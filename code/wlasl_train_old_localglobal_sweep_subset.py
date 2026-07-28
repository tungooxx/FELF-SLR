"""Old-frame LocalGlobal ArcFace architecture sweep for WLASL-N.

This script intentionally uses only the original frame-wise part-aware feature
extractor:

    extract_part_aware_features(frame, RECTIFY_ALPHA)

No seqplus, exp13, raw topology, gates, GEO-OT, auxiliary heads, pretraining,
distillation, reranking, or confusion correction are included.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from tqdm import tqdm

from wlasl_geometry_utils import GeometryConfig, extract_part_aware_features, _prepare_parts
from wlasl_train_local_global_arcface import ArcFaceLoss, CosineClassifier, PositionalEncoding, mixup_three, topk_metrics
from wlasl_train_local_global_arcface_subset import (
    DATA_DIR,
    JSON_PATH,
    RECTIFY_ALPHA,
    SEED,
    SEQUENCE_LENGTH,
    CachedAugmentedPartDataset,
    collect_logits,
    collect_probs,
    eval_accuracy,
    make_loader,
)
from wlasl_train_streams_arcface import augment_fixed, get_class_weights
from spoter_style_augment import augment_spoter_style


class ConfusionRankArcFaceLoss(nn.Module):
    """ArcFace plus validation-mined confused-neighbor ranking pressure."""

    def __init__(
        self,
        scale: float,
        margin: float,
        cw: torch.Tensor | None,
        confused_neighbors: torch.Tensor | None,
        rank_lambda: float,
        rank_margin: float,
    ):
        super().__init__()
        self.arcface = ArcFaceLoss(scale=scale, margin=margin, cw=cw)
        self.scale = float(scale)
        self.rank_lambda = float(rank_lambda)
        self.rank_margin = float(rank_margin)
        if confused_neighbors is not None:
            self.register_buffer("confused_neighbors", confused_neighbors.long())
        else:
            self.confused_neighbors = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        base = self.arcface(logits, labels)
        if self.confused_neighbors is None or self.rank_lambda <= 0:
            return base
        neighbors = self.confused_neighbors[labels]
        valid = neighbors >= 0
        if not bool(valid.any()):
            return base
        safe_neighbors = neighbors.clamp_min(0)
        cos = logits / self.scale
        true_cos = cos.gather(1, labels[:, None])
        neg_cos = cos.gather(1, safe_neighbors)
        rank_terms = F.softplus(self.scale * (neg_cos - true_cos + self.rank_margin))
        rank_loss = rank_terms[valid].mean()
        return base + self.rank_lambda * rank_loss


class SubcenterCosineClassifier(nn.Module):
    """Cosine classifier with K prototypes per class, reduced to class logits."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_subcenters: int = 2,
        scale: float = 16.0,
        reduce: str = "max",
        lse_temperature: float = 0.08,
    ):
        super().__init__()
        if num_subcenters < 1:
            raise ValueError("--num-subcenters must be >= 1.")
        if reduce not in {"max", "lse"}:
            raise ValueError(f"Unknown subcenter reduce mode: {reduce}")
        self.num_classes = int(num_classes)
        self.num_subcenters = int(num_subcenters)
        self.scale = float(scale)
        self.reduce = reduce
        self.lse_temperature = float(lse_temperature)
        self.weight = nn.Parameter(torch.empty(num_classes, num_subcenters, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def subcenter_cosines(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=-1)
        w = F.normalize(self.weight, p=2, dim=-1)
        return torch.einsum("bd,ckd->bck", x, w)

    def reduce_cosines(self, cosines: torch.Tensor) -> torch.Tensor:
        if self.reduce == "max":
            return cosines.max(dim=-1).values
        temp = max(self.lse_temperature, 1e-6)
        return temp * torch.logsumexp(cosines / temp, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.reduce_cosines(self.subcenter_cosines(x))


def build_confusion_neighbors_from_diagnostic(parent_diagnostic: str | None, num_classes: int, top_k: int):
    if not parent_diagnostic or top_k <= 0:
        return None, None
    diag_path = Path(parent_diagnostic)
    if not diag_path.exists():
        raise FileNotFoundError(f"--conf-rank-parent-diagnostic not found: {diag_path}")
    diag = json.loads(diag_path.read_text(encoding="utf-8-sig"))
    logits_info = diag.get("logits") or {}
    val_logits_path = Path(logits_info.get("val_logits", ""))
    val_labels_path = Path(logits_info.get("val_labels", ""))
    if not val_logits_path.exists() or not val_labels_path.exists():
        raise FileNotFoundError(
            f"Parent diagnostic must contain existing validation logits/labels. Got {val_logits_path} and {val_labels_path}"
        )
    val_logits = np.load(val_logits_path)
    val_labels = np.load(val_labels_path).astype(np.int64)
    pred = val_logits.argmax(axis=1).astype(np.int64)
    neighbors = np.full((num_classes, top_k), -1, dtype=np.int64)
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, p in zip(val_labels, pred):
        if 0 <= y < num_classes and 0 <= p < num_classes and p != y:
            counts[y, p] += 1
    for y in range(num_classes):
        order = np.argsort(-counts[y])
        picked = [int(j) for j in order if j != y and counts[y, j] > 0][:top_k]
        if picked:
            neighbors[y, : len(picked)] = picked
    summary = {
        "parent_diagnostic": str(diag_path),
        "source": "validation_predictions_only",
        "top_k": int(top_k),
        "classes_with_neighbors": int((neighbors >= 0).any(axis=1).sum()),
        "total_neighbor_slots": int((neighbors >= 0).sum()),
        "top_confusions": [
            {"true": int(y), "pred": int(j), "count": int(counts[y, j])}
            for y, j in sorted(
                [(y, j) for y in range(num_classes) for j in range(num_classes) if counts[y, j] > 0],
                key=lambda pair: int(counts[pair[0], pair[1]]),
                reverse=True,
            )[:20]
        ],
    }
    return torch.from_numpy(neighbors), summary


EXPECTED_LEFT_DIM = 165
EXPECTED_RIGHT_DIM = 165
EXPECTED_GLOBAL_DIM = 23
PALMNORMVEC_DIM = 64
STAGE2_FEATURE_KINDS = {"b1_wrist", "b2_bodytraj", "b3_dualref", "b4_dualref_lg"}
STAGE3_FEATURE_KINDS = {"c1_tinygcn", "c2_tinygcn_residual", "c3_dynamic_graph"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_actions_for_source(data_dir: str, json_path: str, num_glosses: int, action_source: str):
    if action_source == "json_first_n":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return sorted(entry["gloss"] for entry in data[:num_glosses])
    if action_source == "top_frequency":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ranked = sorted(
            data,
            key=lambda entry: (-len(entry.get("instances", [])), entry.get("gloss", "")),
        )
        return sorted(entry["gloss"] for entry in ranked[:num_glosses])
    if action_source == "train_dirs":
        train_dir = Path(data_dir) / "train"
        actions = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        return actions[:num_glosses]
    raise ValueError(f"Unknown action_source={action_source}")


def load_subset_raw_for_source(data_dir, json_path, num_glosses, action_source, limit_samples=0, labels_only=False):
    actions = read_actions_for_source(data_dir, json_path, num_glosses, action_source)
    label_map = {gloss: i for i, gloss in enumerate(actions)}
    splits = {split: {"raw": [], "labels": [], "ids": []} for split in ["train", "val", "test"]}

    for split in ["train", "val", "test"]:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            continue
        for gloss in actions:
            gloss_dir = split_dir / gloss
            if not gloss_dir.exists():
                continue
            for vid_dir in sorted(p for p in gloss_dir.iterdir() if p.is_dir()):
                valid_vid = True
                frames_raw = []
                for frame_idx in range(SEQUENCE_LENGTH):
                    frame_path = vid_dir / f"{frame_idx}.npy"
                    if not frame_path.exists():
                        valid_vid = False
                        break
                    if not labels_only:
                        frames_raw.append(np.load(frame_path))
                if not valid_vid:
                    continue
                splits[split]["raw"].append(None if labels_only else np.asarray(frames_raw, dtype=np.float32))
                splits[split]["labels"].append(label_map[gloss])
                splits[split]["ids"].append(str(vid_dir))
                if limit_samples and len(splits[split]["labels"]) >= limit_samples:
                    break
            if limit_samples and len(splits[split]["labels"]) >= limit_samples:
                break
    return splits, actions, label_map


def old_frame_cache_dir(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "frame_old"


def feature_cache_dir(cache_dir: str | Path, feature_kind: str) -> Path:
    return old_frame_cache_dir(cache_dir) if feature_kind == "old" else Path(cache_dir) / feature_kind


def old_frame_cache_ready(cache_dir: str | Path, aug_repeats: int) -> bool:
    return feature_cache_ready(cache_dir, "old", aug_repeats)


def feature_cache_ready(cache_dir: str | Path, feature_kind: str, aug_repeats: int) -> bool:
    cache_path = feature_cache_dir(cache_dir, feature_kind)
    required = []
    for split in ["train", "val", "test"]:
        required.extend(
            [
                cache_path / f"{split}_{feature_kind}_left.npy",
                cache_path / f"{split}_{feature_kind}_right.npy",
                cache_path / f"{split}_{feature_kind}_global.npy",
            ]
        )
    if aug_repeats > 0:
        required.extend(
            [
                cache_path / f"aug_r{aug_repeats}_{feature_kind}_left.npy",
                cache_path / f"aug_r{aug_repeats}_{feature_kind}_right.npy",
                cache_path / f"aug_r{aug_repeats}_{feature_kind}_global.npy",
                cache_path / f"aug_r{aug_repeats}_{feature_kind}_labels.npy",
            ]
        )
    return all(p.exists() for p in required)


def extract_old_feature_triplet(seq: np.ndarray, hand_scale_floor: float = 1e-6):
    left_frames, right_frames, global_frames = [], [], []
    for t in range(SEQUENCE_LENGTH):
        left, right, global_features = extract_part_aware_features(seq[t], RECTIFY_ALPHA, hand_scale_floor=hand_scale_floor)
        left_frames.append(left)
        right_frames.append(right)
        global_frames.append(global_features)
    return (
        np.asarray(left_frames, dtype=np.float32),
        np.asarray(right_frames, dtype=np.float32),
        np.asarray(global_frames, dtype=np.float32),
    )


def remove_palm_normal(local_features: np.ndarray) -> np.ndarray:
    # extract_part_aware_features local layout:
    # rel[1:] 60 + bone_vec 60 + bone_len 20 + flex 10 + spread 9 + palm 3 + extras 3 = 165.
    return np.concatenate([local_features[:159], local_features[162:]], axis=-1).astype(np.float32)


def extract_old_no_palm_feature_triplet(seq: np.ndarray, hand_scale_floor: float = 1e-6):
    left_frames, right_frames, global_frames = [], [], []
    for t in range(SEQUENCE_LENGTH):
        left, right, global_features = extract_part_aware_features(seq[t], RECTIFY_ALPHA, hand_scale_floor=hand_scale_floor)
        left_frames.append(remove_palm_normal(left))
        right_frames.append(remove_palm_normal(right))
        global_frames.append(global_features)
    return (
        np.asarray(left_frames, dtype=np.float32),
        np.asarray(right_frames, dtype=np.float32),
        np.asarray(global_frames, dtype=np.float32),
    )


def palmnormvec_hand_features(hand_xyz: np.ndarray, hand_scale_floor: float = 1e-6) -> np.ndarray:
    """Compact palm-centered normalized hand vector: 21x3 coords + valid flag."""
    hand_xyz = np.asarray(hand_xyz, dtype=np.float32)
    valid = float(np.any(np.abs(hand_xyz) > 1e-8))
    if valid <= 0.0:
        return np.zeros(PALMNORMVEC_DIM, dtype=np.float32)
    palm_center = hand_xyz[[0, 5, 9, 13, 17]].mean(axis=0).astype(np.float32)
    scale = float(np.linalg.norm(hand_xyz[0] - hand_xyz[9]))
    if scale <= hand_scale_floor:
        scale = 1.0 if hand_scale_floor <= 1e-6 else float(hand_scale_floor)
    rel = ((hand_xyz - palm_center[None, :]) / scale).astype(np.float32)
    return np.concatenate([rel.reshape(-1), np.array([valid], dtype=np.float32)]).astype(np.float32)


def extract_old_palmnormvec_feature_triplet(seq: np.ndarray, hand_scale_floor: float = 1e-6):
    left_frames, right_frames, global_frames = [], [], []
    cfg = GeometryConfig(name="old_palmnormvec", rectification=True, rectify_alpha=RECTIFY_ALPHA)
    for t in range(SEQUENCE_LENGTH):
        pose, lh, rh = _prepare_parts(seq[t], cfg)
        del pose
        left, right, global_features = extract_part_aware_features(seq[t], RECTIFY_ALPHA, hand_scale_floor=hand_scale_floor)
        left_frames.append(np.concatenate([left, palmnormvec_hand_features(lh, hand_scale_floor)], axis=-1))
        right_frames.append(np.concatenate([right, palmnormvec_hand_features(rh, hand_scale_floor)], axis=-1))
        global_frames.append(global_features)
    return (
        np.asarray(left_frames, dtype=np.float32),
        np.asarray(right_frames, dtype=np.float32),
        np.asarray(global_frames, dtype=np.float32),
    )


def _stage2_frame_features(vec: np.ndarray):
    cfg = GeometryConfig(name="stage2_dual_reference", rectification=True, rectify_alpha=RECTIFY_ALPHA)
    pose, lh, rh = _prepare_parts(vec, cfg)
    face = pose[[0, 2, 5, 9, 10]]
    face_center = face.mean(axis=0).astype(np.float32)
    face_scale = float(np.linalg.norm(face[1] - face[2]))
    if face_scale <= 1e-6:
        face_scale = 1.0
    shoulder_center = (pose[11] + pose[12]).astype(np.float32) * 0.5
    body_scale = float(np.linalg.norm(pose[11] - pose[12]))
    if body_scale <= 1e-6:
        body_scale = 1.0

    def hand_wrist_shape(hand):
        valid = float(np.any(np.abs(hand) > 1e-8))
        wrist = hand[0].astype(np.float32)
        scale = float(np.linalg.norm(hand[9] - wrist))
        if scale <= 1e-6:
            scale = 1.0
        coords = ((hand - wrist[None, :]) / scale).astype(np.float32).reshape(-1)
        return np.concatenate([coords, np.array([valid], dtype=np.float32)]).astype(np.float32)

    def hand_body_traj(hand, other):
        wrist = hand[0].astype(np.float32)
        palm = hand[[0, 5, 9, 13, 17]].mean(axis=0).astype(np.float32)
        other_wrist = other[0].astype(np.float32)
        wrist_body = ((wrist - shoulder_center) / body_scale).astype(np.float32)
        wrist_face = ((wrist - face_center) / face_scale).astype(np.float32)
        palm_body = ((palm - shoulder_center) / body_scale).astype(np.float32)
        other_rel = ((wrist - other_wrist) / body_scale).astype(np.float32)
        distances = np.array(
            [
                np.linalg.norm(wrist - face_center) / body_scale,
                np.linalg.norm(palm - face_center) / body_scale,
                np.linalg.norm(wrist - other_wrist) / body_scale,
                float(np.any(np.abs(hand) > 1e-8)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([wrist_body, wrist_face, palm_body, other_rel, distances]).astype(np.float32)

    old_left, old_right, old_global = extract_part_aware_features(vec, RECTIFY_ALPHA)
    left_wrist = hand_wrist_shape(lh)
    right_wrist = hand_wrist_shape(rh)
    left_traj = hand_body_traj(lh, rh)
    right_traj = hand_body_traj(rh, lh)
    interaction = np.concatenate(
        [
            ((lh[0] - rh[0]) / body_scale).astype(np.float32),
            ((lh[[0, 5, 9, 13, 17]].mean(axis=0) - rh[[0, 5, 9, 13, 17]].mean(axis=0)) / body_scale).astype(np.float32),
            np.array([np.linalg.norm(lh[0] - rh[0]) / body_scale], dtype=np.float32),
            ((face.reshape(-1) - np.tile(face_center, len(face))) / face_scale).astype(np.float32),
        ]
    ).astype(np.float32)
    return old_left, old_right, old_global, left_wrist, right_wrist, left_traj, right_traj, interaction


def extract_stage2_feature_triplet(seq: np.ndarray, feature_kind: str):
    left_frames, right_frames, global_frames = [], [], []
    prev_left_traj = None
    prev_right_traj = None
    for t in range(SEQUENCE_LENGTH):
        old_l, old_r, old_g, lw, rw, lt, rt, inter = _stage2_frame_features(seq[t])
        lv = np.zeros_like(lt) if prev_left_traj is None else (lt - prev_left_traj).astype(np.float32)
        rv = np.zeros_like(rt) if prev_right_traj is None else (rt - prev_right_traj).astype(np.float32)
        left_traj = np.concatenate([lt, lv]).astype(np.float32)
        right_traj = np.concatenate([rt, rv]).astype(np.float32)

        if feature_kind == "b1_wrist":
            left, right, global_features = lw, rw, old_g
        elif feature_kind == "b2_bodytraj":
            left, right, global_features = left_traj, right_traj, inter
        elif feature_kind == "b3_dualref":
            left, right, global_features = np.concatenate([lw, left_traj]), np.concatenate([rw, right_traj]), np.concatenate([old_g, inter])
        elif feature_kind == "b4_dualref_lg":
            left = np.concatenate([old_l, lw, left_traj])
            right = np.concatenate([old_r, rw, right_traj])
            global_features = np.concatenate([old_g, inter])
        else:
            raise ValueError(f"Unknown stage2 feature_kind={feature_kind}")
        left_frames.append(left.astype(np.float32))
        right_frames.append(right.astype(np.float32))
        global_frames.append(global_features.astype(np.float32))
        prev_left_traj = lt
        prev_right_traj = rt
    return (
        np.asarray(left_frames, dtype=np.float32),
        np.asarray(right_frames, dtype=np.float32),
        np.asarray(global_frames, dtype=np.float32),
    )


HAND_GRAPH_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _topology_hand_features(hand: np.ndarray, dynamic: bool = False) -> np.ndarray:
    valid = float(np.any(np.abs(hand) > 1e-8))
    wrist = hand[0].astype(np.float32)
    scale = float(np.linalg.norm(hand[9] - wrist))
    if scale <= 1e-6:
        scale = 1.0
    rel = ((hand - wrist[None, :]) / scale).astype(np.float32)

    agg = np.zeros_like(rel, dtype=np.float32)
    deg = np.zeros((21, 1), dtype=np.float32)
    bone_vecs = []
    bone_lens = []
    for i, j in HAND_GRAPH_EDGES:
        agg[i] += rel[j]
        agg[j] += rel[i]
        deg[i] += 1.0
        deg[j] += 1.0
        bv = (rel[j] - rel[i]).astype(np.float32)
        bone_vecs.append(bv)
        bone_lens.append(np.linalg.norm(bv).astype(np.float32))
    neigh = agg / np.maximum(deg, 1.0)
    topo = np.concatenate(
        [
            rel.reshape(-1),
            (neigh - rel).reshape(-1),
            np.asarray(bone_vecs, dtype=np.float32).reshape(-1),
            np.asarray(bone_lens, dtype=np.float32).reshape(-1),
            np.array([valid, scale], dtype=np.float32),
        ]
    ).astype(np.float32)

    if not dynamic:
        return topo

    # Tiny dynamic adjacency proxy: nearest-joint distances in normalized hand space.
    dist = np.linalg.norm(rel[:, None, :] - rel[None, :, :], axis=-1).astype(np.float32)
    nearest = np.sort(dist + np.eye(21, dtype=np.float32) * 1e6, axis=1)[:, :3]
    fingertips = rel[[4, 8, 12, 16, 20]]
    fingertip_dist = []
    for a in range(len(fingertips)):
        for b in range(a + 1, len(fingertips)):
            fingertip_dist.append(np.linalg.norm(fingertips[a] - fingertips[b]).astype(np.float32))
    return np.concatenate([topo, nearest.reshape(-1), np.asarray(fingertip_dist, dtype=np.float32)]).astype(np.float32)


def extract_stage3_feature_triplet(seq: np.ndarray, feature_kind: str):
    left_frames, right_frames, global_frames = [], [], []
    for t in range(SEQUENCE_LENGTH):
        cfg = GeometryConfig(name="stage3_topology", rectification=True, rectify_alpha=RECTIFY_ALPHA)
        pose, lh, rh = _prepare_parts(seq[t], cfg)
        old_left, old_right, old_global = extract_part_aware_features(seq[t], RECTIFY_ALPHA)
        dynamic = feature_kind == "c3_dynamic_graph"
        left_graph = _topology_hand_features(lh, dynamic=dynamic)
        right_graph = _topology_hand_features(rh, dynamic=dynamic)
        if feature_kind == "c1_tinygcn":
            left, right, global_features = left_graph, right_graph, old_global
        elif feature_kind == "c2_tinygcn_residual":
            left, right, global_features = np.concatenate([old_left, left_graph]), np.concatenate([old_right, right_graph]), old_global
        elif feature_kind == "c3_dynamic_graph":
            left, right, global_features = left_graph, right_graph, old_global
        else:
            raise ValueError(f"Unknown stage3 feature_kind={feature_kind}")
        left_frames.append(left.astype(np.float32))
        right_frames.append(right.astype(np.float32))
        global_frames.append(global_features.astype(np.float32))
    return (
        np.asarray(left_frames, dtype=np.float32),
        np.asarray(right_frames, dtype=np.float32),
        np.asarray(global_frames, dtype=np.float32),
    )


def extract_feature_triplet(seq: np.ndarray, feature_kind: str, hand_scale_floor: float = 1e-6):
    if feature_kind == "old":
        return extract_old_feature_triplet(seq, hand_scale_floor=hand_scale_floor)
    if feature_kind == "old_no_palm":
        return extract_old_no_palm_feature_triplet(seq, hand_scale_floor=hand_scale_floor)
    if feature_kind == "old_palmnormvec":
        return extract_old_palmnormvec_feature_triplet(seq, hand_scale_floor=hand_scale_floor)
    if feature_kind in STAGE2_FEATURE_KINDS:
        return extract_stage2_feature_triplet(seq, feature_kind)
    if feature_kind in STAGE3_FEATURE_KINDS:
        return extract_stage3_feature_triplet(seq, feature_kind)
    raise ValueError(f"Unknown feature_kind={feature_kind}")


def _build_augmented_sample_worker(payload):
    i, raw, label, repeats, feature_kind, augment_mode, hand_scale_floor, seed = payload
    np.random.seed(int(seed))
    left_rows, right_rows, global_rows = [], [], []
    y_rows = np.empty(repeats, dtype=np.int64)
    for r in range(repeats):
        left_seq, right_seq, global_seq = extract_feature_triplet(
            augment_spoter_style(raw, augment_mode),
            feature_kind,
            hand_scale_floor=hand_scale_floor,
        )
        left_rows.append(left_seq.astype(np.float32))
        right_rows.append(right_seq.astype(np.float32))
        global_rows.append(global_seq.astype(np.float32))
        y_rows[r] = label
    return (
        i,
        np.asarray(left_rows, dtype=np.float32),
        np.asarray(right_rows, dtype=np.float32),
        np.asarray(global_rows, dtype=np.float32),
        y_rows,
    )


def encode_feature_parts(
    raw_list,
    split_name: str,
    cache_dir: str | Path,
    feature_kind: str,
    force: bool = False,
    hand_scale_floor: float = 1e-6,
):
    cache_path = feature_cache_dir(cache_dir, feature_kind)
    left_path = cache_path / f"{split_name}_{feature_kind}_left.npy"
    right_path = cache_path / f"{split_name}_{feature_kind}_right.npy"
    global_path = cache_path / f"{split_name}_{feature_kind}_global.npy"
    if not force and all(p.exists() for p in [left_path, right_path, global_path]):
        print(f"Using cached {split_name} {feature_kind} features.")
        return (
            np.load(left_path, mmap_mode="r"),
            np.load(right_path, mmap_mode="r"),
            np.load(global_path, mmap_mode="r"),
        )

    print(f"Encoding {split_name} {feature_kind} features...")
    cache_path.mkdir(parents=True, exist_ok=True)
    left_all, right_all, global_all = [], [], []
    total = len(raw_list)
    for i, seq in enumerate(raw_list):
        left_seq, right_seq, global_seq = extract_feature_triplet(
            np.asarray(seq, dtype=np.float32),
            feature_kind,
            hand_scale_floor=hand_scale_floor,
        )
        left_all.append(left_seq)
        right_all.append(right_seq)
        global_all.append(global_seq)
        if (i + 1) % 250 == 0 or (i + 1) == total:
            print(f"  encoded {split_name}: {i + 1}/{total}")

    np.save(left_path, np.asarray(left_all, dtype=np.float32))
    np.save(right_path, np.asarray(right_all, dtype=np.float32))
    np.save(global_path, np.asarray(global_all, dtype=np.float32))
    return (
        np.load(left_path, mmap_mode="r"),
        np.load(right_path, mmap_mode="r"),
        np.load(global_path, mmap_mode="r"),
    )


def encode_old_parts(raw_list, split_name: str, cache_dir: str | Path, force: bool = False):
    return encode_feature_parts(raw_list, split_name, cache_dir, "old", force=force)


def build_cached_feature_augments(
    raw_train,
    y_train,
    left_dim,
    right_dim,
    global_dim,
    repeats,
    cache_dir,
    feature_kind,
    force=False,
    augment_mode="fixed",
    hand_scale_floor=1e-6,
    cache_workers=0,
):
    if repeats <= 0:
        return None, None, None, None
    cache_path = feature_cache_dir(cache_dir, feature_kind)
    cache_path.mkdir(parents=True, exist_ok=True)
    aug_tag = "aug" if augment_mode == "fixed" else f"aug_{augment_mode}"
    left_path = cache_path / f"{aug_tag}_r{repeats}_{feature_kind}_left.npy"
    right_path = cache_path / f"{aug_tag}_r{repeats}_{feature_kind}_right.npy"
    global_path = cache_path / f"{aug_tag}_r{repeats}_{feature_kind}_global.npy"
    y_path = cache_path / f"{aug_tag}_r{repeats}_{feature_kind}_labels.npy"
    expected = len(y_train) * repeats
    if not force and all(p.exists() for p in [left_path, right_path, global_path, y_path]):
        labels = np.load(y_path)
        if len(labels) == expected:
            print(f"Using cached {feature_kind} augmentation repeats={repeats}.")
            return (
                np.load(left_path, mmap_mode="r"),
                np.load(right_path, mmap_mode="r"),
                np.load(global_path, mmap_mode="r"),
                labels,
            )

    print(f"Building cached {feature_kind} augmentation repeats={repeats}...")
    left_aug = np.lib.format.open_memmap(left_path, mode="w+", dtype=np.float32, shape=(expected, SEQUENCE_LENGTH, left_dim))
    right_aug = np.lib.format.open_memmap(right_path, mode="w+", dtype=np.float32, shape=(expected, SEQUENCE_LENGTH, right_dim))
    global_aug = np.lib.format.open_memmap(global_path, mode="w+", dtype=np.float32, shape=(expected, SEQUENCE_LENGTH, global_dim))
    aug_y = np.empty(expected, dtype=np.int64)

    if cache_workers and cache_workers > 1:
        seeds = np.random.randint(0, 2**31 - 1, size=len(y_train), dtype=np.int64)
        payloads = [
            (i, raw_train[i], int(y_train[i]), repeats, feature_kind, augment_mode, hand_scale_floor, int(seeds[i]))
            for i in range(len(y_train))
        ]
        print(f"Using {cache_workers} cache workers for augmentation.")
        with ProcessPoolExecutor(max_workers=cache_workers) as ex:
            for i, left_rows, right_rows, global_rows, y_rows in tqdm(
                ex.map(_build_augmented_sample_worker, payloads),
                total=len(payloads),
                desc="Aug",
            ):
                start = i * repeats
                end = start + repeats
                left_aug[start:end] = left_rows
                right_aug[start:end] = right_rows
                global_aug[start:end] = global_rows
                aug_y[start:end] = y_rows
                if (i + 1) % 250 == 0 or (i + 1) == len(y_train):
                    print(f"  cached {i + 1}/{len(y_train)} train samples")
    else:
        cursor = 0
        for i in tqdm(range(len(y_train)), desc="Aug"):
            raw = raw_train[i]
            for _ in range(repeats):
                left_seq, right_seq, global_seq = extract_feature_triplet(
                    augment_spoter_style(raw, augment_mode),
                    feature_kind,
                    hand_scale_floor=hand_scale_floor,
                )
                left_aug[cursor] = left_seq
                right_aug[cursor] = right_seq
                global_aug[cursor] = global_seq
                aug_y[cursor] = y_train[i]
                cursor += 1
            if (i + 1) % 250 == 0 or (i + 1) == len(y_train):
                print(f"  cached {i + 1}/{len(y_train)} train samples")

    del left_aug
    del right_aug
    del global_aug
    np.save(y_path, aug_y)
    return (
        np.load(left_path, mmap_mode="r"),
        np.load(right_path, mmap_mode="r"),
        np.load(global_path, mmap_mode="r"),
        np.load(y_path),
    )


def build_cached_old_augments(raw_train, y_train, left_dim, right_dim, global_dim, repeats, cache_dir, force=False):
    return build_cached_feature_augments(raw_train, y_train, left_dim, right_dim, global_dim, repeats, cache_dir, "old", force=force)


class BranchStem(nn.Module):
    def __init__(self, input_dim: int, branch_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Linear(branch_dim, branch_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def temporal_delta(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v = torch.zeros_like(x)
    v[:, 1:] = x[:, 1:] - x[:, :-1]
    a = torch.zeros_like(x)
    a[:, 1:] = v[:, 1:] - v[:, :-1]
    return v, a


class StaticMotionStem(nn.Module):
    """Split raw stream into static morphology/locus and delta-2 motion bases."""

    def __init__(self, input_dim: int, branch_dim: int, gate_bias: float = -1.5):
        super().__init__()
        self.static_stem = BranchStem(input_dim, branch_dim)
        self.motion_stem = BranchStem(input_dim * 2, branch_dim)
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 4, max(32, branch_dim // 2)),
            nn.GELU(),
            nn.Linear(max(32, branch_dim // 2), branch_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)
        self.last_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, a = temporal_delta(x)
        static = self.static_stem(x)
        motion = self.motion_stem(torch.cat([v, a], dim=-1))
        summary = torch.cat(
            [
                static.mean(dim=1),
                static.std(dim=1, unbiased=False),
                motion.mean(dim=1),
                motion.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(summary)).unsqueeze(1)
        self.last_gate = gate.detach()
        return static + gate * motion


class DominantSupportHandStems(nn.Module):
    """Assign shared dominant/support stems by hand motion energy, then restore anatomical outputs."""

    def __init__(self, input_dim: int, branch_dim: int):
        super().__init__()
        self.dominant_stem = BranchStem(input_dim, branch_dim)
        self.support_stem = BranchStem(input_dim, branch_dim)
        self.last_right_dominant = None

    @staticmethod
    def motion_energy(x: torch.Tensor) -> torch.Tensor:
        v, _ = temporal_delta(x)
        return v.pow(2).mean(dim=(1, 2))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        right_dominant = (self.motion_energy(right) >= self.motion_energy(left)).view(-1, 1, 1)
        self.last_right_dominant = right_dominant.detach()
        left_dom = self.dominant_stem(left)
        left_sup = self.support_stem(left)
        right_dom = self.dominant_stem(right)
        right_sup = self.support_stem(right)
        left_out = torch.where(right_dominant, left_sup, left_dom)
        right_out = torch.where(right_dominant, right_dom, right_sup)
        return left_out, right_out


class RelationInputStream(nn.Module):
    """Explicit cross-hand relation token from stemmed hands and relative motion."""

    def __init__(self, branch_dim: int, relation_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(branch_dim * 4, relation_dim),
            nn.LayerNorm(relation_dim),
            nn.GELU(),
            nn.Linear(relation_dim, relation_dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        lv, _ = temporal_delta(left)
        rv, _ = temporal_delta(right)
        return self.net(torch.cat([left, right, left - right, lv - rv], dim=-1))


class GlobalReliabilityGate(nn.Module):
    def __init__(self, left_dim: int, right_dim: int, global_dim: int, hidden_dim: int = 64, bias_init: float = 2.0):
        super().__init__()
        summary_dim = 2 * (left_dim + right_dim + global_dim)
        self.net = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias_init)
        self.last_gate = None

    def forward(self, left: torch.Tensor, right: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        def summarize(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)

        summary = torch.cat([summarize(left), summarize(right), summarize(global_features)], dim=-1)
        gate = torch.sigmoid(self.net(summary)).unsqueeze(1)
        self.last_gate = gate.detach()
        return global_features * gate


class BranchReliabilityGates(nn.Module):
    def __init__(self, left_dim: int, right_dim: int, global_dim: int, hidden_dim: int = 64, bias_init: float = 2.0):
        super().__init__()
        summary_dim = 2 * (left_dim + right_dim + global_dim)
        self.net = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias_init)
        self.last_gate = None

    def forward(self, left: torch.Tensor, right: torch.Tensor, global_features: torch.Tensor):
        def summarize(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)

        summary = torch.cat([summarize(left), summarize(right), summarize(global_features)], dim=-1)
        gate = torch.sigmoid(self.net(summary))
        self.last_gate = gate.detach()
        return (
            left * gate[:, 0].view(-1, 1, 1),
            right * gate[:, 1].view(-1, 1, 1),
            global_features * gate[:, 2].view(-1, 1, 1),
        )


class SideBalancedStreamDropout(nn.Module):
    """Drop full stem streams per sample to reduce right/global co-adaptation."""

    def __init__(self, left_p: float = 0.05, right_p: float = 0.15, global_p: float = 0.15):
        super().__init__()
        self.left_p = float(left_p)
        self.right_p = float(right_p)
        self.global_p = float(global_p)
        self.last_keep = None

    @staticmethod
    def _drop_stream(x: torch.Tensor, p: float, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if (not training) or p <= 0:
            keep = torch.ones(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
            return x, keep
        keep = torch.empty(x.shape[0], 1, 1, device=x.device, dtype=x.dtype).bernoulli_(1.0 - p)
        return x * keep / max(1.0 - p, 1e-6), keep

    def forward(self, left: torch.Tensor, right: torch.Tensor, global_features: torch.Tensor):
        left, left_keep = self._drop_stream(left, self.left_p, self.training)
        right, right_keep = self._drop_stream(right, self.right_p, self.training)
        global_features, global_keep = self._drop_stream(global_features, self.global_p, self.training)
        self.last_keep = torch.cat([left_keep, right_keep, global_keep], dim=1).detach()
        return left, right, global_features


class StreamReliabilityFusion(nn.Module):
    """Per-sample softmax stream reliability weights before concatenation."""

    def __init__(self, left_dim: int, right_dim: int, global_dim: int, hidden_dim: int = 64, bias_init: float = 0.0):
        super().__init__()
        summary_dim = 2 * (left_dim + right_dim + global_dim)
        self.net = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias_init)
        self.last_weights = None

    def forward(self, left: torch.Tensor, right: torch.Tensor, global_features: torch.Tensor):
        def summarize(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)

        summary = torch.cat([summarize(left), summarize(right), summarize(global_features)], dim=-1)
        weights = torch.softmax(self.net(summary), dim=-1)
        self.last_weights = weights.detach()
        return (
            left * weights[:, 0].view(-1, 1, 1),
            right * weights[:, 1].view(-1, 1, 1),
            global_features * weights[:, 2].view(-1, 1, 1),
        )


class ResidualReliabilityFusion(nn.Module):
    """Inject reliability-weighted stream residuals without replacing B6 streams."""

    def __init__(
        self,
        left_dim: int,
        right_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
        beta_init: float = 0.0,
        beta_mode: str = "learnable",
    ):
        super().__init__()
        if beta_mode not in {"learnable", "fixed", "sigmoid"}:
            raise ValueError(f"Unknown residual RF beta mode: {beta_mode}")
        summary_dim = 2 * (left_dim + right_dim + global_dim)
        self.net = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.beta_mode = beta_mode
        self.beta = nn.Parameter(torch.tensor(float(beta_init)), requires_grad=(beta_mode != "fixed"))
        if beta_mode == "sigmoid":
            beta_init = min(max(float(beta_init), 1e-4), 1.0 - 1e-4)
            self.beta.data.fill_(float(np.log(beta_init / (1.0 - beta_init))))
        self.last_weights = None
        self.last_beta = None

    def _beta(self) -> torch.Tensor:
        if self.beta_mode == "sigmoid":
            return torch.sigmoid(self.beta)
        return self.beta

    def forward(self, left: torch.Tensor, right: torch.Tensor, global_features: torch.Tensor):
        def summarize(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)

        summary = torch.cat([summarize(left), summarize(right), summarize(global_features)], dim=-1)
        weights = torch.softmax(self.net(summary), dim=-1)
        beta = self._beta()
        self.last_weights = weights.detach()
        self.last_beta = beta.detach()
        return (
            left + beta * weights[:, 0].view(-1, 1, 1) * left,
            right + beta * weights[:, 1].view(-1, 1, 1) * right,
            global_features + beta * weights[:, 2].view(-1, 1, 1) * global_features,
        )


class KinematicResidualAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.1,
        scale: float = 1.0,
        zero_init: bool = True,
        bounded: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.bounded = bounded
        self.norm = nn.LayerNorm(d_model * 4)
        self.fc1 = nn.Linear(d_model * 4, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        if zero_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        self.last_diag = None

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        v = torch.zeros_like(src)
        v[:, 1:] = src[:, 1:] - src[:, :-1]
        a = torch.zeros_like(src)
        a[:, 1:] = v[:, 1:] - v[:, :-1]
        abs_v = v.abs()
        energy = v * v
        kin = torch.cat([v, a, abs_v, energy], dim=-1)
        residual = self.fc2(self.drop(torch.nn.functional.gelu(self.fc1(self.norm(kin)))))
        injected = torch.tanh(residual) if self.bounded else residual
        with torch.no_grad():
            src_norm = src.norm(dim=-1).mean()
            residual_norm = residual.norm(dim=-1).mean()
            injected_norm = (self.scale * injected).norm(dim=-1).mean()
            self.last_diag = {
                "residual_norm_mean": float(residual_norm.detach().cpu().item()),
                "injected_norm_mean": float(injected_norm.detach().cpu().item()),
                "src_norm_mean": float(src_norm.detach().cpu().item()),
                "residual_to_src_ratio": float((residual_norm / (src_norm + 1e-8)).detach().cpu().item()),
                "injected_to_src_ratio": float((injected_norm / (src_norm + 1e-8)).detach().cpu().item()),
                "velocity_energy_mean": float(energy.mean().detach().cpu().item()),
                "acceleration_energy_mean": float((a * a).mean().detach().cpu().item()),
            }
        return src + self.scale * injected


class ZeroInitCrossHandAttention(nn.Module):
    def __init__(
        self,
        left_dim: int,
        right_dim: int,
        attn_dim: int = 64,
        dropout: float = 0.1,
        scale: float = 1.0,
        direction: str = "bidirectional",
        zero_init: bool = True,
    ):
        super().__init__()
        if direction not in {"bidirectional", "right_queries_left", "left_queries_right"}:
            raise ValueError(f"Unknown cross-hand direction={direction}.")
        self.scale = scale
        self.direction = direction
        self.q_r = nn.Linear(right_dim, attn_dim)
        self.k_l = nn.Linear(left_dim, attn_dim)
        self.v_l = nn.Linear(left_dim, attn_dim)
        self.q_l = nn.Linear(left_dim, attn_dim)
        self.k_r = nn.Linear(right_dim, attn_dim)
        self.v_r = nn.Linear(right_dim, attn_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.right_adapter = nn.Sequential(nn.LayerNorm(attn_dim), nn.Linear(attn_dim, right_dim))
        self.left_adapter = nn.Sequential(nn.LayerNorm(attn_dim), nn.Linear(attn_dim, left_dim))
        if zero_init:
            nn.init.zeros_(self.right_adapter[-1].weight)
            nn.init.zeros_(self.right_adapter[-1].bias)
            nn.init.zeros_(self.left_adapter[-1].weight)
            nn.init.zeros_(self.left_adapter[-1].bias)
        self.last_diag = None

    @staticmethod
    def _entropy(attn: torch.Tensor) -> torch.Tensor:
        return -(attn * (attn.clamp_min(1e-8).log())).sum(dim=-1).mean()

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        scale = float(self.q_r.out_features) ** -0.5
        left_update = torch.zeros_like(left)
        right_update = torch.zeros_like(right)
        attn_entropy_r2l = None
        attn_entropy_l2r = None

        if self.direction in {"bidirectional", "right_queries_left"}:
            scores_r2l = torch.matmul(self.q_r(right), self.k_l(left).transpose(-2, -1)) * scale
            attn_r2l = torch.softmax(scores_r2l, dim=-1)
            context_r = torch.matmul(self.attn_drop(attn_r2l), self.v_l(left))
            right_update = self.right_adapter(context_r)
            attn_entropy_r2l = self._entropy(attn_r2l)

        if self.direction in {"bidirectional", "left_queries_right"}:
            scores_l2r = torch.matmul(self.q_l(left), self.k_r(right).transpose(-2, -1)) * scale
            attn_l2r = torch.softmax(scores_l2r, dim=-1)
            context_l = torch.matmul(self.attn_drop(attn_l2r), self.v_r(right))
            left_update = self.left_adapter(context_l)
            attn_entropy_l2r = self._entropy(attn_l2r)

        with torch.no_grad():
            left_norm = left.norm(dim=-1).mean()
            right_norm = right.norm(dim=-1).mean()
            left_update_norm = left_update.norm(dim=-1).mean()
            right_update_norm = right_update.norm(dim=-1).mean()
            self.last_diag = {
                "attn_entropy_r2l": None if attn_entropy_r2l is None else float(attn_entropy_r2l.detach().cpu().item()),
                "attn_entropy_l2r": None if attn_entropy_l2r is None else float(attn_entropy_l2r.detach().cpu().item()),
                "left_update_norm_mean": float(left_update_norm.detach().cpu().item()),
                "right_update_norm_mean": float(right_update_norm.detach().cpu().item()),
                "left_norm_mean": float(left_norm.detach().cpu().item()),
                "right_norm_mean": float(right_norm.detach().cpu().item()),
                "left_update_to_left_ratio": float((left_update_norm / (left_norm + 1e-8)).detach().cpu().item()),
                "right_update_to_right_ratio": float((right_update_norm / (right_norm + 1e-8)).detach().cpu().item()),
            }
        return left + self.scale * left_update, right + self.scale * right_update


class ZeroInitHandGlobalAttention(nn.Module):
    def __init__(self, hand_dim: int, global_dim: int, attn_dim: int = 64, dropout: float = 0.1, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.q_l = nn.Linear(hand_dim, attn_dim)
        self.q_r = nn.Linear(hand_dim, attn_dim)
        self.k_g = nn.Linear(global_dim, attn_dim)
        self.v_g = nn.Linear(global_dim, attn_dim)
        self.left_adapter = nn.Sequential(nn.LayerNorm(attn_dim), nn.Linear(attn_dim, hand_dim))
        self.right_adapter = nn.Sequential(nn.LayerNorm(attn_dim), nn.Linear(attn_dim, hand_dim))
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.left_adapter[-1].weight)
        nn.init.zeros_(self.left_adapter[-1].bias)
        nn.init.zeros_(self.right_adapter[-1].weight)
        nn.init.zeros_(self.right_adapter[-1].bias)
        self.last_diag = None

    @staticmethod
    def _entropy(attn: torch.Tensor) -> float:
        ent = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1)
        return float(ent.mean().detach().cpu().item())

    def _attend(self, q, k, v):
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        return self.drop(attn) @ v, attn

    def forward(self, left, right, global_features):
        k = self.k_g(global_features)
        v = self.v_g(global_features)
        left_ctx, left_attn = self._attend(self.q_l(left), k, v)
        right_ctx, right_attn = self._attend(self.q_r(right), k, v)
        left_update = self.left_adapter(left_ctx)
        right_update = self.right_adapter(right_ctx)
        with torch.no_grad():
            left_norm = left.norm(dim=-1).mean().clamp_min(1e-8)
            right_norm = right.norm(dim=-1).mean().clamp_min(1e-8)
            left_update_norm = left_update.norm(dim=-1).mean()
            right_update_norm = right_update.norm(dim=-1).mean()
            self.last_diag = {
                "hand_global_entropy_left": self._entropy(left_attn),
                "hand_global_entropy_right": self._entropy(right_attn),
                "left_hg_update_norm_mean": float(left_update_norm.detach().cpu().item()),
                "right_hg_update_norm_mean": float(right_update_norm.detach().cpu().item()),
                "left_hg_update_to_left_ratio": float((left_update_norm / left_norm).detach().cpu().item()),
                "right_hg_update_to_right_ratio": float((right_update_norm / right_norm).detach().cpu().item()),
            }
        return left + self.scale * left_update, right + self.scale * right_update


def sinkhorn_transport(cost: torch.Tensor, eps: float = 0.05, iters: int = 8) -> torch.Tensor:
    kernel = torch.exp(-cost / eps).clamp_min(1e-12)
    u = torch.ones(cost.shape[:-1], device=cost.device, dtype=cost.dtype)
    v = torch.ones(cost.shape[:-2] + cost.shape[-1:], device=cost.device, dtype=cost.dtype)
    for _ in range(iters):
        u = 1.0 / (kernel @ v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)
        v = 1.0 / (kernel.transpose(-2, -1) @ u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)
    plan = u.unsqueeze(-1) * kernel * v.unsqueeze(-2)
    return plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class ZeroInitOTHandGlobalAlign(nn.Module):
    def __init__(
        self,
        hand_dim: int,
        global_dim: int,
        align_dim: int = 64,
        eps: float = 0.05,
        iters: int = 8,
        scale: float = 1.0,
        dustbin: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.iters = iters
        self.scale = scale
        self.dustbin = dustbin
        self.hand_proj = nn.Linear(hand_dim, align_dim)
        self.global_proj = nn.Linear(global_dim, align_dim)
        self.global_val = nn.Linear(global_dim, align_dim)
        self.left_adapter = nn.Sequential(nn.LayerNorm(align_dim * 2), nn.Linear(align_dim * 2, hand_dim))
        self.right_adapter = nn.Sequential(nn.LayerNorm(align_dim * 2), nn.Linear(align_dim * 2, hand_dim))
        self.dustbin_token = nn.Parameter(torch.zeros(1, 1, align_dim))
        self.dustbin_value = nn.Parameter(torch.zeros(1, 1, align_dim))
        nn.init.zeros_(self.left_adapter[-1].weight)
        nn.init.zeros_(self.left_adapter[-1].bias)
        nn.init.zeros_(self.right_adapter[-1].weight)
        nn.init.zeros_(self.right_adapter[-1].bias)
        self.last_diag = None

    def _align(self, hand_tokens, global_features):
        hand = torch.nn.functional.normalize(self.hand_proj(hand_tokens), dim=-1)
        glob = torch.nn.functional.normalize(self.global_proj(global_features), dim=-1)
        values = self.global_val(global_features)
        if self.dustbin:
            glob = torch.cat([glob, self.dustbin_token.expand(glob.shape[0], -1, -1)], dim=1)
            values = torch.cat([values, self.dustbin_value.expand(values.shape[0], -1, -1)], dim=1)
        cost = torch.cdist(hand, glob).pow(2).clamp(0.0, 20.0)
        plan = torch.softmax(-cost / max(self.eps, 1e-6), dim=-1)
        plan = torch.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
        plan = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = plan @ values
        aligned = torch.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
        entropy = -(plan.clamp_min(1e-8) * plan.clamp_min(1e-8).log()).sum(dim=-1).mean()
        dustbin_mass = plan[:, :, -1].mean() if self.dustbin else torch.zeros((), device=plan.device)
        return hand, aligned, entropy, dustbin_mass

    def forward(self, left, right, global_features):
        left_h, left_aligned, left_ent, left_dust = self._align(left, global_features)
        right_h, right_aligned, right_ent, right_dust = self._align(right, global_features)
        left_update = self.left_adapter(torch.cat([left_h, left_aligned], dim=-1))
        right_update = self.right_adapter(torch.cat([right_h, right_aligned], dim=-1))
        with torch.no_grad():
            left_norm = left.norm(dim=-1).mean().clamp_min(1e-8)
            right_norm = right.norm(dim=-1).mean().clamp_min(1e-8)
            left_update_norm = left_update.norm(dim=-1).mean()
            right_update_norm = right_update.norm(dim=-1).mean()
            self.last_diag = {
                "ot_entropy_mean": float(((left_ent + right_ent) * 0.5).detach().cpu().item()),
                "ot_dustbin_mass_mean": float(((left_dust + right_dust) * 0.5).detach().cpu().item()),
                "left_ot_update_norm_mean": float(left_update_norm.detach().cpu().item()),
                "right_ot_update_norm_mean": float(right_update_norm.detach().cpu().item()),
                "left_ot_update_to_left_ratio": float((left_update_norm / left_norm).detach().cpu().item()),
                "right_ot_update_to_right_ratio": float((right_update_norm / right_norm).detach().cpu().item()),
            }
        return left + self.scale * left_update, right + self.scale * right_update


class ResidualConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.drop(torch.nn.functional.gelu(y))
        return self.norm(x + y)


class TemporalConvBridge(nn.Module):
    """Lightweight depthwise temporal bridge before the Transformer."""

    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.2, zero_init: bool = True):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("--temporal-conv-bridge-kernel must be a positive odd integer.")
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.pw = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        if zero_init:
            nn.init.zeros_(self.pw.weight)
            nn.init.zeros_(self.pw.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x.transpose(1, 2)).transpose(1, 2)
        y = self.drop(torch.nn.functional.gelu(self.pw(y)))
        return self.norm(x + y)


class PhaseAwarePool(nn.Module):
    """Start/middle/end slot pooling with light positional phase bias."""

    def __init__(self, d_model: int):
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.out = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.LayerNorm(d_model))
        self.last_slot_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("btd,kd->btk", x, self.slot_queries) / max(x.shape[-1] ** 0.5, 1e-6)
        t = torch.linspace(0.0, 1.0, x.shape[1], device=x.device, dtype=x.dtype).view(1, -1, 1)
        centers = torch.tensor([0.15, 0.50, 0.85], device=x.device, dtype=x.dtype).view(1, 1, 3)
        phase_bias = -((t - centers) ** 2) / 0.08
        weights = torch.softmax(scores + phase_bias, dim=1)
        self.last_slot_weights = weights.detach()
        slots = torch.einsum("btd,btk->bkd", x, weights)
        return self.out(slots.flatten(1))


class LiteTemporalBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.2, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        padding = 0 if causal else kernel_size // 2
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=d_model)
        self.pw = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        if self.causal:
            z = torch.nn.functional.pad(z, (self.kernel_size - 1, 0))
        y = self.dw(z).transpose(1, 2)
        y = self.pw(y)
        x = self.norm1(x + y)
        return self.norm2(x + self.ff(x))


class OldLocalGlobalSweepArcFace(nn.Module):
    def __init__(
        self,
        left_inp: int,
        right_inp: int,
        global_inp: int,
        num_classes: int,
        left_branch_dim: int = 96,
        right_branch_dim: int = 96,
        global_branch_dim: int = 96,
        conv_kernel: int = 3,
        conv_layers: int = 1,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 768,
        dropout: float = 0.3,
        pool_mode: str = "mean",
        temporal_head: str = "transformer",
        lite_kernel: int = 5,
        scale: float = 16.0,
        factorization_mode: str = "none",
        static_motion_gate_bias: float = -1.5,
        relation_dim: int = 96,
        use_global_gate: bool = False,
        global_gate_hidden_dim: int = 64,
        global_gate_bias: float = 2.0,
        use_stream_dropout: bool = False,
        stream_dropout_left: float = 0.05,
        stream_dropout_right: float = 0.15,
        stream_dropout_global: float = 0.15,
        use_reliability_fusion: bool = False,
        reliability_hidden_dim: int = 64,
        reliability_bias: float = 0.0,
        use_residual_reliability_fusion: bool = False,
        residual_rf_hidden_dim: int = 64,
        residual_rf_beta_init: float = 0.0,
        residual_rf_beta_mode: str = "learnable",
        use_lrg_residual: bool = False,
        lrg_residual_gamma_init: float = 0.0,
        lrg_residual_gamma_mode: str = "learnable",
        use_branch_gates: bool = False,
        branch_gate_hidden_dim: int = 64,
        branch_gate_bias: float = 2.0,
        use_temporal_conv_bridge: bool = False,
        temporal_conv_bridge_kernel: int = 5,
        temporal_conv_bridge_dropout: float = 0.2,
        zero_init_temporal_conv_bridge: bool = True,
        use_kinematic_residual: bool = False,
        kinematic_hidden_dim: int = 288,
        kinematic_dropout: float = 0.1,
        kinematic_scale: float = 1.0,
        kinematic_input: str = "concat",
        zero_init_kinematic: bool = True,
        bounded_kinematic: bool = False,
        use_cross_hand_attn: bool = False,
        cross_attn_dim: int = 64,
        cross_attn_dropout: float = 0.1,
        cross_attn_scale: float = 1.0,
        cross_attn_direction: str = "bidirectional",
        zero_init_cross_hand: bool = True,
        use_hand_global_attn: bool = False,
        hand_global_attn_dim: int = 64,
        hand_global_attn_dropout: float = 0.1,
        hand_global_attn_scale: float = 1.0,
        use_ot_align: bool = False,
        ot_align_dim: int = 64,
        ot_epsilon: float = 0.05,
        ot_iters: int = 8,
        ot_scale: float = 1.0,
        ot_dustbin: bool = False,
        classifier_type: str = "cosine",
        num_subcenters: int = 2,
        subcenter_reduce: str = "max",
        subcenter_lse_temperature: float = 0.08,
    ):
        super().__init__()
        if conv_kernel < 1 or conv_kernel % 2 == 0:
            raise ValueError("--conv-kernel must be a positive odd integer.")
        if conv_layers < 1:
            raise ValueError("--conv-layers must be >= 1.")
        if temporal_head not in {"transformer", "lite", "causal_lite"}:
            raise ValueError(f"Unknown temporal_head={temporal_head}.")
        if classifier_type not in {"cosine", "subcenter"}:
            raise ValueError(f"Unknown classifier_type={classifier_type}.")
        if factorization_mode not in {"none", "static_motion_lr", "static_motion_lrg", "dominant_support", "relation_input"}:
            raise ValueError(f"Unknown factorization_mode={factorization_mode}.")
        if lrg_residual_gamma_mode not in {"learnable", "fixed", "sigmoid"}:
            raise ValueError(f"Unknown lrg_residual_gamma_mode={lrg_residual_gamma_mode}.")
        self.pool_mode = pool_mode
        self.temporal_head = temporal_head
        self.factorization_mode = factorization_mode
        self.static_motion_gate_bias = static_motion_gate_bias
        self.relation_dim = int(relation_dim) if factorization_mode == "relation_input" else 0
        self.use_lrg_residual = bool(use_lrg_residual)
        self.lrg_residual_gamma_mode = lrg_residual_gamma_mode
        if factorization_mode in {"static_motion_lr", "static_motion_lrg"}:
            self.left_stem = StaticMotionStem(left_inp, left_branch_dim, gate_bias=static_motion_gate_bias)
            self.right_stem = StaticMotionStem(right_inp, right_branch_dim, gate_bias=static_motion_gate_bias)
            self.global_stem = (
                StaticMotionStem(global_inp, global_branch_dim, gate_bias=static_motion_gate_bias)
                if factorization_mode == "static_motion_lrg"
                else BranchStem(global_inp, global_branch_dim)
            )
            self.dominant_support_stems = None
        elif factorization_mode == "dominant_support":
            self.left_stem = None
            self.right_stem = None
            self.dominant_support_stems = DominantSupportHandStems(left_inp, left_branch_dim)
            if left_branch_dim != right_branch_dim:
                raise ValueError("dominant_support requires left/right branch dims to match.")
            self.global_stem = BranchStem(global_inp, global_branch_dim)
        else:
            self.left_stem = BranchStem(left_inp, left_branch_dim)
            self.dominant_support_stems = None
            self.right_stem = BranchStem(right_inp, right_branch_dim)
            self.global_stem = BranchStem(global_inp, global_branch_dim)
        self.relation_stream = RelationInputStream(left_branch_dim, self.relation_dim) if factorization_mode == "relation_input" else None
        if self.use_lrg_residual:
            self.lrg_left_stem = StaticMotionStem(left_inp, left_branch_dim, gate_bias=static_motion_gate_bias)
            self.lrg_right_stem = StaticMotionStem(right_inp, right_branch_dim, gate_bias=static_motion_gate_bias)
            self.lrg_global_stem = StaticMotionStem(global_inp, global_branch_dim, gate_bias=static_motion_gate_bias)
            self.lrg_project = nn.Linear(left_branch_dim + right_branch_dim + global_branch_dim, left_branch_dim + right_branch_dim + global_branch_dim)
            nn.init.zeros_(self.lrg_project.weight)
            nn.init.zeros_(self.lrg_project.bias)
            gamma = float(lrg_residual_gamma_init)
            if lrg_residual_gamma_mode == "sigmoid":
                gamma = min(max(gamma, 1e-4), 1.0 - 1e-4)
                gamma = float(np.log(gamma / (1.0 - gamma)))
            self.lrg_gamma = nn.Parameter(torch.tensor(gamma), requires_grad=(lrg_residual_gamma_mode != "fixed"))
        else:
            self.lrg_left_stem = None
            self.lrg_right_stem = None
            self.lrg_global_stem = None
            self.lrg_project = None
            self.lrg_gamma = None
        self.stream_dropout = (
            SideBalancedStreamDropout(stream_dropout_left, stream_dropout_right, stream_dropout_global)
            if use_stream_dropout
            else None
        )
        self.reliability_fusion = (
            StreamReliabilityFusion(
                left_branch_dim,
                right_branch_dim,
                global_branch_dim,
                hidden_dim=reliability_hidden_dim,
                bias_init=reliability_bias,
            )
            if use_reliability_fusion
            else None
        )
        self.residual_reliability_fusion = (
            ResidualReliabilityFusion(
                left_branch_dim,
                right_branch_dim,
                global_branch_dim,
                hidden_dim=residual_rf_hidden_dim,
                beta_init=residual_rf_beta_init,
                beta_mode=residual_rf_beta_mode,
            )
            if use_residual_reliability_fusion
            else None
        )
        self.global_gate = (
            GlobalReliabilityGate(
                left_branch_dim,
                right_branch_dim,
                global_branch_dim,
                hidden_dim=global_gate_hidden_dim,
                bias_init=global_gate_bias,
            )
            if use_global_gate
            else None
        )
        self.branch_gates = (
            BranchReliabilityGates(
                left_branch_dim,
                right_branch_dim,
                global_branch_dim,
                hidden_dim=branch_gate_hidden_dim,
                bias_init=branch_gate_bias,
            )
            if use_branch_gates
            else None
        )
        self.d_model = left_branch_dim + right_branch_dim + global_branch_dim + self.relation_dim
        if self.d_model % num_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by num_heads={num_heads}.")
        if kinematic_input not in {"concat", "branch"}:
            raise ValueError(f"Unknown kinematic_input={kinematic_input}.")
        self.kinematic_input = kinematic_input
        self.kinematic_adapter = None
        self.left_kinematic_adapter = None
        self.right_kinematic_adapter = None
        self.global_kinematic_adapter = None
        if use_kinematic_residual:
            if kinematic_input == "concat":
                self.kinematic_adapter = KinematicResidualAdapter(
                    self.d_model,
                    kinematic_hidden_dim,
                    dropout=kinematic_dropout,
                    scale=kinematic_scale,
                    zero_init=zero_init_kinematic,
                    bounded=bounded_kinematic,
                )
            else:
                self.left_kinematic_adapter = KinematicResidualAdapter(
                    left_branch_dim,
                    kinematic_hidden_dim,
                    dropout=kinematic_dropout,
                    scale=kinematic_scale,
                    zero_init=zero_init_kinematic,
                    bounded=bounded_kinematic,
                )
                self.right_kinematic_adapter = KinematicResidualAdapter(
                    right_branch_dim,
                    kinematic_hidden_dim,
                    dropout=kinematic_dropout,
                    scale=kinematic_scale,
                    zero_init=zero_init_kinematic,
                    bounded=bounded_kinematic,
                )
                self.global_kinematic_adapter = KinematicResidualAdapter(
                    global_branch_dim,
                    kinematic_hidden_dim,
                    dropout=kinematic_dropout,
                    scale=kinematic_scale,
                    zero_init=zero_init_kinematic,
                    bounded=bounded_kinematic,
                )
        self.cross_hand_attn = (
            ZeroInitCrossHandAttention(
                left_branch_dim,
                right_branch_dim,
                attn_dim=cross_attn_dim,
                dropout=cross_attn_dropout,
                scale=cross_attn_scale,
                direction=cross_attn_direction,
                zero_init=zero_init_cross_hand,
            )
            if use_cross_hand_attn
            else None
        )
        self.hand_global_attn = (
            ZeroInitHandGlobalAttention(
                left_branch_dim,
                global_branch_dim,
                attn_dim=hand_global_attn_dim,
                dropout=hand_global_attn_dropout,
                scale=hand_global_attn_scale,
            )
            if use_hand_global_attn
            else None
        )
        self.ot_align = (
            ZeroInitOTHandGlobalAlign(
                left_branch_dim,
                global_branch_dim,
                align_dim=ot_align_dim,
                eps=ot_epsilon,
                iters=ot_iters,
                scale=ot_scale,
                dustbin=ot_dustbin,
            )
            if use_ot_align
            else None
        )

        if conv_layers == 1:
            self.conv = nn.Conv1d(self.d_model, self.d_model, conv_kernel, padding=conv_kernel // 2)
            self.res = nn.Linear(self.d_model, self.d_model)
            self.norm = nn.LayerNorm(self.d_model)
            self.proj = nn.Linear(self.d_model, self.d_model)
            self.conv_stack = None
        else:
            self.conv = None
            self.res = None
            self.norm = None
            self.proj = nn.Linear(self.d_model, self.d_model)
            self.conv_stack = nn.Sequential(*[ResidualConvBlock(self.d_model, conv_kernel, dropout) for _ in range(conv_layers)])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model)) if pool_mode == "cls" else None
        self.pe = PositionalEncoding(self.d_model, dropout, SEQUENCE_LENGTH + (1 if pool_mode == "cls" else 0))
        if temporal_head == "transformer":
            enc = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
            self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
            self.lite_head = None
        else:
            self.encoder = None
            self.lite_head = nn.Sequential(
                *[
                    LiteTemporalBlock(
                        self.d_model,
                        kernel_size=lite_kernel,
                        dropout=dropout,
                        causal=(temporal_head == "causal_lite"),
                    )
                    for _ in range(num_layers)
                ]
            )
        self.out_norm = nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_pool = nn.Linear(self.d_model, 1) if pool_mode == "attn" else None
        self.phase_pool = PhaseAwarePool(self.d_model) if pool_mode == "phase" else None
        self.stats_proj = nn.Sequential(nn.Linear(self.d_model * 7, self.d_model), nn.LayerNorm(self.d_model)) if pool_mode == "stats" else None
        self.eventstats_proj = nn.Sequential(nn.Linear(self.d_model * 6, self.d_model), nn.LayerNorm(self.d_model)) if pool_mode == "eventstats" else None
        self.energy_proj = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model), nn.LayerNorm(self.d_model)) if pool_mode == "energy" else None
        if self.energy_proj is not None:
            nn.init.zeros_(self.energy_proj[0].weight)
            nn.init.zeros_(self.energy_proj[0].bias)
        self.temporal_conv_bridge = (
            TemporalConvBridge(
                self.d_model,
                kernel_size=temporal_conv_bridge_kernel,
                dropout=temporal_conv_bridge_dropout,
                zero_init=zero_init_temporal_conv_bridge,
            )
            if use_temporal_conv_bridge
            else None
        )
        self.classifier_type = classifier_type
        if classifier_type == "subcenter":
            self.classifier = SubcenterCosineClassifier(
                self.d_model,
                num_classes,
                num_subcenters=num_subcenters,
                scale=scale,
                reduce=subcenter_reduce,
                lse_temperature=subcenter_lse_temperature,
            )
        else:
            self.classifier = CosineClassifier(self.d_model, num_classes, scale)

    def _pool(self, mem_bt: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "mean":
            return mem_bt.mean(dim=1)
        if self.pool_mode == "attn":
            weights = torch.softmax(self.attn_pool(mem_bt), dim=1)
            return (weights * mem_bt).sum(dim=1)
        if self.pool_mode == "phase":
            return self.phase_pool(mem_bt)
        if self.pool_mode == "cls":
            return mem_bt[:, 0]
        if self.pool_mode == "stats":
            mean = mem_bt.mean(dim=1)
            maxv = mem_bt.amax(dim=1)
            first = mem_bt[:, 0]
            last = mem_bt[:, -1]
            delta = last - first
            vel = mem_bt[:, 1:] - mem_bt[:, :-1]
            mean_vel = vel.mean(dim=1)
            max_vel = vel.abs().amax(dim=1)
            return self.stats_proj(torch.cat([mean, maxv, first, last, delta, mean_vel, max_vel], dim=-1))
        if self.pool_mode == "eventstats":
            vel = torch.zeros_like(mem_bt)
            vel[:, 1:] = mem_bt[:, 1:] - mem_bt[:, :-1]
            event_idx = vel.pow(2).mean(dim=-1).argmax(dim=1)
            event = mem_bt[torch.arange(mem_bt.shape[0], device=mem_bt.device), event_idx]
            z = torch.cat(
                [
                    mem_bt.mean(dim=1),
                    mem_bt.amax(dim=1),
                    mem_bt[:, 0],
                    mem_bt[:, -1],
                    mem_bt[:, -1] - mem_bt[:, 0],
                    event,
                ],
                dim=-1,
            )
            return self.eventstats_proj(z)
        if self.pool_mode == "energy":
            vel = torch.zeros_like(mem_bt)
            vel[:, 1:] = mem_bt[:, 1:] - mem_bt[:, :-1]
            energy = vel.pow(2).mean(dim=-1)
            weights = torch.softmax(energy, dim=1).unsqueeze(-1)
            epool = (weights * mem_bt).sum(dim=1)
            mean = mem_bt.mean(dim=1)
            return mean + self.energy_proj(torch.cat([mean, epool], dim=-1))
        raise ValueError(f"Unknown pool_mode={self.pool_mode}")

    def _lrg_gamma(self) -> torch.Tensor:
        if self.lrg_gamma is None:
            return torch.zeros(())
        if self.lrg_residual_gamma_mode == "sigmoid":
            return torch.sigmoid(self.lrg_gamma)
        return self.lrg_gamma

    def forward_features(self, left: torch.Tensor, right: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        raw_left, raw_right, raw_global = left, right, global_features
        if self.dominant_support_stems is not None:
            left, right = self.dominant_support_stems(left, right)
        else:
            left = self.left_stem(left)
            right = self.right_stem(right)
        global_features = self.global_stem(global_features)
        if self.use_lrg_residual:
            lrg = torch.cat(
                [
                    self.lrg_left_stem(raw_left),
                    self.lrg_right_stem(raw_right),
                    self.lrg_global_stem(raw_global),
                ],
                dim=-1,
            )
            base = torch.cat([left, right, global_features], dim=-1)
            mixed = base + self._lrg_gamma() * self.lrg_project(lrg)
            left, right, global_features = torch.split(
                mixed,
                [
                    left.shape[-1],
                    right.shape[-1],
                    global_features.shape[-1],
                ],
                dim=-1,
            )
        if self.stream_dropout is not None:
            left, right, global_features = self.stream_dropout(left, right, global_features)
        if self.cross_hand_attn is not None:
            left, right = self.cross_hand_attn(left, right)
        if self.hand_global_attn is not None:
            left, right = self.hand_global_attn(left, right, global_features)
        if self.ot_align is not None:
            left, right = self.ot_align(left, right, global_features)
        if self.branch_gates is not None:
            left, right, global_features = self.branch_gates(left, right, global_features)
        if self.reliability_fusion is not None:
            left, right, global_features = self.reliability_fusion(left, right, global_features)
        if self.residual_reliability_fusion is not None:
            left, right, global_features = self.residual_reliability_fusion(left, right, global_features)
        if self.global_gate is not None:
            global_features = self.global_gate(left, right, global_features)
        if self.left_kinematic_adapter is not None:
            left = self.left_kinematic_adapter(left)
            right = self.right_kinematic_adapter(right)
            global_features = self.global_kinematic_adapter(global_features)
        streams = [left, right, global_features]
        if self.relation_stream is not None:
            streams.append(self.relation_stream(left, right))
        src = torch.cat(streams, dim=-1)
        if self.kinematic_adapter is not None:
            src = self.kinematic_adapter(src)

        if self.conv_stack is None:
            residual = self.res(src)
            src = self.conv(src.transpose(1, 2)).transpose(1, 2)
            src = self.norm(src + residual)
            src = self.proj(src)
        else:
            src = self.conv_stack(src)
            src = self.proj(src)
        if self.temporal_conv_bridge is not None:
            src = self.temporal_conv_bridge(src)

        if self.cls_token is not None:
            src = torch.cat([self.cls_token.expand(src.shape[0], -1, -1), src], dim=1)
        if self.temporal_head == "transformer":
            src_t = self.pe(src.transpose(0, 1))
            mem = self.encoder(src_t)
            mem = self.out_norm(mem + src_t).transpose(0, 1)
        else:
            src_pe = self.pe(src.transpose(0, 1)).transpose(0, 1)
            mem = self.lite_head(src_pe)
            mem = self.out_norm(mem + src_pe)
        return self.drop(self._pool(mem))

    def forward(self, left: torch.Tensor, right: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(left, right, global_features))


def parse_args():
    p = argparse.ArgumentParser(description="Old-frame LocalGlobal ArcFace sweep.")
    p.add_argument("--num-glosses", type=int, required=True)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--json-path", default=JSON_PATH)
    p.add_argument("--action-source", choices=["json_first_n", "top_frequency", "train_dirs"], default="json_first_n")
    p.add_argument(
        "--feature-kind",
        choices=[
            "old",
            "old_no_palm",
            "old_palmnormvec",
            "b1_wrist",
            "b2_bodytraj",
            "b3_dualref",
            "b4_dualref_lg",
            "c1_tinygcn",
            "c2_tinygcn_residual",
            "c3_dynamic_graph",
        ],
        default="old",
    )
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--left-branch-dim", type=int, default=96)
    p.add_argument("--right-branch-dim", type=int, default=96)
    p.add_argument("--global-branch-dim", type=int, default=96)
    p.add_argument("--conv-kernel", type=int, default=3)
    p.add_argument("--conv-layers", type=int, default=1)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pool-mode", choices=["mean", "stats", "attn", "cls", "eventstats", "energy", "phase"], default="mean")
    p.add_argument("--temporal-head", choices=["transformer", "lite", "causal_lite"], default="transformer")
    p.add_argument("--lite-kernel", type=int, default=5)
    p.add_argument(
        "--factorization-mode",
        choices=["none", "static_motion_lr", "static_motion_lrg", "dominant_support", "relation_input"],
        default="none",
    )
    p.add_argument("--static-motion-gate-bias", type=float, default=-1.5)
    p.add_argument("--relation-dim", type=int, default=96)
    p.add_argument("--use-stream-dropout", action="store_true")
    p.add_argument("--stream-dropout-left", type=float, default=0.05)
    p.add_argument("--stream-dropout-right", type=float, default=0.15)
    p.add_argument("--stream-dropout-global", type=float, default=0.15)
    p.add_argument("--use-reliability-fusion", action="store_true")
    p.add_argument("--reliability-hidden-dim", type=int, default=64)
    p.add_argument("--reliability-bias", type=float, default=0.0)
    p.add_argument("--use-residual-reliability-fusion", action="store_true")
    p.add_argument("--residual-rf-hidden-dim", type=int, default=64)
    p.add_argument("--residual-rf-beta-init", type=float, default=0.0)
    p.add_argument("--residual-rf-beta-mode", choices=["learnable", "fixed", "sigmoid"], default="learnable")
    p.add_argument("--use-lrg-residual", action="store_true")
    p.add_argument("--lrg-residual-gamma-init", type=float, default=0.0)
    p.add_argument("--lrg-residual-gamma-mode", choices=["learnable", "fixed", "sigmoid"], default="learnable")
    p.add_argument("--use-global-gate", action="store_true")
    p.add_argument("--global-gate-hidden-dim", type=int, default=64)
    p.add_argument("--global-gate-bias", type=float, default=2.0)
    p.add_argument("--use-branch-gates", action="store_true")
    p.add_argument("--branch-gate-hidden-dim", type=int, default=64)
    p.add_argument("--branch-gate-bias", type=float, default=2.0)
    p.add_argument("--use-temporal-conv-bridge", action="store_true")
    p.add_argument("--temporal-conv-bridge-kernel", type=int, default=5)
    p.add_argument("--temporal-conv-bridge-dropout", type=float, default=0.2)
    p.add_argument("--zero-init-temporal-conv-bridge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-kinematic-residual", action="store_true")
    p.add_argument("--kinematic-hidden-dim", type=int, default=288)
    p.add_argument("--kinematic-dropout", type=float, default=0.1)
    p.add_argument("--kinematic-scale", type=float, default=1.0)
    p.add_argument("--kinematic-input", choices=["branch", "concat"], default="concat")
    p.add_argument("--zero-init-kinematic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bounded-kinematic", action="store_true")
    p.add_argument("--use-cross-hand-attn", action="store_true")
    p.add_argument("--cross-attn-dim", type=int, default=64)
    p.add_argument("--cross-attn-dropout", type=float, default=0.1)
    p.add_argument("--cross-attn-scale", type=float, default=1.0)
    p.add_argument(
        "--cross-attn-direction",
        choices=["bidirectional", "right_queries_left", "left_queries_right"],
        default="bidirectional",
    )
    p.add_argument("--zero-init-cross-hand", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-hand-global-attn", action="store_true")
    p.add_argument("--hand-global-attn-dim", type=int, default=64)
    p.add_argument("--hand-global-attn-dropout", type=float, default=0.1)
    p.add_argument("--hand-global-attn-scale", type=float, default=1.0)
    p.add_argument("--use-ot-align", action="store_true")
    p.add_argument("--ot-align-dim", type=int, default=64)
    p.add_argument("--ot-epsilon", type=float, default=0.05)
    p.add_argument("--ot-iters", type=int, default=8)
    p.add_argument("--ot-scale", type=float, default=1.0)
    p.add_argument("--ot-dustbin", action="store_true")
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--scale", type=float, default=16.0)
    p.add_argument("--classifier-type", choices=["cosine", "subcenter"], default="cosine")
    p.add_argument("--num-subcenters", type=int, default=2)
    p.add_argument("--subcenter-reduce", choices=["max", "lse"], default="max")
    p.add_argument("--subcenter-lse-temperature", type=float, default=0.08)
    p.add_argument("--loss-mode", choices=["arcface", "confusion_rank"], default="arcface")
    p.add_argument("--conf-rank-parent-diagnostic", default=None)
    p.add_argument("--conf-rank-top-k", type=int, default=3)
    p.add_argument("--conf-rank-margin", type=float, default=0.05)
    p.add_argument("--conf-rank-lambda", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--swa-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--aug-repeats", type=int, default=10)
    p.add_argument(
        "--augment-mode",
        choices=[
            "fixed",
            "none",
            "rotate",
            "squeeze",
            "perspective",
            "arm_joint_rotate",
            "all",
            "all_noise",
            "fixed_clean_default",
            "fixed_stretch",
            "fixed_warp",
            "fixed_frame_drop",
            "fixed_shift",
            "fixed_affine",
            "fixed_noise",
            "fixed_recommended",
            "fixed_landmark_drop",
            "fixed_frame_drop_noise",
            "fixed_stretch_noise",
            "fixed_temporal",
            "fixed_affine_noise",
            "fixed_temporal_noise",
            "fixed_temporal_affine",
            "fixed_temporal_affine_noise",
            "fixed_all_components",
        ],
        default="fixed",
        help="Training augmentation family. 'fixed' preserves the original augment_fixed policy.",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--hand-scale-floor", type=float, default=1e-6)
    p.add_argument("--cache-workers", type=int, default=0)
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def train_epoch(model, loader, optimizer, criterion, scaler, device, amp, args, desc):
    model.train()
    total_loss = 0.0
    steps = 0
    for left, right, global_features, labels in tqdm(loader, desc=desc, leave=False, mininterval=5.0):
        left = left.to(device)
        right = right.to(device)
        global_features = global_features.to(device)
        labels = labels.to(device)
        if args.mixup_alpha > 0:
            left_m, right_m, global_m, ya, yb, lam = mixup_three(left, right, global_features, labels, alpha=args.mixup_alpha)
        else:
            left_m, right_m, global_m = left, right, global_features
            ya, yb, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(left_m, right_m, global_m)
            loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


def make_model(args, dims, num_classes):
    return OldLocalGlobalSweepArcFace(
        dims[0],
        dims[1],
        dims[2],
        num_classes,
        left_branch_dim=args.left_branch_dim,
        right_branch_dim=args.right_branch_dim,
        global_branch_dim=args.global_branch_dim,
        conv_kernel=args.conv_kernel,
        conv_layers=args.conv_layers,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        pool_mode=args.pool_mode,
        temporal_head=args.temporal_head,
        lite_kernel=args.lite_kernel,
        scale=args.scale,
        factorization_mode=args.factorization_mode,
        static_motion_gate_bias=args.static_motion_gate_bias,
        relation_dim=args.relation_dim,
        use_stream_dropout=args.use_stream_dropout,
        stream_dropout_left=args.stream_dropout_left,
        stream_dropout_right=args.stream_dropout_right,
        stream_dropout_global=args.stream_dropout_global,
        use_reliability_fusion=args.use_reliability_fusion,
        reliability_hidden_dim=args.reliability_hidden_dim,
        reliability_bias=args.reliability_bias,
        use_residual_reliability_fusion=args.use_residual_reliability_fusion,
        residual_rf_hidden_dim=args.residual_rf_hidden_dim,
        residual_rf_beta_init=args.residual_rf_beta_init,
        residual_rf_beta_mode=args.residual_rf_beta_mode,
        use_lrg_residual=args.use_lrg_residual,
        lrg_residual_gamma_init=args.lrg_residual_gamma_init,
        lrg_residual_gamma_mode=args.lrg_residual_gamma_mode,
        use_global_gate=args.use_global_gate,
        global_gate_hidden_dim=args.global_gate_hidden_dim,
        global_gate_bias=args.global_gate_bias,
        use_branch_gates=args.use_branch_gates,
        branch_gate_hidden_dim=args.branch_gate_hidden_dim,
        branch_gate_bias=args.branch_gate_bias,
        use_temporal_conv_bridge=args.use_temporal_conv_bridge,
        temporal_conv_bridge_kernel=args.temporal_conv_bridge_kernel,
        temporal_conv_bridge_dropout=args.temporal_conv_bridge_dropout,
        zero_init_temporal_conv_bridge=args.zero_init_temporal_conv_bridge,
        use_kinematic_residual=args.use_kinematic_residual,
        kinematic_hidden_dim=args.kinematic_hidden_dim,
        kinematic_dropout=args.kinematic_dropout,
        kinematic_scale=args.kinematic_scale,
        kinematic_input=args.kinematic_input,
        zero_init_kinematic=args.zero_init_kinematic,
        bounded_kinematic=args.bounded_kinematic,
        use_cross_hand_attn=args.use_cross_hand_attn,
        cross_attn_dim=args.cross_attn_dim,
        cross_attn_dropout=args.cross_attn_dropout,
        cross_attn_scale=args.cross_attn_scale,
        cross_attn_direction=args.cross_attn_direction,
        zero_init_cross_hand=args.zero_init_cross_hand,
        use_hand_global_attn=args.use_hand_global_attn,
        hand_global_attn_dim=args.hand_global_attn_dim,
        hand_global_attn_dropout=args.hand_global_attn_dropout,
        hand_global_attn_scale=args.hand_global_attn_scale,
        use_ot_align=args.use_ot_align,
        ot_align_dim=args.ot_align_dim,
        ot_epsilon=args.ot_epsilon,
        ot_iters=args.ot_iters,
        ot_scale=args.ot_scale,
        ot_dustbin=args.ot_dustbin,
        classifier_type=args.classifier_type,
        num_subcenters=args.num_subcenters,
        subcenter_reduce=args.subcenter_reduce,
        subcenter_lse_temperature=args.subcenter_lse_temperature,
    )


def get_global_gate_stats(model: nn.Module):
    gate_module = getattr(model, "global_gate", None)
    if gate_module is None or getattr(gate_module, "last_gate", None) is None:
        return None
    gate = gate_module.last_gate.float().cpu()
    return {
        "mean": float(gate.mean().item()),
        "min": float(gate.min().item()),
        "max": float(gate.max().item()),
    }


def get_branch_gate_stats(model: nn.Module):
    gate_module = getattr(model, "branch_gates", None)
    if gate_module is None or getattr(gate_module, "last_gate", None) is None:
        return None
    gate = gate_module.last_gate.float().cpu()
    return {
        "left_mean": float(gate[:, 0].mean().item()),
        "right_mean": float(gate[:, 1].mean().item()),
        "global_mean": float(gate[:, 2].mean().item()),
        "left_min": float(gate[:, 0].min().item()),
        "right_min": float(gate[:, 1].min().item()),
        "global_min": float(gate[:, 2].min().item()),
        "left_max": float(gate[:, 0].max().item()),
        "right_max": float(gate[:, 1].max().item()),
        "global_max": float(gate[:, 2].max().item()),
    }


def get_stream_dropout_stats(model: nn.Module):
    module = getattr(model, "stream_dropout", None)
    if module is None or getattr(module, "last_keep", None) is None:
        return None
    keep = module.last_keep.float().cpu()
    return {
        "left_keep_mean": float(keep[:, 0].mean().item()),
        "right_keep_mean": float(keep[:, 1].mean().item()),
        "global_keep_mean": float(keep[:, 2].mean().item()),
    }


def get_reliability_fusion_stats(model: nn.Module):
    module = getattr(model, "reliability_fusion", None) or getattr(model, "residual_reliability_fusion", None)
    if module is None or getattr(module, "last_weights", None) is None:
        return None
    weights = module.last_weights.float().cpu()
    stats = {
        "left_weight_mean": float(weights[:, 0].mean().item()),
        "right_weight_mean": float(weights[:, 1].mean().item()),
        "global_weight_mean": float(weights[:, 2].mean().item()),
        "left_weight_min": float(weights[:, 0].min().item()),
        "right_weight_min": float(weights[:, 1].min().item()),
        "global_weight_min": float(weights[:, 2].min().item()),
        "left_weight_max": float(weights[:, 0].max().item()),
        "right_weight_max": float(weights[:, 1].max().item()),
        "global_weight_max": float(weights[:, 2].max().item()),
    }
    beta = getattr(module, "last_beta", None)
    if beta is not None:
        stats["residual_beta"] = float(beta.float().cpu().item())
    return stats


def get_phase_pool_stats(model: nn.Module):
    module = getattr(model, "phase_pool", None)
    if module is None or getattr(module, "last_slot_weights", None) is None:
        return None
    weights = module.last_slot_weights.float().cpu()
    peak = weights.mean(dim=0).argmax(dim=0)
    return {
        "start_peak_frame": int(peak[0].item()),
        "middle_peak_frame": int(peak[1].item()),
        "end_peak_frame": int(peak[2].item()),
    }


def get_factorization_stats(model: nn.Module):
    stats = {"factorization_mode": getattr(model, "factorization_mode", "none")}
    if getattr(model, "use_lrg_residual", False):
        stats["use_lrg_residual"] = True
        stats["lrg_residual_gamma"] = float(model._lrg_gamma().detach().float().cpu().item())
        for prefix, module in [
            ("lrg_left", getattr(model, "lrg_left_stem", None)),
            ("lrg_right", getattr(model, "lrg_right_stem", None)),
            ("lrg_global", getattr(model, "lrg_global_stem", None)),
        ]:
            gate = getattr(module, "last_gate", None)
            if gate is not None:
                gate = gate.float().cpu()
                stats[f"{prefix}_static_motion_gate_mean"] = float(gate.mean().item())
    for prefix, module in [
        ("left", getattr(model, "left_stem", None)),
        ("right", getattr(model, "right_stem", None)),
        ("global", getattr(model, "global_stem", None)),
    ]:
        gate = getattr(module, "last_gate", None)
        if gate is not None:
            gate = gate.float().cpu()
            stats[f"{prefix}_static_motion_gate_mean"] = float(gate.mean().item())
            stats[f"{prefix}_static_motion_gate_min"] = float(gate.min().item())
            stats[f"{prefix}_static_motion_gate_max"] = float(gate.max().item())
    ds = getattr(model, "dominant_support_stems", None)
    if ds is not None and getattr(ds, "last_right_dominant", None) is not None:
        stats["right_dominant_ratio"] = float(ds.last_right_dominant.float().mean().cpu().item())
    return stats


def get_kinematic_diagnostics(model: nn.Module):
    adapters = []
    for name in [
        "kinematic_adapter",
        "left_kinematic_adapter",
        "right_kinematic_adapter",
        "global_kinematic_adapter",
    ]:
        adapter = getattr(model, name, None)
        if adapter is not None and getattr(adapter, "last_diag", None) is not None:
            adapters.append(adapter.last_diag)
    if not adapters:
        return None
    keys = adapters[0].keys()
    return {key: float(np.mean([diag[key] for diag in adapters])) for key in keys}


def get_cross_hand_diagnostics(model: nn.Module):
    module = getattr(model, "cross_hand_attn", None)
    if module is None or getattr(module, "last_diag", None) is None:
        return None
    return module.last_diag


def get_hand_global_diagnostics(model: nn.Module):
    module = getattr(model, "hand_global_attn", None)
    if module is None or getattr(module, "last_diag", None) is None:
        return None
    return module.last_diag


def get_ot_diagnostics(model: nn.Module):
    module = getattr(model, "ot_align", None)
    if module is None or getattr(module, "last_diag", None) is None:
        return None
    return module.last_diag


def get_subcenter_usage(model: nn.Module, loader, device, amp: bool = False):
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, SubcenterCosineClassifier):
        return None
    counts = torch.zeros(classifier.num_classes, classifier.num_subcenters, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for left, right, global_features, labels in loader:
            left = left.to(device)
            right = right.to(device)
            global_features = global_features.to(device)
            labels = labels.long()
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                emb = model.forward_features(left, right, global_features)
                cosines = classifier.subcenter_cosines(emb)
            true_cosines = cosines[torch.arange(labels.numel(), device=cosines.device), labels.to(cosines.device)]
            chosen = true_cosines.argmax(dim=-1).cpu()
            for label, center in zip(labels, chosen):
                if 0 <= int(label) < classifier.num_classes:
                    counts[int(label), int(center)] += 1
    active = counts > 0
    classes_with_multiple = (active.sum(dim=1) > 1).sum().item()
    return {
        "num_subcenters": int(classifier.num_subcenters),
        "active_center_fraction": float(active.float().mean().item()),
        "classes_with_multiple_active_centers": int(classes_with_multiple),
        "classes_with_any_active_center": int((active.sum(dim=1) > 0).sum().item()),
        "center_usage_totals": [int(x) for x in counts.sum(dim=0).tolist()],
    }


def experiment_name(args) -> str:
    if args.use_cross_hand_attn:
        return "E4a_B6_zero_init_cross_hand_attention"
    if args.classifier_type == "subcenter":
        return "B6_MP_multi_prototype_classifier"
    if args.use_stream_dropout:
        return "B6_SD_stream_dropout"
    if args.use_reliability_fusion:
        return "B6_RF_stream_reliability_fusion"
    if args.pool_mode == "phase":
        return "B6_PP_phase_aware_pooling"
    if args.use_temporal_conv_bridge:
        return "B6_TC_temporal_conv_bridge"
    if args.factorization_mode != "none":
        return f"B6_{args.factorization_mode}"
    if args.use_lrg_residual:
        return "B6_LRG_residual"
    if args.use_residual_reliability_fusion:
        return "B6_RF_residual"
    if args.use_kinematic_residual:
        return "E6_B6_kinematic_residual"
    return "old_localglobal_sweep"


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    output_prefix = args.output_prefix or f"wlasl{args.num_glosses}_old_localglobal_{args.pool_mode}"
    cache_dir = args.cache_dir or os.path.join("rework_model", "cache", f"wlasl{args.num_glosses}_old_localglobal_sweep")

    use_labels_only = (not args.force_cache) and args.augment_mode == "fixed" and feature_cache_ready(cache_dir, args.feature_kind, args.aug_repeats)
    if use_labels_only:
        print(f"Using complete {args.feature_kind} cache at {feature_cache_dir(cache_dir, args.feature_kind)}; loading labels only.")

    splits, actions, _ = load_subset_raw_for_source(
        args.data_dir,
        args.json_path,
        args.num_glosses,
        args.action_source,
        args.limit_samples,
        labels_only=use_labels_only,
    )
    y_tr = np.asarray(splits["train"]["labels"], dtype=np.int64)
    y_val = np.asarray(splits["val"]["labels"], dtype=np.int64)
    y_test = np.asarray(splits["test"]["labels"], dtype=np.int64)
    num_classes = len(actions)

    left_tr, right_tr, global_tr = encode_feature_parts(
        splits["train"]["raw"],
        "train",
        cache_dir,
        args.feature_kind,
        force=args.force_cache,
        hand_scale_floor=args.hand_scale_floor,
    )
    left_val, right_val, global_val = encode_feature_parts(
        splits["val"]["raw"],
        "val",
        cache_dir,
        args.feature_kind,
        force=args.force_cache,
        hand_scale_floor=args.hand_scale_floor,
    )
    left_test, right_test, global_test = encode_feature_parts(
        splits["test"]["raw"],
        "test",
        cache_dir,
        args.feature_kind,
        force=args.force_cache,
        hand_scale_floor=args.hand_scale_floor,
    )
    dims = (int(left_tr.shape[-1]), int(right_tr.shape[-1]), int(global_tr.shape[-1]))
    print(f"{args.feature_kind} feature dims: left={dims[0]} right={dims[1]} global={dims[2]}")
    if args.feature_kind == "old" and dims != (EXPECTED_LEFT_DIM, EXPECTED_RIGHT_DIM, EXPECTED_GLOBAL_DIM):
        print(f"WARNING: expected old dims 165/165/23, got {dims[0]}/{dims[1]}/{dims[2]}. Continuing.")
    if args.feature_kind == "old_no_palm" and dims != (EXPECTED_LEFT_DIM - 3, EXPECTED_RIGHT_DIM - 3, EXPECTED_GLOBAL_DIM):
        print(f"WARNING: expected old_no_palm dims 162/162/23, got {dims[0]}/{dims[1]}/{dims[2]}. Continuing.")
    if args.feature_kind == "old_palmnormvec" and dims != (
        EXPECTED_LEFT_DIM + PALMNORMVEC_DIM,
        EXPECTED_RIGHT_DIM + PALMNORMVEC_DIM,
        EXPECTED_GLOBAL_DIM,
    ):
        print(
            "WARNING: expected old_palmnormvec dims "
            f"{EXPECTED_LEFT_DIM + PALMNORMVEC_DIM}/{EXPECTED_RIGHT_DIM + PALMNORMVEC_DIM}/23, "
            f"got {dims[0]}/{dims[1]}/{dims[2]}. Continuing."
        )

    left_aug, right_aug, global_aug, aug_y = build_cached_feature_augments(
        splits["train"]["raw"],
        y_tr,
        dims[0],
        dims[1],
        dims[2],
        args.aug_repeats,
        cache_dir,
        args.feature_kind,
        force=args.force_cache,
        augment_mode=args.augment_mode,
        hand_scale_floor=args.hand_scale_floor,
        cache_workers=args.cache_workers,
    )

    train_len = len(y_tr) + (0 if aug_y is None else len(aug_y))
    train_loader = DataLoader(
        CachedAugmentedPartDataset(left_tr, right_tr, global_tr, y_tr, left_aug, right_aug, global_aug, aug_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=train_len >= args.batch_size,
    )
    val_loader = make_loader(left_val, right_val, global_val, y_val, batch_size=args.batch_size)
    test_loader = make_loader(left_test, right_test, global_test, y_test, batch_size=args.batch_size)

    model = make_model(args, dims, num_classes).to(device)
    params = sum(p.numel() for p in model.parameters())
    d_model = args.left_branch_dim + args.right_branch_dim + args.global_branch_dim
    print(
        "Startup: "
        f"feature_kind={args.feature_kind} num_glosses={args.num_glosses} pool={args.pool_mode} "
        f"dims={dims[0]}/{dims[1]}/{dims[2]} branches={args.left_branch_dim}/{args.right_branch_dim}/{args.global_branch_dim} "
        f"d_model={d_model} conv_kernel={args.conv_kernel} conv_layers={args.conv_layers} "
        f"temporal_head={args.temporal_head} "
        f"classifier={args.classifier_type} subcenters={args.num_subcenters} "
        f"augment_mode={args.augment_mode} "
        f"hand_scale_floor={args.hand_scale_floor} "
        f"use_stream_dropout={args.use_stream_dropout} use_reliability_fusion={args.use_reliability_fusion} "
        f"use_residual_rf={args.use_residual_reliability_fusion} use_lrg_residual={args.use_lrg_residual} "
        f"use_temporal_conv_bridge={args.use_temporal_conv_bridge} "
        f"mixup_alpha={args.mixup_alpha} use_global_gate={args.use_global_gate} "
        f"use_branch_gates={args.use_branch_gates} use_kinematic_residual={args.use_kinematic_residual} "
        f"use_cross_hand_attn={args.use_cross_hand_attn} params={params}"
    )

    class_weights = get_class_weights(y_tr, num_classes).to(device)
    confused_neighbors = None
    confusion_rank_summary = None
    if args.loss_mode == "confusion_rank":
        confused_neighbors, confusion_rank_summary = build_confusion_neighbors_from_diagnostic(
            args.conf_rank_parent_diagnostic,
            num_classes,
            args.conf_rank_top_k,
        )
        confused_neighbors = confused_neighbors.to(device)
        print(
            "ConfusionRank: "
            f"top_k={args.conf_rank_top_k} margin={args.conf_rank_margin} lambda={args.conf_rank_lambda} "
            f"classes_with_neighbors={confusion_rank_summary['classes_with_neighbors']}"
        )
    criterion = ConfusionRankArcFaceLoss(
        scale=args.scale,
        margin=args.margin,
        cw=class_weights,
        confused_neighbors=confused_neighbors,
        rank_lambda=args.conf_rank_lambda if args.loss_mode == "confusion_rank" else 0.0,
        rank_margin=args.conf_rank_margin,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    Path("rework_model").mkdir(exist_ok=True)
    pre_path = Path("rework_model") / f"{output_prefix}_single.pth"
    swa_path = Path("rework_model") / f"{output_prefix}_swa.pth"
    best_by_val_path = Path("rework_model") / f"{output_prefix}_best_by_val.pth"

    history = []
    best_val = -1.0
    stale = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, amp, args, f"Epoch {epoch+1}/{args.epochs}")
        val_acc, val_loss = eval_accuracy(model, val_loader, device, criterion=criterion, amp=amp)
        scheduler.step(val_loss)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss}
        print(json.dumps(row))
        history.append(row)
        if val_acc > best_val:
            best_val = val_acc
            stale = 0
            torch.save(model.state_dict(), pre_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(pre_path, map_location=device, weights_only=True))
    pre_probs, _ = collect_probs(model, test_loader, device, amp=amp, tta_passes=0)
    pre_top1, pre_top5 = topk_metrics(pre_probs, y_test)
    pre_val, _ = eval_accuracy(model, val_loader, device, amp=amp)

    swa_metrics = None
    selected_by_val_name = "pre-SWA"
    selected_by_val_state = torch.load(pre_path, map_location=device, weights_only=True)
    selected_by_val = {"val_top1": float(pre_val), "test_top1": float(pre_top1), "test_top5": float(pre_top5)}

    if args.swa_epochs > 0:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        for epoch in range(args.swa_epochs):
            _ = train_epoch(model, train_loader, optimizer, criterion, scaler, device, amp, args, f"SWA {epoch+1}/{args.swa_epochs}")
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"  SWA ep {epoch+1}/{args.swa_epochs}")
        torch.save(swa_model.module.state_dict(), swa_path)
        swa_probs, _ = collect_probs(swa_model, test_loader, device, amp=amp, tta_passes=0)
        swa_top1, swa_top5 = topk_metrics(swa_probs, y_test)
        swa_val, _ = eval_accuracy(swa_model, val_loader, device, amp=amp)
        swa_metrics = {"val_top1": float(swa_val), "test_top1": float(swa_top1), "test_top5": float(swa_top5)}
        if swa_val >= pre_val:
            selected_by_val_name = "SWA"
            selected_by_val_state = torch.load(swa_path, map_location=device, weights_only=True)
            selected_by_val = dict(swa_metrics)

    torch.save(selected_by_val_state, best_by_val_path)

    selected_model = make_model(args, dims, num_classes).to(device)
    selected_model.load_state_dict(selected_by_val_state)
    run_cache = feature_cache_dir(cache_dir, args.feature_kind) / output_prefix
    run_cache.mkdir(parents=True, exist_ok=True)
    val_logits_path = run_cache / "val_logits.npy"
    test_logits_path = run_cache / "test_logits.npy"
    val_labels_path = run_cache / "val_labels.npy"
    test_labels_path = run_cache / "test_labels.npy"
    val_logits = collect_logits(selected_model, val_loader, device, amp=amp)
    test_logits = collect_logits(selected_model, test_loader, device, amp=amp)
    np.save(val_logits_path, val_logits)
    np.save(test_logits_path, test_logits)
    np.save(val_labels_path, y_val)
    np.save(test_labels_path, y_test)
    stream_dropout_stats = get_stream_dropout_stats(selected_model)
    reliability_fusion_stats = get_reliability_fusion_stats(selected_model)
    phase_pool_stats = get_phase_pool_stats(selected_model)
    factorization_stats = get_factorization_stats(selected_model)
    gate_stats = get_global_gate_stats(selected_model)
    branch_gate_stats = get_branch_gate_stats(selected_model)
    kinematic_diagnostics = get_kinematic_diagnostics(selected_model)
    cross_hand_diagnostics = get_cross_hand_diagnostics(selected_model)
    hand_global_diagnostics = get_hand_global_diagnostics(selected_model)
    ot_diagnostics = get_ot_diagnostics(selected_model)
    subcenter_usage = get_subcenter_usage(selected_model, test_loader, device, amp=amp)

    result = {
        "model": "OldLocalGlobalSweepArcFace",
        "experiment": experiment_name(args),
        "base_config": "B6" if (
            args.left_branch_dim == 96
            and args.right_branch_dim == 96
            and args.global_branch_dim == 96
            and abs(args.dropout - 0.2) < 1e-8
            and args.pool_mode == "mean"
        ) else None,
        "num_glosses": int(args.num_glosses),
        "num_classes": int(num_classes),
        "seed": int(args.seed),
        "action_source": args.action_source,
        "hand_scale_floor": float(args.hand_scale_floor),
        "actions": actions,
        "feature_type": "old_frame_part_aware" if args.feature_kind == "old" else (
            "old_frame_part_aware_no_palm_normal" if args.feature_kind == "old_no_palm" else (
                "old_frame_part_aware_plus_palmnormvec" if args.feature_kind == "old_palmnormvec" else f"stage2_{args.feature_kind}"
            )
        ),
        "feature_kind": args.feature_kind,
        "feature_dims": {"left": dims[0], "right": dims[1], "global": dims[2]},
        "left_branch_dim": int(args.left_branch_dim),
        "right_branch_dim": int(args.right_branch_dim),
        "global_branch_dim": int(args.global_branch_dim),
        "d_model": int(d_model),
        "conv_kernel": int(args.conv_kernel),
        "conv_layers": int(args.conv_layers),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "ff_dim": int(args.ff_dim),
        "dropout": float(args.dropout),
        "pool_mode": args.pool_mode,
        "temporal_head": args.temporal_head,
        "lite_kernel": int(args.lite_kernel),
        "factorization_mode": args.factorization_mode,
        "static_motion_gate_bias": float(args.static_motion_gate_bias),
        "relation_dim": int(args.relation_dim),
        "factorization_stats": factorization_stats,
        "use_stream_dropout": bool(args.use_stream_dropout),
        "stream_dropout_left": float(args.stream_dropout_left),
        "stream_dropout_right": float(args.stream_dropout_right),
        "stream_dropout_global": float(args.stream_dropout_global),
        "stream_dropout_stats": stream_dropout_stats,
        "use_reliability_fusion": bool(args.use_reliability_fusion),
        "reliability_hidden_dim": int(args.reliability_hidden_dim),
        "reliability_bias": float(args.reliability_bias),
        "reliability_fusion_stats": reliability_fusion_stats,
        "use_residual_reliability_fusion": bool(args.use_residual_reliability_fusion),
        "residual_rf_hidden_dim": int(args.residual_rf_hidden_dim),
        "residual_rf_beta_init": float(args.residual_rf_beta_init),
        "residual_rf_beta_mode": args.residual_rf_beta_mode,
        "use_lrg_residual": bool(args.use_lrg_residual),
        "lrg_residual_gamma_init": float(args.lrg_residual_gamma_init),
        "lrg_residual_gamma_mode": args.lrg_residual_gamma_mode,
        "phase_pool_stats": phase_pool_stats,
        "use_temporal_conv_bridge": bool(args.use_temporal_conv_bridge),
        "temporal_conv_bridge_kernel": int(args.temporal_conv_bridge_kernel),
        "temporal_conv_bridge_dropout": float(args.temporal_conv_bridge_dropout),
        "zero_init_temporal_conv_bridge": bool(args.zero_init_temporal_conv_bridge),
        "classifier_type": args.classifier_type,
        "num_subcenters": int(args.num_subcenters),
        "subcenter_reduce": args.subcenter_reduce,
        "subcenter_lse_temperature": float(args.subcenter_lse_temperature),
        "subcenter_usage": subcenter_usage,
        "use_global_gate": bool(args.use_global_gate),
        "global_gate_hidden_dim": int(args.global_gate_hidden_dim),
        "global_gate_bias": float(args.global_gate_bias),
        "global_gate_stats": gate_stats,
        "use_branch_gates": bool(args.use_branch_gates),
        "branch_gate_hidden_dim": int(args.branch_gate_hidden_dim),
        "branch_gate_bias": float(args.branch_gate_bias),
        "branch_gate_stats": branch_gate_stats,
        "use_kinematic_residual": bool(args.use_kinematic_residual),
        "kinematic_hidden_dim": int(args.kinematic_hidden_dim),
        "kinematic_dropout": float(args.kinematic_dropout),
        "kinematic_scale": float(args.kinematic_scale),
        "kinematic_input": args.kinematic_input,
        "zero_init_kinematic": bool(args.zero_init_kinematic),
        "bounded_kinematic": bool(args.bounded_kinematic),
        "kinematic_diagnostics": kinematic_diagnostics,
        "use_cross_hand_attn": bool(args.use_cross_hand_attn),
        "cross_attn_dim": int(args.cross_attn_dim),
        "cross_attn_dropout": float(args.cross_attn_dropout),
        "cross_attn_scale": float(args.cross_attn_scale),
        "cross_attn_direction": args.cross_attn_direction,
        "zero_init_cross_hand": bool(args.zero_init_cross_hand),
        "cross_hand_diagnostics": cross_hand_diagnostics,
        "use_hand_global_attn": bool(args.use_hand_global_attn),
        "hand_global_diagnostics": hand_global_diagnostics,
        "use_ot_align": bool(args.use_ot_align),
        "ot_align_dim": int(args.ot_align_dim),
        "ot_epsilon": float(args.ot_epsilon),
        "ot_iters": int(args.ot_iters),
        "ot_scale": float(args.ot_scale),
        "ot_dustbin": bool(args.ot_dustbin),
        "ot_diagnostics": ot_diagnostics,
        "attn_entropy_r2l": None if cross_hand_diagnostics is None else cross_hand_diagnostics.get("attn_entropy_r2l"),
        "attn_entropy_l2r": None if cross_hand_diagnostics is None else cross_hand_diagnostics.get("attn_entropy_l2r"),
        "left_update_norm_mean": None if cross_hand_diagnostics is None else cross_hand_diagnostics.get("left_update_norm_mean"),
        "right_update_norm_mean": None if cross_hand_diagnostics is None else cross_hand_diagnostics.get("right_update_norm_mean"),
        "left_norm_mean": None if cross_hand_diagnostics is None else cross_hand_diagnostics.get("left_norm_mean"),
        "right_norm_mean": None if cross_hand_diagnostics is None else cross_hand_diagnostics.get("right_norm_mean"),
        "left_update_to_left_ratio": None if cross_hand_diagnostics is None else cross_hand_diagnostics.get("left_update_to_left_ratio"),
        "right_update_to_right_ratio": None if cross_hand_diagnostics is None else cross_hand_diagnostics.get("right_update_to_right_ratio"),
        "mixup_alpha": float(args.mixup_alpha),
        "loss_mode": args.loss_mode,
        "conf_rank_parent_diagnostic": args.conf_rank_parent_diagnostic,
        "conf_rank_top_k": int(args.conf_rank_top_k),
        "conf_rank_margin": float(args.conf_rank_margin),
        "conf_rank_lambda": float(args.conf_rank_lambda),
        "confusion_rank_summary": confusion_rank_summary,
        "margin": float(args.margin),
        "arcface_margin": float(args.margin),
        "scale": float(args.scale),
        "arcface_scale": float(args.scale),
        "aug_repeats": int(args.aug_repeats),
        "params": int(params),
        "samples": {"train": int(len(y_tr)), "val": int(len(y_val)), "test": int(len(y_test))},
        "pre_swa": {"val_top1": float(pre_val), "test_top1": float(pre_top1), "test_top5": float(pre_top5)},
        "pre_swa_metrics": {"val_top1": float(pre_val), "test_top1": float(pre_top1), "test_top5": float(pre_top5)},
        "swa": swa_metrics,
        "swa_metrics": swa_metrics,
        "fixed_swa_metrics": swa_metrics,
        "selected_by_val": selected_by_val_name,
        "selected_by_val_metrics": selected_by_val,
        "checkpoints": {
            "pre_swa": str(pre_path),
            "swa": str(swa_path) if args.swa_epochs > 0 else None,
            "best_by_val": str(best_by_val_path),
        },
        "logits": {
            "val_logits": str(val_logits_path),
            "test_logits": str(test_logits_path),
            "val_labels": str(val_labels_path),
            "test_labels": str(test_labels_path),
        },
        "history": history,
    }

    Path("diagnostic").mkdir(exist_ok=True)
    result_path = Path("diagnostic") / f"{output_prefix}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("")
    print("OldLocalGlobalSweepArcFace")
    print(f"  Selected by val: {selected_by_val_name}")
    print(f"  Pre-SWA: Val {pre_val:.2f}% Test top1 {pre_top1:.2f}% Test top5 {pre_top5:.2f}%")
    if swa_metrics is not None:
        print(f"  SWA:     Val {swa_metrics['val_top1']:.2f}% Test top1 {swa_metrics['test_top1']:.2f}% Test top5 {swa_metrics['test_top5']:.2f}%")
    print(f"  Params: {params:,}")
    print(f"  Checkpoint best-by-val: {best_by_val_path}")
    print(f"  Result JSON: {result_path}")


if __name__ == "__main__":
    main()
