"""Train a morphology-trajectory coordinate expert for WLASL.

MorphTrajExpert tests a different hypothesis from B6/FELF/TopoExpert:
handshape, body-relative trajectory, and palm orientation should be represented
as separate factor branches and fused only after each factor has its own temporal
encoding. This avoids early mixing of static morphology with motion/locus.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from wlasl_geometry_utils import (
    FACE_INDICES,
    HAND_BONES,
    GeometryConfig,
    compute_palm_normal,
    _prepare_parts,
)
from wlasl_train_local_global_arcface import ArcFaceLoss, CosineClassifier, topk_metrics
from wlasl_train_local_global_arcface_subset import DATA_DIR, JSON_PATH, SEQUENCE_LENGTH
from wlasl_train_old_localglobal_sweep_subset import load_subset_raw_for_source
from wlasl_train_streams_arcface import augment_fixed, get_class_weights


def _encode_one_factor_job(job):
    seq, rectify_hands, feature_version, augment = job
    arr = np.asarray(seq, dtype=np.float32)
    if augment:
        arr = augment_fixed(arr)
    return sequence_factor_features(arr, rectify_hands=rectify_hands, feature_version=feature_version)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    return topk_metrics(softmax_probs(logits), labels.astype(np.int64))


def load_diag_logits(path: str | None, split: str) -> np.ndarray | None:
    if not path or str(path).lower() in {"none", "null", "skip"}:
        return None
    diag = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    logits_path = Path((diag.get("logits") or {}).get(f"{split}_logits", ""))
    if not logits_path.exists():
        raise FileNotFoundError(f"{path} missing existing {split}_logits: {logits_path}")
    return np.load(logits_path).astype(np.float32)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def hand_morphology(hand: np.ndarray) -> np.ndarray:
    valid = float(np.any(np.abs(hand) > 1e-8))
    if valid <= 0:
        return np.zeros(155, dtype=np.float32)
    palm_center = hand[[0, 5, 9, 13, 17]].mean(axis=0).astype(np.float32)
    scale = float(np.linalg.norm(hand[0] - hand[9]))
    if scale <= 1e-6:
        scale = 1.0
    rel = ((hand - palm_center[None, :]) / scale).astype(np.float32)
    bone_vec = np.asarray([rel[c] - rel[p] for p, c in HAND_BONES], dtype=np.float32)
    bone_len = np.linalg.norm(bone_vec, axis=1).astype(np.float32)
    fingertips = rel[[4, 8, 12, 16, 20]]
    spread = []
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            spread.append(np.linalg.norm(fingertips[i] - fingertips[j]).astype(np.float32))
    return np.concatenate(
        [
            rel.reshape(-1),
            bone_vec.reshape(-1),
            bone_len,
            np.asarray(spread, dtype=np.float32),
            np.array([valid, scale], dtype=np.float32),
        ]
    ).astype(np.float32)


def hand_orientation(hand: np.ndarray, label: str) -> np.ndarray:
    valid = float(np.any(np.abs(hand) > 1e-8))
    if valid <= 0:
        return np.zeros(10, dtype=np.float32)
    wrist = hand[0]
    x_axis = unit(hand[5] - hand[17])
    y_axis = unit(hand[9] - wrist)
    z_axis = compute_palm_normal(hand, label)
    return np.concatenate([x_axis, y_axis, z_axis, np.array([valid], dtype=np.float32)]).astype(np.float32)


def frame_factor_features(frame: np.ndarray, rectify_hands: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = GeometryConfig(name="morph_traj", rectification=rectify_hands)
    pose, lh, rh = _prepare_parts(frame, cfg)
    face = pose[FACE_INDICES]
    face_center = face.mean(axis=0).astype(np.float32)
    face_scale = float(np.linalg.norm(face[1] - face[2]))
    if face_scale <= 1e-6:
        face_scale = 1.0
    shoulder_center = ((pose[11] + pose[12]) * 0.5).astype(np.float32)
    body_scale = float(np.linalg.norm(pose[11] - pose[12]))
    if body_scale <= 1e-6:
        body_scale = face_scale
    if body_scale <= 1e-6:
        body_scale = 1.0

    morph = np.concatenate([hand_morphology(lh), hand_morphology(rh)]).astype(np.float32)
    orient = np.concatenate([hand_orientation(lh, "Left"), hand_orientation(rh, "Right")]).astype(np.float32)

    def hand_traj(hand: np.ndarray, other: np.ndarray) -> np.ndarray:
        valid = float(np.any(np.abs(hand) > 1e-8))
        wrist = hand[0].astype(np.float32)
        palm = hand[[0, 5, 9, 13, 17]].mean(axis=0).astype(np.float32)
        other_wrist = other[0].astype(np.float32)
        return np.concatenate(
            [
                ((wrist - shoulder_center) / body_scale).astype(np.float32),
                ((palm - shoulder_center) / body_scale).astype(np.float32),
                ((wrist - face_center) / face_scale).astype(np.float32),
                ((palm - face_center) / face_scale).astype(np.float32),
                ((wrist - other_wrist) / body_scale).astype(np.float32),
                np.array(
                    [
                        np.linalg.norm(wrist - face_center) / body_scale,
                        np.linalg.norm(palm - face_center) / body_scale,
                        valid,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

    left_traj = hand_traj(lh, rh)
    right_traj = hand_traj(rh, lh)
    inter = np.concatenate(
        [
            ((lh[0] - rh[0]) / body_scale).astype(np.float32),
            ((lh[[0, 5, 9, 13, 17]].mean(axis=0) - rh[[0, 5, 9, 13, 17]].mean(axis=0)) / body_scale).astype(np.float32),
            np.array([np.linalg.norm(lh[0] - rh[0]) / body_scale], dtype=np.float32),
        ]
    )
    traj = np.concatenate([left_traj, right_traj, inter]).astype(np.float32)
    return morph, traj, orient


def sequence_factor_features(seq: np.ndarray, rectify_hands: bool = True, feature_version: str = "v1") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    morph_rows, traj_base_rows, orient_base_rows = [], [], []
    for t in range(SEQUENCE_LENGTH):
        morph, traj, orient = frame_factor_features(seq[t], rectify_hands=rectify_hands)
        morph_rows.append(morph)
        traj_base_rows.append(traj)
        orient_base_rows.append(orient)
    morph_arr = np.asarray(morph_rows, dtype=np.float32)
    traj_base = np.asarray(traj_base_rows, dtype=np.float32)
    orient_base = np.asarray(orient_base_rows, dtype=np.float32)

    traj_vel = np.zeros_like(traj_base)
    traj_vel[1:] = traj_base[1:] - traj_base[:-1]
    traj_acc = np.zeros_like(traj_base)
    traj_acc[1:] = traj_vel[1:] - traj_vel[:-1]
    orient_delta = np.zeros_like(orient_base)
    orient_delta[1:] = orient_base[1:] - orient_base[:-1]

    # Curvature proxy from the two wrist-body velocity vectors.
    curv = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
    for t in range(2, SEQUENCE_LENGTH):
        for hand_idx, start in enumerate([0, 21]):
            v1 = traj_vel[t - 1, start : start + 3]
            v2 = traj_vel[t, start : start + 3]
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom > 1e-8:
                curv[t, hand_idx] = 1.0 - float(np.dot(v1, v2) / denom)

    traj_parts = [traj_base, traj_vel, traj_acc, curv]
    if feature_version == "v3":
        # MT-v3 adds signed path geometry instead of relying only on local deltas.
        # Key 3D trajectory blocks: left/right wrist+palm in body frame, plus cross-hand paths.
        key_slices = [(0, 3), (3, 6), (18, 21), (21, 24), (36, 39), (39, 42)]
        key_pos = np.concatenate([traj_base[:, a:b] for a, b in key_slices], axis=1).astype(np.float32)
        key_vel = np.zeros_like(key_pos)
        key_vel[1:] = key_pos[1:] - key_pos[:-1]
        speed = np.linalg.norm(key_vel.reshape(SEQUENCE_LENGTH, -1, 3), axis=2).astype(np.float32)
        direction = np.zeros_like(key_vel)
        denom = np.linalg.norm(key_vel.reshape(SEQUENCE_LENGTH, -1, 3), axis=2, keepdims=True).clip(1e-6)
        direction = (key_vel.reshape(SEQUENCE_LENGTH, -1, 3) / denom).reshape(SEQUENCE_LENGTH, -1).astype(np.float32)

        first_disp = (key_pos - key_pos[0:1]).astype(np.float32)
        mid = SEQUENCE_LENGTH // 2
        mid_disp = (key_pos - key_pos[mid : mid + 1]).astype(np.float32)
        last_target = (key_pos[-1:] - key_pos).astype(np.float32)

        # Direction-change and curvature for every key path.
        key_curv = np.zeros((SEQUENCE_LENGTH, len(key_slices)), dtype=np.float32)
        for t in range(2, SEQUENCE_LENGTH):
            prev = key_vel[t - 1].reshape(-1, 3)
            curr = key_vel[t].reshape(-1, 3)
            num = (prev * curr).sum(axis=1)
            den = (np.linalg.norm(prev, axis=1) * np.linalg.norm(curr, axis=1)).clip(1e-6)
            key_curv[t] = 1.0 - (num / den)

        # Broadcast global phase/path summaries to every frame so the same trunk can use them.
        first = key_pos[0]
        middle = key_pos[mid]
        last = key_pos[-1]
        total_disp = last - first
        first_middle = middle - first
        middle_last = last - middle
        path_length = speed.sum(axis=0)
        straight = np.linalg.norm(total_disp.reshape(-1, 3), axis=1)
        efficiency = straight / path_length.clip(1e-6)
        summary = np.concatenate([first, middle, last, total_disp, first_middle, middle_last, path_length, efficiency]).astype(np.float32)
        summary_seq = np.repeat(summary[None, :], SEQUENCE_LENGTH, axis=0)

        traj_parts.extend([key_pos, key_vel, speed, direction, first_disp, mid_disp, last_target, key_curv, summary_seq])
    elif feature_version != "v1":
        raise ValueError(f"Unknown MorphTraj feature_version: {feature_version}")

    traj_arr = np.concatenate(traj_parts, axis=1).astype(np.float32)
    orient_arr = np.concatenate([orient_base, orient_delta], axis=1).astype(np.float32)
    return morph_arr, traj_arr, orient_arr


def encode_factors(raw_list, cache_path: Path, split: str, force: bool, rectify_hands: bool, feature_version: str, workers: int = 1):
    paths = {
        "morph": cache_path / f"{split}_morph.npy",
        "traj": cache_path / f"{split}_traj.npy",
        "orient": cache_path / f"{split}_orient.npy",
    }
    if not force and all(p.exists() for p in paths.values()):
        print(f"Using cached {split} MorphTraj factors.")
        return tuple(np.load(paths[k], mmap_mode="r") for k in ["morph", "traj", "orient"])

    cache_path.mkdir(parents=True, exist_ok=True)
    first = sequence_factor_features(np.asarray(raw_list[0], dtype=np.float32), rectify_hands=rectify_hands, feature_version=feature_version)
    arrays = {
        "morph": np.lib.format.open_memmap(paths["morph"], mode="w+", dtype=np.float32, shape=(len(raw_list), *first[0].shape)),
        "traj": np.lib.format.open_memmap(paths["traj"], mode="w+", dtype=np.float32, shape=(len(raw_list), *first[1].shape)),
        "orient": np.lib.format.open_memmap(paths["orient"], mode="w+", dtype=np.float32, shape=(len(raw_list), *first[2].shape)),
    }
    arrays["morph"][0], arrays["traj"][0], arrays["orient"][0] = first
    jobs = [(raw_list[i], rectify_hands, feature_version, False) for i in range(1, len(raw_list))]
    if workers and workers > 1 and jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for offset, (morph, traj, orient) in enumerate(tqdm(ex.map(_encode_one_factor_job, jobs), total=len(jobs), desc=f"Encode {split} MorphTraj w{workers}"), start=1):
                arrays["morph"][offset] = morph
                arrays["traj"][offset] = traj
                arrays["orient"][offset] = orient
    else:
        for i in tqdm(range(1, len(raw_list)), desc=f"Encode {split} MorphTraj"):
            morph, traj, orient = sequence_factor_features(np.asarray(raw_list[i], dtype=np.float32), rectify_hands=rectify_hands, feature_version=feature_version)
            arrays["morph"][i] = morph
            arrays["traj"][i] = traj
            arrays["orient"][i] = orient
    del arrays
    return tuple(np.load(paths[k], mmap_mode="r") for k in ["morph", "traj", "orient"])


def build_aug_cache(raw_train, y_train, cache_path: Path, repeats: int, force: bool, rectify_hands: bool, feature_version: str, workers: int = 1):
    if repeats <= 0:
        return None, None, None, None
    paths = {
        "morph": cache_path / f"aug_r{repeats}_morph.npy",
        "traj": cache_path / f"aug_r{repeats}_traj.npy",
        "orient": cache_path / f"aug_r{repeats}_orient.npy",
        "labels": cache_path / f"aug_r{repeats}_labels.npy",
    }
    if not force and all(p.exists() for p in paths.values()):
        print(f"Using cached MorphTraj augmentation r{repeats}.")
        return (
            np.load(paths["morph"], mmap_mode="r"),
            np.load(paths["traj"], mmap_mode="r"),
            np.load(paths["orient"], mmap_mode="r"),
            np.load(paths["labels"], mmap_mode="r"),
        )

    first = sequence_factor_features(augment_fixed(raw_train[0]), rectify_hands=rectify_hands, feature_version=feature_version)
    total = len(raw_train) * repeats
    morph_aug = np.lib.format.open_memmap(paths["morph"], mode="w+", dtype=np.float32, shape=(total, *first[0].shape))
    traj_aug = np.lib.format.open_memmap(paths["traj"], mode="w+", dtype=np.float32, shape=(total, *first[1].shape))
    orient_aug = np.lib.format.open_memmap(paths["orient"], mode="w+", dtype=np.float32, shape=(total, *first[2].shape))
    y_aug = np.empty(total, dtype=np.int64)

    jobs = []
    labels = []
    for i, seq in enumerate(raw_train):
        for _ in range(repeats):
            jobs.append((seq, rectify_hands, feature_version, True))
            labels.append(y_train[i])
    if workers and workers > 1 and jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for cursor, (morph, traj, orient) in enumerate(tqdm(ex.map(_encode_one_factor_job, jobs), total=len(jobs), desc=f"Aug MorphTraj r{repeats} w{workers}")):
                morph_aug[cursor] = morph
                traj_aug[cursor] = traj
                orient_aug[cursor] = orient
                y_aug[cursor] = labels[cursor]
    else:
        cursor = 0
        for i, seq in enumerate(tqdm(raw_train, desc=f"Aug MorphTraj r{repeats}")):
            for _ in range(repeats):
                morph, traj, orient = sequence_factor_features(augment_fixed(seq), rectify_hands=rectify_hands, feature_version=feature_version)
                morph_aug[cursor] = morph
                traj_aug[cursor] = traj
                orient_aug[cursor] = orient
                y_aug[cursor] = y_train[i]
                cursor += 1
    del morph_aug, traj_aug, orient_aug
    np.save(paths["labels"], y_aug)
    return (
        np.load(paths["morph"], mmap_mode="r"),
        np.load(paths["traj"], mmap_mode="r"),
        np.load(paths["orient"], mmap_mode="r"),
        np.load(paths["labels"], mmap_mode="r"),
    )


class FactorDataset(Dataset):
    def __init__(self, morph, traj, orient, labels, morph_aug=None, traj_aug=None, orient_aug=None, labels_aug=None):
        self.morph = morph
        self.traj = traj
        self.orient = orient
        self.labels = labels
        self.morph_aug = morph_aug
        self.traj_aug = traj_aug
        self.orient_aug = orient_aug
        self.labels_aug = labels_aug
        self.base_len = len(labels)
        self.aug_len = 0 if labels_aug is None else len(labels_aug)

    def __len__(self):
        return self.base_len + self.aug_len

    def __getitem__(self, idx):
        if idx < self.base_len:
            return (
                torch.from_numpy(np.asarray(self.morph[idx], dtype=np.float32)),
                torch.from_numpy(np.asarray(self.traj[idx], dtype=np.float32)),
                torch.from_numpy(np.asarray(self.orient[idx], dtype=np.float32)),
                int(self.labels[idx]),
            )
        j = idx - self.base_len
        return (
            torch.from_numpy(np.asarray(self.morph_aug[j], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.traj_aug[j], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.orient_aug[j], dtype=np.float32)),
            int(self.labels_aug[j]),
        )


class FactorEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, heads: int = 4):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout))
        self.conv = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1, groups=1)
        self.norm = nn.LayerNorm(out_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=out_dim,
                nhead=heads,
                dim_feedforward=out_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )

    def forward(self, x):
        h = self.stem(x)
        y = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm(h + y)
        return self.encoder(h).mean(dim=1)

    def forward_tokens(self, x):
        h = self.stem(x)
        y = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm(h + y)
        return self.encoder(h)


class MorphTrajExpert(nn.Module):
    def __init__(
        self,
        morph_dim: int,
        traj_dim: int,
        orient_dim: int,
        num_classes: int,
        branch_dim: int,
        dropout: float,
        conditioning: str = "none",
    ):
        super().__init__()
        if conditioning not in {"none", "film", "morph_gate_residual", "morph_gate_scale", "traj_cross_attn_morph"}:
            raise ValueError(f"Unknown MorphTraj conditioning: {conditioning}")
        self.conditioning = conditioning
        self.morph = FactorEncoder(morph_dim, branch_dim, dropout)
        self.traj = FactorEncoder(traj_dim, branch_dim, dropout)
        self.orient = FactorEncoder(orient_dim, branch_dim, dropout)
        if conditioning == "film":
            self.traj_film = nn.Sequential(
                nn.LayerNorm(branch_dim),
                nn.Linear(branch_dim, branch_dim * 2),
            )
            self.orient_film = nn.Sequential(
                nn.LayerNorm(branch_dim),
                nn.Linear(branch_dim, branch_dim * 2),
            )
            # Start as identity so v2-A cannot destroy the v1 factor geometry at initialization.
            nn.init.zeros_(self.traj_film[-1].weight)
            nn.init.zeros_(self.traj_film[-1].bias)
            nn.init.zeros_(self.orient_film[-1].weight)
            nn.init.zeros_(self.orient_film[-1].bias)
        elif conditioning in {"morph_gate_residual", "morph_gate_scale"}:
            self.traj_gate = nn.Sequential(
                nn.LayerNorm(branch_dim),
                nn.Linear(branch_dim, branch_dim),
            )
            # Residual gating starts near identity; scale gating starts near 0.5*T.
            nn.init.zeros_(self.traj_gate[-1].weight)
            nn.init.zeros_(self.traj_gate[-1].bias)
        elif conditioning == "traj_cross_attn_morph":
            self.traj_morph_attn = nn.MultiheadAttention(branch_dim, num_heads=4, dropout=dropout, batch_first=True)
            self.traj_attn_norm = nn.LayerNorm(branch_dim)
            self.traj_attn_gate = nn.Parameter(torch.zeros(()))
        self.fuse = nn.Sequential(
            nn.LayerNorm(branch_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(branch_dim * 3, branch_dim * 3),
            nn.GELU(),
            nn.LayerNorm(branch_dim * 3),
        )
        self.morph_classifier = CosineClassifier(branch_dim, num_classes, scale=16.0)
        self.traj_classifier = CosineClassifier(branch_dim, num_classes, scale=16.0)
        self.orient_classifier = CosineClassifier(branch_dim, num_classes, scale=16.0)
        self.classifier = CosineClassifier(branch_dim * 3, num_classes, scale=16.0)

    def forward_features(self, morph, traj, orient):
        if self.conditioning == "traj_cross_attn_morph":
            mtok = self.morph.forward_tokens(morph)
            ttok = self.traj.forward_tokens(traj)
            otok = self.orient.forward_tokens(orient)
            ctx, _ = self.traj_morph_attn(ttok, mtok, mtok, need_weights=False)
            ttok = self.traj_attn_norm(ttok + self.traj_attn_gate * ctx)
            zm = mtok.mean(dim=1)
            zt = ttok.mean(dim=1)
            zo = otok.mean(dim=1)
            z = self.fuse(torch.cat([zm, zt, zo], dim=-1))
            return zm, zt, zo, z

        zm = self.morph(morph)
        zt = self.traj(traj)
        zo = self.orient(orient)
        if self.conditioning == "film":
            gamma_t, beta_t = self.traj_film(zm).chunk(2, dim=-1)
            gamma_o, beta_o = self.orient_film(zm).chunk(2, dim=-1)
            zt = (1.0 + gamma_t) * zt + beta_t
            zo = (1.0 + gamma_o) * zo + beta_o
        elif self.conditioning == "morph_gate_residual":
            gate_t = torch.sigmoid(self.traj_gate(zm))
            zt = zt + gate_t * zt
        elif self.conditioning == "morph_gate_scale":
            gate_t = torch.sigmoid(self.traj_gate(zm))
            zt = gate_t * zt
        z = self.fuse(torch.cat([zm, zt, zo], dim=-1))
        return zm, zt, zo, z

    def forward(self, morph, traj, orient):
        zm, zt, zo, z = self.forward_features(morph, traj, orient)
        return {
            "morph": self.morph_classifier(zm),
            "traj": self.traj_classifier(zt),
            "orient": self.orient_classifier(zo),
            "fused": self.classifier(z),
        }


def mixup_factors(morph, traj, orient, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(morph.size(0), device=morph.device)
    return (
        lam * morph + (1 - lam) * morph[idx],
        lam * traj + (1 - lam) * traj[idx],
        lam * orient + (1 - lam) * orient[idx],
        y,
        y[idx],
        float(lam),
    )


def train_epoch(model, loader, optimizer, criterion, device, mixup_alpha, amp, aux_weight):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    total = 0.0
    for morph, traj, orient, labels in tqdm(loader, desc="Train", leave=False):
        morph, traj, orient, labels = morph.to(device), traj.to(device), orient.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            if mixup_alpha > 0:
                morph, traj, orient, ya, yb, lam = mixup_factors(morph, traj, orient, labels, mixup_alpha)
                outs = model(morph, traj, orient)
                ce = lambda logits: lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            else:
                outs = model(morph, traj, orient)
                ce = lambda logits: criterion(logits, labels)
            loss = ce(outs["fused"]) + aux_weight * (ce(outs["morph"]) + ce(outs["traj"]) + ce(outs["orient"]))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
    return total / max(len(loader), 1)


def eval_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for morph, traj, orient, labels in loader:
            morph, traj, orient, labels = morph.to(device), traj.to(device), orient.to(device), labels.to(device)
            logits = model(morph, traj, orient)["fused"]
            total_loss += float(criterion(logits, labels).detach().cpu())
            correct += int((logits.argmax(1) == labels).sum().item())
            total += int(labels.numel())
    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def collect_logits(model, loader, device):
    model.eval()
    rows = {"fused": [], "morph": [], "traj": [], "orient": []}
    with torch.no_grad():
        for morph, traj, orient, _ in loader:
            outs = model(morph.to(device), traj.to(device), orient.to(device))
            for k in rows:
                rows[k].append(outs[k].detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in rows.items()}


def fusion_sweep(mt_val, mt_test, y_val, y_test, b6_val, b6_test, felf_val, felf_test, out_dir, prefix):
    rows = []
    weights = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    def compatible(val_logits, test_logits, name):
        if val_logits is None or test_logits is None:
            return False
        expected_val = mt_val["fused"].shape
        expected_test = mt_test["fused"].shape
        if val_logits.shape != expected_val or test_logits.shape != expected_test:
            print(
                f"Skipping incompatible {name} fusion logits: "
                f"val={val_logits.shape} test={test_logits.shape}; "
                f"expected val={expected_val} test={expected_test}."
            )
            return False
        return True

    b6_ok = compatible(b6_val, b6_test, "B6")
    felf_ok = compatible(felf_val, felf_test, "FELF")

    def add(name, val_logits, test_logits, weight, expert):
        vt1, vt5 = metrics_from_logits(val_logits, y_val)
        tt1, tt5 = metrics_from_logits(test_logits, y_test)
        rows.append({"run": name, "expert": expert, "weight": weight, "val_top1": vt1, "val_top5": vt5, "test_top1": tt1, "test_top5": tt5})

    for expert in ["fused", "morph", "traj", "orient"]:
        add(f"MorphTraj_{expert}", mt_val[expert], mt_test[expert], 1.0, expert)
        if b6_ok:
            for w in weights:
                add(f"B6_plus_{expert}_w{w}", b6_val + w * mt_val[expert], b6_test + w * mt_test[expert], w, expert)
        if felf_ok:
            for w in weights:
                add(f"FELF_plus_{expert}_w{w}", felf_val + w * mt_val[expert], felf_test + w * mt_test[expert], w, expert)
        if b6_ok and felf_ok:
            for w in weights:
                add(
                    f"B6_plus_FELF_plus_{expert}_w{w}",
                    b6_val + felf_val + w * mt_val[expert],
                    b6_test + felf_test + w * mt_test[expert],
                    w,
                    expert,
                )

    csv_path = out_dir / f"{prefix}_morphtraj_fusion_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-glosses", type=int, default=300)
    parser.add_argument("--action-source", choices=["json_first_n", "top_frequency", "train_dirs"], default="json_first_n")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--json-path", default=JSON_PATH)
    parser.add_argument("--cache-dir", default="rework_model/cache/wlasl300_morphtraj_firstn")
    parser.add_argument("--output-prefix", default="wlasl300_MorphTrajExpert_seed1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--swa-epochs", type=int, default=20)
    parser.add_argument("--aug-repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--branch-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--feature-version", choices=["v1", "v3"], default="v1")
    parser.add_argument(
        "--conditioning",
        choices=["none", "film", "morph_gate_residual", "morph_gate_scale", "traj_cross_attn_morph"],
        default="none",
    )
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--cache-workers", type=int, default=1)
    parser.add_argument("--no-rectify-hands", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--b6-diagnostic", default="diagnostic/wlasl300_seed1_B6.json")
    parser.add_argument("--felf-diagnostic", default="diagnostic/wlasl300_FELF_seed1_lrg10_rf075.json")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rectify = not args.no_rectify_hands
    cache_leaf = "morph_traj" if args.feature_version == "v1" else f"morph_traj_{args.feature_version}"
    cache_path = Path(args.cache_dir) / cache_leaf
    out_dir = cache_path / args.output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    splits, actions, _ = load_subset_raw_for_source(args.data_dir, args.json_path, args.num_glosses, args.action_source, labels_only=False)
    y_train = np.asarray(splits["train"]["labels"], dtype=np.int64)
    y_val = np.asarray(splits["val"]["labels"], dtype=np.int64)
    y_test = np.asarray(splits["test"]["labels"], dtype=np.int64)
    num_classes = len(actions)

    morph_train, traj_train, orient_train = encode_factors(splits["train"]["raw"], cache_path, "train", args.force_cache, rectify, args.feature_version, args.cache_workers)
    morph_val, traj_val, orient_val = encode_factors(splits["val"]["raw"], cache_path, "val", args.force_cache, rectify, args.feature_version, args.cache_workers)
    morph_test, traj_test, orient_test = encode_factors(splits["test"]["raw"], cache_path, "test", args.force_cache, rectify, args.feature_version, args.cache_workers)
    morph_aug, traj_aug, orient_aug, y_aug = build_aug_cache(splits["train"]["raw"], y_train, cache_path, args.aug_repeats, args.force_cache, rectify, args.feature_version, args.cache_workers)

    train_loader = DataLoader(
        FactorDataset(morph_train, traj_train, orient_train, y_train, morph_aug, traj_aug, orient_aug, y_aug),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(TensorDataset(torch.tensor(morph_val), torch.tensor(traj_val), torch.tensor(orient_val), torch.tensor(y_val)), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(torch.tensor(morph_test), torch.tensor(traj_test), torch.tensor(orient_test), torch.tensor(y_test)), batch_size=args.batch_size)

    model = MorphTrajExpert(
        morph_train.shape[-1],
        traj_train.shape[-1],
        orient_train.shape[-1],
        num_classes,
        args.branch_dim,
        args.dropout,
        conditioning=args.conditioning,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"Startup: MorphTrajExpert num_glosses={args.num_glosses} action_source={args.action_source} "
        f"feature_version={args.feature_version} cache_workers={args.cache_workers} dims={morph_train.shape[-1]}/{traj_train.shape[-1]}/{orient_train.shape[-1]} branch={args.branch_dim} "
        f"dropout={args.dropout} conditioning={args.conditioning} aux={args.aux_weight} mixup={args.mixup_alpha} params={params}"
    )

    criterion = ArcFaceLoss(scale=16.0, margin=0.2, cw=get_class_weights(y_train, num_classes).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    pre_path = Path("rework_model") / f"{args.output_prefix}_pre_swa.pth"
    swa_path = Path("rework_model") / f"{args.output_prefix}_swa.pth"
    best_path = Path("rework_model") / f"{args.output_prefix}_best_by_val.pth"

    best_val, patience, history = -1.0, 0, []
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.mixup_alpha, args.amp, args.aux_weight)
        val_loss, val_acc = eval_loss_acc(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row))
        if val_acc > best_val:
            best_val, patience = val_acc, 0
            torch.save(model.state_dict(), pre_path)
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= 12:
                print(f"Early stop epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(pre_path, map_location=device, weights_only=True))
    pre_val_logits = collect_logits(model, val_loader, device)
    pre_test_logits = collect_logits(model, test_loader, device)
    pre_metrics = {
        "val_top1": metrics_from_logits(pre_val_logits["fused"], y_val)[0],
        "test_top1": metrics_from_logits(pre_test_logits["fused"], y_test)[0],
        "test_top5": metrics_from_logits(pre_test_logits["fused"], y_test)[1],
    }

    selected, selected_val_logits, selected_test_logits = "pre_swa", pre_val_logits, pre_test_logits
    swa_metrics = None
    if args.swa_epochs > 0:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.5)
        for ep in range(args.swa_epochs):
            print(f"  SWA ep {ep + 1}/{args.swa_epochs}")
            train_epoch(model, train_loader, optimizer, criterion, device, args.mixup_alpha, args.amp, args.aux_weight)
            swa_model.update_parameters(model)
            swa_scheduler.step()
        torch.save(swa_model.module.state_dict(), swa_path)
        swa_val_logits = collect_logits(swa_model.module.to(device), val_loader, device)
        swa_test_logits = collect_logits(swa_model.module.to(device), test_loader, device)
        swa_metrics = {
            "val_top1": metrics_from_logits(swa_val_logits["fused"], y_val)[0],
            "test_top1": metrics_from_logits(swa_test_logits["fused"], y_test)[0],
            "test_top5": metrics_from_logits(swa_test_logits["fused"], y_test)[1],
        }
        if swa_metrics["val_top1"] >= pre_metrics["val_top1"]:
            selected, selected_val_logits, selected_test_logits = "swa", swa_val_logits, swa_test_logits
            torch.save(swa_model.module.state_dict(), best_path)

    for k, arr in selected_val_logits.items():
        np.save(out_dir / f"val_logits_{k}.npy", arr)
    for k, arr in selected_test_logits.items():
        np.save(out_dir / f"test_logits_{k}.npy", arr)
    np.save(out_dir / "val_labels.npy", y_val)
    np.save(out_dir / "test_labels.npy", y_test)
    (out_dir / "class_names.json").write_text(json.dumps(actions, indent=2), encoding="utf-8")

    b6_val = load_diag_logits(args.b6_diagnostic, "val")
    b6_test = load_diag_logits(args.b6_diagnostic, "test")
    felf_val = load_diag_logits(args.felf_diagnostic, "val")
    felf_test = load_diag_logits(args.felf_diagnostic, "test")
    rows = fusion_sweep(selected_val_logits, selected_test_logits, y_val, y_test, b6_val, b6_test, felf_val, felf_test, out_dir, args.output_prefix)
    best_fusion = max(rows, key=lambda r: (r["test_top1"], r["test_top5"]))
    selected_metrics = {
        "val_top1": metrics_from_logits(selected_val_logits["fused"], y_val)[0],
        "test_top1": metrics_from_logits(selected_test_logits["fused"], y_test)[0],
        "test_top5": metrics_from_logits(selected_test_logits["fused"], y_test)[1],
    }

    diag = {
        "experiment": "MorphTrajExpert_factorized_coordinate_expert",
        "num_glosses": args.num_glosses,
        "action_source": args.action_source,
        "seed": args.seed,
        "feature_type": "palm_morphology_body_trajectory_orientation",
        "feature_version": args.feature_version,
        "conditioning": args.conditioning,
        "feature_dims": {"morphology": int(morph_train.shape[-1]), "trajectory": int(traj_train.shape[-1]), "orientation": int(orient_train.shape[-1])},
        "branch_dim": args.branch_dim,
        "dropout": args.dropout,
        "mixup_alpha": args.mixup_alpha,
        "aux_weight": args.aux_weight,
        "aug_repeats": args.aug_repeats,
        "cache_workers": args.cache_workers,
        "params": params,
        "samples": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
        "pre_swa_metrics": pre_metrics,
        "swa_metrics": swa_metrics,
        "selected_by_val": selected,
        "selected_by_val_metrics": selected_metrics,
        "best_fusion_observed": best_fusion,
        "checkpoints": {"pre_swa": str(pre_path), "swa": str(swa_path), "best_by_val": str(best_path)},
        "logits": {
            "val_logits": str(out_dir / "val_logits_fused.npy"),
            "test_logits": str(out_dir / "test_logits_fused.npy"),
            "val_labels": str(out_dir / "val_labels.npy"),
            "test_labels": str(out_dir / "test_labels.npy"),
        },
        "factor_logits": {
            k: {"val": str(out_dir / f"val_logits_{k}.npy"), "test": str(out_dir / f"test_logits_{k}.npy")}
            for k in ["fused", "morph", "traj", "orient"]
        },
        "fusion_sweep_csv": str(out_dir / f"{args.output_prefix}_morphtraj_fusion_sweep.csv"),
        "history": history,
    }
    Path("diagnostic").mkdir(exist_ok=True)
    diag_path = Path("diagnostic") / f"{args.output_prefix}.json"
    diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print("MorphTrajExpert")
    print(f"  Selected by val: {selected}")
    print(f"  Pre-SWA: Val {pre_metrics['val_top1']:.2f}% Test top1 {pre_metrics['test_top1']:.2f}% Test top5 {pre_metrics['test_top5']:.2f}%")
    if swa_metrics:
        print(f"  SWA:     Val {swa_metrics['val_top1']:.2f}% Test top1 {swa_metrics['test_top1']:.2f}% Test top5 {swa_metrics['test_top5']:.2f}%")
    print(f"  Selected: Test top1 {selected_metrics['test_top1']:.2f}% Test top5 {selected_metrics['test_top5']:.2f}%")
    print(f"  Best fusion observed: {best_fusion['run']} {best_fusion['test_top1']:.2f}%/{best_fusion['test_top5']:.2f}%")
    print(f"  Params: {params:,}")
    print(f"  Result JSON: {diag_path}")


if __name__ == "__main__":
    main()
