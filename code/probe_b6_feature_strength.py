"""Fast feature-strength probes for cached old-frame LocalGlobal features.

This script intentionally avoids the Transformer. It trains simple classifiers
on frozen cached features to estimate how much of B6 comes from handcrafted
feature separability versus temporal modeling.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from wlasl_train_local_global_arcface_subset import DATA_DIR, JSON_PATH
from wlasl_train_old_localglobal_sweep_subset import load_subset_raw_for_source


SEQUENCE_LENGTH = 40


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_cache_root(cache_dir: Path, feature_mode: str = "old") -> Path:
    for candidate in [cache_dir / "frame_old", cache_dir / feature_mode, cache_dir]:
        if (candidate / "train_old_left.npy").exists() or (candidate / "train_left.npy").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate cached features under {cache_dir}")


def load_array(root: Path, split: str, stream: str):
    names = [f"{split}_old_{stream}.npy", f"{split}_{stream}.npy"]
    for name in names:
        path = root / name
        if path.exists():
            return np.load(path, mmap_mode="r")
    raise FileNotFoundError(f"Missing {split}/{stream} in {root}")


def load_labels(root: Path, split: str):
    names = [f"{split}_old_labels.npy", f"{split}_labels.npy", f"{split}_y.npy"]
    for name in names:
        path = root / name
        if path.exists():
            return np.load(path).astype(np.int64)
    # Existing B6 caches sometimes store labels only in run subdirs. Prefer
    # deterministic lookup by split from any child run cache.
    for child in sorted(root.iterdir()):
        if child.is_dir():
            path = child / f"{split}_labels.npy"
            if path.exists():
                return np.load(path).astype(np.int64)
    raise FileNotFoundError(f"Missing labels for {split} in {root}")


def load_split(root: Path, split: str, labels: np.ndarray | None = None):
    left = load_array(root, split, "left")
    right = load_array(root, split, "right")
    global_features = load_array(root, split, "global")
    if labels is None:
        labels = load_labels(root, split)
    return left, right, global_features, labels


def concat_streams(left, right, global_features, streams: str):
    parts = []
    if "left" in streams:
        parts.append(left)
    if "right" in streams:
        parts.append(right)
    if "global" in streams:
        parts.append(global_features)
    if not parts:
        raise ValueError(f"No streams selected by {streams}")
    return np.concatenate(parts, axis=-1)


def pool_features(left, right, global_features, probe: str, streams: str):
    x = concat_streams(left, right, global_features, streams)
    t = x.shape[1]
    center = t // 2
    mid0, mid1 = t // 3, (2 * t) // 3
    if probe == "center":
        return np.asarray(x[:, center], dtype=np.float32)
    if probe == "mean":
        return np.asarray(x.mean(axis=1), dtype=np.float32)
    if probe == "std":
        return np.asarray(x.std(axis=1), dtype=np.float32)
    if probe == "mean_std":
        return np.asarray(np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=-1), dtype=np.float32)
    if probe == "middle_mean":
        return np.asarray(x[:, mid0:mid1].mean(axis=1), dtype=np.float32)
    if probe == "delta_mean":
        v = np.zeros_like(x)
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        return np.asarray(v.mean(axis=1), dtype=np.float32)
    if probe == "delta_stats":
        v = np.zeros_like(x)
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        a = np.zeros_like(x)
        a[:, 1:] = v[:, 1:] - v[:, :-1]
        return np.asarray(
            np.concatenate([v.mean(axis=1), v.std(axis=1), np.abs(v).max(axis=1), a.mean(axis=1), a.std(axis=1)], axis=-1),
            dtype=np.float32,
        )
    if probe == "temporal_stats":
        v = np.zeros_like(x)
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        middle = x[:, mid0:mid1].mean(axis=1)
        return np.asarray(
            np.concatenate([x.mean(axis=1), x.std(axis=1), x[:, 0], middle, x[:, -1], x[:, -1] - x[:, 0], v.mean(axis=1)], axis=-1),
            dtype=np.float32,
        )
    raise ValueError(f"Unknown probe: {probe}")


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class TinyMLPProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CosineProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, input_dim))
        self.scale = scale
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return self.scale * (x @ w.t())


def standardize(train_x, val_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((val_x - mean) / std).astype(np.float32),
        ((test_x - mean) / std).astype(np.float32),
    )


def topk(logits: torch.Tensor, labels: torch.Tensor):
    pred = logits.topk(min(5, logits.shape[1]), dim=1).indices
    top1 = (pred[:, 0] == labels).float().mean().item() * 100.0
    top5 = (pred == labels[:, None]).any(dim=1).float().mean().item() * 100.0
    return top1, top5


def evaluate(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits_all.append(model(x).float().cpu())
            labels_all.append(y)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    return topk(logits, labels)


def train_probe(train_x, train_y, val_x, val_y, test_x, test_y, model_type: str, args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = int(max(train_y.max(), val_y.max(), test_y.max()) + 1)
    if args.standardize:
        train_x, val_x, test_x = standardize(train_x, val_x, test_x)
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.int64)))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y.astype(np.int64)))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y.astype(np.int64)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    if model_type == "linear":
        model = LinearProbe(train_x.shape[1], num_classes)
    elif model_type == "cosine":
        model = CosineProbe(train_x.shape[1], num_classes, scale=args.cosine_scale)
    elif model_type == "mlp":
        model = TinyMLPProbe(train_x.shape[1], num_classes, hidden_dim=args.hidden_dim, dropout=args.dropout)
    else:
        raise ValueError(model_type)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_state = -1.0, None
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y, label_smoothing=args.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_top1, val_top5 = evaluate(model, val_loader, device)
        if val_top1 > best_val:
            best_val = val_top1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_top1, val_top5 = evaluate(model, val_loader, device)
    test_top1, test_top5 = evaluate(model, test_loader, device)
    return {
        "val_top1": val_top1,
        "val_top5": val_top5,
        "test_top1": test_top1,
        "test_top5": test_top5,
        "epochs_ran": epoch + 1,
        "input_dim": int(train_x.shape[1]),
        "params": int(sum(p.numel() for p in model.parameters())),
    }


def default_specs():
    return [
        ("center_left_linear", "center", "left", "linear", "Static left-hand center-frame strength"),
        ("center_right_linear", "center", "right", "linear", "Static right-hand/dominant-hand center-frame strength"),
        ("center_global_linear", "center", "global", "linear", "Static body/locus center-frame strength"),
        ("center_lrg_linear", "center", "left,right,global", "linear", "Pure static full-stream separability"),
        ("mean_lrg_linear", "mean", "left,right,global", "linear", "Order-free feature strength"),
        ("mean_lrg_cosine", "mean", "left,right,global", "cosine", "Order-free cosine classifier strength"),
        ("middle_lrg_linear", "middle_mean", "left,right,global", "linear", "Apex/middle-third strength"),
        ("delta_lrg_linear", "delta_mean", "left,right,global", "linear", "Mean explicit delta/motion strength"),
        ("delta_stats_lrg_linear", "delta_stats", "left,right,global", "linear", "Velocity/acceleration stat strength"),
        ("temporal_stats_lrg_linear", "temporal_stats", "left,right,global", "linear", "Order-light temporal summary strength"),
        ("temporal_stats_lrg_mlp", "temporal_stats", "left,right,global", "mlp", "Tiny MLP without temporal encoder"),
    ]


def infer_num_glosses(label: str, cache_dir: Path) -> int:
    text = f"{label} {cache_dir}".lower()
    if "wlasl100" in text:
        return 100
    if "wlasl300" in text:
        return 300
    raise ValueError(f"Cannot infer num_glosses from label/cache: {label} {cache_dir}")


def load_firstn_labels(label: str, cache_dir: Path, args):
    num_glosses = infer_num_glosses(label, cache_dir)
    splits, actions, _ = load_subset_raw_for_source(
        args.data_dir,
        args.json_path,
        num_glosses,
        "json_first_n",
        0,
        labels_only=True,
    )
    return {
        "train": np.asarray(splits["train"]["labels"], dtype=np.int64),
        "val": np.asarray(splits["val"]["labels"], dtype=np.int64),
        "test": np.asarray(splits["test"]["labels"], dtype=np.int64),
    }, actions


def load_firstn_raw(label: str, cache_dir: Path, args):
    num_glosses = infer_num_glosses(label, cache_dir)
    splits, actions, _ = load_subset_raw_for_source(
        args.data_dir,
        args.json_path,
        num_glosses,
        "json_first_n",
        0,
        labels_only=False,
    )
    return splits, actions


def palm_normalized_hand(hand: np.ndarray) -> np.ndarray:
    hand = np.asarray(hand, dtype=np.float32).reshape(21, 3)
    valid = float(np.isfinite(hand).all() and np.any(np.abs(hand) > 1e-8))
    if valid <= 0:
        return np.zeros(64, dtype=np.float32)
    palm_ids = [0, 5, 9, 13, 17]
    palm_center = hand[palm_ids].mean(axis=0)
    scale = float(np.linalg.norm(hand[9] - hand[0]))
    if scale <= 1e-6 or not np.isfinite(scale):
        scale = float(np.mean([np.linalg.norm(hand[j] - hand[i]) for i, j in [(0, 5), (0, 9), (0, 13), (0, 17)]]))
    if scale <= 1e-6 or not np.isfinite(scale):
        return np.zeros(64, dtype=np.float32)
    coords = ((hand - palm_center[None, :]) / scale).reshape(-1)
    return np.concatenate([coords, np.array([valid], dtype=np.float32)]).astype(np.float32)


def palmnorm_sequence(raw_seq: np.ndarray) -> np.ndarray:
    raw_seq = np.asarray(raw_seq, dtype=np.float32)
    frames = []
    for frame in raw_seq:
        left = frame[132:195].reshape(21, 3)
        right = frame[195:258].reshape(21, 3)
        frames.append(np.concatenate([palm_normalized_hand(left), palm_normalized_hand(right)], axis=0))
    return np.asarray(frames, dtype=np.float32)


def pose_hand_locus_frame(frame: np.ndarray) -> np.ndarray:
    pose = np.asarray(frame[:132], dtype=np.float32).reshape(33, 4)[:, :3]
    left = np.asarray(frame[132:195], dtype=np.float32).reshape(21, 3)
    right = np.asarray(frame[195:258], dtype=np.float32).reshape(21, 3)
    nose = pose[0]
    shoulder_center = (pose[11] + pose[12]) * 0.5
    shoulder_scale = float(np.linalg.norm(pose[11] - pose[12]))
    if shoulder_scale <= 1e-6 or not np.isfinite(shoulder_scale):
        shoulder_scale = 1.0

    def hand_center(hand):
        valid = float(np.isfinite(hand).all() and np.any(np.abs(hand) > 1e-8))
        if valid <= 0:
            return np.zeros(3, dtype=np.float32), valid
        return hand[[0, 5, 9, 13, 17]].mean(axis=0).astype(np.float32), valid

    lc, lv = hand_center(left)
    rc, rv = hand_center(right)
    lw = left[0].astype(np.float32) if lv else np.zeros(3, dtype=np.float32)
    rw = right[0].astype(np.float32) if rv else np.zeros(3, dtype=np.float32)
    vals = [
        (lw - nose) / shoulder_scale,
        (rw - nose) / shoulder_scale,
        (lc - nose) / shoulder_scale,
        (rc - nose) / shoulder_scale,
        (lw - shoulder_center) / shoulder_scale,
        (rw - shoulder_center) / shoulder_scale,
        (rw - lw) / shoulder_scale,
        np.array(
            [
                np.linalg.norm(rw - lw) / shoulder_scale,
                np.linalg.norm(lw - nose) / shoulder_scale,
                np.linalg.norm(rw - nose) / shoulder_scale,
                lv,
                rv,
            ],
            dtype=np.float32,
        ),
    ]
    return np.concatenate(vals).astype(np.float32)


def locus_sequence(raw_seq: np.ndarray) -> np.ndarray:
    return np.asarray([pose_hand_locus_frame(frame) for frame in np.asarray(raw_seq, dtype=np.float32)], dtype=np.float32)


def build_raw_probe_arrays(split, kind: str, cache_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        xs = np.load(cache_path, mmap_mode="r")
    else:
        fn = palmnorm_sequence if kind == "palmnorm" else locus_sequence
        xs = np.asarray([fn(raw) for raw in tqdm(split["raw"], desc=f"{kind} extract", leave=False)], dtype=np.float32)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, xs)
    ys = np.asarray(split["labels"], dtype=np.int64)
    return xs, ys


def load_or_build_raw_probe_sequences(label: str, cache_dir: Path, kind: str, args):
    """Load cached raw-derived probe tensors without touching raw videos when possible."""
    raw_cache_dir = Path(args.raw_probe_cache_dir) / label
    paths = {
        split: raw_cache_dir / f"{split}_{kind}.npy"
        for split in ("train", "val", "test")
    }
    split_labels, actions = load_firstn_labels(label, cache_dir, args)
    if all(path.exists() for path in paths.values()):
        return (
            np.load(paths["train"], mmap_mode="r"),
            split_labels["train"],
            np.load(paths["val"], mmap_mode="r"),
            split_labels["val"],
            np.load(paths["test"], mmap_mode="r"),
            split_labels["test"],
            actions,
        )

    splits, actions = load_firstn_raw(label, cache_dir, args)
    train_x_seq, train_y = build_raw_probe_arrays(splits["train"], kind, paths["train"])
    val_x_seq, val_y = build_raw_probe_arrays(splits["val"], kind, paths["val"])
    test_x_seq, test_y = build_raw_probe_arrays(splits["test"], kind, paths["test"])
    return train_x_seq, train_y, val_x_seq, val_y, test_x_seq, test_y, actions


def palm_pool_features(x: np.ndarray, probe: str, streams: str):
    if streams == "left":
        x = x[:, :, :64]
    elif streams == "right":
        x = x[:, :, 64:]
    elif streams == "both":
        pass
    else:
        raise ValueError(f"Unknown PalmNorm streams={streams}")
    t = x.shape[1]
    center = t // 2
    mid0, mid1 = t // 3, (2 * t) // 3
    if probe == "center":
        return np.asarray(x[:, center], dtype=np.float32)
    if probe == "mean":
        return np.asarray(x.mean(axis=1), dtype=np.float32)
    if probe == "middle_mean":
        return np.asarray(x[:, mid0:mid1].mean(axis=1), dtype=np.float32)
    if probe == "mean_delta_stats":
        v = np.zeros_like(x)
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        return np.asarray(np.concatenate([x.mean(axis=1), x.std(axis=1), v.mean(axis=1), v.std(axis=1)], axis=-1), dtype=np.float32)
    raise ValueError(f"Unknown PalmNorm probe={probe}")


def palmnorm_specs():
    return [
        ("palmnorm_center_left_linear", "center", "left", "linear", "Palm-normalized left-hand static shape"),
        ("palmnorm_center_right_linear", "center", "right", "linear", "Palm-normalized right-hand static shape"),
        ("palmnorm_center_both_linear", "center", "both", "linear", "Palm-normalized both-hand static shape"),
        ("palmnorm_mean_both_linear", "mean", "both", "linear", "Order-free palm-normalized handshape strength"),
        ("palmnorm_middle_both_linear", "middle_mean", "both", "linear", "Middle/apex palm-normalized handshape strength"),
        ("palmnorm_mean_delta_both_linear", "mean_delta_stats", "both", "linear", "Palm-normalized shape plus delta stats"),
        ("palmnorm_mean_both_mlp", "mean", "both", "mlp", "Tiny MLP on order-free palm-normalized handshape"),
    ]


def locus_specs():
    return [
        ("locus_center_linear", "center", "all", "linear", "Center-frame hand/body locus strength"),
        ("locus_mean_linear", "mean", "all", "linear", "Order-free hand/body locus strength"),
        ("locus_middle_linear", "middle_mean", "all", "linear", "Middle/apex hand/body locus strength"),
        ("locus_delta_stats_linear", "mean_delta_stats", "all", "linear", "Hand/body locus trajectory stat strength"),
        ("locus_mean_mlp", "mean", "all", "mlp", "Tiny MLP on order-free hand/body locus"),
    ]


def sequence_pool_features(x: np.ndarray, probe: str):
    t = x.shape[1]
    center = t // 2
    mid0, mid1 = t // 3, (2 * t) // 3
    if probe == "center":
        return np.asarray(x[:, center], dtype=np.float32)
    if probe == "mean":
        return np.asarray(x.mean(axis=1), dtype=np.float32)
    if probe == "middle_mean":
        return np.asarray(x[:, mid0:mid1].mean(axis=1), dtype=np.float32)
    if probe == "mean_delta_stats":
        v = np.zeros_like(x)
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        return np.asarray(np.concatenate([x.mean(axis=1), x.std(axis=1), v.mean(axis=1), v.std(axis=1)], axis=-1), dtype=np.float32)
    raise ValueError(f"Unknown sequence probe={probe}")


def run_palmnorm_dataset(label: str, cache_dir: Path, args):
    train_x_seq, train_y, val_x_seq, val_y, test_x_seq, test_y, actions = load_or_build_raw_probe_sequences(
        label, cache_dir, "palmnorm", args
    )
    rows = []
    for run_id, probe, streams, model_type, interpretation in tqdm(palmnorm_specs(), desc=f"PalmNorm {label}"):
        train_x = palm_pool_features(train_x_seq, probe, streams)
        val_x = palm_pool_features(val_x_seq, probe, streams)
        test_x = palm_pool_features(test_x_seq, probe, streams)
        result = train_probe(train_x, train_y, val_x, val_y, test_x, test_y, model_type, args)
        rows.append(
            {
                "dataset": label,
                "cache_root": str(find_cache_root(cache_dir)),
                "class_protocol": "json_first_n",
                "num_classes": int(len(actions)),
                "run_id": run_id,
                "probe": f"palmnorm_{probe}",
                "streams": streams,
                "model_type": model_type,
                **result,
                "interpretation": interpretation,
            }
        )
    return rows


def run_locus_dataset(label: str, cache_dir: Path, args):
    train_x_seq, train_y, val_x_seq, val_y, test_x_seq, test_y, actions = load_or_build_raw_probe_sequences(
        label, cache_dir, "locus", args
    )
    rows = []
    for run_id, probe, streams, model_type, interpretation in tqdm(locus_specs(), desc=f"Locus {label}"):
        train_x = sequence_pool_features(train_x_seq, probe)
        val_x = sequence_pool_features(val_x_seq, probe)
        test_x = sequence_pool_features(test_x_seq, probe)
        result = train_probe(train_x, train_y, val_x, val_y, test_x, test_y, model_type, args)
        rows.append(
            {
                "dataset": label,
                "cache_root": str(find_cache_root(cache_dir)),
                "class_protocol": "json_first_n",
                "num_classes": int(len(actions)),
                "run_id": run_id,
                "probe": f"locus_{probe}",
                "streams": streams,
                "model_type": model_type,
                **result,
                "interpretation": interpretation,
            }
        )
    return rows


def run_dataset(label: str, cache_dir: Path, args):
    root = find_cache_root(cache_dir)
    split_labels, actions = load_firstn_labels(label, cache_dir, args)
    tr = load_split(root, "train", split_labels["train"])
    val = load_split(root, "val", split_labels["val"])
    test = load_split(root, "test", split_labels["test"])
    for split_name, split in [("train", tr), ("val", val), ("test", test)]:
        if len(split[3]) != split[0].shape[0]:
            raise ValueError(
                f"{label} {split_name} label/cache length mismatch: labels={len(split[3])}, features={split[0].shape[0]}"
            )
    rows = []
    for run_id, probe, streams, model_type, interpretation in tqdm(default_specs(), desc=f"Probes {label}"):
        train_x = pool_features(tr[0], tr[1], tr[2], probe, streams)
        val_x = pool_features(val[0], val[1], val[2], probe, streams)
        test_x = pool_features(test[0], test[1], test[2], probe, streams)
        result = train_probe(train_x, tr[3], val_x, val[3], test_x, test[3], model_type, args)
        rows.append(
            {
                "dataset": label,
                "cache_root": str(root),
                "class_protocol": "json_first_n",
                "num_classes": int(len(actions)),
                "run_id": run_id,
                "probe": probe,
                "streams": streams,
                "model_type": model_type,
                **result,
                "interpretation": interpretation,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Simple frozen-feature probes for B6 old-frame features.")
    parser.add_argument("--wlasl300-cache", default="rework_model/cache/wlasl300_old_B6")
    parser.add_argument("--wlasl100-cache", default="rework_model/cache/wlasl100_old_B6")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--json-path", default=JSON_PATH)
    parser.add_argument("--output-csv", default="diagnostic/b6_feature_probe_firstn.csv")
    parser.add_argument("--output-json", default="diagnostic/b6_feature_probe_firstn.json")
    parser.add_argument("--raw-probe-cache-dir", default="rework_model/cache/raw_probe_features")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--cosine-scale", type=float, default=16.0)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-suite", choices=["cached", "palmnorm", "locus", "raw", "all"], default="cached")
    args = parser.parse_args()
    set_seed(args.seed)
    rows = []
    if args.probe_suite in {"cached", "all"}:
        rows.extend(run_dataset("wlasl300_firstn", Path(args.wlasl300_cache), args))
        rows.extend(run_dataset("wlasl100_firstn", Path(args.wlasl100_cache), args))
    if args.probe_suite in {"palmnorm", "raw", "all"}:
        rows.extend(run_palmnorm_dataset("wlasl300_firstn", Path(args.wlasl300_cache), args))
        rows.extend(run_palmnorm_dataset("wlasl100_firstn", Path(args.wlasl100_cache), args))
    if args.probe_suite in {"locus", "raw", "all"}:
        rows.extend(run_locus_dataset("wlasl300_firstn", Path(args.wlasl300_cache), args))
        rows.extend(run_locus_dataset("wlasl100_firstn", Path(args.wlasl100_cache), args))
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json = Path(args.output_json)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved {out_csv}")
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
