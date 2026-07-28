"""Train the local-global part-aware ArcFace model on WLASL-N subsets.

This script keeps the WLASL-100 champion script untouched and makes the
same architecture usable for WLASL-300, WLASL-1000, and WLASL-2000.
It loads the first N glosses from WLASL_v0.3.json, uses cached extracted
457-dim frame features, writes isolated caches/checkpoints per subset,
and reports validation-selected pre-SWA/SWA test metrics.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

sys.path.insert(0, ".")

from wlasl_geometry_utils import (
    extract_exp13_full_sequence_features,
    extract_part_aware_features,
    extract_part_aware_sequence_features,
)
from wlasl_train_local_global_arcface import (  # noqa: E402
    ArcFaceLoss,
    LocalGlobalArcFace,
    mixup_three,
    topk_metrics,
)
from wlasl_train_streams_arcface import augment_fixed, get_class_weights  # noqa: E402


JSON_PATH = "WLASL_Full/WLASL_v0.3.json"
DATA_DIR = "WLASL2000_Data"
SEQUENCE_LENGTH = 40
RECTIFY_ALPHA = float(os.environ.get("WLASL_RECTIFY_ALPHA", "0.4"))
SEED = 42


class CachedAugmentedPartDataset(Dataset):
    def __init__(self, left_base, right_base, global_base, base_y, left_aug, right_aug, global_aug, aug_y):
        self.left_base = left_base
        self.right_base = right_base
        self.global_base = global_base
        self.base_y = base_y
        self.left_aug = left_aug
        self.right_aug = right_aug
        self.global_aug = global_aug
        self.aug_y = aug_y
        self.base_len = len(base_y)
        self.aug_len = 0 if left_aug is None else len(aug_y)

    def __len__(self):
        return self.base_len + self.aug_len

    def __getitem__(self, idx):
        if idx < self.base_len:
            return (
                torch.from_numpy(np.array(self.left_base[idx], dtype=np.float32, copy=True)),
                torch.from_numpy(np.array(self.right_base[idx], dtype=np.float32, copy=True)),
                torch.from_numpy(np.array(self.global_base[idx], dtype=np.float32, copy=True)),
                int(self.base_y[idx]),
            )
        j = idx - self.base_len
        return (
            torch.from_numpy(np.array(self.left_aug[j], dtype=np.float32)),
            torch.from_numpy(np.array(self.right_aug[j], dtype=np.float32)),
            torch.from_numpy(np.array(self.global_aug[j], dtype=np.float32)),
            int(self.aug_y[j]),
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_actions(json_path, num_glosses):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(entry["gloss"] for entry in data[:num_glosses])


def load_subset_raw(data_dir, json_path, num_glosses, limit_samples=0, labels_only=False):
    actions = read_actions(json_path, num_glosses)
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
                    
                if not labels_only:
                    splits[split]["raw"].append(np.array(frames_raw, dtype=np.float32))
                else:
                    splits[split]["raw"].append(None)
                    
                splits[split]["labels"].append(label_map[gloss])
                splits[split]["ids"].append(f"{split}:{gloss}:{vid_dir.name}")
                if limit_samples and len(splits[split]["labels"]) >= limit_samples:
                    break
            if limit_samples and len(splits[split]["labels"]) >= limit_samples:
                break

    for split in ["train", "val", "test"]:
        print(f"{split}: {len(splits[split]['labels'])}")
    if not splits["train"]["labels"]:
        raise RuntimeError(f"No train samples found in {data_dir} for WLASL-{num_glosses}. Run extraction first.")
    if not splits["val"]["labels"]:
        raise RuntimeError(f"No val samples found in {data_dir} for WLASL-{num_glosses}. Run extraction first.")
    if not splits["test"]["labels"]:
        raise RuntimeError(f"No test samples found in {data_dir} for WLASL-{num_glosses}. Run extraction first.")
    return splits, actions, label_map


def extract_seqplus_features(seq, rectify_alpha):
    left_seq, right_seq, global_seq = extract_part_aware_sequence_features(seq, rectify_alpha)
    left13, right13, global13, orientation13 = extract_exp13_full_sequence_features(seq, rectify_alpha)

    # Exp13 local features start with normalized hand-shape coordinates. The
    # sequence extractor already keeps the stronger morphology, so only append
    # Exp13's dynamic/local residual tail.
    exp13_shape_dims = 60
    left13_dyn = left13[:, exp13_shape_dims:]
    right13_dyn = right13[:, exp13_shape_dims:]

    left_out = np.concatenate([left_seq, left13_dyn], axis=-1).astype(np.float32)
    right_out = np.concatenate([right_seq, right13_dyn], axis=-1).astype(np.float32)
    global_out = np.concatenate([global_seq, global13, orientation13], axis=-1).astype(np.float32)
    return left_out, right_out, global_out


def extract_feature_triplet(seq, feature_mode, rectify_alpha):
    seq = np.asarray(seq, dtype=np.float32)
    if feature_mode == "frame":
        left_frames, right_frames, global_frames = [], [], []
        for t in range(SEQUENCE_LENGTH):
            left, right, global_features = extract_part_aware_features(seq[t], rectify_alpha)
            left_frames.append(left)
            right_frames.append(right)
            global_frames.append(global_features)
        return (
            np.asarray(left_frames, dtype=np.float32),
            np.asarray(right_frames, dtype=np.float32),
            np.asarray(global_frames, dtype=np.float32),
        )
    if feature_mode == "seq":
        return extract_part_aware_sequence_features(seq, rectify_alpha)
    if feature_mode == "exp13":
        left, right, global_features, orientation = extract_exp13_full_sequence_features(seq, rectify_alpha)
        return left, right, np.concatenate([global_features, orientation], axis=-1).astype(np.float32)
    if feature_mode == "seqplus":
        return extract_seqplus_features(seq, rectify_alpha)
    raise ValueError(f"Unknown feature_mode={feature_mode}")


def feature_cache_dir(cache_dir, feature_mode):
    return Path(cache_dir) / feature_mode


def encode_parts(raw_list, split_name, cache_dir, force=False, feature_mode="frame"):
    cache_dir = Path(cache_dir)
    mode_dir = feature_cache_dir(cache_dir, feature_mode)
    left_path = mode_dir / f"{split_name}_left.npy"
    right_path = mode_dir / f"{split_name}_right.npy"
    global_path = mode_dir / f"{split_name}_global.npy"

    if not force and all(p.exists() for p in [left_path, right_path, global_path]):
        print(f"Using cached {split_name} part-aware features.")
        return (
            np.load(left_path, mmap_mode="r"),
            np.load(right_path, mmap_mode="r"),
            np.load(global_path, mmap_mode="r"),
        )

    print(f"Encoding {split_name} part-aware features mode={feature_mode}...")
    left_all, right_all, global_all = [], [], []
    total = len(raw_list)
    for i, seq in enumerate(raw_list):
        left_seq, right_seq, global_seq = extract_feature_triplet(seq, feature_mode, RECTIFY_ALPHA)
        left_all.append(np.array(left_seq, dtype=np.float32))
        right_all.append(np.array(right_seq, dtype=np.float32))
        global_all.append(np.array(global_seq, dtype=np.float32))
        if (i + 1) % 250 == 0 or (i + 1) == total:
            print(f"  encoded {split_name}: {i + 1}/{total}")

    left_arr = np.array(left_all, dtype=np.float32)
    right_arr = np.array(right_all, dtype=np.float32)
    global_arr = np.array(global_all, dtype=np.float32)
    mode_dir.mkdir(parents=True, exist_ok=True)
    np.save(left_path, left_arr)
    np.save(right_path, right_arr)
    np.save(global_path, global_arr)
    return (
        np.load(left_path, mmap_mode="r"),
        np.load(right_path, mmap_mode="r"),
        np.load(global_path, mmap_mode="r"),
    )


def build_cached_part_augments(raw_train, y_train, left_dim, right_dim, global_dim, repeats, cache_dir, force=False, feature_mode="frame"):
    if repeats <= 0:
        return None, None, None, None

    cache_dir = feature_cache_dir(cache_dir, feature_mode)
    cache_dir.mkdir(parents=True, exist_ok=True)
    left_path = cache_dir / f"aug_r{repeats}_left.npy"
    right_path = cache_dir / f"aug_r{repeats}_right.npy"
    global_path = cache_dir / f"aug_r{repeats}_global.npy"
    y_path = cache_dir / f"aug_r{repeats}_labels.npy"

    expected = len(y_train) * repeats
    if not force and all(p.exists() for p in [left_path, right_path, global_path, y_path]):
        labels = np.load(y_path)
        if len(labels) == expected:
            print(f"Using cached local-global augmentation repeats={repeats}.")
            return (
                np.load(left_path, mmap_mode="r"),
                np.load(right_path, mmap_mode="r"),
                np.load(global_path, mmap_mode="r"),
                labels,
            )

    print(f"Building cached local-global augmentation repeats={repeats} mode={feature_mode}...")
    left_aug = np.lib.format.open_memmap(
        left_path, mode="w+", dtype=np.float32, shape=(expected, SEQUENCE_LENGTH, left_dim)
    )
    right_aug = np.lib.format.open_memmap(
        right_path, mode="w+", dtype=np.float32, shape=(expected, SEQUENCE_LENGTH, right_dim)
    )
    global_aug = np.lib.format.open_memmap(
        global_path, mode="w+", dtype=np.float32, shape=(expected, SEQUENCE_LENGTH, global_dim)
    )
    aug_y = np.empty(expected, dtype=np.int64)

    cursor = 0
    for i in tqdm(range(len(y_train)), desc="Aug"):
        raw = raw_train[i]
        for _ in range(repeats):
            aug_raw = augment_fixed(raw)
            left_seq, right_seq, global_seq = extract_feature_triplet(aug_raw, feature_mode, RECTIFY_ALPHA)
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


def make_loader(left, right, global_features, labels, batch_size=32, shuffle=False, drop_last=False):
    return DataLoader(
        TensorDataset(
            torch.tensor(left),
            torch.tensor(right),
            torch.tensor(global_features),
            torch.tensor(labels, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def collect_logits(model, loader, device, amp=False):
    model.eval()
    logits_all = []
    with torch.no_grad():
        for left, right, global_features, _ in loader:
            left = left.to(device)
            right = right.to(device)
            global_features = global_features.to(device)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(left, right, global_features)
            logits_all.append(logits.float().cpu().numpy())
    return np.concatenate(logits_all, axis=0)


def collect_probs(model, loader, device, amp=False, tta_passes=0, noise_std=0.01):
    all_probs = []
    logits = collect_logits(model, loader, device, amp=amp)
    all_probs.append(torch.softmax(torch.tensor(logits), dim=1).numpy())
    if tta_passes <= 0:
        return all_probs[0], all_probs[0]

    model.eval()
    with torch.no_grad():
        for _ in range(tta_passes):
            tta_probs = []
            for left, right, global_features, _ in loader:
                left = left.to(device)
                right = right.to(device)
                global_features = global_features.to(device)
                left = left + torch.randn_like(left) * noise_std
                right = right + torch.randn_like(right) * noise_std
                global_features = global_features + torch.randn_like(global_features) * noise_std
                with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                    logits = model(left, right, global_features)
                tta_probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
            all_probs.append(np.concatenate(tta_probs, axis=0))
    return all_probs[0], np.mean(all_probs, axis=0)


def eval_accuracy(model, loader, device, criterion=None, amp=False):
    model.eval()
    correct, total, loss_total = 0, 0, 0.0
    with torch.no_grad():
        for left, right, global_features, labels in loader:
            left = left.to(device)
            right = right.to(device)
            global_features = global_features.to(device)
            labels = labels.to(device)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(left, right, global_features)
                loss = criterion(logits, labels) if criterion is not None else None
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            if loss is not None:
                loss_total += float(loss.item())
    avg_loss = loss_total / max(len(loader), 1) if criterion is not None else None
    return 100.0 * correct / max(total, 1), avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train local-global ArcFace on WLASL-N.")
    parser.add_argument("--num-glosses", type=int, required=True)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--json-path", default=JSON_PATH)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--swa-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--aug-repeats", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=16.0)
    parser.add_argument("--tta-passes", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0, help="Smoke-test limit per split.")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--feature-mode", choices=["frame", "seq", "exp13", "seqplus"], default="frame")
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    return parser.parse_args()


def default_aug_repeats(num_glosses):
    if num_glosses <= 300:
        return 10
    if num_glosses <= 1000:
        return 5
    return 3


def main():
    args = parse_args()
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    aug_repeats = default_aug_repeats(args.num_glosses) if args.aug_repeats is None else args.aug_repeats
    output_prefix = args.output_prefix or f"wlasl{args.num_glosses}_local_global_arcface_m02"
    cache_dir = args.cache_dir or os.path.join("rework_model", "cache", f"wlasl{args.num_glosses}_local_global_arcface")

    print(f"Using device: {device}; AMP={amp}")
    print(f"Subset: WLASL-{args.num_glosses}; data={args.data_dir}; cache={cache_dir}; aug_repeats={aug_repeats}; feature_mode={args.feature_mode}")

    splits, actions, label_map = load_subset_raw(args.data_dir, args.json_path, args.num_glosses, args.limit_samples)
    nc = len(actions)
    y_tr = np.array(splits["train"]["labels"], dtype=np.int64)
    y_val = np.array(splits["val"]["labels"], dtype=np.int64)
    y_test = np.array(splits["test"]["labels"], dtype=np.int64)
    cw_t = get_class_weights(y_tr, nc).to(device)

    left_tr, right_tr, global_tr = encode_parts(splits["train"]["raw"], "train", cache_dir, force=args.force_cache, feature_mode=args.feature_mode)
    left_val, right_val, global_val = encode_parts(splits["val"]["raw"], "val", cache_dir, force=args.force_cache, feature_mode=args.feature_mode)
    left_test, right_test, global_test = encode_parts(splits["test"]["raw"], "test", cache_dir, force=args.force_cache, feature_mode=args.feature_mode)
    print(f"Left/right/global dims: {left_tr.shape[2]} / {right_tr.shape[2]} / {global_tr.shape[2]}")

    left_aug, right_aug, global_aug, aug_y = build_cached_part_augments(
        splits["train"]["raw"],
        y_tr,
        left_tr.shape[2],
        right_tr.shape[2],
        global_tr.shape[2],
        repeats=aug_repeats,
        cache_dir=cache_dir,
        force=args.force_cache,
        feature_mode=args.feature_mode,
    )

    train_loader = DataLoader(
        CachedAugmentedPartDataset(left_tr, right_tr, global_tr, y_tr, left_aug, right_aug, global_aug, aug_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(y_tr) + (0 if aug_y is None else len(aug_y)) >= args.batch_size,
    )
    val_loader = make_loader(left_val, right_val, global_val, y_val, batch_size=args.batch_size)
    test_loader = make_loader(left_test, right_test, global_test, y_test, batch_size=args.batch_size)

    model = LocalGlobalArcFace(
        left_tr.shape[2],
        right_tr.shape[2],
        global_tr.shape[2],
        nc,
        scale=args.scale,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Classes: {nc}; Params: {params:,}")

    criterion = ArcFaceLoss(scale=args.scale, margin=args.margin, cw=cw_t)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    Path("rework_model").mkdir(exist_ok=True)
    pre_path = Path("rework_model") / f"{output_prefix}_single.pth"
    swa_path = Path("rework_model") / f"{output_prefix}_swa.pth"
    best_path = Path("rework_model") / f"{output_prefix}_best.pth"

    history = []
    best_val = float("-inf")
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        steps = 0
        for left, right, global_features, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False, mininterval=5.0
        ):
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
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(left_m, right_m, global_m)
                loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())
            steps += 1

        val_acc, val_loss = eval_accuracy(model, val_loader, device, criterion=criterion, amp=amp)
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / max(steps, 1), "val_acc": val_acc, "val_loss": val_loss})
        print(f"Ep {epoch + 1}: Val {val_acc:.2f}% loss={val_loss:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            patience = 0
            torch.save(model.state_dict(), pre_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stop epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(pre_path, map_location=device, weights_only=True))
    pre_probs, pre_tta_probs = collect_probs(model, test_loader, device, amp=amp, tta_passes=args.tta_passes)
    pre_top1, pre_top5 = topk_metrics(pre_probs, y_test)
    pre_tta_top1, pre_tta_top5 = topk_metrics(pre_tta_probs, y_test)
    pre_val, _ = eval_accuracy(model, val_loader, device, criterion=None, amp=amp)
    print(f"Pre-SWA: top1={pre_top1:.2f}% top5={pre_top5:.2f}% val={pre_val:.2f}%")

    swa_top1 = swa_top5 = swa_tta_top1 = swa_tta_top5 = swa_val = None
    if args.swa_epochs > 0:
        print(f"Phase 2: SWA ({args.swa_epochs} epochs)")
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        for epoch in range(args.swa_epochs):
            model.train()
            for left, right, global_features, labels in tqdm(
                train_loader, desc=f"SWA {epoch + 1}/{args.swa_epochs}", leave=False, mininterval=5.0
            ):
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
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(left_m, right_m, global_m)
                    loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"  SWA ep {epoch + 1}/{args.swa_epochs}")
        torch.save(swa_model.module.state_dict(), swa_path)

        swa_model.eval()
        swa_probs, swa_tta_probs_arr = collect_probs(swa_model, test_loader, device, amp=amp, tta_passes=args.tta_passes)
        swa_top1, swa_top5 = topk_metrics(swa_probs, y_test)
        swa_tta_top1, swa_tta_top5 = topk_metrics(swa_tta_probs_arr, y_test)
        swa_val, _ = eval_accuracy(swa_model, val_loader, device, criterion=None, amp=amp)
        print(f"SWA: top1={swa_top1:.2f}% top5={swa_top5:.2f}% val={swa_val:.2f}%")

    if swa_val is not None and swa_val >= pre_val:
        selected_name = "SWA"
        selected_state = torch.load(swa_path, map_location=device, weights_only=True)
        selected_top1, selected_top5 = swa_top1, swa_top5
        selected_tta_top1, selected_tta_top5 = swa_tta_top1, swa_tta_top5
        selected_val = swa_val
    else:
        selected_name = "pre-SWA"
        selected_state = torch.load(pre_path, map_location=device, weights_only=True)
        selected_top1, selected_top5 = pre_top1, pre_top5
        selected_tta_top1, selected_tta_top5 = pre_tta_top1, pre_tta_top5
        selected_val = pre_val

    torch.save(selected_state, best_path)

    selected_model = LocalGlobalArcFace(left_tr.shape[2], right_tr.shape[2], global_tr.shape[2], nc, scale=args.scale).to(device)
    selected_model.load_state_dict(selected_state)
    val_logits = collect_logits(selected_model, val_loader, device, amp=amp)
    test_logits = collect_logits(selected_model, test_loader, device, amp=amp)

    cache_dir_path = feature_cache_dir(cache_dir, args.feature_mode)
    np.save(cache_dir_path / "val_logits.npy", val_logits)
    np.save(cache_dir_path / "test_logits.npy", test_logits)
    np.save(cache_dir_path / "val_labels.npy", y_val)
    np.save(cache_dir_path / "test_labels.npy", y_test)

    result = {
        "num_glosses": args.num_glosses,
        "num_classes": nc,
        "data_dir": args.data_dir,
        "cache_dir": str(cache_dir_path),
        "feature_mode": args.feature_mode,
        "feature_dims": {"left": int(left_tr.shape[2]), "right": int(right_tr.shape[2]), "global": int(global_tr.shape[2])},
        "output_prefix": output_prefix,
        "samples": {"train": int(len(y_tr)), "val": int(len(y_val)), "test": int(len(y_test))},
        "params": int(params),
        "amp": bool(amp),
        "batch_size": args.batch_size,
        "aug_repeats": int(aug_repeats),
        "mixup_alpha": float(args.mixup_alpha),
        "epochs_requested": args.epochs,
        "epochs_ran": len(history),
        "swa_epochs": args.swa_epochs,
        "selected": selected_name,
        "pre_swa": {
            "val_top1": float(pre_val),
            "test_top1": float(pre_top1),
            "test_top5": float(pre_top5),
            "tta_top1": float(pre_tta_top1),
            "tta_top5": float(pre_tta_top5),
        },
        "swa": None
        if swa_val is None
        else {
            "val_top1": float(swa_val),
            "test_top1": float(swa_top1),
            "test_top5": float(swa_top5),
            "tta_top1": float(swa_tta_top1),
            "tta_top5": float(swa_tta_top5),
        },
        "selected_metrics": {
            "val_top1": float(selected_val),
            "test_top1": float(selected_top1),
            "test_top5": float(selected_top5),
            "tta_top1": float(selected_tta_top1),
            "tta_top5": float(selected_tta_top5),
        },
        "checkpoints": {
            "pre_swa": str(pre_path),
            "swa": str(swa_path) if args.swa_epochs > 0 else None,
            "best": str(best_path),
        },
        "actions": actions,
        "history": history,
    }

    Path("diagnostic").mkdir(exist_ok=True)
    result_path = Path("diagnostic") / f"{output_prefix}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("")
    print(f"Local-global part-aware ArcFace WLASL-{args.num_glosses}")
    print(f"  Selected: {selected_name}")
    print(f"  Top-1: {selected_top1:.2f}%")
    print(f"  Top-5: {selected_top5:.2f}%")
    if args.tta_passes > 0:
        print(f"  TTA Top-1/Top-5: {selected_tta_top1:.2f}% / {selected_tta_top5:.2f}%")
    print(f"  Val:   {selected_val:.2f}%")
    print(f"  Params: {params:,}")
    print(f"  Checkpoint: {best_path}")
    print(f"  Result JSON: {result_path}")


if __name__ == "__main__":
    main()
