"""Failure-mode diagnostics for trained LocalGlobalArcFace WLASL models.

This script performs evaluation-time perturbations only. It does not train,
rewrite caches, or modify the original training scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wlasl_train_local_global_arcface import ArcFaceLoss, LocalGlobalArcFace, topk_metrics
try:
    from wlasl_train_old_localglobal_sweep_subset import OldLocalGlobalSweepArcFace
except Exception:
    OldLocalGlobalSweepArcFace = None


def parse_noise_stds(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def candidate_cache_dirs(cache_dir: Path, feature_mode: str) -> list[Path]:
    dirs = []
    for d in [cache_dir / feature_mode, cache_dir / "frame_old", cache_dir]:
        if d not in dirs:
            dirs.append(d)
    return dirs


def find_cache_file(cache_dir: Path, feature_mode: str, split: str, stream: str) -> Path:
    names = [
        f"{split}_{stream}.npy",
        f"{split}_{feature_mode}_{stream}.npy",
        f"{split}_old_{stream}.npy",
        f"{split}_{stream}_dr.npy",
    ]
    for d in candidate_cache_dirs(cache_dir, feature_mode):
        for name in names:
            p = d / name
            if p.exists():
                return p
    raise FileNotFoundError(
        f"Could not find {split}/{stream} cache under {cache_dir}. Tried names={names}"
    )


def find_label_file(cache_dir: Path, feature_mode: str, split: str) -> Path | None:
    names = [
        f"{split}_labels.npy",
        f"{split}_y.npy",
        f"y_{split}.npy",
        f"{split}_labels_dr.npy",
    ]
    for d in candidate_cache_dirs(cache_dir, feature_mode):
        for name in names:
            p = d / name
            if p.exists():
                return p
    for d in candidate_cache_dirs(cache_dir, feature_mode):
        for child in d.iterdir() if d.exists() else []:
            if child.is_dir():
                p = child / f"{split}_labels.npy"
                if p.exists():
                    return p
    return None


def load_split(cache_dir: Path, feature_mode: str, split: str):
    left = np.load(find_cache_file(cache_dir, feature_mode, split, "left"), mmap_mode="r")
    right = np.load(find_cache_file(cache_dir, feature_mode, split, "right"), mmap_mode="r")
    global_features = np.load(find_cache_file(cache_dir, feature_mode, split, "global"), mmap_mode="r")
    label_path = find_label_file(cache_dir, feature_mode, split)
    if label_path is None:
        raise FileNotFoundError(
            f"No labels found for split={split}. Provide a cache with {split}_labels.npy or run diagnostics from a run cache."
        )
    labels = np.load(label_path).astype(np.int64)
    return left, right, global_features, labels


def infer_model_kwargs(state: dict, left_dim: int, right_dim: int, global_dim: int, num_classes: int):
    left_branch = state.get("left_stem.net.0.weight")
    classifier = state.get("classifier.weight")
    if left_branch is None or classifier is None:
        return {}
    d_branch = int(left_branch.shape[0])
    d_model = int(classifier.shape[1])
    if d_model != d_branch * 3:
        raise ValueError(
            f"Checkpoint looks incompatible with LocalGlobalArcFace: classifier dim={d_model}, branch dim={d_branch}."
        )
    return {
        "left_inp": left_dim,
        "right_inp": right_dim,
        "global_inp": global_dim,
        "nc": num_classes,
        "d_branch": d_branch,
    }


def diagnostic_for_checkpoint(checkpoint_path: Path) -> Path | None:
    stem = checkpoint_path.stem
    for suffix in ["_best_by_val", "_single", "_swa"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    path = Path("diagnostic") / f"{stem}.json"
    return path if path.exists() else None


def make_sweep_model_from_diagnostic(
    diag: dict,
    left_dim: int,
    right_dim: int,
    global_dim: int,
    num_classes: int,
    scale_override: float | None = None,
):
    if OldLocalGlobalSweepArcFace is None:
        raise RuntimeError("OldLocalGlobalSweepArcFace could not be imported for sweep checkpoint diagnostics.")
    return OldLocalGlobalSweepArcFace(
        left_dim,
        right_dim,
        global_dim,
        num_classes,
        left_branch_dim=int(diag.get("left_branch_dim", 96)),
        right_branch_dim=int(diag.get("right_branch_dim", 96)),
        global_branch_dim=int(diag.get("global_branch_dim", 96)),
        conv_kernel=int(diag.get("conv_kernel", 3)),
        conv_layers=int(diag.get("conv_layers", 1)),
        num_layers=int(diag.get("num_layers", 2)),
        num_heads=int(diag.get("num_heads", 4)),
        ff_dim=int(diag.get("ff_dim", 768)),
        dropout=float(diag.get("dropout", 0.3)),
        pool_mode=diag.get("pool_mode", "mean"),
        temporal_head=diag.get("temporal_head", "transformer"),
        lite_kernel=int(diag.get("lite_kernel", 5)),
        scale=float(scale_override if scale_override is not None else diag.get("scale", diag.get("arcface_scale", 16.0))),
        factorization_mode=diag.get("factorization_mode", "none"),
        static_motion_gate_bias=float(diag.get("static_motion_gate_bias", -1.5)),
        relation_dim=int(diag.get("relation_dim", 96)),
        use_stream_dropout=bool(diag.get("use_stream_dropout", False)),
        stream_dropout_left=float(diag.get("stream_dropout_left", 0.05)),
        stream_dropout_right=float(diag.get("stream_dropout_right", 0.15)),
        stream_dropout_global=float(diag.get("stream_dropout_global", 0.15)),
        use_reliability_fusion=bool(diag.get("use_reliability_fusion", False)),
        reliability_hidden_dim=int(diag.get("reliability_hidden_dim", 64)),
        reliability_bias=float(diag.get("reliability_bias", 0.0)),
        use_residual_reliability_fusion=bool(diag.get("use_residual_reliability_fusion", False)),
        residual_rf_hidden_dim=int(diag.get("residual_rf_hidden_dim", 64)),
        residual_rf_beta_init=float(diag.get("residual_rf_beta_init", 0.0)),
        residual_rf_beta_mode=diag.get("residual_rf_beta_mode", "learnable"),
        use_lrg_residual=bool(diag.get("use_lrg_residual", False)),
        lrg_residual_gamma_init=float(diag.get("lrg_residual_gamma_init", 0.0)),
        lrg_residual_gamma_mode=diag.get("lrg_residual_gamma_mode", "learnable"),
        use_global_gate=bool(diag.get("use_global_gate", False)),
        global_gate_hidden_dim=int(diag.get("global_gate_hidden_dim", 64)),
        global_gate_bias=float(diag.get("global_gate_bias", 2.0)),
        use_branch_gates=bool(diag.get("use_branch_gates", False)),
        branch_gate_hidden_dim=int(diag.get("branch_gate_hidden_dim", 64)),
        branch_gate_bias=float(diag.get("branch_gate_bias", 2.0)),
        use_temporal_conv_bridge=bool(diag.get("use_temporal_conv_bridge", False)),
        temporal_conv_bridge_kernel=int(diag.get("temporal_conv_bridge_kernel", 5)),
        temporal_conv_bridge_dropout=float(diag.get("temporal_conv_bridge_dropout", 0.2)),
        zero_init_temporal_conv_bridge=bool(diag.get("zero_init_temporal_conv_bridge", True)),
        use_kinematic_residual=bool(diag.get("use_kinematic_residual", False)),
        kinematic_hidden_dim=int(diag.get("kinematic_hidden_dim", 288)),
        kinematic_dropout=float(diag.get("kinematic_dropout", 0.1)),
        kinematic_scale=float(diag.get("kinematic_scale", 1.0)),
        kinematic_input=diag.get("kinematic_input", "concat"),
        zero_init_kinematic=bool(diag.get("zero_init_kinematic", True)),
        bounded_kinematic=bool(diag.get("bounded_kinematic", False)),
        use_cross_hand_attn=bool(diag.get("use_cross_hand_attn", False)),
        cross_attn_dim=int(diag.get("cross_attn_dim", 64)),
        cross_attn_dropout=float(diag.get("cross_attn_dropout", 0.1)),
        cross_attn_scale=float(diag.get("cross_attn_scale", 1.0)),
        cross_attn_direction=diag.get("cross_attn_direction", "bidirectional"),
        zero_init_cross_hand=bool(diag.get("zero_init_cross_hand", True)),
        use_hand_global_attn=bool(diag.get("use_hand_global_attn", False)),
        hand_global_attn_dim=int(diag.get("hand_global_attn_dim", 64)),
        hand_global_attn_dropout=float(diag.get("hand_global_attn_dropout", 0.1)),
        hand_global_attn_scale=float(diag.get("hand_global_attn_scale", 1.0)),
        use_ot_align=bool(diag.get("use_ot_align", False)),
        ot_align_dim=int(diag.get("ot_align_dim", 64)),
        ot_epsilon=float(diag.get("ot_epsilon", 0.05)),
        ot_iters=int(diag.get("ot_iters", 8)),
        ot_scale=float(diag.get("ot_scale", 1.0)),
        ot_dustbin=bool(diag.get("ot_dustbin", False)),
        classifier_type=diag.get("classifier_type", "cosine") or "cosine",
        num_subcenters=int(diag.get("num_subcenters", 2)),
        subcenter_reduce=diag.get("subcenter_reduce", "max"),
        subcenter_lse_temperature=float(diag.get("subcenter_lse_temperature", 0.08)),
    )


def load_model_for_checkpoint(
    checkpoint_path: Path,
    state: dict,
    left_dim: int,
    right_dim: int,
    global_dim: int,
    num_classes: int,
    scale: float,
):
    diag_path = diagnostic_for_checkpoint(checkpoint_path)
    if diag_path is not None:
        diag = json.loads(diag_path.read_text(encoding="utf-8-sig"))
        if diag.get("model") == "OldLocalGlobalSweepArcFace":
            return make_sweep_model_from_diagnostic(diag, left_dim, right_dim, global_dim, num_classes, scale), str(diag_path)
    kwargs = infer_model_kwargs(state, left_dim, right_dim, global_dim, num_classes)
    return LocalGlobalArcFace(**kwargs, scale=scale), None


def make_loader(left, right, global_features, labels, batch_size: int) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.asarray(left, dtype=np.float32)),
        torch.from_numpy(np.asarray(right, dtype=np.float32)),
        torch.from_numpy(np.asarray(global_features, dtype=np.float32)),
        torch.from_numpy(np.asarray(labels, dtype=np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def repeat_slice(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    part = x[:, start:end]
    if part.shape[1] <= 0:
        return x
    reps = int(np.ceil(x.shape[1] / part.shape[1]))
    return part.repeat(1, reps, 1)[:, : x.shape[1]]


def get_transform(name: str, seed: int, noise_std: float | None = None) -> Callable:
    def identity(left, right, global_features):
        return left, right, global_features

    def shuffle(left, right, global_features):
        generator = torch.Generator(device=left.device)
        generator.manual_seed(seed)
        outs = []
        for stream in [left, right, global_features]:
            rows = [stream[i, torch.randperm(stream.shape[1], generator=generator, device=left.device)] for i in range(stream.shape[0])]
            outs.append(torch.stack(rows, dim=0))
        return outs[0], outs[1], outs[2]

    def center_repeat(left, right, global_features):
        t = left.shape[1] // 2
        return left[:, t : t + 1].repeat(1, left.shape[1], 1), right[:, t : t + 1].repeat(1, right.shape[1], 1), global_features[:, t : t + 1].repeat(1, global_features.shape[1], 1)

    def mask(which: str):
        def inner(left, right, global_features):
            z_left, z_right, z_global = left, right, global_features
            if which in {"left", "both_hands", "global_only"}:
                z_left = torch.zeros_like(left)
            if which in {"right", "both_hands", "global_only"}:
                z_right = torch.zeros_like(right)
            if which in {"global", "hands_only"}:
                z_global = torch.zeros_like(global_features)
            return z_left, z_right, z_global

        return inner

    def crop(kind: str):
        def inner(left, right, global_features):
            t = left.shape[1]
            if kind == "first_half":
                start, end = 0, max(1, t // 2)
            elif kind == "second_half":
                start, end = t // 2, t
            else:
                span = max(1, t // 3)
                start = (t - span) // 2
                end = start + span
            return repeat_slice(left, start, end), repeat_slice(right, start, end), repeat_slice(global_features, start, end)

        return inner

    def noise(which: str, std: float):
        def inner(left, right, global_features):
            if which in {"left", "all"}:
                left = left + torch.randn_like(left) * std
            if which in {"right", "all"}:
                right = right + torch.randn_like(right) * std
            if which in {"global", "all"}:
                global_features = global_features + torch.randn_like(global_features) * std
            return left, right, global_features

        return inner

    if name == "normal":
        return identity
    if name == "temporal_shuffle":
        return shuffle
    if name == "center_frame_repeat":
        return center_repeat
    if name.startswith("mask_"):
        return mask(name.removeprefix("mask_"))
    if name.startswith("crop_"):
        return crop(name.removeprefix("crop_"))
    if name.startswith("noise_"):
        assert noise_std is not None
        return noise(name.split("_", 1)[1], noise_std)
    raise ValueError(f"Unknown transform: {name}")


def evaluate(model, loader, device, transform, criterion=None, scale: float = 16.0):
    model.eval()
    logits_all, labels_all = [], []
    loss_total, count = 0.0, 0
    with torch.no_grad():
        for left, right, global_features, labels in loader:
            left = left.to(device)
            right = right.to(device)
            global_features = global_features.to(device)
            labels = labels.to(device)
            left, right, global_features = transform(left, right, global_features)
            logits = model(left, right, global_features)
            if criterion is not None:
                loss = criterion(logits, labels)
                loss_total += float(loss.item()) * labels.shape[0]
                count += int(labels.shape[0])
            logits_all.append(logits.float().cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    logits_np = np.concatenate(logits_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    top1, top5 = topk_metrics(probs, labels_np)
    return {
        "top1": float(top1),
        "top5": float(top5),
        "loss": None if criterion is None or count == 0 else float(loss_total / count),
        "logits": logits_np,
        "probs": probs,
        "labels": labels_np,
    }


def interpretation(test_name: str, top1_drop: float, top5_drop: float) -> str:
    if test_name == "temporal_shuffle":
        return "small drop => weak chronology/static bias; large drop => temporal order matters"
    if test_name == "center_frame_repeat":
        return "small drop => static handshape dominates; large drop => motion/trajectory matters"
    if test_name.startswith("mask_global"):
        return "small drop or gain => global may be noisy; large drop => global context is important"
    if test_name.startswith("mask_left") or test_name.startswith("mask_right"):
        return "compare left/right drops to diagnose hand dominance"
    if test_name.startswith("mask_global_only"):
        return "high score => global-context shortcut risk"
    if test_name.startswith("mask_hands_only"):
        return "close to full => global contributes little or adds noise"
    if test_name.startswith("crop_"):
        return "high score on short slice => static/canonical-pose bias; asymmetric halves => phase dependence"
    if test_name.startswith("noise_"):
        return "large drop => stream over-reliance or unstable features"
    return ""


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_outputs(out_dir: Path, probs: np.ndarray, labels: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    top5 = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
    num_classes = probs.shape[1]

    pair_counts: dict[tuple[int, int], int] = {}
    for y, p in zip(labels, pred):
        if int(y) != int(p):
            pair_counts[(int(y), int(p))] = pair_counts.get((int(y), int(p)), 0) + 1
    pair_rows = [
        {"true_class": y, "predicted_class": p, "count": c}
        for (y, p), c in sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    ]
    write_csv(out_dir / "confusion_pairs.csv", pair_rows, ["true_class", "predicted_class", "count"])

    top5_rows = []
    for i, y in enumerate(labels):
        if pred[i] != y and int(y) in set(map(int, top5[i])):
            top5_rows.append(
                {
                    "sample_index": i,
                    "true_label": int(y),
                    "top1_pred": int(pred[i]),
                    "top5_preds": "|".join(map(str, top5[i].tolist())),
                    "top1_prob": float(probs[i, pred[i]]),
                    "true_prob_if_available": float(probs[i, y]),
                }
            )
    write_csv(out_dir / "top5_not_top1.csv", top5_rows, ["sample_index", "true_label", "top1_pred", "top5_preds", "top1_prob", "true_prob_if_available"])

    per_rows = []
    for c in range(num_classes):
        mask = labels == c
        total = int(mask.sum())
        correct = int((pred[mask] == c).sum()) if total else 0
        per_rows.append({"class_id": c, "total": total, "correct": correct, "top1_acc": 100.0 * correct / total if total else 0.0})
    write_csv(out_dir / "per_class_accuracy.csv", per_rows, ["class_id", "total", "correct", "top1_acc"])

    margin_rows = []
    sorted_idx = np.argsort(probs, axis=1)[:, ::-1]
    for i, y in enumerate(labels):
        top1 = int(sorted_idx[i, 0])
        top2 = int(sorted_idx[i, 1])
        margin_rows.append(
            {
                "sample_index": i,
                "true_label": int(y),
                "pred_label": top1,
                "correct": bool(top1 == int(y)),
                "top1_prob": float(probs[i, top1]),
                "top2_prob": float(probs[i, top2]),
                "margin": float(probs[i, top1] - probs[i, top2]),
                "true_in_top5": bool(int(y) in set(map(int, sorted_idx[i, :5]))),
            }
        )
    write_csv(out_dir / "prediction_margins.csv", margin_rows, ["sample_index", "true_label", "pred_label", "correct", "top1_prob", "top2_prob", "margin", "true_in_top5"])

    high_top5_low_top1 = [
        r for r in per_rows if r["total"] > 0 and r["top1_acc"] < 50.0
    ]
    return {
        "num_confusion_pairs": len(pair_rows),
        "top_confusion_pairs": pair_rows[:20],
        "top5_not_top1_count": len(top5_rows),
        "classes_with_low_top1": len(high_top5_low_top1),
    }


def compute_scores(results: dict, baseline: dict, probs: np.ndarray, labels: np.ndarray) -> dict:
    top5_gap = baseline["top5"] - baseline["top1"]
    left_drop = results.get("mask_left_zeroed", {}).get("top1_drop", 0.0)
    right_drop = results.get("mask_right_zeroed", {}).get("top1_drop", 0.0)
    global_only = results.get("mask_global_only", {}).get("top1", 0.0)
    hands_only = results.get("mask_hands_only", {}).get("top1", 0.0)
    shuffle_drop = results.get("temporal_shuffle", {}).get("top1_drop", 0.0)
    center_drop = results.get("center_frame_repeat", {}).get("top1_drop", 0.0)
    pred = probs.argmax(axis=1)
    top5 = np.argsort(probs, axis=1)[:, -5:]
    top5_not_top1 = sum(pred[i] != labels[i] and labels[i] in top5[i] for i in range(len(labels)))
    return {
        "static_bias_score": float(max(0.0, 100.0 - center_drop)),
        "temporal_sensitivity_score": float(shuffle_drop),
        "global_dominance_score": float(global_only),
        "left_right_imbalance_score": float(abs(left_drop - right_drop)),
        "class_crowding_score": float(100.0 * top5_not_top1 / max(1, len(labels))),
        "top5_top1_gap": float(top5_gap),
        "protocol_fragility_warning": bool(abs(shuffle_drop) < 1.0 or abs(center_drop) < 1.0),
        "hands_only_top1": float(hands_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose LocalGlobalArcFace failure modes.")
    parser.add_argument("--num-glosses", type=int, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--feature-mode", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--output-dir", default="diagnostic/failure_diagnosis")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--noise-stds", default="0.005,0.01,0.02")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--scale", type=float, default=16.0)
    parser.add_argument("--margin", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    left, right, global_features, labels = load_split(cache_dir, args.feature_mode, args.split)
    num_classes = int(labels.max()) + 1
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model, model_diagnostic = load_model_for_checkpoint(
        checkpoint_path,
        checkpoint,
        left.shape[-1],
        right.shape[-1],
        global_features.shape[-1],
        num_classes,
        args.scale,
    )
    model = model.to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    criterion = ArcFaceLoss(scale=args.scale, margin=args.margin, cw=None).to(device)
    loader = make_loader(left, right, global_features, labels, args.batch_size)

    tests: list[tuple[str, Callable]] = [
        ("normal", get_transform("normal", args.seed)),
        ("temporal_shuffle", get_transform("temporal_shuffle", args.seed)),
        ("center_frame_repeat", get_transform("center_frame_repeat", args.seed)),
        ("mask_left_zeroed", get_transform("mask_left", args.seed)),
        ("mask_right_zeroed", get_transform("mask_right", args.seed)),
        ("mask_global_zeroed", get_transform("mask_global", args.seed)),
        ("mask_both_hands_zeroed", get_transform("mask_both_hands", args.seed)),
        ("mask_global_only", get_transform("mask_global_only", args.seed)),
        ("mask_hands_only", get_transform("mask_hands_only", args.seed)),
        ("crop_first_half", get_transform("crop_first_half", args.seed)),
        ("crop_second_half", get_transform("crop_second_half", args.seed)),
        ("crop_middle_third", get_transform("crop_middle_third", args.seed)),
    ]
    for std in parse_noise_stds(args.noise_stds):
        for stream in ["left", "right", "global", "all"]:
            tests.append((f"noise_{stream}_std{std:g}", get_transform(f"noise_{stream}", args.seed, noise_std=std)))

    results = {}
    baseline_eval = None
    table_rows = []
    for name, transform in tests:
        item = evaluate(model, loader, device, transform, criterion=criterion, scale=args.scale)
        if name == "normal":
            baseline_eval = item
        assert baseline_eval is not None
        row = {
            "test_name": name,
            "top1": item["top1"],
            "top5": item["top5"],
            "top1_drop": baseline_eval["top1"] - item["top1"],
            "top5_drop": baseline_eval["top5"] - item["top5"],
            "loss": item["loss"],
            "interpretation": interpretation(name, baseline_eval["top1"] - item["top1"], baseline_eval["top5"] - item["top5"]),
        }
        results[name] = {k: v for k, v in row.items() if k != "test_name"}
        table_rows.append(row)
        print(f"{name:28s} top1={row['top1']:.2f} top5={row['top5']:.2f} drop={row['top1_drop']:.2f}/{row['top5_drop']:.2f}")

    write_csv(out_dir / "diagnostic_table.csv", table_rows, ["test_name", "top1", "top5", "top1_drop", "top5_drop", "loss", "interpretation"])
    confusion = write_confusion_outputs(out_dir, baseline_eval["probs"], baseline_eval["labels"])
    scores = compute_scores(results, {"top1": baseline_eval["top1"], "top5": baseline_eval["top5"]}, baseline_eval["probs"], baseline_eval["labels"])

    summary = {
        "num_glosses": int(args.num_glosses),
        "split": args.split,
        "cache_dir": str(cache_dir),
        "feature_mode": args.feature_mode,
        "checkpoint": args.checkpoint,
        "model_diagnostic": model_diagnostic,
        "device": str(device),
        "seed": int(args.seed),
        "feature_dims": {"left": int(left.shape[-1]), "right": int(right.shape[-1]), "global": int(global_features.shape[-1])},
        "num_samples": int(len(labels)),
        "num_classes": int(num_classes),
        "baseline": results["normal"],
        "diagnostics": results,
        "confusion_analysis": confusion,
        "automatic_interpretation": scores,
        "output_files": {
            "diagnostic_table": str(out_dir / "diagnostic_table.csv"),
            "confusion_pairs": str(out_dir / "confusion_pairs.csv"),
            "top5_not_top1": str(out_dir / "top5_not_top1.csv"),
            "per_class_accuracy": str(out_dir / "per_class_accuracy.csv"),
            "prediction_margins": str(out_dir / "prediction_margins.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
