from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from diagnose_local_global_failures import get_transform
from wlasl_train_b6_logit_heads_subset import FELFSLR
from wlasl_train_local_global_arcface import topk_metrics
from wlasl_train_local_global_arcface_subset import DATA_DIR, JSON_PATH
from wlasl_train_old_localglobal_sweep_subset import encode_feature_parts, load_subset_raw_for_source


def make_args(gate: bool, mode: str, high: float):
    return Namespace(
        left_branch_dim=96,
        right_branch_dim=96,
        global_branch_dim=96,
        dropout=0.2,
        scale=16.0,
        use_lrg_head=True,
        use_rf_head=True,
        lrg_logit_weight=1.0,
        rf_logit_weight=0.5,
        normalize_fused_logits=True,
        trainable_fusion_weights=False,
        trainable_fusion_normalizer=False,
        use_global_reliability_gate=gate,
        global_gate_mode=mode,
        global_gate_low=1e-8,
        global_gate_high=high,
        lrg_residual_gamma_init=0.25,
        lrg_residual_gamma_mode="fixed",
        rf_residual_beta_init=0.25,
        rf_residual_beta_mode="fixed",
    )


def make_loader(left, right, global_features, labels, batch_size):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(left, dtype=np.float32)),
            torch.from_numpy(np.asarray(right, dtype=np.float32)),
            torch.from_numpy(np.asarray(global_features, dtype=np.float32)),
            torch.from_numpy(np.asarray(labels, dtype=np.int64)),
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def metrics(logits, labels):
    probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=1).numpy()
    t1, t5 = topk_metrics(probs, labels)
    return float(t1), float(t5)


def global_corrupt(global_features: torch.Tensor, mode: str, seed: int):
    if mode == "normal":
        return global_features
    if mode.startswith("scale_"):
        return global_features * float(mode.split("_", 1)[1])
    if mode == "zero":
        return torch.zeros_like(global_features)
    if mode == "gaussian_noise":
        gen = torch.Generator(device=global_features.device)
        gen.manual_seed(seed)
        return global_features + torch.randn(global_features.shape, generator=gen, device=global_features.device) * 0.25
    if mode == "permute_samples":
        gen = torch.Generator(device=global_features.device)
        gen.manual_seed(seed)
        perm = torch.randperm(global_features.shape[0], generator=gen, device=global_features.device)
        return global_features[perm]
    if mode == "temporal_shuffle":
        gen = torch.Generator(device=global_features.device)
        gen.manual_seed(seed)
        idx = torch.randperm(global_features.shape[1], generator=gen, device=global_features.device)
        return global_features[:, idx]
    if mode == "frame_dropout":
        out = global_features.clone()
        out[:, ::2] = 0
        return out
    if mode == "feature_dropout":
        out = global_features.clone()
        out[:, :, ::2] = 0
        return out
    raise ValueError(mode)


def collect(model, loader, device, transform_name: str = "normal", global_mode: str = "normal", seed: int = 42):
    transform = get_transform(transform_name, seed, None)
    outs, labels, gates, norms = [], [], [], []
    model.eval()
    with torch.no_grad():
        for left, right, global_features, y in loader:
            left = left.to(device)
            right = right.to(device)
            global_features = global_features.to(device)
            left, right, global_features = transform(left, right, global_features)
            global_features = global_corrupt(global_features, global_mode, seed)
            norms.append(global_features.abs().mean(dim=(1, 2)).float().cpu().numpy())
            out = model.forward_all(left, right, global_features)
            outs.append(out["fused"].float().cpu().numpy())
            if "global_reliability_gate" in out:
                gates.append(out["global_reliability_gate"].float().cpu().numpy())
            labels.append(y.numpy())
    return (
        np.concatenate(outs),
        np.concatenate(labels),
        np.concatenate(norms),
        float(np.concatenate(gates).mean()) if gates else "",
    )


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--num-glosses", type=int, required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--gate-high", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    splits, actions, _ = load_subset_raw_for_source(DATA_DIR, JSON_PATH, args.num_glosses, "json_first_n", 0, labels_only=True)
    labels = np.asarray(splits["test"]["labels"], dtype=np.int64)
    left, right, global_features = encode_feature_parts(splits["test"]["raw"], "test", args.cache_dir, "old", force=False)
    loader = make_loader(left, right, global_features, labels, args.batch_size)
    dims = (left.shape[-1], right.shape[-1], global_features.shape[-1])
    high = args.gate_high
    if high is None:
        high = 0.7259677648544312 if args.num_glosses == 300 else 0.7000000000

    models = {
        "FELF-SLR": FELFSLR(dims, len(actions), make_args(False, "soft", high)).to(device),
        "FELF-G-hard": FELFSLR(dims, len(actions), make_args(True, "hard", high)).to(device),
        "FELF-G-soft": FELFSLR(dims, len(actions), make_args(True, "soft", high)).to(device),
    }
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    for model in models.values():
        model.load_state_dict(state, strict=False)

    global_modes = ["normal", "scale_0.75", "scale_0.5", "scale_0.25", "zero", "gaussian_noise", "permute_samples", "temporal_shuffle", "frame_dropout", "feature_dropout"]
    global_rows = []
    norm_rows = []
    for mode in global_modes:
        for name, model in models.items():
            logits, y, norms, gate = collect(model, loader, device, "normal", mode)
            top1, top5 = metrics(logits, y)
            global_rows.append({"condition": mode, "model": name, "top1": top1, "top5": top5, "gate_mean": gate})
            if name == "FELF-G-soft":
                norm_rows.append(
                    {
                        "condition": mode,
                        "norm_mean": float(norms.mean()),
                        "norm_std": float(norms.std(ddof=1)),
                        "norm_min": float(norms.min()),
                        "norm_max": float(norms.max()),
                        "gate_mean": gate,
                    }
                )

    stress_tests = ["normal", "temporal_shuffle", "center_frame_repeat", "mask_left", "mask_right", "mask_global", "crop_first_half", "crop_middle_third", "crop_second_half"]
    stress_rows = []
    for stress in stress_tests:
        for name, model in models.items():
            logits, y, norms, gate = collect(model, loader, device, stress, "normal")
            top1, top5 = metrics(logits, y)
            stress_rows.append({"stress": stress, "model": name, "top1": top1, "top5": top5, "gate_mean": gate})

    prefix = Path(args.output_prefix)
    write_csv(prefix.with_name(prefix.name + "_global_corruptions.csv"), global_rows)
    write_csv(prefix.with_name(prefix.name + "_global_norms.csv"), norm_rows)
    write_csv(prefix.with_name(prefix.name + "_stress.csv"), stress_rows)
    summary = {"global_corruptions": global_rows, "global_norms": norm_rows, "stress": stress_rows, "gate_high": high}
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
