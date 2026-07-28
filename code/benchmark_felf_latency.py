from __future__ import annotations

import csv
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from wlasl_train_b6_logit_heads_subset import FELFSLR, make_b6_head


def make_args(num_classes: int):
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
        lrg_residual_gamma_init=0.25,
        lrg_residual_gamma_mode="fixed",
        rf_residual_beta_init=0.25,
        rf_residual_beta_mode="fixed",
    )


def load_state_if_exists(model: torch.nn.Module, path: str, device: torch.device) -> None:
    ckpt = Path(path)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(model: torch.nn.Module, device: torch.device, batch_size: int, warmup: int, iters: int) -> dict:
    model.eval()
    left = torch.randn(batch_size, 40, 165, device=device)
    right = torch.randn(batch_size, 40, 165, device=device)
    global_features = torch.randn(batch_size, 40, 23, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(left, right, global_features)
        synchronize(device)
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(left, right, global_features)
            synchronize(device)
            times.append(time.perf_counter() - start)
    arr = np.asarray(times, dtype=np.float64)
    per_batch_ms = arr * 1000.0
    per_sample_ms = per_batch_ms / float(batch_size)
    return {
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "per_batch_ms_mean": float(per_batch_ms.mean()),
        "per_batch_ms_std": float(per_batch_ms.std(ddof=1)) if len(per_batch_ms) > 1 else 0.0,
        "per_sample_ms_mean": float(per_sample_ms.mean()),
        "per_sample_ms_std": float(per_sample_ms.std(ddof=1)) if len(per_sample_ms) > 1 else 0.0,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--num-classes", type=int, default=300)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--output-prefix", default="diagnostic/felf_latency")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model_args = make_args(args.num_classes)
    dims = (165, 165, 23)
    ckpt_suffix = "wlasl300" if args.num_classes == 300 else "wlasl100"
    models = {
        "Local-Global ArcFace baseline": make_b6_head(dims, args.num_classes, model_args, kind="main"),
        "LRG expert": make_b6_head(dims, args.num_classes, model_args, kind="lrg"),
        "RF expert": make_b6_head(dims, args.num_classes, model_args, kind="rf"),
        "FELF-SLR staged": FELFSLR(dims, args.num_classes, model_args),
    }
    checkpoints = {
        "Local-Global ArcFace baseline": f"rework_model/{ckpt_suffix}_old_B6_dropout02_swa.pth",
        "LRG expert": f"rework_model/{ckpt_suffix}_B6_LRG_ResidualGamma_fixed025_swa.pth",
        "RF expert": f"rework_model/{ckpt_suffix}_B6_RF_ResidualBeta_fixed025_swa.pth",
        "FELF-SLR staged": f"rework_model/{ckpt_suffix}_FELF_SLR_Staged_FreezeMain_best_by_val.pth",
    }
    rows = []
    for name, model in models.items():
        model = model.to(device)
        load_state_if_exists(model, checkpoints[name], device)
        result = benchmark(model, device, args.batch_size, args.warmup, args.iters)
        result.update(
            {
                "model": name,
                "params": int(sum(p.numel() for p in model.parameters())),
                "device": str(device),
                "checkpoint": checkpoints[name],
            }
        )
        rows.append(result)
        print(f"{name}: {result['per_sample_ms_mean']:.3f} ms/sample")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(r"\begin{tabular}{lll}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Latency} & \textbf{Description} \\")
    print(r"\midrule")
    descriptions = {
        "Local-Global ArcFace baseline": "Single retrieval expert",
        "LRG expert": "Temporal-phase expert",
        "RF expert": "Reliability expert",
        "FELF-SLR staged": "Logit-level expert fusion",
    }
    for row in rows:
        print(f"{row['model']} & {row['per_sample_ms_mean']:.3f} ms & {descriptions[row['model']]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"Saved {prefix.with_suffix('.json')} and {prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
