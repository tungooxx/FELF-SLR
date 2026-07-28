from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round((len(ys) - 1) * p))))
    return ys[idx]


@torch.no_grad()
def measure(fn, warmup: int, iters: int, device: torch.device) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(times)),
        "median_ms": float(statistics.median(times)),
        "p95_ms": percentile(times, 0.95),
        "std_ms": float(statistics.pstdev(times)),
    }


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


class _DecoderSelfAttnCompat:
    batch_first = False


def patch_spoter_for_torch_2(model: nn.Module) -> None:
    for layer in getattr(model.transformer.decoder, "layers", []):
        if not hasattr(layer, "self_attn"):
            layer.self_attn = _DecoderSelfAttnCompat()
        original_forward = layer.forward

        def forward_compat(self, *args, _original_forward=original_forward, **kwargs):
            kwargs.pop("tgt_is_causal", None)
            kwargs.pop("memory_is_causal", None)
            return _original_forward(*args, **kwargs)

        layer.forward = types.MethodType(forward_compat, layer)


def spoter_forward_batched(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Batched equivalent of the official SPOTER forward path.

    Official SPOTER flattens a single sample [T, V, C] to [T, E]. For fair
    batch latency, keep batch as the Transformer batch dimension:
    [B, T, V, C] -> [T, B, E].
    """
    if inputs.dim() == 3:
        inputs = inputs.unsqueeze(0)
    src = inputs.flatten(start_dim=2).transpose(0, 1).float()
    tgt = model.class_query.view(1, 1, -1).repeat(1, inputs.size(0), 1)
    h = model.transformer(model.pos + src, tgt).transpose(0, 1)
    return model.linear_class(h).squeeze(1)


def b6_model(num_classes: int, device: torch.device):
    from wlasl_train_felf_slr_subset import LocalGlobalB6Expert

    model = LocalGlobalB6Expert(165, 165, 23, num_classes, dropout=0.2).to(device).eval()
    return model, lambda b: (
        torch.randn(b, 40, 165, device=device),
        torch.randn(b, 40, 165, device=device),
        torch.randn(b, 40, 23, device=device),
    )


def felf_model(num_classes: int, device: torch.device):
    from argparse import Namespace
    from wlasl_train_felf_slr_subset import FELFSLR

    args = Namespace(
        left_branch_dim=96,
        right_branch_dim=96,
        global_branch_dim=96,
        dropout=0.2,
        scale=16.0,
        num_layers=2,
        num_heads=4,
        ff_dim=768,
        branch_stem_kind="base",
        use_lrg_head=True,
        use_rf_head=True,
        lrg_residual_gamma_init=0.25,
        lrg_residual_gamma_mode="fixed",
        rf_residual_beta_init=0.25,
        rf_residual_beta_mode="fixed",
        trainable_fusion_weights=False,
        trainable_fusion_normalizer=False,
        use_global_reliability_gate=False,
        global_gate_mode="soft",
        global_gate_low=1e-8,
        global_gate_high=0.7259677648544312,
        use_adaptive_router=False,
        trainable_adaptive_router=False,
        router_entropy_weight=2.0,
        router_uncertainty_weight=2.0,
        router_bias=-2.0,
        lrg_logit_weight=1.0,
        rf_logit_weight=0.75,
        normalize_fused_logits=True,
    )
    model = FELFSLR((165, 165, 23), num_classes, args).to(device).eval()
    return model, lambda b: (
        torch.randn(b, 40, 165, device=device),
        torch.randn(b, 40, 165, device=device),
        torch.randn(b, 40, 23, device=device),
    )


def spoter_model(num_classes: int, device: torch.device):
    sys.path.insert(0, str(ROOT / "external_baselines" / "spoter"))
    from spoter.spoter_model import SPOTER  # type: ignore

    model = SPOTER(num_classes=num_classes, hidden_dim=108).to(device).eval()
    patch_spoter_for_torch_2(model)
    return model, lambda b: torch.randn(b, 40, 54, 2, device=device)


def stgcn_model(num_classes: int, device: torch.device):
    sys.path.insert(0, str(ROOT / "external" / "st-gcn-sl" / "st-gcn"))
    from net.st_gcn import Model  # type: ignore
    from train_stgcn_sl_wlasl import CUSTOM_27_EDGE

    model = Model(
        in_channels=3,
        num_class=num_classes,
        edge_importance_weighting=True,
        graph_args={
            "layout": "custom",
            "strategy": "spatial",
            "custom_layout": {"num_node": 27, "center": 0, "edge": CUSTOM_27_EDGE},
        },
    ).to(device).eval()
    return model, lambda b: torch.randn(b, 3, 60, 27, 1, device=device)


def posetgcn_model(num_classes: int, device: torch.device):
    sys.path.insert(0, str(ROOT / "external" / "WLASL" / "code" / "TGCN"))
    from tgcn_model import GCN_muti_att  # type: ignore

    model = GCN_muti_att(input_feature=100, hidden_feature=100, num_class=num_classes, p_dropout=0.3, num_stage=20).to(device).eval()
    return model, lambda b: torch.randn(b, 55, 100, device=device)


def rgb_framepool_model(num_classes: int, device: torch.device):
    try:
        import torchvision.models as models
    except Exception:
        return None, None
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device).eval()
    return model, lambda b: torch.randn(b, 3, 224, 224, device=device)


def call_model(name: str, model: nn.Module, sample):
    if name == "SPOTER":
        return spoter_forward_batched(model, sample)
    return model(*sample) if isinstance(sample, tuple) else model(sample)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--batch-sizes", default="1,32")
    p.add_argument("--output-csv", default="diagnostic/model_only_latency_wlasl100.csv")
    p.add_argument("--output-json", default="diagnostic/model_only_latency_wlasl100.json")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    batches = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    factories = [
        ("B6", b6_model),
        ("FELF-SLR", felf_model),
        ("SPOTER", spoter_model),
        ("ST-GCN-SL27", stgcn_model),
        ("Pose-TGCN", posetgcn_model),
        ("RGB-ResNet18-frame", rgb_framepool_model),
    ]
    rows = []
    for name, factory in factories:
        print(f"Preparing {name}", flush=True)
        model, maker = factory(args.num_classes, device)
        if model is None:
            continue
        params = count_params(model)
        for batch in batches:
            sample = maker(batch)
            stat = measure(lambda: call_model(name, model, sample), args.warmup, args.iters, device)
            row = {
                "model": name,
                "device": str(device),
                "batch_size": batch,
                "params": params,
                "mean_ms": stat["mean_ms"],
                "median_ms": stat["median_ms"],
                "p95_ms": stat["p95_ms"],
                "std_ms": stat["std_ms"],
                "throughput_samples_per_s": 1000.0 * batch / max(stat["mean_ms"], 1e-9),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    Path(args.output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
