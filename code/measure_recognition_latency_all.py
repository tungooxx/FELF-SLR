from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def pct(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    return float(ys[min(len(ys) - 1, max(0, int(round((len(ys) - 1) * q))))])


def summarize(times: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.mean(times)),
        "median_ms": float(statistics.median(times)),
        "p95_ms": pct(times, 0.95),
        "std_ms": float(statistics.pstdev(times)),
    }


def softmax_predict(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return probs.argmax(dim=-1)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def measure_loop(name: str, model: nn.Module, get_sample, call_model, n: int, warmup: int, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        for i in range(min(warmup, n)):
            sample = get_sample(i)
            logits = call_model(model, sample)
            _ = softmax_predict(logits)
        sync(device)

        times = []
        for i in range(n):
            t0 = time.perf_counter()
            sample = get_sample(i)
            logits = call_model(model, sample)
            _ = softmax_predict(logits)
            sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    row = summarize(times)
    row.update({"model": name, "params": count_params(model), "samples": n, "device": str(device)})
    return row


class _DecoderSelfAttnCompat:
    batch_first = False


def patch_spoter(model: nn.Module) -> None:
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
    if inputs.dim() == 3:
        inputs = inputs.unsqueeze(0)
    src = inputs.flatten(start_dim=2).transpose(0, 1).float()
    tgt = model.class_query.view(1, 1, -1).repeat(1, inputs.size(0), 1)
    h = model.transformer(model.pos + src, tgt).transpose(0, 1)
    return model.linear_class(h).squeeze(1)


def bench_b6(device: torch.device, n: int, warmup: int) -> dict:
    from wlasl_train_felf_slr_subset import LocalGlobalB6Expert

    cache = ROOT / "rework_model" / "cache" / "wlasl100_old_B6" / "frame_old"
    left = np.load(cache / "test_old_left.npy", mmap_mode="r")
    right = np.load(cache / "test_old_right.npy", mmap_mode="r")
    glob = np.load(cache / "test_old_global.npy", mmap_mode="r")
    n = min(n, len(left))
    model = LocalGlobalB6Expert(165, 165, 23, 100, dropout=0.2).to(device)

    def get(i):
        return (
            torch.from_numpy(np.asarray(left[i : i + 1])).to(device=device, dtype=torch.float32),
            torch.from_numpy(np.asarray(right[i : i + 1])).to(device=device, dtype=torch.float32),
            torch.from_numpy(np.asarray(glob[i : i + 1])).to(device=device, dtype=torch.float32),
        )

    def call(m, s):
        return m(*s)

    return measure_loop("B6", model, get, call, n, warmup, device)


def bench_felf(device: torch.device, n: int, warmup: int) -> dict:
    from argparse import Namespace
    from wlasl_train_felf_slr_subset import FELFSLR

    cache = ROOT / "rework_model" / "cache" / "wlasl100_old_B6" / "frame_old"
    left = np.load(cache / "test_old_left.npy", mmap_mode="r")
    right = np.load(cache / "test_old_right.npy", mmap_mode="r")
    glob = np.load(cache / "test_old_global.npy", mmap_mode="r")
    n = min(n, len(left))
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
    model = FELFSLR((165, 165, 23), 100, args).to(device)

    def get(i):
        return (
            torch.from_numpy(np.asarray(left[i : i + 1])).to(device=device, dtype=torch.float32),
            torch.from_numpy(np.asarray(right[i : i + 1])).to(device=device, dtype=torch.float32),
            torch.from_numpy(np.asarray(glob[i : i + 1])).to(device=device, dtype=torch.float32),
        )

    def call(m, s):
        return m(*s)

    return measure_loop("FELF-SLR", model, get, call, n, warmup, device)


def bench_spoter(device: torch.device, n: int, warmup: int) -> dict:
    sys.path.insert(0, str(ROOT / "external_baselines" / "spoter"))
    from datasets.czech_slr_dataset import CzechSLRDataset  # type: ignore
    from spoter.spoter_model import SPOTER  # type: ignore

    ckpt = torch.load(ROOT / "diagnostic" / "spoter_wlasl100_firstn_seed1.pth", map_location=device, weights_only=False)
    args = ckpt["args"]
    ds = CzechSLRDataset(args["test_csv"], augmentations=False, normalize=args["normalize"])
    n = min(n, len(ds))
    model = SPOTER(num_classes=args["num_classes"], hidden_dim=args["hidden_dim"]).to(device)
    patch_spoter(model)
    model.load_state_dict(ckpt["model"])

    def get(i):
        x, _y = ds[i]
        return x.to(device=device, dtype=torch.float32)

    def call(m, s):
        return spoter_forward_batched(m, s)

    return measure_loop("SPOTER", model, get, call, n, warmup, device)


def bench_stgcn(device: torch.device, n: int, warmup: int) -> dict:
    sys.path.insert(0, str(ROOT / "external" / "st-gcn-sl" / "st-gcn"))
    from feeder.feeder import Feeder  # type: ignore
    from net.st_gcn import Model  # type: ignore
    from train_stgcn_sl_wlasl import CUSTOM_27_EDGE

    data_dir = ROOT / "rework_model" / "cache" / "stgcn_sl27_wlasl100_firstn"
    ds = Feeder(
        data_path=str(data_dir / "test_data.npy"),
        label_path=str(data_dir / "test_label.pkl"),
        random_choose=False,
        random_move=False,
        window_size=60,
        mmap=True,
    )
    n = min(n, len(ds))
    model = Model(
        in_channels=3,
        num_class=100,
        edge_importance_weighting=True,
        graph_args={
            "layout": "custom",
            "strategy": "spatial",
            "custom_layout": {"num_node": 27, "center": 0, "edge": CUSTOM_27_EDGE},
        },
    ).to(device)
    ckpt = torch.load(ROOT / "diagnostic" / "stgcn_sl27_wlasl100_seed1_gpu.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    def get(i):
        x, _y = ds[i]
        return torch.from_numpy(np.asarray(x)[None]).to(device=device, dtype=torch.float32)

    def call(m, s):
        return m(s)

    return measure_loop("ST-GCN-SL27", model, get, call, n, warmup, device)


def bench_posetgcn(device: torch.device, n: int, warmup: int) -> dict:
    sys.path.insert(0, str(ROOT / "external" / "WLASL" / "code" / "TGCN"))
    from configs import Config  # type: ignore
    from sign_dataset import Sign_Dataset  # type: ignore
    from tgcn_model import GCN_muti_att  # type: ignore

    cfg = Config(str(ROOT / "external" / "WLASL" / "code" / "TGCN" / "configs" / "asl100.ini"))
    export_root = ROOT / "rework_model" / "cache" / "posetgcn_wlasl100_firstn"
    ds = Sign_Dataset(
        index_file_path=str(export_root / "splits" / "asl100_firstn.json"),
        split="test",
        pose_root=str(export_root / "pose_per_individual_videos"),
        img_transforms=None,
        video_transforms=None,
        num_samples=cfg.num_samples,
        sample_strategy="k_copies",
        num_copies=4,
    )
    n = min(n, len(ds))
    model = GCN_muti_att(
        input_feature=cfg.num_samples * 2,
        hidden_feature=cfg.num_samples * 2,
        num_class=100,
        p_dropout=cfg.drop_p,
        num_stage=cfg.num_stages,
    ).to(device)
    state = torch.load(ROOT / "diagnostic" / "posetgcn_wlasl100_seed1_gpu_b32_best.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)

    def get(i):
        x, _y, _vid = ds[i]
        return x.unsqueeze(0).to(device=device, dtype=torch.float32)

    def call(m, s):
        if s.size(2) > m.gc1.in_features and s.size(2) % 4 == 0:
            stride = s.size(2) // 4
            return torch.stack([m(s[:, :, j * stride : (j + 1) * stride]) for j in range(4)], dim=1).mean(dim=1)
        return m(s)

    return measure_loop("Pose-TGCN", model, get, call, n, warmup, device)


def bench_rgb_resnet(device: torch.device, n: int, warmup: int) -> dict:
    import torchvision.models as models

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)
    n = min(n, 100)

    def get(i):
        # Represents already decoded/resized RGB frame tensor.
        return torch.randn(1, 3, 224, 224, device=device)

    def call(m, s):
        return m(s)

    return measure_loop("RGB-ResNet18-frame", model, get, call, n, warmup, device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--models", default="B6,FELF-SLR,SPOTER,ST-GCN-SL27,Pose-TGCN,RGB-ResNet18-frame")
    p.add_argument("--output-csv", default="diagnostic/recognition_latency_wlasl100_gpu.csv")
    p.add_argument("--output-json", default="diagnostic/recognition_latency_wlasl100_gpu.json")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    runners = {
        "B6": bench_b6,
        "FELF-SLR": bench_felf,
        "SPOTER": bench_spoter,
        "ST-GCN-SL27": bench_stgcn,
        "Pose-TGCN": bench_posetgcn,
        "RGB-ResNet18-frame": bench_rgb_resnet,
    }
    rows = []
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        print(f"Measuring {name}", flush=True)
        row = runners[name](device, args.samples, args.warmup)
        rows.append(row)
        print(json.dumps(row), flush=True)
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
