from __future__ import annotations

import csv
import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from wlasl_train_felf_slr_subset import FELFSLR, LocalGlobalB6Expert


ROOT = Path(__file__).resolve().parent


def topk(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(logits, axis=1)[:, ::-1]
    return float(np.any(order[:, :k] == labels[:, None], axis=1).mean() * 100.0)


def per_class_topk(logits: np.ndarray, labels: np.ndarray, num_classes: int, k: int) -> float:
    order = np.argsort(logits, axis=1)[:, ::-1]
    vals = []
    for cls in range(num_classes):
        idx = np.where(labels == cls)[0]
        if idx.size == 0:
            continue
        vals.append(float(np.any(order[idx, :k] == cls, axis=1).mean() * 100.0))
    return float(np.mean(vals)) if vals else 0.0


def class_rows(logits: np.ndarray, labels: np.ndarray, actions: list[str], model_name: str) -> list[dict[str, object]]:
    order = np.argsort(logits, axis=1)[:, ::-1]
    rows = []
    for cls, gloss in enumerate(actions):
        idx = np.where(labels == cls)[0]
        if idx.size == 0:
            continue
        top1 = float((order[idx, 0] == cls).mean() * 100.0)
        top5 = float(np.any(order[idx, :5] == cls, axis=1).mean() * 100.0)
        rows.append({"model": model_name, "class_id": cls, "gloss": gloss, "samples": int(idx.size), "top1": top1, "top5": top5})
    return rows


def load_logits_from_json(path: Path) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    logits_path = ROOT / data["logits"]["test_logits"]
    labels_path = data["logits"].get("test_labels")
    labels = np.load(ROOT / labels_path) if labels_path else None
    return np.load(logits_path), labels, data["actions"] if "actions" in data else []


def drop_frames(x: torch.Tensor, rate: float, seed: int) -> torch.Tensor:
    if rate <= 0:
        return x
    y = x.clone()
    rng = random.Random(seed)
    batch, frames = y.shape[:2]
    n_drop = int(round(frames * rate))
    for i in range(batch):
        idx = rng.sample(range(frames), min(n_drop, frames))
        y[i, idx] = 0.0
    return y


def felf_args() -> Namespace:
    return Namespace(
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


def load_state(model: torch.nn.Module, ckpt_path: Path, strict: bool = True) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=strict)


@torch.no_grad()
def eval_model(model: torch.nn.Module, left: np.ndarray, right: np.ndarray, glob: np.ndarray, labels: np.ndarray, device: torch.device, rate: float, seed: int) -> dict[str, float]:
    model.eval()
    logits_all = []
    for start in range(0, len(labels), 64):
        l = torch.from_numpy(np.asarray(left[start : start + 64])).to(device=device, dtype=torch.float32)
        r = torch.from_numpy(np.asarray(right[start : start + 64])).to(device=device, dtype=torch.float32)
        g = torch.from_numpy(np.asarray(glob[start : start + 64])).to(device=device, dtype=torch.float32)
        l = drop_frames(l, rate, seed + start)
        r = drop_frames(r, rate, seed + 10000 + start)
        g = drop_frames(g, rate, seed + 20000 + start)
        out = model(l, r, g)
        if isinstance(out, dict):
            out = out["fused"]
        logits_all.append(out.detach().cpu().numpy())
    logits = np.concatenate(logits_all, axis=0)
    return {"top1": topk(logits, labels, 1), "top5": topk(logits, labels, 5), "per_class_top1": per_class_topk(logits, labels, 100, 1), "per_class_top5": per_class_topk(logits, labels, 100, 5)}


def make_dropout_rows(model_name: str, model: torch.nn.Module, left, right, glob, labels, device: torch.device) -> list[dict[str, object]]:
    rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    clean = eval_model(model, left, right, glob, labels, device, 0.0, 1234)
    rows = []
    for rate in rates:
        vals = [clean] if rate == 0 else [eval_model(model, left, right, glob, labels, device, rate, 2000 + i) for i in range(5)]
        top1s = np.array([v["top1"] for v in vals])
        top5s = np.array([v["top5"] for v in vals])
        pc1s = np.array([v["per_class_top1"] for v in vals])
        pc5s = np.array([v["per_class_top5"] for v in vals])
        rows.append(
            {
                "model": model_name,
                "condition": "Clean" if rate == 0 else f"Random-{int(rate * 100)}",
                "drop_rate": rate,
                "top1": float(top1s.mean()),
                "top1_std": float(top1s.std()),
                "top5": float(top5s.mean()),
                "top5_std": float(top5s.std()),
                "per_class_top1": float(pc1s.mean()),
                "per_class_top1_std": float(pc1s.std()),
                "per_class_top5": float(pc5s.mean()),
                "per_class_top5_std": float(pc5s.std()),
                "delta_top1": clean["top1"] - float(top1s.mean()),
                "delta_top5": clean["top5"] - float(top5s.mean()),
            }
        )
    return rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ROOT / "rework_model" / "cache" / "wlasl100_old_B6" / "frame_old"
    left = np.load(cache / "test_old_left.npy", mmap_mode="r")
    right = np.load(cache / "test_old_right.npy", mmap_mode="r")
    glob = np.load(cache / "test_old_global.npy", mmap_mode="r")

    b6_logits, labels, actions = load_logits_from_json(ROOT / "diagnostic" / "wlasl100_seed3_B6.json")
    felf_logits, labels2, _ = load_logits_from_json(ROOT / "diagnostic" / "wlasl100_seed3_FELF_SLR_staged_lrg10_rf075.json")
    if labels2 is not None and not np.array_equal(labels, labels2):
        raise RuntimeError("Label mismatch between B6 and FELF logits.")

    summary = [
        {
            "model": "B6 seed3 selected logits",
            "top1": topk(b6_logits, labels, 1),
            "top5": topk(b6_logits, labels, 5),
            "per_class_top1": per_class_topk(b6_logits, labels, 100, 1),
            "per_class_top5": per_class_topk(b6_logits, labels, 100, 5),
        },
        {
            "model": "FELF seed3 LRG1 RF0.75 selected logits",
            "top1": topk(felf_logits, labels, 1),
            "top5": topk(felf_logits, labels, 5),
            "per_class_top1": per_class_topk(felf_logits, labels, 100, 1),
            "per_class_top5": per_class_topk(felf_logits, labels, 100, 5),
        },
    ]

    per_class = class_rows(b6_logits, labels, actions, "B6") + class_rows(felf_logits, labels, actions, "FELF_LRG1_RF075")

    b6 = LocalGlobalB6Expert(165, 165, 23, 100, dropout=0.2).to(device)
    load_state(b6, ROOT / "rework_model" / "wlasl100_seed3_B6_swa.pth")
    felf = FELFSLR((165, 165, 23), 100, felf_args()).to(device)
    load_state(felf, ROOT / "rework_model" / "wlasl100_seed3_FELF_SLR_staged_lrg10_rf075_swa.pth", strict=False)
    dropout = make_dropout_rows("B6_seed3_swa", b6, left, right, glob, labels, device)
    dropout += make_dropout_rows("FELF_seed3_lrg10_rf075_swa", felf, left, right, glob, labels, device)

    out_dir = ROOT / "diagnostic"
    (out_dir / "wlasl100_seed3_perclass_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "wlasl100_seed3_perclass_by_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class[0].keys()))
        writer.writeheader()
        writer.writerows(per_class)
    with (out_dir / "wlasl100_seed3_frame_dropout.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dropout[0].keys()))
        writer.writeheader()
        writer.writerows(dropout)
    (out_dir / "wlasl100_seed3_frame_dropout.json").write_text(json.dumps(dropout, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "dropout": dropout}, indent=2))


if __name__ == "__main__":
    main()
