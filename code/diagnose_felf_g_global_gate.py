from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from diagnose_local_global_failures import load_model_for_checkpoint
from wlasl_train_local_global_arcface import topk_metrics
from wlasl_train_local_global_arcface_subset import DATA_DIR, JSON_PATH
from wlasl_train_old_localglobal_sweep_subset import encode_feature_parts, load_subset_raw_for_source


def make_loader(left, right, global_features, labels, batch_size: int):
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


def load_model(path: str, dims, num_classes: int, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model, _ = load_model_for_checkpoint(Path(path), state, dims[0], dims[1], dims[2], num_classes, 16.0)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def collect_logits(models: dict, loader, device, global_scale: float):
    logits = {name: [] for name in models}
    labels = []
    norms = []
    with torch.no_grad():
        for left, right, global_features, y in loader:
            left = left.to(device)
            right = right.to(device)
            global_features = global_features.to(device) * float(global_scale)
            norms.append(global_features.abs().mean(dim=(1, 2)).detach().cpu().numpy())
            for name, model in models.items():
                logits[name].append(model(left, right, global_features).float().cpu().numpy())
            labels.append(y.numpy())
    return {k: np.concatenate(v) for k, v in logits.items()}, np.concatenate(labels), np.concatenate(norms)


def metrics(logits: np.ndarray, labels: np.ndarray):
    probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=1).numpy()
    t1, t5 = topk_metrics(probs, labels)
    return float(t1), float(t5)


def soft_gate(norm: np.ndarray, a: float, b: float):
    if b <= a:
        return (norm > a).astype(np.float32)
    return np.clip((norm - a) / (b - a), 0.0, 1.0).astype(np.float32)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--num-glosses", type=int, default=300)
    p.add_argument("--cache-dir", default="rework_model/cache/wlasl300_old_B6")
    p.add_argument("--action-source", default="json_first_n", choices=["json_first_n", "top_frequency", "train_dirs"])
    p.add_argument("--b6-checkpoint", default="rework_model/wlasl300_old_B6_dropout02_swa.pth")
    p.add_argument("--lrg-checkpoint", default="rework_model/wlasl300_B6_LRG_ResidualGamma_fixed025_swa.pth")
    p.add_argument("--rf-checkpoint", default="rework_model/wlasl300_B6_RF_ResidualBeta_fixed025_swa.pth")
    p.add_argument("--lambda-lrg", type=float, default=1.0)
    p.add_argument("--lambda-rf", type=float, default=0.5)
    p.add_argument("--normalizer", type=float, default=2.5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-prefix", default="diagnostic/wlasl300_felf_g_global_gate")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    splits, actions, _ = load_subset_raw_for_source(DATA_DIR, JSON_PATH, args.num_glosses, args.action_source, 0, labels_only=True)
    labels = np.asarray(splits["test"]["labels"], dtype=np.int64)
    left, right, global_features = encode_feature_parts(splits["test"]["raw"], "test", args.cache_dir, "old", force=False)
    dims = (left.shape[-1], right.shape[-1], global_features.shape[-1])
    loader = make_loader(left, right, global_features, labels, args.batch_size)
    num_classes = len(actions)
    models = {
        "B6": load_model(args.b6_checkpoint, dims, num_classes, device),
        "LRG": load_model(args.lrg_checkpoint, dims, num_classes, device),
        "RF": load_model(args.rf_checkpoint, dims, num_classes, device),
    }

    normal_logits, _, normal_norm = collect_logits(models, loader, device, 1.0)
    zero_logits, _, zero_norm = collect_logits(models, loader, device, 0.0)
    tau = max(float(zero_norm.max()) + 1e-8, 1e-8)
    normal_norm_q10 = float(np.quantile(normal_norm, 0.10))
    normal_norm_q50 = float(np.quantile(normal_norm, 0.50))

    rows = []
    scales = [1.0, 0.75, 0.5, 0.25, 0.0]
    for scale in scales:
        logit, y, norm = collect_logits(models, loader, device, scale)
        b6 = logit["B6"]
        lrg = logit["LRG"]
        rf = logit["RF"]
        variants = {
            "B6": b6,
            "FELF": (b6 + args.lambda_lrg * lrg + args.lambda_rf * rf) / args.normalizer,
            "B6+RF": (b6 + args.lambda_rf * rf) / (1.0 + abs(args.lambda_rf)),
        }
        hard = (norm > tau).astype(np.float32)[:, None]
        soft = soft_gate(norm, tau, normal_norm_q10)[:, None]
        soft_mid = soft_gate(norm, tau, normal_norm_q50)[:, None]
        variants["FELF-G-hard"] = (b6 + hard * args.lambda_lrg * lrg + args.lambda_rf * rf) / (
            1.0 + hard * abs(args.lambda_lrg) + abs(args.lambda_rf)
        )
        variants["FELF-G-soft-q10"] = (b6 + soft * args.lambda_lrg * lrg + args.lambda_rf * rf) / (
            1.0 + soft * abs(args.lambda_lrg) + abs(args.lambda_rf)
        )
        variants["FELF-G-soft-q50"] = (b6 + soft_mid * args.lambda_lrg * lrg + args.lambda_rf * rf) / (
            1.0 + soft_mid * abs(args.lambda_lrg) + abs(args.lambda_rf)
        )
        for name, arr in variants.items():
            top1, top5 = metrics(arr, y)
            rows.append(
                {
                    "global_scale": scale,
                    "model": name,
                    "top1": top1,
                    "top5": top5,
                    "gate_mean": "" if not name.startswith("FELF-G") else float(
                        {
                            "FELF-G-hard": hard,
                            "FELF-G-soft-q10": soft,
                            "FELF-G-soft-q50": soft_mid,
                        }[name].mean()
                    ),
                }
            )

    # G1 oracle: normal full FELF, mask-global B6+RF.
    oracle = {
        "normal_model": "FELF",
        "mask_global_model": "B6+RF",
        "normal": next(r for r in rows if r["global_scale"] == 1.0 and r["model"] == "FELF"),
        "mask_global": next(r for r in rows if r["global_scale"] == 0.0 and r["model"] == "B6+RF"),
    }
    summary = {
        "tau": tau,
        "normal_global_norm_q10": normal_norm_q10,
        "normal_global_norm_q50": normal_norm_q50,
        "oracle": oracle,
        "rows": rows,
    }
    prefix = Path(args.output_prefix)
    write_csv(prefix.with_suffix(".csv"), rows)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("G1 oracle:", oracle)
    print("Rows:")
    for r in rows:
        print(f"scale={r['global_scale']} {r['model']}: {r['top1']:.2f}/{r['top5']:.2f} gate={r['gate_mean']}")
    print(f"Saved {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
