from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from diagnose_local_global_failures import get_transform, load_model_for_checkpoint
from wlasl_train_b6_logit_heads_subset import FELFSLR
from wlasl_train_local_global_arcface import topk_metrics
from wlasl_train_local_global_arcface_subset import DATA_DIR, JSON_PATH
from wlasl_train_old_localglobal_sweep_subset import encode_feature_parts, load_subset_raw_for_source


def make_loader(left, right, global_features, labels, batch_size: int):
    ds = TensorDataset(
        torch.from_numpy(np.asarray(left, dtype=np.float32)),
        torch.from_numpy(np.asarray(right, dtype=np.float32)),
        torch.from_numpy(np.asarray(global_features, dtype=np.float32)),
        torch.from_numpy(np.asarray(labels, dtype=np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def metrics(logits: np.ndarray, labels: np.ndarray):
    probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=1).numpy()
    t1, t5 = topk_metrics(probs, labels)
    return float(t1), float(t5)


def collect(model, loader, device, transform):
    outs, labels = [], []
    model.eval()
    with torch.no_grad():
        for left, right, global_features, y in loader:
            left = left.to(device)
            right = right.to(device)
            global_features = global_features.to(device)
            left, right, global_features = transform(left, right, global_features)
            outs.append(model(left, right, global_features).float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(outs), np.concatenate(labels)


def load_old_model(path: str, dims, num_classes: int, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model, _ = load_model_for_checkpoint(Path(path), state, dims[0], dims[1], dims[2], num_classes, 16.0)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def felf_args():
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


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
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
    p.add_argument("--felf-checkpoint", default="rework_model/wlasl300_FELF_SLR_Staged_FreezeMain_swa.pth")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-prefix", default="diagnostic/wlasl300_felf_global_mask")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    splits, actions, _ = load_subset_raw_for_source(DATA_DIR, JSON_PATH, args.num_glosses, args.action_source, 0, labels_only=True)
    y_test = np.asarray(splits["test"]["labels"], dtype=np.int64)
    left, right, global_features = encode_feature_parts(splits["test"]["raw"], "test", args.cache_dir, "old", force=False)
    dims = (left.shape[-1], right.shape[-1], global_features.shape[-1])
    loader = make_loader(left, right, global_features, y_test, args.batch_size)
    num_classes = len(actions)

    b6 = load_old_model(args.b6_checkpoint, dims, num_classes, device)
    lrg = load_old_model(args.lrg_checkpoint, dims, num_classes, device)
    rf = load_old_model(args.rf_checkpoint, dims, num_classes, device)
    felf = FELFSLR(dims, num_classes, felf_args()).to(device)
    felf_state = torch.load(args.felf_checkpoint, map_location=device, weights_only=True)
    felf.load_state_dict(felf_state, strict=False)
    felf.eval()

    transforms = {"normal": get_transform("normal", 42), "mask_global": get_transform("mask_global", 42)}
    logits = {}
    labels = None
    for condition, transform in transforms.items():
        logits[condition] = {}
        for name, model in [("B6", b6), ("LRG", lrg), ("RF", rf), ("FELF", felf)]:
            logits[condition][name], labels = collect(model, loader, device, transform)
        logits[condition]["B6+LRG"] = logits[condition]["B6"] + logits[condition]["LRG"]
        logits[condition]["B6+RF"] = logits[condition]["B6"] + 0.5 * logits[condition]["RF"]
        logits[condition]["B6+LRG+RF"] = logits[condition]["B6"] + logits[condition]["LRG"] + 0.5 * logits[condition]["RF"]
        logits[condition]["B6+LRG+RF_norm"] = logits[condition]["B6+LRG+RF"] / 2.5

    metric_rows = []
    model_order = ["B6", "LRG", "RF", "B6+LRG", "B6+RF", "B6+LRG+RF", "B6+LRG+RF_norm", "FELF"]
    normal_metrics = {}
    for name in model_order:
        n1, n5 = metrics(logits["normal"][name], labels)
        m1, m5 = metrics(logits["mask_global"][name], labels)
        normal_metrics[name] = (n1, n5)
        metric_rows.append(
            {
                "model": name,
                "normal_top1": n1,
                "normal_top5": n5,
                "mask_global_top1": m1,
                "mask_global_top5": m5,
                "top1_drop": n1 - m1,
                "top5_drop": n5 - m5,
            }
        )

    pred = {name: logits["mask_global"][name].argmax(axis=1) for name in ["B6", "LRG", "RF", "FELF"]}
    correct = {name: pred[name] == labels for name in pred}
    all_wrong_same = (~correct["B6"] & ~correct["LRG"] & ~correct["RF"] & (pred["B6"] == pred["LRG"]) & (pred["B6"] == pred["RF"]))
    agreement = {
        "total": int(labels.shape[0]),
        "all_experts_wrong_same_class": int(all_wrong_same.sum()),
        "b6_correct_felf_wrong": int((correct["B6"] & ~correct["FELF"]).sum()),
        "felf_correct_b6_wrong": int((correct["FELF"] & ~correct["B6"]).sum()),
        "lrg_or_rf_correct_b6_wrong": int(((correct["LRG"] | correct["RF"]) & ~correct["B6"]).sum()),
        "fusion_confidence_higher_than_b6": int(
            (
                torch.softmax(torch.from_numpy(logits["mask_global"]["FELF"]), dim=1).max(dim=1).values.numpy()
                > torch.softmax(torch.from_numpy(logits["mask_global"]["B6"]), dim=1).max(dim=1).values.numpy()
            ).sum()
        ),
    }

    f_pred = pred["FELF"]
    idx = np.arange(labels.shape[0])
    contribution = {
        "lambda_lrg": 1.0,
        "lambda_rf": 0.5,
        "true_class_lrg_mean": float(logits["mask_global"]["LRG"][idx, labels].mean()),
        "true_class_rf_weighted_mean": float((0.5 * logits["mask_global"]["RF"][idx, labels]).mean()),
        "felf_pred_class_lrg_mean": float(logits["mask_global"]["LRG"][idx, f_pred].mean()),
        "felf_pred_class_rf_weighted_mean": float((0.5 * logits["mask_global"]["RF"][idx, f_pred]).mean()),
        "lrg_wrong_minus_true_mean": float((logits["mask_global"]["LRG"][idx, f_pred] - logits["mask_global"]["LRG"][idx, labels]).mean()),
        "rf_wrong_minus_true_weighted_mean": float((0.5 * (logits["mask_global"]["RF"][idx, f_pred] - logits["mask_global"]["RF"][idx, labels])).mean()),
    }

    oracle_logits_normal = logits["normal"]["FELF"]
    oracle_logits_mask = logits["mask_global"]["B6"]
    oracle = {
        "normal_uses": "FELF",
        "mask_global_uses": "B6",
        "normal_top1": metrics(oracle_logits_normal, labels)[0],
        "normal_top5": metrics(oracle_logits_normal, labels)[1],
        "mask_global_top1": metrics(oracle_logits_mask, labels)[0],
        "mask_global_top5": metrics(oracle_logits_mask, labels)[1],
    }

    prefix = Path(args.output_prefix)
    write_csv(prefix.with_name(prefix.name + "_metrics.csv"), metric_rows)
    summary = {"metrics": metric_rows, "agreement": agreement, "contribution": contribution, "oracle": oracle}
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("D1 metrics")
    for r in metric_rows:
        print(f"{r['model']}: normal {r['normal_top1']:.2f}/{r['normal_top5']:.2f} mask_global {r['mask_global_top1']:.2f}/{r['mask_global_top5']:.2f} drop {r['top1_drop']:.2f}")
    print("D2 agreement", agreement)
    print("D3 contribution", contribution)
    print("D4 oracle", oracle)
    print(f"Saved {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
