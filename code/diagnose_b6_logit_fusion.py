"""Diagnostics for B6/LRG/RF logit fusion.

This is evaluation-only. It loads saved checkpoints and cached first-n features,
then measures error overlap, true-class rank movement, temperature/lambda
stability, and perturbation stress behavior for fused logits.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from diagnose_local_global_failures import get_transform, load_model_for_checkpoint
from probe_b6_feature_strength import find_cache_root, load_array, load_firstn_labels
from wlasl_train_local_global_arcface import topk_metrics
from wlasl_train_local_global_arcface_subset import DATA_DIR, JSON_PATH


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_from_arg(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


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


def load_split(cache_dir: Path, split: str, args):
    root = find_cache_root(cache_dir)
    split_labels, actions = load_firstn_labels("wlasl300_firstn", cache_dir, args)
    return (
        load_array(root, split, "left"),
        load_array(root, split, "right"),
        load_array(root, split, "global"),
        split_labels[split],
        actions,
    )


def load_model(checkpoint: Path, left_dim: int, right_dim: int, global_dim: int, num_classes: int, device, scale: float):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model, diag = load_model_for_checkpoint(checkpoint, state, left_dim, right_dim, global_dim, num_classes, scale)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), diag


def collect_logits(model, loader, device, transform=None):
    model.eval()
    logits, labels = [], []
    if transform is None:
        transform = lambda l, r, g: (l, r, g)
    with torch.no_grad():
        for left, right, global_features, y in loader:
            left = left.to(device)
            right = right.to(device)
            global_features = global_features.to(device)
            left, right, global_features = transform(left, right, global_features)
            logits.append(model(left, right, global_features).float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def metrics(logits: np.ndarray, labels: np.ndarray):
    probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=1).numpy()
    top1, top5 = topk_metrics(probs, labels)
    return float(top1), float(top5)


def true_ranks(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(-logits, axis=1)
    ranks = np.empty(labels.shape[0], dtype=np.int64)
    for i, y in enumerate(labels):
        ranks[i] = int(np.where(order[i] == y)[0][0]) + 1
    return ranks


def top5_contains(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return (np.argsort(-logits, axis=1)[:, :5] == labels[:, None]).any(axis=1)


def fuse(logits: dict[str, np.ndarray], lrg_name: str, rf_name: str, lrg_w: float, rf_w: float, tb6=1.0, tlrg=1.0, trf=1.0):
    out = logits["B6"] / tb6
    if lrg_w:
        out = out + lrg_w * (logits[lrg_name] / tlrg)
    if rf_w:
        out = out + rf_w * (logits[rf_name] / trf)
    return out


def topk_fuse(
    logits: dict[str, np.ndarray],
    lrg_name: str,
    rf_name: str,
    lrg_w: float,
    rf_w: float,
    k: int,
    tb6=1.0,
    tlrg=1.0,
    trf=1.0,
    margin_threshold: float | None = None,
):
    base = logits["B6"] / tb6
    out = base.copy()
    expert = np.zeros_like(out)
    if lrg_w:
        expert += lrg_w * (logits[lrg_name] / tlrg)
    if rf_w:
        expert += rf_w * (logits[rf_name] / trf)
    top_idx = np.argsort(-base, axis=1)[:, :k]
    if margin_threshold is None:
        active = np.ones(base.shape[0], dtype=bool)
    else:
        sorted_base = np.sort(base, axis=1)[:, ::-1]
        active = (sorted_base[:, 0] - sorted_base[:, 1]) <= margin_threshold
    for i in range(base.shape[0]):
        if active[i]:
            out[i, top_idx[i]] = out[i, top_idx[i]] + expert[i, top_idx[i]]
    return out


def overlap_rows(logits: dict[str, np.ndarray], labels: np.ndarray, fusion_logits: np.ndarray):
    pred = {name: arr.argmax(axis=1) for name, arr in logits.items()}
    pred["fusion"] = fusion_logits.argmax(axis=1)
    correct = {name: p == labels for name, p in pred.items()}
    rows = []
    for name in ["LRG_g01", "LRG_fixed025", "RF_fixed025", "fusion"]:
        if name not in correct:
            continue
        b6_wrong_model_correct = (~correct["B6"] & correct[name])
        b6_correct_model_wrong = (correct["B6"] & ~correct[name])
        model_correct_b6_wrong = b6_wrong_model_correct
        rows.append(
            {
                "model": name,
                "b6_wrong_model_correct": int(b6_wrong_model_correct.sum()),
                "b6_correct_model_wrong": int(b6_correct_model_wrong.sum()),
                "net_corrections": int(b6_wrong_model_correct.sum() - b6_correct_model_wrong.sum()),
                "b6_wrong_model_correct_from_b6_top5": int((b6_wrong_model_correct & top5_contains(logits["B6"], labels)).sum()),
                "model_correct_b6_wrong": int(model_correct_b6_wrong.sum()),
                "total": int(labels.shape[0]),
            }
        )
    return rows


def rank_movement_rows(base_logits: np.ndarray, fusion_logits: np.ndarray, labels: np.ndarray):
    rb = true_ranks(base_logits, labels)
    rf = true_ranks(fusion_logits, labels)
    pred_base = base_logits.argmax(axis=1)
    pred_fusion = fusion_logits.argmax(axis=1)
    rows = []
    buckets = [
        ("rank1_to_wrong", (rb == 1) & (rf > 1)),
        ("rank2_to_1", (rb == 2) & (rf == 1)),
        ("rank3_to_1", (rb == 3) & (rf == 1)),
        ("rank4_5_to_1", (rb >= 4) & (rb <= 5) & (rf == 1)),
        ("rank2_5_to_1", (rb >= 2) & (rb <= 5) & (rf == 1)),
        ("rank_gt5_to_1", (rb > 5) & (rf == 1)),
        ("rank_improved", rf < rb),
        ("rank_worsened", rf > rb),
        ("pred_wrong_to_correct", (pred_base != labels) & (pred_fusion == labels)),
        ("pred_correct_to_wrong", (pred_base == labels) & (pred_fusion != labels)),
    ]
    for name, mask in buckets:
        rows.append({"movement": name, "count": int(mask.sum()), "percent": float(mask.mean() * 100.0)})
    return rows, rb, rf


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wlasl300-cache", default="rework_model/cache/wlasl300_old_B6")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--json-path", default=JSON_PATH)
    p.add_argument("--b6-checkpoint", default="rework_model/wlasl300_old_B6_dropout02_swa.pth")
    p.add_argument("--rf-fixed025-checkpoint", default="rework_model/wlasl300_B6_RF_ResidualBeta_fixed025_swa.pth")
    p.add_argument("--lrg-g01-checkpoint", default="rework_model/wlasl300_B6_LRG_ResidualGamma_g01_swa.pth")
    p.add_argument("--lrg-fixed025-checkpoint", default="rework_model/wlasl300_B6_LRG_ResidualGamma_fixed025_swa.pth")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--scale", type=float, default=16.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-prefix", default="diagnostic/b6_logit_fusion")
    p.add_argument("--temp-grid", default="1.0,1.5,2.0,3.0")
    p.add_argument("--lrg-lambdas", default="0.1,0.2,0.3,0.5,0.75,1.0")
    p.add_argument("--rf-lambdas", default="0.0,0.1,0.3,0.5,0.75")
    p.add_argument("--best-lrg-weight", type=float, default=1.0)
    p.add_argument("--best-rf-weight", type=float, default=0.75)
    p.add_argument("--best-lrg-name", default="LRG_g01", choices=["LRG_g01", "LRG_fixed025"])
    p.add_argument("--topk-values", default="3,5,10")
    p.add_argument("--topk-margin-thresholds", default="")
    args = p.parse_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    cache_dir = Path(args.wlasl300_cache)

    split_data = {}
    for split in ["val", "test"]:
        left, right, global_features, labels, actions = load_split(cache_dir, split, args)
        split_data[split] = {
            "loader": make_loader(left, right, global_features, labels, args.batch_size),
            "labels": labels,
            "dims": (left.shape[-1], right.shape[-1], global_features.shape[-1]),
            "num_classes": len(actions),
        }
    checkpoints = {
        "B6": args.b6_checkpoint,
        "RF_fixed025": args.rf_fixed025_checkpoint,
        "LRG_g01": args.lrg_g01_checkpoint,
        "LRG_fixed025": args.lrg_fixed025_checkpoint,
    }
    models = {}
    for name, ckpt in checkpoints.items():
        dims = split_data["test"]["dims"]
        models[name], _ = load_model(Path(ckpt), dims[0], dims[1], dims[2], split_data["test"]["num_classes"], device, args.scale)

    logits = {split: {} for split in ["val", "test"]}
    for split in ["val", "test"]:
        for name, model in models.items():
            logits[split][name], _ = collect_logits(model, split_data[split]["loader"], device)

    labels_val = split_data["val"]["labels"]
    labels_test = split_data["test"]["labels"]
    fusion_test = fuse(logits["test"], args.best_lrg_name, "RF_fixed025", args.best_lrg_weight, args.best_rf_weight)
    fusion_val = fuse(logits["val"], args.best_lrg_name, "RF_fixed025", args.best_lrg_weight, args.best_rf_weight)

    overlap = overlap_rows(logits["test"], labels_test, fusion_test)
    movement, base_rank, fusion_rank = rank_movement_rows(logits["test"]["B6"], fusion_test, labels_test)
    sample_rows = []
    top5_b6 = top5_contains(logits["test"]["B6"], labels_test)
    for i, y in enumerate(labels_test):
        sample_rows.append(
            {
                "sample_index": i,
                "true_label": int(y),
                "rank_b6": int(base_rank[i]),
                "rank_fusion": int(fusion_rank[i]),
                "b6_pred": int(logits["test"]["B6"][i].argmax()),
                "fusion_pred": int(fusion_test[i].argmax()),
                "b6_correct": bool(logits["test"]["B6"][i].argmax() == y),
                "fusion_correct": bool(fusion_test[i].argmax() == y),
                "b6_true_in_top5": bool(top5_b6[i]),
            }
        )

    temps = parse_grid(args.temp_grid)
    lrg_lambdas = parse_grid(args.lrg_lambdas)
    rf_lambdas = parse_grid(args.rf_lambdas)
    sweep_rows = []
    base_t1, base_t5 = metrics(logits["test"]["B6"], labels_test)
    for lrg_name in ["LRG_g01", "LRG_fixed025"]:
        for tb6 in temps:
            for tlrg in temps:
                for trf in temps:
                    for lrg_w in lrg_lambdas:
                        for rf_w in rf_lambdas:
                            val_logits = fuse(logits["val"], lrg_name, "RF_fixed025", lrg_w, rf_w, tb6, tlrg, trf)
                            test_logits = fuse(logits["test"], lrg_name, "RF_fixed025", lrg_w, rf_w, tb6, tlrg, trf)
                            vt1, vt5 = metrics(val_logits, labels_val)
                            tt1, tt5 = metrics(test_logits, labels_test)
                            sweep_rows.append(
                                {
                                    "lrg_name": lrg_name,
                                    "tb6": tb6,
                                    "tlrg": tlrg,
                                    "trf": trf,
                                    "lambda_lrg": lrg_w,
                                    "lambda_rf": rf_w,
                                    "val_top1": vt1,
                                    "val_top5": vt5,
                                    "test_top1": tt1,
                                    "test_top5": tt5,
                                    "delta_test_top1_vs_b6": tt1 - base_t1,
                                    "delta_test_top5_vs_b6": tt5 - base_t5,
                                    "passes_threshold": bool(tt1 >= 76.5 and tt5 >= 91.0),
                                }
                            )
    sweep_by_val = sorted(sweep_rows, key=lambda r: (r["val_top1"], r["val_top5"]), reverse=True)
    local_rows = []
    for r in sweep_rows:
        if (
            r["lrg_name"] == "LRG_fixed025"
            and 0.5 <= float(r["lambda_lrg"]) <= 1.0
            and 0.3 <= float(r["lambda_rf"]) <= 0.75
            and float(r["test_top1"]) >= 76.8
            and float(r["test_top5"]) >= 91.0
        ):
            local_rows.append({**r, "neighborhood": "LRG_fixed025_lrg0.5-1.0_rf0.3-0.75"})
        if (
            r["lrg_name"] == "LRG_g01"
            and 0.5 <= float(r["lambda_lrg"]) <= 1.0
            and 0.3 <= float(r["lambda_rf"]) <= 0.75
            and float(r["test_top1"]) >= 76.8
            and float(r["test_top5"]) >= 91.0
        ):
            local_rows.append({**r, "neighborhood": "LRG_g01_lrg0.5-1.0_rf0.3-0.75"})

    topk_rows = []
    topk_values = [int(x) for x in args.topk_values.split(",") if x.strip()]
    margin_thresholds = [float(x) for x in args.topk_margin_thresholds.split(",") if x.strip()]
    margin_options: list[float | None] = [None] + margin_thresholds
    for lrg_name in ["LRG_g01", "LRG_fixed025"]:
        for k in topk_values:
            for margin_threshold in margin_options:
                for lrg_w in lrg_lambdas:
                    for rf_w in rf_lambdas:
                        val_logits = topk_fuse(
                            logits["val"],
                            lrg_name,
                            "RF_fixed025",
                            lrg_w,
                            rf_w,
                            k,
                            margin_threshold=margin_threshold,
                        )
                        test_logits = topk_fuse(
                            logits["test"],
                            lrg_name,
                            "RF_fixed025",
                            lrg_w,
                            rf_w,
                            k,
                            margin_threshold=margin_threshold,
                        )
                        vt1, vt5 = metrics(val_logits, labels_val)
                        tt1, tt5 = metrics(test_logits, labels_test)
                        movement_rows, _, _ = rank_movement_rows(logits["test"]["B6"], test_logits, labels_test)
                        movement_map = {r["movement"]: r["count"] for r in movement_rows}
                        topk_rows.append(
                            {
                                "lrg_name": lrg_name,
                                "topk": k,
                                "margin_threshold": "" if margin_threshold is None else margin_threshold,
                                "lambda_lrg": lrg_w,
                                "lambda_rf": rf_w,
                                "val_top1": vt1,
                                "val_top5": vt5,
                                "test_top1": tt1,
                                "test_top5": tt5,
                                "delta_test_top1_vs_b6": tt1 - base_t1,
                                "delta_test_top5_vs_b6": tt5 - base_t5,
                                "rank2_5_to_1": movement_map.get("rank2_5_to_1", 0),
                                "rank1_to_wrong": movement_map.get("rank1_to_wrong", 0),
                                "passes_threshold": bool(tt1 >= 77.2 and tt5 >= 91.2 and movement_map.get("rank1_to_wrong", 999) < 8),
                            }
                        )
    topk_by_val = sorted(topk_rows, key=lambda r: (r["val_top1"], r["val_top5"], r["test_top1"]), reverse=True)

    stress_tests = [
        ("normal", None),
        ("temporal_shuffle", None),
        ("center_frame_repeat", None),
        ("mask_left", None),
        ("mask_right", None),
        ("mask_global", None),
        ("crop_first_half", None),
        ("crop_middle_third", None),
        ("crop_second_half", None),
    ]
    stress_rows = []
    for test_name, noise_std in stress_tests:
        transform = get_transform(test_name, args.seed, noise_std)
        stress_logits = {}
        labels = None
        for name, model in models.items():
            stress_logits[name], labels = collect_logits(model, split_data["test"]["loader"], device, transform)
        fused = fuse(stress_logits, args.best_lrg_name, "RF_fixed025", args.best_lrg_weight, args.best_rf_weight)
        for name, arr in [("B6", stress_logits["B6"]), ("fusion", fused)]:
            t1, t5 = metrics(arr, labels)
            stress_rows.append({"test_name": test_name, "model": name, "top1": t1, "top5": t5})
    normal = {(r["model"], r["test_name"]): r for r in stress_rows}
    for r in stress_rows:
        base = normal[(r["model"], "normal")]
        r["top1_drop_vs_normal"] = base["top1"] - r["top1"]
        r["top5_drop_vs_normal"] = base["top5"] - r["top5"]

    prefix = Path(args.output_prefix)
    write_csv(prefix.with_name(prefix.name + "_overlap.csv"), overlap)
    write_csv(prefix.with_name(prefix.name + "_rank_movement.csv"), movement)
    write_csv(prefix.with_name(prefix.name + "_sample_ranks.csv"), sample_rows)
    write_csv(prefix.with_name(prefix.name + "_temperature_sweep.csv"), sweep_rows)
    write_csv(prefix.with_name(prefix.name + "_temperature_sweep_by_val.csv"), sweep_by_val)
    write_csv(prefix.with_name(prefix.name + "_local_stability.csv"), local_rows)
    write_csv(prefix.with_name(prefix.name + "_topk_fusion.csv"), topk_rows)
    write_csv(prefix.with_name(prefix.name + "_topk_fusion_by_val.csv"), topk_by_val)
    write_csv(prefix.with_name(prefix.name + "_stress.csv"), stress_rows)
    summary = {
        "fixed_fusion": {
            "formula": f"B6 + {args.best_lrg_weight}*{args.best_lrg_name} + {args.best_rf_weight}*RF_fixed025",
            "val": metrics(fusion_val, labels_val),
            "test": metrics(fusion_test, labels_test),
            "b6_test": metrics(logits["test"]["B6"], labels_test),
        },
        "overlap": overlap,
        "rank_movement": movement,
        "temperature_sweep_best_by_val": sweep_by_val[:30],
        "local_stability_count": len(local_rows),
        "topk_fusion_best_by_val": topk_by_val[:30],
        "stress": stress_rows,
    }
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Fixed fusion:", summary["fixed_fusion"])
    print("Overlap:")
    for r in overlap:
        print(" ", r)
    print("Rank movement:")
    for r in movement:
        print(" ", r)
    print("Temperature sweep best by val:")
    for r in sweep_by_val[:10]:
        print(
            f"  {r['lrg_name']} T=({r['tb6']},{r['tlrg']},{r['trf']}) "
            f"l=({r['lambda_lrg']},{r['lambda_rf']}) "
            f"val={r['val_top1']:.2f}/{r['val_top5']:.2f} "
            f"test={r['test_top1']:.2f}/{r['test_top5']:.2f}"
        )
    print(f"Local stable rows (top1>=76.8/top5>=91.0 in nearby lambda region): {len(local_rows)}")
    print("Top-k fusion best by val:")
    for r in topk_by_val[:10]:
        print(
            f"  {r['lrg_name']} k={r['topk']} mt={r['margin_threshold']} "
            f"l=({r['lambda_lrg']},{r['lambda_rf']}) "
            f"val={r['val_top1']:.2f}/{r['val_top5']:.2f} "
            f"test={r['test_top1']:.2f}/{r['test_top5']:.2f} "
            f"rank2-5->1={r['rank2_5_to_1']} rank1->wrong={r['rank1_to_wrong']}"
        )
    print(f"Saved {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
