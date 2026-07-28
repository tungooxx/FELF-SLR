from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_logits(run_json: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(run_json.read_text(encoding="utf-8"))
    paths = data["logits"]
    val_logits = np.load(ROOT / paths["val_logits"])
    test_logits = np.load(ROOT / paths["test_logits"])
    val_labels = np.load(ROOT / paths["val_labels"])
    test_labels = np.load(ROOT / paths["test_labels"])
    return val_logits, test_logits, val_labels, test_labels


def acc(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(logits, axis=1)[:, ::-1]
    top1 = np.mean(order[:, 0] == labels) * 100.0
    top5 = np.mean(np.any(order[:, :5] == labels[:, None], axis=1)) * 100.0
    return float(top1), float(top5)


def norm_logits(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mean) / std


def main() -> None:
    out_dir = ROOT / "diagnostic"
    weights_lrg = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    weights_rf = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    rows: list[dict[str, object]] = []
    best_by_seed: list[dict[str, object]] = []

    for seed in [1, 2, 3]:
        b6v, b6t, yv, yt = load_logits(out_dir / f"wlasl100_seed{seed}_B6.json")
        lrgv, lrgt, yv2, yt2 = load_logits(out_dir / f"wlasl100_seed{seed}_LRG_fixed025.json")
        rfv, rft, yv3, yt3 = load_logits(out_dir / f"wlasl100_seed{seed}_RF_fixed025.json")
        assert np.array_equal(yv, yv2) and np.array_equal(yv, yv3)
        assert np.array_equal(yt, yt2) and np.array_equal(yt, yt3)

        variants = {
            "raw": (b6v, b6t, lrgv, lrgt, rfv, rft),
            "zscore": tuple(norm_logits(x) for x in (b6v, b6t, lrgv, lrgt, rfv, rft)),
        }
        seed_rows = []
        for norm_name, (bv, bt, lv, lt, rv, rt) in variants.items():
            for wl in weights_lrg:
                for wr in weights_rf:
                    val_logits = bv + wl * lv + wr * rv
                    test_logits = bt + wl * lt + wr * rt
                    val1, val5 = acc(val_logits, yv)
                    test1, test5 = acc(test_logits, yt)
                    row = {
                        "seed": seed,
                        "norm": norm_name,
                        "w_lrg": wl,
                        "w_rf": wr,
                        "val_top1": val1,
                        "val_top5": val5,
                        "test_top1": test1,
                        "test_top5": test5,
                    }
                    rows.append(row)
                    seed_rows.append(row)
        best = max(seed_rows, key=lambda r: (float(r["val_top1"]), float(r["val_top5"])))
        best_by_seed.append(best)

    # Global setting selected by mean val top1 + 0.25*top5 - 0.5*std top1 over seeds.
    grouped: dict[tuple[str, float, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["norm"]), float(row["w_lrg"]), float(row["w_rf"])), []).append(row)

    global_rows = []
    for (norm_name, wl, wr), rs in grouped.items():
        if len(rs) != 3:
            continue
        val1 = np.array([float(r["val_top1"]) for r in rs])
        val5 = np.array([float(r["val_top5"]) for r in rs])
        test1 = np.array([float(r["test_top1"]) for r in rs])
        test5 = np.array([float(r["test_top5"]) for r in rs])
        score = float(val1.mean() + 0.25 * val5.mean() - 0.5 * val1.std(ddof=1))
        global_rows.append(
            {
                "norm": norm_name,
                "w_lrg": wl,
                "w_rf": wr,
                "score": score,
                "val_top1_mean": float(val1.mean()),
                "val_top1_std": float(val1.std(ddof=1)),
                "val_top5_mean": float(val5.mean()),
                "val_top5_std": float(val5.std(ddof=1)),
                "test_top1_mean": float(test1.mean()),
                "test_top1_std": float(test1.std(ddof=1)),
                "test_top5_mean": float(test5.mean()),
                "test_top5_std": float(test5.std(ddof=1)),
            }
        )
    global_rows.sort(key=lambda r: float(r["score"]), reverse=True)

    csv_path = out_dir / "wlasl100_seedmatched_logit_combination_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    global_csv = out_dir / "wlasl100_seedmatched_logit_combination_global_summary.csv"
    with global_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(global_rows[0].keys()))
        writer.writeheader()
        writer.writerows(global_rows)

    summary = {
        "per_seed_best_by_val": best_by_seed,
        "global_best_by_val_objective": global_rows[0],
        "top10_global": global_rows[:10],
        "files": {"all_rows": str(csv_path), "global_summary": str(global_csv)},
    }
    json_path = out_dir / "wlasl100_seedmatched_logit_combination_sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
