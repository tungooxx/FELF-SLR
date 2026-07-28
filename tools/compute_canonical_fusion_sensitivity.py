"""One-at-a-time sensitivity for the canonical Stage-2 fusion coefficients."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CANONICAL = {"baseline": 0.5, "felf": 1.0, "mt": 0.5}


def topk(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    top1 = np.mean(np.argmax(logits, axis=1) == labels) * 100.0
    idx = np.argpartition(logits, -5, axis=1)[:, -5:]
    top5 = np.mean(np.any(idx == labels[:, None], axis=1)) * 100.0
    return float(top1), float(top5)


def load_arrays(dataset: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    manifest_path = (
        ROOT
        / "diagnostic"
        / "tri_felf_mt_seed_stability"
        / f"{dataset}_seed1_tri_felf_mt_summary.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = manifest["inputs"]
    val = {
        "baseline": np.load(paths["baseline_val"]),
        "felf": np.load(paths["felf_val"]),
        "mt": np.load(paths["mt_val"]),
        "labels": np.load(paths["val_labels"]),
    }
    test = {
        "baseline": np.load(paths["baseline_test"]),
        "felf": np.load(paths["felf_test"]),
        "mt": np.load(paths["mt_test"]),
        "labels": np.load(paths["test_labels"]),
    }
    return val, test


def fuse(arrays: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights[name] * arrays[name] for name in ("baseline", "felf", "mt"))


def main() -> None:
    rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    for dataset in ("wlasl100", "wlasl300"):
        val, test = load_arrays(dataset)
        for varied in CANONICAL:
            for value in GRID:
                weights = dict(CANONICAL)
                weights[varied] = value
                val_top1, val_top5 = topk(fuse(val, weights), val["labels"])
                test_top1, test_top5 = topk(fuse(test, weights), test["labels"])
                rows.append(
                    {
                        "dataset": dataset,
                        "varied_coefficient": varied,
                        "value": value,
                        "baseline_weight": weights["baseline"],
                        "felf_weight": weights["felf"],
                        "mt_weight": weights["mt"],
                        "val_top1": val_top1,
                        "val_top5": val_top5,
                        "test_top1_diagnostic_only": test_top1,
                        "test_top5_diagnostic_only": test_top5,
                    }
                )
        for baseline, felf, mt in itertools.product(GRID, repeat=3):
            if baseline == felf == mt == 0:
                continue
            weights = {"baseline": baseline, "felf": felf, "mt": mt}
            val_top1, val_top5 = topk(fuse(val, weights), val["labels"])
            test_top1, test_top5 = topk(fuse(test, weights), test["labels"])
            grid_rows.append(
                {
                    "dataset": dataset,
                    "baseline_weight": baseline,
                    "felf_weight": felf,
                    "mt_weight": mt,
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "test_top1_diagnostic_only": test_top1,
                    "test_top5_diagnostic_only": test_top5,
                }
            )

    output_dir = ROOT / "diagnostic" / "paper_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "canonical_stage2_weight_sensitivity.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grid_path = output_dir / "canonical_stage2_full_validation_grid.csv"
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    print(output_path)
    print(grid_path)


if __name__ == "__main__":
    main()
