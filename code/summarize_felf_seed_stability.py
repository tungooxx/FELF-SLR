from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_metrics(path: Path) -> tuple[float, float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("swa_metrics", "fixed_swa_metrics", "selected_metrics", "pre_swa_metrics"):
        metrics = data.get(key)
        if isinstance(metrics, dict) and "test_top1" in metrics and "test_top5" in metrics:
            return float(metrics["test_top1"]), float(metrics["test_top5"])
    return None


def summarize(dataset: str, model: str, prefixes: list[str]) -> dict:
    values = []
    missing = []
    for prefix in prefixes:
        metrics = load_metrics(Path("diagnostic") / f"{prefix}.json")
        if metrics is None:
            missing.append(prefix)
        else:
            values.append(metrics)
    arr = np.asarray(values, dtype=np.float64) if values else np.empty((0, 2), dtype=np.float64)
    row = {
        "dataset": dataset,
        "model": model,
        "n": int(arr.shape[0]),
        "missing": missing,
    }
    if arr.shape[0]:
        row.update(
            {
                "top1_mean": float(arr[:, 0].mean()),
                "top1_std": float(arr[:, 0].std(ddof=1)) if arr.shape[0] > 1 else 0.0,
                "top5_mean": float(arr[:, 1].mean()),
                "top5_std": float(arr[:, 1].std(ddof=1)) if arr.shape[0] > 1 else 0.0,
                "values": values,
            }
        )
    return row


def main() -> None:
    seeds = [1, 2, 3]
    rows = []
    for num in (300, 100):
        rows.append(summarize(f"WLASL-{num}", "Baseline", [f"wlasl{num}_seed{s}_B6" for s in seeds]))
        rows.append(summarize(f"WLASL-{num}", "FELF-SLR staged", [f"wlasl{num}_seed{s}_FELF_SLR_staged" for s in seeds]))
    out = Path("diagnostic/felf_seed_stability_summary.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print()
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{Model} & \textbf{Top-1 Mean $\pm$ Std} & \textbf{Top-5 Mean $\pm$ Std} \\")
    print(r"\midrule")
    for row in rows:
        if row["n"] == 0:
            top1 = "TODO"
            top5 = "TODO"
        else:
            top1 = f"{row['top1_mean']:.2f}\\% $\\pm$ {row['top1_std']:.2f}"
            top5 = f"{row['top5_mean']:.2f}\\% $\\pm$ {row['top5_std']:.2f}"
        print(f"{row['dataset']} & {row['model']} & {top1} & {top5} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
