"""Compute submission metrics from the canonical Tri-FELF-MT logit manifests.

The final Tri-FELF-MT summaries are the protocol authority. Loading paths from
those summaries prevents historical branch-logit banks from being mixed with
the canonical Baseline/FELF/MT runs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostic" / "paper_metrics"


def load(path: str | Path) -> np.ndarray:
    return np.load(ROOT / path if not Path(path).is_absolute() else path)


def rank_of_true(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(-logits, axis=1)
    return np.argmax(order == labels[:, None], axis=1) + 1


def topk(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    return float((np.argsort(-logits, axis=1)[:, :k] == labels[:, None]).any(axis=1).mean() * 100.0)


def mrr(ranks: np.ndarray) -> float:
    return float(np.mean(1.0 / ranks))


def per_class_metrics(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    pred = logits.argmax(axis=1)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    values = []
    for cls in np.unique(labels):
        mask = labels == cls
        values.append((float((pred[mask] == cls).mean()), float((top5[mask] == cls).any(axis=1).mean())))
    return float(np.mean([v[0] for v in values]) * 100.0), float(np.mean([v[1] for v in values]) * 100.0)


def ece(logits: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    result = 0.0
    for lo, hi in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(0.0, 1.0, bins + 1)[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if mask.any():
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return result * 100.0


def nll(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def aurc(logits: np.ndarray, labels: np.ndarray) -> float:
    """Area under the selective-prediction risk-coverage curve."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    order = np.argsort(-probs.max(axis=1))
    errors = (probs.argmax(axis=1) != labels).astype(float)[order]
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(cumulative_risk.mean())


def movement(reference: np.ndarray, final: np.ndarray, labels: np.ndarray) -> dict:
    rr = rank_of_true(reference, labels)
    fr = rank_of_true(final, labels)
    ref_correct = rr == 1
    final_correct = fr == 1
    ref_top5 = rr <= 5
    final_top5 = fr <= 5
    ref_top5_sets = np.argsort(-reference, axis=1)[:, :5]
    final_top5_sets = np.argsort(-final, axis=1)[:, :5]
    overlap = np.array([
        len(set(a.tolist()).intersection(b.tolist())) / 5.0
        for a, b in zip(ref_top5_sets, final_top5_sets)
    ])
    corrected = (~ref_correct) & final_correct
    return {
        "wrong_to_correct": int((~ref_correct & final_correct).sum()),
        "correct_to_wrong": int((ref_correct & ~final_correct).sum()),
        "net_corrections": int((final_correct & ~ref_correct).sum() - (ref_correct & ~final_correct).sum()),
        "corrected_already_in_reference_top5": int((corrected & ref_top5).sum()),
        "corrected_total": int(corrected.sum()),
        "rank_2_to_1": int(((rr == 2) & (fr == 1)).sum()),
        "rank_3_to_1": int(((rr == 3) & (fr == 1)).sum()),
        "rank_4_or_5_to_1": int(((rr >= 4) & (rr <= 5) & (fr == 1)).sum()),
        "rank_gt5_to_1": int(((rr > 5) & (fr == 1)).sum()),
        "rank_improved": int((fr < rr).sum()),
        "rank_worsened": int((fr > rr).sum()),
        "rank_unchanged": int((fr == rr).sum()),
        "rank_1_to_wrong": int((ref_correct & ~final_correct).sum()),
        "reference_top5_candidate_retention_percent": float(overlap.mean() * 100.0),
        "overlap_at5_percent": float(overlap.mean() * 100.0),
        "true_class_drop_at5": int((ref_top5 & ~final_top5).sum()),
        "true_class_drop_at5_percent": float((ref_top5 & ~final_top5).mean() * 100.0),
    }


def evaluate(dataset: str, seed: int, labels: np.ndarray, systems: dict[str, np.ndarray]) -> list[dict]:
    rows: list[dict] = []
    for name, logits in systems.items():
        ranks = rank_of_true(logits, labels)
        per_class_top1, per_class_top5 = per_class_metrics(logits, labels)
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "system": name,
            "samples": int(len(labels)),
            "top1": topk(logits, labels, 1),
            "top5": topk(logits, labels, 5),
            "mrr": mrr(ranks),
            "mean_true_rank": float(ranks.mean()),
            "per_class_top1": per_class_top1,
            "per_class_top5": per_class_top5,
            "ece_15bin": ece(logits, labels),
            "nll": nll(logits, labels),
            "aurc": aurc(logits, labels),
        })
    return rows


def canonical_case(
    dataset: str,
    seed: int,
) -> tuple[list[dict], list[dict], dict[str, np.ndarray], np.ndarray, dict]:
    summary_path = (
        ROOT
        / "diagnostic"
        / "tri_felf_mt_seed_stability"
        / f"{dataset}_seed{seed}_tri_felf_mt_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    paths = summary["inputs"]
    weights = summary["weights"]
    labels = load(paths["test_labels"]).astype(int)
    val_labels = load(paths["val_labels"]).astype(int)
    baseline = load(paths["baseline_test"])
    felf = load(paths["felf_test"])
    mt = load(paths["mt_test"])
    baseline_val = load(paths["baseline_val"])
    felf_val = load(paths["felf_val"])
    mt_val = load(paths["mt_val"])

    expected_test_shape = (len(labels), int(dataset.removeprefix("wlasl")))
    expected_val_shape = (len(val_labels), int(dataset.removeprefix("wlasl")))
    for name, array in {
        "baseline_test": baseline,
        "felf_test": felf,
        "mt_test": mt,
    }.items():
        if array.shape != expected_test_shape:
            raise ValueError(f"{name} has shape {array.shape}; expected {expected_test_shape}")
    for name, array in {
        "baseline_val": baseline_val,
        "felf_val": felf_val,
        "mt_val": mt_val,
    }.items():
        if array.shape != expected_val_shape:
            raise ValueError(f"{name} has shape {array.shape}; expected {expected_val_shape}")

    tri = (
        weights["baseline"] * baseline
        + weights["felf"] * felf
        + weights["mt"] * mt
    )
    tri_val = (
        weights["baseline"] * baseline_val
        + weights["felf"] * felf_val
        + weights["mt"] * mt_val
    )
    if not np.isclose(topk(tri, labels, 1), summary["test_top1"], atol=1e-10):
        raise ValueError(f"{dataset} canonical test Top-1 does not reproduce its manifest")
    if not np.isclose(topk(tri_val, val_labels, 1), summary["val_top1"], atol=1e-10):
        raise ValueError(f"{dataset} canonical validation Top-1 does not reproduce its manifest")

    test = {
        "Baseline": baseline,
        "FELF": felf,
        "Tri-FELF-MT": tri,
    }
    rows = evaluate(dataset, seed, labels, test)
    movements = []
    for final_name in ("FELF", "Tri-FELF-MT"):
        for ref_name in ("Baseline", "FELF"):
            if ref_name == final_name or ref_name not in test:
                continue
            item = movement(test[ref_name], test[final_name], labels)
            item.update({"dataset": dataset, "seed": seed, "reference": ref_name, "final": final_name})
            movements.append(item)
    protocol = {
        "dataset": dataset,
        "seed": seed,
        "summary": str(summary_path.relative_to(ROOT)),
        "weights": weights,
        "inputs": paths,
        "test_samples": int(len(labels)),
        "val_samples": int(len(val_labels)),
    }
    return rows, movements, test, labels, protocol


def exact_mcnemar(reference_correct: np.ndarray, final_correct: np.ndarray) -> dict:
    reference_only = int((reference_correct & ~final_correct).sum())
    final_only = int((~reference_correct & final_correct).sum())
    discordant = reference_only + final_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(reference_only, final_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2.0**discordant))
    return {
        "reference_only_correct": reference_only,
        "final_only_correct": final_only,
        "discordant": discordant,
        "mcnemar_exact_p": p_value,
    }


def paired_bootstrap(
    reference: np.ndarray,
    final: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int = 20000,
    seed: int = 20260728,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(labels)
    reference_top1 = reference.argmax(axis=1) == labels
    final_top1 = final.argmax(axis=1) == labels
    reference_top5 = (np.argsort(-reference, axis=1)[:, :5] == labels[:, None]).any(axis=1)
    final_top5 = (np.argsort(-final, axis=1)[:, :5] == labels[:, None]).any(axis=1)
    top1_delta = np.empty(iterations, dtype=np.float64)
    top5_delta = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        count = min(1000, iterations - start)
        indices = rng.integers(0, n, size=(count, n))
        top1_delta[start : start + count] = (
            final_top1[indices].mean(axis=1) - reference_top1[indices].mean(axis=1)
        ) * 100.0
        top5_delta[start : start + count] = (
            final_top5[indices].mean(axis=1) - reference_top5[indices].mean(axis=1)
        ) * 100.0

    def summarize(delta: np.ndarray, observed: float) -> dict:
        non_positive = int((delta <= 0.0).sum())
        non_negative = int((delta >= 0.0).sum())
        p_value = min(
            1.0,
            2.0
            * min(
                (non_positive + 1) / (len(delta) + 1),
                (non_negative + 1) / (len(delta) + 1),
            ),
        )
        return {
            "observed_delta": observed,
            "ci95_low": float(np.percentile(delta, 2.5)),
            "ci95_high": float(np.percentile(delta, 97.5)),
            "paired_bootstrap_p": p_value,
        }

    result = {
        "iterations": iterations,
        "bootstrap_seed": seed,
        "top1": summarize(
            top1_delta,
            float((final_top1.mean() - reference_top1.mean()) * 100.0),
        ),
        "top5": summarize(
            top5_delta,
            float((final_top5.mean() - reference_top5.mean()) * 100.0),
        ),
    }
    result["mcnemar_top1"] = exact_mcnemar(reference_top1, final_top1)
    result["mcnemar_top5"] = exact_mcnemar(reference_top5, final_top5)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_movements: list[dict] = []
    significance_rows: list[dict] = []
    protocols: list[dict] = []
    for dataset in ("wlasl100", "wlasl300"):
        rows, movements, systems, labels, protocol = canonical_case(dataset, 1)
        all_rows.extend(rows)
        all_movements.extend(movements)
        protocols.append(protocol)
        for reference_name in ("Baseline", "FELF"):
            stats = paired_bootstrap(
                systems[reference_name],
                systems["Tri-FELF-MT"],
                labels,
                seed=20260728 + int(dataset.removeprefix("wlasl")),
            )
            significance_rows.append({
                "dataset": dataset,
                "seed": 1,
                "reference": reference_name,
                "final": "Tri-FELF-MT",
                "top1_delta": stats["top1"]["observed_delta"],
                "top1_ci95_low": stats["top1"]["ci95_low"],
                "top1_ci95_high": stats["top1"]["ci95_high"],
                "top1_bootstrap_p": stats["top1"]["paired_bootstrap_p"],
                "top1_mcnemar_p": stats["mcnemar_top1"]["mcnemar_exact_p"],
                "top1_reference_only_correct": stats["mcnemar_top1"]["reference_only_correct"],
                "top1_final_only_correct": stats["mcnemar_top1"]["final_only_correct"],
                "top5_delta": stats["top5"]["observed_delta"],
                "top5_ci95_low": stats["top5"]["ci95_low"],
                "top5_ci95_high": stats["top5"]["ci95_high"],
                "top5_bootstrap_p": stats["top5"]["paired_bootstrap_p"],
                "top5_mcnemar_p": stats["mcnemar_top5"]["mcnemar_exact_p"],
                "top5_reference_only_correct": stats["mcnemar_top5"]["reference_only_correct"],
                "top5_final_only_correct": stats["mcnemar_top5"]["final_only_correct"],
                "bootstrap_iterations": stats["iterations"],
            })
    with (OUT / "classification_rank_metrics_seed1.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (OUT / "rank_movement_metrics_seed1.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_movements[0]))
        writer.writeheader()
        writer.writerows(all_movements)
    with (OUT / "paired_significance_seed1.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(significance_rows[0]))
        writer.writeheader()
        writer.writerows(significance_rows)
    (OUT / "canonical_protocol_manifest.json").write_text(
        json.dumps(protocols, indent=2) + "\n",
        encoding="ascii",
    )
    stability_source = ROOT / "diagnostic" / "tri_felf_mt_seed_stability" / "tri_felf_mt_seed_stability.csv"
    stability_rows = list(csv.DictReader(stability_source.open()))
    stability_out = []
    for dataset in sorted({row["dataset"] for row in stability_rows}):
        rows = [row for row in stability_rows if row["dataset"] == dataset]
        top1 = np.array([float(row["top1"]) for row in rows])
        top5 = np.array([float(row["top5"]) for row in rows])
        stability_out.append({
            "dataset": dataset,
            "seeds": ",".join(row["seed"] for row in rows),
            "n_seeds": len(rows),
            "top1_mean": float(top1.mean()),
            "top1_std_sample": float(top1.std(ddof=1)) if len(top1) > 1 else 0.0,
            "top5_mean": float(top5.mean()),
            "top5_std_sample": float(top5.std(ddof=1)) if len(top5) > 1 else 0.0,
        })
    with (OUT / "tri_felf_mt_seed_stability.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stability_out[0]))
        writer.writeheader()
        writer.writerows(stability_out)
    report = [
        "# Final Paper Metrics",
        "",
        "Top-1/Top-5 are reported as per-instance accuracy; per-class values are macro averages across glosses.",
        "MRR and mean true rank use the complete test ranking.",
        "",
        "## Classification and Ranking",
        "",
        "| Dataset | System | Per-instance Top-1 | Per-instance Top-5 | Per-class Top-1 | Per-class Top-5 | MRR | ECE | NLL | AURC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        report.append(
            f"| {row['dataset']} | {row['system']} | {row['top1']:.2f}% | {row['top5']:.2f}% | "
            f"{row['per_class_top1']:.2f}% | {row['per_class_top5']:.2f}% | {row['mrr']:.3f} | "
            f"{row['ece_15bin']:.2f}% | {row['nll']:.3f} | {row['aurc']:.3f} |"
        )
    report.extend([
        "",
        "## Neighborhood Preservation",
        "",
        "`Overlap@5` is the mean fraction of the reference Top-5 candidates retained in the final Top-5.",
        "`TrueClassDrop@5` counts samples whose true class was in the reference Top-5 but not in the final Top-5.",
        "",
        "| Dataset | Reference | Final | Overlap@5 | TrueClassDrop@5 | Drop rate |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in all_movements:
        report.append(
            f"| {row['dataset']} | {row['reference']} | {row['final']} | {row['overlap_at5_percent']:.2f}% | "
            f"{row['true_class_drop_at5']} | {row['true_class_drop_at5_percent']:.2f}% |"
        )
    report.extend([
        "",
        "## Paired Significance",
        "",
        "Confidence intervals and p-values use 20,000 paired bootstrap resamples of test examples.",
        "McNemar p-values are exact two-sided tests on discordant predictions.",
        "",
        "| Dataset | Reference | Top-1 delta [95% CI] | Bootstrap p | McNemar p |",
        "|---|---|---:|---:|---:|",
    ])
    for row in significance_rows:
        report.append(
            f"| {row['dataset']} | {row['reference']} | {row['top1_delta']:.2f} "
            f"[{row['top1_ci95_low']:.2f}, {row['top1_ci95_high']:.2f}] | "
            f"{row['top1_bootstrap_p']:.4g} | {row['top1_mcnemar_p']:.4g} |"
        )
    report.extend([
        "",
        "## Seed Stability",
        "",
        "| Dataset | Seeds | Top-1 mean +- std | Top-5 mean +- std |",
        "|---|---|---:|---:|",
    ])
    for row in stability_out:
        report.append(
            f"| {row['dataset']} | {row['seeds']} | {row['top1_mean']:.2f} +- {row['top1_std_sample']:.2f}% | "
            f"{row['top5_mean']:.2f} +- {row['top5_std_sample']:.2f}% |"
        )
    (OUT / "final_paper_metrics.md").write_text("\n".join(report) + "\n", encoding="ascii")
    (OUT / "README.md").write_text(
        "# Paper metrics\n\n"
        "`classification_rank_metrics_seed1.csv` reports Top-1, Top-5, MRR, and mean true rank from saved logits.\n\n"
        "`rank_movement_metrics_seed1.csv` reports baseline/FELF/Tri rank movement and Top-5 retention diagnostics.\n\n"
        "`paired_significance_seed1.csv` reports paired bootstrap confidence intervals and exact McNemar tests.\n\n"
        "`canonical_protocol_manifest.json` records every source logit and label path used by the report.\n\n"
        "The manifest paths are taken from the canonical final-run summaries; historical branch-logit banks are intentionally excluded.\n",
        encoding="ascii",
    )
    print(
        f"Wrote {len(all_rows)} classification rows, {len(all_movements)} movement rows, "
        f"and {len(significance_rows)} significance rows to {OUT}"
    )


if __name__ == "__main__":
    main()
