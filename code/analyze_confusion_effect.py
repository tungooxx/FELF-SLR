"""Compare two WLASL diagnostic runs using saved logits.

Outputs:
- parent correct -> candidate wrong cases
- parent wrong -> candidate correct cases
- confusion-pair fixes/breaks/net
- top degraded/improved pairs
- true-label rank movement
- confidence/ECE summary
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ERROR_LABEL_CHOICES = [
    "handshape/morphology",
    "trajectory",
    "location",
    "left-right interaction",
    "global/body context",
    "noisy landmark",
    "unknown",
]


def load_diag(path: str | Path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    logits_path = Path(data["logits"]["test_logits"])
    labels_path = Path(data["logits"]["test_labels"])
    logits = np.load(logits_path)
    labels = np.load(labels_path)
    class_names = None
    class_path = data.get("class_names_path")
    if class_path and Path(class_path).exists():
        class_names = json.loads(Path(class_path).read_text(encoding="utf-8"))
    else:
        actions = data.get("actions")
        if actions:
            class_names = actions
    if class_names is None:
        class_names = [str(i) for i in range(logits.shape[1])]
    return data, logits, labels.astype(np.int64), class_names


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def true_rank(probs, labels):
    order = np.argsort(-probs, axis=1)
    ranks = np.empty(labels.shape[0], dtype=np.int64)
    for i, label in enumerate(labels):
        ranks[i] = int(np.where(order[i] == label)[0][0]) + 1
    return ranks


def ece(probs, labels, bins=15):
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == labels
    total = len(labels)
    value = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc = correct[mask].mean()
        c = conf[mask].mean()
        value += (mask.sum() / total) * abs(acc - c)
    return float(value * 100.0)


def calibration_summary(probs, labels):
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == labels
    return {
        "ece": ece(probs, labels),
        "mean_max_confidence": float(conf.mean()),
        "mean_confidence_correct": float(conf[correct].mean()) if np.any(correct) else None,
        "mean_confidence_wrong": float(conf[~correct].mean()) if np.any(~correct) else None,
        "top1": float(100.0 * correct.mean()),
        "top5": float(100.0 * np.mean([labels[i] in np.argsort(-probs[i])[:5] for i in range(len(labels))])),
    }


def label_name(class_names, idx):
    return class_names[idx] if 0 <= idx < len(class_names) else str(idx)


def write_rows(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze confusion effects between two diagnostic runs.")
    parser.add_argument("--parent", required=True, help="Parent/baseline diagnostic JSON.")
    parser.add_argument("--candidate", required=True, help="Candidate diagnostic JSON.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-pairs", type=int, default=20)
    args = parser.parse_args()

    parent_diag, parent_logits, labels, class_names = load_diag(args.parent)
    candidate_diag, cand_logits, cand_labels, cand_class_names = load_diag(args.candidate)
    if parent_logits.shape != cand_logits.shape:
        raise ValueError(f"Logit shape mismatch: parent={parent_logits.shape} candidate={cand_logits.shape}")
    if not np.array_equal(labels, cand_labels):
        raise ValueError("Parent and candidate labels are not identical.")
    if cand_class_names and len(cand_class_names) == len(class_names):
        class_names = cand_class_names

    parent_probs = softmax(parent_logits)
    cand_probs = softmax(cand_logits)
    parent_pred = parent_probs.argmax(axis=1)
    cand_pred = cand_probs.argmax(axis=1)
    parent_correct = parent_pred == labels
    cand_correct = cand_pred == labels
    parent_rank = true_rank(parent_probs, labels)
    cand_rank = true_rank(cand_probs, labels)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    broken_rows = []
    fixed_rows = []
    rank_rows = []
    for i, label in enumerate(labels):
        base = {
            "sample_index": i,
            "true_label": int(label),
            "true_class": label_name(class_names, int(label)),
            "parent_pred": int(parent_pred[i]),
            "parent_pred_class": label_name(class_names, int(parent_pred[i])),
            "candidate_pred": int(cand_pred[i]),
            "candidate_pred_class": label_name(class_names, int(cand_pred[i])),
            "rank_parent": int(parent_rank[i]),
            "rank_candidate": int(cand_rank[i]),
            "rank_delta": int(cand_rank[i] - parent_rank[i]),
            "parent_conf": float(parent_probs[i, parent_pred[i]]),
            "candidate_conf": float(cand_probs[i, cand_pred[i]]),
        }
        rank_rows.append(base)
        if parent_correct[i] and not cand_correct[i]:
            broken_rows.append(base)
        if (not parent_correct[i]) and cand_correct[i]:
            fixed_rows.append(base)

    common_fields = [
        "sample_index",
        "true_label",
        "true_class",
        "parent_pred",
        "parent_pred_class",
        "candidate_pred",
        "candidate_pred_class",
        "rank_parent",
        "rank_candidate",
        "rank_delta",
        "parent_conf",
        "candidate_conf",
    ]
    write_rows(out / "parent_correct_candidate_wrong.csv", broken_rows, common_fields)
    write_rows(out / "parent_wrong_candidate_correct.csv", fixed_rows, common_fields)
    write_rows(out / "rank_movement.csv", rank_rows, common_fields)

    fix_counter = Counter()
    break_counter = Counter()
    improved_examples = defaultdict(list)
    degraded_examples = defaultdict(list)
    for row in fixed_rows:
        key = (row["true_label"], row["parent_pred"])
        fix_counter[key] += 1
        if len(improved_examples[key]) < 5:
            improved_examples[key].append(row["sample_index"])
    for row in broken_rows:
        key = (row["true_label"], row["candidate_pred"])
        break_counter[key] += 1
        if len(degraded_examples[key]) < 5:
            degraded_examples[key].append(row["sample_index"])

    pair_keys = set(fix_counter) | set(break_counter)
    pair_rows = []
    for true_label, wrong_label in pair_keys:
        fixes = fix_counter[(true_label, wrong_label)]
        breaks = break_counter[(true_label, wrong_label)]
        pair_rows.append(
            {
                "source_label": int(true_label),
                "source_class": label_name(class_names, int(true_label)),
                "wrong_label": int(wrong_label),
                "wrong_class": label_name(class_names, int(wrong_label)),
                "fixes": int(fixes),
                "breaks": int(breaks),
                "net": int(fixes - breaks),
                "manual_error_type": "unknown",
                "manual_error_type_choices": " | ".join(ERROR_LABEL_CHOICES),
                "example_fixed_indices": " ".join(map(str, improved_examples[(true_label, wrong_label)])),
                "example_broken_indices": " ".join(map(str, degraded_examples[(true_label, wrong_label)])),
            }
        )
    pair_rows.sort(key=lambda r: (r["net"], -r["breaks"], r["fixes"]))
    pair_fields = [
        "source_label",
        "source_class",
        "wrong_label",
        "wrong_class",
        "fixes",
        "breaks",
        "net",
        "manual_error_type",
        "manual_error_type_choices",
        "example_fixed_indices",
        "example_broken_indices",
    ]
    write_rows(out / "confusion_pair_net.csv", pair_rows, pair_fields)
    write_rows(out / "top_degraded_pairs.csv", sorted(pair_rows, key=lambda r: (-r["breaks"], r["fixes"]))[: args.top_k_pairs], pair_fields)
    write_rows(out / "top_improved_pairs.csv", sorted(pair_rows, key=lambda r: (-r["fixes"], r["breaks"]))[: args.top_k_pairs], pair_fields)

    summary = {
        "parent": str(args.parent),
        "candidate": str(args.candidate),
        "parent_run": parent_diag.get("output_prefix") or Path(args.parent).stem,
        "candidate_run": candidate_diag.get("output_prefix") or Path(args.candidate).stem,
        "num_samples": int(len(labels)),
        "parent_correct_candidate_wrong": int(len(broken_rows)),
        "parent_wrong_candidate_correct": int(len(fixed_rows)),
        "net_corrections": int(len(fixed_rows) - len(broken_rows)),
        "parent_calibration": calibration_summary(parent_probs, labels),
        "candidate_calibration": calibration_summary(cand_probs, labels),
        "rank_movement": {
            "mean_rank_parent": float(parent_rank.mean()),
            "mean_rank_candidate": float(cand_rank.mean()),
            "mean_rank_delta": float((cand_rank - parent_rank).mean()),
            "true_rank_improved_count": int(np.sum(cand_rank < parent_rank)),
            "true_rank_worsened_count": int(np.sum(cand_rank > parent_rank)),
            "true_rank_same_count": int(np.sum(cand_rank == parent_rank)),
            "top5_lost_count": int(np.sum((parent_rank <= 5) & (cand_rank > 5))),
            "top5_gained_count": int(np.sum((parent_rank > 5) & (cand_rank <= 5))),
        },
        "files": {
            "parent_correct_candidate_wrong": str(out / "parent_correct_candidate_wrong.csv"),
            "parent_wrong_candidate_correct": str(out / "parent_wrong_candidate_correct.csv"),
            "confusion_pair_net": str(out / "confusion_pair_net.csv"),
            "top_degraded_pairs": str(out / "top_degraded_pairs.csv"),
            "top_improved_pairs": str(out / "top_improved_pairs.csv"),
            "rank_movement": str(out / "rank_movement.csv"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
