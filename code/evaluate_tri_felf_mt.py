"""Evaluate the final Tri-FELF-MT logit fusion.

Tri-FELF-MT is a post-hoc class-logit fusion:

    z = a * z_Baseline + b * z_FELF + c * z_MT

The default paper setting is:

    a = 0.5, b = 1.0, c = 0.5

Inputs may be diagnostic JSON files produced by the training scripts or direct
NumPy logit paths. The script intentionally keeps fusion at the logit level so
it does not modify the trained Baseline, FELF, or MorphTraj checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _resolve_path(raw: str | None, base: Path | None = None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    candidates = []
    if base is not None:
        candidates.append(base / path)
        candidates.append(base.parent / path)
    candidates.append(Path.cwd() / path)
    candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _read_json(path: str | None) -> tuple[dict, Path] | tuple[None, None]:
    if not path:
        return None, None
    p = _resolve_path(path)
    if p is None or not p.exists():
        raise FileNotFoundError(f"Diagnostic JSON not found: {path}")
    return json.loads(p.read_text(encoding="utf-8-sig")), p.parent


def _diag_logits(diag: dict | None, split: str, base: Path | None) -> Path | None:
    if not diag:
        return None
    logits = diag.get("logits") or {}
    key = f"{split}_logits"
    if key in logits:
        return _resolve_path(logits[key], base)
    # MorphTraj diagnostics also expose factor logits. The final architecture
    # uses the fused MorphTraj expert unless another path is provided directly.
    factors = diag.get("factor_logits") or {}
    if "fused" in factors and split in factors["fused"]:
        return _resolve_path(factors["fused"][split], base)
    return None


def _diag_labels(diag: dict | None, split: str, base: Path | None) -> Path | None:
    if not diag:
        return None
    logits = diag.get("logits") or {}
    key = f"{split}_labels"
    if key in logits:
        return _resolve_path(logits[key], base)
    return None


def _load_array(path: Path | str | None, name: str) -> np.ndarray:
    if path is None:
        raise ValueError(f"Missing path for {name}")
    p = _resolve_path(str(path))
    if p is None or not p.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return np.load(p)


def _topk(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    labels = labels.astype(np.int64)
    top1 = (logits.argmax(axis=1) == labels).mean() * 100.0
    k = min(5, logits.shape[1])
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    top5 = np.any(topk == labels[:, None], axis=1).mean() * 100.0
    return float(top1), float(top5)


def _movement(reference: np.ndarray, fused: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    ref_pred = reference.argmax(axis=1)
    fused_pred = fused.argmax(axis=1)
    labels = labels.astype(np.int64)
    ref_correct = ref_pred == labels
    fused_correct = fused_pred == labels
    return {
        "reference_wrong_to_fused_correct": int((~ref_correct & fused_correct).sum()),
        "reference_correct_to_fused_wrong": int((ref_correct & ~fused_correct).sum()),
        "net_corrections": int(fused_correct.sum() - ref_correct.sum()),
    }


def _rank_of_true(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(-logits, axis=1)
    ranks = np.empty(len(labels), dtype=np.int64)
    for i, y in enumerate(labels.astype(np.int64)):
        ranks[i] = int(np.where(order[i] == y)[0][0]) + 1
    return ranks


def _rank_movement(reference: np.ndarray, fused: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    r0 = _rank_of_true(reference, labels)
    r1 = _rank_of_true(fused, labels)
    became_rank1 = r1 == 1
    return {
        "rank_2_to_1": int(((r0 == 2) & became_rank1).sum()),
        "rank_3_to_1": int(((r0 == 3) & became_rank1).sum()),
        "rank_4_or_5_to_1": int((((r0 == 4) | (r0 == 5)) & became_rank1).sum()),
        "rank_gt5_to_1": int(((r0 > 5) & became_rank1).sum()),
        "rank_improved": int((r1 < r0).sum()),
        "rank_worsened": int((r1 > r0).sum()),
        "rank_unchanged": int((r1 == r0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tri-FELF-MT logit fusion.")
    parser.add_argument("--baseline-diagnostic")
    parser.add_argument("--felf-diagnostic")
    parser.add_argument("--mt-diagnostic")
    parser.add_argument("--baseline-val-logits")
    parser.add_argument("--baseline-test-logits")
    parser.add_argument("--felf-val-logits")
    parser.add_argument("--felf-test-logits")
    parser.add_argument("--mt-val-logits")
    parser.add_argument("--mt-test-logits")
    parser.add_argument("--val-labels")
    parser.add_argument("--test-labels")
    parser.add_argument("--baseline-weight", type=float, default=0.5)
    parser.add_argument("--felf-weight", type=float, default=1.0)
    parser.add_argument("--mt-weight", type=float, default=0.5)
    parser.add_argument("--output-dir", default="diagnostic/tri_felf_mt")
    parser.add_argument("--output-prefix", default="tri_felf_mt")
    args = parser.parse_args()

    base_diag, base_dir = _read_json(args.baseline_diagnostic)
    felf_diag, felf_dir = _read_json(args.felf_diagnostic)
    mt_diag, mt_dir = _read_json(args.mt_diagnostic)

    paths = {
        "baseline_val": args.baseline_val_logits or _diag_logits(base_diag, "val", base_dir),
        "baseline_test": args.baseline_test_logits or _diag_logits(base_diag, "test", base_dir),
        "felf_val": args.felf_val_logits or _diag_logits(felf_diag, "val", felf_dir),
        "felf_test": args.felf_test_logits or _diag_logits(felf_diag, "test", felf_dir),
        "mt_val": args.mt_val_logits or _diag_logits(mt_diag, "val", mt_dir),
        "mt_test": args.mt_test_logits or _diag_logits(mt_diag, "test", mt_dir),
        "val_labels": args.val_labels or _diag_labels(mt_diag, "val", mt_dir) or _diag_labels(base_diag, "val", base_dir),
        "test_labels": args.test_labels or _diag_labels(mt_diag, "test", mt_dir) or _diag_labels(base_diag, "test", base_dir),
    }

    baseline_val = _load_array(paths["baseline_val"], "baseline val logits").astype(np.float32)
    baseline_test = _load_array(paths["baseline_test"], "baseline test logits").astype(np.float32)
    felf_val = _load_array(paths["felf_val"], "FELF val logits").astype(np.float32)
    felf_test = _load_array(paths["felf_test"], "FELF test logits").astype(np.float32)
    mt_val = _load_array(paths["mt_val"], "MT val logits").astype(np.float32)
    mt_test = _load_array(paths["mt_test"], "MT test logits").astype(np.float32)
    y_val = _load_array(paths["val_labels"], "val labels").astype(np.int64)
    y_test = _load_array(paths["test_labels"], "test labels").astype(np.int64)

    for name, arr in {
        "baseline_val": baseline_val,
        "felf_val": felf_val,
        "mt_val": mt_val,
    }.items():
        if arr.shape != baseline_val.shape:
            raise ValueError(f"{name} shape {arr.shape} != baseline_val shape {baseline_val.shape}")
    for name, arr in {
        "baseline_test": baseline_test,
        "felf_test": felf_test,
        "mt_test": mt_test,
    }.items():
        if arr.shape != baseline_test.shape:
            raise ValueError(f"{name} shape {arr.shape} != baseline_test shape {baseline_test.shape}")

    tri_val = args.baseline_weight * baseline_val + args.felf_weight * felf_val + args.mt_weight * mt_val
    tri_test = args.baseline_weight * baseline_test + args.felf_weight * felf_test + args.mt_weight * mt_test

    rows = []
    for split, labels, arrays in [
        ("val", y_val, {"Baseline": baseline_val, "FELF": felf_val, "MT": mt_val, "Tri-FELF-MT": tri_val}),
        ("test", y_test, {"Baseline": baseline_test, "FELF": felf_test, "MT": mt_test, "Tri-FELF-MT": tri_test}),
    ]:
        for model, logits in arrays.items():
            top1, top5 = _topk(logits, labels)
            rows.append({"split": split, "model": model, "top1": top1, "top5": top5})

    test_top1, test_top5 = _topk(tri_test, y_test)
    val_top1, val_top5 = _topk(tri_val, y_val)
    summary = {
        "architecture": "Tri-FELF-MT",
        "fusion_rule": "z = a*z_Baseline + b*z_FELF + c*z_MT",
        "weights": {
            "baseline": args.baseline_weight,
            "felf": args.felf_weight,
            "mt": args.mt_weight,
        },
        "val_top1": val_top1,
        "val_top5": val_top5,
        "test_top1": test_top1,
        "test_top5": test_top5,
        "movement_vs_baseline": {
            **_movement(baseline_test, tri_test, y_test),
            **_rank_movement(baseline_test, tri_test, y_test),
        },
        "movement_vs_felf": {
            **_movement(felf_test, tri_test, y_test),
            **_rank_movement(felf_test, tri_test, y_test),
        },
        "inputs": {k: str(v) for k, v in paths.items()},
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.output_prefix}_val_logits.npy", tri_val)
    np.save(out_dir / f"{args.output_prefix}_test_logits.npy", tri_test)
    (out_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / f"{args.output_prefix}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "model", "top1", "top5"])
        writer.writeheader()
        writer.writerows(rows)

    print("Tri-FELF-MT")
    print(f"  Weights: Baseline={args.baseline_weight} FELF={args.felf_weight} MT={args.mt_weight}")
    print(f"  Val:  top1 {val_top1:.2f}% top5 {val_top5:.2f}%")
    print(f"  Test: top1 {test_top1:.2f}% top5 {test_top5:.2f}%")
    print(f"  Saved: {out_dir / f'{args.output_prefix}_summary.json'}")


if __name__ == "__main__":
    main()
