"""FineShape-v1: conservative handshape-aware top-5 reranking.

The scorer is trained only on validation samples. Thresholds are selected from
cross-fitted validation predictions, then the scorer is refit on full
validation data and evaluated once on test.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("rework_model/cache/wlasl300_old_B6/frame_old")
QUALITY_ROOT = Path("rework_model/cache/wlasl300_sampling_v1_frozen/frame_old")
RAW_ROOT = Path("rework_model/cache/wlasl300_raw_keypoint_repair")
MT_ROOT = Path("rework_model/cache/wlasl300_morphtraj_firstn/morph_traj")
OUT = Path("diagnostic/wlasl300_fineshape_v1")
SEEDS = (1, 2, 3)
TOPK = 5


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ranks(logits: np.ndarray) -> np.ndarray:
    order = np.argsort(-logits, axis=1)
    inv = np.empty_like(order)
    inv[np.arange(len(order))[:, None], order] = np.arange(logits.shape[1])[None, :]
    return inv + 1


def metrics(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-logits, axis=1)
    return (
        float(100 * np.mean(order[:, 0] == labels)),
        float(100 * np.mean(np.any(order[:, :5] == labels[:, None], axis=1))),
    )


def palmnorm(raw: np.ndarray) -> np.ndarray:
    """Palm-centered normalized hand vectors for both hands."""
    hands = [raw[:, :, 99:162].reshape(len(raw), 40, 21, 3), raw[:, :, 162:225].reshape(len(raw), 40, 21, 3)]
    outputs = []
    for hand in hands:
        center = hand[:, :, [0, 5, 9, 13, 17]].mean(axis=2, keepdims=True)
        scale = np.linalg.norm(hand[:, :, 0] - hand[:, :, 9], axis=-1, keepdims=True)
        valid = np.any(np.abs(hand) > 1e-8, axis=(2, 3), keepdims=True)
        scale = np.where(scale > 1e-6, scale, 1.0)[..., None]
        outputs.append(np.where(valid, (hand - center) / scale, 0.0).reshape(len(raw), 40, -1))
    return np.concatenate(outputs, axis=-1).astype(np.float32)


def shape_columns(x: np.ndarray) -> np.ndarray:
    # Local layout: rel 60, bone vectors 60, lengths 20, flex 15,
    # spread 4, palm 3, pinch/tip-spread/scale 3.
    return np.concatenate([x[..., :60], x[..., 140:159], x[..., 162:165]], axis=-1)


def temporal_stats(x: np.ndarray) -> np.ndarray:
    middle = x[:, x.shape[1] // 3 : 2 * x.shape[1] // 3].mean(axis=1)
    return np.concatenate([x.mean(axis=1), x.std(axis=1), x[:, x.shape[1] // 2], middle], axis=1).astype(np.float32)


def signatures(split: str) -> np.ndarray:
    ul = np.load(ROOT / f"{split}_old_left.npy", mmap_mode="r")
    ur = np.load(ROOT / f"{split}_old_right.npy", mmap_mode="r")
    ql = np.load(QUALITY_ROOT / f"{split}_old_left.npy", mmap_mode="r")
    qr = np.load(QUALITY_ROOT / f"{split}_old_right.npy", mmap_mode="r")
    raw = np.load(RAW_ROOT / f"{split}_raw.npy", mmap_mode="r")
    uniform = temporal_stats(np.concatenate([shape_columns(ul), shape_columns(ur)], axis=-1))
    quality = temporal_stats(np.concatenate([shape_columns(ql), shape_columns(qr)], axis=-1))
    palm = temporal_stats(palmnorm(raw))
    return np.concatenate([uniform, quality, palm], axis=1).astype(np.float32)


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-5] = 1.0
    return tuple(((x - mean) / std).astype(np.float32) for x in (train, *others))


def prototypes(x: np.ndarray, labels: np.ndarray, classes: int) -> np.ndarray:
    out = np.zeros((classes, x.shape[1]), dtype=np.float32)
    for c in range(classes):
        out[c] = x[labels == c].mean(axis=0)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out


def candidate_features(
    base: np.ndarray,
    tri: np.ndarray,
    quality: np.ndarray,
    sig: np.ndarray,
    proto: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate = np.argsort(-base, axis=1)[:, :TOPK]
    row = np.arange(len(base))[:, None]
    base_prob = softmax(base)
    tri_prob = softmax(tri)
    quality_prob = softmax(quality)
    base_rank = ranks(base)[row, candidate].astype(np.float32)
    quality_rank = ranks(quality)[row, candidate].astype(np.float32)
    sig_norm = sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-8)
    morphology_similarity = np.einsum("nd,nkd->nk", sig_norm, proto[candidate])
    pair_similarity = np.einsum("nkd,nd->nk", proto[candidate], proto[candidate[:, 0]])
    sorted_base = np.sort(base, axis=1)
    margin = sorted_base[:, -1] - sorted_base[:, -2]
    margin = np.repeat(margin[:, None], TOPK, axis=1)
    features = np.stack(
        [
            base[row, candidate],
            tri[row, candidate],
            quality[row, candidate],
            base_prob[row, candidate],
            tri_prob[row, candidate],
            quality_prob[row, candidate],
            base_rank,
            quality_rank,
            morphology_similarity,
            morphology_similarity - morphology_similarity[:, :1],
            pair_similarity,
            margin,
        ],
        axis=-1,
    ).astype(np.float32)
    return candidate, features, morphology_similarity, pair_similarity


def fit_model(x: np.ndarray, candidate: np.ndarray, labels: np.ndarray):
    y = (candidate == labels[:, None]).reshape(-1).astype(np.int64)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42),
    )
    model.fit(x.reshape(-1, x.shape[-1]), y)
    return model


def score_model(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(x.shape[:2])


def apply_policy(
    base: np.ndarray,
    candidate: np.ndarray,
    scores: np.ndarray,
    pair_sim: np.ndarray,
    margin_threshold: float,
    advantage_threshold: float,
    pair_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred_local = scores.argmax(axis=1)
    proposed = candidate[np.arange(len(candidate)), pred_local]
    original = candidate[:, 0]
    advantage = scores[np.arange(len(scores)), pred_local] - scores[:, 0]
    vals = np.sort(base, axis=1)
    margin = vals[:, -1] - vals[:, -2]
    pair = pair_sim[np.arange(len(pair_sim)), pred_local]
    change = (proposed != original) & (margin <= margin_threshold) & (advantage >= advantage_threshold) & (pair >= pair_threshold)
    out = base.copy()
    out[np.where(change)[0], proposed[change]] = out[np.where(change)[0], original[change]] + 1e-3
    return out, change


def select_policy(base, candidate, scores, pair_sim, labels) -> dict[str, float]:
    vals = np.sort(base, axis=1)
    margins = vals[:, -1] - vals[:, -2]
    proposed_local = scores.argmax(axis=1)
    advantages = scores[np.arange(len(scores)), proposed_local] - scores[:, 0]
    pairs = pair_sim[np.arange(len(scores)), proposed_local]
    proposed_change = proposed_local != 0
    positive_change = proposed_change & (advantages > 0)
    if not np.any(positive_change):
        return {
            "margin_threshold": float("-inf"),
            "advantage_threshold": float("inf"),
            "pair_threshold": float("inf"),
            "top1": metrics(base, labels)[0],
            "top5": metrics(base, labels)[1],
            "fixes": 0,
            "harms": 0,
            "attempts": 0,
            "score": metrics(base, labels)[0],
        }
    grids = (
        np.quantile(margins[positive_change], [0.4, 0.6, 0.8, 1.0]),
        np.quantile(advantages[positive_change], [0.0, 0.25, 0.5, 0.75]),
        np.quantile(pairs[positive_change], [0.0, 0.25, 0.5, 0.75]),
    )
    base_correct = base.argmax(axis=1) == labels
    rows = []
    for mt in grids[0]:
        for at in grids[1]:
            for pt in grids[2]:
                out, change = apply_policy(base, candidate, scores, pair_sim, float(mt), float(at), float(pt))
                correct = out.argmax(axis=1) == labels
                fixes = int((~base_correct & correct).sum())
                harms = int((base_correct & ~correct).sum())
                top1, top5 = metrics(out, labels)
                rows.append(
                    {
                        "margin_threshold": float(mt),
                        "advantage_threshold": float(at),
                        "pair_threshold": float(pt),
                        "top1": top1,
                        "top5": top5,
                        "fixes": fixes,
                        "harms": harms,
                        "attempts": int(change.sum()),
                        "score": top1 - 0.2 * harms,
                    }
                )
    safe = [r for r in rows if r["harms"] <= 2 and r["attempts"] > 0]
    return max(safe or rows, key=lambda r: (r["score"], r["fixes"], -r["harms"], -r["attempts"]))


def load_logits(seed: int, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b6_dir = ROOT / f"wlasl300_seed{seed}_B6"
    felf_dir = ROOT / f"wlasl300_FELF_seed{seed}_lrg10_rf075"
    mt_dir = MT_ROOT / f"wlasl300_MorphTrajExpert_seed{seed}"
    quality_dir = QUALITY_ROOT / f"wlasl300_sampling_v1_frozen_B6_seed{seed}"
    b6 = np.load(b6_dir / f"{split}_logits.npy")
    felf = np.load(felf_dir / f"{split}_logits.npy")
    mt = np.load(mt_dir / f"{split}_logits_fused.npy")
    quality = np.load(quality_dir / f"{split}_logits.npy")
    return b6, felf, mt, quality


def run_seed(seed: int, sig_train, sig_val, sig_test, proto, y_val, y_test) -> dict:
    vb, vf, vm, vq = load_logits(seed, "val")
    tb, tf, tm, tq = load_logits(seed, "test")
    val_tri = 0.5 * vb + vf + 0.5 * vm
    test_tri = 0.5 * tb + tf + 0.5 * tm
    val_base = val_tri + 0.1 * vq
    test_base = test_tri + 0.1 * tq
    vc, vx, _, vp = candidate_features(val_base, val_tri, vq, sig_val, proto)
    tc, tx, _, tp = candidate_features(test_base, test_tri, tq, sig_test, proto)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(y_val), TOPK), dtype=np.float32)
    for train_idx, held_idx in splitter.split(np.zeros(len(y_val)), y_val):
        model = fit_model(vx[train_idx], vc[train_idx], y_val[train_idx])
        oof[held_idx] = score_model(model, vx[held_idx])
    policy = select_policy(val_base, vc, oof, vp, y_val)
    model = fit_model(vx, vc, y_val)
    test_scores = score_model(model, tx)
    reranked, changed = apply_policy(
        test_base,
        tc,
        test_scores,
        tp,
        policy["margin_threshold"],
        policy["advantage_threshold"],
        policy["pair_threshold"],
    )
    base_correct = test_base.argmax(axis=1) == y_test
    final_correct = reranked.argmax(axis=1) == y_test
    fixes = np.where(~base_correct & final_correct)[0]
    harms = np.where(base_correct & ~final_correct)[0]
    base_top1, base_top5 = metrics(test_base, y_test)
    top1, top5 = metrics(reranked, y_test)
    np.save(OUT / f"seed{seed}_reranked_logits.npy", reranked)
    return {
        "seed": seed,
        "base_top1": base_top1,
        "base_top5": base_top5,
        "top1": top1,
        "top5": top5,
        "fixes": int(len(fixes)),
        "harms": int(len(harms)),
        "net": int(len(fixes) - len(harms)),
        "attempts": int(changed.sum()),
        "fixed_indices": fixes.tolist(),
        "harmed_indices": harms.tolist(),
        "policy": policy,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y_train = np.load(RAW_ROOT / "train_labels.npy").astype(np.int64)
    y_val = np.load(RAW_ROOT / "val_labels.npy").astype(np.int64)
    y_test = np.load(RAW_ROOT / "test_labels.npy").astype(np.int64)
    sig_train, sig_val, sig_test = standardize(signatures("train"), signatures("val"), signatures("test"))
    proto = prototypes(sig_train, y_train, 300)
    rows = [run_seed(seed, sig_train, sig_val, sig_test, proto, y_val, y_test) for seed in SEEDS]
    aggregate = {}
    for key in ("base_top1", "base_top5", "top1", "top5"):
        values = np.asarray([row[key] for row in rows])
        aggregate[f"{key}_mean"] = float(values.mean())
        aggregate[f"{key}_std"] = float(values.std(ddof=1))
    aggregate["fixes_mean"] = float(np.mean([r["fixes"] for r in rows]))
    aggregate["harms_mean"] = float(np.mean([r["harms"] for r in rows]))
    report = {
        "experiment": "FineShape-v1",
        "base": "Tri-FELF-MT + 0.1 Quality-Apex B6",
        "training": "5-fold cross-fitted validation scorer; thresholds selected on OOF validation; test evaluated once",
        "activation": "low Tri+Quality margin AND candidate in top5 AND morphology-similar class pair",
        "seeds": rows,
        "aggregate": aggregate,
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "seed_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["seed", "base_top1", "base_top5", "top1", "top5", "fixes", "harms", "net", "attempts"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
