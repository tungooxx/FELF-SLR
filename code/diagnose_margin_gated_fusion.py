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


def make_loader(left, right, global_features, labels, batch_size):
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


def collect(model, loader, device):
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for left, right, global_features, y in loader:
            logits.append(model(left.to(device), right.to(device), global_features.to(device)).float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def load_model(path: str, dims, num_classes, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model, _ = load_model_for_checkpoint(Path(path), state, dims[0], dims[1], dims[2], num_classes, 16.0)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def metrics(logits, labels):
    probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=1).numpy()
    t1, t5 = topk_metrics(probs, labels)
    return float(t1), float(t5)


def margin(logits):
    top2 = np.partition(logits, -2, axis=1)[:, -2:]
    top2.sort(axis=1)
    return top2[:, 1] - top2[:, 0]


def true_ranks(logits, labels):
    order = np.argsort(-logits, axis=1)
    ranks = np.empty(labels.shape[0], dtype=np.int64)
    for i, y in enumerate(labels):
        ranks[i] = int(np.where(order[i] == y)[0][0]) + 1
    return ranks


def gated_fuse(b6, felf, threshold):
    use_felf = margin(b6) < threshold
    out = b6.copy()
    out[use_felf] = felf[use_felf]
    return out, use_felf


def row_for(name, split, logits, labels, b6_logits, felf_logits, use_felf=None, threshold=None):
    t1, t5 = metrics(logits, labels)
    pred = logits.argmax(axis=1)
    b6_pred = b6_logits.argmax(axis=1)
    felf_pred = felf_logits.argmax(axis=1)
    b6_correct = b6_pred == labels
    pred_correct = pred == labels
    felf_correct = felf_pred == labels
    rb = true_ranks(b6_logits, labels)
    rg = true_ranks(logits, labels)
    return {
        "model": name,
        "split": split,
        "threshold": "" if threshold is None else threshold,
        "top1": t1,
        "top5": t5,
        "use_felf_count": "" if use_felf is None else int(use_felf.sum()),
        "use_felf_percent": "" if use_felf is None else float(use_felf.mean() * 100.0),
        "b6_wrong_to_correct": int((~b6_correct & pred_correct).sum()),
        "b6_correct_to_wrong": int((b6_correct & ~pred_correct).sum()),
        "felf_wrong_to_correct": int((~felf_correct & pred_correct).sum()),
        "felf_correct_to_wrong": int((felf_correct & ~pred_correct).sum()),
        "rank2_5_to_1": int(((rb >= 2) & (rb <= 5) & (rg == 1)).sum()),
        "rank1_to_wrong": int(((rb == 1) & (rg > 1)).sum()),
    }


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
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--thresholds", default="0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-prefix", default="diagnostic/wlasl300_margin_gated_fusion")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    splits, actions, _ = load_subset_raw_for_source(DATA_DIR, JSON_PATH, args.num_glosses, args.action_source, 0, labels_only=True)
    num_classes = len(actions)
    data = {}
    dims = None
    for split in ["val", "test"]:
        labels = np.asarray(splits[split]["labels"], dtype=np.int64)
        left, right, global_features = encode_feature_parts(splits[split]["raw"], split, args.cache_dir, "old", force=False)
        dims = (left.shape[-1], right.shape[-1], global_features.shape[-1])
        data[split] = {"loader": make_loader(left, right, global_features, labels, args.batch_size)}

    models = {
        "B6": load_model(args.b6_checkpoint, dims, num_classes, device),
        "LRG": load_model(args.lrg_checkpoint, dims, num_classes, device),
        "RF": load_model(args.rf_checkpoint, dims, num_classes, device),
    }
    logits = {"val": {}, "test": {}}
    labels = {}
    for split in ["val", "test"]:
        for name, model in models.items():
            logits[split][name], labels[split] = collect(model, data[split]["loader"], device)
        felf = logits[split]["B6"] + args.lambda_lrg * logits[split]["LRG"] + args.lambda_rf * logits[split]["RF"]
        if args.normalize:
            felf = felf / (1.0 + abs(args.lambda_lrg) + abs(args.lambda_rf))
        logits[split]["FELF"] = felf

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    rows = []
    for split in ["val", "test"]:
        rows.append(row_for("B6", split, logits[split]["B6"], labels[split], logits[split]["B6"], logits[split]["FELF"]))
        rows.append(row_for("FELF", split, logits[split]["FELF"], labels[split], logits[split]["B6"], logits[split]["FELF"]))
        for thr in thresholds:
            z, use = gated_fuse(logits[split]["B6"], logits[split]["FELF"], thr)
            rows.append(row_for("margin_gated", split, z, labels[split], logits[split]["B6"], logits[split]["FELF"], use, thr))

    val_rows = [r for r in rows if r["split"] == "val" and r["model"] == "margin_gated"]
    best_val = max(val_rows, key=lambda r: (r["top1"], r["top5"], -float(r["b6_correct_to_wrong"])))
    best_test = next(r for r in rows if r["split"] == "test" and r["model"] == "margin_gated" and r["threshold"] == best_val["threshold"])
    summary = {"best_val": best_val, "corresponding_test": best_test, "rows": rows}
    prefix = Path(args.output_prefix)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
