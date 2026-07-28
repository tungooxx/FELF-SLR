from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
            logits.append(model(left.to(device), right.to(device), global_features.to(device)).float().cpu())
            labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def load_old_model(path: str, dims, num_classes, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model, _ = load_model_for_checkpoint(Path(path), state, dims[0], dims[1], dims[2], num_classes, 16.0)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def metrics(logits: torch.Tensor, labels: torch.Tensor):
    probs = torch.softmax(logits.float().cpu(), dim=1).numpy()
    t1, t5 = topk_metrics(probs, labels.cpu().numpy())
    return float(t1), float(t5)


def fuse(logits: dict[str, torch.Tensor], lrg_w, rf_w, normalizer=True):
    z = logits["B6"] + lrg_w * logits["LRG"] + rf_w * logits["RF"]
    if normalizer:
        z = z / (1.0 + torch.abs(lrg_w) + torch.abs(rf_w))
    return z


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--num-glosses", type=int, default=300)
    p.add_argument("--cache-dir", default="rework_model/cache/wlasl300_old_B6")
    p.add_argument("--action-source", default="json_first_n", choices=["json_first_n", "top_frequency", "train_dirs"])
    p.add_argument("--b6-checkpoint", default="rework_model/wlasl300_old_B6_dropout02_swa.pth")
    p.add_argument("--lrg-checkpoint", default="rework_model/wlasl300_B6_LRG_ResidualGamma_fixed025_swa.pth")
    p.add_argument("--rf-checkpoint", default="rework_model/wlasl300_B6_RF_ResidualBeta_fixed025_swa.pth")
    p.add_argument("--base-lrg", type=float, default=1.0)
    p.add_argument("--base-rf", type=float, default=0.5)
    p.add_argument("--delta-bound", type=float, default=0.25)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-prefix", default="diagnostic/wlasl300_class_conditional_fusion")
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
        data[split] = {"loader": make_loader(left, right, global_features, labels, args.batch_size), "labels": torch.from_numpy(labels)}

    models = {
        "B6": load_old_model(args.b6_checkpoint, dims, num_classes, device),
        "LRG": load_old_model(args.lrg_checkpoint, dims, num_classes, device),
        "RF": load_old_model(args.rf_checkpoint, dims, num_classes, device),
    }
    logits = {"val": {}, "test": {}}
    labels = {}
    for split in ["val", "test"]:
        for name, model in models.items():
            logits[split][name], labels[split] = collect(model, data[split]["loader"], device)
        labels[split] = labels[split].long()

    global_val = fuse(logits["val"], torch.tensor(args.base_lrg), torch.tensor(args.base_rf))
    global_test = fuse(logits["test"], torch.tensor(args.base_lrg), torch.tensor(args.base_rf))

    delta_lrg_raw = torch.zeros(num_classes, device=device, requires_grad=True)
    delta_rf_raw = torch.zeros(num_classes, device=device, requires_grad=True)
    opt = torch.optim.AdamW([delta_lrg_raw, delta_rf_raw], lr=args.lr, weight_decay=0.0)
    val_logits = {k: v.to(device) for k, v in logits["val"].items()}
    val_y = labels["val"].to(device)
    base_lrg = torch.tensor(args.base_lrg, device=device)
    base_rf = torch.tensor(args.base_rf, device=device)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        delta_lrg = args.delta_bound * torch.tanh(delta_lrg_raw)
        delta_rf = args.delta_bound * torch.tanh(delta_rf_raw)
        lrg_w = base_lrg + delta_lrg.view(1, -1)
        rf_w = base_rf + delta_rf.view(1, -1)
        z = fuse(val_logits, lrg_w, rf_w)
        ce = F.cross_entropy(z, val_y)
        reg = args.l2 * (delta_lrg.pow(2).mean() + delta_rf.pow(2).mean())
        loss = ce + reg
        loss.backward()
        opt.step()

    with torch.no_grad():
        delta_lrg = (args.delta_bound * torch.tanh(delta_lrg_raw)).cpu()
        delta_rf = (args.delta_bound * torch.tanh(delta_rf_raw)).cpu()
        lrg_w = torch.tensor(args.base_lrg) + delta_lrg.view(1, -1)
        rf_w = torch.tensor(args.base_rf) + delta_rf.view(1, -1)
        cc_val = fuse(logits["val"], lrg_w, rf_w)
        cc_test = fuse(logits["test"], lrg_w, rf_w)

    rows = [
        {"model": "B6", "split": "test", "top1": metrics(logits["test"]["B6"], labels["test"])[0], "top5": metrics(logits["test"]["B6"], labels["test"])[1]},
        {"model": "global_fusion", "split": "val", "top1": metrics(global_val, labels["val"])[0], "top5": metrics(global_val, labels["val"])[1]},
        {"model": "global_fusion", "split": "test", "top1": metrics(global_test, labels["test"])[0], "top5": metrics(global_test, labels["test"])[1]},
        {"model": "class_conditional", "split": "val", "top1": metrics(cc_val, labels["val"])[0], "top5": metrics(cc_val, labels["val"])[1]},
        {"model": "class_conditional", "split": "test", "top1": metrics(cc_test, labels["test"])[0], "top5": metrics(cc_test, labels["test"])[1]},
    ]
    summary = {
        "settings": vars(args),
        "metrics": rows,
        "delta_lrg_mean": float(delta_lrg.mean()),
        "delta_lrg_std": float(delta_lrg.std()),
        "delta_rf_mean": float(delta_rf.mean()),
        "delta_rf_std": float(delta_rf.std()),
        "delta_lrg_min": float(delta_lrg.min()),
        "delta_lrg_max": float(delta_lrg.max()),
        "delta_rf_min": float(delta_rf.min()),
        "delta_rf_max": float(delta_rf.max()),
    }
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
