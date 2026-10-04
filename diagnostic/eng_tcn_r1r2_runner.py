from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path("/workspace/local-vlm/SLR/FELF-SLR")
CACHE = ROOT / "diagnostic/rep_arch_paper_data/rep_arch_features_v1/wlasl100/engineered"
AUG = CACHE / "aug_r2"
PROTOTYPE = ROOT / "diagnostic/engineered_multiscale_tcn_prototype.py"
REP_RUNNER = ROOT / "code/rep_arch_paper_runner.py"
FACTORIAL_DEV = Path("/workspace/local-vlm/SLR/slr1_factorial/factorial_dev.py")
WLASL_GEOM_UTILS = ROOT / "code/wlasl_geometry_utils.py"
DEFAULT_OUT = ROOT / "diagnostic/eng_tcn_r1r2_runs"
PREFLIGHT = ROOT / "diagnostic/eng_tcn_r1r2_preflight.json"

DESIGN_ID = "61107423-135b-436e-8f76-3b8c6fca44f3"
DESIGN_HASH = "19d644f11f36d142845ea6631b29d9615c5f9e5cba793d25793e22c95a6b7794"
PROTOTYPE_SHA = "d4894a0aab33307a3e6fd1a9e49457c8f22b9f01d213bffe38a97c5f9343888a"
REP_RUNNER_SHA = "ef5b771f6b161b2879dde2e6c6ba2913a244261b962d5a90c2d02556aedba3dd"
FACTORIAL_DEV_SHA = "e5171fdeea7bfbd851c3112a52ca5a05fd9db248a8f86c33fb833e9ef7453662"
WLASL_GEOM_UTILS_SHA = "51a7df8b592e1b9bdce8be19d70b54fa0941c611f19d6052b240cf507fdc3f63"
ARMS = ("ENGINEERED_REPARCH_CONTROL", "ENGINEERED_MULTISCALE_TCN")
SEEDS = (20261031, 20261103, 20261107)
NCLASS = 100
TRAIN_N = 1442
AUG_N = 2884
VAL_N = 338
TOTAL_TRAIN = TRAIN_N + AUG_N
BATCH = 32
MAX_EPOCHS = 50
PATIENCE = 10

sys.path.insert(0, str(ROOT / "diagnostic"))
sys.path.insert(0, "/workspace/local-vlm/SLR/slr1_factorial")
from engineered_multiscale_tcn_prototype import EngineeredTemporalTCN
from factorial_dev import build_model


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bytes_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def class_weights(y: np.ndarray) -> torch.Tensor:
    c = np.bincount(np.asarray(y), minlength=NCLASS).astype(np.float32)
    c = np.maximum(c, 1.0)
    w = 1.0 / c
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def open_array(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


def train_paths() -> dict[str, Path]:
    return {k: CACHE / f"train_{k}.npy" for k in ("l", "r", "g", "y", "ids")}


def val_paths() -> dict[str, Path]:
    return {k: CACHE / f"val_{k}.npy" for k in ("l", "r", "g", "y", "ids")}


def aug_paths() -> dict[str, Path]:
    return {k: AUG / f"train_{k}.npy" for k in ("l", "r", "g", "y")}


def load_train_only():
    tp, ap = train_paths(), aug_paths()
    tr = tuple(open_array(tp[k]) for k in ("l", "r", "g"))
    ytr = open_array(tp["y"])
    atr = tuple(open_array(ap[k]) for k in ("l", "r", "g"))
    ay = open_array(ap["y"])
    assert len(ytr) == TRAIN_N and len(ay) == AUG_N
    assert tr[0].shape == (TRAIN_N, 40, 165)
    assert tr[1].shape == (TRAIN_N, 40, 165)
    assert tr[2].shape == (TRAIN_N, 40, 23)
    assert atr[0].shape == (AUG_N, 40, 165)
    assert atr[1].shape == (AUG_N, 40, 165)
    assert atr[2].shape == (AUG_N, 40, 23)
    return tr, ytr, atr, ay


def load_train_val():
    tr, ytr, atr, ay = load_train_only()
    vp = val_paths()
    va = tuple(open_array(vp[k]) for k in ("l", "r", "g"))
    yva = open_array(vp["y"])
    assert len(yva) == VAL_N
    assert va[0].shape == (VAL_N, 40, 165)
    assert va[1].shape == (VAL_N, 40, 165)
    assert va[2].shape == (VAL_N, 40, 23)
    return tr, ytr, atr, ay, va, yva


class IndexedCombinedDataset(Dataset):
    def __init__(self, base, yb, aug, ya):
        self.base, self.yb, self.aug, self.ya = base, yb, aug, ya

    def __len__(self):
        return len(self.yb) + len(self.ya)

    def __getitem__(self, i):
        if i < len(self.yb):
            src, y, j = self.base, self.yb, i
        else:
            src, y, j = self.aug, self.ya, i - len(self.yb)
        parts = tuple(torch.from_numpy(np.array(x[j], dtype=np.float32)) for x in src)
        return *parts, int(y[j]), int(i)


def val_loader(parts, y):
    ds = TensorDataset(
        *(torch.from_numpy(np.asarray(x)) for x in parts),
        torch.from_numpy(np.asarray(y)),
    )
    return DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)


def make_model(arm: str, dims=(165, 165, 23)):
    if arm == "ENGINEERED_REPARCH_CONTROL":
        model = build_model("anatomical", dims, NCLASS)
        expected = 2063232
    elif arm == "ENGINEERED_MULTISCALE_TCN":
        model = EngineeredTemporalTCN(num_classes=NCLASS, channels=256, depth=3)
        expected = 1257474
    else:
        raise ValueError(arm)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params != expected:
        raise RuntimeError(f"PARAM_MISMATCH arm={arm} got={params} expected={expected}")
    return model, params


def expected_row_order_hashes(seed: int, epochs: int = MAX_EPOCHS):
    ds = TensorDataset(torch.arange(TOTAL_TRAIN, dtype=torch.int64))
    gen = torch.Generator()
    gen.manual_seed(seed)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False, generator=gen, num_workers=0)
    out = []
    for ep in range(1, epochs + 1):
        order = []
        for (idx,) in dl:
            order.extend(idx.tolist())
        arr = np.asarray(order, dtype=np.int64)
        assert len(arr) == TOTAL_TRAIN and len(np.unique(arr)) == TOTAL_TRAIN
        out.append({"epoch": ep, "sha256": bytes_sha256(arr), "n": len(arr)})
    return out


def cache_hash_manifest():
    paths = {}
    for prefix, d in (("train", train_paths()), ("val", val_paths()), ("aug", aug_paths())):
        for k, p in d.items():
            paths[f"{prefix}_{k}"] = {"path": str(p), "sha256": sha256(p), "size": p.stat().st_size}
    return paths


def audit(write: bool = True):
    if sha256(PROTOTYPE) != PROTOTYPE_SHA:
        raise RuntimeError("PROTOTYPE_SHA_DRIFT")
    if sha256(REP_RUNNER) != REP_RUNNER_SHA:
        raise RuntimeError("REP_RUNNER_SHA_DRIFT")
    if sha256(FACTORIAL_DEV) != FACTORIAL_DEV_SHA:
        raise RuntimeError("FACTORIAL_DEV_SHA_DRIFT")
    if sha256(WLASL_GEOM_UTILS) != WLASL_GEOM_UTILS_SHA:
        raise RuntimeError("WLASL_GEOM_UTILS_SHA_DRIFT")

    tr, ytr, atr, ay = load_train_only()
    vp = val_paths()
    va_shapes = {k: list(open_array(vp[k]).shape) for k in ("l", "r", "g", "y")}
    va_shapes["ids"] = [VAL_N]  # provenance file is hashed below; do not deserialize object-array IDs
    _, p0 = make_model(ARMS[0])
    _, p1 = make_model(ARMS[1])
    row_manifest = {str(s): expected_row_order_hashes(s) for s in SEEDS}

    report = {
        "kind": "ENG_TCN_R1R2_PRE_SCIENCE_PREFLIGHT",
        "design_id": DESIGN_ID,
        "design_semantic_hash": DESIGN_HASH,
        "prototype_sha256": PROTOTYPE_SHA,
        "rep_arch_runner_sha256": REP_RUNNER_SHA,
        "factorial_dev_sha256": FACTORIAL_DEV_SHA,
        "wlasl_geometry_utils_sha256": WLASL_GEOM_UTILS_SHA,
        "arms": list(ARMS),
        "frozen_seeds": list(SEEDS),
        "train_count": int(len(ytr)),
        "aug_count": int(len(ay)),
        "total_train_rows": int(len(ytr) + len(ay)),
        "validation_count": VAL_N,
        "train_shapes": [list(x.shape) for x in tr],
        "aug_shapes": [list(x.shape) for x in atr],
        "val_shapes": va_shapes,
        "params": {ARMS[0]: p0, ARMS[1]: p1},
        "cache_hashes": cache_hash_manifest(),
        "row_order_manifest": row_manifest,
        "training_law": {
            "optimizer": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=4)",
            "criterion": "inverse-frequency base-train class-weighted CE label_smoothing=0.05",
            "batch_size": BATCH,
            "epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "grad_clip_l2": 1.0,
            "checkpoint_rule": "strictly greater validation correct count; ties retain earlier checkpoint",
        },
        "firewall": {
            "wlasl100_test258_opened": False,
            "wlasl300_opened": False,
            "external_private_opened": False,
        },
    }
    if write:
        PREFLIGHT.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "audit": "PASS",
        "design_hash": DESIGN_HASH,
        "params": report["params"],
        "cache_hash_count": len(report["cache_hashes"]),
        "row_manifest_seeds": list(report["row_order_manifest"]),
        "firewall": report["firewall"],
    }, sort_keys=True), flush=True)
    return report


@torch.no_grad()
def evaluate(model, dl, dev, criterion, return_logits=False):
    model.eval()
    n = correct = 0
    total_loss = 0.0
    logits_all = []
    for l, r, g, y in dl:
        l, r, g, y = [x.to(dev) for x in (l, r, g, y)]
        z = model(l, r, g)
        n += int(y.numel())
        correct += int((z.argmax(-1) == y).sum())
        total_loss += float(criterion(z, y)) * y.numel()
        if return_logits:
            logits_all.append(z.detach().float().cpu())
    out = {
        "correct": correct,
        "total": n,
        "top1": 100.0 * correct / max(n, 1),
        "loss": total_loss / max(n, 1),
    }
    if return_logits:
        out["logits"] = torch.cat(logits_all).numpy()
    return out


def smoke(arm: str, seed: int):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("FROZEN_ARM_OR_SEED_VIOLATION")
    tr, ytr, _, _ = load_train_only()
    seed_all(seed)
    model, params = make_model(arm)
    model.train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    cw = class_weights(ytr).to(dev)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    l, r, g = [torch.from_numpy(np.array(x[:8], dtype=np.float32)).to(dev) for x in tr]
    y = torch.from_numpy(np.array(ytr[:8], dtype=np.int64)).to(dev)
    z = model(l, r, g)
    loss = criterion(z, y)
    loss.backward()
    finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    if not finite:
        raise RuntimeError("NONFINITE_GRAD")
    print(json.dumps({
        "kind": "TRAIN_ONLY_ENGINEERING_SMOKE",
        "arm": arm,
        "seed": seed,
        "device": str(dev),
        "params": params,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "all_existing_grads_finite": finite,
        "validation_opened": False,
        "test258_opened": False,
        "wlasl300_opened": False,
    }, sort_keys=True), flush=True)


def execute(arm: str, seed: int, out_root: Path):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("FROZEN_ARM_OR_SEED_VIOLATION")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_SCIENCE_EXECUTION")

    pre = audit(write=True)
    expected = pre["row_order_manifest"][str(seed)]
    tr, ytr, atr, ay, va, yva = load_train_val()

    seed_all(seed)
    dev = torch.device("cuda")
    model, params = make_model(arm)
    model = model.to(dev)

    cw = class_weights(ytr).to(dev)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    ds = IndexedCombinedDataset(tr, ytr, atr, ay)
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False, generator=gen, num_workers=0)
    vdl = val_loader(va, yva)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

    outdir = Path(out_root) / arm / f"seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / "best.pt"
    hist_path = outdir / "history.jsonl"
    order_path = outdir / "row_order_observed.json"

    best_correct = -1
    best_epoch = None
    stale = 0
    hist = []
    observed_orders = []

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        order = []
        for l, r, g, y, idx in train_dl:
            order.extend(idx.tolist())
            l, r, g, y = [x.to(dev) for x in (l, r, g, y)]
            opt.zero_grad(set_to_none=True)
            z = model(l, r, g)
            loss = criterion(z, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach())
            n_batches += 1

        order_arr = np.asarray(order, dtype=np.int64)
        order_hash = bytes_sha256(order_arr)
        exp_hash = expected[ep - 1]["sha256"]
        if order_hash != exp_hash:
            raise RuntimeError(f"ROW_ORDER_DRIFT epoch={ep} observed={order_hash} expected={exp_hash}")
        observed_orders.append({"epoch": ep, "sha256": order_hash, "n": len(order)})

        vm = evaluate(model, vdl, dev, criterion)
        sched.step(vm["loss"])
        row = {
            "epoch": ep,
            "train_loss": loss_sum / max(n_batches, 1),
            "val_correct": vm["correct"],
            "val_total": vm["total"],
            "val_top1": vm["top1"],
            "val_loss": vm["loss"],
            "lr": float(opt.param_groups[0]["lr"]),
            "row_order_sha256": order_hash,
        }
        hist.append(row)
        hist_path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in hist) + "\n")
        order_path.write_text(json.dumps(observed_orders, indent=2))
        print(json.dumps({"arm": arm, "seed": seed, **row}, sort_keys=True), flush=True)

        if vm["correct"] > best_correct:
            best_correct = vm["correct"]
            best_epoch = ep
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_epoch is None or not best_path.exists():
        raise RuntimeError("NO_CHECKPOINT")

    model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True))
    final = evaluate(model, vdl, dev, criterion, return_logits=True)
    logits = final.pop("logits")
    preds = logits.argmax(axis=1).astype(np.int64)

    logits_path = outdir / "val_logits.npy"
    pred_path = outdir / "val_pred.npy"
    np.save(logits_path, logits)
    np.save(pred_path, preds)
    logits_sha = sha256(logits_path)
    pred_sha = sha256(pred_path)
    checkpoint_sha = sha256(best_path)

    # Reduction only after immutable prediction/logit bytes have been written and hashed.
    yva_arr = np.asarray(yva, dtype=np.int64)
    reduced_correct = int((preds == yva_arr).sum())
    if reduced_correct != final["correct"]:
        raise RuntimeError("FINAL_REDUCTION_MISMATCH")

    result = {
        "experiment": "ENG-TCN-R1R2",
        "design_id": DESIGN_ID,
        "design_semantic_hash": DESIGN_HASH,
        "arm": arm,
        "seed": seed,
        "params": params,
        "selected_epoch": best_epoch,
        "val": final,
        "checkpoint_sha256": checkpoint_sha,
        "val_logits_sha256": logits_sha,
        "val_pred_sha256": pred_sha,
        "prototype_sha256": PROTOTYPE_SHA,
        "rep_arch_runner_sha256": REP_RUNNER_SHA,
        "factorial_dev_sha256": FACTORIAL_DEV_SHA,
        "wlasl_geometry_utils_sha256": WLASL_GEOM_UTILS_SHA,
        "cache_hashes": pre["cache_hashes"],
        "row_order_observed": observed_orders,
        "firewall": {
            "wlasl100_test258_opened": False,
            "wlasl300_opened": False,
            "external_private_opened": False,
        },
        "scientific_scope": "WLASL100_DEVELOPMENT_ONLY_PACKAGE_LEVEL",
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2))
    print("RESULT", json.dumps({
        "arm": arm,
        "seed": seed,
        "selected_epoch": best_epoch,
        "val_correct": final["correct"],
        "val_total": final["total"],
        "val_top1": final["top1"],
        "val_loss": final["loss"],
        "checkpoint_sha256": checkpoint_sha,
        "val_pred_sha256": pred_sha,
    }, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("audit", "smoke", "execute"))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    a = ap.parse_args()

    if a.mode == "audit":
        audit(write=True)
        return
    if a.arm is None or a.seed is None:
        raise SystemExit("--arm and --seed are required for smoke/execute")
    if a.seed not in SEEDS:
        raise SystemExit(f"seed must be one of {SEEDS}")
    if a.mode == "smoke":
        smoke(a.arm, a.seed)
    else:
        execute(a.arm, a.seed, Path(a.out_root))


if __name__ == "__main__":
    main()

