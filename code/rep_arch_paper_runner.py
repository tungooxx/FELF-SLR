from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path("/workspace/local-vlm/SLR/FELF-SLR")
STD = ROOT / "diagnostic/rep_arch_paper_data/standard258_mp01021_uniform40"
CACHE = ROOT / "diagnostic/rep_arch_paper_data/rep_arch_features_v1"
OUT = ROOT / "diagnostic/rep_arch_paper_runs_v1"
SLR1 = Path("/workspace/local-vlm/SLR/slr1_factorial")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(SLR1))

from wlasl_geometry_utils import extract_part_aware_features
from factorial_dev import build_model

SEQ = 40
RAW_POSE_INDICES = [0, 2, 5, 9, 10, 11, 12]
AUGMENT_SEED = 20261002

RECIPE = {
    "name": "REP-ARCH-PAPER-v1",
    "source": "standard258_mp01021_uniform40",
    "architecture": "exact REP-ARCH LocalGlobalB6 / PermutedBranchB6",
    "optimizer": "AdamW",
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 50,
    "patience": 10,
    "scheduler": "ReduceLROnPlateau(factor=0.5, patience=4)",
    "criterion": "weighted cross entropy + label smoothing 0.05",
    "mixup": False,
    "arcface_margin": False,
    "swa": False,
    "augmentation": {
        "fixed_repeats": 2,
        "seed": AUGMENT_SEED,
        "spatial_probability": 0.8,
        "rotation_degrees": 10.0,
        "scale": [0.95, 1.05],
        "xy_translation": [-0.02, 0.02],
        "noise_probability": 0.5,
        "xyz_noise_std_max": 0.005,
        "temporal_shift_probability": 0.5,
        "temporal_shift_frames": 2,
        "landmark_dropout": False,
        "time_warp": False,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def raw_sufficient_frame(v: np.ndarray):
    pose = v[:132].reshape(33, 4)[:, :3].astype(np.float32)
    lh = v[132:195].astype(np.float32)
    rh = v[195:258].astype(np.float32)
    g = pose[RAW_POSE_INDICES].reshape(-1)
    return lh.copy(), rh.copy(), g.copy()


def frame_features(v: np.ndarray, rep: str):
    if rep == "raw_sufficient":
        return raw_sufficient_frame(v)
    if rep == "engineered":
        return tuple(np.asarray(x, np.float32) for x in extract_part_aware_features(v, 0.4))
    raise ValueError(rep)


def encode_sequence(seq: np.ndarray, rep: str):
    l, r, g = [], [], []
    for v in seq:
        a, b, c = frame_features(v, rep)
        l.append(a)
        r.append(b)
        g.append(c)
    return np.asarray(l, np.float32), np.asarray(r, np.float32), np.asarray(g, np.float32)


def time_shift_edge(seq: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return seq.copy()
    idx = np.clip(np.arange(len(seq)) - shift, 0, len(seq) - 1)
    return seq[idx].copy()


def augment_mild(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = seq.astype(np.float32, copy=True)
    if rng.random() < 0.5:
        out = time_shift_edge(out, int(rng.integers(-2, 3)))

    pose = out[:, :132].reshape(SEQ, 33, 4)
    lh = out[:, 132:195].reshape(SEQ, 21, 3)
    rh = out[:, 195:258].reshape(SEQ, 21, 3)

    if rng.random() < 0.8:
        th = math.radians(float(rng.uniform(-10.0, 10.0)))
        rot = np.asarray([[math.cos(th), -math.sin(th)],
                          [math.sin(th), math.cos(th)]], np.float32)
        scale = float(rng.uniform(0.95, 1.05))
        trans = rng.uniform(-0.02, 0.02, size=(2,)).astype(np.float32)
        for t in range(SEQ):
            pose[t, :, :2] = (pose[t, :, :2] @ rot.T) * scale + trans
            pose[t, :, 2] *= scale
            # Missing hands are all-zero. Preserve missingness exactly.
            for hand in (lh[t], rh[t]):
                valid = np.any(np.abs(hand) > 1e-8, axis=1)
                if np.any(valid):
                    hand[valid, :2] = (hand[valid, :2] @ rot.T) * scale + trans
                    hand[valid, 2] *= scale

    if rng.random() < 0.5:
        sd = float(rng.uniform(0.001, 0.005))
        pose[:, :, :3] += rng.normal(0.0, sd, size=pose[:, :, :3].shape).astype(np.float32)
        for hand in (lh, rh):
            valid = np.any(np.abs(hand) > 1e-8, axis=2)
            noise = rng.normal(0.0, sd, size=hand.shape).astype(np.float32)
            hand[valid] += noise[valid]

    out[:, :132] = pose.reshape(SEQ, -1)
    out[:, 132:195] = lh.reshape(SEQ, -1)
    out[:, 195:258] = rh.reshape(SEQ, -1)
    return out


def selected_indices(y: np.ndarray, nclasses: int) -> np.ndarray:
    idx = np.flatnonzero(y < nclasses)
    return idx.astype(np.int64)


def assert_source_complete(nclasses: int) -> None:
    expected = {"train": 3549, "val": 900, "test": 668}
    for split, n in expected.items():
        d = np.load(STD / f"{split}_done.npy", mmap_mode="r")
        y = np.load(STD / f"{split}_y.npy", mmap_mode="r")
        if len(d) != n or len(y) != n:
            raise RuntimeError(f"SOURCE_SHAPE_MISMATCH {split} done={len(d)} y={len(y)} expected={n}")
        idx = selected_indices(np.asarray(y), nclasses)
        completed = int(np.asarray(d[idx], dtype=np.bool_).sum())
        if completed != len(idx):
            raise RuntimeError(f"SOURCE_INCOMPLETE_WLASL{nclasses} {split} {completed}/{len(idx)}")


def cache_paths(nclasses: int, rep: str):
    d = CACHE / f"wlasl{nclasses}" / rep
    return d, {
        s: {
            k: d / f"{s}_{k}.npy"
            for k in ("l", "r", "g", "y", "ids")
        }
        for s in ("train", "val", "test")
    }


def prepare_cache(nclasses: int, rep: str, aug_repeats: int = 2, force: bool = False):
    assert_source_complete(nclasses)
    d, paths = cache_paths(nclasses, rep)
    d.mkdir(parents=True, exist_ok=True)

    dims = (63, 63, 21) if rep == "raw_sufficient" else (165, 165, 23)

    for split in ("train", "val", "test"):
        y_all = np.load(STD / f"{split}_y.npy", mmap_mode="r")
        x_all = np.load(STD / f"{split}_x.npy", mmap_mode="r")
        ids_all = json.loads((STD / f"{split}_ids.json").read_text())
        idx = selected_indices(np.asarray(y_all), nclasses)
        n = len(idx)
        expected = [paths[split]["l"], paths[split]["r"], paths[split]["g"], paths[split]["y"], paths[split]["ids"]]
        if not force and all(p.exists() for p in expected):
            print("BASE_CACHE_EXISTS", split, n, flush=True)
            continue

        L = np.lib.format.open_memmap(paths[split]["l"], mode="w+", dtype=np.float32, shape=(n, SEQ, dims[0]))
        R = np.lib.format.open_memmap(paths[split]["r"], mode="w+", dtype=np.float32, shape=(n, SEQ, dims[1]))
        G = np.lib.format.open_memmap(paths[split]["g"], mode="w+", dtype=np.float32, shape=(n, SEQ, dims[2]))
        yy = np.asarray(y_all[idx], np.int64)
        out_ids = []
        for j, src_i in enumerate(idx):
            a, b, c = encode_sequence(np.asarray(x_all[src_i], np.float32), rep)
            L[j], R[j], G[j] = a, b, c
            out_ids.append(ids_all[int(src_i)])
            if (j + 1) % 250 == 0 or j + 1 == n:
                print("ENCODE_BASE", nclasses, rep, split, j + 1, "/", n, flush=True)
        del L, R, G
        np.save(paths[split]["y"], yy)
        paths[split]["ids"].write_text(json.dumps(out_ids))

    aug_dir = d / f"aug_r{aug_repeats}"
    aug_dir.mkdir(parents=True, exist_ok=True)
    ap = {k: aug_dir / f"train_{k}.npy" for k in ("l", "r", "g", "y")}
    if force or not all(p.exists() for p in ap.values()):
        y_all = np.load(STD / "train_y.npy", mmap_mode="r")
        x_all = np.load(STD / "train_x.npy", mmap_mode="r")
        idx = selected_indices(np.asarray(y_all), nclasses)
        n = len(idx)
        total = n * aug_repeats
        L = np.lib.format.open_memmap(ap["l"], mode="w+", dtype=np.float32, shape=(total, SEQ, dims[0]))
        R = np.lib.format.open_memmap(ap["r"], mode="w+", dtype=np.float32, shape=(total, SEQ, dims[1]))
        G = np.lib.format.open_memmap(ap["g"], mode="w+", dtype=np.float32, shape=(total, SEQ, dims[2]))
        yy = np.empty(total, np.int64)
        rng = np.random.default_rng(AUGMENT_SEED)
        cur = 0
        for j, src_i in enumerate(idx):
            base = np.asarray(x_all[src_i], np.float32)
            for _ in range(aug_repeats):
                aug = augment_mild(base, rng)
                a, b, c = encode_sequence(aug, rep)
                L[cur], R[cur], G[cur], yy[cur] = a, b, c, int(y_all[src_i])
                cur += 1
            if (j + 1) % 100 == 0 or j + 1 == n:
                print("ENCODE_AUG", nclasses, rep, j + 1, "/", n, flush=True)
        del L, R, G
        np.save(ap["y"], yy)

    meta = {
        "nclasses": nclasses,
        "rep": rep,
        "dims": dims,
        "aug_repeats": aug_repeats,
        "recipe": RECIPE,
        "source_metadata": json.loads((STD / "metadata.json").read_text()) if (STD / "metadata.json").exists() else None,
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("CACHE_READY", json.dumps({"nclasses": nclasses, "rep": rep, "dims": dims}), flush=True)


class CombinedDataset(Dataset):
    def __init__(self, base, yb, aug, ya):
        self.base, self.yb, self.aug, self.ya = base, yb, aug, ya
    def __len__(self):
        return len(self.yb) + len(self.ya)
    def __getitem__(self, i):
        if i < len(self.yb):
            src, y, j = self.base, self.yb, i
        else:
            src, y, j = self.aug, self.ya, i - len(self.yb)
        return *(torch.from_numpy(np.array(x[j], dtype=np.float32)) for x in src), int(y[j])


def loader(parts, y, batch=32, shuffle=False):
    ds = TensorDataset(*(torch.from_numpy(np.asarray(x)) for x in parts), torch.from_numpy(np.asarray(y)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, dl, dev, criterion=None):
    model.eval()
    n = correct = 0
    total_loss = 0.0
    zall, yall = [], []
    for l, r, g, y in dl:
        l, r, g, y = [x.to(dev) for x in (l, r, g, y)]
        z = model(l, r, g)
        n += y.numel()
        correct += int((z.argmax(-1) == y).sum())
        if criterion is not None:
            total_loss += float(criterion(z, y)) * y.numel()
        zall.append(z.detach().float().cpu())
        yall.append(y.detach().cpu())
    z = torch.cat(zall)
    y = torch.cat(yall)
    k = min(5, z.shape[1])
    top5 = float((z.topk(k, dim=1).indices == y[:, None]).any(1).float().mean() * 100)
    return {"top1": 100.0 * correct / max(n, 1), "top5": top5,
            "loss": total_loss / max(n, 1) if criterion is not None else None}


def class_weights(y, nclasses):
    c = np.bincount(np.asarray(y), minlength=nclasses).astype(np.float32)
    c = np.maximum(c, 1.0)
    w = 1.0 / c
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def load_rep_cache(nclasses: int, rep: str, aug_repeats: int):
    d, p = cache_paths(nclasses, rep)
    base = {}
    for s in ("train", "val", "test"):
        base[s] = (
            tuple(np.load(p[s][k], mmap_mode="r") for k in ("l", "r", "g")),
            np.load(p[s]["y"], mmap_mode="r"),
        )
    ad = d / f"aug_r{aug_repeats}"
    aug = (
        tuple(np.load(ad / f"train_{k}.npy", mmap_mode="r") for k in ("l", "r", "g")),
        np.load(ad / "train_y.npy", mmap_mode="r"),
    )
    return base, aug


def train_one(nclasses: int, rep: str, arch: str, seed: int, aug_repeats: int):
    base, aug = load_rep_cache(nclasses, rep, aug_repeats)
    tr, ytr = base["train"]
    va, yva = base["val"]
    te, yte = base["test"]
    atr, ay = aug

    seed_all(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = tuple(int(x.shape[-1]) for x in tr)
    model = build_model(arch, dims, nclasses).to(dev)
    params = sum(p.numel() for p in model.parameters())

    cw = class_weights(ytr, nclasses).to(dev)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    ds = CombinedDataset(tr, ytr, atr, ay)
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=False, generator=gen)
    val_dl = loader(va, yva, 32)
    test_dl = loader(te, yte, 32)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

    outdir = OUT / f"wlasl{nclasses}" / f"{rep}_{arch}" / f"seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / "best.pt"

    best_val = -1.0
    stale = 0
    hist = []
    for ep in range(1, 51):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        for l, r, g, y in train_dl:
            l, r, g, y = [x.to(dev) for x in (l, r, g, y)]
            opt.zero_grad(set_to_none=True)
            z = model(l, r, g)
            loss = criterion(z, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach())
            n_batches += 1
        vm = evaluate(model, val_dl, dev, criterion)
        sched.step(vm["loss"])
        row = {"epoch": ep, "train_loss": loss_sum / max(n_batches, 1), **{f"val_{k}": v for k, v in vm.items()}}
        hist.append(row)
        print(json.dumps({"nclasses": nclasses, "rep": rep, "arch": arch, "seed": seed, **row}), flush=True)
        if vm["top1"] > best_val:
            best_val = vm["top1"]
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= 10:
                break

    model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True))
    val_final = evaluate(model, val_dl, dev, criterion)
    test_final = evaluate(model, test_dl, dev, criterion)
    result = {
        "experiment": "REP-ARCH-PAPER-v1",
        "nclasses": nclasses,
        "rep": rep,
        "arch": arch,
        "seed": seed,
        "dims": dims,
        "params": params,
        "recipe": RECIPE,
        "val": val_final,
        "test": test_final,
        "history": hist,
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2))
    print("RESULT", json.dumps({k: result[k] for k in ("nclasses", "rep", "arch", "seed", "dims", "params")}),
          json.dumps({"val": val_final, "test": test_final}), flush=True)
    return result


def smoke():
    rng = np.random.default_rng(7)
    seq = np.load(next((STD / "smoke").glob("*_x.npy"))).astype(np.float32)
    aug = augment_mild(seq, rng)
    report = {"source_shape": list(seq.shape), "aug_finite": bool(np.isfinite(aug).all()), "arms": {}}
    for rep in ("raw_sufficient", "engineered"):
        parts = encode_sequence(aug, rep)
        dims = tuple(int(x.shape[-1]) for x in parts)
        for arch in ("anatomical", "permuted"):
            m = build_model(arch, dims, 100)
            z = m(*(torch.from_numpy(x[None]) for x in parts))
            report["arms"][f"{rep}+{arch}"] = {"dims": dims, "logits": list(z.shape),
                                                  "params": sum(p.numel() for p in m.parameters())}
    print(json.dumps(report, indent=2))
    if not all(v["logits"] == [1, 100] for v in report["arms"].values()):
        raise RuntimeError("SMOKE_FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "prepare", "train"], required=True)
    ap.add_argument("--nclasses", type=int, choices=[100, 300], default=100)
    ap.add_argument("--rep", choices=["raw_sufficient", "engineered"], default="engineered")
    ap.add_argument("--arch", choices=["anatomical", "permuted"], default="anatomical")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--aug-repeats", type=int, default=2)
    ap.add_argument("--force-cache", action="store_true")
    a = ap.parse_args()

    if a.mode == "smoke":
        smoke()
    elif a.mode == "prepare":
        prepare_cache(a.nclasses, a.rep, a.aug_repeats, a.force_cache)
    else:
        train_one(a.nclasses, a.rep, a.arch, a.seed, a.aug_repeats)


if __name__ == "__main__":
    main()
