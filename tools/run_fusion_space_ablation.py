"""Compare embedding- and decision-level fusion with the same frozen experts.

This is a controlled reviewer ablation. Baseline, LRG, RF, and MT are frozen,
their train/validation/test embeddings and logits are exported once, and every
fusion method uses those exact exports. Fusion heads are trained on the
training split, selected on validation Top-1, and evaluated once on test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from train_morph_traj_expert import MorphTrajExpert
from wlasl_train_felf_slr_subset import FELFSLR, load_expert_checkpoint


DEFAULT_PUBLIC_CHECKPOINTS = ROOT / "checkpoints"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def topk(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    indices = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float((indices == labels[:, None]).any(axis=1).mean() * 100.0)


class NumpyPartsDataset(Dataset):
    def __init__(self, arrays: list[np.ndarray], labels: np.ndarray):
        self.arrays = arrays
        self.labels = labels
        if any(len(array) != len(labels) for array in arrays):
            raise ValueError("Feature and label counts differ")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        parts = [torch.from_numpy(np.asarray(array[index], dtype=np.float32)) for array in self.arrays]
        return (*parts, torch.tensor(int(self.labels[index]), dtype=torch.long))


def load_train_labels(cache: Path, expected: int) -> np.ndarray:
    augmented = np.load(cache / "aug_r10_old_labels.npy", mmap_mode="r")
    if len(augmented) != expected * 10:
        raise ValueError(f"Expected {expected * 10} augmented labels, found {len(augmented)}")
    grouped = np.asarray(augmented).reshape(expected, 10)
    if not np.all(grouped == grouped[:, :1]):
        raise ValueError("The fixed augmentation cache is not sample-major")
    return grouped[:, 0].astype(np.int64)


def model_args() -> Namespace:
    return Namespace(
        left_branch_dim=96,
        right_branch_dim=96,
        global_branch_dim=96,
        dropout=0.2,
        num_layers=2,
        num_heads=4,
        ff_dim=768,
        branch_stem_kind="base",
        scale=16.0,
        margin=0.2,
        use_lrg_head=True,
        use_rf_head=True,
        lrg_logit_weight=1.0,
        rf_logit_weight=0.75,
        normalize_fused_logits=True,
        trainable_fusion_weights=False,
        trainable_fusion_normalizer=False,
        use_adaptive_router=False,
        trainable_adaptive_router=True,
        router_entropy_weight=2.0,
        router_uncertainty_weight=2.0,
        router_bias=-2.0,
        use_global_reliability_gate=False,
        global_gate_mode="soft",
        global_gate_low=1e-8,
        global_gate_high=0.7259677648544312,
        lrg_residual_gamma_init=0.25,
        lrg_residual_gamma_mode="fixed",
        rf_residual_beta_init=0.25,
        rf_residual_beta_mode="fixed",
    )


@dataclass(frozen=True)
class DatasetPaths:
    old_cache: Path
    mt_cache: Path
    summary: Path
    b6_checkpoint: Path
    lrg_checkpoint: Path
    rf_checkpoint: Path
    mt_checkpoint: Path


def paths_for(dataset: int, public_checkpoints: Path) -> DatasetPaths:
    return DatasetPaths(
        old_cache=ROOT / "rework_model" / "cache" / f"wlasl{dataset}_old_B6" / "frame_old",
        mt_cache=ROOT / "rework_model" / "cache" / f"wlasl{dataset}_morphtraj_firstn" / "morph_traj",
        summary=(
            ROOT
            / "diagnostic"
            / "tri_felf_mt_seed_stability"
            / f"wlasl{dataset}_seed1_tri_felf_mt_summary.json"
        ),
        b6_checkpoint=ROOT / "rework_model" / f"wlasl{dataset}_seed1_B6_swa.pth",
        lrg_checkpoint=public_checkpoints / f"wlasl{dataset}_B6_LRG_ResidualGamma_fixed025_swa.pth",
        rf_checkpoint=public_checkpoints / f"wlasl{dataset}_B6_RF_ResidualBeta_fixed025_swa.pth",
        mt_checkpoint=public_checkpoints / f"wlasl{dataset}_MorphTrajExpert_seed1_best_by_val.pth",
    )


def split_arrays(paths: DatasetPaths, split: str) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    old = [
        np.load(paths.old_cache / f"{split}_old_left.npy", mmap_mode="r"),
        np.load(paths.old_cache / f"{split}_old_right.npy", mmap_mode="r"),
        np.load(paths.old_cache / f"{split}_old_global.npy", mmap_mode="r"),
    ]
    mt = [
        np.load(paths.mt_cache / f"{split}_morph.npy", mmap_mode="r"),
        np.load(paths.mt_cache / f"{split}_traj.npy", mmap_mode="r"),
        np.load(paths.mt_cache / f"{split}_orient.npy", mmap_mode="r"),
    ]
    if split == "train":
        labels = load_train_labels(paths.old_cache, len(old[0]))
    else:
        summary = json.loads(paths.summary.read_text())
        labels = np.load(summary["inputs"][f"{split}_labels"]).astype(np.int64)
    if any(len(array) != len(labels) for array in old + mt):
        raise ValueError(f"{split} feature arrays are not aligned")
    return old, mt, labels


def load_models(paths: DatasetPaths, classes: int, device: torch.device):
    for checkpoint in (
        paths.b6_checkpoint,
        paths.lrg_checkpoint,
        paths.rf_checkpoint,
        paths.mt_checkpoint,
    ):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    felf = FELFSLR((165, 165, 23), classes, model_args()).to(device)
    preload = {
        "baseline": load_expert_checkpoint(felf.main, str(paths.b6_checkpoint), device),
        "lrg": load_expert_checkpoint(felf.lrg, str(paths.lrg_checkpoint), device),
        "rf": load_expert_checkpoint(felf.rf, str(paths.rf_checkpoint), device),
    }
    for name, info in preload.items():
        if info["missing"] or info["unexpected"] or info["skipped_shape"]:
            raise RuntimeError(f"Incompatible {name} checkpoint: {info}")
    mt = MorphTrajExpert(310, 131, 40, classes, 128, 0.2, conditioning="none").to(device)
    mt.load_state_dict(torch.load(paths.mt_checkpoint, map_location=device, weights_only=True), strict=True)
    felf.eval()
    mt.eval()
    return felf, mt, preload


@torch.inference_mode()
def export_old(
    model: FELFSLR,
    arrays: list[np.ndarray],
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    split: str,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        NumpyPartsDataset(arrays, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    result = {name: [] for name in ("baseline_embedding", "lrg_embedding", "rf_embedding")}
    result.update({name: [] for name in ("baseline_logits", "lrg_logits", "rf_logits")})
    for left, right, glob, _ in tqdm(loader, desc=f"{split}: B6/LRG/RF", leave=False):
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        glob = glob.to(device, non_blocking=True)
        for name, expert in (
            ("baseline", model.main),
            ("lrg", model.lrg),
            ("rf", model.rf),
        ):
            embedding = expert.forward_features(left, right, glob)
            logits = expert.classifier(embedding)
            result[f"{name}_embedding"].append(embedding.cpu().numpy())
            result[f"{name}_logits"].append(logits.cpu().numpy())
    return {name: np.concatenate(values) for name, values in result.items()}


@torch.inference_mode()
def export_mt(
    model: MorphTrajExpert,
    arrays: list[np.ndarray],
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    split: str,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        NumpyPartsDataset(arrays, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    embeddings = []
    logits = []
    for morph, traj, orient, _ in tqdm(loader, desc=f"{split}: MT", leave=False):
        morph = morph.to(device, non_blocking=True)
        traj = traj.to(device, non_blocking=True)
        orient = orient.to(device, non_blocking=True)
        *_, embedding = model.forward_features(morph, traj, orient)
        embeddings.append(embedding.cpu().numpy())
        logits.append(model.classifier(embedding).cpu().numpy())
    return {
        "mt_embedding": np.concatenate(embeddings),
        "mt_logits": np.concatenate(logits),
    }


def export_frozen_bank(
    dataset: int,
    paths: DatasetPaths,
    output: Path,
    batch_size: int,
    force: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = dataset
    felf, mt, preload = load_models(paths, classes, device)
    bank: dict[str, dict[str, np.ndarray]] = {}
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        expected = [
            output / f"{split}_{name}.npy"
            for name in (
                "baseline_embedding",
                "lrg_embedding",
                "rf_embedding",
                "mt_embedding",
                "baseline_logits",
                "lrg_logits",
                "rf_logits",
                "mt_logits",
                "labels",
            )
        ]
        if not force and all(path.exists() for path in expected):
            bank[split] = {path.stem.removeprefix(f"{split}_"): np.load(path) for path in expected}
            continue
        old, mt_parts, labels = split_arrays(paths, split)
        data = export_old(felf, old, labels, device, batch_size, split)
        data.update(export_mt(mt, mt_parts, labels, device, batch_size, split))
        data["labels"] = labels
        for name, array in data.items():
            np.save(output / f"{split}_{name}.npy", array)
        bank[split] = data
    checkpoint_manifest = {}
    for name, checkpoint in {
        "baseline": paths.b6_checkpoint,
        "lrg": paths.lrg_checkpoint,
        "rf": paths.rf_checkpoint,
        "mt": paths.mt_checkpoint,
    }.items():
        checkpoint_manifest[name] = {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        }
    return bank, {
        "device": str(device),
        "preload": preload,
        "checkpoints": checkpoint_manifest,
    }


class ConcatFusion(nn.Module):
    def __init__(self, dims: list[int], classes: int):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in dims])
        self.head = nn.Sequential(
            nn.Linear(sum(dims), 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, classes),
        )

    def forward(self, *parts):
        return self.head(torch.cat([norm(part) for norm, part in zip(self.norms, parts)], dim=-1))


class ResidualFusion(nn.Module):
    def __init__(self, dims: list[int], classes: int):
        super().__init__()
        base_dim = dims[0]
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in dims])
        self.projections = nn.ModuleList([nn.Linear(dim, base_dim) for dim in dims[1:]])
        self.scales = nn.Parameter(torch.full((len(dims) - 1,), 0.1))
        self.output_norm = nn.LayerNorm(base_dim)
        self.classifier = nn.Linear(base_dim, classes)

    def forward(self, *parts):
        fused = self.norms[0](parts[0])
        for scale, projection, norm, part in zip(
            self.scales,
            self.projections,
            self.norms[1:],
            parts[1:],
        ):
            fused = fused + scale * projection(norm(part))
        return self.classifier(self.output_norm(fused))


def embedding_loader(
    split: dict[str, np.ndarray],
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader:
    names = ("baseline_embedding", "lrg_embedding", "rf_embedding", "mt_embedding")
    tensors = [torch.from_numpy(np.asarray(split[name], dtype=np.float32)) for name in names]
    tensors.append(torch.from_numpy(np.asarray(split["labels"], dtype=np.int64)))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.inference_mode()
def evaluate_head(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, np.ndarray]:
    model.eval()
    logits = []
    labels = []
    for batch in loader:
        *parts, target = batch
        logits.append(model(*(part.to(device) for part in parts)).cpu().numpy())
        labels.append(target.numpy())
    scores = np.concatenate(logits)
    target = np.concatenate(labels)
    return topk(scores, target, 1), topk(scores, target, 5), scores


def train_head(
    method: str,
    model: nn.Module,
    bank: dict[str, dict[str, np.ndarray]],
    output: Path,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = embedding_loader(bank["train"], batch_size, shuffle=True)
    val_loader = embedding_loader(bank["val"], batch_size, shuffle=False)
    test_loader = embedding_loader(bank["test"], batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in train_loader:
            *parts, target = batch
            parts = [part.to(device) for part in parts]
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*parts)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            total += len(target)
        val_top1, val_top5, _ = evaluate_head(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / total,
            "val_top1": val_top1,
            "val_top5": val_top5,
        })
        selection = (val_top1, val_top5)
        if best is None or selection > best["selection"]:
            best = {
                "selection": selection,
                "epoch": epoch,
                "state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best is not None
    model.load_state_dict(best["state"])
    test_top1, test_top5, test_logits = evaluate_head(model, test_loader, device)
    checkpoint = output / f"{method}_best_by_val.pth"
    torch.save(best["state"], checkpoint)
    np.save(output / f"{method}_test_logits.npy", test_logits)
    return {
        "method": method,
        "space": "embeddings",
        "selection": "maximum validation Top-1; validation Top-5 tie-break",
        "best_epoch": best["epoch"],
        "val_top1": best["selection"][0],
        "val_top5": best["selection"][1],
        "test_top1": test_top1,
        "test_top5": test_top5,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint),
        "history": history,
    }


def equal_probability_average(split: dict[str, np.ndarray]) -> np.ndarray:
    probabilities = []
    for name in ("baseline_logits", "lrg_logits", "rf_logits", "mt_logits"):
        logits = split[name]
        shifted = logits - logits.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probabilities.append(probability / probability.sum(axis=1, keepdims=True))
    return sum(probabilities) / len(probabilities)


def select_logit_weights(bank: dict[str, dict[str, np.ndarray]]) -> tuple[tuple[float, ...], dict]:
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    names = ("baseline_logits", "lrg_logits", "rf_logits", "mt_logits")
    labels = bank["val"]["labels"]
    candidates = []
    for baseline_weight in (0.5, 1.0):
        for lrg_weight in grid:
            for rf_weight in grid:
                for mt_weight in grid:
                    weights = (baseline_weight, lrg_weight, rf_weight, mt_weight)
                    logits = sum(weight * bank["val"][name] for weight, name in zip(weights, names))
                    candidates.append({
                        "weights": weights,
                        "val_top1": topk(logits, labels, 1),
                        "val_top5": topk(logits, labels, 5),
                    })
    candidates.sort(
        key=lambda row: (
            -row["val_top1"],
            -row["val_top5"],
            sum(row["weights"]),
            row["weights"],
        )
    )
    best = candidates[0]
    return best["weights"], best


def decision_rows(bank: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    labels = bank["test"]["labels"]
    val_labels = bank["val"]["labels"]
    rows = []
    for name, label in (
        ("baseline_logits", "Baseline only"),
        ("lrg_logits", "LRG only"),
        ("rf_logits", "RF only"),
        ("mt_logits", "MT only"),
    ):
        rows.append({
            "method": label,
            "space": "logits",
            "selection": "frozen expert reference",
            "val_top1": topk(bank["val"][name], val_labels, 1),
            "val_top5": topk(bank["val"][name], val_labels, 5),
            "test_top1": topk(bank["test"][name], labels, 1),
            "test_top5": topk(bank["test"][name], labels, 5),
        })
    probability = equal_probability_average(bank["test"])
    rows.append({
        "method": "equal_probability_averaging",
        "space": "softmax_probabilities",
        "selection": "no learned or selected weights",
        "val_top1": topk(equal_probability_average(bank["val"]), bank["val"]["labels"], 1),
        "val_top5": topk(equal_probability_average(bank["val"]), bank["val"]["labels"], 5),
        "test_top1": topk(probability, labels, 1),
        "test_top5": topk(probability, labels, 5),
        "weights": [0.25, 0.25, 0.25, 0.25],
    })
    weights, selected = select_logit_weights(bank)
    names = ("baseline_logits", "lrg_logits", "rf_logits", "mt_logits")
    test_logits = sum(weight * bank["test"][name] for weight, name in zip(weights, names))
    rows.append({
        "method": "validation_selected_logit_fusion",
        "space": "logits",
        "selection": "validation Top-1; Top-5 and lower L1 weight tie-breaks",
        "val_top1": selected["val_top1"],
        "val_top5": selected["val_top5"],
        "test_top1": topk(test_logits, labels, 1),
        "test_top5": topk(test_logits, labels, 5),
        "weights": {
            "baseline": weights[0],
            "lrg": weights[1],
            "rf": weights[2],
            "mt": weights[3],
        },
    })
    return rows


def run_dataset(args, dataset: int) -> dict:
    public_checkpoints = Path(args.public_checkpoints)
    paths = paths_for(dataset, public_checkpoints)
    output = Path(args.output_dir) / f"wlasl{dataset}"
    bank, export_manifest = export_frozen_bank(
        dataset,
        paths,
        output / "bank",
        args.export_batch_size,
        args.force_export,
    )
    dims = [
        int(bank["train"]["baseline_embedding"].shape[1]),
        int(bank["train"]["lrg_embedding"].shape[1]),
        int(bank["train"]["rf_embedding"].shape[1]),
        int(bank["train"]["mt_embedding"].shape[1]),
    ]
    rows = decision_rows(bank)
    seed_everything(args.seed)
    concat_model = ConcatFusion(dims, dataset)
    rows.append(train_head(
        "embedding_concatenation",
        concat_model,
        bank,
        output,
        args.seed,
        args.epochs,
        args.patience,
        args.head_batch_size,
    ))
    seed_everything(args.seed)
    residual_model = ResidualFusion(dims, dataset)
    rows.append(train_head(
        "embedding_residual_fusion",
        residual_model,
        bank,
        output,
        args.seed,
        args.epochs,
        args.patience,
        args.head_batch_size,
    ))
    result = {
        "dataset": f"WLASL-{dataset}",
        "seed": args.seed,
        "protocol": {
            "experts": ["Baseline", "LRG", "RF", "MT"],
            "expert_training": "frozen",
            "fusion_head_training_split": "train",
            "selection_split": "validation",
            "test_usage": "single final evaluation",
            "embedding_dims": dims,
            "export": export_manifest,
        },
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "fusion_space_ablation.json").write_text(
        json.dumps(result, indent=2),
        encoding="ascii",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=int, choices=(100, 300), default=[100, 300])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--public-checkpoints", default=str(DEFAULT_PUBLIC_CHECKPOINTS))
    parser.add_argument("--output-dir", default="diagnostic/fusion_space_ablation")
    parser.add_argument("--export-batch-size", type=int, default=32)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        results = []
        for dataset in args.datasets:
            path = output / f"wlasl{dataset}" / "fusion_space_ablation.json"
            if not path.exists():
                raise FileNotFoundError(path)
            results.append(json.loads(path.read_text()))
    else:
        results = [run_dataset(args, dataset) for dataset in args.datasets]
    summary_rows = []
    for result in results:
        for row in result["rows"]:
            summary_rows.append({
                "dataset": result["dataset"],
                "seed": result["seed"],
                "method": row["method"],
                "space": row["space"],
                "val_top1": row["val_top1"],
                "val_top5": row["val_top5"],
                "test_top1": row["test_top1"],
                "test_top5": row["test_top5"],
                "weights": json.dumps(row.get("weights", "")),
            })
    with (output / "fusion_space_ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output / "fusion_space_ablation.json").write_text(
        json.dumps({"results": results}, indent=2),
        encoding="ascii",
    )
    print(f"Wrote controlled fusion-space ablation to {output}")


if __name__ == "__main__":
    main()
