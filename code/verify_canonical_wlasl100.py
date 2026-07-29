import argparse
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))

from wlasl_train_old_localglobal_sweep_subset import (
    OldLocalGlobalSweepArcFace,
    load_subset_raw_for_source,
)
from wlasl_train_felf_slr_subset import (
    encode_feature_parts,
    FELFSLR,
    parse_args as parse_felf_args,
    append_clean_missingness_masks,
)
from wlasl_train_local_global_arcface_subset import (
    collect_logits,
    make_loader,
)
from train_morph_traj_expert import (
    encode_factors,
    MorphTrajExpert,
)

class MockArgs:
    def __init__(self, json_path):
        with open(json_path) as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(self, k, v)
        if not hasattr(self, "limit_samples"):
            self.limit_samples = 0
        if not hasattr(self, "force_cache"):
            self.force_cache = False
        if not hasattr(self, "cache_workers"):
            self.cache_workers = 0
        if not hasattr(self, "feature_version"):
            self.feature_version = "v1"

parser = argparse.ArgumentParser(description="Direct checkpoint-only reproduction of canonical WLASL-100 FELF-SLR.")
parser.add_argument("--data-dir", default="WLASL2000_Data")
parser.add_argument("--json-path", default="WLASL_Full/WLASL_v0.3.json")
parser.add_argument("--feature-cache", default="rework_model/cache/wlasl100_old_B6_fair_seed1")
parser.add_argument("--felf-cache", default="rework_model/cache/wlasl100_old_B6")
parser.add_argument("--mt-cache", default="rework_model/cache/wlasl100_morphtraj_fair_seed1")
parser.add_argument("--checkpoint-dir", default=str(REPO_ROOT / "checkpoints" / "canonical"))
parser.add_argument(
    "--output-dir",
    default=str(REPO_ROOT / "diagnostic" / "canonical_checkpoint_reproduction" / "wlasl100"),
)
cli = parser.parse_args()

config_dir = REPO_ROOT / "diagnostic" / "canonical_checkpoint_reproduction" / "wlasl100" / "configs"
b6_args = MockArgs(config_dir / "baseline.json")
b6_args.cache_dir = cli.feature_cache
b6_args.data_dir = cli.data_dir
b6_args.json_path = cli.json_path
mt_args = MockArgs(config_dir / "mt.json")
mt_args.cache_dir = cli.mt_cache
felf_diag = MockArgs(config_dir / "felf.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Loading splits (labels only)...")
splits, actions, _ = load_subset_raw_for_source(
    cli.data_dir, cli.json_path, 100, b6_args.action_source, 0, labels_only=True
)
y_test = np.asarray(splits["test"]["labels"], dtype=np.int64)


# 4. Run B6 Baseline inference on-the-fly
print("Loading B6 Baseline model...")
left_te_b6, right_te_b6, global_te_b6 = encode_feature_parts(
    splits["test"]["raw"], "test", b6_args.cache_dir, b6_args.feature_kind, force=False
)
dims_b6 = (left_te_b6.shape[-1], right_te_b6.shape[-1], global_te_b6.shape[-1])
b6_model = OldLocalGlobalSweepArcFace(
    left_inp=dims_b6[0],
    right_inp=dims_b6[1],
    global_inp=dims_b6[2],
    num_classes=len(actions),
    left_branch_dim=getattr(b6_args, "left_branch_dim", 96),
    right_branch_dim=getattr(b6_args, "right_branch_dim", 96),
    global_branch_dim=getattr(b6_args, "global_branch_dim", 96),
    conv_kernel=getattr(b6_args, "conv_kernel", 3),
    conv_layers=getattr(b6_args, "conv_layers", 1),
    num_layers=getattr(b6_args, "num_layers", 2),
    num_heads=getattr(b6_args, "num_heads", 4),
    ff_dim=getattr(b6_args, "ff_dim", 768),
    dropout=getattr(b6_args, "dropout", 0.3),
    pool_mode=getattr(b6_args, "pool_mode", "mean"),
    temporal_head=getattr(b6_args, "temporal_head", "transformer"),
    lite_kernel=getattr(b6_args, "lite_kernel", 5),
    scale=getattr(b6_args, "scale", 16.0),
    factorization_mode=getattr(b6_args, "factorization_mode", "none"),
    static_motion_gate_bias=getattr(b6_args, "static_motion_gate_bias", -1.5),
    relation_dim=getattr(b6_args, "relation_dim", 96),
    use_global_gate=getattr(b6_args, "use_global_gate", False),
    global_gate_hidden_dim=getattr(b6_args, "global_gate_hidden_dim", 64),
    global_gate_bias=getattr(b6_args, "global_gate_bias", 2.0),
    use_stream_dropout=getattr(b6_args, "use_stream_dropout", False),
    stream_dropout_left=getattr(b6_args, "stream_dropout_left", 0.05),
    stream_dropout_right=getattr(b6_args, "stream_dropout_right", 0.15),
    stream_dropout_global=getattr(b6_args, "stream_dropout_global", 0.15),
    use_reliability_fusion=getattr(b6_args, "use_reliability_fusion", False),
    reliability_hidden_dim=getattr(b6_args, "reliability_hidden_dim", 64),
    reliability_bias=getattr(b6_args, "reliability_bias", 0.0),
    use_residual_reliability_fusion=getattr(b6_args, "use_residual_reliability_fusion", False),
    residual_rf_hidden_dim=getattr(b6_args, "residual_rf_hidden_dim", 64),
    residual_rf_beta_init=getattr(b6_args, "residual_rf_beta_init", 0.0),
    residual_rf_beta_mode=getattr(b6_args, "residual_rf_beta_mode", "learnable"),
    use_lrg_residual=getattr(b6_args, "use_lrg_residual", False),
    lrg_residual_gamma_init=getattr(b6_args, "lrg_residual_gamma_init", 0.0),
    lrg_residual_gamma_mode=getattr(b6_args, "lrg_residual_gamma_mode", "learnable"),
    use_branch_gates=getattr(b6_args, "use_branch_gates", False),
    branch_gate_hidden_dim=getattr(b6_args, "branch_gate_hidden_dim", 64),
    branch_gate_bias=getattr(b6_args, "branch_gate_bias", 2.0),
).to(device)

b6_ckpt = str(Path(cli.checkpoint_dir) / "wlasl100_baseline.pth")
print(f"Loading B6 checkpoint: {b6_ckpt}")
b6_model.load_state_dict(torch.load(b6_ckpt, map_location=device, weights_only=True))
b6_model.eval()

b6_loader = make_loader(left_te_b6, right_te_b6, global_te_b6, y_test, batch_size=32)
b6_logits = []
with torch.no_grad():
    for left, right, glob, _ in b6_loader:
        logits = b6_model(left.to(device), right.to(device), glob.to(device))
        b6_logits.append(logits.cpu().numpy())
b6_logits = np.concatenate(b6_logits, axis=0)
print(f"B6 Baseline Checkpoint Accuracy (Inference): {(b6_logits.argmax(axis=1) == y_test).mean() * 100.0:.2f}%")


# 5. Run MorphTraj Expert inference on-the-fly
print("Loading MorphTraj Expert model...")
rectify = getattr(mt_args, "rectify_hands", True)
feature_version = getattr(mt_args, "feature_version", "v1")
cache_leaf = "morph_traj" if feature_version == "v1" else f"morph_traj_{feature_version}"
mt_cache_path = Path(mt_args.cache_dir) / cache_leaf
morph_te, traj_te, orient_te = encode_factors(
    splits["test"]["raw"], mt_cache_path, "test", force=False, rectify_hands=rectify, feature_version=feature_version
)

conditioning = getattr(mt_args, "conditioning", "none")
mt_model = MorphTrajExpert(
    morph_dim=morph_te.shape[-1],
    traj_dim=traj_te.shape[-1],
    orient_dim=orient_te.shape[-1],
    num_classes=len(actions),
    branch_dim=mt_args.branch_dim,
    dropout=mt_args.dropout,
    conditioning=conditioning,
).to(device)

mt_ckpt = str(Path(cli.checkpoint_dir) / "wlasl100_mt.pth")
print(f"Loading MT checkpoint: {mt_ckpt}")
mt_model.load_state_dict(torch.load(mt_ckpt, map_location=device, weights_only=True))
mt_model.eval()

class MTDataset(Dataset):
    def __init__(self, m, t, o, labels):
        self.m = m
        self.t = t
        self.o = o
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.asarray(self.m[idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.t[idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.o[idx], dtype=np.float32)),
            self.labels[idx],
        )

mt_loader = DataLoader(MTDataset(morph_te, traj_te, orient_te, y_test), batch_size=32, shuffle=False)
mt_logits = []
with torch.no_grad():
    for m, t, o, _ in mt_loader:
        logits = mt_model(m.to(device), t.to(device), o.to(device))["fused"]
        mt_logits.append(logits.cpu().numpy())
mt_logits = np.concatenate(mt_logits, axis=0)
print(f"MT Expert Checkpoint Accuracy (Inference): {(mt_logits.argmax(axis=1) == y_test).mean() * 100.0:.2f}%")


# 6. Run canonical Stage-1 FELF inference.
print("Loading FELF model...")
# FELF arguments parsed mock-style
sys_argv_backup = sys.argv
sys.argv = [
    "export_felf_checkpoint_logits",
    "--num-glosses", "100",
    "--action-source", felf_diag.action_source,
    "--feature-kind", felf_diag.feature_kind,
    "--cache-dir", cli.felf_cache,
    "--no-amp",
    "--lrg-logit-weight", str(felf_diag.lrg_logit_weight),
    "--rf-logit-weight", str(felf_diag.rf_logit_weight),
    "--lrg-residual-gamma-init", str(felf_diag.lrg_residual_gamma_init),
    "--lrg-residual-gamma-mode", felf_diag.lrg_residual_gamma_mode,
    "--rf-residual-beta-init", str(felf_diag.rf_residual_beta_init),
    "--rf-residual-beta-mode", felf_diag.rf_residual_beta_mode,
]
if getattr(felf_diag, "use_lrg_head", True): sys.argv.append("--use-lrg-head")
if getattr(felf_diag, "use_rf_head", True): sys.argv.append("--use-rf-head")
if getattr(felf_diag, "normalize_fused_logits", True): sys.argv.append("--normalize-fused-logits")
if getattr(felf_diag, "freeze_main", False): sys.argv.append("--freeze-main")
try:
    felf_args = parse_felf_args()
finally:
    sys.argv = sys_argv_backup

left_te_felf, right_te_felf, global_te_felf = encode_feature_parts(
    splits["test"]["raw"], "test", felf_args.cache_dir, felf_args.feature_kind, force=False
)
if felf_args.append_missingness_masks:
    global_te_felf = append_clean_missingness_masks(global_te_felf)

dims_felf = (left_te_felf.shape[-1], right_te_felf.shape[-1], global_te_felf.shape[-1])
felf_model = FELFSLR(dims_felf, len(actions), felf_args).to(device)

felf_ckpt = str(Path(cli.checkpoint_dir) / "wlasl100_felf.pth")
print(f"Loading FELF checkpoint: {felf_ckpt}")
felf_model.load_state_dict(torch.load(felf_ckpt, map_location=device, weights_only=True), strict=False)
felf_model.eval()

felf_loader = make_loader(left_te_felf, right_te_felf, global_te_felf, y_test, batch_size=32)
felf_logits = collect_logits(felf_model, felf_loader, device, amp=False)
print(f"FELF Checkpoint Accuracy (Inference): {(felf_logits.argmax(axis=1) == y_test).mean() * 100.0:.2f}%")


# 7. Compute Fusions
print("\n--- INFERENCE-PROVEN FUSION RESULTS ---")
fused_logits = 0.5 * b6_logits + 1.0 * felf_logits + 0.5 * mt_logits
fused_acc = (fused_logits.argmax(axis=1) == y_test).mean() * 100.0
print(f"Weighted Logit Fusion Accuracy: {fused_acc:.4f}%")

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

p_b6 = softmax(b6_logits)
p_felf = softmax(felf_logits)
p_mt = softmax(mt_logits)
p_fused = p_b6 * 0.25 + p_felf * 0.5 + p_mt * 0.25
fused_acc_prob = (p_fused.argmax(axis=1) == y_test).mean() * 100.0
print(f"Probability Averaging Fusion Accuracy: {fused_acc_prob:.4f}%")

out_dir = Path(cli.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "labels.npy", y_test)
np.save(out_dir / "baseline_logits.npy", b6_logits)
np.save(out_dir / "felf_logits.npy", felf_logits)
np.save(out_dir / "mt_logits.npy", mt_logits)
np.save(out_dir / "felf_slr_logits.npy", fused_logits)

def metrics(logits):
    top5 = np.argpartition(-logits, kth=4, axis=1)[:, :5]
    per_class_top1 = []
    per_class_top5 = []
    for class_id in np.unique(y_test):
        mask = y_test == class_id
        per_class_top1.append(np.mean(logits[mask].argmax(axis=1) == y_test[mask]))
        per_class_top5.append(np.mean((top5[mask] == y_test[mask, None]).any(axis=1)))
    return {
        "samples": int(len(y_test)),
        "per_instance_top1": float(np.mean(logits.argmax(axis=1) == y_test) * 100.0),
        "per_instance_top5": float(np.mean((top5 == y_test[:, None]).any(axis=1)) * 100.0),
        "per_class_top1": float(np.mean(per_class_top1) * 100.0),
        "per_class_top5": float(np.mean(per_class_top5) * 100.0),
    }

summary = {
    "protocol": "direct checkpoint inference",
    "fusion": {
        "baseline_weight": 0.5,
        "felf_weight": 1.0,
        "mt_weight": 0.5,
    },
    "checkpoints": {
        "baseline": "checkpoints/canonical/wlasl100_baseline.pth",
        "felf": "checkpoints/canonical/wlasl100_felf.pth",
        "mt": "checkpoints/canonical/wlasl100_mt.pth",
    },
    "metrics": {
        "baseline": metrics(b6_logits),
        "felf": metrics(felf_logits),
        "mt": metrics(mt_logits),
        "felf_slr": metrics(fused_logits),
    },
}
with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(f"Saved canonical checkpoint artifacts: {out_dir}")
