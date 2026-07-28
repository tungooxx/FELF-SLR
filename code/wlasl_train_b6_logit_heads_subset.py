"""FELF-SLR / B6 logit-only multi-head experiment for WLASL-N.

FELF-SLR = Factor-Expert Logit Fusion for Skeleton-based Isolated Sign
Language Recognition.

The architecture keeps B6, RF, and LRG representations separate and fuses only
class logits. B6 is the protected retrieval-geometry expert, LRG is the
temporal/phase expert, and RF is the reliability expert.

No embedding concatenation, no residual embedding fusion, no reranking, and no
post-test class correction are used. The fused logits are produced by one model
during the forward pass. FELF-G optionally gates the LRG logit contribution by
global-stream reliability measured from the input global features.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from tqdm import tqdm

from wlasl_train_local_global_arcface import ArcFaceLoss, topk_metrics, mixup_three
from wlasl_train_local_global_arcface_subset import (
    DATA_DIR,
    JSON_PATH,
    SEED,
    CachedAugmentedPartDataset,
    collect_logits,
    collect_probs,
    eval_accuracy,
    make_loader,
)
from wlasl_train_old_localglobal_sweep_subset import (
    EXPECTED_GLOBAL_DIM,
    EXPECTED_LEFT_DIM,
    EXPECTED_RIGHT_DIM,
    OldLocalGlobalSweepArcFace,
    build_cached_feature_augments,
    encode_feature_parts,
    feature_cache_dir,
    feature_cache_ready,
    load_subset_raw_for_source,
    seed_everything,
)
from wlasl_train_streams_arcface import get_class_weights


def make_b6_head(dims, num_classes, args, *, kind: str) -> OldLocalGlobalSweepArcFace:
    common = dict(
        left_inp=dims[0],
        right_inp=dims[1],
        global_inp=dims[2],
        num_classes=num_classes,
        left_branch_dim=args.left_branch_dim,
        right_branch_dim=args.right_branch_dim,
        global_branch_dim=args.global_branch_dim,
        conv_kernel=3,
        conv_layers=1,
        num_layers=2,
        num_heads=4,
        ff_dim=768,
        dropout=args.dropout,
        pool_mode="mean",
        temporal_head="transformer",
        scale=args.scale,
        classifier_type="cosine",
    )
    if kind == "main":
        return OldLocalGlobalSweepArcFace(**common)
    if kind == "lrg":
        return OldLocalGlobalSweepArcFace(
            **common,
            use_lrg_residual=True,
            lrg_residual_gamma_init=args.lrg_residual_gamma_init,
            lrg_residual_gamma_mode=args.lrg_residual_gamma_mode,
        )
    if kind == "rf":
        return OldLocalGlobalSweepArcFace(
            **common,
            use_residual_reliability_fusion=True,
            residual_rf_beta_init=args.rf_residual_beta_init,
            residual_rf_beta_mode=args.rf_residual_beta_mode,
        )
    raise ValueError(kind)


class B6LogitOnlyFusion(nn.Module):
    def __init__(self, dims, num_classes: int, args):
        super().__init__()
        self.main = make_b6_head(dims, num_classes, args, kind="main")
        self.lrg = make_b6_head(dims, num_classes, args, kind="lrg") if args.use_lrg_head else None
        self.rf = make_b6_head(dims, num_classes, args, kind="rf") if args.use_rf_head else None
        self.trainable_fusion_weights = bool(args.trainable_fusion_weights)
        self.trainable_fusion_normalizer = bool(args.trainable_fusion_normalizer)
        self.use_global_reliability_gate = bool(args.use_global_reliability_gate)
        self.global_gate_mode = args.global_gate_mode
        self.global_gate_low = float(args.global_gate_low)
        self.global_gate_high = float(args.global_gate_high)
        if self.trainable_fusion_weights:
            self.lrg_logit_weight_param = nn.Parameter(torch.tensor(float(args.lrg_logit_weight)))
            self.rf_logit_weight_param = nn.Parameter(torch.tensor(float(args.rf_logit_weight)))
        else:
            self.register_buffer("lrg_logit_weight_param", torch.tensor(float(args.lrg_logit_weight)), persistent=False)
            self.register_buffer("rf_logit_weight_param", torch.tensor(float(args.rf_logit_weight)), persistent=False)
        normalizer = 1.0
        if args.normalize_fused_logits:
            normalizer += abs(float(args.lrg_logit_weight)) if self.lrg is not None else 0.0
            normalizer += abs(float(args.rf_logit_weight)) if self.rf is not None else 0.0
        if self.trainable_fusion_normalizer:
            self.fusion_normalizer_param = nn.Parameter(torch.tensor(float(max(normalizer, 1e-6))))
        else:
            self.register_buffer("fusion_normalizer_param", torch.tensor(float(max(normalizer, 1e-6))), persistent=False)

    def fusion_weight_values(self):
        lrg_w = self.lrg_logit_weight_param
        rf_w = self.rf_logit_weight_param
        norm = torch.clamp(self.fusion_normalizer_param.abs(), min=1e-6)
        return lrg_w, rf_w, norm

    def global_reliability_gate(self, global_features):
        if not self.use_global_reliability_gate:
            return None
        reliability = global_features.detach().abs().mean(dim=(1, 2))
        if self.global_gate_mode == "hard":
            gate = (reliability > self.global_gate_low).float()
        elif self.global_gate_mode == "soft":
            denom = max(self.global_gate_high - self.global_gate_low, 1e-8)
            gate = torch.clamp((reliability - self.global_gate_low) / denom, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown global_gate_mode={self.global_gate_mode}")
        return gate.view(-1, 1)

    def forward_all(self, left, right, global_features):
        out = {"main": self.main(left, right, global_features)}
        if self.lrg is not None:
            out["lrg"] = self.lrg(left, right, global_features)
        if self.rf is not None:
            out["rf"] = self.rf(left, right, global_features)
        lrg_weight, rf_weight, fusion_normalizer = self.fusion_weight_values()
        global_gate = self.global_reliability_gate(global_features)
        if global_gate is not None:
            out["global_reliability_gate"] = global_gate
        fused = out["main"]
        if "lrg" in out:
            lrg_term = lrg_weight * out["lrg"]
            if global_gate is not None:
                lrg_term = global_gate * lrg_term
            fused = fused + lrg_term
        if "rf" in out:
            fused = fused + rf_weight * out["rf"]
        if global_gate is not None:
            fusion_normalizer = 1.0 + global_gate * torch.abs(lrg_weight) + (torch.abs(rf_weight) if self.rf is not None else 0.0)
            fusion_normalizer = torch.clamp(fusion_normalizer, min=1e-6)
        out["fused"] = fused / fusion_normalizer
        return out

    def forward(self, left, right, global_features):
        return self.forward_all(left, right, global_features)["fused"]


class FELFSLR(B6LogitOnlyFusion):
    """Named architecture wrapper for Factor-Expert Logit Fusion."""


def freeze_trunk_keep_classifier(expert: nn.Module) -> None:
    """Freeze an expert trunk while leaving only its classifier trainable."""
    for param in expert.parameters():
        param.requires_grad = False
    for param in expert.classifier.parameters():
        param.requires_grad = True


def build_optimizer(model: nn.Module, args):
    head_lr = args.lr if args.head_lr is None else args.head_lr
    fusion_lr = args.lr if args.fusion_lr is None else args.fusion_lr
    head_wd = args.weight_decay if args.head_weight_decay is None else args.head_weight_decay
    groups = []
    fusion_params = []
    head_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in {"lrg_logit_weight_param", "rf_logit_weight_param", "fusion_normalizer_param"}:
            fusion_params.append(param)
        elif ".classifier." in name:
            head_params.append(param)
        else:
            other_params.append(param)
    if other_params:
        groups.append({"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay, "name": "other"})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": head_wd, "name": "heads"})
    if fusion_params:
        groups.append({"params": fusion_params, "lr": fusion_lr, "weight_decay": args.fusion_weight_decay, "name": "fusion"})
    if not groups:
        raise ValueError("No trainable parameters remain after freeze options.")
    return optim.AdamW(groups)


def train_epoch(model, loader, optimizer, criterion, scaler, device, amp, args, desc):
    model.train()
    total_loss = 0.0
    steps = 0
    for left, right, global_features, labels in tqdm(loader, desc=desc, leave=False, mininterval=5.0):
        left = left.to(device)
        right = right.to(device)
        global_features = global_features.to(device)
        labels = labels.to(device)
        if args.mixup_alpha > 0:
            left_m, right_m, global_m, ya, yb, lam = mixup_three(left, right, global_features, labels, alpha=args.mixup_alpha)
        else:
            left_m, right_m, global_m = left, right, global_features
            ya, yb, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            outs = model.forward_all(left_m, right_m, global_m)

            def ce(logits):
                return lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)

            loss = ce(outs["fused"])
            loss = loss + args.main_aux_weight * ce(outs["main"])
            if "lrg" in outs:
                loss = loss + args.lrg_aux_weight * ce(outs["lrg"])
            if "rf" in outs:
                loss = loss + args.rf_aux_weight * ce(outs["rf"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss detected: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


def parse_args():
    p = argparse.ArgumentParser(description="B6 logit-only multi-head WLASL experiment.")
    p.add_argument("--num-glosses", type=int, required=True)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--json-path", default=JSON_PATH)
    p.add_argument("--action-source", choices=["json_first_n", "top_frequency", "train_dirs"], default="json_first_n")
    p.add_argument("--feature-kind", choices=["old"], default="old")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--architecture-name", default="FELF-SLR")
    p.add_argument("--left-branch-dim", type=int, default=96)
    p.add_argument("--right-branch-dim", type=int, default=96)
    p.add_argument("--global-branch-dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--scale", type=float, default=16.0)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--use-lrg-head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-rf-head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lrg-logit-weight", type=float, default=1.0)
    p.add_argument("--rf-logit-weight", type=float, default=0.75)
    p.add_argument("--normalize-fused-logits", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trainable-fusion-weights", action="store_true")
    p.add_argument("--trainable-fusion-normalizer", action="store_true")
    p.add_argument("--use-global-reliability-gate", action="store_true")
    p.add_argument("--global-gate-mode", choices=["hard", "soft"], default="soft")
    p.add_argument("--global-gate-low", type=float, default=1e-8)
    p.add_argument("--global-gate-high", type=float, default=0.7259677648544312)
    p.add_argument("--lrg-residual-gamma-init", type=float, default=0.1)
    p.add_argument("--lrg-residual-gamma-mode", choices=["learnable", "fixed", "sigmoid"], default="learnable")
    p.add_argument("--rf-residual-beta-init", type=float, default=0.25)
    p.add_argument("--rf-residual-beta-mode", choices=["learnable", "fixed", "sigmoid"], default="fixed")
    p.add_argument("--main-aux-weight", type=float, default=0.25)
    p.add_argument("--lrg-aux-weight", type=float, default=0.25)
    p.add_argument("--rf-aux-weight", type=float, default=0.25)
    p.add_argument("--preload-main-checkpoint", default=None)
    p.add_argument("--preload-lrg-checkpoint", default=None)
    p.add_argument("--preload-rf-checkpoint", default=None)
    p.add_argument("--freeze-main", action="store_true")
    p.add_argument("--freeze-lrg", action="store_true")
    p.add_argument("--freeze-rf", action="store_true")
    p.add_argument("--freeze-lrg-trunk", action="store_true")
    p.add_argument("--freeze-rf-trunk", action="store_true")
    p.add_argument("--head-lr", type=float, default=None)
    p.add_argument("--fusion-lr", type=float, default=None)
    p.add_argument("--head-weight-decay", type=float, default=None)
    p.add_argument("--fusion-weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--swa-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--aug-repeats", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    output_prefix = args.output_prefix or f"wlasl{args.num_glosses}_B6_logit_heads"
    cache_dir = args.cache_dir or os.path.join("rework_model", "cache", f"wlasl{args.num_glosses}_old_B6")

    use_labels_only = (not args.force_cache) and feature_cache_ready(cache_dir, args.feature_kind, args.aug_repeats)
    if use_labels_only:
        print(f"Using complete {args.feature_kind} cache at {feature_cache_dir(cache_dir, args.feature_kind)}; loading labels only.")

    splits, actions, _ = load_subset_raw_for_source(
        args.data_dir,
        args.json_path,
        args.num_glosses,
        args.action_source,
        args.limit_samples,
        labels_only=use_labels_only,
    )
    y_tr = np.asarray(splits["train"]["labels"], dtype=np.int64)
    y_val = np.asarray(splits["val"]["labels"], dtype=np.int64)
    y_test = np.asarray(splits["test"]["labels"], dtype=np.int64)
    num_classes = len(actions)

    left_tr, right_tr, global_tr = encode_feature_parts(splits["train"]["raw"], "train", cache_dir, args.feature_kind, force=args.force_cache)
    left_val, right_val, global_val = encode_feature_parts(splits["val"]["raw"], "val", cache_dir, args.feature_kind, force=args.force_cache)
    left_test, right_test, global_test = encode_feature_parts(splits["test"]["raw"], "test", cache_dir, args.feature_kind, force=args.force_cache)
    dims = (int(left_tr.shape[-1]), int(right_tr.shape[-1]), int(global_tr.shape[-1]))
    print(f"old feature dims: left={dims[0]} right={dims[1]} global={dims[2]}")
    if dims != (EXPECTED_LEFT_DIM, EXPECTED_RIGHT_DIM, EXPECTED_GLOBAL_DIM):
        print(f"WARNING expected old dims 165/165/23, got {dims}.")

    left_aug, right_aug, global_aug, aug_y = build_cached_feature_augments(
        splits["train"]["raw"],
        y_tr,
        dims[0],
        dims[1],
        dims[2],
        args.aug_repeats,
        cache_dir,
        args.feature_kind,
        force=args.force_cache,
    )
    train_len = len(y_tr) + (0 if aug_y is None else len(aug_y))
    train_loader = DataLoader(
        CachedAugmentedPartDataset(left_tr, right_tr, global_tr, y_tr, left_aug, right_aug, global_aug, aug_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=train_len >= args.batch_size,
    )
    val_loader = make_loader(left_val, right_val, global_val, y_val, batch_size=args.batch_size)
    test_loader = make_loader(left_test, right_test, global_test, y_test, batch_size=args.batch_size)

    model = FELFSLR(dims, num_classes, args).to(device)
    preload_info = {}
    if args.preload_main_checkpoint:
        state = torch.load(args.preload_main_checkpoint, map_location=device, weights_only=True)
        missing, unexpected = model.main.load_state_dict(state, strict=False)
        preload_info["main"] = {"path": args.preload_main_checkpoint, "missing": len(missing), "unexpected": len(unexpected)}
    if args.preload_lrg_checkpoint and model.lrg is not None:
        state = torch.load(args.preload_lrg_checkpoint, map_location=device, weights_only=True)
        missing, unexpected = model.lrg.load_state_dict(state, strict=False)
        preload_info["lrg"] = {"path": args.preload_lrg_checkpoint, "missing": len(missing), "unexpected": len(unexpected)}
    if args.preload_rf_checkpoint and model.rf is not None:
        state = torch.load(args.preload_rf_checkpoint, map_location=device, weights_only=True)
        missing, unexpected = model.rf.load_state_dict(state, strict=False)
        preload_info["rf"] = {"path": args.preload_rf_checkpoint, "missing": len(missing), "unexpected": len(unexpected)}
    if args.freeze_main:
        for param in model.main.parameters():
            param.requires_grad = False
    if args.freeze_lrg and model.lrg is not None:
        for param in model.lrg.parameters():
            param.requires_grad = False
    if args.freeze_rf and model.rf is not None:
        for param in model.rf.parameters():
            param.requires_grad = False
    if args.freeze_lrg_trunk and model.lrg is not None:
        freeze_trunk_keep_classifier(model.lrg)
    if args.freeze_rf_trunk and model.rf is not None:
        freeze_trunk_keep_classifier(model.rf)
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Startup: {args.architecture_name} "
        f"num_glosses={args.num_glosses} action_source={args.action_source} dims={dims[0]}/{dims[1]}/{dims[2]} "
        f"use_lrg={args.use_lrg_head} lrg_weight={args.lrg_logit_weight} "
        f"use_rf={args.use_rf_head} rf_weight={args.rf_logit_weight} "
        f"normalize_fused_logits={args.normalize_fused_logits} "
        f"global_gate={args.use_global_reliability_gate}/{args.global_gate_mode} "
        f"aux=main:{args.main_aux_weight}/lrg:{args.lrg_aux_weight}/rf:{args.rf_aux_weight} "
        f"mixup_alpha={args.mixup_alpha} params={params} trainable={trainable_params} "
        f"preload={preload_info} freeze_main={args.freeze_main} "
        f"freeze_lrg_trunk={args.freeze_lrg_trunk} freeze_rf_trunk={args.freeze_rf_trunk} "
        f"trainable_fusion_weights={args.trainable_fusion_weights} "
        f"trainable_fusion_normalizer={args.trainable_fusion_normalizer}"
    )

    class_weights = get_class_weights(y_tr, num_classes).to(device)
    criterion = ArcFaceLoss(scale=args.scale, margin=args.margin, cw=class_weights)
    optimizer = build_optimizer(model, args)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    Path("rework_model").mkdir(exist_ok=True)
    pre_path = Path("rework_model") / f"{output_prefix}_single.pth"
    swa_path = Path("rework_model") / f"{output_prefix}_swa.pth"
    best_by_val_path = Path("rework_model") / f"{output_prefix}_best_by_val.pth"

    history = []
    best_val = -1.0
    stale = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, amp, args, f"Epoch {epoch+1}/{args.epochs}")
        val_acc, val_loss = eval_accuracy(model, val_loader, device, criterion=criterion, amp=amp)
        scheduler.step(val_loss)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss}
        print(json.dumps(row))
        history.append(row)
        if val_acc > best_val:
            best_val = val_acc
            stale = 0
            torch.save(model.state_dict(), pre_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(pre_path, map_location=device, weights_only=True))
    pre_probs, _ = collect_probs(model, test_loader, device, amp=amp, tta_passes=0)
    pre_top1, pre_top5 = topk_metrics(pre_probs, y_test)
    pre_val, _ = eval_accuracy(model, val_loader, device, amp=amp)
    selected_name = "pre-SWA"
    selected_state = torch.load(pre_path, map_location=device, weights_only=True)
    selected_metrics = {"val_top1": float(pre_val), "test_top1": float(pre_top1), "test_top5": float(pre_top5)}
    swa_metrics = None

    if args.swa_epochs > 0:
        optimizer = build_optimizer(model, args)
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        for epoch in range(args.swa_epochs):
            _ = train_epoch(model, train_loader, optimizer, criterion, scaler, device, amp, args, f"SWA {epoch+1}/{args.swa_epochs}")
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"  SWA ep {epoch+1}/{args.swa_epochs}")
        torch.save(swa_model.module.state_dict(), swa_path)
        swa_probs, _ = collect_probs(swa_model, test_loader, device, amp=amp, tta_passes=0)
        swa_top1, swa_top5 = topk_metrics(swa_probs, y_test)
        swa_val, _ = eval_accuracy(swa_model, val_loader, device, amp=amp)
        swa_metrics = {"val_top1": float(swa_val), "test_top1": float(swa_top1), "test_top5": float(swa_top5)}
        if swa_val >= pre_val:
            selected_name = "SWA"
            selected_state = torch.load(swa_path, map_location=device, weights_only=True)
            selected_metrics = dict(swa_metrics)

    torch.save(selected_state, best_by_val_path)
    selected_model = FELFSLR(dims, num_classes, args).to(device)
    selected_model.load_state_dict(selected_state)
    run_cache = feature_cache_dir(cache_dir, args.feature_kind) / output_prefix
    run_cache.mkdir(parents=True, exist_ok=True)
    val_logits_path = run_cache / "val_logits.npy"
    test_logits_path = run_cache / "test_logits.npy"
    np.save(val_logits_path, collect_logits(selected_model, val_loader, device, amp=amp))
    np.save(test_logits_path, collect_logits(selected_model, test_loader, device, amp=amp))
    np.save(run_cache / "val_labels.npy", y_val)
    np.save(run_cache / "test_labels.npy", y_test)

    result = {
        "model": args.architecture_name,
        "experiment": "Factor_Expert_Logit_Fusion",
        "architecture": {
            "name": args.architecture_name,
            "description": "Factor-expert logit fusion with protected B6 retrieval expert, LRG temporal-phase expert, and RF reliability expert.",
            "fusion_space": "class_logits",
            "embedding_fusion": False,
            "posthoc_reranking": False,
            "experts": {
                "main": "B6 retrieval geometry expert",
                "lrg": "temporal/phase static-motion expert",
                "rf": "stream reliability expert",
            },
        },
        "num_glosses": int(args.num_glosses),
        "action_source": args.action_source,
        "feature_kind": args.feature_kind,
        "feature_dims": {"left": dims[0], "right": dims[1], "global": dims[2]},
        "params": int(params),
        "trainable_params": int(trainable_params),
        "preload_info": preload_info,
        "freeze_main": bool(args.freeze_main),
        "freeze_lrg": bool(args.freeze_lrg),
        "freeze_rf": bool(args.freeze_rf),
        "use_lrg_head": bool(args.use_lrg_head),
        "use_rf_head": bool(args.use_rf_head),
        "lrg_logit_weight": float(args.lrg_logit_weight),
        "rf_logit_weight": float(args.rf_logit_weight),
        "normalize_fused_logits": bool(args.normalize_fused_logits),
        "use_global_reliability_gate": bool(args.use_global_reliability_gate),
        "global_gate_mode": args.global_gate_mode,
        "global_gate_low": float(args.global_gate_low),
        "global_gate_high": float(args.global_gate_high),
        "lrg_residual_gamma_init": float(args.lrg_residual_gamma_init),
        "lrg_residual_gamma_mode": args.lrg_residual_gamma_mode,
        "rf_residual_beta_init": float(args.rf_residual_beta_init),
        "rf_residual_beta_mode": args.rf_residual_beta_mode,
        "main_aux_weight": float(args.main_aux_weight),
        "lrg_aux_weight": float(args.lrg_aux_weight),
        "rf_aux_weight": float(args.rf_aux_weight),
        "mixup_alpha": float(args.mixup_alpha),
        "pre_swa_metrics": {"val_top1": float(pre_val), "test_top1": float(pre_top1), "test_top5": float(pre_top5)},
        "swa_metrics": swa_metrics,
        "selected_by_val": selected_name,
        "selected_by_val_metrics": selected_metrics,
        "checkpoints": {"pre_swa": str(pre_path), "swa": str(swa_path), "best_by_val": str(best_by_val_path)},
        "logits": {"val_logits": str(val_logits_path), "test_logits": str(test_logits_path)},
        "history": history,
    }
    Path("diagnostic").mkdir(exist_ok=True)
    result_path = Path("diagnostic") / f"{output_prefix}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.architecture_name)
    print(f"  Selected by val: {selected_name}")
    print(f"  Pre-SWA: Val {pre_val:.2f}% Test top1 {pre_top1:.2f}% Test top5 {pre_top5:.2f}%")
    if swa_metrics is not None:
        print(f"  SWA:     Val {swa_metrics['val_top1']:.2f}% Test top1 {swa_metrics['test_top1']:.2f}% Test top5 {swa_metrics['test_top5']:.2f}%")
    print(f"  Params: {params:,}")
    print(f"  Result JSON: {result_path}")


if __name__ == "__main__":
    main()
