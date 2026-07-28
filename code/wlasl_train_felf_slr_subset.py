"""Clean FELF-SLR training script for WLASL-N.

FELF-SLR = Factor-Expert Logit Fusion for skeleton-based isolated sign
language recognition.

This file intentionally keeps only the architecture needed for FELF-SLR:
  - B6 retrieval expert
  - LRG static-motion residual expert
  - RF residual reliability-fusion expert
  - logit-only fusion, optionally FELF-G global reliability gating

Data loading, cache creation, old frame feature extraction, ArcFace loss,
class weights, metrics, and SWA are reused from existing scripts.
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

from wlasl_train_local_global_arcface import (
    ArcFaceLoss,
    BranchStem,
    CosineClassifier,
    PositionalEncoding,
    SEQUENCE_LENGTH,
    mixup_three,
    topk_metrics,
)
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
    PALMNORMVEC_DIM,
    build_cached_feature_augments,
    encode_feature_parts,
    feature_cache_dir,
    feature_cache_ready,
    load_subset_raw_for_source,
    seed_everything,
)
from wlasl_train_streams_arcface import get_class_weights


MASK_CHANNELS = 6
MASK_LEFT_VALID = 0
MASK_RIGHT_VALID = 1
MASK_LEFT_CORRUPTED = 2
MASK_RIGHT_CORRUPTED = 3
MASK_SWAP_FLAG = 4
MASK_PALM_JITTER_FLAG = 5
PALM_SLICE = slice(159, 162)


def temporal_delta(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = torch.zeros_like(x)
    velocity[:, 1:] = x[:, 1:] - x[:, :-1]
    acceleration = torch.zeros_like(x)
    acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
    return velocity, acceleration


def append_clean_missingness_masks(global_features: np.ndarray) -> np.ndarray:
    """Append per-frame reliability mask channels used by Repair E.

    Channels:
      0 left_valid, 1 right_valid, 2 left_corrupted, 3 right_corrupted,
      4 left/right_swap_flag, 5 palm_jitter_flag.
    """

    masks = np.zeros((*global_features.shape[:2], MASK_CHANNELS), dtype=np.float32)
    masks[..., MASK_LEFT_VALID] = 1.0
    masks[..., MASK_RIGHT_VALID] = 1.0
    return np.concatenate([global_features.astype(np.float32), masks], axis=-1)


def _update_mask(global_features: torch.Tensor, channel: int, frame_mask: torch.Tensor, value: float) -> None:
    if global_features.shape[-1] < EXPECTED_GLOBAL_DIM + MASK_CHANNELS:
        return
    idx = EXPECTED_GLOBAL_DIM + channel
    global_features[..., idx] = torch.where(
        frame_mask,
        torch.full_like(global_features[..., idx], float(value)),
        global_features[..., idx],
    )


def apply_missingness_augmentation(
    left: torch.Tensor,
    right: torch.Tensor,
    global_features: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stochastic Repair-E corruption at training time.

    This intentionally operates on old feature streams rather than raw
    landmarks, so it is cheap and compatible with cached features.
    """

    if (not getattr(args, "missingness_aug", False)) or left.shape[0] == 0:
        return left, right, global_features

    left = left.clone()
    right = right.clone()
    global_features = global_features.clone()
    b, t, _ = left.shape
    device = left.device

    # Random individual hand-frame dropout.
    frame_p = float(args.missingness_frame_drop_p)
    if frame_p > 0:
        l_drop = torch.rand(b, t, device=device) < frame_p
        r_drop = torch.rand(b, t, device=device) < frame_p
        left[l_drop] = 0.0
        right[r_drop] = 0.0
        _update_mask(global_features, MASK_LEFT_VALID, l_drop, 0.0)
        _update_mask(global_features, MASK_RIGHT_VALID, r_drop, 0.0)
        _update_mask(global_features, MASK_LEFT_CORRUPTED, l_drop, 1.0)
        _update_mask(global_features, MASK_RIGHT_CORRUPTED, r_drop, 1.0)

    # Short bursts where one hand disappears.
    burst_p = float(args.missingness_burst_p)
    max_burst = max(1, int(args.missingness_burst_max))
    if burst_p > 0:
        for i in range(b):
            for hand in (0, 1):
                if torch.rand((), device=device).item() >= burst_p:
                    continue
                length = int(torch.randint(1, max_burst + 1, (), device=device).item())
                start = int(torch.randint(0, max(1, t - length + 1), (), device=device).item())
                fm = torch.zeros(t, dtype=torch.bool, device=device)
                fm[start : start + length] = True
                if hand == 0:
                    left[i, fm] = 0.0
                    _update_mask(global_features[i : i + 1], MASK_LEFT_VALID, fm.view(1, -1), 0.0)
                    _update_mask(global_features[i : i + 1], MASK_LEFT_CORRUPTED, fm.view(1, -1), 1.0)
                else:
                    right[i, fm] = 0.0
                    _update_mask(global_features[i : i + 1], MASK_RIGHT_VALID, fm.view(1, -1), 0.0)
                    _update_mask(global_features[i : i + 1], MASK_RIGHT_CORRUPTED, fm.view(1, -1), 1.0)

    # Feature-channel dropout approximates low-confidence joints/partial hand
    # tracking failure without requiring raw keypoints.
    chan_p = float(args.missingness_channel_drop_p)
    if chan_p > 0:
        l_chan = (torch.rand(b, t, left.shape[-1], device=device) < chan_p)
        r_chan = (torch.rand(b, t, right.shape[-1], device=device) < chan_p)
        left = left.masked_fill(l_chan, 0.0)
        right = right.masked_fill(r_chan, 0.0)
        l_any = l_chan.any(dim=-1)
        r_any = r_chan.any(dim=-1)
        _update_mask(global_features, MASK_LEFT_CORRUPTED, l_any, 1.0)
        _update_mask(global_features, MASK_RIGHT_CORRUPTED, r_any, 1.0)

    # Palm-normal jitter: only channels 159:162 in the 165-dim old stream.
    jitter_p = float(args.missingness_palm_jitter_p)
    jitter_std = float(args.missingness_palm_jitter_std)
    if jitter_p > 0 and left.shape[-1] >= PALM_SLICE.stop and jitter_std > 0:
        jitter_mask = torch.rand(b, t, device=device) < jitter_p
        for stream in (left, right):
            palm = stream[..., PALM_SLICE]
            noise = torch.randn_like(palm) * jitter_std
            palm2 = palm + noise
            palm2 = palm2 / palm2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            stream[..., PALM_SLICE] = torch.where(jitter_mask.unsqueeze(-1), palm2, palm)
        _update_mask(global_features, MASK_PALM_JITTER_FLAG, jitter_mask, 1.0)
        _update_mask(global_features, MASK_LEFT_CORRUPTED, jitter_mask, 1.0)
        _update_mask(global_features, MASK_RIGHT_CORRUPTED, jitter_mask, 1.0)

    # Rare short left/right identity swap.
    swap_p = float(args.missingness_swap_p)
    max_swap = max(1, int(args.missingness_swap_max))
    if swap_p > 0:
        for i in range(b):
            if torch.rand((), device=device).item() >= swap_p:
                continue
            length = int(torch.randint(1, max_swap + 1, (), device=device).item())
            start = int(torch.randint(0, max(1, t - length + 1), (), device=device).item())
            ltmp = left[i, start : start + length].clone()
            left[i, start : start + length] = right[i, start : start + length]
            right[i, start : start + length] = ltmp
            fm = torch.zeros(t, dtype=torch.bool, device=device)
            fm[start : start + length] = True
            _update_mask(global_features[i : i + 1], MASK_SWAP_FLAG, fm.view(1, -1), 1.0)
            _update_mask(global_features[i : i + 1], MASK_LEFT_CORRUPTED, fm.view(1, -1), 1.0)
            _update_mask(global_features[i : i + 1], MASK_RIGHT_CORRUPTED, fm.view(1, -1), 1.0)

    return left, right, global_features


class StaticMotionStem(nn.Module):
    """Static morphology/locus stem plus gated delta-2 motion stem."""

    def __init__(self, input_dim: int, branch_dim: int, gate_bias: float = -1.5):
        super().__init__()
        self.static_stem = BranchStem(input_dim, branch_dim)
        self.motion_stem = BranchStem(input_dim * 2, branch_dim)
        hidden = max(32, branch_dim // 2)
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, branch_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)
        self.last_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        velocity, acceleration = temporal_delta(x)
        static = self.static_stem(x)
        motion = self.motion_stem(torch.cat([velocity, acceleration], dim=-1))
        summary = torch.cat(
            [
                static.mean(dim=1),
                static.std(dim=1, unbiased=False),
                motion.mean(dim=1),
                motion.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(summary)).unsqueeze(1)
        self.last_gate = gate.detach()
        return static + gate * motion


class TemporalBranchStem(nn.Module):
    """Base BranchStem plus residual depthwise temporal filtering."""

    def __init__(self, input_dim: int, branch_dim: int):
        super().__init__()
        self.base = BranchStem(input_dim, branch_dim)
        self.dw = nn.Conv1d(branch_dim, branch_dim, kernel_size=3, padding=1, groups=branch_dim)
        self.pw = nn.Conv1d(branch_dim, branch_dim, kernel_size=1)
        self.norm = nn.LayerNorm(branch_dim)
        nn.init.zeros_(self.dw.weight)
        nn.init.zeros_(self.dw.bias)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base(x)
        t = self.pw(self.dw(h.transpose(1, 2))).transpose(1, 2)
        return self.norm(h + t)


class GatedBranchStem(nn.Module):
    """Base BranchStem with an input-conditioned per-frame feature gate."""

    def __init__(self, input_dim: int, branch_dim: int):
        super().__init__()
        self.base = BranchStem(input_dim, branch_dim)
        self.gate = nn.Linear(input_dim, branch_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base(x)
        return h * torch.sigmoid(self.gate(x))


class ResidualBranchStem(nn.Module):
    """Base BranchStem with a learned skip projection from raw descriptors."""

    def __init__(self, input_dim: int, branch_dim: int):
        super().__init__()
        self.base = BranchStem(input_dim, branch_dim)
        self.skip = nn.Linear(input_dim, branch_dim)
        self.norm = nn.LayerNorm(branch_dim)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.base(x) + self.skip(x))


class MotionAwareBranchStem(nn.Module):
    """Stem over static descriptors plus first/second temporal differences."""

    def __init__(self, input_dim: int, branch_dim: int):
        super().__init__()
        self.net = BranchStem(input_dim * 3, branch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        velocity, acceleration = temporal_delta(x)
        return self.net(torch.cat([x, velocity, acceleration], dim=-1))


def make_branch_stem(input_dim: int, branch_dim: int, kind: str) -> nn.Module:
    if kind == "base":
        return BranchStem(input_dim, branch_dim)
    if kind == "temporal":
        return TemporalBranchStem(input_dim, branch_dim)
    if kind == "gated":
        return GatedBranchStem(input_dim, branch_dim)
    if kind == "residual":
        return ResidualBranchStem(input_dim, branch_dim)
    if kind == "motionaware":
        return MotionAwareBranchStem(input_dim, branch_dim)
    raise ValueError(f"Unknown branch stem kind: {kind}")


class ResidualReliabilityFusion(nn.Module):
    """Residual stream reliability injection used by the RF expert."""

    def __init__(
        self,
        left_dim: int,
        right_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
        beta_init: float = 0.25,
        beta_mode: str = "fixed",
    ):
        super().__init__()
        if beta_mode not in {"learnable", "fixed", "sigmoid"}:
            raise ValueError(f"Unknown RF beta mode: {beta_mode}")
        summary_dim = 2 * (left_dim + right_dim + global_dim)
        self.net = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.beta_mode = beta_mode
        self.beta = nn.Parameter(torch.tensor(float(beta_init)), requires_grad=(beta_mode != "fixed"))
        if beta_mode == "sigmoid":
            beta_init = min(max(float(beta_init), 1e-4), 1.0 - 1e-4)
            self.beta.data.fill_(float(np.log(beta_init / (1.0 - beta_init))))
        self.last_weights = None
        self.last_beta = None

    def _beta(self) -> torch.Tensor:
        if self.beta_mode == "sigmoid":
            return torch.sigmoid(self.beta)
        return self.beta

    def forward(self, left: torch.Tensor, right: torch.Tensor, global_features: torch.Tensor):
        def summarize(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)

        summary = torch.cat([summarize(left), summarize(right), summarize(global_features)], dim=-1)
        weights = torch.softmax(self.net(summary), dim=-1)
        beta = self._beta()
        self.last_weights = weights.detach()
        self.last_beta = beta.detach()
        return (
            left + beta * weights[:, 0].view(-1, 1, 1) * left,
            right + beta * weights[:, 1].view(-1, 1, 1) * right,
            global_features + beta * weights[:, 2].view(-1, 1, 1) * global_features,
        )


class LocalGlobalB6Expert(nn.Module):
    """Compact B6-compatible expert with optional LRG or RF additions.

    Module names match the old sweep checkpoints, so existing B6/LRG/RF
    checkpoints can be loaded directly.
    """

    def __init__(
        self,
        left_inp: int,
        right_inp: int,
        global_inp: int,
        num_classes: int,
        left_branch_dim: int = 96,
        right_branch_dim: int = 96,
        global_branch_dim: int = 96,
        dropout: float = 0.2,
        scale: float = 16.0,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 768,
        use_lrg_residual: bool = False,
        lrg_residual_gamma_init: float = 0.25,
        lrg_residual_gamma_mode: str = "fixed",
        use_residual_reliability_fusion: bool = False,
        residual_rf_beta_init: float = 0.25,
        residual_rf_beta_mode: str = "fixed",
        branch_stem_kind: str = "base",
    ):
        super().__init__()
        if lrg_residual_gamma_mode not in {"learnable", "fixed", "sigmoid"}:
            raise ValueError(f"Unknown LRG gamma mode: {lrg_residual_gamma_mode}")
        self.branch_stem_kind = branch_stem_kind
        self.left_stem = make_branch_stem(left_inp, left_branch_dim, branch_stem_kind)
        self.right_stem = make_branch_stem(right_inp, right_branch_dim, branch_stem_kind)
        self.global_stem = make_branch_stem(global_inp, global_branch_dim, branch_stem_kind)
        self.use_lrg_residual = bool(use_lrg_residual)
        self.lrg_residual_gamma_mode = lrg_residual_gamma_mode
        self.lrg_left_stem = None
        self.lrg_right_stem = None
        self.lrg_global_stem = None
        self.lrg_project = None
        self.lrg_gamma = None
        d_model = left_branch_dim + right_branch_dim + global_branch_dim
        if self.use_lrg_residual:
            self.lrg_left_stem = StaticMotionStem(left_inp, left_branch_dim)
            self.lrg_right_stem = StaticMotionStem(right_inp, right_branch_dim)
            self.lrg_global_stem = StaticMotionStem(global_inp, global_branch_dim)
            self.lrg_project = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.lrg_project.weight)
            nn.init.zeros_(self.lrg_project.bias)
            gamma = float(lrg_residual_gamma_init)
            if lrg_residual_gamma_mode == "sigmoid":
                gamma = min(max(gamma, 1e-4), 1.0 - 1e-4)
                gamma = float(np.log(gamma / (1.0 - gamma)))
            self.lrg_gamma = nn.Parameter(torch.tensor(gamma), requires_grad=(lrg_residual_gamma_mode != "fixed"))

        self.residual_reliability_fusion = (
            ResidualReliabilityFusion(
                left_branch_dim,
                right_branch_dim,
                global_branch_dim,
                beta_init=residual_rf_beta_init,
                beta_mode=residual_rf_beta_mode,
            )
            if use_residual_reliability_fusion
            else None
        )
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        self.d_model = d_model
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.res = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.pe = PositionalEncoding(d_model, dropout, SEQUENCE_LENGTH)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.classifier = CosineClassifier(d_model, num_classes, scale)

    def _lrg_gamma(self) -> torch.Tensor:
        if self.lrg_gamma is None:
            return torch.zeros(())
        if self.lrg_residual_gamma_mode == "sigmoid":
            return torch.sigmoid(self.lrg_gamma)
        return self.lrg_gamma

    def forward_features(self, left: torch.Tensor, right: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        raw_left, raw_right, raw_global = left, right, global_features
        left = self.left_stem(left)
        right = self.right_stem(right)
        global_features = self.global_stem(global_features)

        if self.use_lrg_residual:
            lrg = torch.cat(
                [
                    self.lrg_left_stem(raw_left),
                    self.lrg_right_stem(raw_right),
                    self.lrg_global_stem(raw_global),
                ],
                dim=-1,
            )
            base = torch.cat([left, right, global_features], dim=-1)
            mixed = base + self._lrg_gamma() * self.lrg_project(lrg)
            left, right, global_features = torch.split(
                mixed,
                [left.shape[-1], right.shape[-1], global_features.shape[-1]],
                dim=-1,
            )

        if self.residual_reliability_fusion is not None:
            left, right, global_features = self.residual_reliability_fusion(left, right, global_features)

        src = torch.cat([left, right, global_features], dim=-1)
        residual = self.res(src)
        src = self.conv(src.transpose(1, 2)).transpose(1, 2)
        src = self.norm(src + residual)
        src = self.proj(src)
        src_t = self.pe(src.transpose(0, 1))
        mem = self.encoder(src_t)
        mem = self.out_norm(mem + src_t)
        return self.drop(mem.mean(dim=0))

    def forward(self, left: torch.Tensor, right: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(left, right, global_features))


def make_expert(dims, num_classes: int, args, *, kind: str) -> LocalGlobalB6Expert:
    common = dict(
        left_inp=dims[0],
        right_inp=dims[1],
        global_inp=dims[2],
        num_classes=num_classes,
        left_branch_dim=args.left_branch_dim,
        right_branch_dim=args.right_branch_dim,
        global_branch_dim=args.global_branch_dim,
        dropout=args.dropout,
        scale=args.scale,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        branch_stem_kind=args.branch_stem_kind,
    )
    if kind == "main":
        return LocalGlobalB6Expert(**common)
    if kind == "lrg":
        return LocalGlobalB6Expert(
            **common,
            use_lrg_residual=True,
            lrg_residual_gamma_init=args.lrg_residual_gamma_init,
            lrg_residual_gamma_mode=args.lrg_residual_gamma_mode,
        )
    if kind == "rf":
        return LocalGlobalB6Expert(
            **common,
            use_residual_reliability_fusion=True,
            residual_rf_beta_init=args.rf_residual_beta_init,
            residual_rf_beta_mode=args.rf_residual_beta_mode,
        )
    raise ValueError(kind)


class FELFSLR(nn.Module):
    """FELF-SLR logit-fusion model."""

    def __init__(self, dims, num_classes: int, args):
        super().__init__()
        self.main = make_expert(dims, num_classes, args, kind="main")
        self.lrg = make_expert(dims, num_classes, args, kind="lrg") if args.use_lrg_head else None
        self.rf = make_expert(dims, num_classes, args, kind="rf") if args.use_rf_head else None
        self.trainable_fusion_weights = bool(args.trainable_fusion_weights)
        self.trainable_fusion_normalizer = bool(args.trainable_fusion_normalizer)
        self.use_global_reliability_gate = bool(args.use_global_reliability_gate)
        self.global_gate_mode = args.global_gate_mode
        self.global_gate_low = float(args.global_gate_low)
        self.global_gate_high = float(args.global_gate_high)
        self.use_adaptive_router = bool(args.use_adaptive_router)
        self.trainable_adaptive_router = bool(args.trainable_adaptive_router)
        router_values = {
            "entropy": float(args.router_entropy_weight),
            "uncertainty": float(args.router_uncertainty_weight),
            "bias": float(args.router_bias),
        }
        if self.trainable_adaptive_router:
            self.router_entropy_weight = nn.Parameter(torch.tensor(router_values["entropy"], dtype=torch.float32))
            self.router_uncertainty_weight = nn.Parameter(torch.tensor(router_values["uncertainty"], dtype=torch.float32))
            self.router_bias = nn.Parameter(torch.tensor(router_values["bias"], dtype=torch.float32))
        else:
            self.register_buffer("router_entropy_weight", torch.tensor(router_values["entropy"], dtype=torch.float32), persistent=False)
            self.register_buffer("router_uncertainty_weight", torch.tensor(router_values["uncertainty"], dtype=torch.float32), persistent=False)
            self.register_buffer("router_bias", torch.tensor(router_values["bias"], dtype=torch.float32), persistent=False)
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
        return (
            self.lrg_logit_weight_param,
            self.rf_logit_weight_param,
            torch.clamp(self.fusion_normalizer_param.abs(), min=1e-6),
        )

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

    def adaptive_router_gate(self, main_logits: torch.Tensor) -> torch.Tensor | None:
        if not self.use_adaptive_router:
            return None
        probs = torch.softmax(main_logits.detach(), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / torch.log(torch.tensor(float(probs.shape[-1]), device=probs.device))
        top2 = torch.topk(probs, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        uncertainty = 1.0 - margin
        gate = torch.sigmoid(
            self.router_entropy_weight * entropy
            + self.router_uncertainty_weight * uncertainty
            + self.router_bias
        )
        return gate.view(-1, 1)

    def forward_all(self, left, right, global_features):
        out = {"main": self.main(left, right, global_features)}
        if self.lrg is not None:
            out["lrg"] = self.lrg(left, right, global_features)
        if self.rf is not None:
            out["rf"] = self.rf(left, right, global_features)
        lrg_weight, rf_weight, normalizer = self.fusion_weight_values()
        global_gate = self.global_reliability_gate(global_features)
        if global_gate is not None:
            out["global_reliability_gate"] = global_gate
        router_gate = self.adaptive_router_gate(out["main"])
        if router_gate is not None:
            out["adaptive_router_gate"] = router_gate
        fused = out["main"]
        if "lrg" in out:
            lrg_term = lrg_weight * out["lrg"]
            if global_gate is not None:
                lrg_term = global_gate * lrg_term
            if router_gate is not None:
                lrg_term = router_gate * lrg_term
            fused = fused + lrg_term
        if "rf" in out:
            rf_term = rf_weight * out["rf"]
            if router_gate is not None:
                rf_term = router_gate * rf_term
            fused = fused + rf_term
        if global_gate is not None or router_gate is not None:
            lrg_gate = 1.0
            rf_gate = 1.0
            if global_gate is not None:
                lrg_gate = global_gate
            if router_gate is not None:
                lrg_gate = lrg_gate * router_gate
                rf_gate = router_gate
            normalizer = 1.0
            if self.lrg is not None:
                normalizer = normalizer + lrg_gate * torch.abs(lrg_weight)
            if self.rf is not None:
                normalizer = normalizer + rf_gate * torch.abs(rf_weight)
            normalizer = torch.clamp(normalizer, min=1e-6)
        out["fused"] = fused / normalizer
        return out

    def forward(self, left, right, global_features):
        return self.forward_all(left, right, global_features)["fused"]


def freeze_trunk_keep_classifier(expert: nn.Module) -> None:
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
        if name in {
            "lrg_logit_weight_param",
            "rf_logit_weight_param",
            "fusion_normalizer_param",
            "router_entropy_weight",
            "router_uncertainty_weight",
            "router_bias",
        }:
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
        left_m, right_m, global_m = apply_missingness_augmentation(left_m, right_m, global_m, args)

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
    p = argparse.ArgumentParser(description="Clean FELF-SLR WLASL experiment.")
    p.add_argument("--num-glosses", type=int, required=True)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--json-path", default=JSON_PATH)
    p.add_argument("--action-source", choices=["json_first_n", "top_frequency", "train_dirs"], default="json_first_n")
    p.add_argument("--feature-kind", choices=["old", "old_palmnormvec"], default="old")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--architecture-name", default="FELF-SLR")
    p.add_argument("--left-branch-dim", type=int, default=96)
    p.add_argument("--right-branch-dim", type=int, default=96)
    p.add_argument("--global-branch-dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=768)
    p.add_argument("--branch-stem-kind", choices=["base", "temporal", "gated", "residual", "motionaware"], default="base")
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
    p.add_argument("--use-adaptive-router", action="store_true")
    p.add_argument("--trainable-adaptive-router", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--router-entropy-weight", type=float, default=2.0)
    p.add_argument("--router-uncertainty-weight", type=float, default=2.0)
    p.add_argument("--router-bias", type=float, default=-2.0)
    p.add_argument("--use-global-reliability-gate", action="store_true")
    p.add_argument("--global-gate-mode", choices=["hard", "soft"], default="soft")
    p.add_argument("--global-gate-low", type=float, default=1e-8)
    p.add_argument("--global-gate-high", type=float, default=0.7259677648544312)
    p.add_argument("--lrg-residual-gamma-init", type=float, default=0.25)
    p.add_argument("--lrg-residual-gamma-mode", choices=["learnable", "fixed", "sigmoid"], default="fixed")
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
    p.add_argument("--augment-mode", default="fixed")
    p.add_argument("--append-missingness-masks", action="store_true")
    p.add_argument("--missingness-aug", action="store_true")
    p.add_argument("--missingness-frame-drop-p", type=float, default=0.015)
    p.add_argument("--missingness-burst-p", type=float, default=0.08)
    p.add_argument("--missingness-burst-max", type=int, default=5)
    p.add_argument("--missingness-channel-drop-p", type=float, default=0.002)
    p.add_argument("--missingness-palm-jitter-p", type=float, default=0.04)
    p.add_argument("--missingness-palm-jitter-std", type=float, default=0.08)
    p.add_argument("--missingness-swap-p", type=float, default=0.01)
    p.add_argument("--missingness-swap-max", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def load_expert_checkpoint(expert: nn.Module, path: str, device):
    state = torch.load(path, map_location=device, weights_only=True)
    current = expert.state_dict()
    compatible = {}
    skipped_shape = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) != tuple(value.shape):
            skipped_shape.append({"key": key, "checkpoint": list(value.shape), "model": list(current[key].shape)})
            continue
        compatible[key] = value
    missing, unexpected = expert.load_state_dict(compatible, strict=False)
    return {
        "path": path,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "skipped_shape": skipped_shape,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    output_prefix = args.output_prefix or f"wlasl{args.num_glosses}_FELF_SLR"
    cache_dir = args.cache_dir or os.path.join("rework_model", "cache", f"wlasl{args.num_glosses}_old_B6")

    aug_ready = feature_cache_ready(cache_dir, args.feature_kind, args.aug_repeats)
    if args.augment_mode != "fixed" and args.aug_repeats > 0:
        cache_path = feature_cache_dir(cache_dir, args.feature_kind)
        aug_tag = f"aug_{args.augment_mode}_r{args.aug_repeats}_{args.feature_kind}"
        aug_ready = all(
            (cache_path / f"{aug_tag}_{stream}.npy").exists()
            for stream in ["left", "right", "global", "labels"]
        )
    use_labels_only = (not args.force_cache) and aug_ready
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
    if args.append_missingness_masks:
        global_tr = append_clean_missingness_masks(global_tr)
        global_val = append_clean_missingness_masks(global_val)
        global_test = append_clean_missingness_masks(global_test)
    dims = (int(left_tr.shape[-1]), int(right_tr.shape[-1]), int(global_tr.shape[-1]))
    print(f"old feature dims: left={dims[0]} right={dims[1]} global={dims[2]}")
    expected_dims = (EXPECTED_LEFT_DIM, EXPECTED_RIGHT_DIM, EXPECTED_GLOBAL_DIM)
    if args.feature_kind == "old_palmnormvec":
        expected_dims = (
            EXPECTED_LEFT_DIM + PALMNORMVEC_DIM,
            EXPECTED_RIGHT_DIM + PALMNORMVEC_DIM,
            EXPECTED_GLOBAL_DIM,
        )
    if args.append_missingness_masks:
        expected_dims = (expected_dims[0], expected_dims[1], expected_dims[2] + MASK_CHANNELS)
    if dims != expected_dims:
        print(f"WARNING expected {args.feature_kind} dims {expected_dims}, got {dims}.")

    left_aug, right_aug, global_aug, aug_y = build_cached_feature_augments(
        splits["train"]["raw"],
        y_tr,
        left_tr.shape[-1],
        right_tr.shape[-1],
        EXPECTED_GLOBAL_DIM if args.append_missingness_masks else global_tr.shape[-1],
        args.aug_repeats,
        cache_dir,
        args.feature_kind,
        force=args.force_cache,
        augment_mode=args.augment_mode,
    )
    if args.append_missingness_masks and global_aug is not None:
        global_aug = append_clean_missingness_masks(global_aug)
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
        preload_info["main"] = load_expert_checkpoint(model.main, args.preload_main_checkpoint, device)
    if args.preload_lrg_checkpoint and model.lrg is not None:
        preload_info["lrg"] = load_expert_checkpoint(model.lrg, args.preload_lrg_checkpoint, device)
    if args.preload_rf_checkpoint and model.rf is not None:
        preload_info["rf"] = load_expert_checkpoint(model.rf, args.preload_rf_checkpoint, device)
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
        f"Startup: {args.architecture_name} clean_file=True "
        f"num_glosses={args.num_glosses} action_source={args.action_source} dims={dims[0]}/{dims[1]}/{dims[2]} "
        f"use_lrg={args.use_lrg_head} lrg_weight={args.lrg_logit_weight} "
        f"use_rf={args.use_rf_head} rf_weight={args.rf_logit_weight} "
        f"augment_mode={args.augment_mode} "
        f"repairE_missingness={args.missingness_aug} masks={args.append_missingness_masks} "
        f"branch_stem={args.branch_stem_kind} "
        f"encoder_layers={args.num_layers} heads={args.num_heads} ff_dim={args.ff_dim} "
        f"normalize_fused_logits={args.normalize_fused_logits} "
        f"adaptive_router={args.use_adaptive_router}/trainable={args.trainable_adaptive_router} "
        f"global_gate={args.use_global_reliability_gate}/{args.global_gate_mode} "
        f"aux=main:{args.main_aux_weight}/lrg:{args.lrg_aux_weight}/rf:{args.rf_aux_weight} "
        f"mixup_alpha={args.mixup_alpha} params={params} trainable={trainable_params} "
        f"preload={preload_info}"
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
        "script": "wlasl_train_felf_slr_subset.py",
        "experiment": "Factor_Expert_Logit_Fusion",
        "architecture": {
            "name": args.architecture_name,
            "description": "Clean FELF-SLR file with local B6, LRG, and RF expert definitions.",
            "fusion_space": "class_logits",
            "embedding_fusion": False,
            "posthoc_reranking": False,
            "experts": {
                "main": "B6 retrieval geometry expert",
                "lrg": "static-motion temporal/phase expert",
                "rf": "stream reliability expert",
            },
        },
        "num_glosses": int(args.num_glosses),
        "action_source": args.action_source,
        "feature_kind": args.feature_kind,
        "augment_mode": args.augment_mode,
        "repair_e": {
            "append_missingness_masks": bool(args.append_missingness_masks),
            "missingness_aug": bool(args.missingness_aug),
            "frame_drop_p": float(args.missingness_frame_drop_p),
            "burst_p": float(args.missingness_burst_p),
            "burst_max": int(args.missingness_burst_max),
            "channel_drop_p": float(args.missingness_channel_drop_p),
            "palm_jitter_p": float(args.missingness_palm_jitter_p),
            "palm_jitter_std": float(args.missingness_palm_jitter_std),
            "swap_p": float(args.missingness_swap_p),
            "swap_max": int(args.missingness_swap_max),
        },
        "aug_repeats": int(args.aug_repeats),
        "feature_dims": {"left": dims[0], "right": dims[1], "global": dims[2]},
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "ff_dim": int(args.ff_dim),
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
        "use_adaptive_router": bool(args.use_adaptive_router),
        "trainable_adaptive_router": bool(args.trainable_adaptive_router),
        "router_entropy_weight": float(args.router_entropy_weight),
        "router_uncertainty_weight": float(args.router_uncertainty_weight),
        "router_bias": float(args.router_bias),
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
