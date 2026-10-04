from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/workspace/local-vlm/SLR/FELF-SLR")
CACHE = ROOT / "diagnostic/rep_arch_paper_data/rep_arch_features_v1/wlasl100/engineered"
OUT = ROOT / "diagnostic/engineered_multiscale_tcn_prototype_smoke.json"

class BranchStem(nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.proj = nn.Linear(din, dout)
        self.norm = nn.LayerNorm(dout)
    def forward(self, x):
        return F.gelu(self.norm(self.proj(x)))

class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, channels=256, ff=384, dropout=0.0):
        super().__init__()
        specs = [(3,1),(5,2),(9,4)]
        self.dw = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=k, dilation=d,
                      padding=d*(k-1)//2, groups=channels, bias=False)
            for k,d in specs
        ])
        self.mix = nn.Conv1d(channels*len(specs), channels, kernel_size=1, bias=False)
        self.norm1 = nn.LayerNorm(channels)
        self.ff1 = nn.Linear(channels, ff)
        self.ff2 = nn.Linear(ff, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        y = x.transpose(1,2)
        y = torch.cat([conv(y) for conv in self.dw], dim=1)
        y = self.mix(y).transpose(1,2)
        x = self.norm1(x + self.dropout(F.gelu(y)))
        y = self.ff2(F.gelu(self.ff1(x)))
        return self.norm2(x + self.dropout(y))

class EngineeredTemporalTCN(nn.Module):
    def __init__(self, num_classes=100, channels=256, depth=3):
        super().__init__()
        self.left = BranchStem(165, 96)
        self.right = BranchStem(165, 96)
        self.global_stem = BranchStem(23, 64)
        self.blocks = nn.ModuleList([MultiScaleTemporalBlock(channels, ff=384) for _ in range(depth)])
        self.pool_score = nn.Linear(channels, 1)
        self.class_weight = nn.Parameter(torch.empty(num_classes, channels))
        nn.init.normal_(self.class_weight, std=0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(16.0)))
    def forward_features(self, l, r, g):
        x = torch.cat([self.left(l), self.right(r), self.global_stem(g)], dim=-1)
        for blk in self.blocks:
            x = blk(x)
        attn = torch.softmax(self.pool_score(x).squeeze(-1), dim=1)
        pooled = (x * attn.unsqueeze(-1)).sum(dim=1)
        return pooled, attn
    def forward(self, l, r, g):
        pooled, _ = self.forward_features(l,r,g)
        z = F.normalize(pooled, dim=-1)
        w = F.normalize(self.class_weight, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * z @ w.t()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def approximate_macs(T=40, C=256, depth=3):
    stems = T*(165*96 + 165*96 + 23*64)
    per_block_dw = T*C*(3+5+9)
    per_block_mix = T*(3*C)*C
    per_block_ff = T*(C*384 + 384*C)
    pool = T*C
    classifier = C*100
    return {
        "stems": int(stems),
        "per_block_depthwise": int(per_block_dw),
        "per_block_pointwise": int(per_block_mix),
        "per_block_ffn": int(per_block_ff),
        "pool_and_classifier": int(pool+classifier),
        "total": int(stems + depth*(per_block_dw+per_block_mix+per_block_ff) + pool + classifier)
    }

def load_train_batch(n=8):
    l = np.load(CACHE/"train_l.npy", mmap_mode="r")
    r = np.load(CACHE/"train_r.npy", mmap_mode="r")
    g = np.load(CACHE/"train_g.npy", mmap_mode="r")
    y = np.load(CACHE/"train_y.npy", mmap_mode="r")
    assert l.shape[1:] == (40,165) and r.shape[1:] == (40,165) and g.shape[1:] == (40,23)
    return tuple(torch.from_numpy(np.array(a[:n], dtype=np.float32)) for a in (l,r,g)) + (torch.from_numpy(np.array(y[:n], dtype=np.int64)),)

def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(20261002)
    l,r,g,y = load_train_batch(8)
    model = EngineeredTemporalTCN(num_classes=100, channels=256, depth=3)
    model.train()
    logits = model(l,r,g)
    assert logits.shape == (8,100), logits.shape
    loss = F.cross_entropy(logits, y)
    loss.backward()
    finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    grad_params = sum(1 for p in model.parameters() if p.grad is not None)
    params = count_params(model)
    macs = approximate_macs()
    result = {
        "kind":"ENGINEERING_SMOKE_ONLY_NOT_SCIENTIFIC_RESULT",
        "train_split_only":True,
        "validation_accessed":False,
        "test_accessed":False,
        "wlasl300_accessed":False,
        "batch_size":8,
        "input_shapes":{"left":list(l.shape),"right":list(r.shape),"global":list(g.shape)},
        "logits_shape":list(logits.shape),
        "loss_finite":bool(torch.isfinite(loss).item()),
        "loss_value_smoke_only":float(loss.detach()),
        "all_existing_grads_finite":bool(finite),
        "params_with_grad":int(grad_params),
        "trainable_params":int(params),
        "approx_macs_per_clip":macs,
        "approx_gmacs_per_clip":macs["total"]/1e9,
        "architecture":{
            "stems":[165,96,165,96,23,64],
            "temporal_channels":256,
            "blocks":3,
            "kernels":[3,5,9],
            "dilations":[1,2,4],
            "ff_width":384,
            "pool":"learned temporal attention",
            "classifier":"cosine"
        }
    }
    assert finite
    assert params <= 2_000_000, params
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
