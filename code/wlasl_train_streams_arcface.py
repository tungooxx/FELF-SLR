"""Tri-stream benchmark with safer Shape defaults and calibrated val-only fusion.

Defaults:
  - Shape: CE (vanilla ArcFace hurt this stream in prior runs)
  - Motion: ArcFace with a small angular margin
  - D_noshoulder / D_153: loaded from the canonical small-margin ArcFace checkpoint
  - Per-stream temperature scaling on validation
  - Fusion weights tuned on validation only
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, '.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WLASL_DATA_PATH = 'WLASL_Data_Complete'
FACE_INDICES = [0, 2, 5, 9, 10]
SEQUENCE_LENGTH = 40
SEED = 42

DEFAULT_ARCFACE_MARGIN = float(os.environ.get('WLASL_ARCFACE_MARGIN', '0.2'))
SHAPE_LOSS = os.environ.get('WLASL_SHAPE_LOSS', 'ce').lower()
MOTION_LOSS = os.environ.get('WLASL_MOTION_LOSS', 'arcface').lower()
SHAPE_MARGIN = float(os.environ.get('WLASL_SHAPE_MARGIN', '0.1'))
MOTION_MARGIN = float(os.environ.get('WLASL_MOTION_MARGIN', str(DEFAULT_ARCFACE_MARGIN)))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def compute_palm_normal(h, hand):
    w, im, pm = h[0], h[5], h[17]
    v1, v2 = (im - w, pm - w) if hand == 'Right' else (pm - w, im - w)
    n = np.cross(v1, v2)
    nm = np.linalg.norm(n)
    return (n / nm).astype(np.float32) if nm > 1e-8 else np.zeros(3, dtype=np.float32)


def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def extract_D153(vec):
    pose = vec[0:132].reshape(33, 4)
    lh = vec[132:195].reshape(21, 3)
    rh = vec[195:258].reshape(21, 3)
    face = np.stack([pose[i, :3] for i in FACE_INDICES]) - pose[0:1, :3]
    lh_c = lh - lh[0:1]
    rh_c = rh - rh[0:1]
    ls = np.linalg.norm(lh[0] - lh[9])
    rs = np.linalg.norm(rh[0] - rh[9])
    if ls > 1e-6:
        lh_c /= ls
    if rs > 1e-6:
        rh_c /= rs
    ln = compute_palm_normal(lh, 'Left') if np.any(lh != 0) else np.zeros(3, dtype=np.float32)
    rn = compute_palm_normal(rh, 'Right') if np.any(rh != 0) else np.zeros(3, dtype=np.float32)
    return np.concatenate([lh_c.flatten(), rh_c.flatten(), face.flatten(), lh[0], rh[0], ln, rn]).astype(np.float32)


def extract_shape(vec):
    lh = vec[132:195].reshape(21, 3)
    rh = vec[195:258].reshape(21, 3)
    extras = vec[258:]
    lh_c = lh - lh[0:1]
    rh_c = rh - rh[0:1]
    ls = np.linalg.norm(lh[0] - lh[9])
    rs = np.linalg.norm(rh[0] - rh[9])
    if ls > 1e-6:
        lh_c /= ls
    if rs > 1e-6:
        rh_c /= rs
    ln = compute_palm_normal(lh, 'Left') if np.any(lh != 0) else np.zeros(3, dtype=np.float32)
    rn = compute_palm_normal(rh, 'Right') if np.any(rh != 0) else np.zeros(3, dtype=np.float32)
    return np.concatenate([
        lh_c[1:].flatten(),
        rh_c[1:].flatten(),
        ln,
        rn,
        extras[89:105],
        extras[105:119],
        extras[119:199],
        [np.dot(ln, rn)],
    ]).astype(np.float32)


def extract_motion(seq_all):
    frames = []
    for t in range(seq_all.shape[0]):
        vec = seq_all[t]
        pose = vec[0:132].reshape(33, 4)
        lh = vec[132:195].reshape(21, 3)
        rh = vec[195:258].reshape(21, 3)
        extras = vec[258:]
        nose = pose[0, :3]
        l_sh = pose[11, :3]
        r_sh = pose[12, :3]
        l_el = pose[13, :3]
        r_el = pose[14, :3]
        lw = lh[0]
        rw = rh[0]
        lw_n = lw - nose
        rw_n = rw - nose
        lw_ls = lw - l_sh
        rw_rs = rw - r_sh
        h2h = lw - rw
        if t > 0:
            prev = seq_all[t - 1]
            plw = prev[132:135]
            prw = prev[195:198]
            lv = lw - plw
            rv = rw - prw
        else:
            lv = np.zeros(3, dtype=np.float32)
            rv = np.zeros(3, dtype=np.float32)
        ls = np.linalg.norm(lv)
        rs = np.linalg.norm(rv)
        ln = compute_palm_normal(lh, 'Left') if np.any(lh != 0) else np.zeros(3, dtype=np.float32)
        rn = compute_palm_normal(rh, 'Right') if np.any(rh != 0) else np.zeros(3, dtype=np.float32)
        if t > 0:
            plh = seq_all[t - 1, 132:195].reshape(21, 3)
            prh = seq_all[t - 1, 195:258].reshape(21, 3)
            pln = compute_palm_normal(plh, 'Left') if np.any(plh != 0) else np.zeros(3, dtype=np.float32)
            prn = compute_palm_normal(prh, 'Right') if np.any(prh != 0) else np.zeros(3, dtype=np.float32)
            ln_d = ln - pln
            rn_d = rn - prn
        else:
            ln_d = np.zeros(3, dtype=np.float32)
            rn_d = np.zeros(3, dtype=np.float32)
        tips = [4, 8, 12, 16, 20]
        ltd = np.linalg.norm(lh[tips] - lw, axis=1)
        rtd = np.linalg.norm(rh[tips] - rw, axis=1)
        lo = np.mean(ltd)
        ro = np.mean(rtd)
        lsp = np.std(ltd)
        rsp = np.std(rtd)
        lp = np.linalg.norm(lh[4] - lh[8])
        rp = np.linalg.norm(rh[4] - rh[8])
        lc = np.mean(np.linalg.norm(lh[tips] - lh[0:1], axis=1))
        rc = np.mean(np.linalg.norm(rh[tips] - rh[0:1], axis=1))
        le = compute_angle(l_sh, l_el, lw)
        re = compute_angle(r_sh, r_el, rw)
        frames.append(np.concatenate([
            lw_n, rw_n, lw_ls, rw_rs, h2h, lv, rv, [ls, rs], ln, rn, ln_d, rn_d,
            [lo, ro], [lsp, rsp], [lp, rp], [lc, rc], [le, re], extras[0:35], extras[77:79]
        ]).astype(np.float32))
    return np.array(frames, dtype=np.float32)


def time_stretch(seq, factor):
    t, d = seq.shape
    nt = max(2, int(round(t * factor)))
    src, dst = np.arange(t), np.linspace(0, t - 1, nt)
    out = np.empty((nt, d), dtype=seq.dtype)
    for i in range(d):
        out[:, i] = np.interp(dst, src, seq[:, i])
    return out


def time_warp(seq, max_warp=0.15):
    t, d = seq.shape
    cx = np.linspace(0, 1, 4)
    cy = np.clip(cx + np.random.uniform(-max_warp, max_warp, 4), 0, 1)
    cy[0], cy[-1] = 0, 1
    y = np.interp(np.linspace(0, 1, t), cx, cy) * (t - 1)
    out = np.empty_like(seq)
    for i in range(d):
        out[:, i] = np.interp(np.arange(t), y, seq[:, i])
    return out


def random_frame_drop(seq, drop_prob=0.1):
    t, d = seq.shape
    keep = np.random.rand(t) > drop_prob
    if keep.sum() < 2:
        keep[np.random.randint(0, t, size=2)] = True
    kept = seq[keep]
    src, dst = np.linspace(0, 1, kept.shape[0]), np.linspace(0, 1, t)
    out = np.empty_like(seq)
    for i in range(d):
        out[:, i] = np.interp(dst, src, kept[:, i])
    return out


def augment_fixed(seq):
    s = seq.copy()
    if np.random.rand() < 0.7:
        factor = np.random.uniform(0.8, 1.25)
        s = time_stretch(s, factor)
        s = time_stretch(s, SEQUENCE_LENGTH / s.shape[0])
    if np.random.rand() < 0.5:
        s = time_warp(s)
    if np.random.rand() < 0.5:
        s = random_frame_drop(s, np.random.uniform(0.05, 0.2))
    if np.random.rand() < 0.5:
        shift = np.random.randint(-2, 3)
        if shift != 0:
            s = np.roll(s, shift=shift, axis=0)
    if s.shape[0] != SEQUENCE_LENGTH:
        s = time_stretch(s, SEQUENCE_LENGTH / s.shape[0])

    do_aff = np.random.rand() < 0.9
    theta = np.deg2rad(np.random.uniform(-20, 20))
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scale = np.random.uniform(0.9, 1.12)
    trans = np.random.normal(0, np.random.uniform(0, 0.03), (1, 3))
    do_noise = np.random.rand() < 0.9
    noise_std = np.random.uniform(0.005, 0.015)
    do_drop = np.random.rand() < 0.25
    dm_p = np.random.rand(33) < np.random.uniform(0.05, 0.15)
    dm_l = np.random.rand(21) < np.random.uniform(0.05, 0.15)
    dm_r = np.random.rand(21) < np.random.uniform(0.05, 0.15)

    out = s.copy()
    for t in range(SEQUENCE_LENGTH):
        pose = out[t, 0:132].reshape(33, 4)
        lh = out[t, 132:195].reshape(21, 3)
        rh = out[t, 195:258].reshape(21, 3)
        if do_aff:
            pose[:, :2] = (pose[:, :2] - pose[:, :2].mean(0, keepdims=True)) @ r.T * scale + pose[:, :2].mean(0, keepdims=True) + trans[0, :2]
            pose[:, 2] *= scale
            lh[:, :2] = (lh[:, :2] - lh[:, :2].mean(0, keepdims=True)) @ r.T * scale + lh[:, :2].mean(0, keepdims=True) + trans[0, :2]
            lh[:, 2] *= scale
            rh[:, :2] = (rh[:, :2] - rh[:, :2].mean(0, keepdims=True)) @ r.T * scale + rh[:, :2].mean(0, keepdims=True) + trans[0, :2]
            rh[:, 2] *= scale
        if do_noise:
            pose[:, :3] += np.random.normal(0, noise_std, (33, 3))
            lh += np.random.normal(0, noise_std, (21, 3))
            rh += np.random.normal(0, noise_std, (21, 3))
        if do_drop:
            pose[dm_p, :3] = 0.0
            lh[dm_l] = 0.0
            rh[dm_r] = 0.0
        out[t, 0:132] = pose.flatten()
        out[t, 132:195] = lh.flatten()
        out[t, 195:258] = rh.flatten()
    return out


class PositionalEncoding(nn.Module):
    def __init__(self, d, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d // 2])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])


class TemporalBackbone(nn.Module):
    def __init__(self, inp, d=256, nh=4, nl=3, ff=1024, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(inp, d, 3, padding=1)
        self.res = nn.Linear(inp, d)
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, d)
        self.pe = PositionalEncoding(d, dropout, SEQUENCE_LENGTH)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=nh, dim_feedforward=ff, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc, num_layers=nl)
        self.out_norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward_features(self, src):
        res = self.res(src)
        src = src.transpose(1, 2)
        src = self.conv(src)
        src = src.transpose(1, 2)
        src = self.norm(src + res)
        src = self.proj(src)
        src = src.transpose(0, 1)
        src = self.pe(src)
        mem = self.encoder(src)
        mem = self.out_norm(mem + src)
        return self.drop(mem.mean(dim=0))


class LinearModel(nn.Module):
    def __init__(self, inp, nc):
        super().__init__()
        self.backbone = TemporalBackbone(inp)
        self.classifier = nn.Linear(256, nc)

    def forward(self, src):
        return self.classifier(self.backbone.forward_features(src))


class CosineClassifier(nn.Module):
    def __init__(self, d, nc, scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(nc, d))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        w = F.normalize(self.weight, p=2, dim=-1)
        return self.scale * (x @ w.t())


class ArcFaceModel(nn.Module):
    def __init__(self, inp, nc, scale=16.0):
        super().__init__()
        self.backbone = TemporalBackbone(inp)
        self.classifier = CosineClassifier(256, nc, scale)

    def forward(self, src):
        return self.classifier(self.backbone.forward_features(src))


Model = ArcFaceModel


class ArcFaceLoss(nn.Module):
    def __init__(self, scale=16.0, margin=0.2, cw=None):
        super().__init__()
        self.scale = scale
        self.margin = margin
        if cw is not None:
            self.register_buffer('cw', cw)
        else:
            self.cw = None

    def forward(self, logits, labels):
        cos = logits / self.scale
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = F.one_hot(labels, cos.size(1)).float()
        adjusted = torch.cos(theta + one_hot * self.margin)
        return F.cross_entropy(self.scale * adjusted, labels, weight=self.cw, label_smoothing=0.05)


def build_model(inp, nc, loss_type, scale=16.0):
    if loss_type == 'ce':
        return LinearModel(inp, nc).to(device)
    if loss_type == 'arcface':
        return ArcFaceModel(inp, nc, scale=scale).to(device)
    raise ValueError(f'Unsupported loss_type: {loss_type}')


def tensor_loader(x, y, batch_size=32, shuffle=False, drop_last=False):
    return DataLoader(
        TensorDataset(torch.tensor(x), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def collect_logits(model, loader):
    model.eval()
    logits = []
    with torch.no_grad():
        for bx, _ in loader:
            logits.append(model(bx.to(device)).cpu())
    return torch.cat(logits).numpy()


def collect_probs_from_logits(logits, temperature=1.0):
    return torch.softmax(torch.tensor(logits, dtype=torch.float32) / temperature, dim=1).numpy()


def remap_legacy_d_arcface_state(state):
    remapped = {}
    for key, value in state.items():
        if key.startswith('conv1d.'):
            new_key = 'backbone.conv.' + key[len('conv1d.'):]
        elif key.startswith('conv_res.'):
            new_key = 'backbone.res.' + key[len('conv_res.'):]
        elif key.startswith('conv_norm.'):
            new_key = 'backbone.norm.' + key[len('conv_norm.'):]
        elif key.startswith('inp_lin.'):
            new_key = 'backbone.proj.' + key[len('inp_lin.'):]
        elif key.startswith('pe.'):
            new_key = 'backbone.pe.' + key[len('pe.'):]
        elif key.startswith('te.'):
            new_key = 'backbone.encoder.' + key[len('te.'):]
        elif key.startswith('tn.'):
            new_key = 'backbone.out_norm.' + key[len('tn.'):]
        else:
            new_key = key
        remapped[new_key] = value
    return remapped


def fit_temperature(logits, labels):
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    best_t, best_loss = 1.0, float('inf')
    for t in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]:
        loss = F.cross_entropy(logits_t / t, labels_t).item()
        if loss < best_loss:
            best_t, best_loss = t, loss
    return best_t, best_loss


def eval_probs(probs, y, name=''):
    pred = np.argmax(probs, 1)
    top1 = 100 * np.mean(pred == y)
    top5 = 100 * sum(y[i] in np.argsort(probs[i])[-5:] for i in range(len(y))) / len(y)
    if name:
        print(f'  {name:<44} Top-1: {top1:.2f}%  Top-5: {top5:.2f}%')
    return top1, top5


def search_fusion_weights(val_probs, y_val, order):
    best_w, best_acc = None, -1.0
    for w0 in np.arange(0.05, 0.91, 0.05):
        for w1 in np.arange(0.05, 0.91, 0.05):
            w2 = 1.0 - w0 - w1
            if w2 < 0.05:
                continue
            fused = w0 * val_probs[order[0]] + w1 * val_probs[order[1]] + w2 * val_probs[order[2]]
            acc = 100 * np.mean(np.argmax(fused, 1) == y_val)
            if acc > best_acc:
                best_acc = acc
                best_w = (w0, w1, w2)
    return best_w, best_acc


def train_stream(name, fn, is_seq, splits, nc, cw_t, loss_type='arcface', margin=0.2, scale=16.0):
    print(f'\n=== Training {name} ({loss_type}) ===')

    def encode(raw_list):
        if is_seq:
            return np.array([fn(s) for s in raw_list], dtype=np.float32)
        return np.array([np.array([fn(s[t]) for t in range(SEQUENCE_LENGTH)]) for s in raw_list], dtype=np.float32)

    x_tr = encode(splits['train']['raw'])
    x_v = encode(splits['val']['raw'])
    x_te = encode(splits['test']['raw'])
    y_tr = np.array(splits['train']['labels'], dtype=np.int64)
    y_v = np.array(splits['val']['labels'], dtype=np.int64)
    y_te = np.array(splits['test']['labels'], dtype=np.int64)
    dim = x_tr.shape[2]
    print(f'  dim={dim}, train={x_tr.shape}')

    aug_x, aug_y = [], []
    for i in tqdm(range(len(y_tr)), desc=f'Aug-{name}'):
        raw = splits['train']['raw'][i]
        for _ in range(10):
            ar = augment_fixed(raw)
            if is_seq:
                aug_x.append(fn(ar))
            else:
                aug_x.append(np.array([fn(ar[t]) for t in range(SEQUENCE_LENGTH)], dtype=np.float32))
            aug_y.append(y_tr[i])
    x_tr = np.concatenate([x_tr, np.array(aug_x, dtype=np.float32)])
    y_tr = np.concatenate([y_tr, np.array(aug_y, dtype=np.int64)])
    perm = np.random.permutation(len(y_tr))
    x_tr, y_tr = x_tr[perm], y_tr[perm]

    train_loader = tensor_loader(x_tr, y_tr, batch_size=32, shuffle=True, drop_last=True)
    val_loader = tensor_loader(x_v, y_v, batch_size=32)
    test_loader = tensor_loader(x_te, y_te, batch_size=32)

    model = build_model(dim, nc, loss_type, scale=scale)
    if loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(weight=cw_t.to(device), label_smoothing=0.05)
        save_path = f'rework_model/wlasl100_stream_{name}_ce_best.pth'
    else:
        criterion = ArcFaceLoss(scale=scale, margin=margin, cw=cw_t.to(device))
        save_path = f'rework_model/wlasl100_stream_{name}_arcface_best.pth'

    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, patience=5)

    best_va, pat = float('-inf'), 0
    for ep in range(50):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                val_loss += criterion(logits, by).item()
                pred = logits.argmax(1)
                vt += by.size(0)
                vc += (pred == by).sum().item()
        va = 100 * vc / vt
        sch.step(val_loss / len(val_loader))
        if (ep + 1) % 10 == 0:
            print(f'  Ep {ep+1}: Val {va:.2f}%')
        if va > best_va:
            best_va = va
            pat = 0
            torch.save(model.state_dict(), save_path)
        else:
            pat += 1
            if pat >= 12:
                print(f'  Early stop at epoch {ep+1}')
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    val_logits = collect_logits(model, val_loader)
    test_logits = collect_logits(model, test_loader)
    val_probs = collect_probs_from_logits(val_logits)
    test_probs = collect_probs_from_logits(test_logits)
    top1, top5 = eval_probs(test_probs, y_te, f'{name} ({loss_type})')
    return {
        'name': name,
        'dim': dim,
        'loss_type': loss_type,
        'margin': margin,
        'val_logits': val_logits,
        'test_logits': test_logits,
        'val_probs': val_probs,
        'test_probs': test_probs,
        'val_acc': best_va,
        'top1': top1,
        'top5': top5,
        'y_val': y_v,
        'y_test': y_te,
    }


def load_raw():
    td = os.path.join(WLASL_DATA_PATH, 'train')
    actions = sorted([d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))])
    lm = {g: i for i, g in enumerate(actions)}
    splits = {s: {'raw': [], 'labels': []} for s in ['train', 'val', 'test']}
    for sp in ['train', 'val', 'test']:
        sd = os.path.join(WLASL_DATA_PATH, sp)
        if not os.path.exists(sd):
            continue
        for g in sorted(os.listdir(sd)):
            gd = os.path.join(sd, g)
            if not os.path.isdir(gd) or g not in lm:
                continue
            for v in sorted(os.listdir(gd)):
                vd = os.path.join(gd, v)
                if not os.path.isdir(vd):
                    continue
                w = []
                for f in range(SEQUENCE_LENGTH):
                    p = os.path.join(vd, f'{f}.npy')
                    if os.path.exists(p):
                        w.append(np.load(p))
                    else:
                        break
                if len(w) == SEQUENCE_LENGTH:
                    splits[sp]['raw'].append(np.array(w, dtype=np.float32))
                    splits[sp]['labels'].append(lm[g])
    for s in splits:
        print(f'{s}: {len(splits[s]["labels"])}')
    return splits, actions, lm


def get_class_weights(labels, nc):
    cc = np.bincount(labels, minlength=nc).astype(np.float32)
    cc = np.maximum(cc, 1.0)
    cw = 1.0 / cc
    cw = cw / cw.sum() * nc
    return torch.tensor(cw, dtype=torch.float32)


def load_d_arcface_logits(splits, nc):
    def encode(raw):
        return np.array([np.array([extract_D153(s[t]) for t in range(SEQUENCE_LENGTH)]) for s in raw], dtype=np.float32)

    d_candidates = [
        'rework_model/wlasl100_D_noshoulder_arcface_m02_best.pth',
        'rework_model/wlasl100_D153_arcface_m02_best.pth',
        'rework_model/wlasl100_D153_arcface_single.pth',
    ]
    d_path = next((p for p in d_candidates if os.path.exists(p)), None)
    if d_path is None:
        raise FileNotFoundError('Could not find a trained D_noshoulder ArcFace checkpoint.')

    print(f'\n=== Loading D_noshoulder / D_153 ArcFace ===')
    x_val = encode(splits['val']['raw'])
    x_test = encode(splits['test']['raw'])
    val_loader = tensor_loader(x_val, np.array(splits['val']['labels'], dtype=np.int64), batch_size=32)
    test_loader = tensor_loader(x_test, np.array(splits['test']['labels'], dtype=np.int64), batch_size=32)
    model = ArcFaceModel(153, nc, scale=16.0).to(device)
    d_state = torch.load(d_path, weights_only=True)
    if any(key.startswith('conv1d.') for key in d_state):
        d_state = remap_legacy_d_arcface_state(d_state)
    model.load_state_dict(d_state)
    val_logits = collect_logits(model, val_loader)
    test_logits = collect_logits(model, test_loader)
    val_probs = collect_probs_from_logits(val_logits)
    test_probs = collect_probs_from_logits(test_logits)
    top1, top5 = eval_probs(test_probs, np.array(splits['test']['labels'], dtype=np.int64), 'D_noshoulder ArcFace')
    return {
        'name': 'd_noshoulder',
        'loss_type': 'arcface',
        'margin': DEFAULT_ARCFACE_MARGIN,
        'val_logits': val_logits,
        'test_logits': test_logits,
        'val_probs': val_probs,
        'test_probs': test_probs,
        'top1': top1,
        'top5': top5,
    }


def main():
    os.makedirs('rework_model', exist_ok=True)
    print(f'Device: {device}')
    print(f'Config: shape={SHAPE_LOSS}, motion={MOTION_LOSS}, arcface_margin={DEFAULT_ARCFACE_MARGIN:.2f}')
    splits, actions, _ = load_raw()
    nc = len(actions)
    y_tr = np.array(splits['train']['labels'], dtype=np.int64)
    y_val = np.array(splits['val']['labels'], dtype=np.int64)
    y_te = np.array(splits['test']['labels'], dtype=np.int64)
    cw_t = get_class_weights(y_tr, nc)

    shape = train_stream(
        'shape',
        extract_shape,
        False,
        splits,
        nc,
        cw_t,
        loss_type=SHAPE_LOSS,
        margin=SHAPE_MARGIN if SHAPE_LOSS == 'arcface' else 0.0,
    )
    motion = train_stream(
        'motion',
        extract_motion,
        True,
        splits,
        nc,
        cw_t,
        loss_type=MOTION_LOSS,
        margin=MOTION_MARGIN if MOTION_LOSS == 'arcface' else 0.0,
    )
    d_stream = load_d_arcface_logits(splits, nc)

    streams = {
        'd_noshoulder': d_stream,
        'shape': shape,
        'motion': motion,
    }

    print('\n' + '=' * 68)
    print('  TEMPERATURE SCALING (fit on validation only)')
    print('=' * 68)
    temps = {}
    val_probs = {}
    test_probs = {}
    for key, result in streams.items():
        t, loss = fit_temperature(result['val_logits'], y_val)
        temps[key] = t
        val_probs[key] = collect_probs_from_logits(result['val_logits'], t)
        test_probs[key] = collect_probs_from_logits(result['test_logits'], t)
        print(f'  {key:<14} T={t:.2f}  val_nll={loss:.4f}')

    print('\n=== Individual Streams (after calibration) ===')
    eval_probs(test_probs['d_noshoulder'], y_te, 'D_noshoulder / D_153')
    eval_probs(test_probs['shape'], y_te, f'Shape ({shape["loss_type"]})')
    eval_probs(test_probs['motion'], y_te, f'Motion ({motion["loss_type"]})')

    print('\n' + '=' * 68)
    print('  FUSION (weights tuned on validation only)')
    print('=' * 68)
    order = ['d_noshoulder', 'shape', 'motion']
    equal = sum(test_probs[k] for k in order) / len(order)
    eval_probs(equal, y_te, 'Equal-weight fusion')

    best_w, best_val = search_fusion_weights(val_probs, y_val, order)
    fused = best_w[0] * test_probs[order[0]] + best_w[1] * test_probs[order[1]] + best_w[2] * test_probs[order[2]]
    print(f'  Best val weights: D={best_w[0]:.2f}, Shape={best_w[1]:.2f}, Motion={best_w[2]:.2f} (val={best_val:.2f}%)')
    eval_probs(fused, y_te, 'Best calibrated fusion')

    d_ok = np.argmax(test_probs['d_noshoulder'], 1) == y_te
    s_ok = np.argmax(test_probs['shape'], 1) == y_te
    m_ok = np.argmax(test_probs['motion'], 1) == y_te
    print('\n=== Complementarity ===')
    print(f'  All correct:  {(d_ok & s_ok & m_ok).sum()}')
    print(f'  D only:       {(d_ok & ~s_ok & ~m_ok).sum()}')
    print(f'  Shape only:   {(s_ok & ~d_ok & ~m_ok).sum()}')
    print(f'  Motion only:  {(m_ok & ~d_ok & ~s_ok).sum()}')
    print(f'  None:         {(~d_ok & ~s_ok & ~m_ok).sum()}')

    print('\n=== Notes ===')
    print('  Shape defaults to CE because vanilla ArcFace previously degraded this stream.')
    print('  Motion and D_noshoulder use a small ArcFace margin to keep the metric head less brittle.')
    print('  Fusion is calibrated and tuned on validation only; test labels are never used for weight selection.')


if __name__ == '__main__':
    main()
