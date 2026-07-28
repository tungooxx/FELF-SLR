"""Compact local-global part-aware single-model ArcFace experiment."""
import os
import sys

# Configure environment for determinism before importing torch
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import random
import argparse

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args, unknown = parser.parse_known_args()
seed_everything(args.seed)
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

sys.path.insert(0, '.')

from wlasl_train_streams_arcface import (
    load_raw,
    get_class_weights,
    augment_fixed,
)
from wlasl_geometry_utils import extract_part_aware_features


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQUENCE_LENGTH = 40
RECTIFY_ALPHA = float(os.environ.get('WLASL_RECTIFY_ALPHA', '0.4'))
CACHE_DIR = os.environ.get('WLASL_CACHE_DIR', 'champion_aug' if os.path.exists('champion_aug') else (os.path.join('..', 'champion_aug') if os.path.exists(os.path.join('..', 'champion_aug')) else os.path.join('rework_model', 'cache')))


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


class BranchStem(nn.Module):
    def __init__(self, inp, d=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, x):
        return self.net(x)


class LocalGlobalArcFace(nn.Module):
    def __init__(self, left_inp, right_inp, global_inp, nc, d_branch=96, nh=4, nl=2, ff=768, dropout=0.3, scale=16.0):
        super().__init__()
        self.left_stem = BranchStem(left_inp, d_branch)
        self.right_stem = BranchStem(right_inp, d_branch)
        self.global_stem = BranchStem(global_inp, d_branch)
        d_model = d_branch * 3
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.res = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.pe = PositionalEncoding(d_model, dropout, SEQUENCE_LENGTH)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nh, dim_feedforward=ff, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc, num_layers=nl)
        self.out_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.classifier = CosineClassifier(d_model, nc, scale)

    def forward_features(self, left, right, global_features):
        left = self.left_stem(left)
        right = self.right_stem(right)
        global_features = self.global_stem(global_features)
        src = torch.cat([left, right, global_features], dim=-1)
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

    def forward(self, left, right, global_features):
        return self.classifier(self.forward_features(left, right, global_features))


class CachedAugmentedPartDataset(Dataset):
    def __init__(self, left_base, right_base, global_base, base_y, left_aug, right_aug, global_aug, aug_y):
        self.left_base = left_base
        self.right_base = right_base
        self.global_base = global_base
        self.base_y = base_y
        self.left_aug = left_aug
        self.right_aug = right_aug
        self.global_aug = global_aug
        self.aug_y = aug_y
        self.base_len = len(base_y)
        self.aug_len = 0 if left_aug is None else len(aug_y)

    def __len__(self):
        return self.base_len + self.aug_len

    def __getitem__(self, idx):
        if idx < self.base_len:
            return (
                torch.from_numpy(self.left_base[idx]),
                torch.from_numpy(self.right_base[idx]),
                torch.from_numpy(self.global_base[idx]),
                int(self.base_y[idx]),
            )
        j = idx - self.base_len
        return (
            torch.from_numpy(np.array(self.left_aug[j], dtype=np.float32)),
            torch.from_numpy(np.array(self.right_aug[j], dtype=np.float32)),
            torch.from_numpy(np.array(self.global_aug[j], dtype=np.float32)),
            int(self.aug_y[j]),
        )


def mixup_three(left, right, global_features, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(left.size(0), device=left.device)
    left_m = lam * left + (1 - lam) * left[idx]
    right_m = lam * right + (1 - lam) * right[idx]
    global_m = lam * global_features + (1 - lam) * global_features[idx]
    return left_m, right_m, global_m, y, y[idx], lam


def encode_parts(raw_list, split_name='split'):
    left_all, right_all, global_all = [], [], []
    total = len(raw_list)
    for i, seq in enumerate(raw_list):
        left_frames, right_frames, global_frames = [], [], []
        for t in range(SEQUENCE_LENGTH):
            left, right, global_features = extract_part_aware_features(seq[t], RECTIFY_ALPHA)
            left_frames.append(left)
            right_frames.append(right)
            global_frames.append(global_features)
        left_all.append(np.array(left_frames, dtype=np.float32))
        right_all.append(np.array(right_frames, dtype=np.float32))
        global_all.append(np.array(global_frames, dtype=np.float32))
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f'  encoded {split_name}: {i + 1}/{total}')
    return (
        np.array(left_all, dtype=np.float32),
        np.array(right_all, dtype=np.float32),
        np.array(global_all, dtype=np.float32),
    )


def make_loader(left, right, global_features, labels, batch_size=32, shuffle=False, drop_last=False):
    return DataLoader(
        TensorDataset(
            torch.tensor(left),
            torch.tensor(right),
            torch.tensor(global_features),
            torch.tensor(labels, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def build_cached_part_augments(raw_train, y_tr, left_dim, right_dim, global_dim, repeats=10):
    os.makedirs(CACHE_DIR, exist_ok=True)
    left_path = os.path.join(CACHE_DIR, 'local_global_left_aug.npy')
    right_path = os.path.join(CACHE_DIR, 'local_global_right_aug.npy')
    global_path = os.path.join(CACHE_DIR, 'local_global_global_aug.npy')
    y_path = os.path.join(CACHE_DIR, 'local_global_aug_labels.npy')

    if all(os.path.exists(p) for p in [left_path, right_path, global_path, y_path]):
        print('Using cached local-global augmentation.')
        return (
            np.load(left_path, mmap_mode='r'),
            np.load(right_path, mmap_mode='r'),
            np.load(global_path, mmap_mode='r'),
            np.load(y_path),
        )

    total = len(y_tr) * repeats
    print('Building cached local-global augmentation...')
    left_aug = np.lib.format.open_memmap(left_path, mode='w+', dtype=np.float32, shape=(total, 40, left_dim))
    right_aug = np.lib.format.open_memmap(right_path, mode='w+', dtype=np.float32, shape=(total, 40, right_dim))
    global_aug = np.lib.format.open_memmap(global_path, mode='w+', dtype=np.float32, shape=(total, 40, global_dim))
    aug_y = np.empty(total, dtype=np.int64)

    cursor = 0
    for i in tqdm(range(len(y_tr)), desc='Aug'):
        raw = raw_train[i]
        for _ in range(repeats):
            ar = augment_fixed(raw)
            left_seq = np.empty((40, left_dim), dtype=np.float32)
            right_seq = np.empty((40, right_dim), dtype=np.float32)
            global_seq = np.empty((40, global_dim), dtype=np.float32)
            for t in range(SEQUENCE_LENGTH):
                left, right, global_features = extract_part_aware_features(ar[t], RECTIFY_ALPHA)
                left_seq[t] = left
                right_seq[t] = right
                global_seq[t] = global_features
            left_aug[cursor] = left_seq
            right_aug[cursor] = right_seq
            global_aug[cursor] = global_seq
            aug_y[cursor] = y_tr[i]
            cursor += 1
        if (i + 1) % 100 == 0 or (i + 1) == len(y_tr):
            print(f'  cached {i + 1}/{len(y_tr)} train samples')

    del left_aug
    del right_aug
    del global_aug
    np.save(y_path, aug_y)
    return (
        np.load(left_path, mmap_mode='r'),
        np.load(right_path, mmap_mode='r'),
        np.load(global_path, mmap_mode='r'),
        np.load(y_path),
    )


def collect_probs(model, loader, tta_passes=0, noise_std=0.01):
    model.eval()
    all_passes = []
    with torch.no_grad():
        base_probs = []
        for left, right, global_features, _ in loader:
            base_probs.append(torch.softmax(model(left.to(device), right.to(device), global_features.to(device)), 1).cpu().numpy())
        all_passes.append(np.concatenate(base_probs))
        for _ in range(tta_passes):
            tta_probs = []
            for left, right, global_features, _ in loader:
                left = left.to(device) + torch.randn_like(left.to(device)) * noise_std
                right = right.to(device) + torch.randn_like(right.to(device)) * noise_std
                global_features = global_features.to(device) + torch.randn_like(global_features.to(device)) * noise_std
                tta_probs.append(torch.softmax(model(left, right, global_features), 1).cpu().numpy())
            all_passes.append(np.concatenate(tta_probs))
    return all_passes[0], np.mean(all_passes, axis=0)


def eval_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for left, right, global_features, labels in loader:
            pred = model(left.to(device), right.to(device), global_features.to(device)).argmax(1)
            labels = labels.to(device)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100 * correct / total


def topk_metrics(probs, y_true):
    pred = np.argmax(probs, 1)
    top1 = 100 * np.mean(pred == y_true)
    top5 = 100 * sum(y_true[i] in np.argsort(probs[i])[-5:] for i in range(len(y_true))) / len(y_true)
    return top1, top5


def main():
    splits, actions, _ = load_raw()
    nc = len(actions)
    y_tr = np.array(splits['train']['labels'], dtype=np.int64)
    y_v = np.array(splits['val']['labels'], dtype=np.int64)
    y_te = np.array(splits['test']['labels'], dtype=np.int64)
    cw_t = get_class_weights(y_tr, nc).to(device)

    print('Encoding base part-aware features...')
    left_tr, right_tr, global_tr = encode_parts(splits['train']['raw'], 'train')
    left_v, right_v, global_v = encode_parts(splits['val']['raw'], 'val')
    left_te, right_te, global_te = encode_parts(splits['test']['raw'], 'test')
    print(f'Left/right/global dims: {left_tr.shape[2]} / {right_tr.shape[2]} / {global_tr.shape[2]}')

    left_aug, right_aug, global_aug, aug_y = build_cached_part_augments(
        splits['train']['raw'],
        y_tr,
        left_tr.shape[2],
        right_tr.shape[2],
        global_tr.shape[2],
        repeats=10,
    )
    train_loader = DataLoader(
        CachedAugmentedPartDataset(left_tr, right_tr, global_tr, y_tr, left_aug, right_aug, global_aug, aug_y),
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )
    val_loader = make_loader(left_v, right_v, global_v, y_v)
    test_loader = make_loader(left_te, right_te, global_te, y_te)

    model = LocalGlobalArcFace(left_tr.shape[2], right_tr.shape[2], global_tr.shape[2], nc).to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    criterion = ArcFaceLoss(scale=16.0, margin=0.2, cw=cw_t)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=5)

    os.makedirs('rework_model', exist_ok=True)
    pre_path = 'rework_model/wlasl100_local_global_arcface_single.pth'
    swa_path = 'rework_model/wlasl100_local_global_arcface_swa.pth'
    best_va, patience = float('-inf'), 0

    for ep in range(50):
        model.train()
        for left, right, global_features, labels in train_loader:
            left = left.to(device); right = right.to(device); global_features = global_features.to(device); labels = labels.to(device)
            left_m, right_m, global_m, ya, yb, lam = mixup_three(left, right, global_features, labels)
            optimizer.zero_grad()
            logits = model(left_m, right_m, global_m)
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for left, right, global_features, labels in val_loader:
                logits = model(left.to(device), right.to(device), global_features.to(device))
                labels = labels.to(device)
                val_loss += criterion(logits, labels).item()
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        va = 100 * correct / total
        scheduler.step(val_loss / len(val_loader))
        if (ep + 1) % 5 == 0:
            print(f'Ep {ep + 1}: Val {va:.2f}%')
        if va > best_va:
            best_va = va
            patience = 0
            torch.save(model.state_dict(), pre_path)
        else:
            patience += 1
            if patience >= 12:
                print(f'Early stop ep {ep + 1}')
                break

    pre_state = torch.load(pre_path, weights_only=True)
    model.load_state_dict(pre_state)
    pre_probs, _ = collect_probs(model, test_loader)
    pre_t1, pre_t5 = topk_metrics(pre_probs, y_te)
    pre_val = eval_accuracy(model, val_loader)
    print(f'Pre-SWA: Top-1={pre_t1:.2f}%, Top-5={pre_t5:.2f}%, Val={pre_val:.2f}%')

    print('Phase 2: SWA (20 epochs)...')
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
    for ep in range(20):
        model.train()
        for left, right, global_features, labels in train_loader:
            left = left.to(device); right = right.to(device); global_features = global_features.to(device); labels = labels.to(device)
            left_m, right_m, global_m, ya, yb, lam = mixup_three(left, right, global_features, labels)
            optimizer.zero_grad()
            logits = model(left_m, right_m, global_m)
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        swa_model.update_parameters(model)
        swa_scheduler.step()
        if (ep + 1) % 5 == 0:
            print(f'  SWA ep {ep + 1}/20')
    torch.save(swa_model.module.state_dict(), swa_path)

    swa_model.eval()
    swa_probs, _ = collect_probs(swa_model, test_loader)
    swa_t1, swa_t5 = topk_metrics(swa_probs, y_te)
    swa_val = eval_accuracy(swa_model, val_loader)
    print(f'SWA: Top-1={swa_t1:.2f}%, Top-5={swa_t5:.2f}%, Val={swa_val:.2f}%')

    # Keep deployment selection behavior explicit, but reserve the canonical
    # m02_best checkpoint for the best pre-SWA model only.
    if swa_val >= pre_val:
        selected_name, selected_path = 'SWA', swa_path
    else:
        selected_name, selected_path = 'pre-SWA', pre_path

    selected_model = LocalGlobalArcFace(left_tr.shape[2], right_tr.shape[2], global_tr.shape[2], nc).to(device)
    selected_state = torch.load(selected_path, weights_only=True)
    selected_model.load_state_dict(selected_state)
    selected_probs, selected_tta_probs = collect_probs(selected_model, test_loader, tta_passes=5)
    t1, t5 = topk_metrics(selected_probs, y_te)
    tta_t1, _ = topk_metrics(selected_tta_probs, y_te)
    torch.save(selected_state, pre_path)
    torch.save(pre_state, 'rework_model/wlasl100_local_global_arcface_m02_best.pth')

    print('')
    print('Local-global part-aware single model + ArcFace')
    print(f'  Selected: {selected_name} (by val)')
    print(f'  Top-1: {t1:.2f}%')
    print(f'  Top-5: {t5:.2f}%')
    print(f'  TTA:   {tta_t1:.2f}%')
    print(f'  Val:   {max(pre_val, swa_val):.2f}%')
    print('  Promotion threshold: 81.68%')


if __name__ == '__main__':
    main()
