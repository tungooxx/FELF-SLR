from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time
from pathlib import Path
from typing import Iterable

os.environ.setdefault('PYTHONHASHSEED','42')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Dataset, TensorDataset

REPO = Path('/workspace/local-vlm/SLR/FELF-SLR')
OUTROOT = Path('/workspace/local-vlm/SLR/slr1_factorial')
CODE = REPO/'code'
sys.path.insert(0, str(CODE))
from wlasl_geometry_utils import extract_part_aware_features

SEQ=40
FACE=[0,2,5,9,10]
CANONICAL = {
    'branch_dims':[96,96,96], 'd_model':288, 'conv_kernel':3, 'conv_layers':1,
    'num_layers':2, 'num_heads':4, 'ff_dim':768, 'dropout':0.2, 'pool_mode':'mean',
    'classifier':'cosine', 'scale':16.0, 'margin':0.2, 'mixup_alpha':0.2,
    'aug_repeats':10, 'optimizer':'AdamW', 'lr':1e-4, 'weight_decay':1e-4,
    'batch_size':32, 'epochs':50, 'patience':12, 'scheduler_factor':0.5,
    'scheduler_patience':5, 'swa_epochs':20, 'canonical_params':2063232,
    'canonical_seed1_val_top1':82.5,
}



import torch.nn.functional as F

class CosineClassifier(nn.Module):
    def __init__(self,d,nc,scale=16.0):
        super().__init__(); self.weight=nn.Parameter(torch.empty(nc,d)); nn.init.xavier_uniform_(self.weight); self.scale=float(scale)
    def forward(self,x): return self.scale*(F.normalize(x,p=2,dim=-1)@F.normalize(self.weight,p=2,dim=-1).t())

class ArcFaceLoss(nn.Module):
    def __init__(self,scale=16.0,margin=0.2,cw=None):
        super().__init__(); self.scale=float(scale); self.margin=float(margin)
        if cw is not None: self.register_buffer('cw',cw)
        else: self.cw=None
    def forward(self,logits,labels):
        cos=logits/self.scale; theta=torch.acos(cos.clamp(-1+1e-7,1-1e-7)); oh=F.one_hot(labels,cos.size(1)).float()
        return F.cross_entropy(self.scale*torch.cos(theta+oh*self.margin),labels,weight=self.cw,label_smoothing=0.05)

class PositionalEncoding(nn.Module):
    def __init__(self,d,dropout=0.1,max_len=500):
        super().__init__(); self.dropout=nn.Dropout(dropout); pe=torch.zeros(max_len,d); pos=torch.arange(max_len).float().unsqueeze(1)
        div=torch.exp(torch.arange(0,d,2).float()*(-torch.log(torch.tensor(10000.0))/d)); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div[:d//2]); pe=pe.unsqueeze(0).transpose(0,1); self.register_buffer('pe',pe)
    def forward(self,x): return self.dropout(x+self.pe[:x.size(0)])

class BranchStem(nn.Module):
    def __init__(self,inp,d=96):
        super().__init__(); self.net=nn.Sequential(nn.Linear(inp,d),nn.LayerNorm(d),nn.GELU(),nn.Linear(d,d))
    def forward(self,x): return self.net(x)

def mixup_three(left,right,global_features,y,alpha=0.2):
    lam=np.random.beta(alpha,alpha); idx=torch.randperm(left.size(0),device=left.device)
    return lam*left+(1-lam)*left[idx],lam*right+(1-lam)*right[idx],lam*global_features+(1-lam)*global_features[idx],y,y[idx],lam

def get_class_weights(labels,nc):
    cc=np.bincount(labels,minlength=nc).astype(np.float32); cc=np.maximum(cc,1.0); cw=1.0/cc; cw=cw/cw.sum()*nc; return torch.tensor(cw,dtype=torch.float32)

def time_stretch(seq,factor):
    t,d=seq.shape; nt=max(2,int(round(t*factor))); src,dst=np.arange(t),np.linspace(0,t-1,nt); out=np.empty((nt,d),dtype=seq.dtype)
    for i in range(d): out[:,i]=np.interp(dst,src,seq[:,i])
    return out

def time_warp(seq,max_warp=0.15):
    t,d=seq.shape; cx=np.linspace(0,1,4); cy=np.clip(cx+np.random.uniform(-max_warp,max_warp,4),0,1); cy[0],cy[-1]=0,1; y=np.interp(np.linspace(0,1,t),cx,cy)*(t-1); out=np.empty_like(seq)
    for i in range(d): out[:,i]=np.interp(np.arange(t),y,seq[:,i])
    return out

def random_frame_drop(seq,drop_prob=0.1):
    t,d=seq.shape; keep=np.random.rand(t)>drop_prob
    if keep.sum()<2: keep[np.random.randint(0,t,size=2)]=True
    kept=seq[keep]; src,dst=np.linspace(0,1,kept.shape[0]),np.linspace(0,1,t); out=np.empty_like(seq)
    for i in range(d): out[:,i]=np.interp(dst,src,kept[:,i])
    return out

def augment_fixed(seq):
    s=seq.copy()
    if np.random.rand()<0.7:
        factor=np.random.uniform(0.8,1.25); s=time_stretch(s,factor); s=time_stretch(s,SEQ/s.shape[0])
    if np.random.rand()<0.5: s=time_warp(s)
    if np.random.rand()<0.5: s=random_frame_drop(s,np.random.uniform(0.05,0.2))
    if np.random.rand()<0.5:
        shift=np.random.randint(-2,3)
        if shift!=0: s=np.roll(s,shift=shift,axis=0)
    if s.shape[0]!=SEQ: s=time_stretch(s,SEQ/s.shape[0])
    do_aff=np.random.rand()<0.9; theta=np.deg2rad(np.random.uniform(-20,20)); r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]); scale=np.random.uniform(0.9,1.12); trans=np.random.normal(0,np.random.uniform(0,0.03),(1,3)); do_noise=np.random.rand()<0.9; noise_std=np.random.uniform(0.005,0.015); do_drop=np.random.rand()<0.25; dm_p=np.random.rand(33)<np.random.uniform(0.05,0.15); dm_l=np.random.rand(21)<np.random.uniform(0.05,0.15); dm_r=np.random.rand(21)<np.random.uniform(0.05,0.15)
    out=s.copy()
    for t in range(SEQ):
        pose=out[t,0:132].reshape(33,4); lh=out[t,132:195].reshape(21,3); rh=out[t,195:258].reshape(21,3)
        if do_aff:
            pose[:,:2]=(pose[:,:2]-pose[:,:2].mean(0,keepdims=True))@r.T*scale+pose[:,:2].mean(0,keepdims=True)+trans[0,:2]; pose[:,2]*=scale
            lh[:,:2]=(lh[:,:2]-lh[:,:2].mean(0,keepdims=True))@r.T*scale+lh[:,:2].mean(0,keepdims=True)+trans[0,:2]; lh[:,2]*=scale
            rh[:,:2]=(rh[:,:2]-rh[:,:2].mean(0,keepdims=True))@r.T*scale+rh[:,:2].mean(0,keepdims=True)+trans[0,:2]; rh[:,2]*=scale
        if do_noise:
            pose[:,:3]+=np.random.normal(0,noise_std,(33,3)); lh+=np.random.normal(0,noise_std,(21,3)); rh+=np.random.normal(0,noise_std,(21,3))
        if do_drop: pose[dm_p,:3]=0.0; lh[dm_l]=0.0; rh[dm_r]=0.0
        out[t,0:132]=pose.flatten(); out[t,132:195]=lh.flatten(); out[t,195:258]=rh.flatten()
    return out

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def seed_all(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends.cuda,'matmul'): torch.backends.cuda.matmul.allow_tf32=False
    if hasattr(torch.backends,'cudnn'): torch.backends.cudnn.allow_tf32=False


def actions(json_path:Path,n:int)->list[str]:
    data=json.loads(json_path.read_text(encoding='utf-8'))
    return sorted(e['gloss'] for e in data[:n])


def assert_dev_split(split:str)->None:
    if split not in {'train','val'}:
        raise RuntimeError(f'CONFIRMATION_FIREWALL: split={split!r} forbidden; only train/val are allowed')


def load_split(data_dir:Path,json_path:Path,n:int,split:str,limit:int=0):
    assert_dev_split(split)
    names=actions(json_path,n); mp={g:i for i,g in enumerate(names)}
    seqs=[]; ys=[]; ids=[]
    root=data_dir/split
    for g in names:
        gd=root/g
        if not gd.exists(): continue
        for vd in sorted(p for p in gd.iterdir() if p.is_dir()):
            fs=[vd/f'{i}.npy' for i in range(SEQ)]
            if not all(p.exists() for p in fs): continue
            seqs.append(np.stack([np.load(p).astype(np.float32) for p in fs]))
            ys.append(mp[g]); ids.append(f'{split}:{g}:{vd.name}')
            if limit and len(ys)>=limit:
                return np.stack(seqs),np.asarray(ys,np.int64),ids,names
    if not ys: raise RuntimeError(f'no valid {split} samples')
    return np.stack(seqs),np.asarray(ys,np.int64),ids,names


RAW_POSE_INDICES = [0, 2, 5, 9, 10, 11, 12]
AUDITED_GEOMETRY_SHA256 = '51a7df8b592e1b9bdce8be19d70b54fa0941c611f19d6052b240cf507fdc3f63'
PERMUTATION_SEED_BASE = 20260902
DESIGN_ID = '25fa451d-362b-4a5a-9be1-4a10223d7a1f'
DESIGN_SEMANTIC_HASH = '7d5839cd890a75f5c2ffc000d54b5dc5c0dd7efeee2e5f4e3f0311487106f26e'
EXPERIMENT_NAME = 'REP-ARCH-2X2-WLASL100-DEV-v4'


def raw_sufficient_frame(vec:np.ndarray):
    """Information-sufficient raw coordinates for the accepted v4 design.

    This exposes every xyz source coordinate consumed by the canonical engineered
    extractor, but supplies none of its deterministic rectification, normalization,
    bone, angle, normal, distance, or scale-ratio descriptors.
    """
    pose=vec[:132].reshape(33,4)[:,:3].astype(np.float32)
    lh=vec[132:195].reshape(21,3).astype(np.float32)
    rh=vec[195:258].reshape(21,3).astype(np.float32)
    return lh.reshape(-1).copy(), rh.reshape(-1).copy(), pose[RAW_POSE_INDICES].reshape(-1).copy()


def raw_source_sufficiency_audit():
    source=REPO/'code/wlasl_geometry_utils.py'
    source_hash=sha256(source)
    # The exact source hash was manually audited before freezing v4. Under that
    # hash extract_part_aware_features consumes both complete 21x3 hands and only
    # pose indices FACE=[0,2,5,9,10] plus shoulders [11,12].
    vec=np.arange(258,dtype=np.float32)
    l,r,g=raw_sufficient_frame(vec)
    pose=vec[:132].reshape(33,4)[:,:3].astype(np.float32)
    expected_l=vec[132:195].astype(np.float32)
    expected_r=vec[195:258].astype(np.float32)
    expected_g=pose[RAW_POSE_INDICES].reshape(-1).astype(np.float32)
    exact=bool(np.array_equal(l,expected_l) and np.array_equal(r,expected_r) and np.array_equal(g,expected_g))
    return {
        'passed': bool(source_hash==AUDITED_GEOMETRY_SHA256 and exact),
        'audited_source_sha256': AUDITED_GEOMETRY_SHA256,
        'observed_source_sha256': source_hash,
        'hand_xyz_complete': bool(np.array_equal(l,expected_l) and np.array_equal(r,expected_r)),
        'pose_xyz_indices': list(RAW_POSE_INDICES),
        'pose_xyz_exact': bool(np.array_equal(g,expected_g)),
        'raw_dims':[int(l.size),int(r.size),int(g.size)],
    }


def frame_features(vec:np.ndarray,rep:str):
    if rep=='raw_sufficient': return raw_sufficient_frame(vec)
    if rep=='engineered': return extract_part_aware_features(vec,0.4)
    raise ValueError(rep)


def encode_one_sequence(s:np.ndarray,rep:str):
    ll=[];rr=[];gg=[]
    for v in s:
        a,b,c=frame_features(v,rep); ll.append(a);rr.append(b);gg.append(c)
    return np.asarray(ll,np.float32),np.asarray(rr,np.float32),np.asarray(gg,np.float32)

def encode_sequences(seqs:np.ndarray,rep:str):
    L=[];R=[];G=[]
    for i,s in enumerate(seqs):
        a,b,c=encode_one_sequence(s,rep); L.append(a);R.append(b);G.append(c)
        if (i+1)%250==0 or i+1==len(seqs): print(f'encode {rep}: {i+1}/{len(seqs)}',flush=True)
    return tuple(np.asarray(x,np.float32) for x in (L,R,G))


def cache_features(seqs:np.ndarray,y:np.ndarray,rep:str,cache_dir:Path,aug_repeats:int,force:bool=False):
    cache_dir.mkdir(parents=True,exist_ok=True)
    paths={k:cache_dir/f'{rep}_{k}.npy' for k in ['base_l','base_r','base_g','aug_l','aug_r','aug_g','aug_y']}
    if not force and all(p.exists() for p in paths.values()):
        return tuple(np.load(paths[k],mmap_mode='r') for k in ['base_l','base_r','base_g','aug_l','aug_r','aug_g']), np.load(paths['aug_y'],mmap_mode='r')
    base=encode_sequences(seqs,rep)
    for k,a in zip(['base_l','base_r','base_g'],base): np.save(paths[k],a)
    dims=[int(x.shape[-1]) for x in base]; expected=len(y)*aug_repeats
    al=np.lib.format.open_memmap(paths['aug_l'],mode='w+',dtype=np.float32,shape=(expected,SEQ,dims[0]))
    ar=np.lib.format.open_memmap(paths['aug_r'],mode='w+',dtype=np.float32,shape=(expected,SEQ,dims[1]))
    ag=np.lib.format.open_memmap(paths['aug_g'],mode='w+',dtype=np.float32,shape=(expected,SEQ,dims[2]))
    ay=np.empty(expected,np.int64)
    # Canonical B6 seed1 is the first frozen cache generator; later seed runs reuse that cache.
    seed_all(1); cursor=0
    for i,(seq,label) in enumerate(zip(seqs,y)):
        for _ in range(aug_repeats):
            a,b,c=encode_one_sequence(augment_fixed(seq).astype(np.float32),rep)
            al[cursor]=a; ar[cursor]=b; ag[cursor]=c; ay[cursor]=label; cursor+=1
        if (i+1)%100==0 or i+1==len(seqs): print(f'augment-cache {rep}: {i+1}/{len(seqs)}',flush=True)
    del al,ar,ag
    np.save(paths['aug_y'],ay)
    return tuple(np.load(paths[k],mmap_mode='r') for k in ['base_l','base_r','base_g','aug_l','aug_r','aug_g']), np.load(paths['aug_y'],mmap_mode='r')


class CombinedDataset(Dataset):
    def __init__(self,base,base_y,aug,aug_y): self.base=base;self.base_y=base_y;self.aug=aug;self.aug_y=aug_y
    def __len__(self): return len(self.base_y)+len(self.aug_y)
    def __getitem__(self,i):
        if i<len(self.base_y): arrays=self.base;y=self.base_y[i];j=i
        else: arrays=self.aug;y=self.aug_y[i-len(self.base_y)];j=i-len(self.base_y)
        return *(torch.from_numpy(np.array(x[j],dtype=np.float32)) for x in arrays), int(y)



class LocalGlobalB6(nn.Module):
    def __init__(self,ld:int,rd:int,gd:int,nc:int):
        super().__init__(); d=288
        self.left_stem=BranchStem(ld,96); self.right_stem=BranchStem(rd,96); self.global_stem=BranchStem(gd,96)
        self.res=nn.Linear(d,d); self.conv=nn.Conv1d(d,d,3,padding=1); self.norm=nn.LayerNorm(d); self.proj=nn.Linear(d,d); self.pe=PositionalEncoding(d,0.2,SEQ)
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=768,dropout=0.2); self.encoder=nn.TransformerEncoder(enc,num_layers=2); self.out_norm=nn.LayerNorm(d); self.drop=nn.Dropout(0.2); self.classifier=CosineClassifier(d,nc,16.0)
    def forward_features(self,l,r,g):
        src=torch.cat([self.left_stem(l),self.right_stem(r),self.global_stem(g)],-1); residual=self.res(src); src=self.conv(src.transpose(1,2)).transpose(1,2); src=self.norm(src+residual); src=self.proj(src); src_t=self.pe(src.transpose(0,1)); mem=self.encoder(src_t); mem=self.out_norm(mem+src_t).transpose(0,1); return self.drop(mem.mean(1))
    def forward(self,l,r,g): return self.classifier(self.forward_features(l,r,g))

def permutation_spec(dims):
    dims=tuple(int(x) for x in dims); total=sum(dims)
    perm=np.random.default_rng(PERMUTATION_SEED_BASE+total).permutation(total).astype(np.int64)
    groups=np.array_split(perm,3)
    boundaries=(dims[0],dims[0]+dims[1])
    def origin(i):
        return 'L' if i<boundaries[0] else ('R' if i<boundaries[1] else 'G')
    composition=[]
    for grp in groups:
        row={'L':0,'R':0,'G':0}
        for i in grp: row[origin(int(i))]+=1
        composition.append(row)
    bijective=bool(len(perm)==total and np.array_equal(np.sort(perm),np.arange(total,dtype=np.int64)))
    return {
        'seed':int(PERMUTATION_SEED_BASE+total),
        'total_dim':int(total),
        'group_dims':[int(len(x)) for x in groups],
        'group_composition':composition,
        'bijective':bijective,
        'permutation':perm,
    }


class PermutedBranchB6(nn.Module):
    """Exact BranchStem/trunk control with frozen non-anatomical feature grouping."""
    def __init__(self,ld:int,rd:int,gd:int,nc:int):
        super().__init__(); d=288; spec=permutation_spec((ld,rd,gd))
        self.group_dims=tuple(spec['group_dims'])
        self.register_buffer('perm',torch.from_numpy(spec['permutation'].copy()).long(),persistent=True)
        self.stem0=BranchStem(self.group_dims[0],96); self.stem1=BranchStem(self.group_dims[1],96); self.stem2=BranchStem(self.group_dims[2],96)
        self.res=nn.Linear(d,d); self.conv=nn.Conv1d(d,d,3,padding=1); self.norm=nn.LayerNorm(d); self.proj=nn.Linear(d,d); self.pe=PositionalEncoding(d,0.2,SEQ)
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=768,dropout=0.2); self.encoder=nn.TransformerEncoder(enc,num_layers=2); self.out_norm=nn.LayerNorm(d); self.drop=nn.Dropout(0.2); self.classifier=CosineClassifier(d,nc,16.0)
    def forward_features(self,l,r,g):
        raw=torch.cat([l,r,g],-1); mixed=raw.index_select(-1,self.perm); a,b,c=torch.split(mixed,self.group_dims,dim=-1)
        src=torch.cat([self.stem0(a),self.stem1(b),self.stem2(c)],-1); residual=self.res(src); src=self.conv(src.transpose(1,2)).transpose(1,2); src=self.norm(src+residual); src=self.proj(src); src_t=self.pe(src.transpose(0,1)); mem=self.encoder(src_t); mem=self.out_norm(mem+src_t).transpose(0,1); return self.drop(mem.mean(1))
    def forward(self,l,r,g): return self.classifier(self.forward_features(l,r,g))


def build_model(arch:str,dims,nc:int):
    if arch=='anatomical': return LocalGlobalB6(dims[0],dims[1],dims[2],nc)
    if arch=='permuted': return PermutedBranchB6(*dims,nc)
    raise ValueError(arch)


def make_loader(parts,y,batch=32): return DataLoader(TensorDataset(*(torch.from_numpy(np.asarray(x)) for x in parts),torch.from_numpy(np.asarray(y))),batch_size=batch,shuffle=False)

@torch.no_grad()
def evaluate(model,dl,dev,criterion=None):
    model.eval(); correct=total=0;loss=0.; logits=[]; labels=[]
    for l,r,g,y in dl:
        l,r,g,y=[x.to(dev) for x in (l,r,g,y)]
        z=model(l,r,g); correct+=(z.argmax(-1)==y).sum().item(); total+=y.numel(); logits.append(z.float().cpu());labels.append(y.cpu())
        if criterion is not None: loss+=criterion(z,y).item()*y.numel()
    z=torch.cat(logits);yy=torch.cat(labels); top5=(z.topk(min(5,z.shape[1]),dim=1).indices==yy[:,None]).any(1).float().mean().item()*100
    return {'top1':100*correct/max(total,1),'top5':top5,'loss':loss/max(total,1) if criterion is not None else None}


def train_one(rep,arch,seed,train_base,ytr,train_aug,aug_y,val_parts,yval,nc,outdir,max_epochs=50,swa_epochs=20,batch=32):
    seed_all(seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');amp=dev.type=='cuda'
    dims=tuple(int(x.shape[-1]) for x in train_base);model=build_model(arch,dims,nc).to(dev);params=sum(p.numel() for p in model.parameters())
    cw=get_class_weights(np.asarray(ytr),nc).to(dev);crit=ArcFaceLoss(scale=16.0,margin=0.2,cw=cw)
    ds=CombinedDataset(train_base,ytr,train_aug,aug_y);dl=DataLoader(ds,batch_size=batch,shuffle=True,drop_last=len(ds)>=batch);vl=make_loader(val_parts,yval,batch)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',0.5,patience=5);scaler=torch.amp.GradScaler('cuda',enabled=amp)
    outdir.mkdir(parents=True,exist_ok=True);best=outdir/'pre.pt';history=[];best_val=-1.;stale=0
    for ep in range(max_epochs):
        model.train();tot=0.;n=0
        for l,r,g,y in dl:
            l,r,g,y=[x.to(dev) for x in (l,r,g,y)];opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=amp):
                lm,rm,gm,ya,yb,lam=mixup_three(l,r,g,y,alpha=0.2);z=model(lm,rm,gm);loss=lam*crit(z,ya)+(1-lam)*crit(z,yb)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();tot+=loss.item();n+=1
        vm=evaluate(model,vl,dev,crit);sched.step(vm['loss']);row={'epoch':ep+1,'train_loss':tot/max(n,1),'val_top1':vm['top1'],'val_top5':vm['top5'],'val_loss':vm['loss']};history.append(row);print(json.dumps({'arm':f'{rep}+{arch}','seed':seed,**row}),flush=True)
        if vm['top1']>best_val: best_val=vm['top1'];stale=0;torch.save(model.state_dict(),best)
        else:
            stale+=1
            if stale>=12: break
    model.load_state_dict(torch.load(best,map_location=dev,weights_only=True));pre=evaluate(model,vl,dev,crit);selected='pre';selected_state={k:v.detach().cpu() for k,v in model.state_dict().items()};swa=None
    if swa_epochs>0:
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);avg=AveragedModel(model);swalr=SWALR(opt,swa_lr=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=amp)
        for ep in range(swa_epochs):
            model.train()
            for l,r,g,y in dl:
                l,r,g,y=[x.to(dev) for x in (l,r,g,y)];opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=amp):
                    lm,rm,gm,ya,yb,lam=mixup_three(l,r,g,y,alpha=0.2);z=model(lm,rm,gm);loss=lam*crit(z,ya)+(1-lam)*crit(z,yb)
                scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            avg.update_parameters(model);swalr.step();print(json.dumps({'arm':f'{rep}+{arch}','seed':seed,'swa_epoch':ep+1}),flush=True)
        swa=evaluate(avg,vl,dev,crit)
        if swa['top1']>=pre['top1']: selected='swa';selected_state={k:v.detach().cpu() for k,v in avg.module.state_dict().items()}
    torch.save(selected_state,outdir/'selected.pt')
    res={'rep':rep,'arch':arch,'seed':seed,'dims':dims,'params':params,'pre':pre,'swa':swa,'selected':selected,'selected_val_top1':(swa if selected=='swa' else pre)['top1'],'selected_val_top5':(swa if selected=='swa' else pre)['top5'],'history':history}
    (outdir/'result.json').write_text(json.dumps(res,indent=2));return res


def write_recipe(path:Path):
    baseline=json.loads((REPO/'diagnostic/canonical_checkpoint_reproduction/wlasl100/configs/baseline.json').read_text())
    obj={
        'experiment':EXPERIMENT_NAME,'scientific_design_id':DESIGN_ID,'design_semantic_hash':DESIGN_SEMANTIC_HASH,
        'canonical':CANONICAL,'source_head':'3acd6eb96e90de0ff285e64b1fd71cf3e40f3b62',
        'representation':{'raw_sufficient_dims':[63,63,21],'engineered_dims':[165,165,23],'raw_pose_indices':RAW_POSE_INDICES,'raw_source_sufficiency':raw_source_sufficiency_audit()},
        'partition_control':{'algorithm':'default_rng(20260902+total_dim).permutation then array_split(3)','raw':{k:v for k,v in permutation_spec((63,63,21)).items() if k!='permutation'},'engineered':{k:v for k,v in permutation_spec((165,165,23)).items() if k!='permutation'}},
        'baseline_config_subset':{k:baseline.get(k) for k in ['model','base_config','feature_type','feature_dims','dropout','mixup_alpha','margin','scale','aug_repeats','params','samples','selected_by_val','selected_by_val_metrics']},
        'source_hashes':{str(p.relative_to(REPO)):sha256(p) for p in [REPO/'code/wlasl_geometry_utils.py',REPO/'code/wlasl_train_old_localglobal_sweep_subset.py',REPO/'diagnostic/canonical_checkpoint_reproduction/wlasl100/configs/baseline.json']},
        'test_split_forbidden':True,'scientific_result':'NOT_ASSESSED',
    }
    path.write_text(json.dumps(obj,indent=2));return obj


def preflight(data_dir,json_path,n=100):
    fw=False
    try: assert_dev_split('test')
    except RuntimeError: fw=True
    tr,ytr,ids,names=load_split(data_dir,json_path,n,'train',limit=1); _va,_yv,_,_=load_split(data_dir,json_path,n,'val',limit=1)
    dims={}
    for rep in ['raw_sufficient','engineered']:
        dims[rep]=[int(x.shape[-1]) for x in encode_sequences(tr,rep)]
    params={}
    for rep,d in dims.items():
        for arch in ['anatomical','permuted']:
            params[f'{rep}+{arch}']=sum(p.numel() for p in build_model(arch,d,len(names)).parameters())
    expected={
        'engineered+anatomical':2063232,'engineered+permuted':2063232,
        'raw_sufficient+anatomical':2043456,'raw_sufficient+permuted':2043456,
    }
    raw_perm=permutation_spec(tuple(dims['raw_sufficient'])); eng_perm=permutation_spec(tuple(dims['engineered']))
    paired_exact=bool(params['raw_sufficient+anatomical']==params['raw_sufficient+permuted'] and params['engineered+anatomical']==params['engineered+permuted'])
    expected_exact=bool(params==expected)
    rep_delta=abs(params['engineered+anatomical']-params['raw_sufficient+anatomical'])/CANONICAL['canonical_params']
    source_audit=raw_source_sufficiency_audit()
    out={
        'experiment':EXPERIMENT_NAME,'scientific_design_id':DESIGN_ID,'design_semantic_hash':DESIGN_SEMANTIC_HASH,
        'firewall_test_rejected':fw,'dims':dims,'params':params,'expected_params':expected,
        'expected_params_exact':expected_exact,'paired_architecture_params_exact':paired_exact,
        'representation_param_delta_fraction_of_canonical':rep_delta,'representation_param_delta_within_2pct':bool(rep_delta<=0.02),
        'permutations':{
            'raw_sufficient':{k:v for k,v in raw_perm.items() if k!='permutation'},
            'engineered':{k:v for k,v in eng_perm.items() if k!='permutation'},
        },
        'permutation_bijective':bool(raw_perm['bijective'] and eng_perm['bijective']),
        'raw_source_sufficiency':source_audit,
        'paper_head':os.popen(f'git -C {REPO} rev-parse HEAD').read().strip(),
        'test_split_accessed':False,
    }
    out['passed']=bool(
        fw and dims['raw_sufficient']==[63,63,21] and dims['engineered']==[165,165,23]
        and expected_exact and paired_exact and out['representation_param_delta_within_2pct']
        and out['permutation_bijective'] and raw_perm['group_dims']==[49,49,49]
        and eng_perm['group_dims']==[118,118,117] and source_audit['passed']
        and out['paper_head']=='3acd6eb96e90de0ff285e64b1fd71cf3e40f3b62'
    )
    (OUTROOT/'preflight.json').write_text(json.dumps(out,indent=2));write_recipe(OUTROOT/'recipe_manifest.json');return out


def smoke(data_dir,json_path):
    tr,ytr,_,names=load_split(data_dir,json_path,100,'train',limit=8); va,yv,_,_=load_split(data_dir,json_path,100,'val',limit=8);results=[]
    for rep in ['raw_sufficient','engineered']:
        bp=encode_sequences(tr,rep);vp=encode_sequences(va,rep);ap=tuple(x[:0] for x in bp);ay=np.empty(0,np.int64)
        for arch in ['anatomical','permuted']:
            results.append(train_one(rep,arch,20260902,bp,ytr,ap,ay,vp,yv,len(names),OUTROOT/'smoke_runs_v4'/f'{rep}_{arch}',max_epochs=1,swa_epochs=0,batch=4))
    out={'scope':'ENGINEERING_SMOKE_ONLY','experiment':EXPERIMENT_NAME,'scientific_result':'NOT_ASSESSED','test_split_accessed':False,'results':results};(OUTROOT/'smoke.json').write_text(json.dumps(out,indent=2));return out


def parse_seeds(s:str): return [int(x) for x in s.split(',') if x.strip()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['preflight','smoke','run'],required=True);ap.add_argument('--data-dir',default=str(REPO/'WLASL_Data'));ap.add_argument('--json-path',default=str(REPO/'WLASL_Full/WLASL_v0.3.json'));ap.add_argument('--seeds',default='1,2,3');ap.add_argument('--only-arm',default='');ap.add_argument('--force-cache',action='store_true');ap.add_argument('--aug-repeats',type=int,default=10);ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--swa-epochs',type=int,default=20);a=ap.parse_args();OUTROOT.mkdir(parents=True,exist_ok=True);data=Path(a.data_dir);jsonp=Path(a.json_path)
    if a.mode=='preflight': print(json.dumps(preflight(data,jsonp),indent=2));return
    if a.mode=='smoke': preflight(data,jsonp); print(json.dumps(smoke(data,jsonp),indent=2));return
    pf=preflight(data,jsonp)
    if not pf['passed']: raise RuntimeError('v4 preflight failed')
    tr,ytr,_,names=load_split(data,jsonp,100,'train');va,yv,_,_=load_split(data,jsonp,100,'val');cache=OUTROOT/'cache_v4';allres=[]
    for rep in ['raw_sufficient','engineered']:
        cached,ay=cache_features(tr,ytr,rep,cache,a.aug_repeats,a.force_cache);bp=cached[:3];ap=cached[3:];vp=encode_sequences(va,rep)
        for arch in ['anatomical','permuted']:
            arm=f'{rep}+{arch}'
            if a.only_arm and arm!=a.only_arm: continue
            for seed in parse_seeds(a.seeds):
                rd=OUTROOT/'runs_v4'/arm.replace('+','_')/f'seed{seed}';rfile=rd/'result.json'
                if rfile.exists(): allres.append(json.loads(rfile.read_text()));continue
                allres.append(train_one(rep,arch,seed,bp,ytr,ap,ay,vp,yv,len(names),rd,a.epochs,a.swa_epochs,32))
    summary={'experiment':EXPERIMENT_NAME,'scientific_design_id':DESIGN_ID,'scientific_result':'PENDING_ASSESSMENT','test_split_accessed':False,'results':allres};(OUTROOT/'summary_partial_v4.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
