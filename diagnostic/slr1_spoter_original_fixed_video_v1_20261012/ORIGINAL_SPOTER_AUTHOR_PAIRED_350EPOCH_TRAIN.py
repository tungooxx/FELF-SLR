#!/usr/bin/env python3
"""One authorized paper-method original SPOTER v1 run on FELF frozen videos.
Original Git commit 0f909bf9, nn.Transformer 6enc+6dec, 9-head and 54x2,
variable T native source, SGD lr0.001 batch1, paper 350 epochs, no scheduler.
Only source-author augmentations and normalization, VAL-only checkpoints,
explicit no TEST access. PyTorch2.6 dispatch compatibility shim preserves
source cross-attention-only custom decoder layer, not SPOTER2.
"""
import argparse,hashlib,json,math,os,random,sys,time,types
from pathlib import Path
import numpy as np,torch
BASE=Path('/workspace/local-vlm/SLR/FELF-SLR/diagnostic/slr1_spoter_original_fixed_video_v1_20261012')
sys.path.insert(0,str(BASE))
import SPOTER_ORIGINAL_SOURCE_SEMANTIC_ADAPTER as inp
SRC=inp.SOURCE
sys.path.insert(0,str(SRC))
from spoter.spoter_model import SPOTER
from torch import nn
MANIFEST=BASE/'FULL_ORIGINAL_VARIABLE_T_FIXED_COHORT_CACHE/FULL_SPOTER_EXACT_VIDEO_INPUT_MANIFEST.json'
CACHE=MANIFEST.parent
OUT=BASE/'scientific_runs'
AUTHOR_COMMIT='0f909bf92690772f43f0062be41860ed85b461ad'
def require(x,why):
 if not x:raise RuntimeError(why)
def source_contract():
 from subprocess import check_output
 require(check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip()==AUTHOR_COMMIT,'ORIGINAL_SPOTER_GIT_DRIFT')
 require(inp.sha(SRC/'spoter/spoter_model.py')=='569bb31ba864eead8a9f2412d435e9f3f53e9ad6d34edcaef9034e1591f30274','ORIGINAL_MODEL_SHA_DRIFT')
 require(inp.sha(SRC/'train.py')=='ad96e55a907475546aeecaa2e2438c46f33044a296fdc7bdc3286c9eca2011a9','ORIGINAL_TRAIN_SHA_DRIFT')
 require(inp.sha(SRC/'spoter/utils.py')=='87c4c89c68f515417740d00499bf5d0dec7ac77865642680965f3d1bdc1833eb','ORIGINAL_UTILS_SHA_DRIFT')
 require(inp.sha(inp.SIFORMER/'FROZEN_SHARED_VIDEO_IDENTITIES.jsonl')=='e5bb874ef136b0cefff96ee11f06c717a49705172aa718968b68715eaf17354b','SHARED_SIGNED_COHORT_DRIFT')
def install_original_v1_compat(model):
 def source_legacy_dispatch(self,tgt,memory,tgt_mask=None,memory_mask=None,tgt_key_padding_mask=None,memory_key_padding_mask=None,tgt_is_causal=None,memory_is_causal=False,**other):
  require(not tgt_is_causal and not memory_is_causal and not other,'UNAUTHORIZED_PYTORCH_CAUSAL_NEW_SEMANTICS')
  x=tgt
  for layer in self.layers:
   x=layer(x,memory,tgt_mask=tgt_mask,memory_mask=memory_mask,tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=memory_key_padding_mask)
  return self.norm(x) if self.norm is not None else x
 model.transformer.decoder.forward=types.MethodType(source_legacy_dispatch,model.transformer.decoder)
 return model
class NativeSPOTERDataset(torch.utils.data.Dataset):
 def __init__(self,rows,augmentation):
  self.rows=rows
  self.augmentation=augmentation
  self.augmod=inp.source_augment_module() if augmentation else None
  self.sha_to_verify_sample=True
 def __len__(self):return len(self.rows)
 def __getitem__(self,index):
  m=self.rows[index]
  x=np.load(CACHE/m['split']/(m['video_id']+'.npy'),allow_pickle=False)
  assert x.shape==(m['n_frames'],54,2)
  if self.augmentation:
   x=inp.source_augment_one(x,self.augmod)
  # VAL was original SPOTER normalized once and frozen in cache.
  y=torch.from_numpy(np.asarray(x,dtype=np.float32).copy())
  if self.augmentation:y=y+torch.randn(y.size())*0.001 # source GaussianNoise std default .001 mean0
  assert y.shape==(m['n_frames'],54,2) and torch.isfinite(y).all()
  return y,int(m['w100_class_index'] if task=='W100' else m['w300_class_index']),m['video_id']
@torch.inference_mode()
def eval_model(model,rows,classes):
 model.eval()
 all_logits=np.empty((len(rows),classes),dtype=np.float32)
 vids=[];labels=[]
 correct=0;top5=0
 for i,m in enumerate(rows):
  x=np.load(CACHE/m['split']/(m['video_id']+'.npy'),allow_pickle=False)
  x=torch.from_numpy(np.asarray(x,dtype=np.float32).copy()).to(DEVICE)
  logits=model(x).reshape(-1)
  assert tuple(logits.shape)==(classes,) and torch.isfinite(logits).all()
  a=logits.detach().cpu().numpy();all_logits[i]=a
  label=int(m['w100_class_index'] if task=='W100' else m['w300_class_index'])
  correct+=int(np.argmax(a)==label)
  top5+=int(label in np.argpartition(a,-5)[-5:])
  labels.append(label);vids.append(m['video_id'])
 return {'top1_correct':correct,'top5_correct':top5,'total':len(rows),'top1_pct':100*correct/len(rows),'top5_pct':100*top5/len(rows)},np.asarray(vids),np.asarray(labels,dtype=np.int32),all_logits
def save_best(model,opt,ep,upd,metric,meta,ids,y,z,base):
 payload={'epoch':ep,'model':model.state_dict(),'optimizer':opt.state_dict(),'optimizer_updates':upd,'metrics':metric,'meta':meta,'rng':{'torch':torch.get_rng_state(),'numpy':np.random.get_state()[1].tolist(),'python':repr(random.getstate()[1][:10])}}
 p=base/'best.pt';q=base/'best.tmp.pt'
 torch.save(payload,q);os.replace(q,p)
 np.savez_compressed(base/'best_val_logits.npz',video_ids=ids,labels=y,logits=z,classes=meta['classes'],best_epoch=ep)
def main():
 global DEVICE,task
 a=argparse.ArgumentParser()
 a.add_argument('--task',choices=['W100','W300'],required=True)
 a.add_argument('--input-manifest-sha256',required=True)
 a.add_argument('--runner-sha256',required=True)
 a.add_argument('--authorized-gate-id',required=True)
 a.add_argument('--dry-run-no-optimizer',action='store_true')
 args=a.parse_args()
 task=args.task
 require(inp.sha(Path(__file__))==args.runner_sha256,'FROZEN_RUNNER_SHA_DRIFT')
 source_contract()
 require(MANIFEST.exists() and inp.sha(MANIFEST)==args.input_manifest_sha256,'FULL_INPUT_4450_MANIFEST_NOT_FROZEN_OR_MISMATCH')
 manifest=json.loads(MANIFEST.read_text())
 assert manifest['test_sealed'] and manifest['optimizer_updates']==0
 assert manifest['author_commit']==AUTHOR_COMMIT and len(manifest['per_video_metadata'])==4450
 require(len(args.authorized_gate_id)==36 or args.dry_run_no_optimizer,'NO_SCIENTIFIC_GATE_ID')
 arr=list(manifest['per_video_metadata'].values())
 train=[m for m in arr if m['split']=='train' and (m['w100_class_index'] is not None or task=='W300')]
 val=[m for m in arr if m['split']=='val' and (m['w100_class_index'] is not None or task=='W300')]
 train.sort(key=lambda m:m['video_id']);val.sort(key=lambda m:m['video_id'])
 expected={'W100':(1442,338,100),'W300':(3549,901,300)}
 ns,nv,classes=expected[task]
 assert (len(train),len(val))==(ns,nv)
 assert len(set(m['video_id'] for m in train)&set(m['video_id'] for m in val))==0
 for m in train+val:require(12<=m['n_frames']<=209,'SOURCE_NATIVE_VIDEO_FRAME_BOUNDS')
 torch.set_num_threads(2)
 random.seed(379);np.random.seed(379);torch.manual_seed(379);torch.cuda.manual_seed_all(379)
 torch.backends.cudnn.deterministic=True
 DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 require(DEVICE.type=='cuda','FREE_GPU_NOT_AVAILABLE')
 model=install_original_v1_compat(SPOTER(num_classes=classes,hidden_dim=108)).to(DEVICE)
 optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0,weight_decay=0)
 require(model.transformer.encoder.num_layers==6 and model.transformer.decoder.num_layers==6 and model.transformer.nhead==9,'NOT_ORIGINAL_SPOTER')
 require(model.linear_class.out_features==classes,'WRONG_VOCAB_SIZE')
 meta={'task':task,'seed':379,'epochs':350,'source_train_repo_default_epochs':100,'paper_epochs':350,'source_vs_paper_resolution':'paper 350 epochs preferred; source 100 epochs is default CLI and not necessarily original paper experimental run','train_n':ns,'val_n':nv,'classes':classes,'original_native_T':[12,209],'architecture':'original SPOTER 6 enc 6 custom cross-attn-only decoder 9 heads d_model108 no pooling','optimizer':'SGD LR0.001 momentum0 weight_decay0 per source/paper','scheduler':'none (paper); source ReduceLROnPlateau constructed but not called from train_epoch()','train_batch_size':1,'source_original_training_augmentation':'p0.5 four branches rotate/shear-perspective/shear-squeeze/armrotate before source body+hand normalization then subtract0.5 GaussianNoise std0.001','author_modification':'needed to restore original missing (0,0) after author rotate; native OpenPose same-video pose detector, author private Vision API CSV not guaranteed numerically identical; original PyTorch2.6 decoder dispatch shim; no original TEST-based checkpoint selection','model_commit':AUTHOR_COMMIT,'model_sha256':inp.sha(SRC/'spoter/spoter_model.py'),'input_manifest_sha256':args.input_manifest_sha256,'runner_sha256':args.runner_sha256,'authorized_gate_id':args.authorized_gate_id,'test_never_accessed':True}
 sample=train[0]
 smoke_raw=np.load(CACHE/'train'/(sample['video_id']+'.npy'),allow_pickle=False)
 smoke=inp.original_normalize(smoke_raw)
 model.eval()
 with torch.no_grad():
  o=model(torch.from_numpy(smoke).to(DEVICE))
 assert tuple(o.shape)==(1,1,classes) and torch.isfinite(o).all()
 if args.dry_run_no_optimizer:
  print('SPOTER_ORIGINAL_AUTHOR_METHOD_ZERO_OPTIMIZER_DRY_RUN_PASS',json.dumps({'task':task,'model_params':sum(p.numel() for p in model.parameters()),'train':ns,'val':nv,'native_T':len(smoke_raw),'eval_output':list(o.shape),'input_manifest_sha256':args.input_manifest_sha256,'original_git_commit':AUTHOR_COMMIT,'no_optimizer_updates':True,'TEST':False}),flush=True)
  return
 OUT.mkdir(exist_ok=True)
 out=OUT/f'original_spoter_{task}_seed379_350epoch_FROZEN_FELF_VIDEO'
 require(not (out/'history.jsonl').exists(),'NON_IDEMPOTENT_EXISTING_TRAIN_RUN')
 out.mkdir(exist_ok=True)
 (out/'FROZEN_RUN_PROVENANCE.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
 ds=NativeSPOTERDataset(train,True)
 steps=0;best=-1
 for ep in range(1,351):
  start=time.perf_counter()
  model.train()
  shuf=np.random.permutation(len(ds))
  corr=0;losses=0.0
  for i in shuf:
   x,label,vid=ds[int(i)]
   x=x.to(DEVICE)
   optimizer.zero_grad(set_to_none=True)
   logits=model(x).reshape(1,classes)
   loss=nn.functional.cross_entropy(logits,torch.as_tensor([label],device=DEVICE,dtype=torch.long))
   require(torch.isfinite(loss).item(),'TRAIN_NONFINITE_LOSS')
   loss.backward()
   optimizer.step();steps+=1
   corr+=int(logits.detach().argmax()==label)
   losses+=float(loss.item())
  metrics,vids,labels,probs=eval_model(model,val,classes)
  if metrics['top1_correct']>best:
   best=metrics['top1_correct']
   save_best(model,optimizer,ep,steps,metrics,meta,vids,labels,probs,out)
  history={'epoch':ep,'seed':379,'optimizer_updates':steps,'train_correct':corr,'train_loss_mean':losses/ns,'train_n':ns,'val':metrics,'elapsed_sec':time.perf_counter()-start,'test_never_accessed':True}
  with (out/'history.jsonl').open('a') as f:f.write(json.dumps(history,sort_keys=True)+'\n')
  torch.save({'epoch':ep,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'optimizer_updates':steps,'meta':meta},out/'latest.tmp.pt')
  os.replace(out/'latest.tmp.pt',out/'latest.pt')
  print('ORIGINAL_SPOTER_FROZEN_CANONICAL_EPOCH',json.dumps({'task':task,'epoch':ep,'of':350,'optimizer_updates':steps,'val_top1_correct':metrics['top1_correct'],'val_top5_correct':metrics['top5_correct'],'best_val_correct':best,'elapsed_sec':round(history['elapsed_sec'],2),'TEST':False}),flush=True)
 print('ORIGINAL_SPOTER_COMPLETED_350EPOCH',json.dumps({'task':task,'epochs':350,'optimizer_updates':steps,'best_val_correct':best,'TEST':False}),flush=True)
if __name__=='__main__':main()

