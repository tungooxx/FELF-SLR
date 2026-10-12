#!/usr/bin/env python3
"""SPOTER v1 author skeleton semantics on exactly signed native FELF videos.
No author alternate cohort; native original T=12..209 no frame repetition.
Source-normalizer and source-augmentations AST-loaded unchanged except
explicit post-augmentation restoration of originally missing joints.
"""
import ast, hashlib, json, logging, sys, types
from pathlib import Path
import numpy as np,torch
F=Path('/workspace/local-vlm/SLR/FELF-SLR')
HERE=F/'diagnostic/slr1_spoter_original_fixed_video_v1_20261012'
SOURCE=F/'diagnostic/crowding_baseline_repro_v1/vendor_spoter_0f909bf9'
SIFORMER=F/'diagnostic/slr1_siformer_fixed_cohort_input_v1_20261012'
BUNDLES=F/'diagnostic/slr1_w100_w300_author_paired_20261010/cache_io_only_repair/native_video_bundles_W300_full_from_zip_v3_20261012'
BODY_INDEX=(0,1,9,10,11,12,2,5,3,6,4,7)
HAND_INDEX=(0,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,4,3,2,1)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(1048576),b''):h.update(z)
 return h.hexdigest()
def orig_func(rel,name,identifier):
 p=SOURCE/rel
 tree=ast.parse(p.read_text())
 assignments=[x for node in tree.body if isinstance(node,ast.Assign) for x in node.targets if isinstance(x,ast.Name) and x.id==identifier]
 funcs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name]
 assert len(assignments)==len(funcs)==1,(p,identifier,name)
 value=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==identifier for x in n.targets)))
 ns={identifier:value,'logging':logging}
 exec(compile(ast.Module(body=funcs,type_ignores=[]),str(p),'exec'),ns)
 return ns[name],value
body_normalize,BODY_NAMES=orig_func('normalization/body_normalization.py','normalize_single_dict','BODY_IDENTIFIERS')
hand_normalize,HAND_NAMES=orig_func('normalization/hand_normalization.py','normalize_single_dict','HAND_IDENTIFIERS')
assert BODY_NAMES==['nose','neck','rightEye','leftEye','rightEar','leftEar','rightShoulder','leftShoulder','rightElbow','leftElbow','rightWrist','leftWrist']
assert HAND_NAMES==['wrist','indexTip','indexDIP','indexPIP','indexMCP','middleTip','middleDIP','middlePIP','middleMCP','ringTip','ringDIP','ringPIP','ringMCP','littleTip','littleDIP','littlePIP','littleMCP','thumbTip','thumbIP','thumbMP','thumbCMC']
ALL_NAMES=BODY_NAMES+[x+'_'+str(i) for i in (0,1) for x in HAND_NAMES]
assert len(ALL_NAMES)==54
def read_source(row):
 assert row['split'] in ('train','val')
 p=BUNDLES/row['split']/(row['video_id']+'.pt')
 assert sha(p)==row['native_bundle_sha256'],('native signed bundle digest mismatch',p)
 clip=torch.load(p,map_location='cpu',weights_only=True)
 assert str(clip['video_id'])==row['video_id'] and clip['split']==row['split']
 xy=clip['xy'].numpy()
 m=clip['valid_mask'].numpy().astype(bool)
 assert xy.ndim==3 and xy.shape[1:]==(55,2) and len(xy)==len(m)==row['native_frame_n']
 assert np.isfinite(xy).all()
 raw=np.maximum(0.0,(xy.astype(np.float64)+1.0)*0.5)
 raw[~m]=0.
 landmarks=np.concatenate([raw[:,BODY_INDEX,:],raw[:,13+np.array(HAND_INDEX),:],raw[:,34+np.array(HAND_INDEX),:]],axis=1)
 assert landmarks.shape==(len(xy),54,2) and np.isfinite(landmarks).all()
 assert np.all(landmarks[~m]==0)
 return landmarks.astype(np.float32),m
def to_row(arr):
 assert arr.ndim==3 and arr.shape[1:]==(54,2)
 return {name:arr[:,i,:].astype(np.float64).tolist() for i,name in enumerate(ALL_NAMES)}
def original_normalize(arr):
 row=to_row(arr)
 body_normalize(row)
 hand_normalize(row)
 output=np.asarray([row[name] for name in ALL_NAMES],dtype=np.float32).transpose(1,0,2).copy()
 output-=0.5
 assert output.shape==arr.shape and np.isfinite(output).all()
 return output
def source_augment_module():
 # Import original augmentation source unchanged; source body/hand normalization
 # top-level pandas import is unavailable in isolated runtime so bind our exact
 # AST-extracted normalizers and identifier sets into in-memory source modules.
 pkg=types.ModuleType('normalization')
 pkg.__path__=[str(SOURCE/'normalization')]
 bm=types.ModuleType('normalization.body_normalization')
 bm.BODY_IDENTIFIERS=BODY_NAMES;bm.normalize_single_dict=body_normalize
 hm=types.ModuleType('normalization.hand_normalization')
 hm.HAND_IDENTIFIERS=HAND_NAMES;hm.normalize_single_dict=hand_normalize
 sys.modules.update({'normalization':pkg,'normalization.body_normalization':bm,'normalization.hand_normalization':hm})
 sys.path.insert(0,str(SOURCE))
 import augmentations
 assert Path(augmentations.__file__).resolve()==(SOURCE/'augmentations/__init__.py').resolve()
 return augmentations
def source_augment_one(arr,mod):
 import random
 row=to_row(arr)
 if random.random()<0.5:
  which=random.randrange(4)
  if which==0:row=mod.augment_rotate(row,(-13,13))
  elif which==1:row=mod.augment_shear(row,'perspective',(0,0.1))
  elif which==2:row=mod.augment_shear(row,'squeeze',(0,0.15))
  else:row=mod.augment_arm_joint_rotate(row,0.3,(-4,4))
  # Some original geometric transforms change absent (0,0) to nonzero;
  # preserve original undetected semantics exactly, do not hallucinate joints.
  zero=np.all(arr==0,axis=2)
  for i,name in enumerate(ALL_NAMES):
   v=np.asarray(row[name],dtype=np.float64)
   v[zero[:,i]]=[0,0]
   row[name]=v.tolist()
 body_normalize(row);hand_normalize(row)
 out=np.asarray([row[name] for name in ALL_NAMES],dtype=np.float32).transpose(1,0,2).copy()-0.5
 assert np.isfinite(out).all() and out.shape==arr.shape
 return out
if __name__=='__main__':
 print('ORIGINAL_SPOTER_OP55_INPUT_INTERFACE_READY',json.dumps({'source_commit':'0f909bf92690772f43f0062be41860ed85b461ad','semantic_joint_count':54,'body_names':BODY_NAMES,'hand_names':HAND_NAMES,'body_index':BODY_INDEX,'hand_index':HAND_INDEX,'source_normalization_body_sha':sha(SOURCE/'normalization/body_normalization.py'),'source_normalization_hand_sha':sha(SOURCE/'normalization/hand_normalization.py'),'source_augment_sha':sha(SOURCE/'augmentations/__init__.py'),'variable_original_T':True,'test_firewall':True}),flush=True)

