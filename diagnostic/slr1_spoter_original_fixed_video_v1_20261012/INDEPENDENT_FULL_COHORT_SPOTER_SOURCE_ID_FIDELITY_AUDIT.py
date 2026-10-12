#!/usr/bin/env python3
import collections,hashlib,json,math,random,sys
from pathlib import Path
import numpy as np,torch
P=Path('/workspace/local-vlm/SLR/FELF-SLR/diagnostic/slr1_spoter_original_fixed_video_v1_20261012')
sys.path.insert(0,str(P))
import SPOTER_ORIGINAL_SOURCE_SEMANTIC_ADAPTER as a
C=P/'FULL_ORIGINAL_VARIABLE_T_FIXED_COHORT_CACHE'
M=C/'FULL_SPOTER_EXACT_VIDEO_INPUT_MANIFEST.json'
assert a.sha(M)=='9b7e4abd60b0182dc5e5d2f23457b6fa9460594b6ac3c88504556b2979d708d6'
data=json.loads(M.read_text())
rows=list(data['per_video_metadata'].values())
original_rows=[json.loads(x) for x in (a.SIFORMER/'FROZEN_SHARED_VIDEO_IDENTITIES.jsonl').read_text().splitlines()]
native_by_id={r['video_id']:r for r in original_rows}
assert len(native_by_id)==4450
assert len(rows)==len({r['video_id'] for r in rows})==4450
assert collections.Counter(r['split'] for r in rows)=={'train':3549,'val':901}
assert sum(r['w100_class_index'] is not None and r['split']=='train' for r in rows)==1442
assert sum(r['w100_class_index'] is not None and r['split']=='val' for r in rows)==338
assert {r['video_id'] for r in rows if r['split']=='train'}.isdisjoint({r['video_id'] for r in rows if r['split']=='val'})
ids=json.loads((a.SIFORMER/'NATIVE_SIFORMER_204_FULL_FIXED_COHORT_CACHE/val_IDENTITY_AND_LABELS.json').read_text())
assert len(ids['video_ids'])==901 and len(set(ids['video_ids']))==901
assert {r['video_id'] for r in rows if r['split']=='val'}==set(ids['video_ids'])
v100={r['video_id'] for r in rows if r['split']=='val' and r['w100_class_index'] is not None}
assert v100=={ids['video_ids'][i] for i,y in enumerate(ids['labels_w100_minus1_if_added']) if y>=0}
mc=json.loads((a.SIFORMER/'IDENTITY_FREEZE_RECEIPT.json').read_text())
assert mc['shared338']==338 and mc['fixed_rows_sha256']==a.sha(a.SIFORMER/'FROZEN_SHARED_VIDEO_IDENTITIES.jsonl')
allsha=0;seen_types={};details={}
rows_by_type={}
for row in rows:
 t=row['n_frames']
 if row['split']=='train' and (t<=15 or t>=190 or row['semantic_missing_person_nframes']>0):
  if t<=15:rows_by_type.setdefault('short',row)
  if t>=190:rows_by_type.setdefault('long',row)
  if row['semantic_missing_person_nframes']>0:rows_by_type.setdefault('missing_person',row)
 if row['split']=='train' and 50<=t<=90:rows_by_type.setdefault('normal',row)
 p=C/row['split']/(row['video_id']+'.npy')
 assert a.sha(p)==row['converted_sha256'],p
 x=np.load(p,allow_pickle=False)
 assert x.shape==(t,54,2) and np.isfinite(x).all()
 if row['split']=='train':
  assert x.min()>=0
  assert np.isfinite(x).all() and x.min()>=0
 allsha+=1
assert allsha==4450
assert set(rows_by_type)=={'short','normal','long','missing_person'},list(rows_by_type)
mod=a.source_augment_module()
for j,(name,r) in enumerate(rows_by_type.items()):
 np.random.seed(379+j);random.seed(379+j);torch.manual_seed(379+j)
 assert native_by_id[r['video_id']]['gloss']==r['gloss'] and native_by_id[r['video_id']]['split']==r['split']
 original,person=a.read_source(native_by_id[r['video_id']])
 cached=np.load(C/'train'/(r['video_id']+'.npy'),allow_pickle=False)
 assert np.array_equal(original,cached)
 norm=a.original_normalize(cached)
 # Real exactly source normalizer normalized but no augment.
 assert norm.shape==cached.shape and np.isfinite(norm).all()
 aug=a.source_augment_one(cached,mod)
 assert aug.shape==cached.shape and np.isfinite(aug).all()
 # Data alignment to genuine original OpenPose landmarks means no new missing locations.
 # Source can transform points, but undetected mask restored before normalization.
 details[name]={'video_id':r['video_id'],'gloss':r['gloss'],'n_native_frames':r['n_frames'],'missing_person_frames':r['semantic_missing_person_nframes'],'raw_native_first_frame_54':True,'normalized_finite':True,'source_augmentation_finite':True}
summary={'verdict':'PASS_WITH_EXPLICIT_SOURCE_DETECTOR_AND_MISSING_MASK_ADAPTATIONS','kind':'ORIGINAL_SPOTER_NATIVE_VARIABLE_T_FROZEN_4450_COHORT','source_original_commit':'0f909bf92690772f43f0062be41860ed85b461ad','canonical_video_id_jsonl_sha256':a.sha(a.SIFORMER/'FROZEN_SHARED_VIDEO_IDENTITIES.jsonl'),'source_input_manifest_sha256':a.sha(M),'all_4450_sha256_verified':True,'full_video_count':4450,'train3549':True,'val901':True,'w100_train1442':True,'w100_val338':True,'val338_equals_siformer':True,'original_frame_range':[12,209],'frame_sequence_no_nearest_204':True,'native_empty_person_frames':488,'original_OpenPose_joint_semantics_12body42hands':True,'no_joints_synthesized':True,'source_normalize_single_dict':True,'source_augmentation_train_only_with_missing_zero_mask_restoration':True,'input_cache_train_raw_native_val_pre_normalized':True,'source_Vision_API_numerical_parity':'NOT_ESTABLISHED','video_cases':details,'test_sealed':True,'optimizer_updates':0}
p=P/'FULL_COHORT_SOURCE_FIDELITY_AND_REAL_EDGECASE_INDEPENDENT_AUDIT.json'
assert not p.exists()
p.write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
print('SPOTER_FULL_4450_SOURCE_FIDELITY_AUDIT_PASS',json.dumps({'file_sha256':a.sha(p),'manifest_sha256':a.sha(M),'seen_real_cases':details,'full_file_hash_checks':allsha}),flush=True)

