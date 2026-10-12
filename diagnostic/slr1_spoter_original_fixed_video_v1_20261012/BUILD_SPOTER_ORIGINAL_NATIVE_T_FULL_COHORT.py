#!/usr/bin/env python3
"""Create original-variable-T SPOTER real-input cache for exactly 4450 FELF video IDs.
No 204-frame repetition, no alternate CSV samples, no optimizer and no TEST.
"""
import collections,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
H=Path('/workspace/local-vlm/SLR/FELF-SLR/diagnostic/slr1_spoter_original_fixed_video_v1_20261012')
sys.path.insert(0,str(H))
import SPOTER_ORIGINAL_SOURCE_SEMANTIC_ADAPTER as a
S=a.SIFORMER
OUT=H/'FULL_ORIGINAL_VARIABLE_T_FIXED_COHORT_CACHE'
if OUT.exists():raise RuntimeError('EXISTING_CACHE_DONT_SILENTLY_REWRITE')
assert shutil.disk_usage(H).free>520000000
manifest_file=S/'FROZEN_SHARED_VIDEO_IDENTITIES.jsonl'
assert a.sha(manifest_file)=='e5bb874ef136b0cefff96ee11f06c717a49705172aa718968b68715eaf17354b'
rows=[json.loads(line) for line in manifest_file.read_text().splitlines()]
assert len(rows)==4450
assert [len([r for r in rows if r['split']==sp]) for sp in ('train','val')]==[3549,901]
assert sum(r['w100_class_index'] is not None and r['split']=='train' for r in rows)==1442
assert sum(r['w100_class_index'] is not None and r['split']=='val' for r in rows)==338
assert len({r['video_id'] for r in rows})==4450
assert len({r['video_id'] for r in rows if r['split']=='train'}&{r['video_id'] for r in rows if r['split']=='val'})==0
OUT.mkdir()
for sp in ('train','val'): (OUT/sp).mkdir()
report={'source':'EXACT_SOURCE_SIGNED_NATIVE_OPENPOSE55','model':'ORIGINAL_SPOTER_6ENC6DEC_54x2_NATIVET','native_video_manifest_sha256':a.sha(manifest_file),'adapter_sha256':a.sha(H/'SPOTER_ORIGINAL_SOURCE_SEMANTIC_ADAPTER.py'),'author_commit':'0f909bf92690772f43f0062be41860ed85b461ad','test_sealed':True,'optimizer_updates':0,'no_204_frame_resampling':True,'source_normalization':'identical source SPOTER normalize_single_dict from AST; missing0 coordinates preserved before subtract0.5','author_source_modality_deviation':'Original author Vision API/CSV extraction unavailable for EXACT FELF video IDs; semantic OpenPose body/hand source from SAME signed video assets, physical indices verified. Numerical same-detector equivalence NOT certified','files':{},'per_video_metadata':{},'quality':collections.Counter()}
start=time.perf_counter()
for i,row in enumerate(rows):
 raw,person=a.read_source(row)
 p=OUT/row['split']/(row['video_id']+'.npy')
 if row['split']=='val':
  converted=a.original_normalize(raw)
 else:
  converted=raw
 assert len(converted)==row['native_frame_n'] and converted.shape[1:]==(54,2)
 assert np.isfinite(converted).all()
 np.save(p,converted,allow_pickle=False)
 metadata={'video_id':row['video_id'],'split':row['split'],'gloss':row['gloss'],'n_frames':len(converted),'w300_class_index':int(row['w300_class_index']),'w100_class_index':row['w100_class_index'],'native_bundle_sha256':row['native_bundle_sha256'],'semantic_missing_person_nframes':int((~person).sum()),'converted_sha256':a.sha(p)}
 report['per_video_metadata'][row['video_id']]=metadata
 report['quality']['original_frames']+=len(raw)
 report['quality']['missing_person_frames']+=int((~person).sum())
 report['quality']['missing_body_nose_frames']+=int(np.sum(raw[:,0,0]==0))
 report['quality']['missing_left_wrist_frames']+=int(np.sum(raw[:,12,0]==0))
 report['quality']['missing_right_wrist_frames']+=int(np.sum(raw[:,33,0]==0))
 report['quality'][row['split']+'_videos']+=1
 report['quality']['num_frames_under_20']+=len(raw)<20
 report['quality']['num_frames_over_180']+=len(raw)>180
 report['quality']['fully_missing_person_videos']+=bool(not person.any())
 if (i+1)%100==0 or i+1==len(rows):
  print('SPOTER_FULL_SAME_VIDEO_INPUT_CACHE_PROGRESS',json.dumps({'done':i+1,'total':4450,'split':row['split'],'elapsed_sec':round(time.perf_counter()-start,1),'test_sealed':True,'no_training':True}),flush=True)
assert report['quality']['missing_person_frames']==488,report['quality']
assert report['quality']['train_videos']==3549 and report['quality']['val_videos']==901
report['quality']=dict(report['quality'])
assert min(x['n_frames'] for x in report['per_video_metadata'].values())==12
assert max(x['n_frames'] for x in report['per_video_metadata'].values())==209
m=OUT/'FULL_SPOTER_EXACT_VIDEO_INPUT_MANIFEST.json';m.write_text(json.dumps(report,sort_keys=True)+'\n')
summary={'manifest_sha256':a.sha(m),'all_native_source_videos':4450,'splits':{'train':3549,'val':901},'native_frames':report['quality']['original_frames'],'missing_person_frames':488,'native_T_range':[12,209],'total_time_sec':round(time.perf_counter()-start,1),'no_other_cohort':True,'no_test':True}
p=H/'SOURCE_TO_SPOTER_VARIABLE_T_CACHE_RECEIPT.json';p.write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
print('SPOTER_AUTHOR_SEMANTIC_4450_COHORT_READY',json.dumps(summary),flush=True)

