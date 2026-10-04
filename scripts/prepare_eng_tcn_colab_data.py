#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, zipfile
from pathlib import Path

EXPECTED = {
    'runner':'9ee29dedc32f2be0061d2325c4a579a112ed29d4d4acad2401b8e9524d35b77c',
    'preflight':'37a523ba5a6f5c597140d7bdd9cf0930a3d365e0b58e6b09a318ef80824bfa28',
    'prototype':'d4894a0aab33307a3e6fd1a9e49457c8f22b9f01d213bffe38a97c5f9343888a',
    'rep_runner':'ef5b771f6b161b2879dde2e6c6ba2913a244261b962d5a90c2d02556aedba3dd',
    'factorial_dev':'e5171fdeea7bfbd851c3112a52ca5a05fd9db248a8f86c33fb833e9ef7453662',
    'wlasl_geometry_utils':'51a7df8b592e1b9bdce8be19d70b54fa0941c611f19d6052b240cf507fdc3f63',
}
EXPECTED_ROOT = Path('/workspace/local-vlm/SLR/FELF-SLR')
FACTORIAL_ROOT = Path('/workspace/local-vlm/SLR/slr1_factorial')
CACHE_REL = Path('diagnostic/rep_arch_paper_data/rep_arch_features_v1/wlasl100/engineered')

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def assert_sha(path:Path, expected:str, label:str):
    got=sha(path)
    if got!=expected: raise SystemExit(f'{label} SHA mismatch: {got} != {expected}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,default=EXPECTED_ROOT)
    ap.add_argument('--dataset-dir',type=Path,required=True)
    ap.add_argument('--archive',default='eng_tcn_r1r2_wlasl100_dev_cache.zip')
    a=ap.parse_args()
    repo=a.repo.resolve(); ds=a.dataset_dir.resolve()
    if repo != EXPECTED_ROOT:
        EXPECTED_ROOT.parent.mkdir(parents=True,exist_ok=True)
        if EXPECTED_ROOT.exists() or EXPECTED_ROOT.is_symlink():
            if EXPECTED_ROOT.resolve()!=repo: raise SystemExit(f'{EXPECTED_ROOT} already exists and is not {repo}')
        else:
            EXPECTED_ROOT.symlink_to(repo,target_is_directory=True)
    assert_sha(repo/'diagnostic/eng_tcn_r1r2_runner.py',EXPECTED['runner'],'runner')
    assert_sha(repo/'diagnostic/eng_tcn_r1r2_preflight.json',EXPECTED['preflight'],'preflight')
    assert_sha(repo/'diagnostic/engineered_multiscale_tcn_prototype.py',EXPECTED['prototype'],'prototype')
    assert_sha(repo/'code/rep_arch_paper_runner.py',EXPECTED['rep_runner'],'rep_runner')
    assert_sha(repo/'code/wlasl_geometry_utils.py',EXPECTED['wlasl_geometry_utils'],'wlasl_geometry_utils')
    vendor=repo/'vendor/eng_tcn_r1r2/factorial_dev.py'
    assert_sha(vendor,EXPECTED['factorial_dev'],'vendored factorial_dev')
    FACTORIAL_ROOT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(vendor,FACTORIAL_ROOT/'factorial_dev.py')
    assert_sha(FACTORIAL_ROOT/'factorial_dev.py',EXPECTED['factorial_dev'],'installed factorial_dev')

    archive=ds/a.archive
    if not archive.exists(): raise SystemExit(f'missing archive: {archive}')
    tmp=ds/'_eng_tcn_extract'
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()
    with zipfile.ZipFile(archive) as z: z.extractall(tmp)
    manifest=json.loads((tmp/'manifest.json').read_text())
    if manifest.get('experiment_id')!='27e54160-30b8-440b-b517-2bffbeae897c': raise SystemExit('wrong experiment manifest')
    target=repo/CACHE_REL
    target.mkdir(parents=True,exist_ok=True)
    for e in manifest['files']:
        rel=Path(e['relative_path'])
        if rel.name.startswith('test_') or 'wlasl300' in str(rel).lower(): raise SystemExit(f'forbidden file in bundle: {rel}')
        src=tmp/rel
        assert_sha(src,e['sha256'],str(rel))
        dst=target/rel.relative_to('cache')
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        assert_sha(dst,e['sha256'],str(dst))
    shutil.rmtree(tmp)
    print(json.dumps({'status':'PASS','files':len(manifest['files']),'repo':str(repo),'cache':str(target),'test258_in_bundle':False,'wlasl300_in_bundle':False},sort_keys=True))
if __name__=='__main__': main()
