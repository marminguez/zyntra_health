"""Integrity gate before training the V15.4 Scale-10 data ablation."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

EXPECTED_VAL=10686
EXPECTED_CHANNELS=8
EXPECTED_SEQ=(288,15)
EXPECTED_FUTURE=(24,8)
EXPECTED_Y=4

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def shards(root:Path,split:str):return sorted((root/split).glob('*.npz'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scale10',default='ml/data/v15_4_scale10');ap.add_argument('--frozen',default='ml/data/v15_master');a=ap.parse_args();scale=Path(a.scale10);frozen=Path(a.frozen)
 errors=[]
 # Validation must be byte-identical to frozen V15 master.
 sv=shards(scale,'validation');fv=shards(frozen,'validation');sm={p.name:p for p in sv};fm={p.name:p for p in fv}
 if set(sm)!=set(fm):errors.append(f'validation shard names differ: scale={len(sm)} frozen={len(fm)}')
 val_n=0
 for name in sorted(set(sm)&set(fm)):
  if sha256(sm[name])!=sha256(fm[name]):errors.append(f'validation bytes differ: {name}')
  with np.load(sm[name],allow_pickle=False) as z:
   n=len(z['y']);val_n+=n
   if z['x'].shape[1:]!=EXPECTED_SEQ:errors.append(f'{name}: validation x shape {z["x"].shape}')
   if z['future_known'].shape[1:]!=EXPECTED_FUTURE:errors.append(f'{name}: validation future shape {z["future_known"].shape}')
   if z['y'].shape[1:]!=(EXPECTED_Y,):errors.append(f'{name}: validation y shape {z["y"].shape}')
 if val_n!=EXPECTED_VAL:errors.append(f'validation windows={val_n}, expected={EXPECTED_VAL}')
 # Train shape/content checks.
 train=shards(scale,'train');train_n=0;bad_nonfinite=0
 for p in train:
  with np.load(p,allow_pickle=False) as z:
   x=z['x'];f=z['future_known'];y=z['y'];t=z['timestamp'];n=len(y);train_n+=n
   if not (len(x)==len(f)==len(y)==len(t)):errors.append(f'{p.name}: inconsistent row counts')
   if x.shape[1:]!=EXPECTED_SEQ:errors.append(f'{p.name}: x shape {x.shape}')
   if f.shape[1:]!=EXPECTED_FUTURE:errors.append(f'{p.name}: future shape {f.shape}')
   if y.shape[1:]!=(EXPECTED_Y,):errors.append(f'{p.name}: y shape {y.shape}')
   if n>240:errors.append(f'{p.name}: {n} windows exceeds cap 240')
   if not np.all(np.isfinite(x)):bad_nonfinite+=int(np.size(x)-np.isfinite(x).sum())
   if not np.all(np.isfinite(f)):bad_nonfinite+=int(np.size(f)-np.isfinite(f).sum())
   if not np.all(np.isfinite(y)):bad_nonfinite+=int(np.size(y)-np.isfinite(y).sum())
 if bad_nonfinite:errors.append(f'non-finite numeric values: {bad_nonfinite}')
 meta=json.loads((scale/'metadata.json').read_text(encoding='utf-8'))
 if meta.get('train_windows')!=train_n:errors.append(f'metadata train_windows={meta.get("train_windows")} actual={train_n}')
 if meta.get('validation_frozen_windows')!=val_n:errors.append(f'metadata validation={meta.get("validation_frozen_windows")} actual={val_n}')
 if meta.get('future_cgm_input') is not False:errors.append('future_cgm_input is not false')
 print('=== V15.4 SCALE-10 INTEGRITY CHECK ===');print(f'train shards:       {len(train):,}');print(f'train windows:      {train_n:,}');print(f'validation shards:  {len(sv):,}');print(f'validation windows: {val_n:,}');print(f'validation byte-identical: {set(sm)==set(fm) and all(sha256(sm[n])==sha256(fm[n]) for n in set(sm)&set(fm))}');print(f'future CGM input:   {meta.get("future_cgm_input")}')
 if errors:
  print('\nFAIL');
  for e in errors[:30]:print(f' - {e}')
  raise SystemExit(1)
 print('\nPASS — Scale-10 is compatible and frozen validation is exact.')
if __name__=='__main__':main()
