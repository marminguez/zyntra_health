"""Build V15.4 Scale-10 training data while freezing V15 master validation.

Controlled experiment:
- training cap increases from 24 to 240 windows/patient;
- split/seed/history/targets/features remain V15.1/V15.4 compatible;
- future-known cache contains insulin/basal/bolus/carbs + missing masks;
- validation shards are copied byte-for-byte from frozen ml/data/v15_master;
- no future CGM is used as an input.
"""
from __future__ import annotations
import argparse,json,random,shutil,sys
from collections import deque
from pathlib import Path
import duckdb,numpy as np,pandas as pd,pyarrow.parquet as pq
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from forecasting.splits_v14 import development_split
from prepare_v14_training_data import BASE_FEATURES,FEATURE_NAMES,HISTORY_MINUTES,HORIZONS,SEED,SEQ_LEN,STEP_MINUTES,_feature_vector,_num,_reservoir_add
FUTURE_STEPS=24;FUTURE_NUMERIC=('insulin','basal','bolus','carbs');COLUMNS=['id','source_file','date',*BASE_FEATURES,'subject_split_across_traintest']

def write_subject(outdir,source,pid,res):
 if not res:return 0
 d=outdir/'train';d.mkdir(parents=True,exist_ok=True);ss=''.join(c if c.isalnum() or c in '-_' else '_' for c in source);sp=''.join(c if c.isalnum() or c in '-_' else '_' for c in pid)
 np.savez_compressed(d/f'{ss}__{sp}.npz',x=np.stack([r[0] for r in res]).astype(np.float32),future_known=np.stack([r[1] for r in res]).astype(np.float32),y=np.stack([r[2] for r in res]).astype(np.float32),timestamp=np.asarray([str(r[3]) for r in res]));return len(res)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train-parquet',required=True);ap.add_argument('--frozen-master',default='ml/data/v15_master');ap.add_argument('--outdir',default='ml/data/v15_4_scale10');ap.add_argument('--train-windows-per-patient',type=int,default=240);ap.add_argument('--batch-size',type=int,default=25000);ap.add_argument('--memory-limit',default='3GB');a=ap.parse_args()
 data=Path(a.train_parquet);frozen=Path(a.frozen_master);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 if a.train_windows_per_patient!=240:print(f'WARNING: non-registered cap {a.train_windows_per_patient}; Scale-10 reference is 240')
 # Freeze validation exactly by copying existing NPZ bytes.
 vd=out/'validation';vd.mkdir(exist_ok=True)
 for p in sorted((frozen/'validation').glob('*.npz')):shutil.copy2(p,vd/p.name)
 frozen_val=sum(len(np.load(p,allow_pickle=False)['y']) for p in sorted(vd.glob('*.npz')))
 pf=pq.ParquetFile(data);tmp=data.parent/'.v15_scale10_duckdb_tmp';tmp.mkdir(exist_ok=True);con=duckdb.connect(str(tmp/'prepare.duckdb'));con.execute(f"SET temp_directory='{str(tmp).replace(chr(39),chr(39)*2)}'");con.execute(f"SET memory_limit='{a.memory_limit}'");con.execute('SET preserve_insertion_order=false');ps=str(data).replace("'","''");sel=', '.join(f'"{c}"' for c in COLUMNS)
 q=f'''SELECT {sel} FROM read_parquet('{ps}') WHERE id IS NOT NULL AND source_file IS NOT NULL AND date IS NOT NULL ORDER BY source_file,id,date''';reader=con.execute(q).fetch_record_batch(rows_per_batch=a.batch_size);rng=random.Random(SEED)
 key=None;split=None;overlap=False;gh={};fb={};gb={};future_rows={};res=[];seen=0;stats={'rows_scanned':0,'train_subjects':0,'train_windows':0,'train_candidates_before_cap':0,'validation_frozen_windows':frozen_val,'overlap_subjects':0}
 def flush():
  nonlocal res,seen
  if key is None:return
  if overlap or split=='metabonet_overlap_excluded':stats['overlap_subjects']+=1
  elif split=='train':stats['train_subjects']+=1;stats['train_candidates_before_cap']+=seen;stats['train_windows']+=write_subject(out,*key,res)
  res=[];seen=0
 for bi,b in enumerate(reader,1):
  frame=b.to_pandas();stats['rows_scanned']+=len(frame)
  for r in frame.itertuples(index=False):
   k=(str(r.source_file),str(r.id))
   if k!=key:
    flush();key=k;overlap=bool(r.subject_split_across_traintest) if r.subject_split_across_traintest is not None else False;split=development_split(*k,overlap);gh={};fb={};gb={};future_rows={}
   ts=pd.Timestamp(r.date);vals={n:_num(getattr(r,n)) for n in BASE_FEATURES};future_rows[ts]=vals;g=vals['CGM']
   if np.isfinite(g):fb[ts]=_feature_vector(ts,vals,gh);gh[ts]=g;gb[ts]=g
   anchor=ts-pd.Timedelta(minutes=120)
   if not overlap and split=='train' and anchor in gb:
    hist=pd.date_range(anchor-pd.Timedelta(minutes=HISTORY_MINUTES-STEP_MINUTES),anchor,freq=f'{STEP_MINUTES}min');targets=[anchor+pd.Timedelta(minutes=h) for h in HORIZONS];future=pd.date_range(anchor+pd.Timedelta(minutes=STEP_MINUTES),anchor+pd.Timedelta(minutes=120),freq=f'{STEP_MINUTES}min')
    if len(hist)==SEQ_LEN and all(t in fb for t in hist) and all(t in gb for t in targets) and len(future)==FUTURE_STEPS and all(t in future_rows for t in future):
     x=np.stack([fb[t] for t in hist]).astype(np.float32);fk=[]
     for t in future:
      row=future_rows[t];v=[]
      for n in FUTURE_NUMERIC:
       z=row[n];m=0. if np.isfinite(z) else 1.;v.extend([0. if m else z,m])
      fk.append(v)
     y=np.asarray([gb[t] for t in targets],dtype=np.float32);seen+=1;_reservoir_add(res,(x,np.asarray(fk,dtype=np.float32),y,anchor),seen,a.train_windows_per_patient,rng)
   cutoff=ts-pd.Timedelta(minutes=HISTORY_MINUTES+120)
   for store in (gh,fb,gb,future_rows):
    for old in [z for z in store if z<cutoff]:store.pop(old,None)
  if bi%20==0:print(f"scanned {stats['rows_scanned']:,}/{pf.metadata.num_rows:,} ({100*stats['rows_scanned']/pf.metadata.num_rows:.1f}%)",flush=True)
 flush();con.close();meta={'version':'v15.4-scale10-data','controlled_change':'train windows/patient cap 24 -> 240 only','seed':SEED,'train_windows_per_patient_cap':a.train_windows_per_patient,'validation_source':'byte copy of frozen v15_master validation','future_channels':[v for c in FUTURE_NUMERIC for v in (c,f'{c}_missing')],'future_cgm_input':False,**stats};(out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print('\n=== V15.4 SCALE-10 DATASET ===');print(json.dumps(meta,indent=2));print(f'Artifacts: {out.resolve()}')
if __name__=='__main__':main()
