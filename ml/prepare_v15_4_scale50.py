"""Build resumable V15.4 Scale-50 training data with frozen validation.

Controlled change vs frozen V15.4: training cap 24 -> 1200 windows/patient.
Validation is copied byte-for-byte from v15_master. No future CGM input.
Resume is at completed-subject boundaries with deterministic subject RNG.
"""
from __future__ import annotations
import argparse,hashlib,json,random,shutil,sys
from pathlib import Path
import duckdb,numpy as np,pandas as pd,pyarrow.parquet as pq
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from forecasting.splits_v14 import development_split
from prepare_v14_training_data import BASE_FEATURES,HISTORY_MINUTES,HORIZONS,SEED,SEQ_LEN,STEP_MINUTES,_feature_vector,_num,_reservoir_add
FUTURE_STEPS=24;FUTURE_NUMERIC=('insulin','basal','bolus','carbs');COLUMNS=['id','source_file','date',*BASE_FEATURES,'subject_split_across_traintest'];CAP=1200;CHECKPOINT_VERSION=1
def safe(v):return ''.join(c if c.isalnum() or c in '-_' else '_' for c in v)
def srng(s,p):return random.Random(int.from_bytes(hashlib.sha256(f'{SEED}|{s}|{p}'.encode()).digest()[:8],'big'))
def write_subject(out,s,p,res):
 if not res:return 0
 d=out/'train';d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/f'{safe(s)}__{safe(p)}.npz',x=np.stack([r[0] for r in res]).astype(np.float32),future_known=np.stack([r[1] for r in res]).astype(np.float32),y=np.stack([r[2] for r in res]).astype(np.float32),timestamp=np.asarray([str(r[3]) for r in res]));return len(res)
def save_cp(path,s,p,stats):
 tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps({'checkpoint_version':1,'last_completed_source':s,'last_completed_id':p,'stats':stats},indent=2));tmp.replace(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train-parquet',required=True);ap.add_argument('--frozen-master',default='ml/data/v15_master');ap.add_argument('--outdir',default='ml/data/v15_4_scale50');ap.add_argument('--batch-size',type=int,default=25000);ap.add_argument('--memory-limit',default='3GB');ap.add_argument('--reset',action='store_true');a=ap.parse_args();data=Path(a.train_parquet);frozen=Path(a.frozen_master);out=Path(a.outdir);cp=out/'checkpoint.json';out.mkdir(parents=True,exist_ok=True)
 if a.reset:shutil.rmtree(out/'train',ignore_errors=True);cp.unlink(missing_ok=True);(out/'metadata.json').unlink(missing_ok=True)
 checkpoint=json.loads(cp.read_text()) if cp.exists() else None
 if checkpoint and checkpoint.get('checkpoint_version')!=CHECKPOINT_VERSION:raise RuntimeError('Unsupported checkpoint')
 if not checkpoint and (out/'train').exists() and any((out/'train').glob('*.npz')):raise RuntimeError('Train shards exist without checkpoint; use --reset or restore checkpoint')
 vd=out/'validation';vd.mkdir(exist_ok=True)
 for p in sorted((frozen/'validation').glob('*.npz')):
  if not (vd/p.name).exists():shutil.copy2(p,vd/p.name)
 valn=sum(len(np.load(p,allow_pickle=False)['y']) for p in vd.glob('*.npz'));stats={'rows_scanned_this_run':0,'train_subjects':0,'train_windows':0,'train_candidates_before_cap':0,'validation_frozen_windows':valn,'overlap_subjects':0,'validation_subjects_skipped':0,'completed_subjects':0};ls=lp=None
 if checkpoint:
  for k,v in checkpoint.get('stats',{}).items():
   if k in stats and k!='rows_scanned_this_run':stats[k]=v
  ls=str(checkpoint['last_completed_source']);lp=str(checkpoint['last_completed_id']);print(f'RESUME: continuing after {ls} / {lp}');print(f"Already written train windows: {stats['train_windows']:,}")
 else:print('Starting Scale-50 from the beginning')
 pf=pq.ParquetFile(data);tmp=data.parent/'.v15_scale50_duckdb_tmp';tmp.mkdir(exist_ok=True);con=duckdb.connect(str(tmp/'prepare.duckdb'));con.execute(f"SET temp_directory='{str(tmp).replace(chr(39),chr(39)*2)}'");con.execute(f"SET memory_limit='{a.memory_limit}'");con.execute('SET preserve_insertion_order=false');ps=str(data).replace("'","''");sel=', '.join(f'"{c}"' for c in COLUMNS);resume=''
 if ls is not None:resume=f" AND (source_file > '{ls.replace(chr(39),chr(39)*2)}' OR (source_file = '{ls.replace(chr(39),chr(39)*2)}' AND CAST(id AS VARCHAR) > '{lp.replace(chr(39),chr(39)*2)}'))"
 q=f'''SELECT {sel} FROM read_parquet('{ps}') WHERE id IS NOT NULL AND source_file IS NOT NULL AND date IS NOT NULL {resume} ORDER BY source_file,CAST(id AS VARCHAR),date''';reader=con.execute(q).fetch_record_batch(rows_per_batch=a.batch_size)
 key=None;split=None;overlap=False;gh={};fb={};gb={};fr={};res=[];seen=0;rng=None
 def flush():
  nonlocal res,seen
  if key is None:return
  s,p=key
  if overlap or split=='metabonet_overlap_excluded':stats['overlap_subjects']+=1
  elif split=='train':stats['train_subjects']+=1;stats['train_candidates_before_cap']+=seen;stats['train_windows']+=write_subject(out,s,p,res)
  elif split=='validation':stats['validation_subjects_skipped']+=1
  stats['completed_subjects']+=1;save_cp(cp,s,p,stats);res=[];seen=0
  if stats['completed_subjects']%25==0:print(f"checkpoint: {s}/{p} | completed={stats['completed_subjects']:,} | train_windows={stats['train_windows']:,}",flush=True)
 for bi,b in enumerate(reader,1):
  f=b.to_pandas();stats['rows_scanned_this_run']+=len(f)
  for r in f.itertuples(index=False):
   k=(str(r.source_file),str(r.id))
   if k!=key:flush();key=k;overlap=bool(r.subject_split_across_traintest) if r.subject_split_across_traintest is not None else False;split=development_split(*k,overlap);rng=srng(*k);gh={};fb={};gb={};fr={}
   ts=pd.Timestamp(r.date);vals={n:_num(getattr(r,n)) for n in BASE_FEATURES};fr[ts]=vals;g=vals['CGM']
   if np.isfinite(g):fb[ts]=_feature_vector(ts,vals,gh);gh[ts]=g;gb[ts]=g
   anchor=ts-pd.Timedelta(minutes=120)
   if not overlap and split=='train' and anchor in gb:
    ht=pd.date_range(anchor-pd.Timedelta(minutes=HISTORY_MINUTES-STEP_MINUTES),anchor,freq=f'{STEP_MINUTES}min');tt=[anchor+pd.Timedelta(minutes=h) for h in HORIZONS];ft=pd.date_range(anchor+pd.Timedelta(minutes=STEP_MINUTES),anchor+pd.Timedelta(minutes=120),freq=f'{STEP_MINUTES}min')
    if len(ht)==SEQ_LEN and all(t in fb for t in ht) and all(t in gb for t in tt) and len(ft)==FUTURE_STEPS and all(t in fr for t in ft):
     x=np.stack([fb[t] for t in ht]).astype(np.float32);fk=[]
     for t in ft:
      v=[]
      for n in FUTURE_NUMERIC:
       z=fr[t][n];m=0. if np.isfinite(z) else 1.;v.extend([0. if m else z,m])
      fk.append(v)
     y=np.asarray([gb[t] for t in tt],dtype=np.float32);seen+=1;_reservoir_add(res,(x,np.asarray(fk,dtype=np.float32),y,anchor),seen,CAP,rng)
   cutoff=ts-pd.Timedelta(minutes=HISTORY_MINUTES+120)
   for store in (gh,fb,gb,fr):
    for old in [z for z in store if z<cutoff]:store.pop(old,None)
  if bi%20==0:print(f"this run scanned {stats['rows_scanned_this_run']:,} rows | completed={stats['completed_subjects']:,} | train_windows={stats['train_windows']:,}",flush=True)
 flush();con.close();meta={'version':'v15.4-scale50-data','controlled_change':'train windows/patient cap 24 -> 1200 only','seed':SEED,'reservoir_rng':'deterministic subject-specific SHA256(seed|source|id)','train_windows_per_patient_cap':CAP,'validation_source':'byte copy of frozen v15_master validation','future_channels':[v for c in FUTURE_NUMERIC for v in (c,f'{c}_missing')],'future_cgm_input':False,'resumable':True,**stats};(out/'metadata.json').write_text(json.dumps(meta,indent=2));print('\n=== V15.4 SCALE-50 DATASET ===');print(json.dumps(meta,indent=2));print(f'Artifacts: {out.resolve()}')
if __name__=='__main__':main()
