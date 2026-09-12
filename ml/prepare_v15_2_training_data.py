"""Build V15.2 data: V14.1 history + known future insulin and carbs.

Controlled extension of V15.1. Same seed, split, history, targets and per-patient
caps. Future input contains only insulin/carbs and their missingness indicators
for t+5..t+120. Future CGM is never an input.
"""
from __future__ import annotations
import argparse
from collections import deque
import json
from pathlib import Path
import random
import sys
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from forecasting.splits_v14 import development_split
from prepare_v14_training_data import BASE_FEATURES, FEATURE_NAMES, HISTORY_MINUTES, HORIZONS, SEED, SEQ_LEN, STEP_MINUTES, _feature_vector, _num, _reservoir_add

FUTURE_STEPS=120//STEP_MINUTES
COLUMNS=["id","source_file","date",*BASE_FEATURES,"subject_split_across_traintest"]

def write_subject(outdir,split,source,pid,reservoir):
    if not reservoir:return 0
    d=outdir/split; d.mkdir(parents=True,exist_ok=True)
    ss="".join(c if c.isalnum() or c in "-_" else "_" for c in source); sp="".join(c if c.isalnum() or c in "-_" else "_" for c in pid)
    np.savez_compressed(d/f"{ss}__{sp}.npz",x=np.stack([r[0] for r in reservoir]).astype(np.float32),future_known=np.stack([r[1] for r in reservoir]).astype(np.float32),y=np.stack([r[2] for r in reservoir]).astype(np.float32),timestamp=np.asarray([str(r[3]) for r in reservoir]))
    return len(reservoir)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--outdir',default='ml/data/v15_2'); p.add_argument('--train-windows-per-patient',type=int,default=24); p.add_argument('--val-windows-per-patient',type=int,default=48); p.add_argument('--batch-size',type=int,default=25000); p.add_argument('--memory-limit',default='2GB'); a=p.parse_args()
    data_path=Path(a.data_dir)/'train.parquet'; out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); tmp=Path(a.data_dir)/'.v15_2_duckdb_tmp'; tmp.mkdir(parents=True,exist_ok=True)
    pf=pq.ParquetFile(data_path); ps=str(data_path).replace("'","''"); tsql=str(tmp).replace("'","''")
    con=duckdb.connect(str(tmp/'v15_2_prepare.duckdb')); con.execute(f"SET temp_directory='{tsql}'"); con.execute(f"SET memory_limit='{a.memory_limit}'"); con.execute('SET preserve_insertion_order=false')
    cols=', '.join(f'"{c}"' for c in COLUMNS)
    q=f'''SELECT {cols} FROM read_parquet('{ps}') WHERE id IS NOT NULL AND source_file IS NOT NULL AND date IS NOT NULL ORDER BY source_file,id,date'''
    print(f"V15.2 dataset preparation from {pf.metadata.num_rows:,} parquet rows"); print('Controlled change vs V15.1: add future carbs; no future CGM input')
    reader=con.execute(q).fetch_record_batch(rows_per_batch=a.batch_size); rng=random.Random(SEED)
    key=None; split=None; overlap=False; gh={}; fb={}; gb={}; future={}; history=deque(); reservoir=[]; seen=0
    stats={'train_subjects':0,'validation_subjects':0,'overlap_subjects':0,'train_windows':0,'validation_windows':0,'rows_scanned':0}
    def flush():
        nonlocal reservoir,seen
        if key is None:return
        source,pid=key
        if overlap or split=='metabonet_overlap_excluded':stats['overlap_subjects']+=1
        else:
            n=write_subject(out,split,source,pid,reservoir); stats[f'{split}_subjects']+=1; stats[f'{split}_windows']+=n
        reservoir=[];seen=0
    for bi,batch in enumerate(reader,1):
        frame=batch.to_pandas();stats['rows_scanned']+=len(frame)
        for r in frame.itertuples(index=False):
            source,pid=str(r.source_file),str(r.id); k=(source,pid)
            if k!=key:
                flush();key=k;overlap=bool(r.subject_split_across_traintest) if r.subject_split_across_traintest is not None else False;split=development_split(source,pid,overlap);gh={};fb={};gb={};future={};history=deque()
            t=pd.Timestamp(r.date); vals={n:_num(getattr(r,n)) for n in BASE_FEATURES}; future[t]=(vals['insulin'],vals['carbs']); g=vals['CGM']
            if np.isfinite(g):
                feat=_feature_vector(t,vals,gh);history.append(t);gh[t]=g;fb[t]=feat;gb[t]=g
            anchor=t-pd.Timedelta(minutes=120)
            if not overlap and anchor in gb:
                ht=pd.date_range(anchor-pd.Timedelta(minutes=HISTORY_MINUTES-STEP_MINUTES),anchor,freq=f'{STEP_MINUTES}min'); tt=[anchor+pd.Timedelta(minutes=h) for h in HORIZONS]; ft=pd.date_range(anchor+pd.Timedelta(minutes=STEP_MINUTES),anchor+pd.Timedelta(minutes=120),freq=f'{STEP_MINUTES}min')
                if len(ht)==SEQ_LEN and all(x in fb for x in ht) and all(x in gb for x in tt) and len(ft)==FUTURE_STEPS and all(x in future for x in ft):
                    x=np.stack([fb[z] for z in ht]).astype(np.float32); fk=[]
                    for z in ft:
                        ins,carb=future[z]; im=0. if np.isfinite(ins) else 1.; cm=0. if np.isfinite(carb) else 1.; fk.append([0. if im else ins,im,0. if cm else carb,cm])
                    y=np.asarray([gb[z] for z in tt],dtype=np.float32);seen+=1;limit=a.train_windows_per_patient if split=='train' else a.val_windows_per_patient;_reservoir_add(reservoir,(x,np.asarray(fk,dtype=np.float32),y,anchor),seen,limit,rng)
            cutoff=t-pd.Timedelta(minutes=HISTORY_MINUTES+120)
            for store in (gh,fb,gb,future):
                for old in [z for z in store if z<cutoff]:store.pop(old,None)
            while history and history[0]<cutoff:history.popleft()
        if bi%20==0:
            pct=min(stats['rows_scanned']/max(pf.metadata.num_rows,1)*100,100);print(f"  scanned {stats['rows_scanned']:,}/{pf.metadata.num_rows:,} rows ({pct:.1f}%)",flush=True)
    flush();con.close()
    meta={'version':'v15.2','seed':SEED,'history_minutes':HISTORY_MINUTES,'sequence_length':SEQ_LEN,'sample_minutes':STEP_MINUTES,'horizons_minutes':list(HORIZONS),'historical_features':list(FEATURE_NAMES),'future_features':['insulin','insulin_missing','carbs','carbs_missing'],'future_steps':FUTURE_STEPS,'future_window':'t+5..t+120','future_cgm_as_input':False,'split_policy':'same deterministic patient-level policy as V14.1/V15.1; MetaboNet overlap excluded','train_windows_per_patient_cap':a.train_windows_per_patient,'validation_windows_per_patient_cap':a.val_windows_per_patient,**stats}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print('\nV15.2 dataset ready:');print(json.dumps(meta,indent=2));print(f'Artifacts written to {out}')
if __name__=='__main__':main()
