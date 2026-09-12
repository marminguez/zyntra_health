"""Audit how strongly V15.1/V15 master window caps undersample MetaboNet.

Read-only: scans train.parquet in subject/date order and counts every window that
satisfies the exact V15.1 history/target/future-insulin availability rules. It does
not materialize windows, train a model, or read test/Live targets.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict,deque
from pathlib import Path
import duckdb,numpy as np,pandas as pd,pyarrow.parquet as pq
from ml.forecasting.splits_v14 import development_split

HISTORY_MINUTES=1440; STEP=5; SEQ_LEN=288; HORIZONS=(30,60,90,120); FUTURE_STEPS=24
COLS=['id','source_file','date','CGM','insulin','subject_split_across_traintest']

def finite(v):
    try:return np.isfinite(float(v))
    except (TypeError,ValueError):return False

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--output',default='ml/results/top3_audit/data_scale_audit.json');ap.add_argument('--memory-limit',default='3GB');ap.add_argument('--batch-size',type=int,default=50000);a=ap.parse_args()
    path=Path(a.train);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);pf=pq.ParquetFile(path)
    tmp=path.parent/'.top3_scale_duckdb_tmp';tmp.mkdir(exist_ok=True);con=duckdb.connect(str(tmp/'audit.duckdb'));con.execute(f"SET temp_directory='{str(tmp).replace(chr(39),chr(39)*2)}'");con.execute(f"SET memory_limit='{a.memory_limit}'");con.execute('SET preserve_insertion_order=false')
    ps=str(path).replace("'","''");sel=', '.join(f'"{c}"' for c in COLS);q=f'''SELECT {sel} FROM read_parquet('{ps}') WHERE id IS NOT NULL AND source_file IS NOT NULL AND date IS NOT NULL ORDER BY source_file,id,date'''
    reader=con.execute(q).fetch_record_batch(rows_per_batch=a.batch_size)
    current=None;split=None;overlap=False;cgm=set();ins=set();seen=0;stats=defaultdict(int);per_source=defaultdict(lambda:defaultdict(int));per_patient=[]
    def flush():
        nonlocal seen
        if current is None:return
        s,p=current;stats['subjects_total']+=1;per_source[s]['subjects_total']+=1
        if overlap or split=='metabonet_overlap_excluded':stats['overlap_subjects']+=1;per_source[s]['overlap_subjects']+=1
        else:
            stats[f'{split}_subjects']+=1;stats[f'{split}_valid_windows']+=seen;per_source[s][f'{split}_subjects']+=1;per_source[s][f'{split}_valid_windows']+=seen;per_patient.append((split,seen))
        seen=0
    for bi,b in enumerate(reader,1):
        f=b.to_pandas();stats['rows_scanned']+=len(f)
        for r in f.itertuples(index=False):
            key=(str(r.source_file),str(r.id))
            if key!=current:
                flush();current=key;overlap=bool(r.subject_split_across_traintest) if r.subject_split_across_traintest is not None else False;split=development_split(*key,overlap);cgm=set();ins=set()
            ts=pd.Timestamp(r.date)
            if finite(r.CGM):cgm.add(ts)
            # V15.1 requires the future timestamp row to exist, but insulin itself may be missing.
            ins.add(ts)
            anchor=ts-pd.Timedelta(minutes=120)
            if not overlap and anchor in cgm:
                hist=pd.date_range(anchor-pd.Timedelta(minutes=HISTORY_MINUTES-STEP),anchor,freq=f'{STEP}min')
                targets=[anchor+pd.Timedelta(minutes=h) for h in HORIZONS];future=pd.date_range(anchor+pd.Timedelta(minutes=STEP),anchor+pd.Timedelta(minutes=120),freq=f'{STEP}min')
                if len(hist)==SEQ_LEN and all(t in cgm for t in hist) and all(t in cgm for t in targets) and len(future)==FUTURE_STEPS and all(t in ins for t in future):seen+=1
            cutoff=ts-pd.Timedelta(minutes=HISTORY_MINUTES+120)
            cgm={t for t in cgm if t>=cutoff};ins={t for t in ins if t>=cutoff}
        if bi%20==0:print(f"scanned {stats['rows_scanned']:,}/{pf.metadata.num_rows:,} ({100*stats['rows_scanned']/pf.metadata.num_rows:.1f}%)",flush=True)
    flush();con.close()
    train_counts=np.array([n for s,n in per_patient if s=='train']);val_counts=np.array([n for s,n in per_patient if s=='validation'])
    def desc(x):return {'patients':int(len(x)),'valid_windows':int(x.sum()),'median_per_patient':float(np.median(x)) if len(x) else 0,'p90_per_patient':float(np.percentile(x,90)) if len(x) else 0,'max_per_patient':int(x.max()) if len(x) else 0}
    train_valid=int(stats['train_valid_windows']);val_valid=int(stats['validation_valid_windows']);train_cap24=int(sum(min(n,24) for n in train_counts));val_cap48=int(sum(min(n,48) for n in val_counts))
    result={'parquet_rows':int(pf.metadata.num_rows),'exact_v15_1_rules':True,'train':desc(train_counts),'validation':desc(val_counts),'current_caps':{'train_per_patient':24,'validation_per_patient':48,'train_windows_expected_after_cap':train_cap24,'validation_windows_expected_after_cap':val_cap48},'train_scale_vs_cap':float(train_valid/train_cap24) if train_cap24 else None,'validation_scale_vs_cap':float(val_valid/val_cap48) if val_cap48 else None,'by_source':{s:dict(v) for s,v in sorted(per_source.items())},'test_parquet_used':False,'live_targets_used':False,'model_trained':False}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('\n=== TOP-3 DATA SCALE AUDIT ===');print(f"train valid windows before cap: {train_valid:,}");print(f"train windows with cap=24:       {train_cap24:,}");print(f"train scale available:           {result['train_scale_vs_cap']:.2f}x");print(f"train median/p90/max per patient: {result['train']['median_per_patient']:.0f} / {result['train']['p90_per_patient']:.0f} / {result['train']['max_per_patient']:,}");print(f"validation valid before cap:      {val_valid:,}");print(f"validation with cap=48:           {val_cap48:,}");print(f"validation scale available:       {result['validation_scale_vs_cap']:.2f}x");print('\nBy source (train valid windows):')
    for s,v in sorted(per_source.items(),key=lambda kv:kv[1].get('train_valid_windows',0),reverse=True):print(f"  {s:14s} {v.get('train_valid_windows',0):12,d}  patients={v.get('train_subjects',0):4d}")
    print(f'\nReport: {out.resolve()}')
if __name__=='__main__':main()
