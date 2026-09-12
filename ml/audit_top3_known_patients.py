"""Top-3 data audit for MetaboNet Live.

No model training and no targets.parquet access. Measures how much of the official
Live template belongs to subjects with train/test overlap, how much prior train
history exists for them, and how anchors are distributed by source.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import duckdb
import pandas as pd
import pyarrow.parquet as pq


def qpath(p: Path) -> str:
    return str(p.resolve()).replace("'", "''")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',required=True)
    ap.add_argument('--test',required=True)
    ap.add_argument('--template',required=True)
    ap.add_argument('--output',default='ml/results/top3_audit/known_patient_audit.json')
    ap.add_argument('--memory-limit',default='3GB')
    a=ap.parse_args()
    train,test,template,out=map(Path,[a.train,a.test,a.template,a.output]); out=out.resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    for p in (train,test,template):
        if not p.exists(): raise FileNotFoundError(p)
    t=pq.read_table(template).to_pandas(); t['date']=pd.to_datetime(t['date'],errors='raise')
    con=duckdb.connect(); con.execute(f"SET memory_limit='{a.memory_limit}'"); con.register('template_df',t[['source_file','id','date']])
    tr=qpath(train); te=qpath(test)
    # Subject-level train stats. This is the only full train scan.
    con.execute(f"""CREATE TEMP TABLE train_subjects AS
      SELECT CAST(source_file AS VARCHAR) source_file, CAST(id AS VARCHAR) id,
             MIN(date) train_first, MAX(date) train_last, COUNT(*) train_rows,
             SUM(CASE WHEN CGM IS NOT NULL THEN 1 ELSE 0 END) train_cgm_rows,
             MAX(CASE WHEN COALESCE(subject_split_across_traintest,false) THEN 1 ELSE 0 END) overlap_flag
      FROM read_parquet('{tr}') WHERE source_file IS NOT NULL AND id IS NOT NULL AND date IS NOT NULL
      GROUP BY 1,2""")
    con.execute(f"""CREATE TEMP TABLE test_subjects AS
      SELECT CAST(source_file AS VARCHAR) source_file, CAST(id AS VARCHAR) id,
             MIN(date) test_first, MAX(date) test_last, COUNT(*) test_rows,
             MAX(CASE WHEN COALESCE(subject_split_across_traintest,false) THEN 1 ELSE 0 END) overlap_flag
      FROM read_parquet('{te}') WHERE source_file IS NOT NULL AND id IS NOT NULL AND date IS NOT NULL
      GROUP BY 1,2""")
    con.execute("""CREATE TEMP TABLE live AS
      SELECT CAST(source_file AS VARCHAR) source_file, CAST(id AS VARCHAR) id, date FROM template_df""")
    overall=con.execute("""SELECT COUNT(*) anchors,
      SUM(CASE WHEN tr.id IS NOT NULL THEN 1 ELSE 0 END) anchors_subject_seen_in_train,
      SUM(CASE WHEN tr.overlap_flag=1 OR ts.overlap_flag=1 THEN 1 ELSE 0 END) anchors_flagged_overlap,
      COUNT(DISTINCT l.source_file||'|'||l.id) live_subjects,
      COUNT(DISTINCT CASE WHEN tr.id IS NOT NULL THEN l.source_file||'|'||l.id END) live_subjects_seen_in_train,
      COUNT(DISTINCT CASE WHEN tr.overlap_flag=1 OR ts.overlap_flag=1 THEN l.source_file||'|'||l.id END) live_subjects_flagged_overlap
      FROM live l LEFT JOIN train_subjects tr USING(source_file,id) LEFT JOIN test_subjects ts USING(source_file,id)""").fetchdf().iloc[0].to_dict()
    by_source=con.execute("""SELECT l.source_file,COUNT(*) anchors,
      SUM(CASE WHEN tr.id IS NOT NULL THEN 1 ELSE 0 END) seen_train,
      SUM(CASE WHEN tr.overlap_flag=1 OR ts.overlap_flag=1 THEN 1 ELSE 0 END) flagged_overlap,
      COUNT(DISTINCT l.id) subjects,
      COUNT(DISTINCT CASE WHEN tr.id IS NOT NULL THEN l.id END) subjects_seen_train
      FROM live l LEFT JOIN train_subjects tr USING(source_file,id) LEFT JOIN test_subjects ts USING(source_file,id)
      GROUP BY 1 ORDER BY anchors DESC""").fetchdf()
    prior=con.execute("""SELECT COUNT(*) anchors_seen_train,
      SUM(CASE WHEN tr.train_last < l.date THEN 1 ELSE 0 END) anchors_with_strictly_prior_train,
      MEDIAN(date_diff('day',tr.train_first,tr.train_last)) median_train_span_days,
      MEDIAN(tr.train_rows) median_train_rows,
      MEDIAN(tr.train_cgm_rows) median_train_cgm_rows,
      MEDIAN(date_diff('day',tr.train_last,l.date)) median_days_trainlast_to_anchor
      FROM live l JOIN train_subjects tr USING(source_file,id)""").fetchdf().iloc[0].to_dict()
    # Check whether the explicit flag and actual same-subject presence agree.
    mismatch=con.execute("""SELECT
      SUM(CASE WHEN tr.id IS NOT NULL AND NOT (tr.overlap_flag=1 OR ts.overlap_flag=1) THEN 1 ELSE 0 END) seen_but_not_flagged,
      SUM(CASE WHEN tr.id IS NULL AND (ts.overlap_flag=1) THEN 1 ELSE 0 END) flagged_but_not_seen
      FROM live l LEFT JOIN train_subjects tr USING(source_file,id) LEFT JOIN test_subjects ts USING(source_file,id)""").fetchdf().iloc[0].to_dict()
    result={'overall':overall,'prior_train_history':prior,'flag_consistency':mismatch,'by_source':by_source.to_dict(orient='records'),'targets_used':False,'model_trained':False}
    def clean(v):
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list): return [clean(x) for x in v]
        if hasattr(v,'item'): return v.item()
        return v
    result=clean(result); out.write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print('\n=== TOP-3 KNOWN-PATIENT AUDIT ===')
    for k,v in overall.items(): print(f'{k}: {v:,}' if isinstance(v,(int,float)) else f'{k}: {v}')
    print('\nPrior train history:'); [print(f'  {k}: {v}') for k,v in prior.items()]
    print('\nFlag consistency:'); [print(f'  {k}: {v}') for k,v in mismatch.items()]
    print('\nBy source:'); print(by_source.to_string(index=False))
    print(f'\nReport: {out}')
    con.close()
if __name__=='__main__': main()
