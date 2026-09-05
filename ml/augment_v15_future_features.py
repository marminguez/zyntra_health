"""Build a reusable V15 future-feature cache from frozen V15.1 anchors.

Reuses the exact x/y/timestamps already sampled for V15.1 and extracts future
insulin, basal, bolus and carbs at t+5..t+120 in one DuckDB equality join.
No resampling and no future CGM input.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

FUTURE_STEPS=24
FUTURE_NUMERIC=("insulin","basal","bolus","carbs")

def collect(root:Path):
    rows=[]; shards={}; aid=0
    for split in ("train","validation"):
        for p in sorted((root/split).glob("*.npz")):
            if "__" not in p.stem: raise ValueError(f"Unexpected shard name: {p.name}")
            source,pid=p.stem.split("__",1)
            with np.load(p,allow_pickle=False) as z: stamps=z["timestamp"].astype(str)
            ids=[]
            for i,t in enumerate(stamps): rows.append((aid,source,pid,pd.Timestamp(str(t))));ids.append(aid);aid+=1
            shards[(split,p.name)]={"path":p,"ids":np.asarray(ids,dtype=np.int64)}
    return pd.DataFrame(rows,columns=["anchor_id","source_file","id","anchor_date"]),shards

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--train-parquet",required=True);ap.add_argument("--v15-1-dir",default="ml/data/v15_1");ap.add_argument("--outdir",default="ml/data/v15_master");ap.add_argument("--memory-limit",default="3GB");a=ap.parse_args()
    root=Path(a.v15_1_dir);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);anchors,shards=collect(root)
    print(f"Frozen V15.1 anchors: {len(anchors):,}");print("Reusing x/y/timestamps exactly; no resampling");print(f"Caching future covariates: {FUTURE_NUMERIC}")
    con=duckdb.connect();con.execute(f"SET memory_limit='{a.memory_limit}'");con.register("anchors",anchors);ps=str(Path(a.train_parquet)).replace("'","''")
    cols=", ".join(f'"{c}"' for c in FUTURE_NUMERIC);selected=", ".join(f'p."{c}"' for c in FUTURE_NUMERIC)
    q=f"""WITH grid AS (
      SELECT a.anchor_id,s.step,a.source_file,a.id,a.anchor_date+s.step*INTERVAL '5 minutes' AS future_date
      FROM anchors a CROSS JOIN range(1,25) s(step)
    ), p AS (
      SELECT source_file,CAST(id AS VARCHAR) AS id,date,{cols} FROM read_parquet('{ps}')
    )
    SELECT g.anchor_id,g.step,{selected} FROM grid g LEFT JOIN p
      ON p.source_file=g.source_file AND p.id=g.id AND p.date=g.future_date
    ORDER BY g.anchor_id,g.step"""
    print("Scanning train.parquet once with DuckDB...",flush=True);f=con.execute(q).fetchdf();con.close()
    expected=len(anchors)*FUTURE_STEPS
    if len(f)!=expected: raise RuntimeError(f"Expected {expected:,} future rows, got {len(f):,}")
    vals={};miss={}
    for c in FUTURE_NUMERIC:
        ar=pd.to_numeric(f[c],errors="coerce").to_numpy(dtype=np.float32).reshape(len(anchors),FUTURE_STEPS);m=(~np.isfinite(ar)).astype(np.float32);vals[c]=np.where(np.isfinite(ar),ar,0).astype(np.float32);miss[c]=m
    counts={"train_windows":0,"validation_windows":0};pos={int(x):i for i,x in enumerate(anchors.anchor_id.to_numpy())}
    for (split,name),info in shards.items():
        with np.load(info["path"],allow_pickle=False) as z:x=z["x"].astype(np.float32);y=z["y"].astype(np.float32);stamp=z["timestamp"];old=z["future_insulin"].astype(np.float32)
        idx=np.asarray([pos[int(i)] for i in info["ids"]]);channels=[]
        for c in FUTURE_NUMERIC:channels.extend([vals[c][idx],miss[c][idx]])
        fk=np.stack(channels,axis=-1).astype(np.float32)
        comp=(old[:,:,1]<.5)&(fk[:,:,1]<.5)
        if np.any(comp) and not np.allclose(old[:,:,0][comp],fk[:,:,0][comp],atol=1e-6):raise RuntimeError(f"Future insulin mismatch: {name}")
        d=out/split;d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/name,x=x,future_known=fk,y=y,timestamp=stamp);counts[f"{split}_windows"]+=len(x)
    meta={"version":"v15_master","source":"exact frozen V15.1 sampled windows","resampled":False,"future_steps":24,"future_window":"t+5..t+120","future_channels":[v for c in FUTURE_NUMERIC for v in (c,f"{c}_missing")],"future_cgm_input":False,**counts};(out/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("\nV15 master dataset ready:");print(json.dumps(meta,indent=2));print(f"Artifacts written to {out}")
if __name__=="__main__":main()
