"""Prepare V18 24-step future-CGM targets for the exact existing Scale-10 anchors.

Intermediate CGM can be missing even when the frozen +30/+60/+90/+120 targets
exist. We preserve every frozen anchor and write a finite-value mask; V18 must
mask missing intermediate targets during training. Existing shards are untouched.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import duckdb,numpy as np,pandas as pd

STEPS=np.arange(5,121,5,dtype=np.int32)

def safe(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True,help="Directory containing original train.parquet")
    ap.add_argument("--base-shards",default="ml/data/v15_4_scale10")
    ap.add_argument("--outdir",default="ml/data/v18_scale10")
    ap.add_argument("--memory-limit",default="2GB")
    a=ap.parse_args()
    parquet=Path(a.data_dir)/"train.parquet"
    if not parquet.exists(): raise FileNotFoundError(parquet)
    base=Path(a.base_shards);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect();con.execute(f"SET memory_limit='{a.memory_limit}'")
    total=written=incomplete_anchors=missing_points=0
    missing_by_step=np.zeros(len(STEPS),dtype=np.int64)
    for split in ("train","validation"):
        srcdir=base/split
        if not srcdir.exists(): raise FileNotFoundError(srcdir)
        od=out/split;od.mkdir(parents=True,exist_ok=True)
        for shard in sorted(srcdir.glob("*.npz")):
            with np.load(shard,allow_pickle=False) as z:
                ts=pd.to_datetime(z["timestamp"]).to_numpy(dtype="datetime64[ns]")
            lo=(pd.Timestamp(ts.min())-pd.Timedelta(minutes=1)).isoformat()
            hi=(pd.Timestamp(ts.max())+pd.Timedelta(minutes=120)).isoformat()
            p=str(parquet).replace("'","''")
            df=con.execute(f"""SELECT source_file,id,date,CGM FROM read_parquet('{p}')
                WHERE date >= ? AND date <= ? AND CGM IS NOT NULL
                ORDER BY source_file,id,date""",[lo,hi]).df()
            stem=shard.stem;matches=[]
            for (source,pid),g in df.groupby(["source_file","id"],sort=False):
                if f"{safe(source)}__{safe(pid)}"==stem: matches.append(g)
            if len(matches)!=1:
                raise RuntimeError(f"{shard}: expected one source/patient match, got {len(matches)}")
            g=matches[0];lookup={pd.Timestamp(t):float(v) for t,v in zip(g.date,g.CGM)}
            fut=np.full((len(ts),len(STEPS)),np.nan,dtype=np.float32)
            for i,t in enumerate(pd.to_datetime(ts)):
                for j,m in enumerate(STEPS):
                    fut[i,j]=lookup.get(t+pd.Timedelta(minutes=int(m)),np.nan)
            mask=np.isfinite(fut)
            incomplete=~mask.all(axis=1)
            total+=len(ts);written+=len(ts)
            incomplete_anchors+=int(incomplete.sum());missing_points+=int((~mask).sum())
            missing_by_step+=(~mask).sum(axis=0)
            # Fill only for storage/numerical safety; mask is authoritative.
            # Linear interpolation is NOT treated as ground truth.
            filled=fut.copy()
            for i in range(len(filled)):
                good=np.flatnonzero(mask[i])
                if len(good):
                    bad=np.flatnonzero(~mask[i])
                    filled[i,bad]=np.interp(bad,good,filled[i,good])
            np.savez_compressed(od/shard.name,y_trajectory=filled,target_mask=mask.astype(np.uint8),
                                timestamp=np.asarray([str(pd.Timestamp(t)) for t in ts]))
            print(f"{split}: {shard.name} -> {len(ts)} | incomplete anchors: {int(incomplete.sum())}",flush=True)
    con.close()
    np.savez_compressed(out/"metadata.npz",future_steps_minutes=STEPS,missing_by_step=missing_by_step,
                        total_anchors=np.asarray(total),incomplete_anchors=np.asarray(incomplete_anchors),
                        missing_points=np.asarray(missing_points))
    print(f"V18 targets written: {written:,}/{total:,} | incomplete anchors retained: {incomplete_anchors:,} | missing points masked: {missing_points:,}")
    print("Missing points by step:",dict(zip(STEPS.tolist(),missing_by_step.tolist())))
    print(f"out: {out}")
if __name__=="__main__":main()
