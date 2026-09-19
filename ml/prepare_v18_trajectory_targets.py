"""Prepare V18 24-step future-CGM targets while preserving existing V14/V15 inputs.

This is intentionally a companion dataset: for each existing Scale-10 shard,
match its anchor timestamps back to the original MetaboNet parquet and write
future CGM at +5,+10,...,+120 min. Existing x/y shards are not modified.

Usage:
 python -m ml.prepare_v18_trajectory_targets --data-dir <MetaboNet dir>
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
    total=written=missing=0
    for split in ("train","validation"):
        srcdir=base/split
        if not srcdir.exists(): raise FileNotFoundError(srcdir)
        od=out/split;od.mkdir(parents=True,exist_ok=True)
        for shard in sorted(srcdir.glob("*.npz")):
            with np.load(shard,allow_pickle=False) as z:
                ts=pd.to_datetime(z["timestamp"]).to_numpy(dtype="datetime64[ns]")
            # Shard filename is source__patient; recover identifiers from the
            # corresponding data by querying anchors against sanitized names.
            anchors=[pd.Timestamp(t).isoformat() for t in ts]
            # Query only the time envelope; then identify the source/patient by
            # the sanitized shard stem, avoiding assumptions about raw IDs.
            lo=(pd.Timestamp(ts.min())-pd.Timedelta(minutes=1)).isoformat()
            hi=(pd.Timestamp(ts.max())+pd.Timedelta(minutes=120)).isoformat()
            p=str(parquet).replace("'","''")
            df=con.execute(f"""SELECT source_file,id,date,CGM FROM read_parquet('{p}')
                WHERE date >= ? AND date <= ? AND CGM IS NOT NULL ORDER BY source_file,id,date""",[lo,hi]).df()
            stem=shard.stem
            matches=[]
            for (source,pid),g in df.groupby(["source_file","id"],sort=False):
                if f"{safe(source)}__{safe(pid)}"==stem:
                    matches.append(g)
            if len(matches)!=1:
                raise RuntimeError(f"{shard}: expected one source/patient match, got {len(matches)}")
            g=matches[0];lookup={pd.Timestamp(t):float(v) for t,v in zip(g.date,g.CGM)}
            fut=np.full((len(ts),len(STEPS)),np.nan,dtype=np.float32)
            for i,t in enumerate(pd.to_datetime(ts)):
                for j,m in enumerate(STEPS):
                    fut[i,j]=lookup.get(t+pd.Timedelta(minutes=int(m)),np.nan)
            ok=np.isfinite(fut).all(axis=1);total+=len(ts);missing+=int((~ok).sum())
            if not ok.all():
                raise RuntimeError(f"{shard}: {int((~ok).sum())} anchors lack complete 5-min future CGM")
            np.savez_compressed(od/shard.name,y_trajectory=fut,timestamp=np.asarray([str(pd.Timestamp(t)) for t in ts]))
            written+=len(ts)
            print(f"{split}: {shard.name} -> {len(ts)}",flush=True)
    con.close()
    np.savez_compressed(out/"metadata.npz",future_steps_minutes=STEPS)
    print(f"V18 targets written: {written:,}/{total:,} | incomplete: {missing:,} | out: {out}")
if __name__=="__main__":main()
