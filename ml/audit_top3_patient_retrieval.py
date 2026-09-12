"""Top-3 patient-memory retrieval audit using a temporal split inside V15 master train.

Why: the frozen V15 train/validation split is patient-disjoint by design, so it
cannot test same-patient memory. This diagnostic instead uses each patient's own
V15 master train windows, sorted by time: the earliest fraction is retrieval
memory and the latest fraction is query/evaluation. Retrieval candidates must end
at least 120 minutes before the query anchor. No Live targets are read and no
model is trained.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

HORIZONS=(30,60,90,120)

def load_train(root:Path):
    files=sorted((root/'train').glob('*.npz'))
    if not files: raise FileNotFoundError(f'No NPZ shards in {root/"train"}')
    patients=[]
    for p in files:
        if '__' not in p.stem: raise ValueError(f'Unexpected V15 master shard name: {p.name}')
        source,pid=p.stem.split('__',1)
        with np.load(p,allow_pickle=False) as z:
            x=z['x'].astype(np.float32); y=z['y'].astype(np.float32); ts=z['timestamp'].astype('datetime64[ns]')
        if not (len(x)==len(y)==len(ts)): raise ValueError(f'{p}: inconsistent lengths')
        order=np.argsort(ts)
        patients.append((source,pid,x[order],y[order],ts[order]))
    return patients

def rep(x):
    # Compact history-only state: last 2h, every 15 min, selected V14 channels.
    tail=x[:,-24:,:]
    idx=[0,1,2,3,5,7,9,11]
    return tail[:,::3,idx].reshape(len(x),-1).astype(np.float32)

def metrics(y,p):
    e=p-y; rmse=np.sqrt(np.mean(e*e,axis=0)); mae=np.mean(np.abs(e),axis=0)
    mard=100*np.mean(np.abs(e)/np.maximum(np.abs(y),1e-6),axis=0)
    return {'n':int(len(y)),'rmse':rmse.tolist(),'mae':mae.tolist(),'mard':mard.tolist(),'rmse_mean':float(rmse.mean()),'mard_mean':float(mard.mean())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--master',default='ml/data/v15_master')
    ap.add_argument('--output',default='ml/results/top3_audit/patient_retrieval.json')
    ap.add_argument('--k',type=int,default=5)
    ap.add_argument('--memory-fraction',type=float,default=.70)
    ap.add_argument('--min-windows',type=int,default=20)
    ap.add_argument('--max-memory-per-patient',type=int,default=5000)
    a=ap.parse_args()
    if not 0<a.memory_fraction<1: raise ValueError('--memory-fraction must be between 0 and 1')
    root=Path(a.master).resolve(); out=Path(a.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    print('Loading frozen V15 master TRAIN windows...')
    patients=load_train(root)
    pred=[]; truth=[]; oracle=[]; distances=[]; candidate_counts=[]; used_subjects=set(); query_count=0; skipped_small=0; skipped_no_prior=0
    gap=np.timedelta64(120,'m')
    for source,pid,x,y,ts in patients:
        n=len(x)
        if n<a.min_windows: skipped_small+=1; continue
        cut=max(1,min(n-1,int(np.floor(n*a.memory_fraction))))
        mem_idx=np.arange(cut,dtype=np.int64); qry_idx=np.arange(cut,n,dtype=np.int64)
        if len(mem_idx)>a.max_memory_per_patient:
            sel=np.linspace(0,len(mem_idx)-1,a.max_memory_per_patient,dtype=int); mem_idx=mem_idx[sel]
        rmem=rep(x[mem_idx]); rqry=rep(x[qry_idx])
        for qpos,j in enumerate(qry_idx):
            query_count+=1
            # Candidate's +120 target must be in the past relative to the query anchor.
            valid=ts[mem_idx]+gap <= ts[j]
            if not np.any(valid): skipped_no_prior+=1; continue
            cand_local=np.flatnonzero(valid); cand_idx=mem_idx[cand_local]
            d=np.mean((rmem[cand_local]-rqry[qpos])**2,axis=1)
            order=np.argsort(d)[:max(1,a.k)]; nn=cand_idx[order]
            w=1.0/(d[order]+1e-6); w=w/w.sum(); pp=(y[nn]*w[:,None]).sum(axis=0)
            pred.append(pp); truth.append(y[j]); distances.append(float(d[order[0]])); candidate_counts.append(len(cand_idx)); used_subjects.add((source,pid))
            best=nn[np.argmin(np.mean(np.abs(y[nn]-y[j]),axis=1))]; oracle.append(y[best])
    if not pred: raise RuntimeError('No eligible temporally separated same-patient retrieval queries found')
    pred=np.asarray(pred); truth=np.asarray(truth); oracle=np.asarray(oracle)
    result={
      'design':'within-patient chronological split of frozen V15 master train windows',
      'memory_fraction':a.memory_fraction,'k':a.k,
      'patients_total':len(patients),'patients_used':len(used_subjects),
      'queries_considered':query_count,'queries_eligible':len(pred),
      'skipped_small_patients':skipped_small,'skipped_no_temporally_safe_memory':skipped_no_prior,
      'median_prior_candidates':float(np.median(candidate_counts)),
      'median_nearest_distance':float(np.median(distances)),
      'knn':metrics(truth,pred),'oracle_best_of_k':metrics(truth,oracle),
      'live_targets_used':False,'model_trained':False,
      'temporal_guard':'candidate anchor + 120 min <= query anchor'
    }
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('\n=== TOP-3 PATIENT RETRIEVAL AUDIT ===')
    print(f"patients used: {len(used_subjects):,}/{len(patients):,}")
    print(f"eligible queries: {len(pred):,}/{query_count:,} ({100*len(pred)/max(query_count,1):.2f}%)")
    print(f"median prior candidates: {np.median(candidate_counts):,.0f}")
    print(f"median nearest distance: {np.median(distances):.5f}")
    for name in ('knn','oracle_best_of_k'):
        m=result[name]; print(f"\n{name}:")
        for h,r,ma in zip(HORIZONS,m['rmse'],m['mard']): print(f"  +{h}: RMSE={r:.3f} MARD={ma:.3f}%")
        print(f"  mean: RMSE={m['rmse_mean']:.3f} MARD={m['mard_mean']:.3f}%")
    print(f'\nReport: {out}')
if __name__=='__main__': main()
