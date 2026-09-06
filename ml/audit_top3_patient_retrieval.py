"""Top-3 patient-memory retrieval audit on frozen V15 master validation windows.

No Live targets are read and no model is trained. V15 master shards preserve the
original V15.1 shard filename as source__patient.npz and store anchor timestamps
under the singular key `timestamp`.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

HORIZONS=(30,60,90,120)

def load_split(root:Path, split:str):
    files=sorted((root/split).glob('*.npz'))
    if not files: raise FileNotFoundError(f'No NPZ shards in {root/split}')
    xs=[]; ys=[]; ts=[]; src=[]; ids=[]
    for p in files:
        if '__' not in p.stem: raise ValueError(f'Unexpected V15 master shard name: {p.name}')
        source,pid=p.stem.split('__',1)
        with np.load(p,allow_pickle=False) as z:
            x=z['x'].astype(np.float32); y=z['y'].astype(np.float32)
            if 'timestamp' not in z.files: raise KeyError(f'{p}: expected timestamp; keys={z.files}')
            stamp=z['timestamp'].astype(str)
        n=len(x)
        if len(y)!=n or len(stamp)!=n: raise ValueError(f'{p}: inconsistent lengths x={n}, y={len(y)}, timestamp={len(stamp)}')
        xs.append(x); ys.append(y); ts.append(stamp)
        src.append(np.full(n,source,dtype=object)); ids.append(np.full(n,pid,dtype=object))
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(ts),np.concatenate(src),np.concatenate(ids)

def rep(x):
    tail=x[:,-24:,:]
    idx=[0,1,2,3,5,7,9,11]
    return tail[:,::3,idx].reshape(len(x),-1).astype(np.float32)

def metrics(y,p):
    e=p-y; rmse=np.sqrt(np.mean(e*e,axis=0)); mae=np.mean(np.abs(e),axis=0)
    denom=np.maximum(np.abs(y),1e-6); mard=100*np.mean(np.abs(e)/denom,axis=0)
    return {'n':int(len(y)),'rmse':rmse.tolist(),'mae':mae.tolist(),'mard':mard.tolist(),'rmse_mean':float(rmse.mean()),'mard_mean':float(mard.mean())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--master',default='ml/data/v15_master'); ap.add_argument('--output',default='ml/results/top3_audit/patient_retrieval.json'); ap.add_argument('--k',type=int,default=5); ap.add_argument('--max-train-per-patient',type=int,default=5000); a=ap.parse_args()
    root=Path(a.master).resolve(); out=Path(a.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    print('Loading frozen V15 master train/validation...')
    xtr,ytr,ttr,str_,itr=load_split(root,'train'); xv,yv,tv,sv,iv=load_split(root,'validation')
    rtr=rep(xtr); rv=rep(xv); groups={}
    for i,(s,p) in enumerate(zip(str_,itr)): groups.setdefault((s,p),[]).append(i)
    pred=[]; truth=[]; oracle=[]; distances=[]; subjects=set(); eligible=0; prior_candidates=[]
    for j,(s,p,t) in enumerate(zip(sv,iv,tv)):
        inds=groups.get((s,p))
        if not inds: continue
        cand=np.asarray([i for i in inds if ttr[i] < t],dtype=np.int64)
        if len(cand)==0: continue
        if len(cand)>a.max_train_per_patient:
            sel=np.linspace(0,len(cand)-1,a.max_train_per_patient,dtype=int); cand=cand[sel]
        eligible+=1; subjects.add((s,p)); prior_candidates.append(len(cand))
        d=np.mean((rtr[cand]-rv[j])**2,axis=1); order=np.argsort(d)[:max(1,a.k)]; nn=cand[order]
        w=1.0/(d[order]+1e-6); w=w/w.sum(); pred.append((ytr[nn]*w[:,None]).sum(axis=0)); truth.append(yv[j]); distances.append(float(d[order[0]]))
        best=nn[np.argmin(np.mean(np.abs(ytr[nn]-yv[j]),axis=1))]; oracle.append(ytr[best])
    if not pred: raise RuntimeError('No same-patient temporally prior retrieval matches found')
    pred=np.asarray(pred); truth=np.asarray(truth); oracle=np.asarray(oracle)
    result={'eligible_validation_windows':eligible,'total_validation_windows':int(len(yv)),'eligible_fraction':eligible/len(yv),'eligible_subjects':len(subjects),'k':a.k,'median_prior_candidates':float(np.median(prior_candidates)),'median_nearest_distance':float(np.median(distances)),'knn':metrics(truth,pred),'oracle_best_of_k':metrics(truth,oracle),'targets_used':'frozen internal validation y only','live_targets_used':False,'model_trained':False}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('\n=== TOP-3 PATIENT RETRIEVAL AUDIT ==='); print(f"eligible windows: {eligible:,}/{len(yv):,} ({100*eligible/len(yv):.2f}%)"); print(f"eligible subjects: {len(subjects):,}"); print(f"median prior candidates: {np.median(prior_candidates):,.0f}"); print(f"median nearest distance: {np.median(distances):.5f}")
    for name in ('knn','oracle_best_of_k'):
        m=result[name]; print(f"\n{name}:")
        for h,r,ma in zip(HORIZONS,m['rmse'],m['mard']): print(f"  +{h}: RMSE={r:.3f} MARD={ma:.3f}%")
        print(f"  mean: RMSE={m['rmse_mean']:.3f} MARD={m['mard_mean']:.3f}%")
    print(f'\nReport: {out}')
if __name__=='__main__': main()
