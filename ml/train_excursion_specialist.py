"""Leakage-safe rapid screen for a FIR-1 excursion residual specialist.

FIR-1 predictions stay frozen. A prospective gate estimates P(large rise) and a
small ridge residual model estimates FIR-1 underprediction. Both are fitted on a
deterministic calibration half only. Final predictions are evaluated once on the
disjoint holdout half.

No future CGM is used as an input. Future targets are labels/residual targets only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS=(30,60,90,120); SEED=42

def load_validation(d):
    xs,fs,ys=[],[],[]
    for p in sorted((Path(d)/"validation").glob("*.npz")):
        with np.load(p,allow_pickle=False) as z:
            xs.append(z["x"].astype(np.float32)); fs.append(z["future_known"].astype(np.float32)); ys.append(z["y"].astype(np.float32))
    if not xs: raise ValueError("No validation shards")
    return np.concatenate(xs),np.concatenate(fs),np.concatenate(ys)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))

def fit_logistic(x,y,steps=1500,lr=.03,l2=1e-3):
    w=np.zeros(x.shape[1]); b=0.; pos=max(y.sum(),1.); neg=max(len(y)-y.sum(),1.)
    sw=np.where(y>0,len(y)/(2*pos),len(y)/(2*neg))
    for _ in range(steps):
        p=sigmoid(x@w+b); e=(p-y)*sw
        w-=lr*((x.T@e)/len(y)+l2*w); b-=lr*e.mean()
    return w,b

def ridge(x,y,l2=10.):
    xa=np.c_[np.ones(len(x)),x]
    reg=np.eye(xa.shape[1])*l2; reg[0,0]=0
    return np.linalg.solve(xa.T@xa+reg,xa.T@y)

def make_features(x,future,fir,current):
    recent=min(12,x.shape[1])
    bolus=np.where(x[:,-recent:,8]<.5,x[:,-recent:,7],0).sum(1)
    carbs=np.where(x[:,-recent:,12]<.5,x[:,-recent:,11],0).sum(1)
    insulin=np.where(x[:,-recent:,10]<.5,x[:,-recent:,9],0).sum(1)
    basal=np.where(x[:,-1,6]<.5,x[:,-1,5],0)
    summaries=[]
    for steps in (6,12,18,24):
        f=future[:,:steps,:]
        summaries += [f.sum(1),f.max(1),f.mean(1)]
    parts=[current[:,None],x[:,-1,1:5],bolus[:,None],carbs[:,None],insulin[:,None],basal[:,None],fir,fir-current[:,None]]+summaries
    a=np.concatenate(parts,1).astype(np.float64); a[~np.isfinite(a)]=0
    return a

def metric(y,p,current):
    out=[]
    for j,h in enumerate(HORIZONS):
        e=p[:,j]-y[:,j]
        dc=lambda d:np.where(d>5,1,np.where(d<-5,-1,0))
        out.append({"horizon_minutes":h,"n":len(y),"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e**2))),"mard":float(np.mean(np.abs(e)/np.maximum(np.abs(y[:,j]),1e-6))*100),"direction_accuracy":float(np.mean(dc(y[:,j]-current)==dc(p[:,j]-current)))})
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="ml/data/v15_4_scale50"); ap.add_argument("--predictions",default="ml/results/fir1_scale50/validation_predictions.npz"); ap.add_argument("--outdir",default="ml/results/excursion_specialist"); a=ap.parse_args()
    x,f,ydata=load_validation(a.data_dir)
    with np.load(a.predictions,allow_pickle=False) as z:
        y=z["y_true"].astype(np.float64); current=z["current_glucose"].astype(np.float64); fir=z["y_pred"].astype(np.float64)
    if len(y)!=len(x) or not np.allclose(y,ydata,atol=1e-4,rtol=0): raise ValueError("alignment failed")
    feat=make_features(x,f,fir,current); rng=np.random.default_rng(SEED); order=rng.permutation(len(y)); cut=len(y)//2; cal,hold=order[:cut],order[cut:]
    mu=feat[cal].mean(0); sd=feat[cal].std(0); sd[sd<1e-6]=1; xc=(feat[cal]-mu)/sd; xh=(feat[hold]-mu)/sd
    corrected=fir[hold].copy(); details=[]
    for j,h in enumerate(HORIZONS):
        large=(y[:,j]-current>=40).astype(float); w,b=fit_logistic(xc,large[cal]); pc=sigmoid(xc@w+b); ph=sigmoid(xh@w+b)
        # Residual specialist learns positive FIR-1 underprediction on calibration only.
        residual=np.maximum(y[cal,j]-fir[cal,j],0)
        rw=ridge(np.c_[xc,pc],residual,l2=10.)
        raw=np.c_[np.ones(len(hold)),xh,ph]@rw
        raw=np.clip(raw,0,80)
        # Continuous gating: no post-hoc threshold. High excursion probability receives more correction.
        corr=ph*raw
        corrected[:,j]=fir[hold,j]+corr
        mask=(y[hold,j]-current[hold]>=40)
        base_lr=float(np.sqrt(np.mean((fir[hold,j][mask]-y[hold,j][mask])**2))) if mask.any() else np.nan
        new_lr=float(np.sqrt(np.mean((corrected[:,j][mask]-y[hold,j][mask])**2))) if mask.any() else np.nan
        details.append({"horizon_minutes":h,"large_rise_n":int(mask.sum()),"mean_gate_probability":float(ph.mean()),"mean_correction":float(corr.mean()),"large_rise_rmse_baseline":base_lr,"large_rise_rmse_specialist":new_lr,"large_rise_rmse_relative_pct":100*(new_lr/base_lr-1)})
    mb=metric(y[hold],fir[hold],current[hold]); ms=metric(y[hold],corrected,current[hold])
    bm,sm=mb.mard.mean(),ms.mard.mean(); br,sr=mb.rmse.mean(),ms.rmse.mean(); bd,sdirection=mb.direction_accuracy.mean(),ms.direction_accuracy.mean()
    dm=100*(sm/bm-1); dr=100*(sr/br-1); dd=100*(sdirection-bd)
    # Frozen structural gate: >=3% improvement in either MARD or RMSE, no regression in the other (>0.25%), and large-rise RMSE improves in >=3 horizons.
    lr_good=sum(d["large_rise_rmse_relative_pct"]<=-3 for d in details)
    go=bool(((dm<=-3) or (dr<=-3)) and dm<=.25 and dr<=.25 and lr_good>=3)
    print("=== FIR-1 EXCURSION SPECIALIST HOLDOUT ==="); print(f"rows total: {len(y):,} | calibration: {len(cal):,} | holdout: {len(hold):,}"); print("FIR-1 frozen: True | future CGM input: False | live targets used: False")
    print("\nBASELINE FIR-1"); print(mb.to_string(index=False)); print("\nFIR-1 + EXCURSION SPECIALIST"); print(ms.to_string(index=False))
    print("\n4H HOLDOUT COMPARISON"); print(f"MARD: {bm:.6f} -> {sm:.6f} ({dm:+.3f}% relative)"); print(f"RMSE: {br:.6f} -> {sr:.6f} ({dr:+.3f}% relative)"); print(f"Direction: {100*bd:.4f}% -> {100*sdirection:.4f}% ({dd:+.3f} pp)")
    print("\nLARGE-RISE RMSE");
    for d in details: print(f"+{d['horizon_minutes']:3d} | n={d['large_rise_n']:4d} | RMSE {d['large_rise_rmse_baseline']:.3f} -> {d['large_rise_rmse_specialist']:.3f} ({d['large_rise_rmse_relative_pct']:+.2f}%) | mean correction {d['mean_correction']:.2f}")
    print("\nSTRUCTURAL GATE"); print(f"large-rise horizons improved >=3%: {lr_good}/4"); print(f"EXCURSION SPECIALIST: {'GO' if go else 'NO-GO'}")
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); mb.to_csv(out/"holdout_baseline.csv",index=False); ms.to_csv(out/"holdout_specialist.csv",index=False); np.savez_compressed(out/"holdout_predictions.npz",indices=hold,y_true=y[hold],current_glucose=current[hold],fir_pred=fir[hold],specialist_pred=corrected)
    report={"experiment":"FIR-1 excursion specialist rapid holdout screen","seed":SEED,"calibration_n":len(cal),"holdout_n":len(hold),"future_cgm_input":False,"live_targets_used":False,"fir1_frozen":True,"correction":"P(large rise) * positive-residual ridge estimate","global_change":{"mard_relative_pct":dm,"rmse_relative_pct":dr,"direction_pp":dd},"large_rise":details,"gate":">=3% global MARD or RMSE improvement; other metric <=0.25% regression; >=3/4 large-rise RMSE improve >=3%","go":go}
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(f"\nReport: {out/'report.json'}")
if __name__=="__main__": main()
