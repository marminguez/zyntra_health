"""Selective routing audit for the FIR-1 excursion specialist.

Fits gate + positive-residual specialist on calibration only, then selects a
probability threshold and correction strength on calibration only. The chosen
routing policy is frozen before one holdout evaluation.

No future CGM input. Future targets are labels/residual targets only.
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
    return np.concatenate(xs),np.concatenate(fs),np.concatenate(ys)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
def fit_logistic(x,y,steps=1500,lr=.03,l2=1e-3):
    w=np.zeros(x.shape[1]); b=0.; pos=max(y.sum(),1.); neg=max(len(y)-y.sum(),1.)
    sw=np.where(y>0,len(y)/(2*pos),len(y)/(2*neg))
    for _ in range(steps):
        p=sigmoid(x@w+b); e=(p-y)*sw; w-=lr*((x.T@e)/len(y)+l2*w); b-=lr*e.mean()
    return w,b
def ridge(x,y,l2=10.):
    xa=np.c_[np.ones(len(x)),x]; reg=np.eye(xa.shape[1])*l2; reg[0,0]=0
    return np.linalg.solve(xa.T@xa+reg,xa.T@y)
def make_features(x,future,fir,current):
    r=min(12,x.shape[1]); bol=np.where(x[:,-r:,8]<.5,x[:,-r:,7],0).sum(1); car=np.where(x[:,-r:,12]<.5,x[:,-r:,11],0).sum(1); ins=np.where(x[:,-r:,10]<.5,x[:,-r:,9],0).sum(1); bas=np.where(x[:,-1,6]<.5,x[:,-1,5],0)
    sm=[]
    for s in (6,12,18,24):
        q=future[:,:s,:]; sm += [q.sum(1),q.max(1),q.mean(1)]
    a=np.concatenate([current[:,None],x[:,-1,1:5],bol[:,None],car[:,None],ins[:,None],bas[:,None],fir,fir-current[:,None]]+sm,1).astype(float); a[~np.isfinite(a)]=0; return a
def scores(y,p,current):
    e=p-y; mae=np.mean(np.abs(e),0); rmse=np.sqrt(np.mean(e**2,0)); mard=np.mean(np.abs(e)/np.maximum(np.abs(y),1e-6),0)*100
    dc=lambda d:np.where(d>5,1,np.where(d<-5,-1,0)); direction=np.mean(dc(y-current[:,None])==dc(p-current[:,None]),0)
    return mae,rmse,mard,direction
def objective(y,p,current,base):
    _,r,m,_=scores(y,p,current); _,br,bm,_=base
    # Competition-aware but conservative: average relative MARD + RMSE change.
    dm=np.mean((m/bm)-1); dr=np.mean((r/br)-1)
    return float(dm+dr),float(dm),float(dr)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="ml/data/v15_4_scale50"); ap.add_argument("--predictions",default="ml/results/fir1_scale50/validation_predictions.npz"); ap.add_argument("--outdir",default="ml/results/selective_excursion_routing"); a=ap.parse_args()
    x,f,ydata=load_validation(a.data_dir)
    with np.load(a.predictions,allow_pickle=False) as z: y=z["y_true"].astype(float); current=z["current_glucose"].astype(float); fir=z["y_pred"].astype(float)
    if not np.allclose(y,ydata,atol=1e-4,rtol=0) or not np.allclose(current,x[:,-1,0],atol=1e-4,rtol=0): raise ValueError("alignment failed")
    feat=make_features(x,f,fir,current); order=np.random.default_rng(SEED).permutation(len(y)); cut=len(y)//2; cal,hold=order[:cut],order[cut:]
    mu=feat[cal].mean(0); sd=feat[cal].std(0); sd[sd<1e-6]=1; xc=(feat[cal]-mu)/sd; xh=(feat[hold]-mu)/sd
    pc=np.zeros((len(cal),4)); ph=np.zeros((len(hold),4)); rc=np.zeros_like(pc); rh=np.zeros_like(ph)
    for j in range(4):
        lab=(y[:,j]-current>=40).astype(float); w,b=fit_logistic(xc,lab[cal]); pc[:,j]=sigmoid(xc@w+b); ph[:,j]=sigmoid(xh@w+b)
        target=np.maximum(y[cal,j]-fir[cal,j],0); rw=ridge(np.c_[xc,pc[:,j]],target); rc[:,j]=np.clip(np.c_[np.ones(len(cal)),xc,pc[:,j]]@rw,0,80); rh[:,j]=np.clip(np.c_[np.ones(len(hold)),xh,ph[:,j]]@rw,0,80)
    basecal=scores(y[cal],fir[cal],current[cal]); candidates=[]
    # One global policy for all horizons: avoids 8-parameter per-horizon overfitting.
    for t in np.arange(.50,.91,.05):
        for strength in (.25,.50,.75,1.0):
            pred=fir[cal]+(pc>=t)*rc*strength; obj,dm,dr=objective(y[cal],pred,current[cal],basecal)
            flag=float(np.mean(pc>=t)); candidates.append((obj,t,strength,dm,dr,flag))
    candidates.sort(key=lambda q:q[0]); best=candidates[0]; _,threshold,strength,cal_dm,cal_dr,cal_flag=best
    final=fir[hold]+(ph>=threshold)*rh*strength
    mb=scores(y[hold],fir[hold],current[hold]); ms=scores(y[hold],final,current[hold])
    mae_b,r_b,m_b,d_b=mb; mae_s,r_s,m_s,d_s=ms
    dm=100*(m_s.mean()/m_b.mean()-1); dr=100*(r_s.mean()/r_b.mean()-1); dd=100*(d_s.mean()-d_b.mean())
    rows=[]; lr_good=0
    for j,h in enumerate(HORIZONS):
        mask=y[hold,j]-current[hold]>=40; rb=np.sqrt(np.mean((fir[hold,j][mask]-y[hold,j][mask])**2)); rs=np.sqrt(np.mean((final[:,j][mask]-y[hold,j][mask])**2)); rel=100*(rs/rb-1); lr_good+=rel<=-3
        rows.append({"horizon_minutes":h,"baseline_rmse":r_b[j],"specialist_rmse":r_s[j],"baseline_mard":m_b[j],"specialist_mard":m_s[j],"baseline_direction":d_b[j],"specialist_direction":d_s[j],"large_rise_n":int(mask.sum()),"large_rise_rmse_baseline":rb,"large_rise_rmse_routed":rs,"large_rise_rmse_relative_pct":rel,"holdout_flag_rate":float(np.mean(ph[:,j]>=threshold))})
    df=pd.DataFrame(rows); go=bool(((dm<=-3) or (dr<=-3)) and dm<=.25 and dr<=.25 and lr_good>=3)
    print("=== SELECTIVE EXCURSION ROUTING AUDIT ==="); print(f"rows: {len(y):,} | calibration: {len(cal):,} | holdout: {len(hold):,}"); print("future CGM input: False | live targets used: False | FIR-1 frozen: True")
    print("\nCALIBRATION-SELECTED POLICY"); print(f"threshold: {threshold:.2f} | correction strength: {strength:.2f} | mean flag rate: {100*cal_flag:.2f}%"); print(f"calibration MARD relative: {100*cal_dm:+.3f}% | RMSE relative: {100*cal_dr:+.3f}%")
    print("\nHOLDOUT BY HORIZON"); print(df.to_string(index=False))
    print("\n4H HOLDOUT COMPARISON"); print(f"MARD: {m_b.mean():.6f} -> {m_s.mean():.6f} ({dm:+.3f}% relative)"); print(f"RMSE: {r_b.mean():.6f} -> {r_s.mean():.6f} ({dr:+.3f}% relative)"); print(f"Direction: {100*d_b.mean():.4f}% -> {100*d_s.mean():.4f}% ({dd:+.3f} pp)")
    print("\nLARGE-RISE HOLDOUT");
    for q in rows: print(f"+{q['horizon_minutes']:3d} | n={q['large_rise_n']:4d} | RMSE {q['large_rise_rmse_baseline']:.3f} -> {q['large_rise_rmse_routed']:.3f} ({q['large_rise_rmse_relative_pct']:+.2f}%) | flagged {100*q['holdout_flag_rate']:.1f}%")
    print("\nSTRUCTURAL GATE"); print(f"large-rise horizons improved >=3%: {lr_good}/4"); print(f"SELECTIVE ROUTING: {'GO' if go else 'NO-GO'}")
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); df.to_csv(out/"holdout_metrics.csv",index=False); pd.DataFrame(candidates,columns=["objective","threshold","strength","mard_rel","rmse_rel","flag_rate"]).to_csv(out/"calibration_grid.csv",index=False); np.savez_compressed(out/"holdout_predictions.npz",indices=hold,y_true=y[hold],current_glucose=current[hold],fir_pred=fir[hold],routed_pred=final)
    report={"experiment":"selective excursion routing","seed":SEED,"future_cgm_input":False,"live_targets_used":False,"fir1_frozen":True,"policy":{"threshold":threshold,"strength":strength,"calibration_flag_rate":cal_flag},"holdout":{"mard_relative_pct":dm,"rmse_relative_pct":dr,"direction_pp":dd,"large_rise_horizons_improved_ge3pct":int(lr_good)},"gate":">=3% global MARD or RMSE improvement; other <=0.25% regression; >=3/4 large-rise RMSE improve >=3%","go":go}
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(f"\nReport: {out/'report.json'}")
if __name__=="__main__": main()
