"""Prospective audit for FIR-1 hidden large excursions.

Hidden excursion at horizon h:
  true rise >= 40 mg/dL AND frozen FIR-1 predicted rise < 20 mg/dL.

The FIR-1 prediction is known at inference, so this target asks whether anchor-available
signals can identify cases where the frozen champion is about to miss a large rise.
Future glucose targets define labels only and are never features.
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
    w=np.zeros(x.shape[1]); b=0.; pos=max(float(y.sum()),1.); neg=max(float(len(y)-y.sum()),1.); sw=np.where(y>0,len(y)/(2*pos),len(y)/(2*neg))
    for _ in range(steps):
        p=sigmoid(x@w+b); e=(p-y)*sw; w-=lr*((x.T@e)/len(y)+l2*w); b-=lr*e.mean()
    return w,b
def confusion(y,p,t):
    q=p>=t; y=y.astype(bool); tp=int(np.sum(q&y)); fp=int(np.sum(q&~y)); fn=int(np.sum(~q&y)); tn=int(np.sum(~q&~y)); prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-12)
    return {"precision":prec,"recall":rec,"f1":f1,"specificity":tn/max(tn+fp,1),"flag_rate":float(q.mean())}
def auc_rank(y,s):
    y=y.astype(bool); np_=int(y.sum()); nn=int((~y).sum())
    if not np_ or not nn:return float('nan')
    o=np.argsort(s); ranks=np.empty(len(s)); ranks[o]=np.arange(1,len(s)+1); return float((ranks[y].sum()-np_*(np_+1)/2)/(np_*nn))
def features(x,f,fir,current):
    r=min(12,x.shape[1]); bol=np.where(x[:,-r:,8]<.5,x[:,-r:,7],0).sum(1); car=np.where(x[:,-r:,12]<.5,x[:,-r:,11],0).sum(1); ins=np.where(x[:,-r:,10]<.5,x[:,-r:,9],0).sum(1); bas=np.where(x[:,-1,6]<.5,x[:,-1,5],0)
    sm=[]
    for s in (6,12,18,24): q=f[:,:s,:]; sm += [q.sum(1),q.max(1),q.mean(1)]
    a=np.concatenate([current[:,None],x[:,-1,1:5],bol[:,None],car[:,None],ins[:,None],bas[:,None],fir,fir-current[:,None]]+sm,1).astype(float); a[~np.isfinite(a)]=0; return a

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='ml/data/v15_4_scale50'); ap.add_argument('--predictions',default='ml/results/fir1_scale50/validation_predictions.npz'); ap.add_argument('--outdir',default='ml/results/hidden_excursion_predictability'); a=ap.parse_args()
    x,f,yd=load_validation(Path(a.data_dir))
    with np.load(a.predictions,allow_pickle=False) as z: y=z['y_true'].astype(float); current=z['current_glucose'].astype(float); fir=z['y_pred'].astype(float)
    if not np.allclose(y,yd,atol=1e-4,rtol=0) or not np.allclose(current,x[:,-1,0],atol=1e-4,rtol=0): raise ValueError('alignment failed')
    ft=features(x,f,fir,current); order=np.random.default_rng(SEED).permutation(len(y)); cut=len(y)//2; cal,hold=order[:cut],order[cut:]; mu=ft[cal].mean(0); sd=ft[cal].std(0); sd[sd<1e-6]=1; xc=(ft[cal]-mu)/sd; xh=(ft[hold]-mu)/sd
    rows=[]
    print('=== HIDDEN EXCURSION PREDICTABILITY AUDIT ==='); print(f'rows: {len(y):,} | calibration: {len(cal):,} | holdout: {len(hold):,} | features: {ft.shape[1]}'); print('label: true rise >=40 AND FIR-1 predicted rise <20 mg/dL'); print('future CGM input: False | live targets used: False | FIR-1 frozen: True')
    for j,h in enumerate(HORIZONS):
        true_rise=y[:,j]-current; fir_rise=fir[:,j]-current; label=((true_rise>=40)&(fir_rise<20)).astype(float); yc,yh=label[cal],label[hold]; w,b=fit_logistic(xc,yc); pc=sigmoid(xc@w+b); ph=sigmoid(xh@w+b)
        cand=[]
        for t in np.arange(.05,.951,.01): cand.append((float(t),confusion(yc,pc,float(t))))
        eligible=[q for q in cand if q[1]['recall']>=.75]; pool=eligible if eligible else cand; t,cm=max(pool,key=lambda q:(q[1]['f1'],q[1]['precision'])); hm=confusion(yh,ph,t); auc=auc_rank(yh,ph); prev=float(yh.mean()); lift=hm['precision']/prev if prev else float('nan')
        # How much frozen FIR-1 squared error lives in this hidden cohort?
        err=(fir[hold,j]-y[hold,j])**2; m=yh.astype(bool); sse_share=float(err[m].sum()/max(err.sum(),1e-12)); concentration=sse_share/max(prev,1e-12)
        row={'horizon_minutes':h,'positive_n':int(yh.sum()),'prevalence':prev,'auc':auc,'threshold':t,'precision':hm['precision'],'recall':hm['recall'],'f1':hm['f1'],'flag_rate':hm['flag_rate'],'precision_lift':lift,'fir1_sse_share':sse_share,'error_concentration':concentration}; rows.append(row)
        print(f"+{h:3d} | hidden {100*prev:5.1f}% ({int(yh.sum()):4d}) | FIR1 SSE {100*sse_share:5.1f}% ({concentration:.2f}x) | AUC {auc:.3f} | thr {t:.2f} | precision {100*hm['precision']:5.1f}% | recall {100*hm['recall']:5.1f}% | flag {100*hm['flag_rate']:5.1f}% | lift {lift:.2f}x")
    df=pd.DataFrame(rows); qualified=((df.auc>=.80)&(df.recall>=.70)&(df.precision_lift>=2.0)).sum(); go=bool(qualified>=3 and df.auc.mean()>=.80)
    print('\nRESCUE SIGNAL GATE'); print(f'mean holdout AUC: {df.auc.mean():.3f}'); print(f'qualified horizons (AUC>=.80, recall>=70%, lift>=2x): {qualified}/4'); print(f'HIDDEN EXCURSION RESCUE: {"GO" if go else "NO-GO"}')
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); df.to_csv(out/'holdout_metrics.csv',index=False); report={'experiment':'hidden excursion predictability audit','seed':SEED,'label':'true rise >=40 AND FIR1 predicted rise <20','future_cgm_input':False,'live_targets_used':False,'fir1_frozen':True,'threshold_selection':'calibration only: max F1 among recall>=75%, else max F1','gate':'mean AUC>=.80 and >=3/4 horizons AUC>=.80 recall>=70% lift>=2x','metrics':rows,'go':go}; (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(f"\nReport: {out/'report.json'}")
if __name__=='__main__': main()
