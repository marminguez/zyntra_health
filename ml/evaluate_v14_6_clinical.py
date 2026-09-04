"""Matched clinical comparison: frozen V14.1 vs V14.5 vs V14.6.
No fitting or optimization is performed.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

HORIZONS=(30,60,90,120)
SLICES=("all","hypoglycemia_<70","target_70_180","hyperglycemia_>180","severe_hyper_>250","rapid_drop_>=30mgdl","rapid_rise_>=30mgdl")

def load_validation(root):
    xs,ys=[],[]
    for p in sorted((root/'validation').glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            xs.append(z['x'].astype(np.float32)); ys.append(z['y'].astype(np.float32))
    if not xs: raise ValueError('No validation shards found')
    return np.concatenate(xs),np.concatenate(ys)

def dclass(d): return np.where(d>5,1,np.where(d<-5,-1,0))
def calc(yt,yp,cur,mask):
    yt,yp,cur=yt[mask],yp[mask],cur[mask]; e=yp-yt
    return {'n':int(len(yt)),'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e**2))),'mard':float(np.mean(np.abs(e)/np.maximum(np.abs(yt),1e-6))*100),'direction_accuracy':float(np.mean(dclass(yt-cur)==dclass(yp-cur)))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='ml/data/v14_1'); ap.add_argument('--v14-1-dir',default='ml/results/v14_1'); ap.add_argument('--v14-5-dir',default='ml/results/v14_5'); ap.add_argument('--v14-6-dir',default='ml/results/v14_6'); ap.add_argument('--outdir',default='ml/results/v14_6_clinical'); ap.add_argument('--batch-size',type=int,default=64); a=ap.parse_args()
    data,d1,d5,d6,out=map(Path,[a.data_dir,a.v14_1_dir,a.v14_5_dir,a.v14_6_dir,a.outdir]); out.mkdir(parents=True,exist_ok=True)
    x,y=load_validation(data); cur=x[:,-1,0].astype(np.float64)
    with np.load(d1/'normalization.npz') as z: mean,std=z['mean'],z['std']
    xn=((x-mean)/std).astype(np.float32)
    m1=tf.keras.models.load_model(d1/'best_model.keras'); raw1=m1.predict(xn,batch_size=a.batch_size,verbose=1); p1=np.column_stack([v.reshape(-1) for v in raw1]).astype(np.float64)
    preds={}
    for name,d in [('v14_5',d5),('v14_6',d6)]:
        with np.load(d/'validation_predictions.npz') as z:
            preds[name]=(z['y_pred'].astype(np.float64),z['y_true'].astype(np.float64),z['current_glucose'].astype(np.float64))
        p,yy,cc=preds[name]
        if p.shape!=p1.shape or yy.shape!=y.shape or not np.allclose(yy,y,atol=1e-5) or not np.allclose(cc,cur,atol=1e-5): raise ValueError(f'{name} predictions do not align with validation windows')
    rows=[]
    for i,h in enumerate(HORIZONS):
        yt=y[:,i].astype(np.float64); delta=yt-cur
        masks={'all':np.ones(len(yt),dtype=bool),'hypoglycemia_<70':yt<70,'target_70_180':(yt>=70)&(yt<=180),'hyperglycemia_>180':yt>180,'severe_hyper_>250':yt>250,'rapid_drop_>=30mgdl':delta<=-30,'rapid_rise_>=30mgdl':delta>=30}
        model_preds=[('v14_1',p1[:,i]),('v14_5',preds['v14_5'][0][:,i]),('v14_6',preds['v14_6'][0][:,i])]
        for model,pred in model_preds:
            for s in SLICES: rows.append({'model':model,'horizon_minutes':h,'slice':s,**calc(yt,pred,cur,masks[s])})
    df=pd.DataFrame(rows); df.to_csv(out/'v14_6_vs_v14_5_vs_v14_1_clinical_metrics.csv',index=False)
    comp=[]
    for h in HORIZONS:
        for s in SLICES:
            r1=df[(df.model=='v14_1')&(df.horizon_minutes==h)&(df.slice==s)].iloc[0]; r5=df[(df.model=='v14_5')&(df.horizon_minutes==h)&(df.slice==s)].iloc[0]; r6=df[(df.model=='v14_6')&(df.horizon_minutes==h)&(df.slice==s)].iloc[0]
            comp.append({'horizon_minutes':h,'slice':s,'n':int(r1.n),'v14_1_rmse':r1.rmse,'v14_5_rmse':r5.rmse,'v14_6_rmse':r6.rmse,'rmse_change_pct_v14_6_vs_v14_1':(r6.rmse-r1.rmse)/r1.rmse*100,'rmse_change_pct_v14_6_vs_v14_5':(r6.rmse-r5.rmse)/r5.rmse*100,'v14_1_mard':r1.mard,'v14_5_mard':r5.mard,'v14_6_mard':r6.mard,'mard_change_pct_v14_6_vs_v14_1':(r6.mard-r1.mard)/r1.mard*100,'mard_change_pct_v14_6_vs_v14_5':(r6.mard-r5.mard)/r5.mard*100,'v14_1_direction':r1.direction_accuracy,'v14_5_direction':r5.direction_accuracy,'v14_6_direction':r6.direction_accuracy,'direction_change_pp_v14_6_vs_v14_1':(r6.direction_accuracy-r1.direction_accuracy)*100,'direction_change_pp_v14_6_vs_v14_5':(r6.direction_accuracy-r5.direction_accuracy)*100})
    comp=pd.DataFrame(comp); comp.to_csv(out/'v14_6_clinical_comparison.csv',index=False)
    (out/'report.json').write_text(json.dumps({'version':'v14.6_matched_clinical_comparison','validation_windows':int(len(x)),'models':['v14_1','v14_5','v14_6'],'matched':True,'test_parquet_used':False,'optimization_performed':False},indent=2),encoding='utf-8')
    print('\nV14.6 vs V14.5 vs V14.1 — CLINICAL COMPARISON'); print(comp.to_string(index=False)); print(f'\nArtifacts written to {out}')
if __name__=='__main__': main()
