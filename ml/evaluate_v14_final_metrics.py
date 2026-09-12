"""Competition-aligned internal validation for frozen V14 candidates.

Evaluates V14.1, V14.5 and V14.6 on the same untouched internal validation
windows using per-horizon MARD, RMSE, MAE and the DTS Error Grid percentages.
No fitting, model selection logic, threshold tuning, or test.parquet access.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

HORIZONS=(30,60,90,120)

def load_validation(root):
    xs,ys=[],[]
    for p in sorted((root/'validation').glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            xs.append(z['x'].astype(np.float32)); ys.append(z['y'].astype(np.float32))
    if not xs: raise ValueError('No validation shards found')
    return np.concatenate(xs),np.concatenate(ys)

def dts_error_grid(pred_glucose,true_glucose):
    pred=np.asarray(pred_glucose,dtype=float); ref=np.asarray(true_glucose,dtype=float); total=len(ref)
    b_up=np.where(ref<=50,60,(540/450)*(ref-50)+60)
    c_up=np.where(ref<=50,86.5,(513.5/297)*(ref-50)+86.5)
    d_up=np.where(ref<=50,124,(476/191)*(ref-50)+124)
    e_up=np.where(ref<=50,179,(421/117)*(ref-50)+179)
    b_low=np.where(ref<=62.5,0,(430/537.5)*(ref-62.5)+50)
    c_low=np.where(ref<=97.5,0,(257/502.5)*(ref-97.5)+50)
    d_low=np.where(ref<=153,0,(147/447)*(ref-153)+50)
    e_low=np.where(ref<=238,0,(76/362)*(ref-238)+50)
    za=(pred<=b_up)&(pred>=b_low)
    zb=((pred<=c_up)&(pred>b_up))|((pred<b_low)&(pred>=c_low))
    zc=((pred<=d_up)&(pred>c_up))|((pred<c_low)&(pred>=d_low))
    zd=((pred<=e_up)&(pred>d_up))|((pred<d_low)&(pred>=e_low))
    ze=(pred>e_up)|(pred<e_low)
    return {f'dts_{z}_percent':float(np.sum(m)/total*100) for z,m in [('a',za),('b',zb),('c',zc),('d',zd),('e',ze)]}

def metrics(yt,yp):
    e=yp-yt
    out={'n':int(len(yt)),'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e**2))),'mard':float(np.mean(np.abs(e)/np.maximum(np.abs(yt),1e-6))*100)}
    out.update(dts_error_grid(yp,yt)); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='ml/data/v14_1'); ap.add_argument('--v14-1-dir',default='ml/results/v14_1'); ap.add_argument('--v14-5-dir',default='ml/results/v14_5'); ap.add_argument('--v14-6-dir',default='ml/results/v14_6'); ap.add_argument('--outdir',default='ml/results/v14_final_metrics'); ap.add_argument('--batch-size',type=int,default=64); a=ap.parse_args()
    data,d1,d5,d6,out=map(Path,[a.data_dir,a.v14_1_dir,a.v14_5_dir,a.v14_6_dir,a.outdir]); out.mkdir(parents=True,exist_ok=True)
    x,y=load_validation(data)
    with np.load(d1/'normalization.npz') as z: mean,std=z['mean'],z['std']
    xn=((x-mean)/std).astype(np.float32)
    m1=tf.keras.models.load_model(d1/'best_model.keras'); raw=m1.predict(xn,batch_size=a.batch_size,verbose=1); p1=np.column_stack([v.reshape(-1) for v in raw]).astype(float)
    candidates={'v14_1':p1}
    for name,d in [('v14_5',d5),('v14_6',d6)]:
        with np.load(d/'validation_predictions.npz') as z:
            yy=z['y_true'].astype(float); pp=z['y_pred'].astype(float)
        if yy.shape!=y.shape or pp.shape!=y.shape or not np.allclose(yy,y,atol=1e-5): raise ValueError(f'{name} predictions do not align with validation windows')
        candidates[name]=pp
    rows=[]
    for name,p in candidates.items():
        for i,h in enumerate(HORIZONS): rows.append({'model':name,'horizon_minutes':h,**metrics(y[:,i].astype(float),p[:,i])})
    df=pd.DataFrame(rows); df.to_csv(out/'competition_aligned_metrics.csv',index=False)
    summary=[]
    for name,g in df.groupby('model',sort=False):
        summary.append({'model':name,'mean_mard_4h':float(g.mard.mean()),'mean_dts_a_4h':float(g.dts_a_percent.mean()),'mean_rmse_4h':float(g.rmse.mean()),'mean_mae_4h':float(g.mae.mean())})
    sm=pd.DataFrame(summary); sm.to_csv(out/'competition_aligned_summary.csv',index=False)
    (out/'report.json').write_text(json.dumps({'version':'v14_competition_aligned_internal_validation','models':list(candidates),'horizons':list(HORIZONS),'validation_windows':int(len(y)),'metrics':['MARD','DTS Error Grid A-E','RMSE','MAE'],'averaging':'unweighted mean across 30/60/90/120 horizons','test_parquet_used':False,'optimization_performed':False},indent=2),encoding='utf-8')
    print('\nCOMPETITION-ALIGNED METRICS — PER HORIZON'); print(df.to_string(index=False))
    print('\n4-HORIZON SUMMARY'); print(sm.to_string(index=False))
    print(f'\nArtifacts written to {out}')
if __name__=='__main__': main()
