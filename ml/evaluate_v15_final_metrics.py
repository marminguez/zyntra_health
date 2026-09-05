"""Competition-aligned internal validation for frozen V15.1-V15.4 predictions.
No fitting, tuning, test.parquet, or live targets are used.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
HORIZONS=(30,60,90,120)

def dts_error_grid(pred_glucose,true_glucose):
 pred=np.asarray(pred_glucose,dtype=float);ref=np.asarray(true_glucose,dtype=float);total=len(ref)
 b_up=np.where(ref<=50,60,(540/450)*(ref-50)+60);c_up=np.where(ref<=50,86.5,(513.5/297)*(ref-50)+86.5);d_up=np.where(ref<=50,124,(476/191)*(ref-50)+124);e_up=np.where(ref<=50,179,(421/117)*(ref-50)+179)
 b_low=np.where(ref<=62.5,0,(430/537.5)*(ref-62.5)+50);c_low=np.where(ref<=97.5,0,(257/502.5)*(ref-97.5)+50);d_low=np.where(ref<=153,0,(147/447)*(ref-153)+50);e_low=np.where(ref<=238,0,(76/362)*(ref-238)+50)
 za=(pred<=b_up)&(pred>=b_low);zb=((pred<=c_up)&(pred>b_up))|((pred<b_low)&(pred>=c_low));zc=((pred<=d_up)&(pred>c_up))|((pred<c_low)&(pred>=d_low));zd=((pred<=e_up)&(pred>d_up))|((pred<d_low)&(pred>=e_low));ze=(pred>e_up)|(pred<e_low)
 return {f'dts_{z}_percent':float(np.sum(m)/total*100) for z,m in [('a',za),('b',zb),('c',zc),('d',zd),('e',ze)]}
def metrics(yt,yp):
 e=yp-yt;out={'n':int(len(yt)),'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e**2))),'mard':float(np.mean(np.abs(e)/np.maximum(np.abs(yt),1e-6))*100)};out.update(dts_error_grid(yp,yt));return out
def load_predictions(path):
 with np.load(path,allow_pickle=False) as z:return z['y_true'].astype(float),z['y_pred'].astype(float)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--v15-1-dir',default='ml/results/v15_1');ap.add_argument('--v15-2-dir',default='ml/results/v15_2');ap.add_argument('--v15-3-dir',default='ml/results/v15_3');ap.add_argument('--v15-4-dir',default='ml/results/v15_4');ap.add_argument('--outdir',default='ml/results/v15_final_metrics');a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 dirs={'v15_1':Path(a.v15_1_dir),'v15_2':Path(a.v15_2_dir),'v15_3':Path(a.v15_3_dir),'v15_4':Path(a.v15_4_dir)};candidates={};reference_y=None
 for name,d in dirs.items():
  y,p=load_predictions(d/'validation_predictions.npz')
  if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=4:raise ValueError(f'{name}: unexpected y/p shapes {y.shape}/{p.shape}')
  if reference_y is None:reference_y=y
  elif y.shape!=reference_y.shape or not np.array_equal(y,reference_y,equal_nan=True):raise ValueError(f'{name}: validation targets do not exactly match V15.1')
  candidates[name]=p
 rows=[]
 for name,p in candidates.items():
  for i,h in enumerate(HORIZONS):rows.append({'model':name,'horizon_minutes':h,**metrics(reference_y[:,i],p[:,i])})
 df=pd.DataFrame(rows);df.to_csv(out/'competition_aligned_metrics.csv',index=False)
 summary=[]
 for name,g in df.groupby('model',sort=False):summary.append({'model':name,'mean_mard_4h':float(g.mard.mean()),'mean_dts_a_4h':float(g.dts_a_percent.mean()),'mean_rmse_4h':float(g.rmse.mean()),'mean_mae_4h':float(g.mae.mean())})
 sm=pd.DataFrame(summary);sm.to_csv(out/'competition_aligned_summary.csv',index=False)
 report={'version':'v15_competition_aligned_internal_validation','models':list(candidates),'horizons':list(HORIZONS),'validation_windows':int(len(reference_y)),'target_alignment':'exact np.array_equal across V15.1-V15.4','metrics':['MARD','DTS Error Grid A-E','RMSE','MAE'],'averaging':'unweighted mean across 30/60/90/120 horizons','test_parquet_used':False,'live_targets_used':False,'optimization_performed':False};(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('\nV15 COMPETITION-ALIGNED METRICS — PER HORIZON');print(df.to_string(index=False));print('\nV15 4-HORIZON SUMMARY');print(sm.to_string(index=False));print(f'\nArtifacts written to {out}')
if __name__=='__main__':main()
