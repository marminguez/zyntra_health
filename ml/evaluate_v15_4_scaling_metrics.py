"""Competition-aligned internal comparison for V15.4, Scale-10 and Scale-50.
Uses the exact DTS Error Grid implementation already used by evaluate_v15_final_metrics.py.
No fitting, tuning, test.parquet or live targets are used.
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

def direction_class(d):return np.where(d>5,1,np.where(d<-5,-1,0))

def metrics(yt,yp,current):
 e=yp-yt
 out={'n':int(len(yt)),'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e**2))),'mard':float(np.mean(np.abs(e)/np.maximum(np.abs(yt),1e-6))*100),'direction_accuracy':float(np.mean(direction_class(yt-current)==direction_class(yp-current)))}
 out.update(dts_error_grid(yp,yt));return out

def load_predictions(path):
 with np.load(path,allow_pickle=False) as z:
  y=z['y_true'].astype(float);p=z['y_pred'].astype(float);current=z['current_glucose'].astype(float)
 return y,p,current

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--v15-4-dir',default='ml/results/v15_4');ap.add_argument('--scale10-dir',default='ml/results/v15_4_scale10');ap.add_argument('--scale50-dir',default='ml/results/v15_4_scale50');ap.add_argument('--outdir',default='ml/results/v15_4_scaling_metrics');a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 dirs={'v15_4':Path(a.v15_4_dir),'scale10':Path(a.scale10_dir),'scale50':Path(a.scale50_dir)};preds={};reference_y=None;reference_current=None
 for name,d in dirs.items():
  path=d/'validation_predictions.npz'
  if not path.exists():raise FileNotFoundError(path)
  y,p,current=load_predictions(path)
  if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=4:raise ValueError(f'{name}: unexpected y/p shapes {y.shape}/{p.shape}')
  if current.ndim!=1 or len(current)!=len(y):raise ValueError(f'{name}: unexpected current_glucose shape {current.shape}')
  if reference_y is None:reference_y=y;reference_current=current
  else:
   if y.shape!=reference_y.shape or not np.array_equal(y,reference_y,equal_nan=True):raise ValueError(f'{name}: validation targets do not exactly match V15.4')
   if current.shape!=reference_current.shape or not np.array_equal(current,reference_current,equal_nan=True):raise ValueError(f'{name}: current_glucose does not exactly match V15.4')
  preds[name]=p
 rows=[]
 for name,p in preds.items():
  for i,h in enumerate(HORIZONS):rows.append({'model':name,'horizon_minutes':h,**metrics(reference_y[:,i],p[:,i],reference_current)})
 df=pd.DataFrame(rows);df.to_csv(out/'competition_aligned_scaling_metrics.csv',index=False)
 summary=[]
 for name,g in df.groupby('model',sort=False):summary.append({'model':name,'mean_mard_4h':float(g.mard.mean()),'mean_dts_a_4h':float(g.dts_a_percent.mean()),'mean_rmse_4h':float(g.rmse.mean()),'mean_mae_4h':float(g.mae.mean()),'mean_direction_accuracy_4h':float(g.direction_accuracy.mean())})
 sm=pd.DataFrame(summary);sm.to_csv(out/'competition_aligned_scaling_summary.csv',index=False)
 report={'version':'v15_4_scaling_competition_aligned_internal_validation','models':list(preds),'horizons':list(HORIZONS),'validation_windows':int(len(reference_y)),'target_alignment':'exact np.array_equal across V15.4/Scale-10/Scale-50','current_glucose_alignment':'exact np.array_equal across V15.4/Scale-10/Scale-50','metrics':['MARD','DTS Error Grid A-E','RMSE','MAE','direction_accuracy'],'averaging':'unweighted mean across 30/60/90/120 horizons','dts_implementation_source':'same equations as ml/evaluate_v15_final_metrics.py','test_parquet_used':False,'live_targets_used':False,'optimization_performed':False};(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('\nV15.4 SCALING — COMPETITION-ALIGNED METRICS PER HORIZON');print(df.to_string(index=False));print('\nV15.4 SCALING — 4-HORIZON SUMMARY');print(sm.to_string(index=False));print(f'\nArtifacts written to {out}')
if __name__=='__main__':main()
