"""Contrastive atlas for FIR-1 hidden excursions.

For each horizon, compare hidden excursions (true rise>=40 while FIR-1 rise<20)
with nearest non-hidden controls matched ONLY on current glucose and FIR-1 predicted
rise. Then inspect pre-anchor temporal sequence differences.

Diagnostic only: future target defines cases; matching/features use anchor-available data.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
H=(30,60,90,120)
def load(d):
 xs,fs,ys=[],[],[]
 for p in sorted((Path(d)/'validation').glob('*.npz')):
  with np.load(p,allow_pickle=False) as z: xs.append(z['x'].astype(float));fs.append(z['future_known'].astype(float));ys.append(z['y'].astype(float))
 return np.concatenate(xs),np.concatenate(fs),np.concatenate(ys)
def valid_event(x,v,miss): return np.where(x[:,:,miss]<.5,x[:,:,v],0)
def stats(a): return float(np.mean(a)),float(np.median(a))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='ml/data/v15_4_scale50');ap.add_argument('--predictions',default='ml/results/fir1_scale50/validation_predictions.npz');ap.add_argument('--outdir',default='ml/results/hidden_excursion_contrastive');a=ap.parse_args()
 x,f,yd=load(a.data_dir)
 with np.load(a.predictions,allow_pickle=False) as z:y=z['y_true'].astype(float);cur=z['current_glucose'].astype(float);fir=z['y_pred'].astype(float)
 if not np.allclose(y,yd,atol=1e-4,rtol=0) or not np.allclose(cur,x[:,-1,0],atol=1e-4,rtol=0):raise ValueError('alignment failed')
 bol=valid_event(x,7,8);ins=valid_event(x,9,10);car=valid_event(x,11,12); glucose=x[:,:,0]
 rows=[]; matchrows=[]
 print('=== HIDDEN EXCURSION CONTRASTIVE ATLAS ===');print('cases: true rise>=40 AND FIR-1 predicted rise<20');print('controls: nearest non-hidden match on current glucose + FIR-1 predicted rise only');print('future target used for cohort only: True | future CGM feature: False')
 for j,h in enumerate(H):
  rise=y[:,j]-cur;fr=fir[:,j]-cur;case=(rise>=40)&(fr<20);ctrl=~case; ci=np.where(case)[0]; pool=np.where(ctrl)[0]
  # standardize the two matching dimensions globally; deterministic nearest control, reuse allowed
  M=np.c_[cur,fr];sd=M.std(0);sd[sd<1e-6]=1;Z=(M-M.mean(0))/sd; chosen=[]
  for k in ci:
   d=np.sum((Z[pool]-Z[k])**2,axis=1);chosen.append(pool[int(np.argmin(d))])
  chosen=np.asarray(chosen); matchrows.append({'horizon_minutes':h,'cases':len(ci),'unique_controls':int(len(np.unique(chosen))),'mean_abs_current_diff':float(np.mean(np.abs(cur[ci]-cur[chosen]))),'mean_abs_fir_rise_diff':float(np.mean(np.abs(fr[ci]-fr[chosen])))})
  def add(name,cv,tv):
   delta=cv-tv;pooled=np.sqrt((np.var(cv)+np.var(tv))/2);effect=float(np.mean(delta)/pooled) if pooled>1e-9 else 0
   rows.append({'horizon_minutes':h,'feature':name,'case_mean':float(np.mean(cv)),'control_mean':float(np.mean(tv)),'mean_delta':float(np.mean(delta)),'median_delta':float(np.median(delta)),'standardized_effect':effect})
  # Full-history shape: glucose changes over multiple lookbacks, volatility, min/max/range.
  for steps,mins in ((3,15),(6,30),(12,60),(24,120),(48,240),(96,480),(288,1440)):
   s=min(steps,x.shape[1]);add(f'glucose_change_{mins}m',glucose[ci,-1]-glucose[ci,-s],glucose[chosen,-1]-glucose[chosen,-s]);add(f'glucose_std_{mins}m',glucose[ci,-s:].std(1),glucose[chosen,-s:].std(1));add(f'glucose_range_{mins}m',np.ptp(glucose[ci,-s:],axis=1),np.ptp(glucose[chosen,-s:],axis=1))
  # Exact event timing/sums, preserving temporal information that 60m aggregates can hide.
  for arr,nm in ((car,'carbs'),(bol,'bolus'),(ins,'insulin')):
   for steps,mins in ((3,15),(6,30),(12,60),(24,120),(48,240)):
    s=min(steps,x.shape[1]);add(f'{nm}_sum_{mins}m',arr[ci,-s:].sum(1),arr[chosen,-s:].sum(1));add(f'{nm}_events_{mins}m',(arr[ci,-s:]!=0).sum(1),(arr[chosen,-s:]!=0).sum(1))
   # steps since last nonzero event, capped at 24h
   def since(q):
    out=np.full(len(q),x.shape[1],dtype=float)
    for ii,row in enumerate(q):
     nz=np.where(row!=0)[0]
     if len(nz):out[ii]=len(row)-1-nz[-1]
    return out*5
   add(f'minutes_since_{nm}',since(arr[ci]),since(arr[chosen]))
  # Current derivative channels and missingness burden.
  for idx,nm in ((1,'delta5'),(2,'delta15'),(3,'delta30'),(4,'acceleration')):add(nm,x[ci,-1,idx],x[chosen,-1,idx])
  for idx,nm in ((6,'basal_missing'),(8,'bolus_missing'),(10,'insulin_missing'),(12,'carbs_missing')):
   add(f'{nm}_60m',x[ci,-12:,idx].mean(1),x[chosen,-12:,idx].mean(1))
 df=pd.DataFrame(rows);md=pd.DataFrame(matchrows);df['abs_effect']=df.standardized_effect.abs();top=df.sort_values('abs_effect',ascending=False).groupby('horizon_minutes').head(15)
 print('\nMATCH QUALITY');print(md.to_string(index=False));print('\nTOP CONTRASTIVE FEATURES PER HORIZON');print(top[['horizon_minutes','feature','case_mean','control_mean','mean_delta','standardized_effect']].to_string(index=False))
 # Cross-horizon consistency: mean signed effect and number of horizons with |d|>=.20 and same sign.
 piv=df.pivot(index='feature',columns='horizon_minutes',values='standardized_effect');cons=[]
 for name,r in piv.iterrows():
  vals=r.dropna().values; mean=float(np.mean(vals)); strong=int(np.sum(np.abs(vals)>=.20)); same=int(np.sum(np.sign(vals)==np.sign(mean))) if mean else 0;cons.append({'feature':name,'mean_effect':mean,'mean_abs_effect':float(np.mean(np.abs(vals))),'strong_horizons':strong,'same_sign_horizons':same})
 cdf=pd.DataFrame(cons).sort_values(['strong_horizons','mean_abs_effect'],ascending=False)
 print('\nCROSS-HORIZON CONSISTENCY');print(cdf.head(25).to_string(index=False))
 # Discovery gate: at least one anchor-available temporal feature has |d|>=.20 in >=3 horizons with consistent sign.
 eligible=cdf[(cdf.strong_horizons>=3)&(cdf.same_sign_horizons>=3)];go=len(eligible)>0
 print('\nREPRESENTATION GATE');print(f'consistent temporal signals (|effect|>=0.20 in >=3 horizons): {len(eligible)}');print(f'HIDDEN EXCURSION REPRESENTATION: {"GO" if go else "NO-GO"}')
 out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);df.to_csv(out/'contrasts.csv',index=False);md.to_csv(out/'match_quality.csv',index=False);cdf.to_csv(out/'consistency.csv',index=False);(out/'report.json').write_text(json.dumps({'experiment':'hidden excursion contrastive atlas','matching':'nearest non-hidden on current glucose + FIR1 predicted rise','future_target_cohort_only':True,'representation_gate':'at least one feature |standardized effect|>=.20 in >=3 horizons, same sign','eligible_features':eligible.to_dict('records'),'go':bool(go)},indent=2),encoding='utf-8');print(f"\nReport: {out/'report.json'}")
if __name__=='__main__':main()
