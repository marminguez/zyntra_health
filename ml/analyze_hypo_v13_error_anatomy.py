"""V13 Error Anatomy: diagnose predictor vs alert-policy bottleneck.
Research evaluation only. No model or policy optimization is applied to blind test.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import lopo_hypo_v12_1 as base
from lopo_hypo_v12_2 import conservative_gate_decision
from audit_hypo_v12_2_canonical_alerts import starts,H

def anatomy(meta,probs,policy,pid):
 _,notify=base.state_machine(meta,probs,**policy)
 f=meta.reset_index(drop=True).copy(); f['timestamp']=pd.to_datetime(f.timestamp); f['prob']=np.asarray(probs); f['notify']=np.asarray(notify)>=.5
 events=starts(f.glucose<70,f.timestamp); horizon=pd.Timedelta(minutes=H); er=[]
 for ei,e in enumerate(events):
  w=f[(f.timestamp>=e-horizon)&(f.timestamp<e)]
  alerts=w[w.notify]
  maxp=float(w.prob.max()) if len(w) else np.nan; maxi=w.prob.idxmax() if len(w) else None
  er.append({'p_id':pid,'event_id':ei,'event_start':e,'canonical_detected':not alerts.empty,'max_raw_prob_30m':maxp,'max_prob_time':f.loc[maxi,'timestamp'] if maxi is not None else None,'glucose_at_max_prob':float(f.loc[maxi,'glucose']) if maxi is not None else None,'notification_count_30m':len(alerts)})
 ed=pd.DataFrame(er)
 # Raw-score threshold sweep, deliberately diagnostic only: shows attainable tradeoff without state machine.
 rows=[]
 days=max((f.timestamp.max()-f.timestamp.min()).total_seconds()/86400.,5/1440.)
 for t in np.linspace(.05,.95,19):
  raw=f.prob>=t; onset=raw & ~raw.shift(1,fill_value=False); at=list(f.loc[onset,'timestamp']); covered=sum(any(e-horizon<=a<e for a in at) for e in events); useful=sum(any(a<e<=a+horizon for e in events) for a in at); fp=len(at)-useful
  rows.append({'p_id':pid,'threshold':float(t),'events':len(events),'detected_events':covered,'event_recall':covered/len(events) if events else np.nan,'raw_alerts':len(at),'tp_alerts':useful,'fp_alerts':fp,'alert_ppv':useful/len(at) if at else np.nan,'false_alerts_per_day':fp/days})
 return ed,pd.DataFrame(rows)

def reconstruct(data,pid,outdir):
 row=base.run_patient(data,pid,outdir); base.seed_all(); others=[int(x) for x in sorted(data.p_id.unique()) if int(x)!=pid]; vid=base.val_patient(others,pid); trids=[x for x in others if x!=vid]; train=data[data.p_id.isin(trids)]; val=data[data.p_id==vid]; target=data[data.p_id==pid].sort_index(); sc=base.StandardScaler().fit(train[base.FEATURES].fillna(0.)); xtr,ytr,_=base.sequences(train,sc); xv,yv,_=base.sequences(val,sc); xt,yt,mt=base.sequences(target,sc); classes=np.unique(ytr); ws=base.compute_class_weight('balanced',classes=classes,y=ytr); cw={int(c):float(w) for c,w in zip(classes,ws)}; pop=base.build_hypoglycemia_classifier(base.LOOKBACK,len(base.FEATURES)); pop.compile(optimizer='adam',loss='binary_crossentropy',metrics=[base.tf.keras.metrics.AUC(name='auc')]); pop.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=35,batch_size=128,shuffle=False,callbacks=[base.EarlyStopping(monitor='val_auc',patience=8,restore_best_weights=True,mode='max')],verbose=0); pt=pop.predict(xt,verbose=0).ravel(); start=pd.to_datetime(mt.timestamp).min(); _,_,fi=base.subset(mt,pt,end=start+pd.Timedelta(days=base.FT_END_DAY)); mtest,ptest,ti=base.subset(mt,pt,start=start+pd.Timedelta(days=base.GATE_END_DAY));
 if row['activation']=='personalized': ptest=base.fine_tune(pop,xt[fi],yt[fi]).predict(xt[ti],verbose=0).ravel()
 ed,sw=anatomy(mtest,ptest,row['active_policy'],pid); p=outdir/f'patient_{pid}'; p.mkdir(parents=True,exist_ok=True); ed.to_csv(p/'event_raw_score_anatomy.csv',index=False); sw.to_csv(p/'raw_threshold_sweep.csv',index=False); return ed,sw

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True,type=Path); ap.add_argument('--output-dir',default=Path('ml/models/v13_error_anatomy'),type=Path); a=ap.parse_args(); base.gate_decision=conservative_gate_decision; data=base.add_context(base.load_ohio_directory(a.data_dir)); a.output_dir.mkdir(parents=True,exist_ok=True); E=[];S=[]
 for pid in sorted(int(x) for x in data.p_id.unique()): print(f'V13 error anatomy patient {pid}',flush=True); e,s=reconstruct(data,pid,a.output_dir); E.append(e);S.append(s)
 e=pd.concat(E,ignore_index=True); s=pd.concat(S,ignore_index=True); e.to_csv(a.output_dir/'v13_event_anatomy.csv',index=False); s.to_csv(a.output_dir/'v13_raw_threshold_sweep_per_patient.csv',index=False)
 agg=s.groupby('threshold').agg(events=('events','sum'),detected_events=('detected_events','sum'),raw_alerts=('raw_alerts','sum'),tp_alerts=('tp_alerts','sum'),fp_alerts=('fp_alerts','sum')).reset_index(); agg['event_recall']=agg.detected_events/agg.events; agg['alert_ppv']=agg.tp_alerts/agg.raw_alerts; agg.to_csv(a.output_dir/'v13_raw_threshold_sweep_pooled.csv',index=False)
 missed=e[~e.canonical_detected]; r={'model':'v13_error_anatomy','purpose':'diagnose raw predictor vs state-machine bottleneck','events':len(e),'canonical_detected':int(e.canonical_detected.sum()),'canonical_missed':int((~e.canonical_detected).sum()),'median_max_raw_prob_detected':float(e.loc[e.canonical_detected,'max_raw_prob_30m'].median()),'median_max_raw_prob_missed':float(missed.max_raw_prob_30m.median()),'missed_with_raw_prob_ge_0_5':int((missed.max_raw_prob_30m>=.5).sum()),'missed_with_raw_prob_ge_0_3':int((missed.max_raw_prob_30m>=.3).sum()),'clinical_status':'retrospective research diagnostic only; threshold sweep is descriptive, not blind-test optimization'}; (a.output_dir/'v13_error_anatomy_report.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2)); print('\nRaw-score pooled sweep:\n',agg.to_string(index=False))
if __name__=='__main__': main()
