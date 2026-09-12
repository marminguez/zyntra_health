"""Canonical retrospective alert audit for frozen V12.2.

One alert = one notification emitted by the state machine. A notification is
useful when at least one glucose<70 episode starts strictly after it and within
30 minutes. Multiple hypo events may be covered by one notification. Event
recall is computed with the same notification-level definition so PPV/recall
are interpretable together.

Research evaluation only; no model/policy/gate optimization is performed.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import lopo_hypo_v12_1 as base
from lopo_hypo_v12_2 import conservative_gate_decision
H=30

def starts(mask,timestamps):
 m=pd.Series(mask).astype(bool); ts=pd.Series(pd.to_datetime(timestamps)); return list(ts[m & ~m.shift(1,fill_value=False)])

def canonical(meta,probs,policy,pid):
 _,notify=base.state_machine(meta,probs,**policy); f=meta.reset_index(drop=True).copy(); f['timestamp']=pd.to_datetime(f.timestamp); f['notify']=np.asarray(notify)>=.5
 ev=starts(f.glucose<70,f.timestamp); alerts=list(f.loc[f.notify,'timestamp']); h=pd.Timedelta(minutes=H)
 ar=[]; covered=set()
 for ai,a in enumerate(alerts):
  matches=[(ei,e) for ei,e in enumerate(ev) if a<e<=a+h]
  for ei,_ in matches: covered.add(ei)
  ar.append({'p_id':pid,'alert_id':ai,'alert_time':a,'classification':'TP' if matches else 'FP','matched_event_count':len(matches),'matched_event_ids':';'.join(str(ei) for ei,_ in matches),'first_event_time':min((e for _,e in matches),default=None),'warning_minutes':min(((e-a).total_seconds()/60 for _,e in matches),default=None)})
 er=[]
 for ei,e in enumerate(ev):
  prior=[a for a in alerts if e-h<=a<e]; best=max(prior) if prior else None
  er.append({'p_id':pid,'event_id':ei,'event_start':e,'detected':bool(prior),'matched_alert_time':best,'warning_minutes':((e-best).total_seconds()/60 if best is not None else None)})
 ad=pd.DataFrame(ar); ed=pd.DataFrame(er); tp=int((ad.classification=='TP').sum()) if len(ad) else 0; fp=len(ad)-tp; det=int(ed.detected.sum()) if len(ed) else 0
 days=max((f.timestamp.max()-f.timestamp.min()).total_seconds()/86400.,5/1440.)
 return ad,ed,{'events':len(ev),'detected_events':det,'missed_events':len(ev)-det,'event_recall':det/len(ev) if ev else None,'alerts':len(alerts),'tp_alerts':tp,'fp_alerts':fp,'alert_ppv':tp/len(alerts) if alerts else None,'false_alerts_per_day':fp/days,'observed_days':days}

def reconstruct(data,pid,outdir):
 row=base.run_patient(data,pid,outdir); base.seed_all(); others=[int(x) for x in sorted(data.p_id.unique()) if int(x)!=pid]; vid=base.val_patient(others,pid); trids=[x for x in others if x!=vid]; train=data[data.p_id.isin(trids)]; val=data[data.p_id==vid]; target=data[data.p_id==pid].sort_index(); sc=base.StandardScaler().fit(train[base.FEATURES].fillna(0.)); xtr,ytr,_=base.sequences(train,sc); xv,yv,_=base.sequences(val,sc); xt,yt,mt=base.sequences(target,sc); classes=np.unique(ytr); ws=base.compute_class_weight('balanced',classes=classes,y=ytr); cw={int(c):float(w) for c,w in zip(classes,ws)}; pop=base.build_hypoglycemia_classifier(base.LOOKBACK,len(base.FEATURES)); pop.compile(optimizer='adam',loss='binary_crossentropy',metrics=[base.tf.keras.metrics.AUC(name='auc')]); pop.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=35,batch_size=128,shuffle=False,callbacks=[base.EarlyStopping(monitor='val_auc',patience=8,restore_best_weights=True,mode='max')],verbose=0); pt=pop.predict(xt,verbose=0).ravel(); start=pd.to_datetime(mt.timestamp).min(); _,_,fi=base.subset(mt,pt,end=start+pd.Timedelta(days=base.FT_END_DAY)); mtest,ptest,ti=base.subset(mt,pt,start=start+pd.Timedelta(days=base.GATE_END_DAY));
 if row['activation']=='personalized': ptest=base.fine_tune(pop,xt[fi],yt[fi]).predict(xt[ti],verbose=0).ravel()
 ad,ed,s=canonical(mtest,ptest,row['active_policy'],pid); p=outdir/f'patient_{pid}'; p.mkdir(parents=True,exist_ok=True); ad.to_csv(p/'canonical_alerts.csv',index=False); ed.to_csv(p/'canonical_events.csv',index=False); return {'test_patient':pid,'activation':row['activation'],**s}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True,type=Path); ap.add_argument('--output-dir',default=Path('ml/models/v12_2_canonical_alert_audit'),type=Path); a=ap.parse_args(); base.gate_decision=conservative_gate_decision; data=base.add_context(base.load_ohio_directory(a.data_dir)); a.output_dir.mkdir(parents=True,exist_ok=True); rows=[]
 for pid in sorted(int(x) for x in data.p_id.unique()): print(f'Canonical V12.2 audit patient {pid}',flush=True); rows.append(reconstruct(data,pid,a.output_dir)); pd.DataFrame(rows).to_csv(a.output_dir/'v12_2_canonical_per_patient.csv',index=False)
 d=pd.DataFrame(rows); events=int(d.events.sum()); det=int(d.detected_events.sum()); alerts=int(d.alerts.sum()); tp=int(d.tp_alerts.sum()); fp=int(d.fp_alerts.sum()); days=float(d.observed_days.sum()); r={'model':'v12_2_canonical_alert_audit','baseline':'V12.2 frozen','canonical_alert':'state-machine notification','horizon_minutes':H,'events':events,'detected_events':det,'missed_events':events-det,'event_recall':det/events if events else None,'alerts':alerts,'tp_alerts':tp,'fp_alerts':fp,'alert_ppv':tp/alerts if alerts else None,'false_alert_fraction':fp/alerts if alerts else None,'false_alerts_per_patient_day':fp/days if days else None,'clinical_status':'retrospective research evaluation only; not clinically validated; not for live alerting'}; (a.output_dir/'v12_2_canonical_report.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
