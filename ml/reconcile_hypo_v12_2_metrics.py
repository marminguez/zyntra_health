"""Reconcile V12.2 event-level and notification-level metrics.

Retrospective research audit only; does not alter the frozen V12.2 model,
policy, gate, or test split. Uses the exact event_evaluation definition and
separately explains which notification episode is associated with each event.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import lopo_hypo_v12_1 as base
from lopo_hypo_v12_2 import conservative_gate_decision

H=30

def episodes(mask,timestamps):
    out=[]; start=end=None; gap=pd.Timedelta(minutes=5)
    for active,ts in zip(mask,timestamps):
        ts=pd.Timestamp(ts)
        if active:
            if start is None: start=end=ts
            elif ts-end<=gap: end=ts
            else: out.append((start,end)); start=end=ts
        elif start is not None: out.append((start,end)); start=end=None
    if start is not None: out.append((start,end))
    return out

def reconcile(meta,probs,policy,pid):
    alert,notify=base.state_machine(meta,probs,**policy)
    f=meta.reset_index(drop=True).copy(); f["timestamp"]=pd.to_datetime(f.timestamp); f["alert"]=np.asarray(alert)>=.5; f["notify"]=np.asarray(notify)>=.5
    real=episodes(f.glucose<70,f.timestamp); ae=episodes(f.alert,f.timestamp); notifications=list(f.loc[f.notify,"timestamp"])
    matched_alert_eps=set(); event_rows=[]
    for ei,(es,ee) in enumerate(real):
        win=f[f.alert & (f.timestamp<es) & (f.timestamp>=es-pd.Timedelta(minutes=H))]
        if win.empty:
            event_rows.append({"p_id":pid,"event_id":ei,"event_start":es,"event_end":ee,"detected":False,"first_alert_time":None,"warning_minutes":None,"alert_episode_id":None,"notification_start":None,"notification_to_event_minutes":None})
            continue
        first=win.timestamp.min(); aid=next((i for i,(a,b) in enumerate(ae) if a<=first<=b),None)
        if aid is not None: matched_alert_eps.add(aid)
        nstart=ae[aid][0] if aid is not None else None
        event_rows.append({"p_id":pid,"event_id":ei,"event_start":es,"event_end":ee,"detected":True,"first_alert_time":first,"warning_minutes":float((es-first).total_seconds()/60.),"alert_episode_id":aid,"notification_start":nstart,"notification_to_event_minutes":float((es-nstart).total_seconds()/60.) if nstart is not None else None})
    alert_rows=[]
    for aid,(a,b) in enumerate(ae):
        ev=[r for r in event_rows if r["alert_episode_id"]==aid]
        alert_rows.append({"p_id":pid,"alert_episode_id":aid,"alert_start":a,"alert_end":b,"duration_minutes":float((b-a).total_seconds()/60.+5),"matched_event_count":len(ev),"classification":"matched_alert_episode" if ev else "false_alert_episode","matched_event_ids":";".join(str(r["event_id"]) for r in ev),"first_matched_event_start":min((r["event_start"] for r in ev),default=None),"start_to_first_event_minutes":min(((r["event_start"]-a).total_seconds()/60. for r in ev),default=None)})
    er=pd.DataFrame(event_rows); ar=pd.DataFrame(alert_rows)
    return er,ar,{"events":len(real),"detected_events":int(er.detected.sum()) if len(er) else 0,"alert_episodes":len(ae),"matched_alert_episodes":len(matched_alert_eps),"false_alert_episodes":len(ae)-len(matched_alert_eps),"notifications":len(notifications),"detected_events_per_matched_alert_episode":(int(er.detected.sum())/len(matched_alert_eps) if matched_alert_eps else None)}

def reconstruct(data,pid,outdir):
    row=base.run_patient(data,pid,outdir); base.seed_all(); others=[int(x) for x in sorted(data.p_id.unique()) if int(x)!=pid]; vid=base.val_patient(others,pid); train_ids=[x for x in others if x!=vid]
    train=data[data.p_id.isin(train_ids)]; val=data[data.p_id==vid]; target=data[data.p_id==pid].sort_index(); scaler=base.StandardScaler().fit(train[base.FEATURES].fillna(0.)); xtr,ytr,_=base.sequences(train,scaler); xv,yv,_=base.sequences(val,scaler); xt,yt,mt=base.sequences(target,scaler)
    classes=np.unique(ytr); ws=base.compute_class_weight("balanced",classes=classes,y=ytr); cw={int(c):float(w) for c,w in zip(classes,ws)}; pop=base.build_hypoglycemia_classifier(base.LOOKBACK,len(base.FEATURES)); pop.compile(optimizer="adam",loss="binary_crossentropy",metrics=[base.tf.keras.metrics.AUC(name="auc")]); pop.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=35,batch_size=128,shuffle=False,callbacks=[base.EarlyStopping(monitor="val_auc",patience=8,restore_best_weights=True,mode="max")],verbose=0)
    pt=pop.predict(xt,verbose=0).ravel(); start=pd.to_datetime(mt.timestamp).min(); ft_end=start+pd.Timedelta(days=base.FT_END_DAY); gate_end=start+pd.Timedelta(days=base.GATE_END_DAY); _,_,fi=base.subset(mt,pt,end=ft_end); mtest,ptest,ti=base.subset(mt,pt,start=gate_end)
    if row["activation"]=="personalized": ptest=base.fine_tune(pop,xt[fi],yt[fi]).predict(xt[ti],verbose=0).ravel()
    er,ar,s=reconcile(mtest,ptest,row["active_policy"],pid); pdir=outdir/f"patient_{pid}"; pdir.mkdir(parents=True,exist_ok=True); er.to_csv(pdir/"event_reconciliation.csv",index=False); ar.to_csv(pdir/"alert_episode_reconciliation.csv",index=False)
    return {"test_patient":pid,"activation":row["activation"],"event_recall_reference":row["test_active_recall"],**s}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True,type=Path); ap.add_argument("--output-dir",default=Path("ml/models/v12_2_metric_reconciliation"),type=Path); a=ap.parse_args(); base.gate_decision=conservative_gate_decision; data=base.add_context(base.load_ohio_directory(a.data_dir)); a.output_dir.mkdir(parents=True,exist_ok=True); rows=[]
    for pid in sorted(int(x) for x in data.p_id.unique()):
        print(f"V12.2 metric reconciliation held-out patient {pid}",flush=True); rows.append(reconstruct(data,pid,a.output_dir)); pd.DataFrame(rows).to_csv(a.output_dir/"v12_2_metric_reconciliation_per_patient.csv",index=False)
    d=pd.DataFrame(rows); report={"model":"v12_2_metric_reconciliation","baseline":"V12.2 frozen","event_definition":"glucose <70 episode; detected when ALERTED at any sample strictly before onset and within previous 30 min","alert_episode_definition":"continuous ALERTED state grouped at 5-min sampling; episode is FP if it matches no hypo event","patients":len(d),"events":int(d.events.sum()),"detected_events":int(d.detected_events.sum()),"event_recall":float(d.detected_events.sum()/d.events.sum()),"alert_episodes":int(d.alert_episodes.sum()),"matched_alert_episodes":int(d.matched_alert_episodes.sum()),"false_alert_episodes":int(d.false_alert_episodes.sum()),"notifications":int(d.notifications.sum()),"clinical_status":"retrospective research evaluation only; not clinically validated; not for live alerting"}; (a.output_dir/"v12_2_metric_reconciliation_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
