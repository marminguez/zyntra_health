"""Episode-level retrospective audit for frozen V12.2.

Research evaluation only; not clinically validated and not for live alerting.
This audit does not optimize or alter V12.2. It reconstructs the blind day-31+
alert stream and separates notification starts into: first TP alert for an
upcoming hypo event, redundant/duplicate TP alert for an already-covered event,
and FP alert with no hypo onset in the next 30 minutes.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import audit_hypo_v12_2_fp as audit1
import lopo_hypo_v12_1 as base
from lopo_hypo_v12_2 import conservative_gate_decision

HORIZON_MIN=30

def classify_notifications(meta, probs, policy):
    alert, notify=base.state_machine(meta,probs,**policy)
    m=meta.reset_index(drop=True).copy(); m["timestamp"]=pd.to_datetime(m.timestamp); m["notify"]=np.asarray(notify)>=.5
    horizon=pd.Timedelta(minutes=HORIZON_MIN)
    rows=[]; days=0.0
    for pid,f in m.groupby("p_id"):
        f=f.sort_values("timestamp").reset_index(drop=True)
        days+=max((f.timestamp.max()-f.timestamp.min()).total_seconds()/86400.,5./1440.)
        low=f.glucose<70
        starts=list(f.loc[low & ~low.shift(1,fill_value=False),"timestamp"])
        covered=set()
        for ts in f.loc[f.notify,"timestamp"]:
            upcoming=[(i,e) for i,e in enumerate(starts) if ts<e<=ts+horizon]
            if not upcoming:
                cls="false_positive"
                event_time=None
            else:
                i,event_time=min(upcoming,key=lambda x:x[1])
                if i in covered: cls="duplicate_true_positive"
                else: cls="true_positive"; covered.add(i)
            rows.append({"p_id":int(pid),"notification_time":ts,"class":cls,"matched_event_time":event_time,"lead_minutes":((event_time-ts).total_seconds()/60. if event_time is not None else None)})
    r=pd.DataFrame(rows)
    counts=r["class"].value_counts() if len(r) else pd.Series(dtype=int)
    tp=int(counts.get("true_positive",0)); dup=int(counts.get("duplicate_true_positive",0)); fp=int(counts.get("false_positive",0)); total=tp+dup+fp
    return r,{"notification_alerts":total,"unique_true_positive_alerts":tp,"duplicate_true_positive_alerts":dup,"false_positive_alerts":fp,"unique_alert_precision_ppv":tp/total if total else None,"useful_or_redundant_tp_fraction":(tp+dup)/total if total else None,"false_positive_fraction":fp/total if total else None,"false_alerts_per_patient_day":fp/days if days else None,"duplicate_alerts_per_patient_day":dup/days if days else None}

def reconstruct(data,pid,outdir):
    row=base.run_patient(data,pid,outdir)
    base.seed_all(); others=[int(x) for x in sorted(data.p_id.unique()) if int(x)!=pid]; vid=base.val_patient(others,pid); train_ids=[x for x in others if x!=vid]
    train=data[data.p_id.isin(train_ids)]; val=data[data.p_id==vid]; target=data[data.p_id==pid].sort_index(); scaler=base.StandardScaler().fit(train[base.FEATURES].fillna(0.))
    xtr,ytr,_=base.sequences(train,scaler); xv,yv,_=base.sequences(val,scaler); xt,yt,mt=base.sequences(target,scaler)
    classes=np.unique(ytr); ws=base.compute_class_weight("balanced",classes=classes,y=ytr); cw={int(c):float(w) for c,w in zip(classes,ws)}
    pop=base.build_hypoglycemia_classifier(base.LOOKBACK,len(base.FEATURES)); pop.compile(optimizer="adam",loss="binary_crossentropy",metrics=[base.tf.keras.metrics.AUC(name="auc")]); pop.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=35,batch_size=128,shuffle=False,callbacks=[base.EarlyStopping(monitor="val_auc",patience=8,restore_best_weights=True,mode="max")],verbose=0)
    ptarget=pop.predict(xt,verbose=0).ravel(); start=pd.to_datetime(mt.timestamp).min(); ft_end=start+pd.Timedelta(days=base.FT_END_DAY); gate_end=start+pd.Timedelta(days=base.GATE_END_DAY); _,_,ft_idx=base.subset(mt,ptarget,end=ft_end); mtest,ptest,test_idx=base.subset(mt,ptarget,start=gate_end)
    if row["activation"]=="personalized":
        cand=base.fine_tune(pop,xt[ft_idx],yt[ft_idx]); ptest=cand.predict(xt[test_idx],verbose=0).ravel()
    timeline,stats=classify_notifications(mtest,ptest,row["active_policy"]); timeline.to_csv(outdir/f"patient_{pid}"/"alert_timeline_audit.csv",index=False)
    stats["hypoglycemia_events"]=int(row["test_events"]); stats["detected_events"]=int(row["test_active_detected"]); stats["missed_events"]=int(row["test_events"]-row["test_active_detected"]); stats["event_recall"]=float(row["test_active_recall"]); stats["warning_minutes"]=row["test_active_warning_minutes"]
    return {"test_patient":pid,"activation":row["activation"],**stats}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True,type=Path); ap.add_argument("--output-dir",default=Path("ml/models/v12_2_alert_audit_v2"),type=Path); a=ap.parse_args(); base.gate_decision=conservative_gate_decision
    data=base.add_context(base.load_ohio_directory(a.data_dir)); a.output_dir.mkdir(parents=True,exist_ok=True); rows=[]
    for pid in sorted(int(x) for x in data.p_id.unique()):
        print(f"V12.2 alert audit v2 held-out patient {pid}",flush=True); rows.append(reconstruct(data,pid,a.output_dir)); pd.DataFrame(rows).to_csv(a.output_dir/"v12_2_alert_audit_v2_per_patient.csv",index=False)
    d=pd.DataFrame(rows); total=int(d.notification_alerts.sum()); tp=int(d.unique_true_positive_alerts.sum()); dup=int(d.duplicate_true_positive_alerts.sum()); fp=int(d.false_positive_alerts.sum()); events=int(d.hypoglycemia_events.sum()); det=int(d.detected_events.sum())
    report={"model":"v12_2_alert_audit_v2","baseline":"V12.2 frozen","definition":f"notification episode matched to hypo onset strictly after alert and within {HORIZON_MIN} minutes; repeat notifications for an already-covered onset are duplicates","patients":len(d),"events":events,"detected_events":det,"event_recall":det/events if events else None,"notification_alerts":total,"unique_true_positive_alerts":tp,"duplicate_true_positive_alerts":dup,"false_positive_alerts":fp,"unique_alert_precision_ppv":tp/total if total else None,"false_positive_fraction":fp/total if total else None,"mean_false_alerts_per_patient_day":float(d.false_alerts_per_patient_day.mean()),"mean_duplicate_alerts_per_patient_day":float(d.duplicate_alerts_per_patient_day.mean()),"clinical_status":"retrospective research evaluation only; not clinically validated; not for live alerting"}; (a.output_dir/"v12_2_alert_audit_v2_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
