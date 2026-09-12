"""V12.2 retrospective false-positive audit.

Research evaluation only. Re-runs the frozen V12.2 temporal protocol and adds
alert-level audit metrics on the blind day-31+ test. It does not change model
training, personalization activation, alert policy, or any decision threshold.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lopo_hypo_v12_1 as base
from lopo_hypo_v12_2 import conservative_gate_decision


def alert_audit(meta, probs, policy):
    alert, notify = base.state_machine(meta, probs, **policy)
    m = meta.reset_index(drop=True).copy()
    m["timestamp"] = pd.to_datetime(m.timestamp)
    m["notify"] = np.asarray(notify) >= .5
    horizon = pd.Timedelta(minutes=30)
    total_alerts = true_alerts = 0
    for _, f in m.groupby("p_id"):
        f = f.sort_values("timestamp").reset_index(drop=True)
        low = f.glucose < 70
        event_starts = f.loc[low & ~low.shift(1, fill_value=False), "timestamp"].tolist()
        for ts in f.loc[f.notify, "timestamp"]:
            total_alerts += 1
            if any(ts < e <= ts + horizon for e in event_starts):
                true_alerts += 1
    false_alerts = total_alerts - true_alerts
    days = 0.0
    for _, f in m.groupby("p_id"):
        days += max((f.timestamp.max()-f.timestamp.min()).total_seconds()/86400., 5./1440.)
    return {
        "notification_alerts": total_alerts,
        "true_positive_alerts": true_alerts,
        "false_positive_alerts": false_alerts,
        "alert_precision_ppv": true_alerts/total_alerts if total_alerts else None,
        "false_alerts_per_patient_day": false_alerts/days if days else None,
    }


def run_patient_audited(data, pid, outdir):
    # Reproduce V12.2, then deterministically reconstruct its blind-test predictions
    # so the audit cannot influence policy selection.
    row = base.run_patient(data, pid, outdir)
    base.seed_all()
    others=[int(x) for x in sorted(data.p_id.unique()) if int(x)!=pid]
    vid=base.val_patient(others,pid); train_ids=[x for x in others if x!=vid]
    train=data[data.p_id.isin(train_ids)]; val=data[data.p_id==vid]; target=data[data.p_id==pid].sort_index()
    scaler=base.StandardScaler().fit(train[base.FEATURES].fillna(0.))
    xtr,ytr,_=base.sequences(train,scaler); xv,yv,_=base.sequences(val,scaler); xt,yt,mt=base.sequences(target,scaler)
    classes=np.unique(ytr); ws=base.compute_class_weight("balanced",classes=classes,y=ytr); cw={int(c):float(w) for c,w in zip(classes,ws)}
    pop=base.build_hypoglycemia_classifier(base.LOOKBACK,len(base.FEATURES)); pop.compile(optimizer="adam",loss="binary_crossentropy",metrics=[base.tf.keras.metrics.AUC(name="auc")])
    pop.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=35,batch_size=128,shuffle=False,callbacks=[base.EarlyStopping(monitor="val_auc",patience=8,restore_best_weights=True,mode="max")],verbose=0)
    ptarget=pop.predict(xt,verbose=0).ravel(); start=pd.to_datetime(mt.timestamp).min(); ft_end=start+pd.Timedelta(days=base.FT_END_DAY); gate_end=start+pd.Timedelta(days=base.GATE_END_DAY)
    _,_,ft_idx=base.subset(mt,ptarget,end=ft_end); mtest,ptest,test_idx=base.subset(mt,ptarget,start=gate_end)
    if row["activation"]=="personalized":
        cand=base.fine_tune(pop,xt[ft_idx],yt[ft_idx]); ptest=cand.predict(xt[test_idx],verbose=0).ravel()
    audit=alert_audit(mtest,ptest,row["active_policy"])
    audit["false_negative_events"] = int(row["test_events"]-row["test_active_detected"])
    row.update({f"audit_{k}":v for k,v in audit.items()})
    return row


def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True,type=Path); ap.add_argument("--output-dir",default=Path("ml/models/v12_2_fp_audit"),type=Path); a=ap.parse_args()
    base.gate_decision=conservative_gate_decision
    data=base.add_context(base.load_ohio_directory(a.data_dir)); a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for pid in sorted(int(x) for x in data.p_id.unique()):
        print(f"V12.2 FP audit held-out patient {pid}",flush=True)
        rows.append(run_patient_audited(data,pid,a.output_dir))
        pd.DataFrame(rows).to_csv(a.output_dir/"v12_2_fp_audit_per_patient.csv",index=False)
    d=pd.DataFrame(rows)
    tp=int(d.audit_true_positive_alerts.sum()); fp=int(d.audit_false_positive_alerts.sum()); total=tp+fp
    events=int(d.test_events.sum()); detected=int(d.test_active_detected.sum())
    report={"model":"v12_2_false_positive_audit","baseline":"V12.2 frozen","test":"day 31+ blind future test","patients":len(d),"events":events,"detected_events":detected,"event_recall":detected/events if events else None,"notification_alerts":total,"true_positive_alerts":tp,"false_positive_alerts":fp,"alert_precision_ppv":tp/total if total else None,"false_positive_fraction":fp/total if total else None,"mean_false_alerts_per_patient_day":float(d.audit_false_alerts_per_patient_day.mean()),"clinical_status":"retrospective research evaluation only; not clinically validated; not for live alerting"}
    (a.output_dir/"v12_2_fp_audit_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
