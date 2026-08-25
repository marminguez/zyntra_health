"""Hypo V7: optimize alert policy without test leakage.

For each LOPO fold, train the same CGM+dynamics LSTM as V6. Sweep thresholds on
validation only, select the threshold with the fewest false alerts per day while
meeting recall/warning constraints, then evaluate that frozen threshold on the
held-out patient. Test threshold sweeps are saved for diagnostics only and are
never used for selection.
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
os.environ.setdefault("PYTHONHASHSEED","42"); os.environ.setdefault("TF_DETERMINISTIC_OPS","1")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from event_evaluation import evaluate_events
from lstm_hypoglycemia_classifier import build_hypoglycemia_classifier
from ohio_t1dm_loader import load_ohio_directory, dataset_summary

SEED=42; LOOKBACK=48
FEATURES=["glucose","glucose_delta_5m","glucose_delta_15m","glucose_delta_30m","glucose_acceleration_15m"]
THRESHOLDS=np.round(np.arange(.30,.951,.05),2)
TARGET_RECALL=.90; TARGET_WARNING=15.0

def seed_all():
    random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass

def sequences(frame,scaler):
    xs=[]; ys=[]; meta=[]
    for pid,p in frame.groupby("p_id"):
        p=p.sort_index().copy(); scaled=scaler.transform(p[FEATURES].fillna(0.0))
        for i in range(LOOKBACK,len(p)):
            xs.append(scaled[i-LOOKBACK:i]); ys.append(int(p.iloc[i].target_hypo)); meta.append({"p_id":int(pid),"timestamp":p.index[i],"glucose":float(p.iloc[i].glucose)})
    return np.asarray(xs,dtype=np.float32),np.asarray(ys,dtype=np.int32),pd.DataFrame(meta)

def choose_validation_patient(train_ids,held_out):
    candidates=[p for p in sorted(train_ids) if p!=held_out]
    return candidates[(SEED+int(held_out))%len(candidates)]

def sweep(meta,probs):
    rows=[]
    for t in THRESHOLDS:
        ev=evaluate_events(meta,probs,float(t),30)
        rows.append({"threshold":float(t),"hypoglycemia_events":ev["hypoglycemia_events"],"detected_events":ev["detected_events"],"event_recall":ev["event_recall"],"median_warning_minutes":ev["median_warning_minutes"],"false_alert_episodes":ev["false_alert_episodes"],"false_alerts_per_patient_day":ev["false_alerts_per_patient_day"]})
    return pd.DataFrame(rows)

def select_threshold(validation_sweep):
    eligible=validation_sweep[(validation_sweep.event_recall>=TARGET_RECALL)&(validation_sweep.median_warning_minutes.fillna(0)>=TARGET_WARNING)].copy()
    if not eligible.empty:
        # Minimize burden; break ties in favor of recall, then higher threshold.
        eligible=eligible.sort_values(["false_alerts_per_patient_day","event_recall","threshold"],ascending=[True,False,False])
        return float(eligible.iloc[0].threshold),"constrained_min_fp"
    # Fallback: maximize recall first, then minimize false alerts.
    fallback=validation_sweep.sort_values(["event_recall","false_alerts_per_patient_day","threshold"],ascending=[False,True,False])
    return float(fallback.iloc[0].threshold),"fallback_max_recall"

def train_fold(data,test_id,outdir):
    seed_all(); remaining=[int(x) for x in sorted(data.p_id.unique()) if int(x)!=int(test_id)]; val_id=choose_validation_patient(remaining,test_id); train_ids=[x for x in remaining if x!=val_id]
    train=data[data.p_id.isin(train_ids)].copy(); val=data[data.p_id==val_id].copy(); test=data[data.p_id==test_id].copy()
    scaler=StandardScaler().fit(train[FEATURES].fillna(0.0)); xtr,ytr,_=sequences(train,scaler); xv,yv,mv=sequences(val,scaler); xt,yt,mt=sequences(test,scaler)
    classes=np.unique(ytr); weights=compute_class_weight("balanced",classes=classes,y=ytr); cw={int(c):float(w) for c,w in zip(classes,weights)}
    model=build_hypoglycemia_classifier(LOOKBACK,len(FEATURES)); model.compile(optimizer="adam",loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=35,batch_size=128,shuffle=False,callbacks=[EarlyStopping(monitor="val_auc",patience=8,restore_best_weights=True,mode="max")],verbose=0)
    pv=model.predict(xv,verbose=0).ravel(); pt=model.predict(xt,verbose=0).ravel(); val_sweep=sweep(mv,pv); threshold,policy=select_threshold(val_sweep); test_sweep=sweep(mt,pt); ev=evaluate_events(mt,pt,threshold,30)
    fold_dir=outdir/f"patient_{test_id}"; fold_dir.mkdir(parents=True,exist_ok=True); val_sweep.to_csv(fold_dir/"validation_threshold_sweep.csv",index=False); test_sweep.to_csv(fold_dir/"test_threshold_sweep_diagnostic.csv",index=False)
    auc=float(roc_auc_score(yt,pt)) if len(np.unique(yt))>1 else None
    return {"test_patient":int(test_id),"validation_patient":int(val_id),"train_patients":train_ids,"selected_threshold":threshold,"selection_policy":policy,"roc_auc":auc,"hypoglycemia_events":ev["hypoglycemia_events"],"detected_events":ev["detected_events"],"missed_events":ev["missed_events"],"event_recall":ev["event_recall"],"median_warning_minutes":ev["median_warning_minutes"],"false_alert_episodes":ev["false_alert_episodes"],"false_alerts_per_patient_day":ev["false_alerts_per_patient_day"],"observed_patient_days":ev["observed_patient_days"]}

def aggregate(folds):
    df=pd.DataFrame(folds); total_events=int(df.hypoglycemia_events.sum()); total_detected=int(df.detected_events.sum())
    return {"patients":len(df),"pooled_events":total_events,"pooled_detected":total_detected,"pooled_event_recall":total_detected/total_events if total_events else None,"mean_patient_event_recall":float(df.event_recall.mean()),"median_patient_event_recall":float(df.event_recall.median()),"mean_false_alerts_per_patient_day":float(df.false_alerts_per_patient_day.mean()),"median_false_alerts_per_patient_day":float(df.false_alerts_per_patient_day.median()),"median_warning_minutes_across_patients":float(df.median_warning_minutes.dropna().median()) if df.median_warning_minutes.notna().any() else None,"patients_meeting_full_target":int(((df.event_recall>=TARGET_RECALL)&(df.false_alerts_per_patient_day<=1.0)&(df.median_warning_minutes>=TARGET_WARNING)).sum())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True,type=Path); ap.add_argument("--output-dir",default=Path("models/v7_alert_intelligence"),type=Path); args=ap.parse_args(); data=load_ohio_directory(args.data_dir); ids=sorted(int(x) for x in data.p_id.unique()); args.output_dir.mkdir(parents=True,exist_ok=True)
    dataset_summary(data).to_csv(args.output_dir/"dataset_summary.csv",index=False); folds=[]
    for pid in ids:
        print(f"V7 held-out patient {pid}",flush=True); folds.append(train_fold(data,pid,args.output_dir)); pd.DataFrame(folds).to_csv(args.output_dir/"v7_per_patient.csv",index=False)
    report={"model":"hypo_v7_alert_intelligence","seed":SEED,"features":FEATURES,"protocol":"LOPO; threshold selected on validation only","threshold_grid":THRESHOLDS.tolist(),"selection_constraints":{"event_recall_min":TARGET_RECALL,"median_warning_minutes_min":TARGET_WARNING},"folds":folds,"aggregate":aggregate(folds),"research_target":{"event_recall_min":.90,"false_alerts_per_patient_day_max":1.0,"median_warning_minutes_min":15},"clinical_status":"research only; not clinically validated"}
    (args.output_dir/"v7_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report["aggregate"],indent=2))
if __name__=="__main__": main()
