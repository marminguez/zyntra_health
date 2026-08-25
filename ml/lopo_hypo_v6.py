"""Hypo V6: deterministic Leave-One-Patient-Out evaluation on OhioT1DM.

Raw OhioT1DM files are intentionally NOT committed. Point --data-dir at a local
folder containing the authorised XML files.
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
from alert_quality import build_alert_analysis
from event_evaluation import evaluate_events
from lstm_hypoglycemia_classifier import build_hypoglycemia_classifier
from ohio_t1dm_loader import load_ohio_directory, dataset_summary
SEED=42; LOOKBACK=48
FEATURES=["glucose","glucose_delta_5m","glucose_delta_15m","glucose_delta_30m","glucose_acceleration_15m"]
def seed_all():
    random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass
def sequences(frame, scaler):
    xs=[]; ys=[]; meta=[]
    for pid,p in frame.groupby("p_id"):
        p=p.sort_index().copy(); scaled=scaler.transform(p[FEATURES].fillna(0.0))
        for i in range(LOOKBACK,len(p)):
            xs.append(scaled[i-LOOKBACK:i]); ys.append(int(p.iloc[i].target_hypo)); meta.append({"p_id":int(pid),"timestamp":p.index[i],"glucose":float(p.iloc[i].glucose)})
    return np.asarray(xs,dtype=np.float32),np.asarray(ys,dtype=np.int32),pd.DataFrame(meta)
def choose_validation_patient(train_ids,held_out):
    candidates=[p for p in sorted(train_ids) if p!=held_out]; return candidates[(SEED+int(held_out))%len(candidates)]
def choose_threshold(model,xv,yv):
    probs=model.predict(xv,verbose=0).ravel(); best=(0.5,-1.)
    for t in np.arange(.30,.951,.05):
        pred=probs>=t; tp=np.sum((pred==1)&(yv==1)); fp=np.sum((pred==1)&(yv==0)); fn=np.sum((pred==0)&(yv==1)); beta2=25.
        score=(1+beta2)*tp/((1+beta2)*tp+beta2*fn+fp) if tp else 0.
        if score>best[1]: best=(float(t),float(score))
    return best[0]
def train_fold(data,test_id):
    seed_all(); remaining=[int(x) for x in sorted(data.p_id.unique()) if int(x)!=int(test_id)]; val_id=choose_validation_patient(remaining,test_id); train_ids=[x for x in remaining if x!=val_id]
    train=data[data.p_id.isin(train_ids)].copy(); val=data[data.p_id==val_id].copy(); test=data[data.p_id==test_id].copy()
    scaler=StandardScaler().fit(train[FEATURES].fillna(0.0)); xtr,ytr,_=sequences(train,scaler); xv,yv,_=sequences(val,scaler); xt,yt,meta=sequences(test,scaler)
    classes=np.unique(ytr); weights=compute_class_weight("balanced",classes=classes,y=ytr); cw={int(c):float(w) for c,w in zip(classes,weights)}
    model=build_hypoglycemia_classifier(LOOKBACK,len(FEATURES)); model.compile(optimizer="adam",loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=35,batch_size=128,shuffle=False,callbacks=[EarlyStopping(monitor="val_auc",patience=8,restore_best_weights=True,mode="max")],verbose=0)
    threshold=choose_threshold(model,xv,yv); probs=model.predict(xt,verbose=0).ravel(); auc=float(roc_auc_score(yt,probs)) if len(np.unique(yt))>1 else None; ev=evaluate_events(meta,probs,threshold,30); analysis=build_alert_analysis(meta,probs,threshold,30); fps=analysis[analysis.classification=="FP"]
    return {"test_patient":int(test_id),"validation_patient":int(val_id),"train_patients":train_ids,"threshold":threshold,"roc_auc":auc,"hypoglycemia_events":ev["hypoglycemia_events"],"detected_events":ev["detected_events"],"missed_events":ev["missed_events"],"event_recall":ev["event_recall"],"median_warning_minutes":ev["median_warning_minutes"],"false_alert_episodes":ev["false_alert_episodes"],"false_alerts_per_patient_day":ev["false_alerts_per_patient_day"],"observed_patient_days":ev["observed_patient_days"],"near_miss_alerts":int((fps.fp_subtype=="near_miss").sum()) if len(fps) else 0,"clear_false_alerts":int((fps.fp_subtype=="clear_false_positive").sum()) if len(fps) else 0}
def aggregate(folds):
    df=pd.DataFrame(folds); total_events=int(df.hypoglycemia_events.sum()); total_detected=int(df.detected_events.sum())
    return {"patients":len(df),"pooled_events":total_events,"pooled_detected":total_detected,"pooled_event_recall":total_detected/total_events if total_events else None,"mean_patient_event_recall":float(df.event_recall.mean()),"median_patient_event_recall":float(df.event_recall.median()),"std_patient_event_recall":float(df.event_recall.std(ddof=0)),"mean_false_alerts_per_patient_day":float(df.false_alerts_per_patient_day.mean()),"median_false_alerts_per_patient_day":float(df.false_alerts_per_patient_day.median()),"median_warning_minutes_across_patients":float(df.median_warning_minutes.dropna().median()) if df.median_warning_minutes.notna().any() else None}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True,type=Path); ap.add_argument("--output-dir",default=Path("models/v6_lopo"),type=Path); args=ap.parse_args(); data=load_ohio_directory(args.data_dir); ids=sorted(int(x) for x in data.p_id.unique())
    if len(ids)<3: raise RuntimeError("LOPO requires at least 3 patients")
    args.output_dir.mkdir(parents=True,exist_ok=True); dataset_summary(data).to_csv(args.output_dir/"dataset_summary.csv",index=False); folds=[]
    for pid in ids:
        print(f"LOPO held-out patient {pid}",flush=True); folds.append(train_fold(data,pid)); pd.DataFrame(folds).to_csv(args.output_dir/"lopo_per_patient.csv",index=False)
    report={"model":"hypo_v6_lopo_cgm_dynamics","seed":SEED,"features":FEATURES,"lookback_minutes":LOOKBACK*5,"prediction_horizon_minutes":30,"protocol":"leave-one-patient-out with one additional patient reserved for validation inside each fold","folds":folds,"aggregate":aggregate(folds),"research_target":{"event_recall_min":.80,"false_alerts_per_patient_day_max":1.,"median_warning_minutes_min":15},"clinical_status":"research only; not clinically validated"}
    (args.output_dir/"lopo_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report["aggregate"],indent=2))
if __name__=="__main__": main()
