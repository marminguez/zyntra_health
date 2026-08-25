"""Leakage-safe, event-aware training/evaluation pipeline for Zyntra Hypo V5."""
from pathlib import Path
import json
import os
import pickle
import random

# Set deterministic flags before TensorFlow initializes.
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

from alert_quality import build_alert_analysis, threshold_sweep
from event_evaluation import evaluate_events
from lstm_hypoglycemia_classifier import build_hypoglycemia_classifier, create_sequences_for_classification, find_optimal_threshold, process_full_data

SEED=42
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
try: tf.config.experimental.enable_op_determinism()
except Exception: pass

ML_DIR=Path(__file__).resolve().parent; MODELS_DIR=ML_DIR/"models"
MODEL_PATH=MODELS_DIR/"lstm_hypoglycemia_classifier_v2.h5"; THRESHOLD_PATH=MODELS_DIR/"optimal_threshold_v2.npy"; SCALER_PATH=MODELS_DIR/"lstm_feature_scaler_v2.pkl"; FEATURE_NAMES_PATH=MODELS_DIR/"lstm_feature_names_v2.json"; EVALUATION_PATH=MODELS_DIR/"evaluation_v2.json"; ALERT_ANALYSIS_PATH=MODELS_DIR/"alert_analysis.csv"; THRESHOLD_SWEEP_PATH=MODELS_DIR/"threshold_sweep.csv"
LOOKBACK=48; BETA=5.0; EXCLUDED_COLUMNS=["glucose_future","target_hypo","p_id"]


def _json_default(v):
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,np.floating): return float(v)
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,(pd.Timestamp,np.datetime64)): return str(v)
    raise TypeError(type(v).__name__)


def add_glucose_dynamics(df):
    """Add causal CGM dynamics; only past/current glucose is used."""
    out=df.copy()
    grouped=out.groupby("p_id",sort=False)["glucose"]
    out["glucose_delta_5m"]=grouped.diff(1)
    out["glucose_delta_15m"]=grouped.diff(3)
    out["glucose_delta_30m"]=grouped.diff(6)
    previous_15=grouped.shift(3)-grouped.shift(6)
    out["glucose_acceleration_15m"]=out["glucose_delta_15m"]-previous_15
    dynamics=["glucose_delta_5m","glucose_delta_15m","glucose_delta_30m","glucose_acceleration_15m"]
    out[dynamics]=out.groupby("p_id",sort=False)[dynamics].transform(lambda x:x.fillna(0.0))
    return out


def chronological_split(df,train_fraction=.70,val_fraction=.15):
    n=len(df); a=int(n*train_fraction); b=int(n*(train_fraction+val_fraction)); return df.iloc[:a].copy(),df.iloc[a:b].copy(),df.iloc[b:].copy()


def split_dataset(df):
    ids=sorted(df.p_id.unique())
    if len(ids)>=3:
        rng=np.random.default_rng(SEED); arr=np.asarray(ids); rng.shuffle(arr); n=len(arr); nt=max(1,int(n*.70)); nv=max(1,int(n*.15))
        if nt+nv>=n: nt,nv=n-2,1
        tr,va,te=arr[:nt].tolist(),arr[nt:nt+nv].tolist(),arr[nt+nv:].tolist()
        return df[df.p_id.isin(tr)].copy(),df[df.p_id.isin(va)].copy(),df[df.p_id.isin(te)].copy(),{"strategy":"patient_disjoint","train_ids":tr,"val_ids":va,"test_ids":te}
    if len(ids)==2:
        tv=df[df.p_id==ids[0]].copy(); test=df[df.p_id==ids[1]].copy(); train,val,_=chronological_split(tv,.80,.20); return train,val,test,{"strategy":"hybrid_two_patient","train_ids":[ids[0]],"val_ids":[ids[0]],"test_ids":[ids[1]]}
    train,val,test=chronological_split(df); return train,val,test,{"strategy":"chronological_single_patient","train_ids":ids,"val_ids":ids,"test_ids":ids,"warning":"Single-patient evaluation does not demonstrate generalization to unseen patients."}


def fit_and_apply_scaler(train,val,test,features):
    scaler=StandardScaler().fit(train[features]); result=[]
    for frame in (train,val,test):
        x=frame.copy(); x[features]=scaler.transform(frame[features]); result.append(x)
    return (*result,scaler)


def frames_to_sequences(frame,include_metadata=False):
    xs,ys,metas=[],[],[]
    for pid in sorted(frame.p_id.unique()):
        p=frame[frame.p_id==pid].sort_index(); x,y=create_sequences_for_classification(p,lookback=LOOKBACK)
        if not len(x): continue
        xs.append(x); ys.append(y)
        if include_metadata: metas.append(pd.DataFrame({"p_id":p.iloc[LOOKBACK:].p_id.values,"timestamp":p.iloc[LOOKBACK:].index}))
    if not xs: raise RuntimeError("Not enough data to create sequences")
    X,Y=np.concatenate(xs),np.concatenate(ys); return (X,Y,pd.concat(metas,ignore_index=True)) if include_metadata else (X,Y)


def attach_unscaled_metadata(meta,test):
    source=test.reset_index().rename(columns={test.index.name or "index":"timestamp"}); source.timestamp=pd.to_datetime(source.timestamp); meta=meta.copy(); meta.timestamp=pd.to_datetime(meta.timestamp)
    merged=meta.merge(source[["p_id","timestamp","glucose"]],on=["p_id","timestamp"],how="left")
    if merged.glucose.isna().any(): raise RuntimeError("Could not align original glucose metadata")
    return merged


def calculate_metrics(model,x,y,threshold):
    p=model.predict(x,verbose=0).flatten(); pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); auc=roc_auc_score(y,p) if len(np.unique(y))>1 else None
    return {"roc_auc":auc,"recall_sensitivity":tp/(tp+fn) if tp+fn else 0.,"precision_ppv":tp/(tp+fp) if tp+fp else 0.,"specificity":tn/(tn+fp) if tn+fp else 0.,"npv":tn/(tn+fn) if tn+fn else 0.,"f1":f1_score(y,pred,zero_division=0),"confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},"test_samples":int(len(y)),"test_positive_samples":int(y.sum())},p


def main():
    print("\n=== ZYNTRA HYPO V5: DYNAMICS + REPRODUCIBILITY ===")
    df=add_glucose_dynamics(process_full_data()); features=[c for c in df.columns if c not in EXCLUDED_COLUMNS]
    train_raw,val_raw,test_raw,split_info=split_dataset(df); train,val,test,scaler=fit_and_apply_scaler(train_raw,val_raw,test_raw,features)
    xtr,ytr=frames_to_sequences(train); xv,yv=frames_to_sequences(val); xt,yt,meta=frames_to_sequences(test,True); meta=attach_unscaled_metadata(meta,test_raw)
    classes=np.unique(ytr); weights=compute_class_weight("balanced",classes=classes,y=ytr); cw={int(c):float(w) for c,w in zip(classes,weights)}
    model=build_hypoglycemia_classifier(LOOKBACK,len(features)); model.compile(optimizer="adam",loss="binary_crossentropy",metrics=[tf.keras.metrics.Precision(name="precision"),tf.keras.metrics.Recall(name="recall"),tf.keras.metrics.AUC(name="auc")])
    model.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=50,batch_size=64,shuffle=False,callbacks=[EarlyStopping(monitor="val_auc",patience=15,restore_best_weights=True,mode="max")],verbose=1)
    threshold=float(find_optimal_threshold(model,xv,yv,beta=BETA)); metrics,prob=calculate_metrics(model,xt,yt,threshold); events=evaluate_events(meta,prob,threshold,30); alerts=build_alert_analysis(meta,prob,threshold,30); sweep=threshold_sweep(meta,prob,horizon_minutes=30)
    MODELS_DIR.mkdir(parents=True,exist_ok=True); model.save(MODEL_PATH); np.save(THRESHOLD_PATH,threshold)
    with open(SCALER_PATH,"wb") as f: pickle.dump(scaler,f)
    with open(FEATURE_NAMES_PATH,"w") as f: json.dump(features,f,indent=2)
    alerts.to_csv(ALERT_ANALYSIS_PATH,index=False); sweep.to_csv(THRESHOLD_SWEEP_PATH,index=False)
    report={"model":"lstm_hypoglycemia_classifier_v5_dynamics","seed":SEED,"deterministic_training":True,"prediction_horizon_minutes":30,"lookback_minutes":LOOKBACK*5,"threshold":threshold,"threshold_metric":f"F{BETA}","features":features,"new_v5_features":["glucose_delta_5m","glucose_delta_15m","glucose_delta_30m","glucose_acceleration_15m"],"split":split_info,"patients_total":int(df.p_id.nunique()),"sample_metrics":metrics,"event_metrics":events,"alert_quality":{"tp_events":int((alerts.classification=="TP").sum()),"fp_alerts":int((alerts.classification=="FP").sum()),"fn_events":int((alerts.classification=="FN").sum())},"product_research_target":{"event_recall_min":.80,"false_alerts_per_patient_day_max":1.0,"median_warning_minutes_min":15},"limitations":["Current demo contains too few patients for robust external generalization.","IOB remains a simple rolling bolus sum.","Product research targets are experimental, not clinically validated thresholds."]}
    with open(EVALUATION_PATH,"w") as f: json.dump(report,f,indent=2,default=_json_default)
    print(json.dumps(report,indent=2,default=_json_default))

if __name__=="__main__": main()
