"""Hypo V9: post-model alert intelligence for false-positive suppression.

The contextual V8 LSTM remains the risk model. V9 adds a validation-selected
alert policy that requires persistence of risk, with a rescue path for fast
falls / already-low glucose. Policy selection never uses the held-out patient.
"""
from __future__ import annotations
import argparse,json,os,random
from pathlib import Path
os.environ.setdefault("PYTHONHASHSEED","42")
os.environ.setdefault("TF_DETERMINISTIC_OPS","1")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from event_evaluation import evaluate_events
from lstm_hypoglycemia_classifier import build_hypoglycemia_classifier
from ohio_t1dm_loader import load_ohio_directory,dataset_summary

SEED=42; LOOKBACK=48
BASE=["glucose","glucose_delta_5m","glucose_delta_15m","glucose_delta_30m","glucose_acceleration_15m"]
CONTEXT=["iob_simple","carbs_recent_30m","carbs_recent_60m","basal_rate","exercise_intensity","hour_sin","hour_cos"]
FEATURES=BASE+CONTEXT
THRESHOLDS=np.arange(.30,.951,.05)
PERSISTENCE=(1,2,3,4)
RESCUE_GLUCOSE=(80.,90.,100.)
RESCUE_DELTA15=(-10.,-15.,-20.)

def seed_all():
    random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass

def add_context(df):
    out=[]
    for pid,p in df.groupby("p_id"):
        p=p.sort_index().copy()
        p["carbs_recent_30m"]=p.carbs_g.rolling(6,min_periods=1).sum()
        p["carbs_recent_60m"]=p.carbs_g.rolling(12,min_periods=1).sum()
        h=p.index.hour+p.index.minute/60.
        p["hour_sin"]=np.sin(2*np.pi*h/24.)
        p["hour_cos"]=np.cos(2*np.pi*h/24.)
        out.append(p)
    return pd.concat(out).sort_index()

def seq(frame,scaler):
    xs=[];ys=[];meta=[]
    for pid,p in frame.groupby("p_id"):
        p=p.sort_index().copy(); z=scaler.transform(p[FEATURES].fillna(0.))
        for i in range(LOOKBACK,len(p)):
            r=p.iloc[i]
            xs.append(z[i-LOOKBACK:i]); ys.append(int(r.target_hypo))
            meta.append({"p_id":int(pid),"timestamp":p.index[i],"glucose":float(r.glucose),
                         "delta15":float(r.glucose_delta_15m) if pd.notna(r.glucose_delta_15m) else 0.0})
    return np.asarray(xs,np.float32),np.asarray(ys,np.int32),pd.DataFrame(meta)

def val_patient(ids,test):
    c=[x for x in sorted(ids) if x!=test]
    return c[(SEED+int(test))%len(c)]

def policy_scores(meta,probs,threshold,persistence,rescue_glucose,rescue_delta15):
    """Binary alert score after persistence filtering, independently per patient."""
    out=np.zeros(len(meta),dtype=float)
    work=meta.reset_index(drop=True).copy(); work["p"]=np.asarray(probs)
    for _,idx in work.groupby("p_id").groups.items():
        idx=np.asarray(list(idx),dtype=int)
        p=work.loc[idx,"p"].to_numpy(); above=(p>=threshold).astype(int)
        # Consecutive persistence: rolling sum equals requested run length.
        run=np.zeros(len(idx),dtype=int); count=0
        for j,a in enumerate(above):
            count=count+1 if a else 0; run[j]=count
        glucose=work.loc[idx,"glucose"].to_numpy(); delta15=work.loc[idx,"delta15"].to_numpy()
        rescue=(glucose<=rescue_glucose)|(delta15<=rescue_delta15)
        alert=(above==1)&((run>=persistence)|rescue)
        out[idx]=alert.astype(float)
    return out

def evaluate_policy(meta,probs,params):
    score=policy_scores(meta,probs,**params)
    return evaluate_events(meta,score,0.5,30)

def policy_grid(meta,probs):
    rows=[]
    for t in THRESHOLDS:
        for k in PERSISTENCE:
            for g in RESCUE_GLUCOSE:
                for d in RESCUE_DELTA15:
                    params={"threshold":float(t),"persistence":int(k),"rescue_glucose":float(g),"rescue_delta15":float(d)}
                    e=evaluate_policy(meta,probs,params)
                    rows.append({**params,"event_recall":e["event_recall"],"median_warning_minutes":e["median_warning_minutes"],
                                 "false_alerts_per_patient_day":e["false_alerts_per_patient_day"],"false_alert_episodes":e["false_alert_episodes"],
                                 "detected_events":e["detected_events"],"hypoglycemia_events":e["hypoglycemia_events"]})
    return pd.DataFrame(rows)

def choose_policy(grid):
    ok=grid[(grid.event_recall>=.90)&(grid.median_warning_minutes>=15.)]
    if len(ok):
        row=ok.sort_values(["false_alerts_per_patient_day","event_recall","median_warning_minutes"],ascending=[True,False,False]).iloc[0]
        policy="constrained_min_fp"
    else:
        row=grid.sort_values(["event_recall","false_alerts_per_patient_day","median_warning_minutes"],ascending=[False,True,False]).iloc[0]
        policy="fallback_max_recall"
    params={k:(int(row[k]) if k=="persistence" else float(row[k])) for k in ["threshold","persistence","rescue_glucose","rescue_delta15"]}
    return params,policy

def fold(data,test_id,outdir):
    seed_all(); rem=[int(x) for x in sorted(data.p_id.unique()) if int(x)!=test_id]; vid=val_patient(rem,test_id); tids=[x for x in rem if x!=vid]
    tr=data[data.p_id.isin(tids)]; va=data[data.p_id==vid]; te=data[data.p_id==test_id]
    scaler=StandardScaler().fit(tr[FEATURES].fillna(0.)); xtr,ytr,_=seq(tr,scaler); xv,yv,mv=seq(va,scaler); xt,yt,mt=seq(te,scaler)
    classes=np.unique(ytr); w=compute_class_weight("balanced",classes=classes,y=ytr); cw={int(c):float(v) for c,v in zip(classes,w)}
    model=build_hypoglycemia_classifier(LOOKBACK,len(FEATURES)); model.compile(optimizer="adam",loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=35,batch_size=128,shuffle=False,
              callbacks=[EarlyStopping(monitor="val_auc",patience=8,restore_best_weights=True,mode="max")],verbose=0)
    pv=model.predict(xv,verbose=0).ravel(); pt=model.predict(xt,verbose=0).ravel()
    grid=policy_grid(mv,pv); params,selection=choose_policy(grid)
    d=outdir/f"patient_{test_id}"; d.mkdir(parents=True,exist_ok=True); grid.to_csv(d/"validation_policy_grid.csv",index=False)
    e=evaluate_policy(mt,pt,params); auc=float(roc_auc_score(yt,pt)) if len(np.unique(yt))>1 else None
    return {"test_patient":int(test_id),"validation_patient":int(vid),"train_patients":tids,"selected_policy":params,
            "selection_policy":selection,"roc_auc":auc,"hypoglycemia_events":e["hypoglycemia_events"],"detected_events":e["detected_events"],
            "missed_events":e["missed_events"],"event_recall":e["event_recall"],"median_warning_minutes":e["median_warning_minutes"],
            "false_alert_episodes":e["false_alert_episodes"],"false_alerts_per_patient_day":e["false_alerts_per_patient_day"],
            "observed_patient_days":e["observed_patient_days"]}

def aggregate(folds):
    d=pd.DataFrame(folds); ev=int(d.hypoglycemia_events.sum()); det=int(d.detected_events.sum())
    full=(d.event_recall>=.9)&(d.false_alerts_per_patient_day<=1.)&(d.median_warning_minutes>=15.)
    return {"patients":len(d),"pooled_events":ev,"pooled_detected":det,"pooled_event_recall":det/ev if ev else None,
            "mean_patient_event_recall":float(d.event_recall.mean()),"median_patient_event_recall":float(d.event_recall.median()),
            "mean_false_alerts_per_patient_day":float(d.false_alerts_per_patient_day.mean()),
            "median_false_alerts_per_patient_day":float(d.false_alerts_per_patient_day.median()),
            "median_warning_minutes_across_patients":float(d.median_warning_minutes.dropna().median()),
            "patients_meeting_full_target":int(full.sum())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True,type=Path); ap.add_argument("--output-dir",default=Path("models/v9_alert_intelligence"),type=Path); a=ap.parse_args()
    data=add_context(load_ohio_directory(a.data_dir)); a.output_dir.mkdir(parents=True,exist_ok=True); dataset_summary(data).to_csv(a.output_dir/"dataset_summary.csv",index=False); folds=[]
    for pid in sorted(int(x) for x in data.p_id.unique()):
        print(f"V9 held-out patient {pid}",flush=True); folds.append(fold(data,pid,a.output_dir)); pd.DataFrame(folds).to_csv(a.output_dir/"v9_per_patient.csv",index=False)
    report={"model":"hypo_v9_alert_intelligence","seed":SEED,"features":FEATURES,"lookback_minutes":240,"prediction_horizon_minutes":30,
            "protocol":"LOPO; V8 contextual LSTM plus validation-selected persistence/rescue alert policy",
            "policy_grid":{"thresholds":[round(float(x),2) for x in THRESHOLDS],"persistence_steps":list(PERSISTENCE),
                           "rescue_glucose":list(RESCUE_GLUCOSE),"rescue_delta15":list(RESCUE_DELTA15)},
            "selection_constraints":{"event_recall_min":.9,"median_warning_minutes_min":15.},"folds":folds,"aggregate":aggregate(folds),
            "research_target":{"event_recall_min":.9,"false_alerts_per_patient_day_max":1.,"median_warning_minutes_min":15},
            "clinical_status":"research only; not clinically validated"}
    (a.output_dir/"v9_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report["aggregate"],indent=2))
if __name__=="__main__": main()
