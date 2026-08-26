"""Hypo V10: LOPO with alert state machine and cooldown suppression."""
from __future__ import annotations
import argparse,json,os,random
from pathlib import Path
os.environ.setdefault("PYTHONHASHSEED","42"); os.environ.setdefault("TF_DETERMINISTIC_OPS","1")
import numpy as np, pandas as pd, tensorflow as tf
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
THRESHOLDS=np.arange(.40,.91,.05); PERSISTENCE=(1,2,3); COOLDOWN_MIN=(30,45,60,90)

def seed_all():
 random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
 try: tf.config.experimental.enable_op_determinism()
 except Exception: pass

def add_context(df):
 out=[]
 for pid,p in df.groupby("p_id"):
  p=p.sort_index().copy(); p["carbs_recent_30m"]=p.carbs_g.rolling(6,min_periods=1).sum(); p["carbs_recent_60m"]=p.carbs_g.rolling(12,min_periods=1).sum(); h=p.index.hour+p.index.minute/60.; p["hour_sin"]=np.sin(2*np.pi*h/24.); p["hour_cos"]=np.cos(2*np.pi*h/24.); out.append(p)
 return pd.concat(out).sort_index()

def seq(frame,scaler):
 xs=[];ys=[];meta=[]
 for pid,p in frame.groupby("p_id"):
  p=p.sort_index().copy(); z=scaler.transform(p[FEATURES].fillna(0.))
  for i in range(LOOKBACK,len(p)):
   xs.append(z[i-LOOKBACK:i]); ys.append(int(p.iloc[i].target_hypo)); meta.append({"p_id":int(pid),"timestamp":p.index[i],"glucose":float(p.iloc[i].glucose)})
 return np.asarray(xs,np.float32),np.asarray(ys,np.int32),pd.DataFrame(meta)

def val_patient(ids,test):
 c=[x for x in sorted(ids) if x!=test]; return c[(SEED+int(test))%len(c)]

def state_machine(meta,probs,threshold,persistence,cooldown_minutes):
 out=np.zeros(len(meta),float); work=meta.reset_index(drop=True).copy(); work["p"]=np.asarray(probs); cooldown_steps=int(cooldown_minutes/5)
 for _,idxs in work.groupby("p_id").groups.items():
  idxs=list(idxs); streak=0; cooldown=0
  for pos in idxs:
   p=float(work.at[pos,"p"])
   if cooldown>0:
    cooldown-=1; streak=0; continue
   if p>=threshold: streak+=1
   else: streak=0
   if streak>=persistence:
    out[pos]=1.0; cooldown=cooldown_steps; streak=0
 return out

def eval_policy(meta,probs,params): return evaluate_events(meta,state_machine(meta,probs,**params),0.5,30)

def grid(meta,probs):
 rows=[]
 for t in THRESHOLDS:
  for k in PERSISTENCE:
   for cd in COOLDOWN_MIN:
    p={"threshold":float(t),"persistence":int(k),"cooldown_minutes":int(cd)}; e=eval_policy(meta,probs,p)
    rows.append({**p,"event_recall":e["event_recall"],"median_warning_minutes":e["median_warning_minutes"],"false_alerts_per_patient_day":e["false_alerts_per_patient_day"],"false_alert_episodes":e["false_alert_episodes"],"detected_events":e["detected_events"],"hypoglycemia_events":e["hypoglycemia_events"]})
 return pd.DataFrame(rows)

def choose_policy(g):
 ok=g[(g.event_recall>=.90)&(g.median_warning_minutes>=15.)]
 if len(ok): row=ok.sort_values(["false_alerts_per_patient_day","event_recall","median_warning_minutes"],ascending=[True,False,False]).iloc[0]; tag="constrained_min_fp"
 else: row=g.sort_values(["event_recall","false_alerts_per_patient_day"],ascending=[False,True]).iloc[0]; tag="fallback_max_recall"
 return {"threshold":float(row.threshold),"persistence":int(row.persistence),"cooldown_minutes":int(row.cooldown_minutes)},tag

def fold(data,test_id,outdir):
 seed_all(); rem=[int(x) for x in sorted(data.p_id.unique()) if int(x)!=test_id]; vid=val_patient(rem,test_id); tids=[x for x in rem if x!=vid]; tr=data[data.p_id.isin(tids)]; va=data[data.p_id==vid]; te=data[data.p_id==test_id]
 sc=StandardScaler().fit(tr[FEATURES].fillna(0.)); xtr,ytr,_=seq(tr,sc); xv,yv,mv=seq(va,sc); xt,yt,mt=seq(te,sc)
 classes=np.unique(ytr); w=compute_class_weight("balanced",classes=classes,y=ytr); cw={int(c):float(v) for c,v in zip(classes,w)}
 m=build_hypoglycemia_classifier(LOOKBACK,len(FEATURES)); m.compile(optimizer="adam",loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(name="auc")]); m.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=35,batch_size=128,shuffle=False,callbacks=[EarlyStopping(monitor="val_auc",patience=8,restore_best_weights=True,mode="max")],verbose=0)
 pv=m.predict(xv,verbose=0).ravel(); pt=m.predict(xt,verbose=0).ravel(); g=grid(mv,pv); params,policy=choose_policy(g); d=outdir/f"patient_{test_id}"; d.mkdir(parents=True,exist_ok=True); g.to_csv(d/"validation_state_machine_grid.csv",index=False); e=eval_policy(mt,pt,params); auc=float(roc_auc_score(yt,pt)) if len(np.unique(yt))>1 else None
 return {"test_patient":test_id,"validation_patient":vid,"train_patients":tids,"selected_policy":params,"selection_policy":policy,"roc_auc":auc,"hypoglycemia_events":e["hypoglycemia_events"],"detected_events":e["detected_events"],"missed_events":e["missed_events"],"event_recall":e["event_recall"],"median_warning_minutes":e["median_warning_minutes"],"false_alert_episodes":e["false_alert_episodes"],"false_alerts_per_patient_day":e["false_alerts_per_patient_day"],"observed_patient_days":e["observed_patient_days"]}

def aggregate(f):
 d=pd.DataFrame(f); ev=int(d.hypoglycemia_events.sum()); det=int(d.detected_events.sum()); full=(d.event_recall>=.9)&(d.false_alerts_per_patient_day<=1.)&(d.median_warning_minutes>=15.)
 return {"patients":len(d),"pooled_events":ev,"pooled_detected":det,"pooled_event_recall":det/ev if ev else None,"mean_patient_event_recall":float(d.event_recall.mean()),"median_patient_event_recall":float(d.event_recall.median()),"mean_false_alerts_per_patient_day":float(d.false_alerts_per_patient_day.mean()),"median_false_alerts_per_patient_day":float(d.false_alerts_per_patient_day.median()),"median_warning_minutes_across_patients":float(d.median_warning_minutes.dropna().median()),"patients_meeting_full_target":int(full.sum())}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True,type=Path); ap.add_argument("--output-dir",default=Path("models/v10_alert_state_machine"),type=Path); a=ap.parse_args(); data=add_context(load_ohio_directory(a.data_dir)); a.output_dir.mkdir(parents=True,exist_ok=True); dataset_summary(data).to_csv(a.output_dir/"dataset_summary.csv",index=False); folds=[]
 for pid in sorted(int(x) for x in data.p_id.unique()): print(f"V10 held-out patient {pid}",flush=True); folds.append(fold(data,pid,a.output_dir)); pd.DataFrame(folds).to_csv(a.output_dir/"v10_per_patient.csv",index=False)
 report={"model":"hypo_v10_alert_state_machine","seed":SEED,"features":FEATURES,"lookback_minutes":240,"prediction_horizon_minutes":30,"protocol":"LOPO; contextual LSTM plus validation-selected alert cooldown state machine","state_machine_grid":{"thresholds":[round(float(x),2) for x in THRESHOLDS],"persistence_steps":list(PERSISTENCE),"cooldown_minutes":list(COOLDOWN_MIN)},"selection_constraints":{"event_recall_min":.9,"median_warning_minutes_min":15.},"folds":folds,"aggregate":aggregate(folds),"research_target":{"event_recall_min":.9,"false_alerts_per_patient_day_max":1.,"median_warning_minutes_min":15},"clinical_status":"research only; not clinically validated"}; (a.output_dir/"v10_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report["aggregate"],indent=2))
if __name__=="__main__": main()
