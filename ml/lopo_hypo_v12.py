"""Hypo V12: controlled personalized fine-tuning with a population safety gate.

For each held-out patient, train a population LSTM without that patient. Then
compare 21-day and 30-day patient adaptation. Fine-tuning starts from the
population weights, uses a low learning rate and few epochs, and never sees the
common future evaluation period (day 30 onward).

A safety gate compares the fine-tuned candidate with the unchanged population
model on the personal calibration history. The candidate is activated only if
it preserves calibration recall (within a small safety margin) and improves the
selection objective; otherwise the population model remains active.
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

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

SEED=42; LOOKBACK=48; WINDOWS=(21,30); COMMON_EVAL_START_DAY=30
MIN_CAL_EVENTS=3; FT_EPOCHS=10; FT_LR=1e-4; SAFETY_RECALL_MARGIN=0.02
BASE=["glucose","glucose_delta_5m","glucose_delta_15m","glucose_delta_30m","glucose_acceleration_15m"]
CONTEXT=["iob_simple","carbs_recent_30m","carbs_recent_60m","basal_rate","exercise_intensity","hour_sin","hour_cos"]
FEATURES=BASE+CONTEXT
THRESHOLDS=np.arange(0.35,0.91,0.05); PERSISTENCE=(1,2,3); CLEAR_STEPS=(2,3,4,6); REARM_MARGIN=(0.05,0.10,0.15)

def seed_all():
 random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
 try: tf.config.experimental.enable_op_determinism()
 except Exception: pass

def add_context(df):
 out=[]
 for pid,p in df.groupby("p_id"):
  p=p.sort_index().copy(); p["carbs_recent_30m"]=p.carbs_g.rolling(6,min_periods=1).sum(); p["carbs_recent_60m"]=p.carbs_g.rolling(12,min_periods=1).sum()
  h=p.index.hour+p.index.minute/60.; p["hour_sin"]=np.sin(2*np.pi*h/24.); p["hour_cos"]=np.cos(2*np.pi*h/24.); out.append(p)
 return pd.concat(out).sort_index()

def sequences(frame,scaler):
 xs=[]; ys=[]; meta=[]
 for pid,p in frame.groupby("p_id"):
  p=p.sort_index().copy(); z=scaler.transform(p[FEATURES].fillna(0.0))
  for i in range(LOOKBACK,len(p)):
   r=p.iloc[i]; xs.append(z[i-LOOKBACK:i]); ys.append(int(r.target_hypo)); meta.append({"p_id":int(pid),"timestamp":p.index[i],"glucose":float(r.glucose)})
 return np.asarray(xs,np.float32),np.asarray(ys,np.int32),pd.DataFrame(meta)

def val_patient(ids,held):
 c=[x for x in sorted(ids) if x!=held]; return c[(SEED+int(held))%len(c)]

def state_machine(meta,probs,threshold,persistence,clear_steps,rearm_margin):
 w=meta.reset_index(drop=True).copy(); w["p"]=np.asarray(probs); alert=np.zeros(len(w)); notify=np.zeros(len(w))
 for _,idxs in w.groupby("p_id").groups.items():
  state="NORMAL"; high=clear=0; clear_t=max(0.,threshold-rearm_margin)
  for pos in list(idxs):
   p=float(w.at[pos,"p"])
   if state=="NORMAL":
    if p>=threshold:
     high=1
     if persistence==1: state="ALERTED"; alert[pos]=notify[pos]=1.; clear=0
     else: state="WATCH"
   elif state=="WATCH":
    if p>=threshold:
     high+=1
     if high>=persistence: state="ALERTED"; alert[pos]=notify[pos]=1.; clear=0
    else: state="NORMAL"; high=0
   else:
    alert[pos]=1.
    if p<clear_t:
     clear+=1
     if clear>=clear_steps: state="NORMAL"; high=clear=0
    else: clear=0
 return alert,notify

def notification_metrics(meta,sig):
 m=meta.reset_index(drop=True).copy(); m["timestamp"]=pd.to_datetime(m.timestamp); m["n"]=np.asarray(sig)>=.5; n=int(m.n.sum()); days=0.
 for _,f in m.groupby("p_id"): days+=max((f.timestamp.max()-f.timestamp.min()).total_seconds()/86400.,5./1440.)
 return n/days if days else None

def evaluate(meta,probs,params):
 a,n=state_machine(meta,probs,**params); ev=evaluate_events(meta,a,.5,30); return ev,notification_metrics(meta,n)

def grid(meta,probs):
 rows=[]
 for t in THRESHOLDS:
  for k in PERSISTENCE:
   for c in CLEAR_STEPS:
    for r in REARM_MARGIN:
     q={"threshold":float(t),"persistence":int(k),"clear_steps":int(c),"rearm_margin":float(r)}; ev,nt=evaluate(meta,probs,q)
     rows.append({**q,"recall":ev["event_recall"],"warning":ev["median_warning_minutes"],"false_alerts":ev["false_alerts_per_patient_day"],"notifications":nt})
 return pd.DataFrame(rows)

def choose(g):
 ok=g[(g.recall>=.90)&(g.warning>=15.)]
 if len(ok): row=ok.sort_values(["notifications","false_alerts","recall"],ascending=[True,True,False]).iloc[0]; tag="constrained_min_notifications"
 else: row=g.sort_values(["recall","notifications","false_alerts"],ascending=[False,True,True]).iloc[0]; tag="fallback_max_recall"
 return {"threshold":float(row.threshold),"persistence":int(row.persistence),"clear_steps":int(row.clear_steps),"rearm_margin":float(row.rearm_margin)},tag

def subset(meta,probs,start=None,end=None):
 ts=pd.to_datetime(meta.timestamp); mask=np.ones(len(meta),dtype=bool)
 if start is not None: mask&=(ts>=start).to_numpy()
 if end is not None: mask&=(ts<end).to_numpy()
 idx=np.flatnonzero(mask); return meta.iloc[idx].reset_index(drop=True),np.asarray(probs)[idx],idx

def count_events(meta):
 if meta.empty:return 0
 m=meta.sort_values("timestamp"); low=m.glucose<70; return int((low&~low.shift(1,fill_value=False)).sum())

def clone_population(pop):
 m=build_hypoglycemia_classifier(LOOKBACK,len(FEATURES)); m.set_weights(pop.get_weights()); return m

def fine_tune(pop,x,y):
 m=clone_population(pop)
 # Freeze the recurrent feature extractor; adapt only dense decision layers.
 for layer in m.layers:
  layer.trainable = not isinstance(layer,(tf.keras.layers.LSTM,tf.keras.layers.Bidirectional))
 m.compile(optimizer=tf.keras.optimizers.Adam(FT_LR),loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(name="auc")])
 classes=np.unique(y); cw=None
 if len(classes)>1:
  ws=compute_class_weight("balanced",classes=classes,y=y); cw={int(c):float(w) for c,w in zip(classes,ws)}
 m.fit(x,y,epochs=FT_EPOCHS,batch_size=64,shuffle=False,class_weight=cw,verbose=0)
 return m

def candidate_score(ev,nt):
 # Recall dominates; notification burden breaks near-ties.
 return float(ev["event_recall"])*10.0-float(nt)

def run_patient(data,pid,outdir):
 seed_all(); others=[int(x) for x in sorted(data.p_id.unique()) if int(x)!=pid]; vid=val_patient(others,pid); train_ids=[x for x in others if x!=vid]
 train=data[data.p_id.isin(train_ids)]; val=data[data.p_id==vid]; target=data[data.p_id==pid].sort_index()
 scaler=StandardScaler().fit(train[FEATURES].fillna(0.)); xtr,ytr,_=sequences(train,scaler); xv,yv,mv=sequences(val,scaler); xt,yt,mt=sequences(target,scaler)
 classes=np.unique(ytr); ws=compute_class_weight("balanced",classes=classes,y=ytr); cw={int(c):float(w) for c,w in zip(classes,ws)}
 pop=build_hypoglycemia_classifier(LOOKBACK,len(FEATURES)); pop.compile(optimizer="adam",loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(name="auc")])
 pop.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=35,batch_size=128,shuffle=False,callbacks=[EarlyStopping(monitor="val_auc",patience=8,restore_best_weights=True,mode="max")],verbose=0)
 pval=pop.predict(xv,verbose=0).ravel(); ptarget=pop.predict(xt,verbose=0).ravel(); pop_policy,_=choose(grid(mv,pval))
 start=pd.to_datetime(mt.timestamp).min(); eval_start=start+pd.Timedelta(days=COMMON_EVAL_START_DAY); meval,peval,eval_idx=subset(mt,ptarget,start=eval_start); xeval=xt[eval_idx]; yeval=yt[eval_idx]
 pop_ev,pop_nt=evaluate(meval,peval,pop_policy); rows=[]; pdir=outdir/f"patient_{pid}"; pdir.mkdir(parents=True,exist_ok=True)
 for days in WINDOWS:
  cutoff=start+pd.Timedelta(days=days); mcal,pcal,cal_idx=subset(mt,ptarget,end=cutoff); xcal=xt[cal_idx]; ycal=yt[cal_idx]; n_events=count_events(mcal)
  # Baseline personal policy using unchanged population probabilities.
  base_policy,_=choose(grid(mcal,pcal)); base_cal_ev,base_cal_nt=evaluate(mcal,pcal,base_policy)
  base_eval_ev,base_eval_nt=evaluate(meval,peval,base_policy)
  if n_events<MIN_CAL_EVENTS:
   ft=pop; ft_policy=pop_policy; gate="population_insufficient_events"
  else:
   ft=fine_tune(pop,xcal,ycal); pft_cal=ft.predict(xcal,verbose=0).ravel(); ft_policy,_=choose(grid(mcal,pft_cal)); ft_cal_ev,ft_cal_nt=evaluate(mcal,pft_cal,ft_policy)
   safe=ft_cal_ev["event_recall"]>=base_cal_ev["event_recall"]-SAFETY_RECALL_MARGIN
   better=candidate_score(ft_cal_ev,ft_cal_nt)>=candidate_score(base_cal_ev,base_cal_nt)
   gate="fine_tuned" if safe and better else "population_safety_fallback"
  if gate=="fine_tuned": pft_eval=ft.predict(xeval,verbose=0).ravel(); active_policy=ft_policy
  else: pft_eval=peval; active_policy=pop_policy
  active_ev,active_nt=evaluate(meval,pft_eval,active_policy)
  auc=float(roc_auc_score(yeval,pft_eval)) if len(np.unique(yeval))>1 else None
  rows.append({"test_patient":pid,"adaptation_days":days,"calibration_hypoglycemia_events":n_events,"activation":gate,"population_policy":pop_policy,"calibrated_policy_only":base_policy,"active_policy":active_policy,"evaluation_events":int(active_ev["hypoglycemia_events"]),"population_detected":int(pop_ev["detected_events"]),"population_recall":pop_ev["event_recall"],"population_notifications_per_day":pop_nt,"policy_only_detected":int(base_eval_ev["detected_events"]),"policy_only_recall":base_eval_ev["event_recall"],"policy_only_notifications_per_day":base_eval_nt,"active_detected":int(active_ev["detected_events"]),"active_recall":active_ev["event_recall"],"active_notifications_per_day":active_nt,"active_warning_minutes":active_ev["median_warning_minutes"],"active_auc":auc,"recall_delta_vs_population_pp":(active_ev["event_recall"]-pop_ev["event_recall"])*100.,"notifications_delta_vs_population":active_nt-pop_nt})
 return rows

def summarize(rows):
 d=pd.DataFrame(rows); out=[]
 for days,g in d.groupby("adaptation_days"):
  events=int(g.evaluation_events.sum()); pdect=int(g.population_detected.sum()); adect=int(g.active_detected.sum()); full=(g.active_recall>=.90)&(g.active_notifications_per_day<=1.)&(g.active_warning_minutes>=15.)
  out.append({"adaptation_days":int(days),"patients":len(g),"fine_tuned_activated":int((g.activation=="fine_tuned").sum()),"events":events,"population_pooled_recall":pdect/events if events else None,"active_pooled_recall":adect/events if events else None,"mean_population_notifications_per_day":float(g.population_notifications_per_day.mean()),"mean_active_notifications_per_day":float(g.active_notifications_per_day.mean()),"mean_recall_delta_pp":float(g.recall_delta_vs_population_pp.mean()),"patients_meeting_full_target":int(full.sum())})
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True,type=Path); ap.add_argument("--output-dir",default=Path("models/v12_personalized_finetuning"),type=Path); args=ap.parse_args()
 data=add_context(load_ohio_directory(args.data_dir)); args.output_dir.mkdir(parents=True,exist_ok=True); dataset_summary(data).to_csv(args.output_dir/"dataset_summary.csv",index=False)
 rows=[]
 for pid in sorted(int(x) for x in data.p_id.unique()):
  print(f"V12 held-out patient {pid}: 21/30-day controlled fine-tuning",flush=True); rows.extend(run_patient(data,pid,args.output_dir)); pd.DataFrame(rows).to_csv(args.output_dir/"v12_per_patient_window.csv",index=False)
 summary=summarize(rows); pd.DataFrame(summary).to_csv(args.output_dir/"v12_window_summary.csv",index=False)
 report={"model":"hypo_v12_personalized_finetuning","seed":SEED,"adaptation_windows_days":list(WINDOWS),"common_evaluation_start_day":COMMON_EVAL_START_DAY,"fine_tuning":{"epochs":FT_EPOCHS,"learning_rate":FT_LR,"recurrent_layers":"frozen","dense_layers":"trainable","safety_recall_margin":SAFETY_RECALL_MARGIN},"protocol":"population LSTM excludes held-out patient; 21/30-day chronological personal data may fine-tune dense decision layers; safety gate evaluated only on calibration history; common future evaluation begins day 30","research_target":{"event_recall_min":.90,"notifications_per_patient_day_max":1.,"median_warning_minutes_min":15},"window_summary":summary,"rows":rows,"clinical_status":"research only; not clinically validated"}
 (args.output_dir/"v12_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
