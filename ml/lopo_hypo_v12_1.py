"""Hypo V12.1: temporal safety gate for personalized fine-tuning.

Timeline for each held-out patient:
- Days 1-21: personal fine-tuning only.
- Days 22-30: untouched personal validation / safety gate.
- Day 31+: completely blind future test.

The population LSTM is trained without the held-out patient. Fine-tuning adapts
only dense layers. A personalized candidate is activated only if it proves on
the temporal gate window that it preserves/improves recall versus population
and meets notification-burden safeguards. Otherwise population stays active.
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

SEED=42; LOOKBACK=48
FT_END_DAY=21; GATE_END_DAY=30
MIN_FT_EVENTS=3; MIN_GATE_EVENTS=3
FT_EPOCHS=10; FT_LR=1e-4
MATERIAL_RECALL_GAIN_PP=5.0
MAX_NOTIFICATION_INCREASE_IF_MATERIAL_GAIN=0.50
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

def notifications_per_day(meta,sig):
 m=meta.reset_index(drop=True).copy(); m["timestamp"]=pd.to_datetime(m.timestamp); m["n"]=np.asarray(sig)>=.5; n=int(m.n.sum()); days=0.
 for _,f in m.groupby("p_id"): days+=max((f.timestamp.max()-f.timestamp.min()).total_seconds()/86400.,5./1440.)
 return n/days if days else None

def evaluate(meta,probs,params):
 a,n=state_machine(meta,probs,**params); ev=evaluate_events(meta,a,.5,30); return ev,notifications_per_day(meta,n)

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
 for layer in m.layers: layer.trainable=not isinstance(layer,(tf.keras.layers.LSTM,tf.keras.layers.Bidirectional))
 m.compile(optimizer=tf.keras.optimizers.Adam(FT_LR),loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(name="auc")])
 classes=np.unique(y); cw=None
 if len(classes)>1:
  ws=compute_class_weight("balanced",classes=classes,y=y); cw={int(c):float(w) for c,w in zip(classes,ws)}
 m.fit(x,y,epochs=FT_EPOCHS,batch_size=64,shuffle=False,class_weight=cw,verbose=0)
 return m

def gate_decision(pop_ev,pop_nt,cand_ev,cand_nt):
 pr=float(pop_ev["event_recall"]); cr=float(cand_ev["event_recall"]); gain_pp=(cr-pr)*100.
 if cand_ev["median_warning_minutes"] is None or cand_ev["median_warning_minutes"]<15.: return False,"blocked_warning_time",gain_pp
 if cr<pr: return False,"blocked_recall_degradation",gain_pp
 if gain_pp>=MATERIAL_RECALL_GAIN_PP:
  if cand_nt<=pop_nt+MAX_NOTIFICATION_INCREASE_IF_MATERIAL_GAIN: return True,"activated_material_recall_gain",gain_pp
  return False,"blocked_excess_notifications_despite_recall_gain",gain_pp
 if cand_nt<=pop_nt: return True,"activated_noninferior_recall_lower_notifications",gain_pp
 return False,"blocked_no_product_benefit",gain_pp

def run_patient(data,pid,outdir):
 seed_all(); others=[int(x) for x in sorted(data.p_id.unique()) if int(x)!=pid]; vid=val_patient(others,pid); train_ids=[x for x in others if x!=vid]
 train=data[data.p_id.isin(train_ids)]; val=data[data.p_id==vid]; target=data[data.p_id==pid].sort_index()
 scaler=StandardScaler().fit(train[FEATURES].fillna(0.)); xtr,ytr,_=sequences(train,scaler); xv,yv,mv=sequences(val,scaler); xt,yt,mt=sequences(target,scaler)
 classes=np.unique(ytr); ws=compute_class_weight("balanced",classes=classes,y=ytr); cw={int(c):float(w) for c,w in zip(classes,ws)}
 pop=build_hypoglycemia_classifier(LOOKBACK,len(FEATURES)); pop.compile(optimizer="adam",loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(name="auc")])
 pop.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=35,batch_size=128,shuffle=False,callbacks=[EarlyStopping(monitor="val_auc",patience=8,restore_best_weights=True,mode="max")],verbose=0)
 pval=pop.predict(xv,verbose=0).ravel(); ptarget=pop.predict(xt,verbose=0).ravel(); pop_policy,_=choose(grid(mv,pval))
 start=pd.to_datetime(mt.timestamp).min(); ft_end=start+pd.Timedelta(days=FT_END_DAY); gate_end=start+pd.Timedelta(days=GATE_END_DAY)
 mft,pft_base,ft_idx=subset(mt,ptarget,end=ft_end); mgate,pgate_base,gate_idx=subset(mt,ptarget,start=ft_end,end=gate_end); mtest,ptest_base,test_idx=subset(mt,ptarget,start=gate_end)
 xft=xt[ft_idx]; yft=yt[ft_idx]; xgate=xt[gate_idx]; ygate=yt[gate_idx]; xtest=xt[test_idx]; ytest=yt[test_idx]
 ft_events=count_events(mft); gate_events=count_events(mgate)
 pop_gate_ev,pop_gate_nt=evaluate(mgate,pgate_base,pop_policy); pop_test_ev,pop_test_nt=evaluate(mtest,ptest_base,pop_policy)
 pdir=outdir/f"patient_{pid}"; pdir.mkdir(parents=True,exist_ok=True)
 if ft_events<MIN_FT_EVENTS:
  activated=False; reason="fallback_insufficient_finetune_events"; cand_policy=pop_policy; cand_gate_ev=pop_gate_ev; cand_gate_nt=pop_gate_nt; cand_model=None
 elif gate_events<MIN_GATE_EVENTS:
  activated=False; reason="fallback_insufficient_gate_events"; cand_policy=pop_policy; cand_gate_ev=pop_gate_ev; cand_gate_nt=pop_gate_nt; cand_model=None
 else:
  cand_model=fine_tune(pop,xft,yft); pgate_cand=cand_model.predict(xgate,verbose=0).ravel(); cand_grid=grid(mgate,pgate_cand); cand_grid.to_csv(pdir/"candidate_gate_policy_grid.csv",index=False); cand_policy,_=choose(cand_grid); cand_gate_ev,cand_gate_nt=evaluate(mgate,pgate_cand,cand_policy); activated,reason,_=gate_decision(pop_gate_ev,pop_gate_nt,cand_gate_ev,cand_gate_nt)
 if activated:
  ptest_active=cand_model.predict(xtest,verbose=0).ravel(); active_policy=cand_policy; active_model_name="personalized"
 else:
  ptest_active=ptest_base; active_policy=pop_policy; active_model_name="population"
 active_test_ev,active_test_nt=evaluate(mtest,ptest_active,active_policy)
 active_auc=float(roc_auc_score(ytest,ptest_active)) if len(np.unique(ytest))>1 else None
 gate_gain=(float(cand_gate_ev["event_recall"])-float(pop_gate_ev["event_recall"]))*100.
 return {"test_patient":pid,"population_validation_patient":vid,"population_train_patients":train_ids,"finetune_days":FT_END_DAY,"gate_days":GATE_END_DAY-FT_END_DAY,"test_start_day":GATE_END_DAY+1,"finetune_hypoglycemia_events":ft_events,"gate_hypoglycemia_events":gate_events,"activation":active_model_name,"gate_reason":reason,"population_policy":pop_policy,"candidate_policy":cand_policy,"active_policy":active_policy,"gate_population_recall":pop_gate_ev["event_recall"],"gate_candidate_recall":cand_gate_ev["event_recall"],"gate_recall_delta_pp":gate_gain,"gate_population_notifications_per_day":pop_gate_nt,"gate_candidate_notifications_per_day":cand_gate_nt,"test_events":int(active_test_ev["hypoglycemia_events"]),"test_population_detected":int(pop_test_ev["detected_events"]),"test_population_recall":pop_test_ev["event_recall"],"test_population_notifications_per_day":pop_test_nt,"test_active_detected":int(active_test_ev["detected_events"]),"test_active_recall":active_test_ev["event_recall"],"test_active_notifications_per_day":active_test_nt,"test_active_warning_minutes":active_test_ev["median_warning_minutes"],"test_active_auc":active_auc,"test_recall_delta_vs_population_pp":(active_test_ev["event_recall"]-pop_test_ev["event_recall"])*100.,"test_notifications_delta_vs_population":active_test_nt-pop_test_nt}

def summarize(rows):
 d=pd.DataFrame(rows); events=int(d.test_events.sum()); pdect=int(d.test_population_detected.sum()); adect=int(d.test_active_detected.sum()); full=(d.test_active_recall>=.90)&(d.test_active_notifications_per_day<=1.)&(d.test_active_warning_minutes>=15.)
 return {"patients":len(d),"personalized_activated":int((d.activation=="personalized").sum()),"population_fallback":int((d.activation=="population").sum()),"test_events":events,"population_pooled_recall":pdect/events if events else None,"active_pooled_recall":adect/events if events else None,"mean_population_notifications_per_day":float(d.test_population_notifications_per_day.mean()),"mean_active_notifications_per_day":float(d.test_active_notifications_per_day.mean()),"mean_test_recall_delta_pp":float(d.test_recall_delta_vs_population_pp.mean()),"patients_meeting_full_target":int(full.sum())}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True,type=Path); ap.add_argument("--output-dir",default=Path("models/v12_1_temporal_safety_gate"),type=Path); args=ap.parse_args()
 data=add_context(load_ohio_directory(args.data_dir)); args.output_dir.mkdir(parents=True,exist_ok=True); dataset_summary(data).to_csv(args.output_dir/"dataset_summary.csv",index=False)
 rows=[]
 for pid in sorted(int(x) for x in data.p_id.unique()):
  print(f"V12.1 held-out patient {pid}: days 1-21 FT, 22-30 gate, 31+ blind test",flush=True); rows.append(run_patient(data,pid,args.output_dir)); pd.DataFrame(rows).to_csv(args.output_dir/"v12_1_per_patient.csv",index=False)
 agg=summarize(rows); report={"model":"hypo_v12_1_temporal_safety_gate","seed":SEED,"timeline":{"finetune_days":"1-21","temporal_gate_days":"22-30","blind_test":"31+"},"fine_tuning":{"epochs":FT_EPOCHS,"learning_rate":FT_LR,"recurrent_layers":"frozen","dense_layers":"trainable"},"gate_rules":{"minimum_finetune_events":MIN_FT_EVENTS,"minimum_gate_events":MIN_GATE_EVENTS,"candidate_recall_must_be_at_least_population":True,"material_recall_gain_pp":MATERIAL_RECALL_GAIN_PP,"max_notification_increase_if_material_gain":MAX_NOTIFICATION_INCREASE_IF_MATERIAL_GAIN},"research_target":{"event_recall_min":.90,"notifications_per_patient_day_max":1.,"median_warning_minutes_min":15},"aggregate":agg,"rows":rows,"clinical_status":"research only; not clinically validated"}; (args.output_dir/"v12_1_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(agg,indent=2))
if __name__=="__main__": main()
