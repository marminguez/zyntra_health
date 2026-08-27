"""Hypo V12.3: conservative model gate + patient-adaptive alert policy.

Timeline remains leakage-safe:
- Days 1-21: personal fine-tuning.
- Days 22-30: model safety gate AND alert-policy calibration.
- Day 31+: untouched blind future test.

V12.2 decides whether population or personalized probabilities are active. V12.3
then calibrates the alert state-machine policy for that active model using only
days 22-30. The blind future test is never used to select model or policy.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import lopo_hypo_v12_1 as b

MIN_RECALL_GAIN_PP=5.0
MAX_NOTIFICATION_INCREASE=0.50
TARGET_NOTIFICATIONS_PER_DAY=1.50


def conservative_gate(pop_ev,pop_nt,cand_ev,cand_nt):
    gain=(float(cand_ev['event_recall'])-float(pop_ev['event_recall']))*100.
    warning=cand_ev['median_warning_minutes']
    if warning is None or float(warning)<15.: return False,'blocked_warning_time',gain
    if gain<MIN_RECALL_GAIN_PP: return False,'blocked_insufficient_recall_gain',gain
    if float(cand_nt)>float(pop_nt)+MAX_NOTIFICATION_INCREASE:
        return False,'blocked_excess_notifications_despite_recall_gain',gain
    return True,'activated_material_recall_gain',gain


def choose_adaptive(g):
    """Prefer >=90% recall, >=15 min warning and <=1.5 notifications/day.

    If the burden target is infeasible on the gate window, preserve the safety
    constraints and choose the lowest-burden policy. If safety itself is
    infeasible, maximize recall first.
    """
    safe=g[(g.recall>=.90)&(g.warning>=15.)]
    target=safe[safe.notifications<=TARGET_NOTIFICATIONS_PER_DAY]
    if len(target):
        row=target.sort_values(['notifications','false_alerts','recall'],ascending=[True,True,False]).iloc[0]
        tag='adaptive_target_met'
    elif len(safe):
        row=safe.sort_values(['notifications','false_alerts','recall'],ascending=[True,True,False]).iloc[0]
        tag='adaptive_safety_preserved_burden_unmet'
    else:
        row=g.sort_values(['recall','notifications','false_alerts'],ascending=[False,True,True]).iloc[0]
        tag='adaptive_fallback_max_recall'
    p={'threshold':float(row.threshold),'persistence':int(row.persistence),'clear_steps':int(row.clear_steps),'rearm_margin':float(row.rearm_margin)}
    return p,tag


def run_patient(data,pid,outdir):
    b.seed_all(); others=[int(x) for x in sorted(data.p_id.unique()) if int(x)!=pid]; vid=b.val_patient(others,pid); train_ids=[x for x in others if x!=vid]
    train=data[data.p_id.isin(train_ids)]; val=data[data.p_id==vid]; target=data[data.p_id==pid].sort_index()
    scaler=StandardScaler().fit(train[b.FEATURES].fillna(0.)); xtr,ytr,_=b.sequences(train,scaler); xv,yv,mv=b.sequences(val,scaler); xt,yt,mt=b.sequences(target,scaler)
    classes=np.unique(ytr); ws=compute_class_weight('balanced',classes=classes,y=ytr); cw={int(c):float(w) for c,w in zip(classes,ws)}
    pop=b.build_hypoglycemia_classifier(b.LOOKBACK,len(b.FEATURES)); pop.compile(optimizer='adam',loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(name='auc')])
    pop.fit(xtr,ytr,validation_data=(xv,yv),class_weight=cw,epochs=35,batch_size=128,shuffle=False,callbacks=[EarlyStopping(monitor='val_auc',patience=8,restore_best_weights=True,mode='max')],verbose=0)
    pval=pop.predict(xv,verbose=0).ravel(); ptarget=pop.predict(xt,verbose=0).ravel(); population_reference_policy,_=b.choose(b.grid(mv,pval))
    start=pd.to_datetime(mt.timestamp).min(); ft_end=start+pd.Timedelta(days=b.FT_END_DAY); gate_end=start+pd.Timedelta(days=b.GATE_END_DAY)
    mft,_,ft_idx=b.subset(mt,ptarget,end=ft_end); mgate,pgate_pop,gate_idx=b.subset(mt,ptarget,start=ft_end,end=gate_end); mtest,ptest_pop,test_idx=b.subset(mt,ptarget,start=gate_end)
    xft=xt[ft_idx]; yft=yt[ft_idx]; xgate=xt[gate_idx]; xtest=xt[test_idx]; ytest=yt[test_idx]
    ft_events=b.count_events(mft); gate_events=b.count_events(mgate)
    # V12.2 model gate compares each model with its best safety-oriented policy on gate.
    pop_gate_grid=b.grid(mgate,pgate_pop); pop_gate_policy,_=b.choose(pop_gate_grid); pop_gate_ev,pop_gate_nt=b.evaluate(mgate,pgate_pop,pop_gate_policy)
    pdir=outdir/f'patient_{pid}'; pdir.mkdir(parents=True,exist_ok=True); pop_gate_grid.to_csv(pdir/'population_gate_policy_grid.csv',index=False)
    cand=None; pgate_cand=None; cand_gate_policy=pop_gate_policy; cand_gate_ev=pop_gate_ev; cand_gate_nt=pop_gate_nt
    if ft_events<b.MIN_FT_EVENTS: activated=False; reason='fallback_insufficient_finetune_events'
    elif gate_events<b.MIN_GATE_EVENTS: activated=False; reason='fallback_insufficient_gate_events'
    else:
        cand=b.fine_tune(pop,xft,yft); pgate_cand=cand.predict(xgate,verbose=0).ravel(); cg=b.grid(mgate,pgate_cand); cg.to_csv(pdir/'candidate_gate_policy_grid.csv',index=False); cand_gate_policy,_=b.choose(cg); cand_gate_ev,cand_gate_nt=b.evaluate(mgate,pgate_cand,cand_gate_policy); activated,reason,_=conservative_gate(pop_gate_ev,pop_gate_nt,cand_gate_ev,cand_gate_nt)
    # NEW IN V12.3: calibrate alert policy on this patient's gate window for whichever model V12.2 selected.
    if activated:
        active_model='personalized'; active_gate_probs=pgate_cand; ptest_active=cand.predict(xtest,verbose=0).ravel()
    else:
        active_model='population'; active_gate_probs=pgate_pop; ptest_active=ptest_pop
    adaptive_grid=b.grid(mgate,active_gate_probs); adaptive_grid.to_csv(pdir/'active_adaptive_policy_grid.csv',index=False); active_policy,policy_reason=choose_adaptive(adaptive_grid)
    gate_active_ev,gate_active_nt=b.evaluate(mgate,active_gate_probs,active_policy)
    pop_test_ev,pop_test_nt=b.evaluate(mtest,ptest_pop,population_reference_policy)
    active_test_ev,active_test_nt=b.evaluate(mtest,ptest_active,active_policy)
    auc=float(roc_auc_score(ytest,ptest_active)) if len(np.unique(ytest))>1 else None
    return {'test_patient':pid,'activation':active_model,'gate_reason':reason,'adaptive_policy_reason':policy_reason,'finetune_hypoglycemia_events':ft_events,'gate_hypoglycemia_events':gate_events,'population_reference_policy':population_reference_policy,'active_policy':active_policy,'gate_active_recall':gate_active_ev['event_recall'],'gate_active_notifications_per_day':gate_active_nt,'test_events':int(active_test_ev['hypoglycemia_events']),'test_population_detected':int(pop_test_ev['detected_events']),'test_population_recall':pop_test_ev['event_recall'],'test_population_notifications_per_day':pop_test_nt,'test_active_detected':int(active_test_ev['detected_events']),'test_active_recall':active_test_ev['event_recall'],'test_active_notifications_per_day':active_test_nt,'test_active_warning_minutes':active_test_ev['median_warning_minutes'],'test_active_auc':auc,'test_recall_delta_vs_population_pp':(active_test_ev['event_recall']-pop_test_ev['event_recall'])*100.,'test_notifications_delta_vs_population':active_test_nt-pop_test_nt}


def summarize(rows):
    d=pd.DataFrame(rows); events=int(d.test_events.sum()); pdect=int(d.test_population_detected.sum()); adect=int(d.test_active_detected.sum()); full=(d.test_active_recall>=.90)&(d.test_active_notifications_per_day<=TARGET_NOTIFICATIONS_PER_DAY)&(d.test_active_warning_minutes>=15.)
    return {'patients':len(d),'personalized_activated':int((d.activation=='personalized').sum()),'population_fallback':int((d.activation=='population').sum()),'test_events':events,'population_pooled_recall':pdect/events if events else None,'active_pooled_recall':adect/events if events else None,'mean_population_notifications_per_day':float(d.test_population_notifications_per_day.mean()),'mean_active_notifications_per_day':float(d.test_active_notifications_per_day.mean()),'mean_test_recall_delta_pp':float(d.test_recall_delta_vs_population_pp.mean()),'patients_meeting_v12_3_target':int(full.sum())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True,type=Path); ap.add_argument('--output-dir',default=Path('ml/models/v12_3_adaptive_alert_policy'),type=Path); a=ap.parse_args()
    data=b.add_context(b.load_ohio_directory(a.data_dir)); a.output_dir.mkdir(parents=True,exist_ok=True); b.dataset_summary(data).to_csv(a.output_dir/'dataset_summary.csv',index=False)
    rows=[]
    for pid in sorted(int(x) for x in data.p_id.unique()):
        print(f'V12.3 held-out patient {pid}: FT 1-21, gate+policy 22-30, blind test 31+',flush=True); rows.append(run_patient(data,pid,a.output_dir)); pd.DataFrame(rows).to_csv(a.output_dir/'v12_3_per_patient.csv',index=False)
    agg=summarize(rows); report={'model':'hypo_v12_3_adaptive_alert_policy','seed':b.SEED,'timeline':{'finetune_days':'1-21','model_gate_and_policy_calibration_days':'22-30','blind_test':'31+'},'model_gate':{'minimum_recall_gain_pp':MIN_RECALL_GAIN_PP,'max_notification_increase_per_day':MAX_NOTIFICATION_INCREASE},'adaptive_alert_policy':{'target_recall_min':.90,'target_notifications_per_patient_day_max':TARGET_NOTIFICATIONS_PER_DAY,'minimum_warning_minutes':15,'selection_data':'days 22-30 only'},'aggregate':agg,'rows':rows,'clinical_status':'research only; not clinically validated'}; (a.output_dir/'v12_3_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(agg,indent=2)); print(f'V12.3 completed successfully. Results: {a.output_dir}',flush=True)
if __name__=='__main__': main()
