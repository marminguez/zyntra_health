"""V16 Event-Aware Trajectory Scale-10 experiment.

Controlled change vs converged Trajectory: auxiliary excursion-dynamics heads
condition each horizon prediction end-to-end. Dataset/split/normalization,
future-known inputs, 50/50 fusion and seed remain frozen.
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,tensorflow as tf
from ml.forecasting.model_v16_event_aware import HORIZONS,build_v16_forecaster
from ml.train_v15_4_scale10 import SEED,ShardSequence as BaseSequence,evaluate,load_validation,normalize_future,train_manifest

EVENT_THRESHOLD=40.0

def event_labels(y,current):
    d=y-current[:,None]
    return np.where(d<=-EVENT_THRESHOLD,0,np.where(d>=EVENT_THRESHOLD,2,1)).astype(np.int32)

class V16Sequence(BaseSequence):
    def __getitem__(self,index):
        inputs,targets=super().__getitem__(index)
        # Base targets are abs(4)+delta(4); current = absolute target - delta target.
        current=np.asarray(targets[0])-np.asarray(targets[4])
        y=np.column_stack([np.asarray(targets[i]) for i in range(4)])
        ev=event_labels(y,current)
        return inputs,tuple(targets)+tuple(ev[:,i] for i in range(4))

def regime_metrics(y,p,current):
    d=y-current[:,None]; rows=[]
    regimes={"large_fall":d<=-40,"stable":np.abs(d)<20,"large_rise":d>=40,"hypo":y<70,"hyper":y>180}
    for name,mask in regimes.items():
        for i,h in enumerate(HORIZONS):
            m=mask[:,i]
            if not np.any(m):continue
            e=p[m,i]-y[m,i]
            rows.append({"regime":name,"horizon_minutes":h,"n":int(m.sum()),"rmse":float(np.sqrt(np.mean(e**2))),"mae":float(np.mean(np.abs(e)))})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data-dir",default="ml/data/v15_4_scale10");ap.add_argument("--v14-1-dir",default="ml/results/v14_1");ap.add_argument("--outdir",default="ml/results/v16_event_aware_scale10");ap.add_argument("--epochs",type=int,default=40);ap.add_argument("--batch-size",type=int,default=64);a=ap.parse_args()
    random.seed(SEED);np.random.seed(SEED);tf.random.set_seed(SEED)
    data=Path(a.data_dir);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);manifest=train_manifest(data);ntrain=sum(n for _,n in manifest)
    xr,fr,y=load_validation(data);cur=xr[:,-1,0].astype(float);dv=y-cur[:,None];ev=event_labels(y,cur)
    with np.load(Path(a.v14_1_dir)/"normalization.npz") as z:mean=z["mean"].astype(np.float32);std=z["std"].astype(np.float32)
    xv=((xr-mean)/std).astype(np.float32);fv=normalize_future(fr,mean,std)
    targets=tuple(y[:,i] for i in range(4))+tuple(dv[:,i] for i in range(4))+tuple(ev[:,i] for i in range(4))
    seq=V16Sequence(manifest,mean,std,batch_size=a.batch_size,seed=SEED);model=build_v16_forecaster(xv.shape[1],xv.shape[2])
    cb=[tf.keras.callbacks.BackupAndRestore(backup_dir=str(out/"training_backup"),save_freq="epoch",delete_checkpoint=False),tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=2,min_lr=1e-5),tf.keras.callbacks.ModelCheckpoint(str(out/"best.weights.h5"),monitor="val_loss",save_best_only=True,save_weights_only=True),tf.keras.callbacks.CSVLogger(str(out/"training_history.csv"),append=True)]
    print("=== V16 EVENT-AWARE TRAJECTORY SCALE-10 ===");print(f"train windows: {ntrain:,} | validation: {len(y):,} | epochs ceiling: {a.epochs}");print("event target: delta <= -40 / neutral / delta >= +40 | future CGM: False")
    hist=model.fit(seq,validation_data=([xv,fv],targets),epochs=a.epochs,callbacks=cb,verbose=2)
    best=out/"best.weights.h5"
    if best.exists():model.load_weights(best)
    raw=model.predict([xv,fv],batch_size=a.batch_size,verbose=1)
    absolute=np.column_stack([raw[i].ravel() for i in range(4)]);delta=np.column_stack([raw[i+4].ravel() for i in range(4)]);recon=cur[:,None]+delta;final=.5*absolute+.5*recon
    mf=evaluate(y,final,cur);rg=regime_metrics(y,final,cur)
    mf.to_csv(out/"v16_scale10_validation_metrics.csv",index=False);rg.to_csv(out/"v16_regime_metrics.csv",index=False)
    np.savez_compressed(out/"validation_predictions.npz",y_true=y,current_glucose=cur,y_pred=final,absolute_head=absolute,delta_head=delta)
    bestep=int(np.argmin(hist.history["val_loss"])+1) if hist.history.get("val_loss") else None
    mm=float(mf.mard.mean());rr=float(mf.rmse.mean());dd=float(mf.direction_accuracy.mean())
    gate=bool(mm<=13.5 and rr<=29.0 and dd>=.698671)
    report={"experiment":"V16 Event-Aware Trajectory Scale-10","event_threshold_mgdl":EVENT_THRESHOLD,"event_loss_weight":.25,"future_cgm_input":False,"epochs_ceiling":a.epochs,"best_val_loss_epoch_this_process":bestep,"mean_mard":mm,"mean_rmse":rr,"mean_direction_accuracy":dd,"scale50_gate":"MARD<=13.5 AND RMSE<=29.0 AND Direction>=69.8671%","gate_pass":gate,"seed":SEED}
    (out/"v16_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\nV16 EVENT-AWARE TRAJECTORY SCALE-10 VALIDATION METRICS");print(mf.to_string(index=False));print("\n4H MEANS");print(f"MARD: {mm:.6f}%");print(f"RMSE: {rr:.6f}");print(f"Direction: {dd*100:.4f}%");print(f"Best val_loss epoch in this process: {bestep} | epochs run in this process: {len(hist.history.get('loss',[]))}");print(f"Scale50 gate: {'PASS' if gate else 'FAIL'}")
    print("\nREGIME RMSE");print(rg.to_string(index=False));print(f"\nArtifacts written to {out}")
if __name__=="__main__":main()
