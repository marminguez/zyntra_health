"""V17 Distributional Trajectory Scale-10 convergence experiment.

Controlled change vs Trajectory: replace point Huber heads with conditional
quantile heads P10/P25/P50/P75/P90. P50 is evaluated as the point forecast.
Dataset/split/normalization/future-known inputs/seed remain frozen.
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,tensorflow as tf
from ml.forecasting.model_v17_distributional import HORIZONS,QUANTILES,build_v17_forecaster
from ml.train_v15_4_scale10 import SEED,ShardSequence as BaseSequence,evaluate,load_validation,normalize_future,train_manifest

class QuantileSequence(BaseSequence):
    def __getitem__(self,index):
        inputs,targets=super().__getitem__(index)
        y=tuple(np.asarray(targets[i]) for i in range(4))
        return inputs,tuple(y[i] for i in range(4) for _ in QUANTILES)

def regime_metrics(y,p,current):
    d=y-current[:,None];rows=[]
    regimes={"large_fall":d<=-40,"stable":np.abs(d)<20,"large_rise":d>=40,"hypo":y<70,"hyper":y>180}
    for name,mask in regimes.items():
        for i,h in enumerate(HORIZONS):
            m=mask[:,i]
            if np.any(m):
                e=p[m,i]-y[m,i]
                rows.append({"regime":name,"horizon_minutes":h,"n":int(m.sum()),"rmse":float(np.sqrt(np.mean(e**2))),"mae":float(np.mean(np.abs(e)))})
    return pd.DataFrame(rows)

def calibration(y,quant):
    rows=[]
    for i,h in enumerate(HORIZONS):
        for j,q in enumerate(QUANTILES):
            pred=quant[:,i,j]
            rows.append({"horizon_minutes":h,"quantile":q,"empirical_coverage":float(np.mean(y[:,i]<=pred)),"mean_prediction":float(np.mean(pred))})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data-dir",default="ml/data/v15_4_scale10");ap.add_argument("--v14-1-dir",default="ml/results/v14_1");ap.add_argument("--outdir",default="ml/results/v17_distributional_scale10");ap.add_argument("--epochs",type=int,default=40);ap.add_argument("--batch-size",type=int,default=64);a=ap.parse_args()
    random.seed(SEED);np.random.seed(SEED);tf.random.set_seed(SEED)
    data=Path(a.data_dir);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);manifest=train_manifest(data);ntrain=sum(n for _,n in manifest)
    xr,fr,y=load_validation(data);cur=xr[:,-1,0].astype(float)
    with np.load(Path(a.v14_1_dir)/"normalization.npz") as z:mean=z["mean"].astype(np.float32);std=z["std"].astype(np.float32)
    xv=((xr-mean)/std).astype(np.float32);fv=normalize_future(fr,mean,std)
    targets=tuple(y[:,i] for i in range(4) for _ in QUANTILES)
    seq=QuantileSequence(manifest,mean,std,batch_size=a.batch_size,seed=SEED);model=build_v17_forecaster(xv.shape[1],xv.shape[2])
    cb=[tf.keras.callbacks.BackupAndRestore(backup_dir=str(out/"training_backup"),save_freq="epoch",delete_checkpoint=False),tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=2,min_lr=1e-5),tf.keras.callbacks.ModelCheckpoint(str(out/"best.weights.h5"),monitor="val_loss",save_best_only=True,save_weights_only=True),tf.keras.callbacks.CSVLogger(str(out/"training_history.csv"),append=True)]
    print("=== V17 DISTRIBUTIONAL TRAJECTORY SCALE-10 ===");print(f"train windows: {ntrain:,} | validation: {len(y):,} | epochs ceiling: {a.epochs}");print("quantiles: P10/P25/P50/P75/P90 | point forecast: P50 | future CGM: False")
    hist=model.fit(seq,validation_data=([xv,fv],targets),epochs=a.epochs,callbacks=cb,verbose=2)
    best=out/"best.weights.h5"
    if best.exists():model.load_weights(best)
    raw=model.predict([xv,fv],batch_size=a.batch_size,verbose=1)
    quant=np.empty((len(y),4,len(QUANTILES)),dtype=np.float32)
    k=0
    for i in range(4):
        for j in range(len(QUANTILES)):
            quant[:,i,j]=raw[k].ravel();k+=1
    p50=quant[:,:,2]
    mf=evaluate(y,p50,cur);rg=regime_metrics(y,p50,cur);cal=calibration(y,quant)
    mf.to_csv(out/"v17_scale10_validation_metrics.csv",index=False);rg.to_csv(out/"v17_regime_metrics.csv",index=False);cal.to_csv(out/"v17_quantile_calibration.csv",index=False)
    np.savez_compressed(out/"validation_predictions.npz",y_true=y,current_glucose=cur,quantiles=quant,y_pred=p50,quantile_levels=np.asarray(QUANTILES))
    bestep=int(np.argmin(hist.history["val_loss"])+1) if hist.history.get("val_loss") else None
    mm=float(mf.mard.mean());rr=float(mf.rmse.mean());dd=float(mf.direction_accuracy.mean())
    lr=rg[rg.regime=="large_rise"].rmse.mean();lf=rg[rg.regime=="large_fall"].rmse.mean()
    gate=bool(mm<=13.5 and rr<=29.0 and dd>=.698671)
    report={"experiment":"V17 Distributional Trajectory Scale-10","quantiles":QUANTILES,"point_forecast":"P50","future_cgm_input":False,"epochs_ceiling":a.epochs,"best_val_loss_epoch_this_process":bestep,"mean_mard":mm,"mean_rmse":rr,"mean_direction_accuracy":dd,"large_rise_mean_rmse":float(lr),"large_fall_mean_rmse":float(lf),"scale50_gate":"MARD<=13.5 AND RMSE<=29.0 AND Direction>=69.8671%","gate_pass":gate,"seed":SEED}
    (out/"v17_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\nV17 DISTRIBUTIONAL TRAJECTORY SCALE-10 VALIDATION METRICS");print(mf.to_string(index=False));print("\n4H MEANS");print(f"MARD: {mm:.6f}%");print(f"RMSE: {rr:.6f}");print(f"Direction: {dd*100:.4f}%");print(f"Best val_loss epoch in this process: {bestep} | epochs run in this process: {len(hist.history.get('loss',[]))}");print(f"Scale50 gate: {'PASS' if gate else 'FAIL'}");print("\nREGIME RMSE");print(rg.to_string(index=False));print("\nQUANTILE CALIBRATION");print(cal.to_string(index=False));print(f"\nArtifacts written to {out}")
if __name__=="__main__":main()
