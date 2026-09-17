"""Full-convergence Scale-10 screen for the causal trajectory forecaster.

Dataset, frozen validation, normalization, targets, future-known channels,
50/50 absolute/delta fusion and seed remain aligned with the original screen.
Training runs to convergence with a 40-epoch safety ceiling.
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,tensorflow as tf
from ml.forecasting.model_trajectory import HORIZONS,READ_STEPS,build_trajectory_forecaster
from ml.train_v15_4_scale10 import SEED,ShardSequence,evaluate,load_validation,normalize_future,train_manifest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='ml/data/v15_4_scale10');ap.add_argument('--v14-1-dir',default='ml/results/v14_1');ap.add_argument('--outdir',default='ml/results/trajectory_scale10_20ep');ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--batch-size',type=int,default=64);a=ap.parse_args()
 random.seed(SEED);np.random.seed(SEED);tf.random.set_seed(SEED)
 data=Path(a.data_dir);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);manifest=train_manifest(data);ntrain=sum(n for _,n in manifest);xr,fr,y=load_validation(data);cur=xr[:,-1,0].astype(float);dv=y-cur[:,None]
 with np.load(Path(a.v14_1_dir)/'normalization.npz') as z:mean=z['mean'].astype(np.float32);std=z['std'].astype(np.float32)
 xv=((xr-mean)/std).astype(np.float32);fv=normalize_future(fr,mean,std);targets=tuple(y[:,i] for i in range(4))+tuple(dv[:,i] for i in range(4));seq=ShardSequence(manifest,mean,std,batch_size=a.batch_size,seed=SEED);model=build_trajectory_forecaster(xv.shape[1],xv.shape[2])
 cb=[tf.keras.callbacks.BackupAndRestore(backup_dir=str(out/'training_backup'),save_freq='epoch',delete_checkpoint=False),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=2,min_lr=1e-5),tf.keras.callbacks.ModelCheckpoint(str(out/'best.weights.h5'),monitor='val_loss',save_best_only=True,save_weights_only=True),tf.keras.callbacks.CSVLogger(str(out/'training_history.csv'),append=True)]
 print('=== TRAJECTORY SCALE-10 — CONVERGENCE / RESUMABLE ===');print(f'train windows: {ntrain:,} | validation: {len(y):,} | epochs ceiling: {a.epochs}');print('decoder: 24 x 5 min | reads: 6/12/18/24 | future CGM: False');print('BackupAndRestore ON | EarlyStopping patience=5 | ReduceLROnPlateau patience=2')
 hist=model.fit(seq,validation_data=([xv,fv],targets),epochs=a.epochs,callbacks=cb,verbose=2)
 best=out/'best.weights.h5'
 if best.exists():model.load_weights(best)
 raw=model.predict([xv,fv],batch_size=a.batch_size,verbose=1);absolute=np.column_stack([raw[i].ravel() for i in range(4)]);delta=np.column_stack([raw[i+4].ravel() for i in range(4)]);recon=cur[:,None]+delta;final=.5*absolute+.5*recon
 ma=evaluate(y,absolute,cur);md=evaluate(y,recon,cur);mf=evaluate(y,final,cur);ma.to_csv(out/'absolute_head_metrics.csv',index=False);md.to_csv(out/'delta_reconstructed_metrics.csv',index=False);mf.to_csv(out/'trajectory_scale10_validation_metrics.csv',index=False);np.savez_compressed(out/'validation_predictions.npz',y_true=y,current_glucose=cur,absolute_head=absolute,delta_head=delta,delta_reconstructed=recon,y_pred=final)
 means={'mean_mard_4h':float(mf.mard.mean()),'mean_rmse_4h':float(mf.rmse.mean()),'mean_mae_4h':float(mf.mae.mean()),'mean_direction_accuracy_4h':float(mf.direction_accuracy.mean())};bestep=int(np.argmin(hist.history['val_loss'])+1) if hist.history.get('val_loss') else None
 report={'experiment':'trajectory-scale10-convergence','scientific_change':'single causal recurrent future trajectory decoder with reads at 30/60/90/120 min','training_protocol':'40ep ceiling; ES5; ReduceLR2; resumable','read_steps':READ_STEPS,'seed':SEED,'train_windows':int(ntrain),'validation_windows':int(len(y)),'epochs_ceiling':a.epochs,'best_val_loss_epoch_this_process':bestep,'future_cgm_input':False,**means};(out/'trajectory_scale10_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('\nTRAJECTORY SCALE-10 VALIDATION METRICS');print(mf.to_string(index=False));print('\n4H MEANS');print(f"MARD: {means['mean_mard_4h']:.6f}%");print(f"RMSE: {means['mean_rmse_4h']:.6f}");print(f"Direction: {means['mean_direction_accuracy_4h']*100:.4f}%");print(f'Best val_loss epoch in this process: {bestep} | epochs run in this process: {len(hist.history.get("loss",[]))}');print('\nComponent RMSEs');print(pd.DataFrame({'horizon':HORIZONS,'absolute_rmse':ma.rmse,'delta_reconstructed_rmse':md.rmse,'hybrid_rmse':mf.rmse}).to_string(index=False));print(f'\nArtifacts written to {out}')
if __name__=='__main__':main()
