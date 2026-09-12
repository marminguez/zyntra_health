"""Train V15.4: V15.3 inputs with horizon-specific future prefixes."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from ml.forecasting.model_v15_4 import build_v15_4_forecaster,HORIZONS,PREFIX_STEPS
SEED=42;CHANNELS=(2,3,4,5,6,7)
def load(root,split):
 xs,fs,ys=[],[],[]
 for p in sorted((root/split).glob('*.npz')):
  with np.load(p,allow_pickle=False) as z:x=z['x'].astype(np.float32);f=z['future_known'].astype(np.float32);y=z['y'].astype(np.float32)
  if f.shape[-1]<8:raise ValueError(f'{p}: expected v15_master, got {f.shape}')
  xs.append(x);fs.append(f[:,:,CHANNELS]);ys.append(y)
 if not xs:raise ValueError(f'No shards for {split}')
 return np.concatenate(xs),np.concatenate(fs),np.concatenate(ys)
def dc(d):return np.where(d>5,1,np.where(d<-5,-1,0))
def ev(y,p,c):
 rows=[]
 for i,h in enumerate(HORIZONS):
  yt=y[:,i].astype(float);yp=p[:,i].astype(float);e=yp-yt;rows.append({'horizon_minutes':h,'n':len(yt),'mae':np.mean(abs(e)),'rmse':np.sqrt(np.mean(e**2)),'mard':np.mean(abs(e)/np.maximum(abs(yt),1e-6))*100,'direction_accuracy':np.mean(dc(yt-c)==dc(yp-c))})
 return pd.DataFrame(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='ml/data/v15_master');ap.add_argument('--v14-1-dir',default='ml/results/v14_1');ap.add_argument('--outdir',default='ml/results/v15_4');ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--batch-size',type=int,default=64);a=ap.parse_args();random.seed(SEED);np.random.seed(SEED);tf.random.set_seed(SEED)
 data,d1,out=Path(a.data_dir),Path(a.v14_1_dir),Path(a.outdir);out.mkdir(parents=True,exist_ok=True);xt,ft,yt=load(data,'train');xv,fv,yv=load(data,'validation');ct=xt[:,-1,0].copy();cv=xv[:,-1,0].astype(float);dt=yt-ct[:,None];dv=yv-cv[:,None]
 with np.load(d1/'normalization.npz') as z:mean,std=z['mean'],z['std']
 xt=((xt-mean)/std).astype(np.float32);xv=((xv-mean)/std).astype(np.float32)
 for ar in (ft,fv):
  for vi,mi,hi in ((0,1,5),(2,3,7),(4,5,11)):
   present=ar[:,:,mi]<.5;ar[:,:,vi]=np.where(present,(ar[:,:,vi]-float(mean[hi]))/float(std[hi]),0.)
 m=build_v15_4_forecaster(xt.shape[1],xt.shape[2]);tt=[yt[:,i] for i in range(4)]+[dt[:,i] for i in range(4)];vt=[yv[:,i] for i in range(4)]+[dv[:,i] for i in range(4)];cb=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=2,min_lr=1e-5),tf.keras.callbacks.ModelCheckpoint(str(out/'best_model.keras'),monitor='val_loss',save_best_only=True)]
 hist=m.fit([xt,ft],tt,validation_data=([xv,fv],vt),epochs=a.epochs,batch_size=a.batch_size,shuffle=True,callbacks=cb,verbose=2);raw=m.predict([xv,fv],batch_size=a.batch_size,verbose=1);pa=np.column_stack([raw[i].ravel() for i in range(4)]);dd=np.column_stack([raw[i+4].ravel() for i in range(4)]);pr=cv[:,None]+dd;pf=.5*pa+.5*pr;ma,mr,mf=ev(yv,pa,cv),ev(yv,pr,cv),ev(yv,pf,cv)
 ma.to_csv(out/'absolute_head_metrics.csv',index=False);mr.to_csv(out/'delta_reconstructed_metrics.csv',index=False);mf.to_csv(out/'v15_4_validation_metrics.csv',index=False);pd.DataFrame(hist.history).to_csv(out/'training_history.csv',index=False);np.savez_compressed(out/'validation_predictions.npz',y_true=yv,current_glucose=cv,absolute_head=pa,delta_head=dd,delta_reconstructed=pr,y_pred=pf)
 report={'version':'v15.4','data_source':'v15_master exact frozen V15.1 anchors','resampled':False,'hypothesis':'restricting future-known inputs to each target horizon improves cleanliness and may improve forecasting','controlled_change_vs_v15_3':'same basal+bolus+carbs inputs, but each horizon only sees future inputs through its target time','future_features':['basal','basal_missing','bolus','bolus_missing','carbs','carbs_missing'],'future_prefix_steps':PREFIX_STEPS,'future_cgm_input':False,'fusion':'fixed 0.5 absolute + 0.5 reconstructed delta','normalization':'V14.1 train-only statistics','seed':SEED,'train_windows':int(len(xt)),'validation_windows':int(len(xv)),'test_parquet_used':False,'live_targets_used':False,'clinical_status':'conditional challenge/research model; not causal clinical deployment'};(out/'v15_4_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('\nV15.4 VALIDATION METRICS — HORIZON-MASKED FUTURE INPUTS');print(mf.to_string(index=False));print('\nComponent RMSEs');print(pd.DataFrame({'horizon':HORIZONS,'absolute_rmse':ma.rmse,'delta_reconstructed_rmse':mr.rmse,'hybrid_rmse':mf.rmse}).to_string(index=False));print(f'\nArtifacts written to {out}')
if __name__=='__main__':main()
