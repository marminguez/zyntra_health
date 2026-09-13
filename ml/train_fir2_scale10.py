"""Fast FIR-2 screen on frozen V15.4 Scale-10 data (default: 5 epochs)."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from ml.forecasting.future_intervention_features_v2 import SUMMARY_FEATURE_NAMES, build_future_intervention_summary_v2
from ml.forecasting.model_fir2 import HORIZONS, PREFIX_STEPS, build_fir2_forecaster

SEED=42
CHANNELS=(2,3,4,5,6,7)

def load_validation(root):
 xs,fs,ys=[],[],[]
 for p in sorted((root/'validation').glob('*.npz')):
  with np.load(p,allow_pickle=False) as z:
   xs.append(z['x'].astype(np.float32));fs.append(z['future_known'].astype(np.float32)[:,:,CHANNELS]);ys.append(z['y'].astype(np.float32))
 if not xs: raise ValueError('No validation shards')
 return np.concatenate(xs),np.concatenate(fs),np.concatenate(ys)

def manifest(root):
 items=[]
 for p in sorted((root/'train').glob('*.npz')):
  with np.load(p,allow_pickle=False) as z:n=len(z['y'])
  if n:items.append((p,n))
 if not items:raise ValueError('No train shards')
 return items

def norm_future(f,mean,std):
 f=f.copy()
 for vi,mi,hi in ((0,1,5),(2,3,7),(4,5,11)):
  present=f[:,:,mi]<.5
  f[:,:,vi]=np.where(present,(f[:,:,vi]-float(mean[hi]))/float(std[hi]),0.)
 return f.astype(np.float32)

class ShardSequence(tf.keras.utils.Sequence):
 def __init__(self,items,mean,std,batch_size=64,seed=SEED):
  super().__init__();self.items=items;self.mean=mean.astype(np.float32);self.std=std.astype(np.float32);self.bs=batch_size;self.seed=seed;self.epoch=0;self._rebuild()
 def _rebuild(self):
  rng=np.random.default_rng(self.seed+self.epoch);b=[]
  for mi in rng.permutation(len(self.items)):
   _,n=self.items[int(mi)];idx=rng.permutation(n)
   for start in range(0,n,self.bs):b.append((int(mi),idx[start:start+self.bs]))
  rng.shuffle(b);self.batches=b
 def __len__(self):return len(self.batches)
 def __getitem__(self,i):
  mi,rows=self.batches[i];p,_=self.items[mi]
  with np.load(p,allow_pickle=False) as z:
   x=z['x'][rows].astype(np.float32);fr=z['future_known'][rows].astype(np.float32)[:,:,CHANNELS];y=z['y'][rows].astype(np.float32)
  current=x[:,-1,0].copy();delta=y-current[:,None]
  s=build_future_intervention_summary_v2(fr,float(self.mean[5]),float(self.std[5]))
  x=((x-self.mean)/self.std).astype(np.float32);f=norm_future(fr,self.mean,self.std)
  targets=tuple(y[:,j] for j in range(4))+tuple(delta[:,j] for j in range(4))
  return (x,f,s),targets
 def on_epoch_end(self):self.epoch+=1;self._rebuild()

def direction_class(d):return np.where(d>5,1,np.where(d<-5,-1,0))
def evaluate(y,p,current):
 rows=[]
 for i,h in enumerate(HORIZONS):
  yt=y[:,i].astype(float);yp=p[:,i].astype(float);e=yp-yt
  rows.append({'horizon_minutes':h,'n':len(yt),'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e**2))),'mard':float(np.mean(np.abs(e)/np.maximum(np.abs(yt),1e-6))*100),'direction_accuracy':float(np.mean(direction_class(yt-current)==direction_class(yp-current)))})
 return pd.DataFrame(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='ml/data/v15_4_scale10');ap.add_argument('--v14-1-dir',default='ml/results/v14_1');ap.add_argument('--outdir',default='ml/results/fir2_scale10');ap.add_argument('--epochs',type=int,default=5);ap.add_argument('--batch-size',type=int,default=64);a=ap.parse_args()
 random.seed(SEED);np.random.seed(SEED);tf.random.set_seed(SEED)
 data=Path(a.data_dir);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);items=manifest(data);ntrain=sum(n for _,n in items)
 xr,fr,yv=load_validation(data);current=xr[:,-1,0].astype(float);dv=yv-current[:,None]
 with np.load(Path(a.v14_1_dir)/'normalization.npz') as z:mean=z['mean'].astype(np.float32);std=z['std'].astype(np.float32)
 sv=build_future_intervention_summary_v2(fr,float(mean[5]),float(std[5]));xv=((xr-mean)/std).astype(np.float32);fv=norm_future(fr,mean,std);vt=tuple(yv[:,i] for i in range(4))+tuple(dv[:,i] for i in range(4))
 seq=ShardSequence(items,mean,std,a.batch_size);model=build_fir2_forecaster(xv.shape[1],xv.shape[2])
 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=1,min_lr=1e-5),tf.keras.callbacks.ModelCheckpoint(str(out/'best.weights.h5'),monitor='val_loss',save_best_only=True,save_weights_only=True),tf.keras.callbacks.CSVLogger(str(out/'training_history.csv'))]
 print('=== FIR-2 SCALE-10 FAST SCREEN ===');print(f'train windows:      {ntrain:,}');print(f'validation windows: {len(yv):,}');print(f'summary features:   {len(SUMMARY_FEATURE_NAMES)} per horizon');print('future CGM input:   False');print(f'epochs max:         {a.epochs}');print(f'steps/epoch:        {len(seq):,}')
 model.fit(seq,validation_data=([xv,fv,sv],vt),epochs=a.epochs,callbacks=callbacks,verbose=2)
 best=out/'best.weights.h5'
 if best.exists():model.load_weights(best)
 raw=model.predict([xv,fv,sv],batch_size=a.batch_size,verbose=1);absolute=np.column_stack([raw[i].ravel() for i in range(4)]);delta=np.column_stack([raw[i+4].ravel() for i in range(4)]);recon=current[:,None]+delta;final=.5*absolute+.5*recon
 mf=evaluate(yv,final,current);mf.to_csv(out/'fir2_scale10_validation_metrics.csv',index=False);np.savez_compressed(out/'validation_predictions.npz',y_true=yv,current_glucose=current,absolute_head=absolute,delta_head=delta,delta_reconstructed=recon,y_pred=final)
 report={'experiment':'FIR-2 Scale-10 fast screen','baseline':'V15.4 Scale-10','controlled_change':'FIR-1 summaries plus 8 fixed action/timing features','summary_features':list(SUMMARY_FEATURE_NAMES),'future_prefix_steps':PREFIX_STEPS,'future_cgm_input':False,'train_windows':int(ntrain),'validation_windows':int(len(yv)),'epochs_max':a.epochs,'seed':SEED,'test_parquet_used':False,'live_targets_used':False};(out/'fir2_scale10_report.json').write_text(json.dumps(report,indent=2))
 print('\nFIR-2 SCALE-10 VALIDATION METRICS');print(mf.to_string(index=False));print('\n4H MEANS');print(f"MARD: {mf.mard.mean():.6f}%");print(f"RMSE: {mf.rmse.mean():.6f}");print(f"Direction: {mf.direction_accuracy.mean()*100:.4f}%");print(f'\nArtifacts written to {out}')
if __name__=='__main__':main()
