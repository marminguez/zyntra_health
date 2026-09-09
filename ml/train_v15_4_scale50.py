"""Train V15.4 Scale-50 with frozen validation and epoch-boundary resume.

Controlled scientific change vs frozen V15.4: train cap 24 -> 1200 windows/patient.
Architecture, normalization, targets, future inputs, losses and 50/50 fusion are V15.4.
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,tensorflow as tf
from ml.forecasting.model_v15_4 import HORIZONS,PREFIX_STEPS,build_v15_4_forecaster
SEED=42;CHANNELS=(2,3,4,5,6,7)
def load_validation(root):
 xs,fs,ys=[],[],[]
 for p in sorted((root/'validation').glob('*.npz')):
  with np.load(p,allow_pickle=False) as z:xs.append(z['x'].astype(np.float32));fs.append(z['future_known'].astype(np.float32)[:,:,CHANNELS]);ys.append(z['y'].astype(np.float32))
 if not xs:raise ValueError('No validation shards')
 return np.concatenate(xs),np.concatenate(fs),np.concatenate(ys)
def manifest(root):
 a=[]
 for p in sorted((root/'train').glob('*.npz')):
  with np.load(p,allow_pickle=False) as z:n=len(z['y'])
  if n:a.append((p,n))
 if not a:raise ValueError('No train shards')
 return a
def norm_future(f,mean,std):
 f=f.copy()
 for vi,mi,hi in ((0,1,5),(2,3,7),(4,5,11)):
  present=f[:,:,mi]<.5;f[:,:,vi]=np.where(present,(f[:,:,vi]-float(mean[hi]))/float(std[hi]),0.)
 return f.astype(np.float32)
class ShardSequence(tf.keras.utils.Sequence):
 def __init__(self,m,mean,std,batch_size=64,seed=SEED):
  super().__init__();self.m=m;self.mean=mean.astype(np.float32);self.std=std.astype(np.float32);self.bs=batch_size;self.seed=seed;self.epoch=0;self._rebuild()
 def _rebuild(self):
  rng=np.random.default_rng(self.seed+self.epoch);b=[]
  for mi in rng.permutation(len(self.m)):
   _,n=self.m[int(mi)];idx=rng.permutation(n)
   for s in range(0,n,self.bs):b.append((int(mi),idx[s:s+self.bs]))
  rng.shuffle(b);self.b=b
 def __len__(self):return len(self.b)
 def __getitem__(self,i):
  mi,rows=self.b[i];p,_=self.m[mi]
  with np.load(p,allow_pickle=False) as z:x=z['x'][rows].astype(np.float32);f=z['future_known'][rows].astype(np.float32)[:,:,CHANNELS];y=z['y'][rows].astype(np.float32)
  cur=x[:,-1,0].copy();delta=y-cur[:,None];x=((x-self.mean)/self.std).astype(np.float32);f=norm_future(f,self.mean,self.std);targets=tuple(y[:,j] for j in range(4))+tuple(delta[:,j] for j in range(4));return (x,f),targets
 def on_epoch_end(self):self.epoch+=1;self._rebuild()
def dc(d):return np.where(d>5,1,np.where(d<-5,-1,0))
def evaluate(y,p,current):
 rows=[]
 for i,h in enumerate(HORIZONS):
  yt=y[:,i].astype(float);yp=p[:,i].astype(float);e=yp-yt;rows.append({'horizon_minutes':h,'n':len(yt),'mae':float(np.mean(abs(e))),'rmse':float(np.sqrt(np.mean(e**2))),'mard':float(np.mean(abs(e)/np.maximum(abs(yt),1e-6))*100),'direction_accuracy':float(np.mean(dc(yt-current)==dc(yp-current)))})
 return pd.DataFrame(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='ml/data/v15_4_scale50');ap.add_argument('--v14-1-dir',default='ml/results/v14_1');ap.add_argument('--outdir',default='ml/results/v15_4_scale50');ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--batch-size',type=int,default=64);a=ap.parse_args();random.seed(SEED);np.random.seed(SEED);tf.random.set_seed(SEED);data=Path(a.data_dir);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);backup=out/'training_backup';m=manifest(data);nw=sum(n for _,n in m);xv,fv,yv=load_validation(data);cur=xv[:,-1,0].astype(float);dv=yv-cur[:,None]
 with np.load(Path(a.v14_1_dir)/'normalization.npz') as z:mean=z['mean'].astype(np.float32);std=z['std'].astype(np.float32)
 xv=((xv-mean)/std).astype(np.float32);fv=norm_future(fv,mean,std);vt=tuple(yv[:,i] for i in range(4))+tuple(dv[:,i] for i in range(4));seq=ShardSequence(m,mean,std,a.batch_size);model=build_v15_4_forecaster(xv.shape[1],xv.shape[2]);callbacks=[tf.keras.callbacks.BackupAndRestore(backup_dir=str(backup),save_freq='epoch',delete_checkpoint=False),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=2,min_lr=1e-5),tf.keras.callbacks.ModelCheckpoint(str(out/'best.weights.h5'),monitor='val_loss',save_best_only=True,save_weights_only=True),tf.keras.callbacks.CSVLogger(str(out/'training_history.csv'),append=True)]
 print('=== V15.4 SCALE-50 TRAINING ===');print(f'train shards: {len(m):,}');print(f'train windows: {nw:,}');print(f'validation windows: {len(yv):,}');print(f'batch size: {a.batch_size}');print(f'steps/epoch: {len(seq):,}');print(f'backup dir: {backup.resolve()}')
 if backup.exists() and any(backup.iterdir()):print('Resume checkpoint detected: Keras will restore training state automatically.')
 model.fit(x=seq,validation_data=([xv,fv],vt),epochs=a.epochs,callbacks=callbacks,verbose=2)
 best=out/'best.weights.h5'
 if best.exists():model.load_weights(best)
 raw=model.predict([xv,fv],batch_size=a.batch_size,verbose=1);absolute=np.column_stack([raw[i].ravel() for i in range(4)]);dh=np.column_stack([raw[i+4].ravel() for i in range(4)]);recon=cur[:,None]+dh;final=.5*absolute+.5*recon;ma=evaluate(yv,absolute,cur);mr=evaluate(yv,recon,cur);mf=evaluate(yv,final,cur);ma.to_csv(out/'absolute_head_metrics.csv',index=False);mr.to_csv(out/'delta_reconstructed_metrics.csv',index=False);mf.to_csv(out/'v15_4_scale50_validation_metrics.csv',index=False);np.savez_compressed(out/'validation_predictions.npz',y_true=yv,current_glucose=cur,absolute_head=absolute,delta_head=dh,delta_reconstructed=recon,y_pred=final);report={'experiment':'v15.4-scale50','scientific_change_vs_v15_4':'training cap per patient 24 -> 1200','architecture':'exact V15.4 model builder','future_features':['basal','basal_missing','bolus','bolus_missing','carbs','carbs_missing'],'future_prefix_steps':PREFIX_STEPS,'future_cgm_input':False,'fusion':'fixed 0.5 absolute + 0.5 reconstructed delta','normalization':'V14.1 frozen train-only statistics','seed':SEED,'train_windows':int(nw),'validation_windows':int(len(yv)),'batch_size':a.batch_size,'epochs_requested':a.epochs,'resumable':True,'resume_mechanism':'Keras BackupAndRestore at epoch boundaries','best_checkpoint':'best.weights.h5','test_parquet_used':False,'live_targets_used':False};(out/'v15_4_scale50_report.json').write_text(json.dumps(report,indent=2));print('\nV15.4 SCALE-50 VALIDATION METRICS');print(mf.to_string(index=False));print('\nComponent RMSEs');print(pd.DataFrame({'horizon':HORIZONS,'absolute_rmse':ma.rmse,'delta_reconstructed_rmse':mr.rmse,'hybrid_rmse':mf.rmse}).to_string(index=False));print(f'\nArtifacts written to {out}')
if __name__=='__main__':main()
