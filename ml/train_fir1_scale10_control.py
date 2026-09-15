"""Controlled Scale-10 FIR-1 baseline for the volatility experiment."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,tensorflow as tf
from ml.forecasting.future_intervention_features import build_future_intervention_summary
from ml.forecasting.model_fir1 import HORIZONS,build_fir1_forecaster
SEED=42;CHANNELS=(2,3,4,5,6,7)
def loadval(root):
 X,F,Y=[],[],[]
 for p in sorted((root/'validation').glob('*.npz')):
  with np.load(p,allow_pickle=False) as z:X.append(z['x'].astype(np.float32));F.append(z['future_known'].astype(np.float32)[:,:,CHANNELS]);Y.append(z['y'].astype(np.float32))
 if not X:raise ValueError('No validation shards')
 return np.concatenate(X),np.concatenate(F),np.concatenate(Y)
def manifest(root):
 r=[]
 for p in sorted((root/'train').glob('*.npz')):
  with np.load(p,allow_pickle=False) as z:n=len(z['y'])
  if n:r.append((p,n))
 if not r:raise ValueError('No train shards')
 return r
def nf(f,mean,std):
 f=f.copy()
 for vi,mi,hi in ((0,1,5),(2,3,7),(4,5,11)):
  present=f[:,:,mi]<.5;f[:,:,vi]=np.where(present,(f[:,:,vi]-mean[hi])/std[hi],0)
 return f.astype(np.float32)
class Seq(tf.keras.utils.Sequence):
 def __init__(self,it,mean,std,bs=64):super().__init__();self.it=it;self.mean=mean;self.std=std;self.bs=bs;self.ep=0;self.rebuild()
 def rebuild(self):
  r=np.random.default_rng(SEED+self.ep);self.b=[]
  for mi in r.permutation(len(self.it)):
   _,n=self.it[int(mi)];ix=r.permutation(n)
   for s in range(0,n,self.bs):self.b.append((int(mi),ix[s:s+self.bs]))
  r.shuffle(self.b)
 def __len__(self):return len(self.b)
 def __getitem__(self,i):
  mi,rows=self.b[i];p,_=self.it[mi]
  with np.load(p,allow_pickle=False) as z:x=z['x'][rows].astype(np.float32);fr=z['future_known'][rows].astype(np.float32)[:,:,CHANNELS];y=z['y'][rows].astype(np.float32)
  cur=x[:,-1,0].copy();summ=build_future_intervention_summary(fr,basal_mean=float(self.mean[5]),basal_std=float(self.std[5]));xn=((x-self.mean)/self.std).astype(np.float32);targets=tuple(y[:,j] for j in range(4))+tuple((y-cur[:,None])[:,j] for j in range(4));return (xn,nf(fr,self.mean,self.std),summ),targets
 def on_epoch_end(self):self.ep+=1;self.rebuild()
def dc(d):return np.where(d>5,1,np.where(d<-5,-1,0))
def evalm(y,p,c):
 r=[]
 for j,h in enumerate(HORIZONS):
  e=p[:,j]-y[:,j];r.append({'horizon_minutes':h,'mae':float(np.mean(abs(e))),'rmse':float(np.sqrt(np.mean(e**2))),'mard':float(np.mean(abs(e)/np.maximum(abs(y[:,j]),1e-6))*100),'direction_accuracy':float(np.mean(dc(y[:,j]-c)==dc(p[:,j]-c)))})
 return pd.DataFrame(r)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='ml/data/v15_4_scale10');ap.add_argument('--v14-1-dir',default='ml/results/v14_1');ap.add_argument('--outdir',default='ml/results/fir1_scale10_control_20ep');ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--batch-size',type=int,default=64);a=ap.parse_args();random.seed(SEED);np.random.seed(SEED);tf.random.set_seed(SEED)
 root=Path(a.data_dir);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);it=manifest(root);x,fr,y=loadval(root);cur=x[:,-1,0].astype(float)
 with np.load(Path(a.v14_1_dir)/'normalization.npz') as z:mean=z['mean'].astype(np.float32);std=z['std'].astype(np.float32)
 sv=build_future_intervention_summary(fr,basal_mean=float(mean[5]),basal_std=float(std[5]));xv=((x-mean)/std).astype(np.float32);fv=nf(fr,mean,std);tar=tuple(y[:,j] for j in range(4))+tuple((y-cur[:,None])[:,j] for j in range(4));seq=Seq(it,mean,std,a.batch_size);m=build_fir1_forecaster(x.shape[1],x.shape[2])
 cb=[tf.keras.callbacks.BackupAndRestore(backup_dir=str(out/'training_backup'),delete_checkpoint=False),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=2,min_lr=1e-5),tf.keras.callbacks.ModelCheckpoint(str(out/'best.weights.h5'),monitor='val_loss',save_best_only=True,save_weights_only=True),tf.keras.callbacks.CSVLogger(str(out/'training_history.csv'),append=True)]
 print('=== FIR-1 SCALE-10 CONTROL — FULL CONVERGENCE / RESUMABLE ===');print(f'train windows: {sum(n for _,n in it):,} | validation: {len(y):,} | epochs max: {a.epochs}');print('BackupAndRestore ON | EarlyStopping patience=4 | ReduceLROnPlateau patience=2 | future CGM: False')
 hist=m.fit(seq,validation_data=([xv,fv,sv],tar),epochs=a.epochs,callbacks=cb,verbose=2);m.load_weights(out/'best.weights.h5');raw=m.predict([xv,fv,sv],batch_size=a.batch_size,verbose=1);ab=np.column_stack([raw[i].ravel() for i in range(4)]);dr=cur[:,None]+np.column_stack([raw[i+4].ravel() for i in range(4)]);pred=.5*ab+.5*dr;df=evalm(y,pred,cur);df.to_csv(out/'validation_metrics.csv',index=False);np.savez_compressed(out/'validation_predictions.npz',y_true=y,current_glucose=cur,y_pred=pred)
 base={'mard':14.5988565,'rmse':30.88723775,'direction':.6916525};now={'mard':float(df.mard.mean()),'rmse':float(df.rmse.mean()),'direction':float(df.direction_accuracy.mean())};relm=100*(now['mard']/base['mard']-1);relr=100*(now['rmse']/base['rmse']-1);dpp=100*(now['direction']-base['direction']);best_epoch=int(np.argmin(hist.history['val_loss'])+1) if hist.history.get('val_loss') else None
 print('\nVALIDATION');print(df.to_string(index=False));print('\n4H VS SCALE10');print(f"MARD {base['mard']:.6f} -> {now['mard']:.6f} ({relm:+.2f}% relative)");print(f"RMSE {base['rmse']:.6f} -> {now['rmse']:.6f} ({relr:+.2f}% relative)");print(f"Direction {100*base['direction']:.4f}% -> {100*now['direction']:.4f}% ({dpp:+.3f} pp)");print(f'Best val_loss epoch in this process: {best_epoch} | epochs run in this process: {len(hist.history.get("loss",[]))}')
 (out/'report.json').write_text(json.dumps({'experiment':'FIR-1 Scale10 full convergence control','epochs_max':a.epochs,'best_val_loss_epoch_this_process':best_epoch,'early_stopping_patience':4,'reduce_lr_patience':2,'resumable':True,'metrics':now,'relative_mard_pct':relm,'relative_rmse_pct':relr,'direction_pp':dpp},indent=2))
if __name__=='__main__':main()
