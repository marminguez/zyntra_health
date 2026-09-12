"""Train V15.2 from V15 master: future insulin + future carbs only."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from ml.forecasting.model_v15_2 import build_v15_2_forecaster,HORIZONS
SEED=42
# v15_master channel order:
# 0 insulin,1 insulin_missing,2 basal,3 basal_missing,
# 4 bolus,5 bolus_missing,6 carbs,7 carbs_missing
V15_2_CHANNELS=(0,1,6,7)

def load(root,split):
    xs,fs,ys=[],[],[]
    for p in sorted((root/split).glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            xs.append(z['x'].astype(np.float32))
            master=z['future_known'].astype(np.float32)
            if master.shape[-1] < 8:
                raise ValueError(f'{p} does not look like v15_master: future_known shape={master.shape}')
            fs.append(master[:,:,V15_2_CHANNELS])
            ys.append(z['y'].astype(np.float32))
    if not xs:raise ValueError(f'No shards found for {split}')
    return np.concatenate(xs),np.concatenate(fs),np.concatenate(ys)
def dclass(d):return np.where(d>5,1,np.where(d<-5,-1,0))
def evaluate(y,p,cur):
    rows=[]
    for i,h in enumerate(HORIZONS):
        yt=y[:,i].astype(float);yp=p[:,i].astype(float);e=yp-yt;rows.append({'horizon_minutes':h,'n':len(yt),'mae':np.mean(np.abs(e)),'rmse':np.sqrt(np.mean(e**2)),'mard':np.mean(np.abs(e)/np.maximum(np.abs(yt),1e-6))*100,'direction_accuracy':np.mean(dclass(yt-cur)==dclass(yp-cur))})
    return pd.DataFrame(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='ml/data/v15_master');ap.add_argument('--v14-1-dir',default='ml/results/v14_1');ap.add_argument('--outdir',default='ml/results/v15_2');ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--batch-size',type=int,default=64);a=ap.parse_args()
    random.seed(SEED);np.random.seed(SEED);tf.random.set_seed(SEED);data,d1,out=Path(a.data_dir),Path(a.v14_1_dir),Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    xt,ft,yt=load(data,'train');xv,fv,yv=load(data,'validation');ct=xt[:,-1,0].copy();cv=xv[:,-1,0].astype(float);dt=yt-ct[:,None];dv=yv-cv[:,None]
    with np.load(d1/'normalization.npz') as z:mean,std=z['mean'],z['std']
    xt=((xt-mean)/std).astype(np.float32);xv=((xv-mean)/std).astype(np.float32)
    # ft/fv order after channel selection: insulin, insulin_missing, carbs, carbs_missing.
    # Reuse V14.1 train-only statistics: insulin index 9, carbs index 11.
    for arr in (ft,fv):
        ip=arr[:,:,1]<.5;cp=arr[:,:,3]<.5
        arr[:,:,0]=np.where(ip,(arr[:,:,0]-float(mean[9]))/float(std[9]),0.)
        arr[:,:,2]=np.where(cp,(arr[:,:,2]-float(mean[11]))/float(std[11]),0.)
    model=build_v15_2_forecaster(xt.shape[1],xt.shape[2]);tt=[yt[:,i] for i in range(4)]+[dt[:,i] for i in range(4)];vt=[yv[:,i] for i in range(4)]+[dv[:,i] for i in range(4)]
    cb=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=2,min_lr=1e-5),tf.keras.callbacks.ModelCheckpoint(str(out/'best_model.keras'),monitor='val_loss',save_best_only=True)]
    hist=model.fit([xt,ft],tt,validation_data=([xv,fv],vt),epochs=a.epochs,batch_size=a.batch_size,shuffle=True,callbacks=cb,verbose=2);raw=model.predict([xv,fv],batch_size=a.batch_size,verbose=1)
    pa=np.column_stack([raw[i].reshape(-1) for i in range(4)]).astype(float);pdlt=np.column_stack([raw[i+4].reshape(-1) for i in range(4)]).astype(float);pr=cv[:,None]+pdlt;pf=.5*pa+.5*pr
    ma=evaluate(yv,pa,cv);mr=evaluate(yv,pr,cv);mf=evaluate(yv,pf,cv);ma.to_csv(out/'absolute_head_metrics.csv',index=False);mr.to_csv(out/'delta_reconstructed_metrics.csv',index=False);mf.to_csv(out/'v15_2_validation_metrics.csv',index=False);pd.DataFrame(hist.history).to_csv(out/'training_history.csv',index=False);np.savez_compressed(out/'validation_predictions.npz',y_true=yv,current_glucose=cv,absolute_head=pa,delta_head=pdlt,delta_reconstructed=pr,y_pred=pf)
    report={'version':'v15.2','data_source':'v15_master exact frozen V15.1 anchors','resampled':False,'hypothesis':'adding known future carbs to future insulin further improves forecasting','controlled_change_vs_v15_1':'add future carbs + missingness only','future_features':['insulin','insulin_missing','carbs','carbs_missing'],'future_cgm_input':False,'fusion':'fixed 0.5 absolute + 0.5 reconstructed delta; not validation-tuned','normalization':'V14.1 train-only statistics','seed':SEED,'train_windows':int(len(xt)),'validation_windows':int(len(xv)),'test_parquet_used':False,'live_targets_used':False,'clinical_status':'conditional challenge/research model; not a causal clinical deployment model'};(out/'v15_2_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nV15.2 VALIDATION METRICS — FUTURE INSULIN + CARBS');print(mf.to_string(index=False));print('\nComponent RMSEs');print(pd.DataFrame({'horizon':HORIZONS,'absolute_rmse':ma.rmse,'delta_reconstructed_rmse':mr.rmse,'hybrid_rmse':mf.rmse}).to_string(index=False));print(f'\nArtifacts written to {out}')
if __name__=='__main__':main()
