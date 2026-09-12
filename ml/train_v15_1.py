"""Train Zyntra V15.1: V14.5 hybrid forecast + known future insulin."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from ml.forecasting.model_v15_1 import build_v15_1_forecaster, HORIZONS

SEED=42


def load(root,split):
    xs,fs,ys=[],[],[]
    for p in sorted((root/split).glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            xs.append(z['x'].astype(np.float32)); fs.append(z['future_insulin'].astype(np.float32)); ys.append(z['y'].astype(np.float32))
    if not xs: raise ValueError(f'No shards found for {split}')
    return np.concatenate(xs),np.concatenate(fs),np.concatenate(ys)


def dclass(d): return np.where(d>5,1,np.where(d<-5,-1,0))


def evaluate(y,p,cur):
    rows=[]
    for i,h in enumerate(HORIZONS):
        yt=y[:,i].astype(float); yp=p[:,i].astype(float); e=yp-yt
        rows.append({'horizon_minutes':h,'n':len(yt),'mae':np.mean(np.abs(e)),
                     'rmse':np.sqrt(np.mean(e**2)),
                     'mard':np.mean(np.abs(e)/np.maximum(np.abs(yt),1e-6))*100,
                     'direction_accuracy':np.mean(dclass(yt-cur)==dclass(yp-cur))})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='ml/data/v15_1'); ap.add_argument('--v14-1-dir',default='ml/results/v14_1'); ap.add_argument('--outdir',default='ml/results/v15_1'); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=64); a=ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    data,d1,out=Path(a.data_dir),Path(a.v14_1_dir),Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    xt,ft,yt=load(data,'train'); xv,fv,yv=load(data,'validation')
    ct=xt[:,-1,0].copy(); cv=xv[:,-1,0].astype(float); dt=yt-ct[:,None]; dv=yv-cv[:,None]

    with np.load(d1/'normalization.npz') as z: mean,std=z['mean'],z['std']
    xt=((xt-mean)/std).astype(np.float32); xv=((xv-mean)/std).astype(np.float32)

    # Future insulin uses the V14.1 train-only insulin normalization (feature index 9).
    insulin_mean=float(mean[9]); insulin_std=float(std[9])
    for arr in (ft,fv):
        present=arr[:,:,1] < 0.5
        arr[:,:,0]=np.where(present,(arr[:,:,0]-insulin_mean)/insulin_std,0.0)
    model=build_v15_1_forecaster(xt.shape[1],xt.shape[2])
    train_targets=[yt[:,i] for i in range(4)]+[dt[:,i] for i in range(4)]
    val_targets=[yv[:,i] for i in range(4)]+[dv[:,i] for i in range(4)]
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=2,min_lr=1e-5),
               tf.keras.callbacks.ModelCheckpoint(str(out/'best_model.keras'),monitor='val_loss',save_best_only=True)]
    hist=model.fit([xt,ft],train_targets,validation_data=([xv,fv],val_targets),epochs=a.epochs,batch_size=a.batch_size,shuffle=True,callbacks=callbacks,verbose=2)
    raw=model.predict([xv,fv],batch_size=a.batch_size,verbose=1)
    pa=np.column_stack([raw[i].reshape(-1) for i in range(4)]).astype(float)
    pdlt=np.column_stack([raw[i+4].reshape(-1) for i in range(4)]).astype(float)
    pr=cv[:,None]+pdlt; pf=.5*pa+.5*pr
    ma=evaluate(yv,pa,cv); mr=evaluate(yv,pr,cv); mf=evaluate(yv,pf,cv)
    ma.to_csv(out/'absolute_head_metrics.csv',index=False); mr.to_csv(out/'delta_reconstructed_metrics.csv',index=False); mf.to_csv(out/'v15_1_validation_metrics.csv',index=False)
    pd.DataFrame(hist.history).to_csv(out/'training_history.csv',index=False)
    np.savez_compressed(out/'validation_predictions.npz',y_true=yv,current_glucose=cv,absolute_head=pa,delta_head=pdlt,delta_reconstructed=pr,y_pred=pf)
    report={'version':'v15.1','hypothesis':'known future insulin t+5..t+120 improves multi-horizon forecasting over V14.5','controlled_change':'future insulin encoder only','future_features':['insulin','insulin_missing'],'future_cgm_input':False,'fusion':'fixed 0.5 absolute + 0.5 reconstructed delta; not validation-tuned','normalization':'V14.1 train-only historical statistics; future insulin uses V14.1 insulin mean/std','seed':SEED,'train_windows':int(len(xt)),'validation_windows':int(len(xv)),'test_parquet_used':False,'live_targets_used':False,'clinical_status':'conditional challenge/research model; not a causal clinical deployment model'}
    (out/'v15_1_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nV15.1 VALIDATION METRICS — FUTURE INSULIN'); print(mf.to_string(index=False))
    print('\nComponent RMSEs'); print(pd.DataFrame({'horizon':HORIZONS,'absolute_rmse':ma.rmse,'delta_reconstructed_rmse':mr.rmse,'hybrid_rmse':mf.rmse}).to_string(index=False)); print(f'\nArtifacts written to {out}')

if __name__=='__main__': main()
