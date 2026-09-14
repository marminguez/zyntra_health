"""Large-rise residual atlas for frozen FIR-1 Scale-50 validation predictions.

Diagnostic only. Within true large-rise cases (target-current >=40 mg/dL), quantify
where FIR-1 underpredicts magnitude and how that residual relates to anchor-available
trend/event context. Future targets define the diagnostic cohort only, never features.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS=(30,60,90,120)

def load_validation(d):
    xs,fs,ys=[],[],[]
    for p in sorted((Path(d)/"validation").glob("*.npz")):
        with np.load(p,allow_pickle=False) as z:
            xs.append(z["x"].astype(np.float32)); fs.append(z["future_known"].astype(np.float32)); ys.append(z["y"].astype(np.float32))
    if not xs: raise ValueError("No validation shards")
    return np.concatenate(xs),np.concatenate(fs),np.concatenate(ys)

def binned(name,values,bins,labels,residual,h):
    cat=pd.cut(values,bins=bins,labels=labels,include_lowest=True,right=False)
    rows=[]
    for lab in labels:
        m=np.asarray(cat==lab)
        if not m.any(): continue
        r=residual[m]
        rows.append({"horizon_minutes":h,"factor":name,"segment":str(lab),"n":int(m.sum()),"share_pct":100*float(m.mean()),"mean_residual":float(r.mean()),"median_residual":float(np.median(r)),"underprediction_rate":float(np.mean(r>0)),"mae_residual":float(np.mean(np.abs(r))),"rmse_residual":float(np.sqrt(np.mean(r*r)))})
    return rows

def binary(name,mask,residual,h):
    rows=[]
    for lab,m in (("yes",mask),("no",~mask)):
        r=residual[m]
        rows.append({"horizon_minutes":h,"factor":name,"segment":lab,"n":int(m.sum()),"share_pct":100*float(m.mean()),"mean_residual":float(r.mean()),"median_residual":float(np.median(r)),"underprediction_rate":float(np.mean(r>0)),"mae_residual":float(np.mean(np.abs(r))),"rmse_residual":float(np.sqrt(np.mean(r*r)))})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="ml/data/v15_4_scale50"); ap.add_argument("--predictions",default="ml/results/fir1_scale50/validation_predictions.npz"); ap.add_argument("--outdir",default="ml/results/large_rise_residual_atlas"); a=ap.parse_args()
    x,f,ydata=load_validation(a.data_dir)
    with np.load(a.predictions,allow_pickle=False) as z:
        y=z["y_true"].astype(float); current=z["current_glucose"].astype(float); pred=z["y_pred"].astype(float)
    if not np.allclose(y,ydata,atol=1e-4,rtol=0) or not np.allclose(current,x[:,-1,0],atol=1e-4,rtol=0): raise ValueError("alignment failed")
    r=min(12,x.shape[1]); bol=np.where(x[:,-r:,8]<.5,x[:,-r:,7],0).sum(1); car=np.where(x[:,-r:,12]<.5,x[:,-r:,11],0).sum(1); ins=np.where(x[:,-r:,10]<.5,x[:,-r:,9],0).sum(1); d5=x[:,-1,1]; d15=x[:,-1,2]; d30=x[:,-1,3]; accel=x[:,-1,4]
    rows=[]; summary=[]
    for j,h in enumerate(HORIZONS):
        rise=y[:,j]-current; mask=rise>=40; res=(y[:,j]-pred[:,j])[mask]
        summary.append({"horizon_minutes":h,"n":int(mask.sum()),"prevalence_pct":100*float(mask.mean()),"actual_rise_mean":float(rise[mask].mean()),"predicted_rise_mean":float((pred[:,j]-current)[mask].mean()),"mean_residual":float(res.mean()),"median_residual":float(np.median(res)),"underprediction_rate":float(np.mean(res>0)),"rmse":float(np.sqrt(np.mean(res*res)))})
        c=current[mask]; q30=d30[mask]; q5=d5[mask]; qa=accel[mask]; cb=bol[mask]; cc=car[mask]; ci=ins[mask]
        rows += binned("current_glucose",c,[-np.inf,70,120,180,250,np.inf],["<70","70-120","120-180","180-250",">=250"],res,h)
        rows += binned("trend_30m",q30,[-np.inf,0,15,30,50,np.inf],["<0","0-15","15-30","30-50",">=50"],res,h)
        rows += binned("trend_5m",q5,[-np.inf,0,5,10,20,np.inf],["<0","0-5","5-10","10-20",">=20"],res,h)
        rows += binned("acceleration",qa,[-np.inf,-5,0,5,10,np.inf],["<-5","-5-0","0-5","5-10",">=10"],res,h)
        rows += binary("bolus_last60",cb>0,res,h); rows += binary("carbs_last60",cc>0,res,h); rows += binary("insulin_last60",ci>0,res,h)
        rows += binned("carbs_amount_last60",cc,[-np.inf,1e-9,20,50,100,np.inf],["0","0-20","20-50","50-100",">=100"],res,h)
        rows += binned("bolus_amount_last60",cb,[-np.inf,1e-9,2,5,10,np.inf],["0","0-2","2-5","5-10",">=10"],res,h)
        # FIR-1's own prospective expected rise is legal at inference and can reveal magnitude compression.
        pr=(pred[:,j]-current)[mask]
        rows += binned("fir1_predicted_rise",pr,[-np.inf,0,20,40,60,np.inf],["<0","0-20","20-40","40-60",">=60"],res,h)
    sdf=pd.DataFrame(summary); df=pd.DataFrame(rows)
    # Rank segments with enough support by absolute systematic bias, not merely noise RMSE.
    eligible=df[df.n>=30].copy(); eligible["abs_mean_residual"]=eligible.mean_residual.abs(); priority=eligible.sort_values(["abs_mean_residual","n"],ascending=[False,False]).head(30)
    print("=== LARGE-RISE FIR-1 RESIDUAL ATLAS ==="); print("cohort: true target-current >=40 mg/dL | diagnostic only"); print("future CGM feature: False | live targets used: False")
    print("\nLARGE-RISE SUMMARY"); print(sdf.to_string(index=False))
    print("\nTOP SYSTEMATIC RESIDUAL SEGMENTS (n>=30)"); print(priority[["horizon_minutes","factor","segment","n","share_pct","mean_residual","median_residual","underprediction_rate","rmse_residual"]].to_string(index=False))
    print("\nBY FIR-1 PREDICTED RISE"); print(df[df.factor=="fir1_predicted_rise"].to_string(index=False))
    print("\nBY RECENT CARBS"); print(df[df.factor.isin(["carbs_last60","carbs_amount_last60"])].to_string(index=False))
    print("\nBY RECENT BOLUS"); print(df[df.factor.isin(["bolus_last60","bolus_amount_last60"])].to_string(index=False))
    print("\nBY 30-MIN TREND"); print(df[df.factor=="trend_30m"].to_string(index=False))
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); sdf.to_csv(out/"summary.csv",index=False); df.to_csv(out/"segments.csv",index=False); priority.to_csv(out/"priority_segments.csv",index=False)
    report={"experiment":"large-rise FIR-1 residual atlas","cohort":"target-current >=40 mg/dL","future_cgm_feature":False,"live_targets_used":False,"summary":summary,"interpretation_rule":"Look for supported anchor-available factors with monotonic/systematic residual magnitude; future target defines cohort only."}; (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(f"\nReport: {out/'report.json'}")
if __name__=="__main__": main()
