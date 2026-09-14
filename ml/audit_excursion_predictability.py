"""Prospective predictability audit for FIR-1's dominant error regime.

Question: can large future glucose rises (target-current >=40 mg/dL) be detected
using ONLY information available at the prediction anchor plus frozen FIR-1
predictions? This is a diagnostic classifier audit, not a competition model.

Leakage controls:
- future targets are labels only
- no future CGM is used as a feature
- deterministic calibration/holdout split
- preprocessing fitted on calibration only
- threshold selected on calibration only, then frozen on holdout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (30, 60, 90, 120)
SEED = 42


def load_validation(data_dir: Path):
    xs, fs, ys = [], [], []
    for p in sorted((data_dir / "validation").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            xs.append(z["x"].astype(np.float32))
            fs.append(z["future_known"].astype(np.float32))
            ys.append(z["y"].astype(np.float32))
    if not xs:
        raise ValueError("No validation shards")
    return np.concatenate(xs), np.concatenate(fs), np.concatenate(ys)


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(x, y, steps=1500, lr=0.03, l2=1e-3):
    # Lightweight NumPy logistic regression to avoid adding sklearn dependency.
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y)-y.sum()), 1.0)
    weights = np.where(y > 0, len(y)/(2*pos), len(y)/(2*neg))
    for _ in range(steps):
        p = sigmoid(x @ w + b)
        e = (p-y) * weights
        w -= lr * ((x.T @ e)/len(y) + l2*w)
        b -= lr * float(np.mean(e))
    return w, b


def confusion(y, p, threshold):
    pred = p >= threshold
    yb = y.astype(bool)
    tp = int(np.sum(pred & yb)); fp = int(np.sum(pred & ~yb))
    fn = int(np.sum(~pred & yb)); tn = int(np.sum(~pred & ~yb))
    precision = tp/max(tp+fp,1); recall = tp/max(tp+fn,1)
    f1 = 2*precision*recall/max(precision+recall,1e-12)
    specificity = tn/max(tn+fp,1)
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":precision,"recall":recall,"f1":f1,"specificity":specificity,"flag_rate":float(np.mean(pred))}


def auc_rank(y, score):
    y = y.astype(bool)
    npos = int(y.sum()); nneg = int((~y).sum())
    if not npos or not nneg: return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float); ranks[order] = np.arange(1,len(score)+1)
    # adequate here; exact ties are rare for continuous logistic scores
    return float((ranks[y].sum()-npos*(npos+1)/2)/(npos*nneg))


def make_features(x, future, fir_pred, current):
    # Anchor-available raw history features from V14.x.
    recent = min(12, x.shape[1])
    bolus = np.where(x[:,-recent:,8] < .5, x[:,-recent:,7], 0).sum(axis=1)
    carbs = np.where(x[:,-recent:,12] < .5, x[:,-recent:,11], 0).sum(axis=1)
    insulin = np.where(x[:,-recent:,10] < .5, x[:,-recent:,9], 0).sum(axis=1)
    basal = np.where(x[:,-1,6] < .5, x[:,-1,5], 0)

    # Future-known interventions only; never future CGM. FIR-1 prediction itself is
    # legal at inference and summarizes the frozen global model's expectation.
    future_summary = []
    for steps in (6,12,18,24):
        f = future[:,:steps,:]
        # Use robust aggregate statistics over all future-known channels. Missing
        # indicators are retained as channels in the source tensor.
        future_summary.extend([f.sum(axis=1), f.max(axis=1), f.mean(axis=1)])

    parts = [
        current[:,None], x[:,-1,1:5],
        bolus[:,None], carbs[:,None], insulin[:,None], basal[:,None],
        fir_pred, fir_pred-current[:,None],
    ] + future_summary
    feats = np.concatenate(parts, axis=1).astype(np.float64)
    feats[~np.isfinite(feats)] = 0.0
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/v15_4_scale50")
    ap.add_argument("--predictions", default="ml/results/fir1_scale50/validation_predictions.npz")
    ap.add_argument("--outdir", default="ml/results/excursion_predictability")
    args = ap.parse_args()

    x, future, y_data = load_validation(Path(args.data_dir))
    with np.load(args.predictions, allow_pickle=False) as z:
        y = z["y_true"].astype(np.float32)
        current = z["current_glucose"].astype(np.float32)
        fir = z["y_pred"].astype(np.float32)
    if len(y) != len(x) or not np.allclose(y, y_data, atol=1e-4, rtol=0):
        raise ValueError("Prediction/data alignment failed")
    if not np.allclose(current, x[:,-1,0], atol=1e-4, rtol=0):
        raise ValueError("Current glucose alignment failed")

    feats = make_features(x, future, fir, current)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y)); cut = len(y)//2
    cal, hold = order[:cut], order[cut:]
    mu = feats[cal].mean(axis=0); sd = feats[cal].std(axis=0); sd[sd < 1e-6] = 1.0
    xc = (feats[cal]-mu)/sd; xh = (feats[hold]-mu)/sd

    rows=[]
    print("=== EXCURSION PREDICTABILITY AUDIT ===")
    print(f"rows: {len(y):,} | calibration: {len(cal):,} | holdout: {len(hold):,}")
    print(f"features: {feats.shape[1]} | future CGM input: False | live targets used: False")
    print("label: target glucose - current glucose >= 40 mg/dL")

    for j,h in enumerate(HORIZONS):
        label = (y[:,j]-current >= 40).astype(float)
        yc, yh = label[cal], label[hold]
        w,b = fit_logistic(xc,yc)
        pc=sigmoid(xc@w+b); ph=sigmoid(xh@w+b)

        # Frozen selection objective: among thresholds with calibration recall >=80%,
        # maximize F1; if none, maximize F1 globally. This favors catching the error
        # regime while constraining excessive false positives.
        candidates=[]
        for t in np.arange(.05,.951,.01):
            m=confusion(yc,pc,float(t)); candidates.append((float(t),m))
        eligible=[q for q in candidates if q[1]["recall"] >= .80]
        pool=eligible if eligible else candidates
        threshold, cm=max(pool,key=lambda q:(q[1]["f1"],q[1]["precision"]))
        hm=confusion(yh,ph,threshold)
        auc=auc_rank(yh,ph)
        prevalence=float(yh.mean())
        lift=hm["precision"]/prevalence if prevalence else float("nan")

        row={"horizon_minutes":h,"holdout_n":len(hold),"positive_n":int(yh.sum()),"prevalence":prevalence,"threshold":threshold,"auc":auc,"precision":hm["precision"],"recall":hm["recall"],"f1":hm["f1"],"specificity":hm["specificity"],"flag_rate":hm["flag_rate"],"precision_lift_vs_prevalence":lift}
        rows.append(row)
        print(f"+{h:3d} | prevalence {100*prevalence:5.1f}% | AUC {auc:.3f} | threshold {threshold:.2f} | precision {100*hm['precision']:5.1f}% | recall {100*hm['recall']:5.1f}% | F1 {hm['f1']:.3f} | flag {100*hm['flag_rate']:5.1f}% | lift {lift:.2f}x")

    df=pd.DataFrame(rows)
    # Gate: prospective signal must be useful across horizons, not just one cherry-picked
    # case. Require mean AUC >=.75 and >=3/4 horizons with recall >=.75 and lift >=1.5.
    qualified=((df.recall>=.75)&(df.precision_lift_vs_prevalence>=1.5)).sum()
    go=bool(df.auc.mean()>=.75 and qualified>=3)
    print("\nSIGNAL GATE")
    print(f"mean holdout AUC: {df.auc.mean():.3f}")
    print(f"qualified horizons (recall>=75%, precision lift>=1.5x): {qualified}/4")
    print(f"EXCURSION SPECIALIST: {'GO' if go else 'NO-GO'}")

    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"holdout_metrics.csv",index=False)
    report={"experiment":"prospective large-rise predictability audit","seed":SEED,"label":"target-current >=40 mg/dL","future_cgm_input":False,"live_targets_used":False,"calibration_n":len(cal),"holdout_n":len(hold),"features":int(feats.shape[1]),"threshold_selection":"calibration only: max F1 among recall>=80%, else max F1","gate":"mean holdout AUC>=0.75 and >=3/4 horizons recall>=75% with precision lift>=1.5x","qualified_horizons":int(qualified),"mean_holdout_auc":float(df.auc.mean()),"excursion_specialist_go":go,"metrics":rows}
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"\nReport: {out/'report.json'}")

if __name__ == "__main__":
    main()
