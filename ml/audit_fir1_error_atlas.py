"""FIR-1 Scale-50 Error Atlas.

Diagnostic only: no training, no threshold tuning, no Live targets.
Uses frozen FIR-1 validation predictions and the matching Scale-50 validation
windows to identify which physiological/context regimes dominate squared error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (30, 60, 90, 120)


def load_validation(data_dir: Path):
    xs, fs, ys = [], [], []
    for p in sorted((data_dir / "validation").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            xs.append(z["x"].astype(np.float32))
            fs.append(z["future_known"].astype(np.float32))
            ys.append(z["y"].astype(np.float32))
    if not xs:
        raise ValueError("No validation shards found")
    return np.concatenate(xs), np.concatenate(fs), np.concatenate(ys)


def direction_class(d):
    return np.where(d > 5, 1, np.where(d < -5, -1, 0))


def summarize(mask, y, pred, current, total_sse):
    n = int(mask.sum())
    if n == 0:
        return None
    rows = []
    for j, h in enumerate(HORIZONS):
        yt = y[mask, j].astype(float)
        yp = pred[mask, j].astype(float)
        cur = current[mask].astype(float)
        err = yp - yt
        sse = float(np.sum(err ** 2))
        rows.append({
            "horizon_minutes": h,
            "n": n,
            "share_pct": 100.0 * n / len(y),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mard": float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-6)) * 100),
            "direction_accuracy": float(np.mean(direction_class(yt-cur) == direction_class(yp-cur))),
            "squared_error_share_pct": 100.0 * sse / total_sse[j] if total_sse[j] else 0.0,
        })
    return rows


def add_group(out, group, labels, y, pred, current, total_sse):
    for label, mask in labels:
        rows = summarize(mask, y, pred, current, total_sse)
        if rows:
            for r in rows:
                r["group"] = group
                r["segment"] = label
                out.append(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/v15_4_scale50")
    ap.add_argument("--predictions", default="ml/results/fir1_scale50/validation_predictions.npz")
    ap.add_argument("--outdir", default="ml/results/fir1_error_atlas")
    args = ap.parse_args()

    x, future, y_data = load_validation(Path(args.data_dir))
    with np.load(args.predictions, allow_pickle=False) as z:
        y = z["y_true"].astype(np.float32)
        current = z["current_glucose"].astype(np.float32)
        pred = z["y_pred"].astype(np.float32)

    if len(y) != len(x):
        raise ValueError(f"Prediction/data row mismatch: predictions={len(y)}, validation={len(x)}")
    if y.shape != y_data.shape or not np.allclose(y, y_data, atol=1e-4, rtol=0):
        raise ValueError("Prediction targets do not align with Scale-50 validation targets")
    if not np.allclose(current, x[:, -1, 0], atol=1e-4, rtol=0):
        raise ValueError("Current glucose does not align with validation windows")

    # V14 history feature indices: glucose=0, basal=5, basal_missing=6,
    # bolus=7, bolus_missing=8, insulin=9, insulin_missing=10,
    # carbs=11, carbs_missing=12. Use raw windows only.
    d5 = x[:, -1, 1]
    d30 = x[:, -1, 3]
    future_change = y - current[:, None]
    abs_future_change = np.max(np.abs(future_change), axis=1)

    recent = min(12, x.shape[1])  # last hour at 5-minute cadence
    recent_bolus = np.any((x[:, -recent:, 8] < 0.5) & (x[:, -recent:, 7] > 0), axis=1)
    recent_carbs = np.any((x[:, -recent:, 12] < 0.5) & (x[:, -recent:, 11] > 0), axis=1)
    recent_insulin = np.any((x[:, -recent:, 10] < 0.5) & (x[:, -recent:, 9] > 0), axis=1)

    total_sse = np.sum((pred.astype(float) - y.astype(float)) ** 2, axis=0)
    atlas = []

    add_group(atlas, "current_glucose", [
        ("<70", current < 70),
        ("70-180", (current >= 70) & (current <= 180)),
        (">180", current > 180),
    ], y, pred, current, total_sse)

    add_group(atlas, "recent_trend_30m", [
        ("falling_fast_<-30", d30 < -30),
        ("falling_-30_-10", (d30 >= -30) & (d30 < -10)),
        ("stable_-10_10", (d30 >= -10) & (d30 <= 10)),
        ("rising_10_30", (d30 > 10) & (d30 <= 30)),
        ("rising_fast_>30", d30 > 30),
    ], y, pred, current, total_sse)

    add_group(atlas, "recent_trend_5m", [
        ("falling_<-5", d5 < -5),
        ("stable_-5_5", (d5 >= -5) & (d5 <= 5)),
        ("rising_>5", d5 > 5),
    ], y, pred, current, total_sse)

    add_group(atlas, "recent_events_60m", [
        ("bolus_yes", recent_bolus), ("bolus_no", ~recent_bolus),
        ("carbs_yes", recent_carbs), ("carbs_no", ~recent_carbs),
        ("insulin_yes", recent_insulin), ("insulin_no", ~recent_insulin),
        ("bolus_or_carbs", recent_bolus | recent_carbs),
        ("neither_bolus_nor_carbs", ~(recent_bolus | recent_carbs)),
    ], y, pred, current, total_sse)

    add_group(atlas, "future_excursion_max_4h", [
        ("<20", abs_future_change < 20),
        ("20-40", (abs_future_change >= 20) & (abs_future_change < 40)),
        ("40-80", (abs_future_change >= 40) & (abs_future_change < 80)),
        (">=80", abs_future_change >= 80),
    ], y, pred, current, total_sse)

    # Clinically interpretable transition regimes per horizon.
    for j, h in enumerate(HORIZONS):
        target = y[:, j]
        masks = [
            ("normal_to_normal", (current >= 70) & (current <= 180) & (target >= 70) & (target <= 180)),
            ("to_hypo_<70", target < 70),
            ("to_hyper_>180", target > 180),
            ("large_drop_>=40", target-current <= -40),
            ("large_rise_>=40", target-current >= 40),
        ]
        for label, mask in masks:
            n = int(mask.sum())
            if not n:
                continue
            yt, yp, cur = target[mask].astype(float), pred[mask, j].astype(float), current[mask].astype(float)
            err = yp-yt
            atlas.append({
                "group": f"transition_{h}", "segment": label, "horizon_minutes": h,
                "n": n, "share_pct": 100*n/len(y),
                "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2))),
                "mard": float(np.mean(np.abs(err)/np.maximum(np.abs(yt),1e-6))*100),
                "direction_accuracy": float(np.mean(direction_class(yt-cur)==direction_class(yp-cur))),
                "squared_error_share_pct": 100*float(np.sum(err**2))/total_sse[j],
            })

    df = pd.DataFrame(atlas)
    cols = ["group","segment","horizon_minutes","n","share_pct","mae","rmse","mard","direction_accuracy","squared_error_share_pct"]
    df = df[cols]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "error_atlas.csv", index=False)

    # Priority view: segments that consume disproportionately more SSE than rows.
    priority = df[df.n >= max(50, int(0.01*len(y)))].copy()
    priority["error_concentration_ratio"] = priority.squared_error_share_pct / priority.share_pct
    priority = priority.sort_values(["error_concentration_ratio", "squared_error_share_pct"], ascending=False)
    priority.to_csv(out / "priority_segments.csv", index=False)

    top = priority.head(20)
    report = {
        "experiment": "FIR-1 Scale-50 Error Atlas",
        "validation_rows": int(len(y)),
        "future_cgm_input": False,
        "live_targets_used": False,
        "diagnostic_only": True,
        "priority_definition": "squared-error share divided by row share; minimum 1% of validation rows",
        "top_priority_segments": top[["group","segment","horizon_minutes","n","share_pct","rmse","mard","squared_error_share_pct","error_concentration_ratio"]].to_dict("records"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== FIR-1 SCALE-50 ERROR ATLAS ===")
    print(f"validation rows: {len(y):,}")
    print("future CGM input: False")
    print("live targets used: False")
    print("\nTOP ERROR-CONCENTRATION SEGMENTS")
    print(top[["group","segment","horizon_minutes","n","share_pct","rmse","mard","squared_error_share_pct","error_concentration_ratio"]].to_string(index=False))

    print("\nCURRENT GLUCOSE")
    print(df[df.group=="current_glucose"].to_string(index=False))
    print("\nRECENT 30-MIN TREND")
    print(df[df.group=="recent_trend_30m"].to_string(index=False))
    print("\nRECENT EVENTS (LAST 60 MIN)")
    print(df[df.group=="recent_events_60m"].to_string(index=False))
    print("\nFUTURE EXCURSION")
    print(df[df.group=="future_excursion_max_4h"].to_string(index=False))
    print(f"\nReport: {out / 'report.json'}")
    print(f"Full atlas: {out / 'error_atlas.csv'}")


if __name__ == "__main__":
    main()
