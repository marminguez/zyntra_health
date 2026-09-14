"""Leakage-safe audit of FIR-1 absolute/delta fusion weights.

No model is trained. The frozen FIR-1 Scale-50 validation predictions are split
once, deterministically, into calibration and holdout subsets. Per-horizon alpha
weights are selected ONLY on calibration, then evaluated unchanged on holdout.

final = alpha * absolute + (1-alpha) * delta_reconstructed

The 0.50 baseline is evaluated on the exact same holdout rows. This audit asks
whether useful performance is already hidden in the two trained heads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (30, 60, 90, 120)
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
SEED = 42


def direction_class(delta):
    return np.where(delta > 5, 1, np.where(delta < -5, -1, 0))


def metrics(y, p, current):
    rows = []
    for i, h in enumerate(HORIZONS):
        yt = y[:, i].astype(float)
        yp = p[:, i].astype(float)
        err = yp - yt
        rows.append({
            "horizon_minutes": h,
            "n": len(yt),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mard": float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-6)) * 100),
            "direction_accuracy": float(np.mean(
                direction_class(yt-current) == direction_class(yp-current)
            )),
        })
    return pd.DataFrame(rows)


def fuse(abs_p, delta_p, alpha):
    a = np.asarray(alpha, dtype=np.float32).reshape(1, 4)
    return a * abs_p + (1.0-a) * delta_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="ml/results/fir1_scale50/validation_predictions.npz")
    ap.add_argument("--outdir", default="ml/results/fir1_fusion_audit")
    ap.add_argument("--calibration-fraction", type=float, default=0.50)
    args = ap.parse_args()

    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration fraction must be between 0 and 1")

    with np.load(args.predictions, allow_pickle=False) as z:
        y = z["y_true"].astype(np.float32)
        current = z["current_glucose"].astype(np.float32)
        abs_p = z["absolute_head"].astype(np.float32)
        delta_p = z["delta_reconstructed"].astype(np.float32)

    if y.shape != abs_p.shape or y.shape != delta_p.shape or y.shape[1] != 4:
        raise ValueError(f"Unexpected prediction shapes: y={y.shape}, abs={abs_p.shape}, delta={delta_p.shape}")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y))
    n_cal = int(round(len(y) * args.calibration_fraction))
    cal_idx, hold_idx = order[:n_cal], order[n_cal:]

    selected = []
    calibration_rows = []
    for j, h in enumerate(HORIZONS):
        yt = y[cal_idx, j].astype(float)
        best = None
        for alpha in ALPHAS:
            pred = alpha * abs_p[cal_idx, j] + (1.0-alpha) * delta_p[cal_idx, j]
            err = pred - yt
            mard = float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-6)) * 100)
            rmse = float(np.sqrt(np.mean(err ** 2)))
            calibration_rows.append({"horizon_minutes": h, "alpha_absolute": float(alpha), "mard": mard, "rmse": rmse})
            # Primary objective MARD; RMSE breaks ties. Grid is frozen before seeing results.
            score = (mard, rmse)
            if best is None or score < best[0]:
                best = (score, float(alpha))
        selected.append(best[1])

    selected = np.asarray(selected, dtype=np.float32)
    baseline_alpha = np.full(4, 0.50, dtype=np.float32)
    baseline_hold = fuse(abs_p[hold_idx], delta_p[hold_idx], baseline_alpha)
    tuned_hold = fuse(abs_p[hold_idx], delta_p[hold_idx], selected)

    mb = metrics(y[hold_idx], baseline_hold, current[hold_idx])
    mt = metrics(y[hold_idx], tuned_hold, current[hold_idx])

    base_mard = float(mb.mard.mean())
    tuned_mard = float(mt.mard.mean())
    base_rmse = float(mb.rmse.mean())
    tuned_rmse = float(mt.rmse.mean())
    base_dir = float(mb.direction_accuracy.mean())
    tuned_dir = float(mt.direction_accuracy.mean())
    rel_mard = 100.0 * (tuned_mard / base_mard - 1.0)
    rel_rmse = 100.0 * (tuned_rmse / base_rmse - 1.0)
    dir_pp = 100.0 * (tuned_dir - base_dir)

    # Same rapid gate used for structural Scale-10 screens: material MARD or
    # direction gain, while preventing a meaningful RMSE regression.
    go = ((rel_mard <= -2.0) or (dir_pp >= 0.5)) and rel_rmse <= 1.0

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(calibration_rows).to_csv(out / "calibration_grid.csv", index=False)
    mb.to_csv(out / "holdout_baseline_050_metrics.csv", index=False)
    mt.to_csv(out / "holdout_selected_fusion_metrics.csv", index=False)

    report = {
        "experiment": "FIR-1 Scale-50 leakage-safe fusion audit",
        "predictions": args.predictions,
        "seed": SEED,
        "calibration_fraction": args.calibration_fraction,
        "calibration_n": int(len(cal_idx)),
        "holdout_n": int(len(hold_idx)),
        "alpha_definition": "alpha*absolute + (1-alpha)*delta_reconstructed",
        "alpha_grid": ALPHAS.tolist(),
        "selection_objective": "per-horizon minimum calibration MARD; RMSE tie-break",
        "selected_alpha_absolute": {str(h): float(a) for h, a in zip(HORIZONS, selected)},
        "holdout_baseline": {"mard": base_mard, "rmse": base_rmse, "direction": base_dir},
        "holdout_selected": {"mard": tuned_mard, "rmse": tuned_rmse, "direction": tuned_dir},
        "holdout_change": {"mard_relative_pct": rel_mard, "rmse_relative_pct": rel_rmse, "direction_pp": dir_pp},
        "promotion_gate": "MARD improves >=2% relative OR direction >=+0.5pp, with RMSE regression <=1%",
        "go": bool(go),
        "important_scope": "weights selected on calibration only and evaluated once on disjoint holdout; this is not the official Live score",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== FIR-1 FUSION AUDIT ===")
    print(f"validation rows: {len(y):,}")
    print(f"calibration rows: {len(cal_idx):,}")
    print(f"holdout rows: {len(hold_idx):,}")
    print("alpha = absolute weight; 1-alpha = reconstructed-delta weight")
    print("selected alphas:")
    for h, a in zip(HORIZONS, selected):
        print(f"  +{h}: {a:.2f}")

    print("\nHOLDOUT BASELINE 0.50/0.50")
    print(mb.to_string(index=False))
    print("\nHOLDOUT SELECTED FUSION")
    print(mt.to_string(index=False))
    print("\n4H HOLDOUT COMPARISON")
    print(f"MARD: {base_mard:.6f} -> {tuned_mard:.6f} ({rel_mard:+.3f}% relative)")
    print(f"RMSE: {base_rmse:.6f} -> {tuned_rmse:.6f} ({rel_rmse:+.3f}% relative)")
    print(f"Direction: {100*base_dir:.4f}% -> {100*tuned_dir:.4f}% ({dir_pp:+.3f} pp)")
    print(f"FUSION PROMOTION: {'GO' if go else 'NO-GO'}")
    print(f"\nReport: {out / 'report.json'}")


if __name__ == "__main__":
    main()
