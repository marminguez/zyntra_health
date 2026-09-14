"""Audit whether same-patient history can improve the FIR-1 global model.

This is a PERSONALIZATION SIGNAL AUDIT, not the final unbiased competition estimate.
FIR-1 Scale-50 was trained on these train windows already, so the purpose here is
narrow: test whether its residual error contains a persistent patient-specific
component that can be learned strictly from earlier observations of the same
patient and improve later predictions.

Design:
- load frozen FIR-1 Scale-50 weights
- for each Scale-50 train patient shard, sort windows chronologically
- earliest 70% = personalization memory; latest 30% = query/evaluation
- memory anchors whose +120 target would overlap the query period are excluded
- learn only a per-horizon residual offset from past errors
- shrink the patient offset toward zero when little memory is available
- report global vs personalized metrics for several memory budgets

No Live targets and no future CGM are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.forecasting.future_intervention_features import build_future_intervention_summary
from ml.forecasting.model_fir1 import HORIZONS, build_fir1_forecaster

CHANNELS = (2, 3, 4, 5, 6, 7)
GAP = np.timedelta64(120, "m")
MEMORY_BUDGETS = (24, 48, 96, 192, 0)  # 0 = all temporally safe memory


def norm_future(f: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    f = f.copy()
    for vi, mi, hi in ((0, 1, 5), (2, 3, 7), (4, 5, 11)):
        present = f[:, :, mi] < 0.5
        f[:, :, vi] = np.where(
            present,
            (f[:, :, vi] - float(mean[hi])) / float(std[hi]),
            0.0,
        )
    return f.astype(np.float32)


def direction_class(delta):
    return np.where(delta > 5, 1, np.where(delta < -5, -1, 0))


def metrics(y: np.ndarray, p: np.ndarray, current: np.ndarray) -> dict:
    err = p - y
    rmse = np.sqrt(np.mean(err * err, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    mard = 100.0 * np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-6), axis=0)
    direction = np.mean(
        direction_class(y - current[:, None]) == direction_class(p - current[:, None]),
        axis=0,
    )
    return {
        "n": int(len(y)),
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "mard": mard.tolist(),
        "direction": direction.tolist(),
        "rmse_mean": float(rmse.mean()),
        "mard_mean": float(mard.mean()),
        "direction_mean": float(direction.mean()),
    }


def predict_fir1(model, x_raw, future_raw, mean, std, batch_size):
    current = x_raw[:, -1, 0].astype(np.float32)
    summary = build_future_intervention_summary(
        future_raw,
        basal_mean=float(mean[5]),
        basal_std=float(std[5]),
    )
    x = ((x_raw - mean) / std).astype(np.float32)
    f = norm_future(future_raw, mean, std)
    raw = model.predict([x, f, summary], batch_size=batch_size, verbose=0)
    absolute = np.column_stack([raw[i].ravel() for i in range(4)])
    delta_head = np.column_stack([raw[i + 4].ravel() for i in range(4)])
    reconstructed = current[:, None] + delta_head
    return (0.5 * absolute + 0.5 * reconstructed).astype(np.float32)


def patient_offset(residual_memory: np.ndarray, shrink_k: float) -> np.ndarray:
    """Robust past residual with empirical-Bayes style shrinkage toward zero."""
    n = len(residual_memory)
    if n == 0:
        return np.zeros(4, dtype=np.float32)
    center = np.median(residual_memory, axis=0)
    weight = n / (n + shrink_k)
    return (weight * center).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/v15_4_scale50")
    ap.add_argument("--v14-1-dir", default="ml/results/v14_1")
    ap.add_argument("--weights", default="ml/results/fir1_scale50/best.weights.h5")
    ap.add_argument("--output", default="ml/results/personalized_residual_adapter/audit.json")
    ap.add_argument("--memory-fraction", type=float, default=0.70)
    ap.add_argument("--min-windows", type=int, default=40)
    ap.add_argument("--shrink-k", type=float, default=24.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-patients", type=int, default=0)
    args = ap.parse_args()

    if not 0.0 < args.memory_fraction < 1.0:
        raise ValueError("--memory-fraction must be between 0 and 1")

    root = Path(args.data_dir)
    files = sorted((root / "train").glob("*.npz"))
    if not files:
        raise FileNotFoundError(root / "train")
    if args.max_patients > 0:
        files = files[: args.max_patients]

    with np.load(Path(args.v14_1_dir) / "normalization.npz") as z:
        mean = z["mean"].astype(np.float32)
        std = z["std"].astype(np.float32)

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)

    # Infer shapes from first non-empty shard, then restore the frozen champion.
    sample = None
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            if len(z["y"]):
                sample = z["x"]
                break
    if sample is None:
        raise RuntimeError("No non-empty train shards")

    model = build_fir1_forecaster(sample.shape[1], sample.shape[2])
    model.load_weights(weights)

    buckets = {
        b: {"y": [], "global": [], "personalized": [], "current": [], "patients": set()}
        for b in MEMORY_BUDGETS
    }
    skipped_small = 0
    skipped_no_safe_memory = 0
    total_patients = 0

    print("=== PERSONALIZED RESIDUAL ADAPTER SIGNAL AUDIT ===")
    print(f"patient shards: {len(files):,}")
    print(f"global model: FIR-1 Scale-50")
    print(f"memory/query split: {args.memory_fraction:.0%}/{1-args.memory_fraction:.0%}")
    print(f"shrink_k: {args.shrink_k:g}")
    print("future CGM input: False")
    print("live targets used: False")

    for fi, p in enumerate(files, 1):
        if "__" not in p.stem:
            raise ValueError(f"Unexpected shard name: {p.name}")
        source, pid = p.stem.split("__", 1)
        with np.load(p, allow_pickle=False) as z:
            x = z["x"].astype(np.float32)
            future = z["future_known"].astype(np.float32)[:, :, CHANNELS]
            y = z["y"].astype(np.float32)
            ts = z["timestamp"].astype("datetime64[ns]")

        n = len(y)
        if n < args.min_windows:
            skipped_small += 1
            continue
        order = np.argsort(ts)
        x, future, y, ts = x[order], future[order], y[order], ts[order]
        cut = max(1, min(n - 1, int(np.floor(n * args.memory_fraction))))
        query_idx = np.arange(cut, n, dtype=np.int64)
        first_query_ts = ts[query_idx[0]]
        safe_memory_idx = np.flatnonzero(ts[:cut] + GAP <= first_query_ts)
        if len(safe_memory_idx) == 0:
            skipped_no_safe_memory += 1
            continue

        pred = predict_fir1(model, x, future, mean, std, args.batch_size)
        residual = y - pred
        current = x[:, -1, 0].astype(np.float32)
        patient_key = f"{source}|{pid}"
        total_patients += 1

        for budget in MEMORY_BUDGETS:
            mem_idx = safe_memory_idx if budget == 0 else safe_memory_idx[-budget:]
            offset = patient_offset(residual[mem_idx], args.shrink_k)
            qy = y[query_idx]
            qg = pred[query_idx]
            qp = qg + offset[None, :]
            buckets[budget]["y"].append(qy)
            buckets[budget]["global"].append(qg)
            buckets[budget]["personalized"].append(qp)
            buckets[budget]["current"].append(current[query_idx])
            buckets[budget]["patients"].add(patient_key)

        if fi % 100 == 0:
            print(f"processed {fi:,}/{len(files):,} shards")

    results = {}
    for budget in MEMORY_BUDGETS:
        d = buckets[budget]
        if not d["y"]:
            continue
        y = np.concatenate(d["y"])
        g = np.concatenate(d["global"])
        q = np.concatenate(d["personalized"])
        current = np.concatenate(d["current"])
        mg = metrics(y, g, current)
        mp = metrics(y, q, current)
        results["all" if budget == 0 else str(budget)] = {
            "memory_budget": "all_safe" if budget == 0 else budget,
            "patients": len(d["patients"]),
            "queries": len(y),
            "global": mg,
            "personalized": mp,
            "mard_relative_change_pct": 100.0 * (mp["mard_mean"] / mg["mard_mean"] - 1.0),
            "rmse_relative_change_pct": 100.0 * (mp["rmse_mean"] / mg["rmse_mean"] - 1.0),
            "direction_change_pp": 100.0 * (mp["direction_mean"] - mg["direction_mean"]),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "personalized-residual-adapter-signal-audit-v1",
        "purpose": "detect persistent same-patient residual signal before building a learned adapter",
        "global_model": "FIR-1 Scale-50",
        "data": str(root),
        "memory_fraction": args.memory_fraction,
        "temporal_guard": "memory anchor +120min <= first query anchor",
        "residual_estimator": "per-horizon median residual with shrinkage toward zero",
        "shrink_k": args.shrink_k,
        "patients_used": total_patients,
        "skipped_small": skipped_small,
        "skipped_no_safe_memory": skipped_no_safe_memory,
        "future_cgm_input": False,
        "live_targets_used": False,
        "important_limitation": "FIR-1 was trained on these train windows; this measures incremental residual persistence, not an unbiased competition score",
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== MEMORY CURVE ===")
    for key, r in results.items():
        print(
            f"memory={key:>4} | patients={r['patients']:>4} | queries={r['queries']:>7} | "
            f"MARD {r['global']['mard_mean']:.3f}->{r['personalized']['mard_mean']:.3f} "
            f"({r['mard_relative_change_pct']:+.2f}%) | "
            f"RMSE {r['global']['rmse_mean']:.3f}->{r['personalized']['rmse_mean']:.3f} "
            f"({r['rmse_relative_change_pct']:+.2f}%) | "
            f"DIR {100*r['global']['direction_mean']:.2f}->{100*r['personalized']['direction_mean']:.2f}% "
            f"({r['direction_change_pp']:+.2f}pp)"
        )

    # Frozen signal gate. We require a material personalization effect before
    # spending time on a learned MLP/adapter.
    best = min(results.values(), key=lambda r: r["mard_relative_change_pct"])
    go = best["mard_relative_change_pct"] <= -5.0 and best["rmse_relative_change_pct"] < 0.0
    print("\nSIGNAL GATE")
    print(f"best memory budget: {best['memory_budget']}")
    print(f"best MARD relative change: {best['mard_relative_change_pct']:+.2f}%")
    print(f"best RMSE relative change: {best['rmse_relative_change_pct']:+.2f}%")
    print(f"LEARNED PERSONALIZED ADAPTER: {'GO' if go else 'NO-GO'}")
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
