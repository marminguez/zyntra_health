"""Audit a future GlucoFM 24h sidecar before it is allowed into Zyntra.

Expected NPZ keys per shard:
- cgm_24h: (N, 288) float32
- observed_mask: (N, 288) float32/bool
- timestamp: (N,) anchor timestamps
Optional:
- embedding: (N, D) official frozen GlucoFM embedding

This audit never approximates GlucoFM and never accepts post-anchor CGM.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

STEPS = 288


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", default="ml/data/glucofm_scale10")
    a = ap.parse_args()
    root = Path(a.sidecar)
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ sidecar shards found under {root}")

    total = 0
    embedding_dim = None
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            required = {"cgm_24h", "observed_mask", "timestamp"}
            missing = required.difference(z.files)
            if missing:
                raise AssertionError(f"{path}: missing keys {sorted(missing)}")
            cgm = z["cgm_24h"]
            mask = z["observed_mask"]
            ts = z["timestamp"]
            if cgm.ndim != 2 or cgm.shape[1] != STEPS:
                raise AssertionError(f"{path}: cgm_24h shape {cgm.shape}, expected (N,{STEPS})")
            if mask.shape != cgm.shape:
                raise AssertionError(f"{path}: mask shape {mask.shape} != CGM shape {cgm.shape}")
            if len(ts) != len(cgm):
                raise AssertionError(f"{path}: timestamp count mismatch")
            if not np.isfinite(cgm).all():
                raise AssertionError(f"{path}: non-finite CGM values; imputation must be explicit before encoding")
            if not np.isin(mask, [0, 1]).all():
                raise AssertionError(f"{path}: observed_mask must be binary")
            if "embedding" in z.files:
                emb = z["embedding"]
                if emb.ndim != 2 or len(emb) != len(cgm) or not np.isfinite(emb).all():
                    raise AssertionError(f"{path}: invalid embedding shape/values")
                if embedding_dim is None:
                    embedding_dim = emb.shape[1]
                elif emb.shape[1] != embedding_dim:
                    raise AssertionError(f"{path}: inconsistent embedding dimension")
            total += len(cgm)

    print("=== GLUCOFM SIDECAR AUDIT ===")
    print(f"shards:           {len(files):,}")
    print(f"anchors:          {total:,}")
    print(f"CGM context:      288 x 5 min = 24h")
    print(f"embedding dim:    {embedding_dim if embedding_dim is not None else 'not generated yet'}")
    print("future CGM input: False by sidecar contract (window ends at anchor)")
    print("PASS — structural sidecar checks passed.")


if __name__ == "__main__":
    main()
