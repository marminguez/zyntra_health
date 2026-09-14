"""Audit GlucoFM 24h sidecars before they are allowed into Zyntra.

Checks structural validity and exact 1:1 alignment against the frozen Scale-10
reference shards. This script never approximates GlucoFM itself.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

STEPS = 288


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", default="ml/data/glucofm_scale10")
    ap.add_argument("--reference", default="ml/data/v15_4_scale10")
    a = ap.parse_args()

    sidecar_root = Path(a.sidecar)
    reference_root = Path(a.reference)
    files = sorted(sidecar_root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ sidecar shards found under {sidecar_root}")

    reference_files = sorted(reference_root.rglob("*.npz"))
    if not reference_files:
        raise FileNotFoundError(f"No reference NPZ shards found under {reference_root}")

    sidecar_rel = {p.relative_to(sidecar_root) for p in files}
    reference_rel = {p.relative_to(reference_root) for p in reference_files}
    if sidecar_rel != reference_rel:
        missing = sorted(reference_rel - sidecar_rel)
        extra = sorted(sidecar_rel - reference_rel)
        raise AssertionError(
            f"Shard set mismatch: missing={missing[:5]} extra={extra[:5]} "
            f"(missing_count={len(missing)}, extra_count={len(extra)})"
        )

    total = 0
    observed = 0
    embedding_dim = None
    for path in files:
        rel = path.relative_to(sidecar_root)
        ref_path = reference_root / rel

        with np.load(path, allow_pickle=False) as z, np.load(ref_path, allow_pickle=False) as ref:
            required = {"cgm_24h", "observed_mask", "timestamp"}
            missing = required.difference(z.files)
            if missing:
                raise AssertionError(f"{path}: missing keys {sorted(missing)}")
            if "timestamp" not in ref.files:
                raise AssertionError(f"{ref_path}: reference missing timestamp")

            cgm = z["cgm_24h"]
            mask = z["observed_mask"]
            ts = z["timestamp"]
            ref_ts = ref["timestamp"]

            if cgm.ndim != 2 or cgm.shape[1] != STEPS:
                raise AssertionError(f"{path}: cgm_24h shape {cgm.shape}, expected (N,{STEPS})")
            if mask.shape != cgm.shape:
                raise AssertionError(f"{path}: mask shape {mask.shape} != CGM shape {cgm.shape}")
            if len(ts) != len(cgm):
                raise AssertionError(f"{path}: timestamp count mismatch")
            if ts.shape != ref_ts.shape or not np.array_equal(ts, ref_ts):
                raise AssertionError(f"{path}: timestamps are not exact 1:1 match with {ref_path}")
            if not np.isfinite(cgm).all():
                raise AssertionError(f"{path}: non-finite CGM values")
            if not np.isin(mask, [0, 1]).all():
                raise AssertionError(f"{path}: observed_mask must be binary")
            if np.any((mask == 0) & (cgm != 0.0)):
                raise AssertionError(f"{path}: missing CGM slots must use neutral fill 0.0")

            if "embedding" in z.files:
                emb = z["embedding"]
                if emb.ndim != 2 or len(emb) != len(cgm) or not np.isfinite(emb).all():
                    raise AssertionError(f"{path}: invalid embedding shape/values")
                if embedding_dim is None:
                    embedding_dim = emb.shape[1]
                elif emb.shape[1] != embedding_dim:
                    raise AssertionError(f"{path}: inconsistent embedding dimension")

            total += len(cgm)
            observed += int(mask.sum())

    slots = total * STEPS
    print("=== GLUCOFM SIDECAR AUDIT ===")
    print(f"shards:           {len(files):,}")
    print(f"anchors:          {total:,}")
    print("alignment:        exact shard + row + timestamp match")
    print("CGM context:      288 x 5 min = 24h, ending at anchor")
    print(f"observed CGM:     {observed:,}/{slots:,} ({100.0 * observed / slots:.2f}%)")
    print(f"embedding dim:    {embedding_dim if embedding_dim is not None else 'not generated yet'}")
    print("future CGM input: False by construction")
    print("PASS — sidecar is structurally valid and exactly aligned to frozen Scale-10 anchors.")


if __name__ == "__main__":
    main()
