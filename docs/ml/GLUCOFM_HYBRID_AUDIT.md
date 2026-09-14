# GlucoFM hybrid integration audit

## Decision

GlucoFM will be used as an additional frozen CGM representation source. It will **not** replace the current Zyntra forecaster at this stage.

Target architecture:

`Zyntra history encoder + Zyntra future-known intervention encoder + frozen GlucoFM context embedding -> existing multi-horizon prediction heads`

The purpose is to test whether GlucoFM contributes complementary 24-hour glycemic context while preserving Zyntra's multimodal and future-intervention logic.

## What is verified from the public GlucoFM paper/blog

- Input CGM is aligned to a 24-hour, 5-minute grid (288 positions).
- Observation masks are preserved; missing values are not treated as true measurements.
- The encoder separates slower state dynamics and shorter-term event/deviation dynamics before fusion.
- The same frozen encoder is used as a reusable representation in downstream tasks.
- For postprandial response forecasting, frozen GlucoFM representations are combined with recent CGM and other context rather than replacing those downstream inputs.
- The paper states that code and reproducibility scripts will be released.

## Current blocker

As of 2026-09-14, the official public paper and Google Research post do not expose a released GlucoFM checkpoint/repository that we can safely integrate and reproduce. The paper explicitly says the authors *will release* code and reproducibility scripts.

Therefore we must not fabricate weights, recreate unpublished implementation details, or call an independently reimplemented architecture 'GlucoFM'.

## Zyntra integration plan

### Stage G0 — integration seam

Prepare Zyntra to consume a precomputed frozen GlucoFM embedding as an optional third context input. No GlucoFM weights are required for this stage.

Requirements:

- Existing Zyntra history branch remains present.
- Existing future-known basal/bolus/carbs branch remains present.
- GlucoFM embedding is projected through a small trainable adapter.
- Only the adapter and Zyntra downstream fusion/heads are trained in the first experiment; GlucoFM stays frozen.
- No future CGM may enter GlucoFM context. The 24-hour window must end at the forecast anchor.

### Stage G1 — 24-hour causal context extraction

When the official checkpoint/code is available, construct a sidecar dataset keyed exactly to each Zyntra anchor:

- 288 x 5-minute CGM grid ending at anchor time;
- observation mask retained;
- no measurements after anchor;
- subject/split provenance retained for leakage auditing;
- embeddings precomputed once and cached locally.

This must be generated from the original MetaboNet time series, not from the current Scale-10 NPZ shards, because the current model shards only carry Zyntra's shorter history window.

### Stage G2 — frozen hybrid fast screen

Run on frozen Scale-10 validation first.

Compare against Scale-10 reference:

- mean MARD: 14.598856%
- mean RMSE: 30.887238
- mean direction accuracy: 69.1653%
- competition DTS-A: 76.525360%

Only a clear structural improvement graduates to Scale-50.

### Stage G3 — controlled fine-tuning

Only if G2 is clearly positive and licensing/competition rules permit it, test partial GlucoFM fine-tuning. This is deliberately not the first experiment.

## Competition/legal status

The currently public MetaboNet leaderboard instructions describe submission format/frequency but do not explicitly state whether externally pretrained weights are permitted in the annual competition. Before an annual submission using GlucoFM weights, obtain confirmation from the organizers or an updated written rule.

## Scientific guardrails

- GlucoFM remains an auxiliary representation, not a replacement baseline.
- No post-anchor CGM.
- Frozen validation remains unchanged.
- GlucoFM availability/version/checkpoint hash must be recorded.
- Never tune preprocessing on the secret/Live targets.
- Report Zyntra-only and Zyntra+GlucoFM side-by-side.
