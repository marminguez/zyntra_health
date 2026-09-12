# V15.4 Scaling Experiment Report

Date: 2026-09-12
Branch: `feat/v14-multi-horizon-forecasting`

## Objective

Measure how much performance improves when increasing the number of training windows per patient while keeping the V15.4 architecture, validation set, targets, normalization, future inputs and 50/50 absolute+delta fusion fixed.

This experiment was designed to answer one specific question: **is data scale still a strong enough lever to reach Top-3 performance, or is the current architecture beginning to saturate?**

## Frozen experimental setup

The following elements were held constant across V15.4, Scale-10 and Scale-50:

- same V15.4 model architecture;
- same 4 forecast horizons: +30, +60, +90 and +120 minutes;
- same history input shape and feature definitions;
- same future-known input family: basal, bolus and carbs with missingness masks;
- no future CGM input;
- same V14.1 train-only normalization statistics;
- same absolute-head + delta-head design;
- same fixed 0.5 absolute + 0.5 reconstructed-delta fusion;
- same seed 42;
- same frozen validation set with 10,686 windows;
- no `test.parquet` targets or live targets used in fitting or internal evaluation.

The scaled dataset builders copy validation byte-for-byte from `ml/data/v15_master`, and `ml/check_v15_4_scaled.py` verifies exact compatibility before training.

## Controlled changes

| Experiment | Train cap / patient | Train windows | Relative to V15.4 |
|---|---:|---:|---:|
| V15.4 | 24 | 29,366 | 1.00x |
| Scale-10 | 240 | 285,096 | 9.71x |
| Scale-50 | 1,200 | 1,331,220 | 45.33x |

Scale-10 therefore increases the usable training set by almost 10x. Scale-50 increases it by more than 45x compared with the original V15.4 training sample.

## Data integrity

Scale-10 and Scale-50 use deterministic per-subject reservoir sampling derived from `SHA256(seed|source|id)` and resumable subject-level dataset construction.

Scale-50 integrity check passed with:

- train shards: 1,235;
- train windows: 1,331,220;
- max windows per shard: 1,200 / cap 1,200;
- validation shards: 224;
- validation windows: 10,686;
- validation byte-identical to frozen V15 master: `True`;
- future CGM input: `False`.

## Competition-aligned evaluation

`ml/evaluate_v15_4_scaling_metrics.py` evaluates all three model variants against the exact same validation targets and `current_glucose`. It uses the same DTS Error Grid equations already used by `ml/evaluate_v15_final_metrics.py`.

### Per-horizon results

| Model | Horizon | MAE | RMSE | MARD | Direction Accuracy | DTS-A | DTS-B | DTS-C | DTS-D | DTS-E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V15.4 | 30 | 13.881 | 20.571 | 9.890% | 62.02% | 87.64% | 11.92% | 0.43% | 0.01% | 0.00% |
| V15.4 | 60 | 21.668 | 31.530 | 15.094% | 66.53% | 75.33% | 23.00% | 1.53% | 0.12% | 0.02% |
| V15.4 | 90 | 26.631 | 38.451 | 18.407% | 68.89% | 67.66% | 29.43% | 2.58% | 0.27% | 0.06% |
| V15.4 | 120 | 29.882 | 42.808 | 20.704% | 69.40% | 62.56% | 33.42% | 3.63% | 0.35% | 0.05% |
| Scale-10 | 30 | 12.667 | 18.964 | 8.970% | 65.16% | 89.34% | 10.40% | 0.26% | 0.00% | 0.00% |
| Scale-10 | 60 | 19.910 | 29.348 | 13.736% | 68.73% | 78.39% | 20.29% | 1.21% | 0.09% | 0.02% |
| Scale-10 | 90 | 24.356 | 35.613 | 16.856% | 70.97% | 71.31% | 26.21% | 2.25% | 0.21% | 0.03% |
| Scale-10 | 120 | 27.137 | 39.624 | 18.834% | 71.80% | 67.06% | 29.52% | 3.05% | 0.36% | 0.02% |
| Scale-50 | 30 | 12.231 | 18.415 | 8.693% | 66.22% | 90.16% | 9.55% | 0.28% | 0.00% | 0.00% |
| Scale-50 | 60 | 19.197 | 28.502 | 13.372% | 69.55% | 79.12% | 19.58% | 1.20% | 0.08% | 0.02% |
| Scale-50 | 90 | 23.501 | 34.646 | 16.385% | 71.36% | 72.86% | 24.84% | 2.06% | 0.22% | 0.03% |
| Scale-50 | 120 | 26.232 | 38.485 | 18.506% | 72.40% | 68.19% | 28.51% | 2.89% | 0.38% | 0.02% |

### Four-horizon summary

| Model | Mean MARD | Mean DTS-A | Mean RMSE | Mean MAE | Mean Direction Accuracy |
|---|---:|---:|---:|---:|---:|
| V15.4 | 16.0237% | 73.2968% | 33.3400 | 23.0156 | 66.7088% |
| Scale-10 | 14.5989% | 76.5254% | 30.8872 | 21.0175 | 69.1653% |
| **Scale-50** | **14.2392%** | **77.5852%** | **30.0121** | **20.2903** | **69.8812%** |

## Improvement vs original V15.4

Scale-50 vs V15.4:

- MARD: 16.0237% -> 14.2392% (**-11.14% relative**);
- RMSE: 33.3400 -> 30.0121 (**-9.98% relative**);
- DTS-A: 73.2968% -> 77.5852% (**+4.29 percentage points**);
- direction accuracy: 66.71% -> 69.88% (**+3.17 percentage points**).

DTS-A improves at every horizon:

- +30: 87.64% -> 90.16%;
- +60: 75.33% -> 79.12%;
- +90: 67.66% -> 72.86%;
- +120: 62.56% -> 68.19%.

## Scaling curve and saturation

The first scaling step produced a large gain:

- V15.4 -> Scale-10: +3.23 pp DTS-A and -1.42 pp absolute MARD.

The second scaling step required roughly 4.67x more training windows than Scale-10 but produced a much smaller gain:

- Scale-10 -> Scale-50: +1.06 pp DTS-A and -0.36 pp absolute MARD.

This is a clear diminishing-return signal.

The experiment therefore supports two conclusions simultaneously:

1. **Training data scale is a validated performance lever.** The original V15.4 model was substantially undertrained relative to the amount of available data.
2. **Pure scaling is now entering saturation.** Further multiplying the number of windows without changing the representation or model is unlikely to close the remaining Top-3 gap efficiently.

## Decision

**Scale-50 becomes the new internal baseline for Top-3 experiments.**

Do not launch Scale-100/Scale-200 as the immediate next experiment. Future work should preserve the Scale-50 dataset size and frozen validation set, while changing one high-upside modeling component at a time.

The next controlled experiment is **Future Intervention Representation**: improve how known future basal, bolus and carbohydrate interventions are encoded, while keeping the Scale-50 training population, targets, validation, architecture and evaluation protocol fixed.

## Reproducibility files

Key files involved in the scaling program:

- `ml/prepare_v15_4_scale10.py`
- `ml/train_v15_4_scale10.py`
- `ml/prepare_v15_4_scale50.py`
- `ml/train_v15_4_scale50.py`
- `ml/check_v15_4_scaled.py`
- `ml/evaluate_v15_4_scaling_metrics.py`

Key commits:

- `2932fe1` — Scale-10 dataset builder
- `e4eb431` — Scale-10 integrity checker
- `b765da9` — Scale-10 trainer
- `43d58c3` — Scale-50 dataset builder
- `1a6ee3c` — reusable scaled-data integrity checker
- `b97181c` — Scale-50 trainer
- `75bbfe6` — competition-aligned scaling evaluator

## Current external reference

The official Live result for the existing V15.4 submission remains separate from this internal scaling experiment. Scale-10 and Scale-50 have not yet been externally submitted, so no Live DTS-A/MARD/RMSE should be claimed for them.

The purpose of this report is internal scientific comparison and experiment tracking.