# Zyntra Scientific Baseline — September 2026

**Version:** 1.1  
**Date:** 2026-09-12  
**Branch:** `feat/v14-multi-horizon-forecasting`

## 1. Purpose

This document is the single scientific baseline for Zyntra. It consolidates the current forecasting evidence, preserves historical baselines, separates internal from external results, and defines what must be true before a new model replaces the current internal baseline.

Zyntra is evolving from glucose prediction toward **personalized metabolic state forecasting**: combining continuous physiological signals to anticipate glucose trajectories and clinically relevant events before they occur.

---

## 2. Current scientific status

### Historical external reference

**V14.5 Live** remains the historical external/Live reference currently documented for the challenge pipeline.

| Horizon | RMSE | MARD | DTS-A |
|---|---:|---:|---:|
| +30 min | 20.57 | 10.79% | 85.7% |
| +60 min | 33.06 | 17.73% | 69.1% |
| +90 min | 40.58 | 22.34% | 59.5% |
| +120 min | 45.24 | 25.20% | 53.1% |

These results must remain clearly separated from later **internal validation** experiments.

### Current internal baseline

**V15.4 Scale-50** is now the internal baseline for Top-3 experiments.

The decision is supported by a controlled scaling experiment in which model architecture, validation set, targets, normalization, future inputs, 50/50 absolute+delta fusion and seed were held fixed. Only the number of training windows per patient was increased.

---

## 3. V15.4 scaling experiment

### Scientific question

Does increasing usable training data still provide enough performance gain to justify pure scaling, or is the V15.4 architecture beginning to saturate?

### Frozen experimental setup

Across original V15.4, Scale-10 and Scale-50:

- same V15.4 architecture;
- same four forecast horizons: +30, +60, +90 and +120 min;
- same history inputs and feature definitions;
- same future-known inputs: basal, bolus and carbs with missingness masks;
- no future CGM input;
- same V14.1 train-only normalization statistics;
- same absolute-head + delta-head design;
- same fixed 0.5 absolute + 0.5 reconstructed-delta fusion;
- same seed: 42;
- same frozen validation set: **10,686 windows**;
- no `test.parquet` targets or Live targets used for fitting or internal evaluation.

The scaled dataset builders copy validation byte-for-byte from `ml/data/v15_master`.

### Training scale

| Experiment | Train cap / patient | Train windows | Relative to V15.4 |
|---|---:|---:|---:|
| V15.4 | 24 | 29,366 | 1.00x |
| Scale-10 | 240 | 285,096 | 9.71x |
| Scale-50 | 1,200 | 1,331,220 | 45.33x |

Scale-50 integrity checks documented in the repository:

- train shards: 1,235;
- train windows: 1,331,220;
- max windows per shard: 1,200 / cap 1,200;
- validation shards: 224;
- validation windows: 10,686;
- validation byte-identical to frozen V15 master: `True`;
- future CGM input: `False`.

---

## 4. Competition-aligned metrics per horizon

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
| **Scale-50** | **30** | **12.231** | **18.415** | **8.693%** | **66.22%** | **90.16%** | **9.55%** | **0.28%** | **0.00%** | **0.00%** |
| **Scale-50** | **60** | **19.197** | **28.502** | **13.372%** | **69.55%** | **79.12%** | **19.58%** | **1.20%** | **0.08%** | **0.02%** |
| **Scale-50** | **90** | **23.501** | **34.646** | **16.385%** | **71.36%** | **72.86%** | **24.84%** | **2.06%** | **0.22%** | **0.03%** |
| **Scale-50** | **120** | **26.232** | **38.485** | **18.506%** | **72.40%** | **68.19%** | **28.51%** | **2.89%** | **0.38%** | **0.02%** |

---

## 5. Four-horizon summary

| Model | Mean MARD | Mean DTS-A | Mean RMSE | Mean MAE | Mean Direction Accuracy |
|---|---:|---:|---:|---:|---:|
| V15.4 | 16.0237% | 73.2968% | 33.3400 | 23.0156 | 66.7088% |
| Scale-10 | 14.5989% | 76.5254% | 30.8872 | 21.0175 | 69.1653% |
| **Scale-50** | **14.2392%** | **77.5852%** | **30.0121** | **20.2903** | **69.8812%** |

### Scale-50 vs original V15.4

- Mean MARD: 16.0237% → **14.2392%** (**-11.14% relative**)
- Mean RMSE: 33.3400 → **30.0121** (**-9.98% relative**)
- Mean DTS-A: 73.2968% → **77.5852%** (**+4.29 pp**)
- Mean direction accuracy: 66.71% → **69.88%** (**+3.17 pp**)

DTS-A improves at every horizon:

- +30: 87.64% → **90.16%**
- +60: 75.33% → **79.12%**
- +90: 67.66% → **72.86%**
- +120: 62.56% → **68.19%**

---

## 6. Scaling curve and saturation

The first increase in scale produced a large gain:

- V15.4 → Scale-10: **+3.23 pp DTS-A** and **-1.42 pp absolute MARD**.

The second increase required roughly **4.67x more training windows** than Scale-10 but produced a smaller incremental gain:

- Scale-10 → Scale-50: **+1.06 pp DTS-A** and **-0.36 pp absolute MARD**.

### Scientific interpretation

The experiment supports two conclusions:

1. **Training scale is a validated performance lever.** Original V15.4 was substantially undertrained relative to the available data.
2. **Pure scaling is entering diminishing returns.** Further multiplying windows without improving representation or model design is unlikely to be the most efficient route to Top-3 performance.

Therefore Scale-100 / Scale-200 should **not** be the immediate next experiment.

---

## 7. Internal baseline decision

### Historical external baseline

**V14.5 Live** remains the historical externally evaluated reference.

### Internal scientific baseline

**V15.4 Scale-50** becomes the new internal baseline for controlled Top-3 experiments.

This wording is intentional: Scale-10 and Scale-50 have **not yet been externally submitted**, so no Live RMSE/MARD/DTS claims should be made for them.

### Baseline promotion rule

A future challenger should only replace Scale-50 if it:

- uses the same frozen validation protocol;
- improves the priority metrics materially;
- does not degrade clinically important subgroups;
- is reproducible from code + config + metadata;
- passes leakage and integrity checks.

---

## 8. V15 scientific target status

The earlier V15 target was:

- +30 RMSE < 20.57, preferably <20;
- +30 MARD <10%;
- no material degradation across longer horizons.

Scale-50 achieves internally:

- **+30 RMSE: 18.415**
- **+30 MARD: 8.693%**
- **+30 DTS-A: 90.16%**

It therefore exceeds the original V15 internal target under the frozen validation protocol.

This does **not** constitute external or clinical validation.

---

## 9. Experiment registry — current decisions

| Experiment | Main learning | Decision |
|---|---|---|
| V14.1 | Historical multi-horizon baseline | Archive |
| V14.4 | Strong internal improvement, especially in low-glucose analysis | Keep as historical reference |
| V14.5 | Strong V14 challenger and Live pipeline basis | Keep |
| V14.5 Live | Historical external/Live reference | Keep as external reference |
| V14.6 | Hybrid/drop-aware line did not become main baseline | Research/archive |
| Top-3 KNN retrieval | Classical nearest-neighbour retrieval not competitive | Drop classical KNN |
| Oracle retrieval | Historical episodes contain potential signal but ranking is weak | Research |
| V15.4 | Stronger representation/model line, undertrained at original scale | Keep as control |
| V15.4 Scale-10 | Scaling clearly improves performance | Archive as scaling evidence |
| **V15.4 Scale-50** | Best controlled internal result; diminishing returns becoming visible | **Internal baseline** |

---

## 10. Next controlled experiment

The next experiment documented in the scaling program is **Future Intervention Representation**.

The principle is to keep the successful Scale-50 experimental foundation fixed while changing one high-upside modeling component at a time.

Focus:

- improve representation of known future basal insulin;
- improve representation of bolus interventions;
- improve carbohydrate intervention encoding;
- preserve Scale-50 training population;
- preserve frozen validation;
- preserve targets and evaluation protocol.

This is preferable to brute-force additional data scaling because the scaling curve is already showing saturation.

---

## 11. Product-science interpretation

The strongest current internal evidence is no longer simply that Zyntra can forecast glucose at +30 minutes.

The Scale-50 results show meaningful predictive signal across all four horizons, with the strongest reliability still at +30 and progressively higher uncertainty at longer horizons.

The product direction remains:

**Data layer → metabolic state representation → forecast → risk intelligence → adaptive alert policy → user experience.**

The research objective is not only lower RMSE. Zyntra ultimately needs predictions that are:

- robust between patients;
- reliable in hypo/hyperglycemic ranges;
- calibrated under rapid changes;
- useful enough to drive low-noise actionable alerts.

---

## 12. Validation gaps

Before stronger scientific or product claims are made, priority validation should include:

1. **Patient-level robustness:** verify that gains are distributed rather than dominated by a subset of subjects.
2. **Clinical subgroup analysis:** hypo, hyper, rapid fall, rapid rise, post-bolus, post-meal and overnight.
3. **Calibration analysis:** understand why scaling improves long-horizon predictions and whether systematic bias remains.
4. **External evaluation:** submit the selected scaled model through the official challenge pipeline before presenting its internal metrics as Live results.
5. **Leakage audit:** continue verifying that model selection and tuning do not use final test/Live targets.

---

## 13. Current headline metrics

### Internal — V15.4 Scale-50

| Metric | Result |
|---|---:|
| +30 RMSE | **18.415** |
| +30 MARD | **8.693%** |
| +30 DTS-A | **90.16%** |
| +60 RMSE | **28.502** |
| +90 RMSE | **34.646** |
| +120 RMSE | **38.485** |
| Mean 4-horizon RMSE | **30.012** |
| Mean 4-horizon MARD | **14.239%** |
| Mean 4-horizon DTS-A | **77.585%** |
| Mean direction accuracy | **69.881%** |

### External historical reference — V14.5 Live

| Metric | Result |
|---|---:|
| +30 RMSE | **20.57** |
| +30 MARD | **10.79%** |
| +30 DTS-A | **85.7%** |

---

## 14. Scientific one-line thesis

> **Zyntra is evolving from glucose prediction toward personalized metabolic state forecasting: combining continuous physiological signals to anticipate glucose trajectories and clinically relevant events before they occur.**

---

## 15. Reproducibility references

Detailed V15.4 scaling documentation: [`V15_4_SCALING_REPORT.md`](./V15_4_SCALING_REPORT.md)

Key scripts documented by the experiment:

- `ml/prepare_v15_4_scale10.py`
- `ml/train_v15_4_scale10.py`
- `ml/prepare_v15_4_scale50.py`
- `ml/train_v15_4_scale50.py`
- `ml/check_v15_4_scaled.py`
- `ml/evaluate_v15_4_scaling_metrics.py`

**Status:** internal scientific baseline updated on 2026-09-12.  
**Internal baseline:** V15.4 Scale-50.  
**External historical reference:** V14.5 Live.  
**Next modeling direction:** Future Intervention Representation.