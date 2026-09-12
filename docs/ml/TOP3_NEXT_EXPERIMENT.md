# Zyntra Top-3 Program — Next Controlled Experiment

Date: 2026-09-12
Status: planned, not yet implemented
Baseline: V15.4 Scale-50

## Strategic target

The objective is not incremental leaderboard movement. The modeling program is aimed at reaching competitive Top-3 territory, with DTS Error Grid Zone A as the main clinical metric and MARD/RMSE as complementary analytical metrics.

The Scale-50 experiment established a stronger internal baseline:

- mean DTS-A: 77.5852%;
- mean MARD: 14.2392%;
- mean RMSE: 30.0121;
- mean direction accuracy: 69.8812%.

The +30 minute horizon is already comparatively strong at 90.16% DTS-A. The remaining weakness is concentrated increasingly at longer horizons:

- +60: 79.12% DTS-A;
- +90: 72.86% DTS-A;
- +120: 68.19% DTS-A.

This motivates an intervention-aware experiment before a larger architectural rewrite.

## Hypothesis

Raw 5-minute future basal/bolus/carbohydrate samples force the model to learn useful pharmacological and meal-event summaries implicitly.

At +60 to +120 minutes, prediction quality may improve if the model receives explicit causal summaries of known future interventions, while still respecting the competition rule that future CGM targets are never used as input.

## Controlled experiment: Future Intervention Representation

Keep frozen:

- Scale-50 training population and sampling policy;
- 1,331,220 training windows;
- frozen 10,686-window validation set;
- history encoder and V15.4 architecture for the first ablation;
- four targets (+30/+60/+90/+120);
- V14.1 normalization policy where applicable;
- absolute + delta heads;
- fixed 50/50 fusion;
- seed and training/evaluation protocol;
- no future CGM.

Change only the representation of known future interventions.

### Candidate derived future features

For each prediction horizon/prefix, derive features from future-known basal, bolus and carbohydrate events such as:

1. cumulative bolus/insulin through the target horizon;
2. cumulative carbohydrates through the target horizon;
3. time to next bolus;
4. magnitude of next bolus;
5. bolus event count;
6. time to next carbohydrate event;
7. magnitude of next carbohydrate event;
8. carbohydrate event count;
9. current/future basal level and basal change summaries;
10. approximate future insulin-action / IOB-style summary;
11. approximate carbohydrate-action summary;
12. simple carbohydrate-to-insulin relationship summaries when both are known.

The exact feature set must be documented before training and generated only from information legally available at the prediction anchor and known future intervention stream. No future CGM value may enter any derived feature.

## Experimental discipline

This should remain an ablation, not a bundle of simultaneous architecture changes.

First test whether better intervention representation improves Scale-50 with the existing V15.4 architecture. Only after measuring this experiment should the program consider a trajectory decoder, source-aware modeling, patient personalization or other structural changes.

## Success criteria

The most important signal is improvement at +60/+90/+120 without sacrificing +30 performance.

A meaningful result should show consistent gains across multiple metrics, especially DTS-A and MARD, rather than a tiny change in one aggregate metric.

If the intervention representation produces a material jump, it becomes part of the new baseline before testing a trajectory model. If gains are marginal, the next major lever should be architecture rather than further pure scaling.

## Next architecture candidate if representation alone is insufficient

A trajectory model would predict the glucose path from +5 through +120 minutes rather than four largely independent horizons. Conceptually:

- history encoder;
- future-intervention encoder;
- sequence/trajectory decoder;
- read predictions at +30/+60/+90/+120.

This is intentionally deferred until the intervention-representation ablation is complete.

## Deferred / standby ideas

Foundation-model approaches such as GlucoFM or TimesFM remain standby options rather than part of this experiment. They should not be mixed into the controlled intervention-representation ablation.

Likewise, pure Scale-100/Scale-200 training is deferred because Scale-10 -> Scale-50 showed clear diminishing returns.

## Baseline reference

See `docs/ml/V15_4_SCALING_REPORT.md` for the complete scaling experiment, integrity controls, metrics and decision rationale.