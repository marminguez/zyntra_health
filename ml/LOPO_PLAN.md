# Hypo V6 — Multi-Patient Generalization

## Objective
Measure whether Zyntra's hypoglycemia predictor generalizes to a patient never seen during training.

## Dataset
OhioT1DM XML files for patients 559, 563, 570, 575, 588 and 591.

## Evaluation protocol
Use Leave-One-Patient-Out (LOPO) cross-validation:

1. Hold out one patient as test.
2. Train only on the remaining patients.
3. Select the operating threshold without using the held-out patient.
4. Evaluate the held-out patient with the strict event-aware 30-minute pre-onset definition.
5. Repeat until every patient has served once as the unseen test patient.

## Per-patient outputs
- ROC AUC (diagnostic; secondary to event metrics)
- hypoglycemia event count
- detected events
- event recall
- median warning time
- false alert episodes
- false alerts per patient-day
- near-miss alerts
- clear false-positive alerts

## Aggregate outputs
Report mean, median, standard deviation, minimum and maximum across held-out patients. Also report the pooled number of detected/missed events.

## Research target
Experimental product target only; not clinically validated:
- event recall >= 80%
- false alerts <= 1 per patient-day
- median warning >= 15 minutes

## Leakage rules
- No rows from the held-out patient may be used to fit the scaler, model or threshold.
- Feature engineering must be causal and use only information available at prediction time.
- Threshold selection must never use held-out test performance.
- Patient identity must not be used as a predictive feature.

## V6 baseline
Start with CGM + causal glucose dynamics so the multi-patient result is comparable to Hypo V5. Add physiological IOB, meals and activity only in later ablations after the baseline is measured.
