# Private LSTM validation datasets

Keep patient/raw validation data out of git. This directory documents the expected local layout for verifying the bundled `ml/lstm_hypoglycemia_classifier.h5` model.

Recommended file:

```text
data/private/lstm_hypoglycemia_holdout.json
```

Expected JSON shape:

```json
{
  "sequences": [
    [
      [0.0, 0.0, 0.0, 0.0, 0.0]
    ]
  ],
  "labels": [0]
}
```

Each sequence must contain exactly 48 rows, one row per 5-minute interval, and each row must use this feature order:

1. `glucose`
2. `bolus`
3. `carbs_g`
4. `step_count`
5. `iob`

Use the same holdout split that produced the original 72% accuracy when you need to confirm the model has not regressed:

```bash
npm run ml:lstm:evaluate -- --dataset data/private/lstm_hypoglycemia_holdout.json
```

The `.h5` model is enough for inference, but this private holdout dataset is required to reproduce or guard the reported accuracy metric.
