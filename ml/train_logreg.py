"""
Trains an XGBoost deterioration risk model on the synthetic T1DM dataset.
Features: z-scores of 5 signal domains.
Target: binary deterioration in next 24h.
Outputs: ml/model.pkl (joblib) + ml/model_meta.json
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression

FEATURES = ["z_glucose", "z_hrv", "z_sleep_score", "z_steps", "z_adherence"]
TARGET   = "label_24h"

print("Loading synthetic dataset...")
df = pd.read_parquet("ml/synthetic_t1dm.parquet")
df = df.dropna(subset=FEATURES)

X = df[FEATURES].values
y = df[TARGET].values
groups = df["patient_id"].values

# Patient-aware split: train on 10 patients, test on 2
splitter = GroupShuffleSplit(n_splits=1, test_size=2/12, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Positive rate train: {np.mean(y_train):.1%} | test: {np.mean(y_test):.1%}")

# Scale + LogReg
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        random_state=42,
    )),
])

print("Training Logistic Regression...")
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"\nTest AUC: {auc:.4f}")
print(classification_report(y_test, (y_prob > 0.5).astype(int),
                             target_names=["stable", "deterioration"]))

# Save model
Path("ml").mkdir(exist_ok=True)
joblib.dump(model, "ml/model.pkl")

# Save metadata for the API
meta = {
    "features": FEATURES,
    "auc": float(np.round(auc, 4)),
    "threshold": 0.5,
    "model_type": "logistic_regression",
    "dataset": "synthetic_ohio_calibrated",
    "version": "1.0.0",
}
with open("ml/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved ml/model.pkl and ml/model_meta.json")
