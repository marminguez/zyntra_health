"""Leakage-safe, event-aware training/evaluation pipeline for Zyntra hypoglycemia LSTM."""
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

from event_evaluation import evaluate_events
from lstm_hypoglycemia_classifier import build_hypoglycemia_classifier, create_sequences_for_classification, find_optimal_threshold, process_full_data

ML_DIR = Path(__file__).resolve().parent
MODELS_DIR = ML_DIR / "models"
MODEL_PATH = MODELS_DIR / "lstm_hypoglycemia_classifier_v2.h5"
THRESHOLD_PATH = MODELS_DIR / "optimal_threshold_v2.npy"
SCALER_PATH = MODELS_DIR / "lstm_feature_scaler_v2.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "lstm_feature_names_v2.json"
EVALUATION_PATH = MODELS_DIR / "evaluation_v2.json"
LOOKBACK = 48
BETA = 5.0
EXCLUDED_COLUMNS = ["glucose_future", "target_hypo", "p_id"]


def json_default(value):
    """Convert numpy/pandas scalar values used in evaluation output to JSON-safe Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def chronological_split(df, train_fraction=0.70, val_fraction=0.15):
    n = len(df)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def split_dataset(df):
    patient_ids = [int(x) for x in sorted(df["p_id"].unique())]
    if len(patient_ids) >= 3:
        rng = np.random.default_rng(42)
        ids = np.asarray(patient_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15))
        if n_train + n_val >= n:
            n_train, n_val = n - 2, 1
        train_ids = [int(x) for x in ids[:n_train]]
        val_ids = [int(x) for x in ids[n_train:n_train+n_val]]
        test_ids = [int(x) for x in ids[n_train+n_val:]]
        return df[df.p_id.isin(train_ids)].copy(), df[df.p_id.isin(val_ids)].copy(), df[df.p_id.isin(test_ids)].copy(), {"strategy":"patient_disjoint","train_ids":train_ids,"val_ids":val_ids,"test_ids":test_ids}
    if len(patient_ids) == 2:
        train_val = df[df.p_id == patient_ids[0]].copy()
        test_df = df[df.p_id == patient_ids[1]].copy()
        train_df, val_df, _ = chronological_split(train_val, 0.80, 0.20)
        return train_df, val_df, test_df, {"strategy":"hybrid_two_patient","train_ids":[patient_ids[0]],"val_ids":[patient_ids[0]],"test_ids":[patient_ids[1]]}
    train_df, val_df, test_df = chronological_split(df)
    return train_df, val_df, test_df, {"strategy":"chronological_single_patient","train_ids":patient_ids,"val_ids":patient_ids,"test_ids":patient_ids,"warning":"Single-patient evaluation does not demonstrate generalization to unseen patients."}


def fit_and_apply_scaler(train_df, val_df, test_df, feature_names):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_names])
    scaled = []
    for frame in (train_df, val_df, test_df):
        out = frame.copy()
        out[feature_names] = scaler.transform(frame[feature_names])
        scaled.append(out)
    return (*scaled, scaler)


def frames_to_sequences(frame, include_metadata=False):
    xs, ys, metas = [], [], []
    for pid in sorted(frame["p_id"].unique()):
        patient = frame[frame["p_id"] == pid].sort_index()
        x, y = create_sequences_for_classification(patient, lookback=LOOKBACK)
        if not len(x):
            continue
        xs.append(x); ys.append(y)
        if include_metadata:
            target_rows = patient.iloc[LOOKBACK:].copy()
            metas.append(pd.DataFrame({"p_id":target_rows["p_id"].values,"timestamp":target_rows.index}))
    if not xs:
        empty_x = np.empty((0, LOOKBACK, 0)); empty_y = np.empty((0,), dtype=int)
        return (empty_x, empty_y, pd.DataFrame()) if include_metadata else (empty_x, empty_y)
    x_all, y_all = np.concatenate(xs), np.concatenate(ys)
    return (x_all, y_all, pd.concat(metas, ignore_index=True)) if include_metadata else (x_all, y_all)


def attach_unscaled_metadata(sequence_meta, unscaled_test_df):
    source = unscaled_test_df.reset_index().rename(columns={unscaled_test_df.index.name or "index":"timestamp"})
    source["timestamp"] = pd.to_datetime(source["timestamp"])
    sequence_meta = sequence_meta.copy(); sequence_meta["timestamp"] = pd.to_datetime(sequence_meta["timestamp"])
    merged = sequence_meta.merge(source[["p_id","timestamp","glucose"]], on=["p_id","timestamp"], how="left")
    if merged["glucose"].isna().any(): raise RuntimeError("Could not align original glucose metadata with test sequences.")
    return merged


def calculate_metrics(model, x_test, y_test, threshold):
    probabilities = model.predict(x_test, verbose=0).flatten(); predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0,1]).ravel()
    auc = roc_auc_score(y_test, probabilities) if len(np.unique(y_test)) > 1 else None
    return {"roc_auc":auc,"recall_sensitivity":tp/(tp+fn) if tp+fn else 0.0,"precision_ppv":tp/(tp+fp) if tp+fp else 0.0,"specificity":tn/(tn+fp) if tn+fp else 0.0,"npv":tn/(tn+fn) if tn+fn else 0.0,"f1":f1_score(y_test,predictions,zero_division=0),"confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},"test_samples":int(len(y_test)),"test_positive_samples":int(y_test.sum())}, probabilities


def main():
    print("\n=== ZYNTRA LSTM V2: LEAKAGE-SAFE + EVENT EVALUATION ===")
    df = process_full_data(); feature_names = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    train_raw, val_raw, test_raw, split_info = split_dataset(df)
    print(f"Split strategy: {split_info['strategy']}")
    if split_info.get("warning"): print(f"WARNING: {split_info['warning']}")
    train_df, val_df, test_df, scaler = fit_and_apply_scaler(train_raw,val_raw,test_raw,feature_names)
    x_train,y_train = frames_to_sequences(train_df); x_val,y_val = frames_to_sequences(val_df); x_test,y_test,test_meta = frames_to_sequences(test_df,include_metadata=True)
    test_meta = attach_unscaled_metadata(test_meta,test_raw)
    if min(len(x_train),len(x_val),len(x_test)) == 0: raise RuntimeError("Not enough data to create train/validation/test sequences.")
    present_classes=np.unique(y_train); weights=compute_class_weight("balanced",classes=present_classes,y=y_train); class_weights={int(c):float(w) for c,w in zip(present_classes,weights)}
    model=build_hypoglycemia_classifier(LOOKBACK,len(feature_names)); model.compile(optimizer="adam",loss="binary_crossentropy",metrics=[tf.keras.metrics.Precision(name="precision"),tf.keras.metrics.Recall(name="recall"),tf.keras.metrics.AUC(name="auc")])
    model.fit(x_train,y_train,validation_data=(x_val,y_val),class_weight=class_weights,epochs=50,batch_size=64,callbacks=[EarlyStopping(monitor="val_auc",patience=15,restore_best_weights=True,mode="max")],verbose=1)
    threshold=float(find_optimal_threshold(model,x_val,y_val,beta=BETA)); sample_metrics,probabilities=calculate_metrics(model,x_test,y_test,threshold); event_metrics=evaluate_events(test_meta,probabilities,threshold,horizon_minutes=30)
    MODELS_DIR.mkdir(parents=True,exist_ok=True); model.save(MODEL_PATH); np.save(THRESHOLD_PATH,threshold)
    with open(SCALER_PATH,"wb") as fh: pickle.dump(scaler,fh)
    with open(FEATURE_NAMES_PATH,"w",encoding="utf-8") as fh: json.dump(feature_names,fh,indent=2)
    report={"model":"lstm_hypoglycemia_classifier_v2","prediction_horizon_minutes":30,"lookback_minutes":LOOKBACK*5,"threshold":threshold,"threshold_metric":f"F{BETA}","split":split_info,"patients_total":int(df.p_id.nunique()),"rows":{"train":len(train_raw),"validation":len(val_raw),"test":len(test_raw)},"sequences":{"train":len(x_train),"validation":len(x_val),"test":len(x_test)},"sample_metrics":sample_metrics,"event_metrics":event_metrics,"limitations":["Current demo contains too few patients for robust external generalization.","IOB is still a simple rolling bolus sum and should be replaced by an insulin-action model.","Event metrics require validation on a multi-patient dataset before clinical interpretation."]}
    with open(EVALUATION_PATH,"w",encoding="utf-8") as fh: json.dump(report,fh,indent=2,default=json_default)
    print(json.dumps(report,indent=2,default=json_default)); print(f"Evaluation saved to {EVALUATION_PATH}")


if __name__ == "__main__": main()
