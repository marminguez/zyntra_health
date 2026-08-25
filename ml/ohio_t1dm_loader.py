"""OhioT1DM XML loader for Zyntra multi-patient experiments.

Converts OhioT1DM train/test XML files into a common 5-minute dataframe with
patient id, CGM, bolus, meal carbs and wearable/context signals where present.
The loader keeps the original source split for auditing, but downstream model
validation should use patient-disjoint splits rather than the source train/test
labels when measuring generalization.
"""
from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

TS_FORMAT = "%d-%m-%Y %H:%M:%S"
FIVE_MIN = "5min"


def _dt(value):
    return pd.to_datetime(value, format=TS_FORMAT, errors="coerce")


def _float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _events(root, tag):
    node = root.find(tag)
    return list(node) if node is not None else []


def _series_events(root, tag, value_attr="value"):
    rows = []
    for event in _events(root, tag):
        ts = _dt(event.attrib.get("ts"))
        value = _float(event.attrib.get(value_attr))
        if pd.notna(ts) and np.isfinite(value):
            rows.append((ts, value))
    return pd.DataFrame(rows, columns=["timestamp", "value"])


def _aggregate_point_events(frame, index, column, agg="sum"):
    if frame.empty:
        return pd.Series(0.0, index=index, name=column)
    frame = frame.copy()
    frame["bucket"] = frame["timestamp"].dt.floor(FIVE_MIN)
    grouped = frame.groupby("bucket")["value"]
    values = grouped.sum() if agg == "sum" else grouped.mean()
    return values.reindex(index).fillna(0.0).rename(column)


def load_ohio_xml(path):
    path = Path(path)
    root = ET.parse(path).getroot()
    patient_id = int(root.attrib["id"])
    source_split = "training" if "training" in path.name else "testing"

    glucose = _series_events(root, "glucose_level")
    if glucose.empty:
        raise ValueError(f"No CGM data found in {path}")
    glucose = glucose.rename(columns={"value": "glucose"}).sort_values("timestamp")
    glucose["bucket"] = glucose["timestamp"].dt.floor(FIVE_MIN)
    glucose = glucose.groupby("bucket")["glucose"].mean()

    index = pd.date_range(glucose.index.min(), glucose.index.max(), freq=FIVE_MIN)
    df = pd.DataFrame(index=index)
    df.index.name = "timestamp"
    # Interpolate only short CGM gaps; long gaps remain missing and are dropped later.
    df["glucose"] = glucose.reindex(index).interpolate(method="time", limit=3, limit_area="inside")

    bolus_rows = []
    for event in _events(root, "bolus"):
        ts = _dt(event.attrib.get("ts_begin"))
        if pd.notna(ts):
            bolus_rows.append((ts, _float(event.attrib.get("dose"), 0.0)))
    bolus = pd.DataFrame(bolus_rows, columns=["timestamp", "value"])
    df["bolus"] = _aggregate_point_events(bolus, index, "bolus")

    meal_rows = []
    for event in _events(root, "meal"):
        ts = _dt(event.attrib.get("ts"))
        if pd.notna(ts):
            meal_rows.append((ts, _float(event.attrib.get("carbs"), 0.0)))
    meals = pd.DataFrame(meal_rows, columns=["timestamp", "value"])
    df["carbs_g"] = _aggregate_point_events(meals, index, "carbs_g")

    for xml_tag, column in [
        ("basis_steps", "steps"),
        ("basis_heart_rate", "heart_rate"),
        ("basis_gsr", "gsr"),
        ("basis_skin_temperature", "skin_temperature"),
        ("basis_air_temperature", "air_temperature"),
    ]:
        values = _series_events(root, xml_tag)
        if values.empty:
            df[column] = np.nan
        else:
            values["bucket"] = values["timestamp"].dt.floor(FIVE_MIN)
            df[column] = values.groupby("bucket")["value"].mean().reindex(index)

    # Scheduled basal: forward-fill the latest basal rate.
    basal = _series_events(root, "basal")
    if basal.empty:
        df["basal_rate"] = np.nan
    else:
        basal = basal.sort_values("timestamp").set_index("timestamp")["value"]
        df["basal_rate"] = basal.reindex(index, method="ffill")

    # Exercise intensity active during the reported duration.
    df["exercise_intensity"] = 0.0
    for event in _events(root, "exercise"):
        start = _dt(event.attrib.get("ts"))
        duration = _float(event.attrib.get("duration"), 0.0)
        intensity = _float(event.attrib.get("intensity"), 0.0)
        if pd.notna(start) and duration > 0:
            end = start + pd.Timedelta(minutes=duration)
            df.loc[(df.index >= start) & (df.index <= end), "exercise_intensity"] = intensity

    df["p_id"] = patient_id
    df["source_split"] = source_split
    df["weight"] = _float(root.attrib.get("weight"))
    df["insulin_type"] = root.attrib.get("insulin_type", "")

    # Causal CGM dynamics used by Hypo V5.
    df["glucose_delta_5m"] = df["glucose"].diff(1)
    df["glucose_delta_15m"] = df["glucose"].diff(3)
    df["glucose_delta_30m"] = df["glucose"].diff(6)
    previous_15 = df["glucose"].shift(3) - df["glucose"].shift(6)
    df["glucose_acceleration_15m"] = df["glucose_delta_15m"] - previous_15

    # Initial baseline IOB; replaced in a later ablation with physiological insulin action.
    df["iob_simple"] = df["bolus"].rolling(48, min_periods=1).sum()

    # 30-minute prediction target.
    df["glucose_future"] = df["glucose"].shift(-6)
    df["target_hypo"] = (df["glucose_future"] < 70).astype(int)

    return df.dropna(subset=["glucose", "glucose_future"])


def load_ohio_directory(data_dir):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*-ws-*.xml"))
    if not files:
        raise FileNotFoundError(f"No OhioT1DM XML files found in {data_dir}")
    frames = [load_ohio_xml(path) for path in files]
    return pd.concat(frames).sort_index()


def dataset_summary(df):
    summary = []
    for pid, frame in df.groupby("p_id"):
        summary.append({
            "p_id": int(pid),
            "rows": int(len(frame)),
            "hypo_samples": int((frame["glucose"] < 70).sum()),
            "min_glucose": float(frame["glucose"].min()),
            "max_glucose": float(frame["glucose"].max()),
            "positive_targets_30m": int(frame["target_hypo"].sum()),
        })
    return pd.DataFrame(summary)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    data = load_ohio_directory(args.data_dir)
    print(dataset_summary(data).to_string(index=False))
