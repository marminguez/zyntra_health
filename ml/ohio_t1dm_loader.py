"""OhioT1DM XML loader for Zyntra multi-patient experiments."""
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
TS_FORMAT="%d-%m-%Y %H:%M:%S"; FIVE_MIN="5min"
def _dt(v): return pd.to_datetime(v,format=TS_FORMAT,errors="coerce")
def _float(v,default=np.nan):
    try:return float(v)
    except (TypeError,ValueError):return default
def _events(root,tag):
    node=root.find(tag); return list(node) if node is not None else []
def _series_events(root,tag,value_attr="value"):
    rows=[]
    for event in _events(root,tag):
        ts=_dt(event.attrib.get("ts")); value=_float(event.attrib.get(value_attr))
        if pd.notna(ts) and np.isfinite(value): rows.append((ts,value))
    return pd.DataFrame(rows,columns=["timestamp","value"])
def _aggregate(frame,index,column):
    if frame.empty:return pd.Series(0.,index=index,name=column)
    frame=frame.copy(); frame["bucket"]=frame.timestamp.dt.floor(FIVE_MIN)
    return frame.groupby("bucket").value.sum().reindex(index).fillna(0.).rename(column)
def load_ohio_xml(path):
    path=Path(path); root=ET.parse(path).getroot(); pid=int(root.attrib["id"]); source="training" if "training" in path.name else "testing"
    glucose=_series_events(root,"glucose_level")
    if glucose.empty: raise ValueError(f"No CGM data found in {path}")
    glucose["bucket"]=glucose.timestamp.dt.floor(FIVE_MIN); glucose=glucose.groupby("bucket").value.mean()
    index=pd.date_range(glucose.index.min(),glucose.index.max(),freq=FIVE_MIN); df=pd.DataFrame(index=index); df.index.name="timestamp"
    df["glucose"]=glucose.reindex(index).interpolate(method="time",limit=3,limit_area="inside")
    bolus=[]
    for e in _events(root,"bolus"):
        ts=_dt(e.attrib.get("ts_begin"))
        if pd.notna(ts): bolus.append((ts,_float(e.attrib.get("dose"),0.)))
    df["bolus"]=_aggregate(pd.DataFrame(bolus,columns=["timestamp","value"]),index,"bolus")
    meals=[]
    for e in _events(root,"meal"):
        ts=_dt(e.attrib.get("ts"))
        if pd.notna(ts): meals.append((ts,_float(e.attrib.get("carbs"),0.)))
    df["carbs_g"]=_aggregate(pd.DataFrame(meals,columns=["timestamp","value"]),index,"carbs_g")
    basal=_series_events(root,"basal")
    df["basal_rate"]=np.nan if basal.empty else basal.sort_values("timestamp").set_index("timestamp").value.reindex(index,method="ffill")
    df["exercise_intensity"]=0.
    for e in _events(root,"exercise"):
        start=_dt(e.attrib.get("ts")); duration=_float(e.attrib.get("duration"),0.); intensity=_float(e.attrib.get("intensity"),0.)
        if pd.notna(start) and duration>0: df.loc[(df.index>=start)&(df.index<=start+pd.Timedelta(minutes=duration)),"exercise_intensity"]=intensity
    df["p_id"]=pid; df["source_split"]=source; df["weight"]=_float(root.attrib.get("weight")); df["insulin_type"]=root.attrib.get("insulin_type","")
    df["glucose_delta_5m"]=df.glucose.diff(); df["glucose_delta_15m"]=df.glucose.diff(3); df["glucose_delta_30m"]=df.glucose.diff(6)
    df["glucose_acceleration_15m"]=df.glucose_delta_15m-(df.glucose.shift(3)-df.glucose.shift(6)); df["iob_simple"]=df.bolus.rolling(48,min_periods=1).sum()
    df["glucose_future"]=df.glucose.shift(-6); df["target_hypo"]=(df.glucose_future<70).astype(int)
    return df.dropna(subset=["glucose","glucose_future"])
def load_ohio_directory(data_dir):
    files=sorted(Path(data_dir).glob("*-ws-*.xml"))
    if not files: raise FileNotFoundError(f"No OhioT1DM XML files found in {data_dir}")
    return pd.concat([load_ohio_xml(p) for p in files]).sort_index()
def dataset_summary(df):
    return pd.DataFrame([{"p_id":int(pid),"rows":len(f),"hypo_samples":int((f.glucose<70).sum()),"min_glucose":float(f.glucose.min()),"max_glucose":float(f.glucose.max()),"positive_targets_30m":int(f.target_hypo.sum())} for pid,f in df.groupby("p_id")])
