"""Fast CI tests for V6 LOPO invariants; no OhioT1DM raw data required."""
import pandas as pd
from lopo_hypo_v6 import FEATURES, aggregate, choose_validation_patient

def test_features_do_not_include_patient_identity_or_future():
    forbidden={"p_id","glucose_future","target_hypo","source_split"}
    assert forbidden.isdisjoint(FEATURES)

def test_validation_patient_is_not_test_patient():
    ids=[559,563,570,575,588]
    assert choose_validation_patient(ids,591) in ids

def test_aggregate_pooled_recall():
    folds=[{"hypoglycemia_events":10,"detected_events":8,"event_recall":.8,"false_alerts_per_patient_day":1.0,"median_warning_minutes":20},
           {"hypoglycemia_events":5,"detected_events":5,"event_recall":1.0,"false_alerts_per_patient_day":.5,"median_warning_minutes":25}]
    out=aggregate(folds)
    assert out["pooled_events"]==15
    assert out["pooled_detected"]==13
    assert abs(out["pooled_event_recall"]-13/15)<1e-9
