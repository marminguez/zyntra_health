"""Deterministic patient-level development splits for Zyntra V14.1.

Internal validation is restricted to subjects whose MetaboNet
subject_split_across_traintest flag is false. This prevents calling a subject
'unseen' when MetaboNet explicitly indicates that the subject spans its public
train/test files.
"""
from __future__ import annotations

import hashlib


def stable_bucket(source_file: str, patient_id: str, modulo: int = 100) -> int:
    key = f"{source_file}::{patient_id}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16) % modulo


def development_split(source_file: str, patient_id: str, split_across_train_test: bool) -> str:
    if split_across_train_test:
        return "metabonet_overlap_excluded"
    bucket = stable_bucket(source_file, patient_id)
    if bucket < 15:
        return "validation"
    return "train"
