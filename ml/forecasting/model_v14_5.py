"""Zyntra V14.5 hybrid absolute + delta multi-horizon forecaster."""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model

HORIZONS = (30, 60, 90, 120)


def build_v14_5_forecaster(sequence_length: int, n_features: int) -> Model:
    """V14.1 temporal backbone with paired absolute and delta heads.

    The backbone capacity is intentionally unchanged. Each horizon receives an
    absolute-glucose head and a glucose-change head. Both are supervised; no
    extreme weighting is used in this ablation.
    """
    inputs = Input(shape=(sequence_length, n_features), name="metabolic_history")
    x = LSTM(64, return_sequences=True, name="lstm_1")(inputs)
    x = Dropout(0.20, name="dropout_1")(x)
    x = LSTM(32, name="lstm_2")(x)
    x = Dense(32, activation="relu", name="shared_dense")(x)
    x = Dropout(0.10, name="dropout_2")(x)
    abs_outputs = [Dense(1, name=f"abs_{h}")(x) for h in HORIZONS]
    delta_outputs = [Dense(1, name=f"delta_{h}")(x) for h in HORIZONS]
    outputs = abs_outputs + delta_outputs
    model = Model(inputs=inputs, outputs=outputs, name="zyntra_v14_5_hybrid_forecaster")
    losses = {name: tf.keras.losses.Huber() for name in [f"abs_{h}" for h in HORIZONS] + [f"delta_{h}" for h in HORIZONS]}
    metrics = {name: [tf.keras.metrics.MeanAbsoluteError(name="mae")] for name in losses}
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses, metrics=metrics)
    return model
