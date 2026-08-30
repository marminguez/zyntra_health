"""Zyntra V14.1 shared-backbone multi-horizon glucose forecaster."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model

HORIZONS = (30, 60, 90, 120)


def build_v14_forecaster(sequence_length: int, n_features: int) -> Model:
    """One temporal encoder with four glucose-regression heads."""
    inputs = Input(shape=(sequence_length, n_features), name="metabolic_history")
    x = LSTM(64, return_sequences=True, name="lstm_1")(inputs)
    x = Dropout(0.20, name="dropout_1")(x)
    x = LSTM(32, name="lstm_2")(x)
    x = Dense(32, activation="relu", name="shared_dense")(x)
    x = Dropout(0.10, name="dropout_2")(x)
    outputs = [Dense(1, name=f"pred_{h}")(x) for h in HORIZONS]
    model = Model(inputs=inputs, outputs=outputs, name="zyntra_v14_1_forecaster")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={f"pred_{h}": tf.keras.losses.Huber() for h in HORIZONS},
        metrics={f"pred_{h}": [tf.keras.metrics.MeanAbsoluteError(name="mae")] for h in HORIZONS},
    )
    return model
