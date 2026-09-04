"""Zyntra V15.1 conditional forecaster with known future insulin.

Controlled extension of V14.5: the historical encoder and paired absolute/delta
supervision are retained, while a second encoder receives only insulin values
for t+5..t+120. No future CGM or other future covariates are inputs.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model

HORIZONS = (30, 60, 90, 120)
FUTURE_STEPS = 24


def build_v15_1_forecaster(sequence_length: int, n_features: int) -> Model:
    history = Input(shape=(sequence_length, n_features), name="metabolic_history")
    h = LSTM(64, return_sequences=True, name="history_lstm_1")(history)
    h = Dropout(0.20, name="history_dropout_1")(h)
    h = LSTM(32, name="history_lstm_2")(h)

    future_insulin = Input(shape=(FUTURE_STEPS, 2), name="future_insulin")
    # Channels: normalized insulin value and explicit missingness indicator.
    f = LSTM(16, name="future_insulin_lstm")(future_insulin)

    x = Concatenate(name="history_future_fusion")([h, f])
    x = Dense(32, activation="relu", name="shared_dense")(x)
    x = Dropout(0.10, name="dropout_2")(x)

    abs_outputs = [Dense(1, name=f"abs_{horizon}")(x) for horizon in HORIZONS]
    delta_outputs = [Dense(1, name=f"delta_{horizon}")(x) for horizon in HORIZONS]
    outputs = abs_outputs + delta_outputs

    model = Model(
        inputs=[history, future_insulin],
        outputs=outputs,
        name="zyntra_v15_1_future_insulin_forecaster",
    )
    names = [f"abs_{h}" for h in HORIZONS] + [f"delta_{h}" for h in HORIZONS]
    losses = {name: tf.keras.losses.Huber() for name in names}
    metrics = {name: [tf.keras.metrics.MeanAbsoluteError(name="mae")] for name in names}
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=losses,
        metrics=metrics,
    )
    return model
