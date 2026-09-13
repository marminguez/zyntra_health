"""Trajectory fast-screen model for the Zyntra Top-3 program.

Unlike V15.4, which re-encodes a separate future prefix for each horizon, this
model runs one causal recurrent decoder across all 24 future-known 5-minute
steps. The hidden trajectory is read at steps 6/12/18/24 to predict
+30/+60/+90/+120 minutes.

Intermediate decoder states are latent; supervision remains the same frozen
four-horizon targets. No future CGM is accepted as an input.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Lambda
from tensorflow.keras.models import Model

HORIZONS = (30, 60, 90, 120)
READ_STEPS = {30: 6, 60: 12, 90: 18, 120: 24}
FUTURE_STEPS = 24


def build_trajectory_forecaster(sequence_length: int, n_features: int) -> Model:
    history = Input(shape=(sequence_length, n_features), name="metabolic_history")
    h = LSTM(64, return_sequences=True, name="history_lstm_1")(history)
    h = Dropout(0.20, name="history_dropout_1")(h)
    _, history_h, history_c = LSTM(
        32,
        return_state=True,
        name="history_lstm_2",
    )(h)

    future = Input(shape=(FUTURE_STEPS, 6), name="future_known")
    future_projected = Dense(
        32,
        activation="relu",
        name="future_step_projection",
    )(future)

    trajectory = LSTM(
        32,
        return_sequences=True,
        name="causal_future_trajectory_lstm",
    )(
        future_projected,
        initial_state=[history_h, history_c],
    )

    shared_dense = Dense(32, activation="relu", name="shared_dense")
    shared_dropout = Dropout(0.10, name="shared_dropout")

    abs_outputs = []
    delta_outputs = []
    for horizon in HORIZONS:
        idx = READ_STEPS[horizon] - 1
        z = Lambda(
            lambda t, i=idx: t[:, i, :],
            name=f"trajectory_read_{horizon}",
        )(trajectory)
        z = shared_dense(z)
        z = shared_dropout(z)
        abs_outputs.append(Dense(1, name=f"abs_{horizon}")(z))
        delta_outputs.append(Dense(1, name=f"delta_{horizon}")(z))

    outputs = abs_outputs + delta_outputs
    names = [f"abs_{h}" for h in HORIZONS] + [f"delta_{h}" for h in HORIZONS]

    model = Model(
        [history, future],
        outputs,
        name="zyntra_causal_trajectory_forecaster",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={name: tf.keras.losses.Huber() for name in names},
        metrics={
            name: [tf.keras.metrics.MeanAbsoluteError(name="mae")]
            for name in names
        },
    )
    return model
