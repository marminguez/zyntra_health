"""V15.4 ablation with independent dense towers per forecast horizon.

Controlled change vs V15.4:
- history encoder unchanged
- future-known encoder unchanged
- original Huber losses unchanged
- only the shared Dense(32)+Dropout(0.10) representation is replaced by
  one independent Dense(32)+Dropout(0.10) tower for each horizon
- final absolute/delta heads remain horizon-specific
"""
from __future__ import annotations

import tensorflow as tf

HORIZONS = (30, 60, 90, 120)
PREFIX_STEPS = (6, 12, 18, 24)


def build_v15_4_horizon_towers_forecaster(sequence_length: int, n_features: int):
    history_input = tf.keras.Input(
        shape=(sequence_length, n_features), name="history"
    )
    h = tf.keras.layers.LSTM(64, return_sequences=True, name="history_lstm_1")(
        history_input
    )
    h = tf.keras.layers.Dropout(0.20, name="history_dropout")(h)
    history_encoding = tf.keras.layers.LSTM(32, name="history_lstm_2")(h)

    future_input = tf.keras.Input(shape=(24, 6), name="future_known")
    future_lstm = tf.keras.layers.LSTM(16, name="shared_future_lstm")

    absolute_outputs = []
    delta_outputs = []

    for horizon, prefix in zip(HORIZONS, PREFIX_STEPS):
        future_prefix = tf.keras.layers.Lambda(
            lambda z, p=prefix: z[:, :p, :],
            name=f"future_prefix_{horizon}",
        )(future_input)
        future_encoding = future_lstm(future_prefix)
        combined = tf.keras.layers.Concatenate(name=f"combined_{horizon}")(
            [history_encoding, future_encoding]
        )

        # Experimental change: no Dense representation is shared across horizons.
        tower = tf.keras.layers.Dense(
            32, activation="relu", name=f"tower_dense_{horizon}"
        )(combined)
        tower = tf.keras.layers.Dropout(
            0.10, name=f"tower_dropout_{horizon}"
        )(tower)

        absolute_outputs.append(
            tf.keras.layers.Dense(1, name=f"absolute_{horizon}")(tower)
        )
        delta_outputs.append(
            tf.keras.layers.Dense(1, name=f"delta_{horizon}")(tower)
        )

    model = tf.keras.Model(
        inputs=[history_input, future_input],
        outputs=absolute_outputs + delta_outputs,
        name="zyntra_v15_4_horizon_towers",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=[tf.keras.losses.Huber() for _ in range(8)],
        metrics=[[tf.keras.metrics.MeanAbsoluteError()] for _ in range(8)],
    )
    return model
