"""Scale-10 control: Zyntra V15.4 plus a separate 24h CGM-only encoder.

Important: this is NOT GlucoFM. Zyntra already consumes 24h multivariable history.
This control tests whether a dedicated CGM-only temporal branch adds value before
we have an official frozen GlucoFM checkpoint.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, LSTM, Lambda
from tensorflow.keras.models import Model

HORIZONS = (30, 60, 90, 120)
PREFIX_STEPS = {30: 6, 60: 12, 90: 18, 120: 24}
FUTURE_STEPS = 24
CGM_STEPS = 288


def build_cgm24_aux_forecaster(sequence_length: int, n_features: int) -> Model:
    history = Input(shape=(sequence_length, n_features), name="metabolic_history")
    h = LSTM(64, return_sequences=True, name="history_lstm_1")(history)
    h = Dropout(0.20, name="history_dropout_1")(h)
    h = LSTM(32, name="history_lstm_2")(h)

    future = Input(shape=(FUTURE_STEPS, 6), name="future_known")
    future_encoder = LSTM(16, name="future_known_lstm_shared")

    # Dedicated CGM-only branch. No new information is introduced: this is an
    # architectural control for the future frozen-embedding fusion seam.
    cgm24 = Input(shape=(CGM_STEPS, 1), name="cgm_24h")
    c = LSTM(16, name="cgm24_lstm")(cgm24)
    c = Dropout(0.10, name="cgm24_dropout")(c)

    shared_dense = Dense(32, activation="relu", name="shared_dense")
    shared_dropout = Dropout(0.10, name="shared_dropout")

    abs_outputs = []
    delta_outputs = []
    for horizon in HORIZONS:
        steps = PREFIX_STEPS[horizon]
        prefix = Lambda(lambda z, s=steps: z[:, :s, :], name=f"future_prefix_{horizon}")(future)
        f = future_encoder(prefix)
        z = Concatenate(name=f"history_future_cgm24_fusion_{horizon}")([h, f, c])
        z = shared_dense(z)
        z = shared_dropout(z)
        abs_outputs.append(Dense(1, name=f"abs_{horizon}")(z))
        delta_outputs.append(Dense(1, name=f"delta_{horizon}")(z))

    outputs = abs_outputs + delta_outputs
    names = [f"abs_{h}" for h in HORIZONS] + [f"delta_{h}" for h in HORIZONS]
    model = Model(
        [history, future, cgm24],
        outputs,
        name="zyntra_v15_4_cgm24_aux_forecaster",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={n: tf.keras.losses.Huber() for n in names},
        metrics={n: [tf.keras.metrics.MeanAbsoluteError(name="mae")] for n in names},
    )
    return model
