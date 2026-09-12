"""FIR-1: V15.4 Scale-50 plus explicit future-intervention summaries.

Controlled change vs V15.4 Scale-50:
- keep the history encoder unchanged;
- keep the raw horizon-masked future LSTM unchanged;
- add one shared Dense encoder over 13 deterministic intervention-summary
  features for each horizon;
- fuse history + raw-future embedding + summary embedding before the same
  shared Dense/Dropout and absolute/delta heads.

No future CGM is accepted by this model.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, LSTM, Lambda
from tensorflow.keras.models import Model

HORIZONS = (30, 60, 90, 120)
PREFIX_STEPS = {30: 6, 60: 12, 90: 18, 120: 24}
FUTURE_STEPS = 24
SUMMARY_FEATURES = 13


def build_fir1_forecaster(sequence_length: int, n_features: int) -> Model:
    history = Input(shape=(sequence_length, n_features), name="metabolic_history")
    h = LSTM(64, return_sequences=True, name="history_lstm_1")(history)
    h = Dropout(0.20, name="history_dropout_1")(h)
    h = LSTM(32, name="history_lstm_2")(h)

    future = Input(shape=(FUTURE_STEPS, 6), name="future_known")
    summary = Input(shape=(len(HORIZONS), SUMMARY_FEATURES), name="future_intervention_summary")

    future_encoder = LSTM(16, name="future_known_lstm_shared")
    summary_encoder = Dense(16, activation="relu", name="future_summary_dense_shared")
    shared_dense = Dense(32, activation="relu", name="shared_dense")
    shared_dropout = Dropout(0.10, name="shared_dropout")

    abs_outputs = []
    delta_outputs = []

    for hi, horizon in enumerate(HORIZONS):
        steps = PREFIX_STEPS[horizon]
        prefix = Lambda(
            lambda z, s=steps: z[:, :s, :],
            name=f"future_prefix_{horizon}",
        )(future)
        f = future_encoder(prefix)

        s = Lambda(
            lambda z, i=hi: z[:, i, :],
            name=f"summary_slice_{horizon}",
        )(summary)
        s = summary_encoder(s)

        z = Concatenate(name=f"history_future_summary_fusion_{horizon}")([h, f, s])
        z = shared_dense(z)
        z = shared_dropout(z)
        abs_outputs.append(Dense(1, name=f"abs_{horizon}")(z))
        delta_outputs.append(Dense(1, name=f"delta_{horizon}")(z))

    outputs = abs_outputs + delta_outputs
    names = [f"abs_{h}" for h in HORIZONS] + [f"delta_{h}" for h in HORIZONS]

    model = Model(
        [history, future, summary],
        outputs,
        name="zyntra_fir1_future_intervention_forecaster",
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
