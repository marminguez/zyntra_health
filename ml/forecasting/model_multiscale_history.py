"""Top-3 fast-screen: multi-scale history encoder for V15.4-style forecasting.

Rationale: the current V15.4 baseline compresses the full 24h x 5-min history
through a single LSTM stack. This model keeps the same inputs/targets/future-known
branch but splits historical representation into:
- recent high-resolution dynamics: last 2h at 5-min cadence
- long-context trend: full 24h pooled to 30-min cadence

No future CGM is used. Output heads and fixed downstream fusion protocol remain
compatible with the V15.4 Scale-10 trainer/evaluator.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    AveragePooling1D,
    Concatenate,
    Dense,
    Dropout,
    Input,
    LSTM,
    Lambda,
)
from tensorflow.keras.models import Model

HORIZONS = (30, 60, 90, 120)
PREFIX_STEPS = {30: 6, 60: 12, 90: 18, 120: 24}
FUTURE_STEPS = 24
RECENT_STEPS = 24  # 2h at 5-min cadence
POOL_SIZE = 6      # 30-min long-context cadence


def build_multiscale_forecaster(sequence_length: int, n_features: int) -> Model:
    if sequence_length % POOL_SIZE != 0:
        raise ValueError(
            f"sequence_length={sequence_length} must be divisible by pool size {POOL_SIZE}"
        )
    if sequence_length < RECENT_STEPS:
        raise ValueError(
            f"sequence_length={sequence_length} shorter than recent branch {RECENT_STEPS}"
        )

    history = Input(shape=(sequence_length, n_features), name="metabolic_history")

    # Recent branch: preserve high-frequency dynamics over the last 2h.
    recent = Lambda(lambda z: z[:, -RECENT_STEPS:, :], name="recent_2h_slice")(history)
    recent = LSTM(48, return_sequences=True, name="recent_lstm_1")(recent)
    recent = Dropout(0.15, name="recent_dropout")(recent)
    recent = LSTM(24, name="recent_lstm_2")(recent)

    # Long branch: preserve circadian/metabolic context without forcing one LSTM
    # to process all 288 fine-grained steps.
    long_ctx = AveragePooling1D(
        pool_size=POOL_SIZE,
        strides=POOL_SIZE,
        name="long_context_pool_30m",
    )(history)
    long_ctx = LSTM(32, name="long_context_lstm")(long_ctx)

    history_state = Concatenate(name="multiscale_history_fusion")([recent, long_ctx])
    history_state = Dense(48, activation="relu", name="history_projection")(history_state)
    history_state = Dropout(0.10, name="history_projection_dropout")(history_state)

    future = Input(shape=(FUTURE_STEPS, 6), name="future_known")
    future_encoder = LSTM(16, name="future_known_lstm_shared")
    shared_dense = Dense(32, activation="relu", name="shared_dense")
    shared_dropout = Dropout(0.10, name="shared_dropout")

    abs_outputs = []
    delta_outputs = []
    for horizon in HORIZONS:
        steps = PREFIX_STEPS[horizon]
        prefix = Lambda(
            lambda z, s=steps: z[:, :s, :],
            name=f"future_prefix_{horizon}",
        )(future)
        f = future_encoder(prefix)
        z = Concatenate(name=f"history_future_fusion_{horizon}")([history_state, f])
        z = shared_dense(z)
        z = shared_dropout(z)
        abs_outputs.append(Dense(1, name=f"abs_{horizon}")(z))
        delta_outputs.append(Dense(1, name=f"delta_{horizon}")(z))

    outputs = abs_outputs + delta_outputs
    names = [f"abs_{h}" for h in HORIZONS] + [f"delta_{h}" for h in HORIZONS]
    model = Model(
        [history, future],
        outputs,
        name="zyntra_multiscale_history_forecaster",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={n: tf.keras.losses.Huber() for n in names},
        metrics={n: [tf.keras.metrics.MeanAbsoluteError(name="mae")] for n in names},
    )
    return model
