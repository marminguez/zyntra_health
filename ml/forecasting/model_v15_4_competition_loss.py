"""V15.4 architecture with a competition-aligned training objective.

Controlled ablation:
- architecture is unchanged from V15.4
- future-known masking is unchanged
- absolute heads use a blended absolute + relative robust loss
- delta heads keep standard Huber loss
- longer horizons receive slightly larger loss weights because they dominate the
  current error gap, while evaluation still reports each horizon separately

This does not use future CGM.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, LSTM, Lambda
from tensorflow.keras.models import Model

HORIZONS = (30, 60, 90, 120)
PREFIX_STEPS = {30: 6, 60: 12, 90: 18, 120: 24}
FUTURE_STEPS = 24


def absolute_relative_huber(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Blend robust mg/dL error with a robust relative-error term.

    The denominator floor avoids exploding gradients at low glucose while still
    making the objective more sensitive to relative error, which is closer to
    MARD than pure Huber.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    abs_err = y_pred - y_true
    abs_loss = tf.keras.losses.huber(y_true, y_pred, delta=15.0)

    denom = tf.maximum(tf.abs(y_true), 70.0)
    rel_true = tf.zeros_like(y_true)
    rel_pred = abs_err / denom
    rel_loss = tf.keras.losses.huber(rel_true, rel_pred, delta=0.10)

    # Put the dimensionless relative term on a comparable numerical scale.
    return 0.75 * abs_loss + 0.25 * (70.0 * rel_loss)


def build_v15_4_competition_loss_forecaster(sequence_length: int, n_features: int) -> Model:
    history = Input(shape=(sequence_length, n_features), name="metabolic_history")
    h = LSTM(64, return_sequences=True, name="history_lstm_1")(history)
    h = Dropout(.20, name="history_dropout_1")(h)
    h = LSTM(32, name="history_lstm_2")(h)

    future = Input(shape=(FUTURE_STEPS, 6), name="future_known")
    future_encoder = LSTM(16, name="future_known_lstm_shared")
    shared_dense = Dense(32, activation="relu", name="shared_dense")
    shared_dropout = Dropout(.10, name="shared_dropout")

    abs_outputs = []
    delta_outputs = []
    for horizon in HORIZONS:
        steps = PREFIX_STEPS[horizon]
        prefix = Lambda(lambda z, s=steps: z[:, :s, :], name=f"future_prefix_{horizon}")(future)
        f = future_encoder(prefix)
        z = Concatenate(name=f"history_future_fusion_{horizon}")([h, f])
        z = shared_dense(z)
        z = shared_dropout(z)
        abs_outputs.append(Dense(1, name=f"abs_{horizon}")(z))
        delta_outputs.append(Dense(1, name=f"delta_{horizon}")(z))

    outputs = abs_outputs + delta_outputs
    abs_names = [f"abs_{h}" for h in HORIZONS]
    delta_names = [f"delta_{h}" for h in HORIZONS]
    names = abs_names + delta_names

    losses = {name: absolute_relative_huber for name in abs_names}
    losses.update({name: tf.keras.losses.Huber() for name in delta_names})

    horizon_weights = {30: 1.00, 60: 1.10, 90: 1.20, 120: 1.30}
    loss_weights = {f"abs_{h}": horizon_weights[h] for h in HORIZONS}
    loss_weights.update({f"delta_{h}": horizon_weights[h] for h in HORIZONS})

    model = Model([history, future], outputs, name="zyntra_v15_4_competition_loss")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=losses,
        loss_weights=loss_weights,
        metrics={n: [tf.keras.metrics.MeanAbsoluteError(name="mae")] for n in names},
    )
    return model
