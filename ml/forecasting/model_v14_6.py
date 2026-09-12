"""Zyntra V14.6 hybrid forecaster with auxiliary rapid-drop tasks."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model

HORIZONS = (30, 60, 90, 120)
AUX_LOSS_WEIGHT = 5.0


def build_v14_6_forecaster(sequence_length: int, n_features: int) -> Model:
    """V14.5 backbone/forecast heads plus rapid-drop auxiliary heads.

    Forecasting capacity is unchanged from V14.5. Four sigmoid auxiliary heads
    predict whether glucose will fall by at least 30 mg/dL at each horizon.
    These heads are training-only representation constraints: they are not used
    in the final glucose forecast.
    """
    inputs = Input(shape=(sequence_length, n_features), name="metabolic_history")
    x = LSTM(64, return_sequences=True, name="lstm_1")(inputs)
    x = Dropout(0.20, name="dropout_1")(x)
    x = LSTM(32, name="lstm_2")(x)
    x = Dense(32, activation="relu", name="shared_dense")(x)
    x = Dropout(0.10, name="dropout_2")(x)

    abs_outputs = [Dense(1, name=f"abs_{h}")(x) for h in HORIZONS]
    delta_outputs = [Dense(1, name=f"delta_{h}")(x) for h in HORIZONS]
    drop_outputs = [Dense(1, activation="sigmoid", name=f"drop_{h}")(x) for h in HORIZONS]
    outputs = abs_outputs + delta_outputs + drop_outputs

    model = Model(inputs=inputs, outputs=outputs, name="zyntra_v14_6_drop_aware_forecaster")

    regression_names = [f"abs_{h}" for h in HORIZONS] + [f"delta_{h}" for h in HORIZONS]
    drop_names = [f"drop_{h}" for h in HORIZONS]
    losses = {name: tf.keras.losses.Huber() for name in regression_names}
    losses.update({name: tf.keras.losses.BinaryCrossentropy() for name in drop_names})
    loss_weights = {name: 1.0 for name in regression_names}
    loss_weights.update({name: AUX_LOSS_WEIGHT for name in drop_names})
    metrics = {
        name: [tf.keras.metrics.MeanAbsoluteError(name="mae")]
        for name in regression_names
    }
    metrics.update({
        name: [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ]
        for name in drop_names
    })

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return model
