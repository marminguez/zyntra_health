"""Optional frozen GlucoFM context adapter for Zyntra.

This module deliberately does not implement GlucoFM itself. It defines the
integration seam for embeddings produced by an official frozen GlucoFM encoder.
Zyntra history and future-intervention branches remain intact.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization


def adapt_glucofm_embedding(x: tf.Tensor, units: int = 32) -> tf.Tensor:
    """Project a frozen GlucoFM embedding into Zyntra fusion space."""
    x = LayerNormalization(name="glucofm_adapter_norm")(x)
    x = Dense(units, activation="relu", name="glucofm_adapter_dense")(x)
    return Dropout(0.10, name="glucofm_adapter_dropout")(x)
