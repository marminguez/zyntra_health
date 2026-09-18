"""V16 event-aware causal trajectory forecaster.

The causal trajectory backbone is preserved, but each horizon also predicts
future excursion dynamics from information available at inference time.
Excursion logits are learned jointly and condition the glucose heads end-to-end.
No future CGM is accepted as input.
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, LSTM, Lambda
from tensorflow.keras.models import Model

HORIZONS=(30,60,90,120)
READ_STEPS={30:6,60:12,90:18,120:24}
FUTURE_STEPS=24

def build_v16_forecaster(sequence_length:int,n_features:int)->Model:
    history=Input((sequence_length,n_features),name="metabolic_history")
    h=LSTM(64,return_sequences=True,name="history_lstm_1")(history)
    h=Dropout(.20,name="history_dropout_1")(h)
    _,hh,hc=LSTM(32,return_state=True,name="history_lstm_2")(h)
    future=Input((FUTURE_STEPS,6),name="future_known")
    fp=Dense(32,activation="relu",name="future_step_projection")(future)
    traj=LSTM(32,return_sequences=True,name="causal_future_trajectory_lstm")(fp,initial_state=[hh,hc])
    shared=Dense(32,activation="relu",name="shared_dense")
    drop=Dropout(.10,name="shared_dropout")
    abs_out=[]; delta_out=[]; event_out=[]
    for horizon in HORIZONS:
        idx=READ_STEPS[horizon]-1
        z=Lambda(lambda t,i=idx:t[:,i,:],name=f"trajectory_read_{horizon}")(traj)
        z=shared(z)
        # Auxiliary 3-way dynamics: large fall / non-event / large rise.
        event=Dense(3,activation="softmax",name=f"event_{horizon}")(z)
        conditioned=Concatenate(name=f"event_condition_{horizon}")([z,event])
        conditioned=Dense(32,activation="relu",name=f"event_fusion_{horizon}")(conditioned)
        conditioned=drop(conditioned)
        abs_out.append(Dense(1,name=f"abs_{horizon}")(conditioned))
        delta_out.append(Dense(1,name=f"delta_{horizon}")(conditioned))
        event_out.append(event)
    outputs=abs_out+delta_out+event_out
    reg_names=[f"abs_{h}" for h in HORIZONS]+[f"delta_{h}" for h in HORIZONS]
    event_names=[f"event_{h}" for h in HORIZONS]
    losses={n:tf.keras.losses.Huber() for n in reg_names}
    losses.update({n:tf.keras.losses.SparseCategoricalCrossentropy() for n in event_names})
    weights={n:1.0 for n in reg_names}; weights.update({n:.25 for n in event_names})
    model=Model([history,future],outputs,name="zyntra_v16_event_aware_trajectory")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=losses,loss_weights=weights)
    return model
