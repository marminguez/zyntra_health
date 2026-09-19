"""V17 distributional causal trajectory forecaster.

Predicts conditional glucose quantiles at each horizon from the proven causal
Trajectory backbone. The median (P50) is the point forecast; outer quantiles
represent future uncertainty. No future CGM is accepted as input.
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,Input,LSTM,Lambda
from tensorflow.keras.models import Model

HORIZONS=(30,60,90,120)
READ_STEPS={30:6,60:12,90:18,120:24}
QUANTILES=(.10,.25,.50,.75,.90)
FUTURE_STEPS=24

def pinball(q):
    def loss(y_true,y_pred):
        e=y_true-y_pred
        return tf.reduce_mean(tf.maximum(q*e,(q-1.0)*e))
    loss.__name__=f"pinball_{int(q*100):02d}"
    return loss

def build_v17_forecaster(sequence_length:int,n_features:int)->Model:
    history=Input((sequence_length,n_features),name="metabolic_history")
    h=LSTM(64,return_sequences=True,name="history_lstm_1")(history)
    h=Dropout(.20,name="history_dropout_1")(h)
    _,hh,hc=LSTM(32,return_state=True,name="history_lstm_2")(h)
    future=Input((FUTURE_STEPS,6),name="future_known")
    fp=Dense(32,activation="relu",name="future_step_projection")(future)
    traj=LSTM(32,return_sequences=True,name="causal_future_trajectory_lstm")(fp,initial_state=[hh,hc])
    shared=Dense(32,activation="relu",name="shared_dense")
    drop=Dropout(.10,name="shared_dropout")
    outputs=[]; losses={}
    for horizon in HORIZONS:
        idx=READ_STEPS[horizon]-1
        z=Lambda(lambda t,i=idx:t[:,i,:],name=f"trajectory_read_{horizon}")(traj)
        z=drop(shared(z))
        for q in QUANTILES:
            name=f"q{int(q*100):02d}_{horizon}"
            outputs.append(Dense(1,name=name)(z))
            losses[name]=pinball(q)
    model=Model([history,future],outputs,name="zyntra_v17_distributional_trajectory")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=losses)
    return model
