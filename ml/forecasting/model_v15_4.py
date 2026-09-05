"""V15.4: horizon-masked future-known inputs on frozen V15 master windows.

Each prediction horizon may only consume future-known covariates up to its own
target time: +30 sees steps 1..6, +60 sees 1..12, +90 sees 1..18, +120 sees
1..24. The same future LSTM and shared dense weights are reused across horizons
to keep this as close as possible to V15.3 while removing post-target context.
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import Concatenate,Dense,Dropout,Input,LSTM,Lambda
from tensorflow.keras.models import Model
HORIZONS=(30,60,90,120)
PREFIX_STEPS={30:6,60:12,90:18,120:24}
FUTURE_STEPS=24

def build_v15_4_forecaster(sequence_length:int,n_features:int)->Model:
    history=Input(shape=(sequence_length,n_features),name='metabolic_history')
    h=LSTM(64,return_sequences=True,name='history_lstm_1')(history)
    h=Dropout(.20,name='history_dropout_1')(h)
    h=LSTM(32,name='history_lstm_2')(h)

    future=Input(shape=(FUTURE_STEPS,6),name='future_known')
    future_encoder=LSTM(16,name='future_known_lstm_shared')
    shared_dense=Dense(32,activation='relu',name='shared_dense')
    shared_dropout=Dropout(.10,name='shared_dropout')

    abs_outputs=[];delta_outputs=[]
    for horizon in HORIZONS:
        steps=PREFIX_STEPS[horizon]
        prefix=Lambda(lambda z,s=steps:z[:,:s,:],name=f'future_prefix_{horizon}')(future)
        f=future_encoder(prefix)
        z=Concatenate(name=f'history_future_fusion_{horizon}')([h,f])
        z=shared_dense(z)
        z=shared_dropout(z)
        abs_outputs.append(Dense(1,name=f'abs_{horizon}')(z))
        delta_outputs.append(Dense(1,name=f'delta_{horizon}')(z))

    outputs=abs_outputs+delta_outputs
    names=[f'abs_{h}' for h in HORIZONS]+[f'delta_{h}' for h in HORIZONS]
    m=Model([history,future],outputs,name='zyntra_v15_4_horizon_masked_forecaster')
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss={n:tf.keras.losses.Huber() for n in names},metrics={n:[tf.keras.metrics.MeanAbsoluteError(name='mae')] for n in names})
    return m
