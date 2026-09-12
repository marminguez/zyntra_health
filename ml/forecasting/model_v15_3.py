"""V15.3: V15.2 architecture with future basal + bolus + carbs, no aggregate insulin."""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import Concatenate,Dense,Dropout,Input,LSTM
from tensorflow.keras.models import Model
HORIZONS=(30,60,90,120);FUTURE_STEPS=24

def build_v15_3_forecaster(sequence_length:int,n_features:int)->Model:
    history=Input(shape=(sequence_length,n_features),name='metabolic_history');h=LSTM(64,return_sequences=True,name='history_lstm_1')(history);h=Dropout(.20,name='history_dropout_1')(h);h=LSTM(32,name='history_lstm_2')(h)
    future=Input(shape=(FUTURE_STEPS,6),name='future_known')
    # normalized basal, basal_missing, normalized bolus, bolus_missing, normalized carbs, carbs_missing
    f=LSTM(16,name='future_known_lstm')(future);x=Concatenate(name='history_future_fusion')([h,f]);x=Dense(32,activation='relu',name='shared_dense')(x);x=Dropout(.10,name='dropout_2')(x)
    outputs=[Dense(1,name=f'abs_{q}')(x) for q in HORIZONS]+[Dense(1,name=f'delta_{q}')(x) for q in HORIZONS]
    m=Model([history,future],outputs,name='zyntra_v15_3_basal_bolus_carbs_forecaster');names=[f'abs_{q}' for q in HORIZONS]+[f'delta_{q}' for q in HORIZONS]
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss={n:tf.keras.losses.Huber() for n in names},metrics={n:[tf.keras.metrics.MeanAbsoluteError(name='mae')] for n in names});return m
