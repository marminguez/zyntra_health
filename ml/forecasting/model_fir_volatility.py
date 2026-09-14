"""FIR-1 architecture plus explicit pre-anchor CGM volatility context."""
import tensorflow as tf
from tensorflow.keras.layers import Concatenate,Dense,Dropout,Input,LSTM,Lambda
from tensorflow.keras.models import Model
HORIZONS=(30,60,90,120); PREFIX_STEPS={30:6,60:12,90:18,120:24}; FUTURE_STEPS=24; SUMMARY_FEATURES=13; VOL_FEATURES=8
def build_fir_volatility_forecaster(sequence_length,n_features):
 history=Input((sequence_length,n_features),name='metabolic_history'); h=LSTM(64,return_sequences=True,name='history_lstm_1')(history);h=Dropout(.20,name='history_dropout_1')(h);h=LSTM(32,name='history_lstm_2')(h)
 future=Input((FUTURE_STEPS,6),name='future_known');summary=Input((4,SUMMARY_FEATURES),name='future_intervention_summary');vol=Input((VOL_FEATURES,),name='cgm_volatility_context')
 fe=LSTM(16,name='future_known_lstm_shared');se=Dense(16,activation='relu',name='future_summary_dense_shared');ve=Dense(8,activation='relu',name='volatility_dense')(vol);sd=Dense(32,activation='relu',name='shared_dense');do=Dropout(.10,name='shared_dropout');ao=[];dd=[]
 for hi,hz in enumerate(HORIZONS):
  f=fe(Lambda(lambda z,s=PREFIX_STEPS[hz]:z[:,:s,:],name=f'future_prefix_{hz}')(future));s=se(Lambda(lambda z,i=hi:z[:,i,:],name=f'summary_slice_{hz}')(summary));z=Concatenate(name=f'history_future_summary_volatility_fusion_{hz}')([h,f,s,ve]);z=do(sd(z));ao.append(Dense(1,name=f'abs_{hz}')(z));dd.append(Dense(1,name=f'delta_{hz}')(z))
 names=[f'abs_{h}' for h in HORIZONS]+[f'delta_{h}' for h in HORIZONS];m=Model([history,future,summary,vol],ao+dd,name='zyntra_fir_volatility_forecaster');m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss={n:tf.keras.losses.Huber() for n in names},metrics={n:[tf.keras.metrics.MeanAbsoluteError(name='mae')] for n in names});return m
