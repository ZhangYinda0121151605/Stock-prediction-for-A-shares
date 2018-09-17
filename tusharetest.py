import tushare as ts
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense, LSTM, merge
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# stimu function
def atan(x): 
    return tf.atan(x)

# basic config

class conf:
    instrument = '600000' #code of shares
    start_date = '2005-01-01'
    split_date = '2015-01-01' #before split: training data
    end_date = '2015-12-01' #after split: test data
    fields = ['close', 'open', 'high', 'low', 'volume', 'amount'] #amount = close*volume
    seq_len = 30 #each sample
    batch = 100 #one gradient descent with 100 samples 

# geting data from tushare and preprocessing

df = ts.get_k_data(conf.instrument, conf.start_date, conf.end_date)
#df.to_csv("600000data.csv", index=False, sep=',')
df['amount'] = df['close']*df['volume']
df['return'] = df['close'].shift(-5) / df['close'].shift(-1) - 1 #return(yield rate) = close price 5 days after / close price tomorrow
#df['return'] = df['return'].apply(lambda x:np.where(x>=0.2,0.2,np.where(x>-0.2,x,-0.2)))
df['return'] = df['return']*10 #just for training
df.dropna(inplace=True)
dftime = df['date'][df.date>=conf.split_date]
df.reset_index(drop=True, inplace=True)
scaledf = df[conf.fields]
traindf = df[df.date<conf.split_date]

train_input = []
train_output = []
test_input = []
test_output = []
for i in range(conf.seq_len-1, len(traindf)):
    a = scale(scaledf[i+1-conf.seq_len:i+1])
    train_input.append(a)
    c = df['return'][i]
    train_output.append(c)
    
for i in range(len(traindf), len(df)):
    a = scale(scaledf[i+1-conf.seq_len:i+1])
    test_input.append(a)
    c = df['return'][i]
    test_output.append(c)

train_x = np.array(train_input)
train_y = np.array(train_output)
test_x = np.array(test_input) 
test_y = np.array(test_output)

lstm_input = Input(shape=(30,6), name='lstm_input')
lstm_output = LSTM(128, activation=atan, dropout_W=0.2, dropout_U=0.1)(lstm_input)
Dense_output_1 = Dense(64, activation='linear', kernel_regularizer=keras.regularizers.l1(0.))(lstm_output)
Dense_output_2 = Dense(16, activation='linear')(Dense_output_1)
predictions = Dense(1, activation=atan)(Dense_output_2)
model = Model(input=lstm_input, output=predictions)
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(train_x, train_y, batch_size=conf.batch, nb_epoch=10, verbose=2)

predictions = model.predict(test_x)

plt.title('Return Rate Estimation')

xindex = range(len(test_y))
test_y = test_y/10
plt.plot(xindex, test_y, color = 'red', label='true return rate')
plt.plot(xindex, predictions, color = 'blue', label='estimation')
plt.xlabel('Date')
plt.ylabel('Return Rate')

plt.show()


