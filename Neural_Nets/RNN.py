import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM
import keras
import pickle
import os
import numpy as np
from tensorflow import reset_default_graph
reset_default_graph()
path = 'E:\local-repo\Healthify'
os.chdir(path)

f = open('X.pickle', 'rb')
X = pickle.load(f)
f = open('y.pickle', 'rb')
y = pickle.load(f)

X = np.array(X).reshape(-1, 300, 1)
y = np.array(y).reshape(-1, 1)

model = Sequential()

# IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)

model.add(LSTM(64, input_shape=(X.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(lr=0.0005, decay=1e-6)
# Compile model
model.compile(
    loss='mse',
    optimizer=opt,
    metrics=['accuracy'])

class_weight = {0.: 1,
                1.: 5.277}


model.fit(X, y,
          batch_size=64,
          epochs=3,
          validation_split=0.3,
          class_weight=class_weight)
