# -*- coding: utf-8 -*-
import keras
from keras.layers import merge
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import *
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import time
import os
import pickle
from tensorflow import reset_default_graph
reset_default_graph()

path = 'E:\local-repo\Healthify'
os.chdir(path)

f = open('X_train.pickle', 'rb')
X_train = pickle.load(f)
f = open('y_train.pickle', 'rb')
y_train = pickle.load(f)
f = open('X_test.pickle', 'rb')
X_test = pickle.load(f)
f = open('y_test.pickle', 'rb')
y_test = pickle.load(f)


X_train = np.array(X_train).reshape(-1, 300, 1)
X_test = np.array(X_test).reshape(-1, 300, 1)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

def block_type1(x, nb_filter, filter_len=16):
    out = Conv1D(nb_filter, filter_len, padding='same')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(0.5)(out)
    out = Conv1D(nb_filter, filter_len, padding='same')(out)
    return out

def block_type2(x, nb_filter, filter_len=16):
    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out = Dropout(0.5)(out)
    out = Conv1D(nb_filter, filter_len, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(0.5)(out)
    out = Conv1D(nb_filter, filter_len, padding='same')(out)
    return out


def modelGenerator(signal_size):
    inp = Input(shape=(signal_size,1))
    inp_begin= Conv1D(64, 16, padding='same')(inp)
    inp_begin = BatchNormalization()(inp_begin)
    inp_begin= Activation('relu')(inp_begin)

    out_1=block_type1(inp_begin, 64, filter_len=16)
    #maxpooling1
    inp_1=MaxPooling1D(pool_size=2, padding='valid')(inp_begin)
    out_1=MaxPooling1D(pool_size=2, padding='valid')(out_1)
    out_1=keras.layers.add([out_1, inp_1])

    out_2=block_type2(out_1, 64, filter_len=16)

    out_2=block_type2(out_2, 64, filter_len=16)
    #maxpooling2
    out_2=MaxPooling1D(pool_size=2, padding='valid')(out_2)
    inp_2=MaxPooling1D(pool_size=2, padding='valid')(inp_1)
    out_2=keras.layers.add([out_2, inp_2])

    out_3=block_type2(out_2, 64, filter_len=16)

    inp_3= Conv1D(64*2, 1, padding='same')(inp_2)

    out_3=block_type2(out_3, 64*2, filter_len=16)
    #maxpooling3
    out_3=MaxPooling1D(pool_size=2, padding='valid')(out_3)
    inp_3=MaxPooling1D(pool_size=2, padding='valid')(inp_3)
    out_3=keras.layers.add([out_3, inp_3])

    out_4=block_type2(out_3, 64*2)

    out_4=block_type2(out_4, 64*2)
    #maxpooling4
    out_4=MaxPooling1D(pool_size=2, padding='valid')(out_4)
    inp_4=MaxPooling1D(pool_size=2, padding='valid')(inp_3)
    out_4=keras.layers.add([out_4, inp_4])

    out_5=block_type2(out_4, 64*2)

    inp_5= Conv1D(64*3, 1, padding='same')(inp_4)

    out_5=block_type2(out_5, 64*3)
    #maxpooling5
    out_5=MaxPooling1D(pool_size=2, padding='valid')(out_5)
    inp_5=MaxPooling1D(pool_size=2, padding='valid')(inp_5)
    out_5=keras.layers.add([out_5, inp_5])

    out_6=block_type2(out_5, 64*3)
    out_6=block_type2(out_6, 64*3)
    #maxpooling 6
    out_6=MaxPooling1D(pool_size=2, padding='valid')(out_6)
    inp_6=MaxPooling1D(pool_size=2, padding='valid')(inp_5)
    out_6=keras.layers.add([out_6, inp_6])

    out_7=block_type2(out_6, 64*3)

    inp_7= Conv1D(64*4, 1, padding='same')(inp_6)
    out_7=block_type2(out_7, 64*4)
    #maxpooling 7
    out_7=MaxPooling1D(pool_size=2, padding='valid')(out_7)
    inp_7=MaxPooling1D(pool_size=2, padding='valid')(inp_7)
    out_7=keras.layers.add([out_7, inp_7])

    #out_8=block_type2(out_7, 64*4)
    #out_8=block_type2(out_8, 64*4)
    #out_8=MaxPooling1D(pool_size=2, padding='valid')(out_8)
    #inp_8=MaxPooling1D(pool_size=2, padding='valid')(inp_7)
    #out_8=keras.layers.add([out_8, inp_8])
    #out_8=block_type2(out_8, 64*4)

    out_final = BatchNormalization()(out_7)
    out_final= Activation('relu')(out_final)

    out_final=Flatten()(out_final)
    out_final=Dense(1)(out_final)
    out_final= Activation('sigmoid')(out_final)
    model = Model(inp, out_final)
    name = "ecg-cnn-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir="logs/{}".format(name))   # initialize Tensorboard
    model.compile(optimizer='adam', loss='mse')
    return model, tensorboard


def main():
    fs = 100 # 125Hz
    period = 3 # 10s
    signal_size = fs * period # 10 stands for 10s of signal
    model, tensorboard = modelGenerator(signal_size)
    class_weight ={0.: 1,
                   1.: 6}
    model.fit(X, y,
              batch_size=32,
              validation_split=0.1,
              epochs=10,
              callbacks=[tensorboard],
              class_weight=class_weight)  # train the model, 10 epochs with callback for tensorboard
    model.save('CNN_model.h5')

if __name__ == '__main__':
    main()
