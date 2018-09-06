# -*- coding: utf-8 -*-

from keras.layers import merge
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import *
from keras.layers.core import Dense,Activation,Flatten,Dropout,Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
from dataGenerator import data # dataGenerator library file generates data for model
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import time

name = "ecg-cnn-{}".format(int(time.time()))


def modelGenerator(signal_size):
    inp = Input(shape=(signal_size,1))
    inp_begin= Conv1D(64,16,padding='same')(inp)
    inp_begin = BatchNormalization()(inp_begin)
    inp_begin= Activation('relu')(inp_begin)
    ###

    out_1=block_type1(inp_begin,64,filter_len=16)
    #maxpooling1
    inp_1=MaxPooling1D(pool_size=2,padding='valid')(inp_begin)
    out_1=MaxPooling1D(pool_size=2,padding='valid')(out_1)
    out_1=merge([out_1,inp_1],mode='sum')


    out_2=block_type2(out_1,64,filter_len=16)

    out_2=block_type2(out_2,64,filter_len=16)
    #maxpooling2
    out_2=MaxPooling1D(pool_size=2,padding='valid')(out_2)
    inp_2=MaxPooling1D(pool_size=2,padding='valid')(inp_1)
    out_2=merge([out_2,inp_2],mode='sum')

    out_3=block_type2(out_2,64,filter_len=16)

    inp_3= Conv1D(64*2,1,padding='same')(inp_2)

    out_3=block_type2(out_3,64*2,filter_len=16)
    #maxpooling3
    out_3=MaxPooling1D(pool_size=2,padding='valid')(out_3)
    inp_3=MaxPooling1D(pool_size=2,padding='valid')(inp_3)
    out_3=merge([out_3,inp_3],mode='sum')

    out_4=block_type2(out_3,64*2)

    out_4=block_type2(out_4,64*2)
    #maxpooling4
    out_4=MaxPooling1D(pool_size=2,padding='valid')(out_4)
    inp_4=MaxPooling1D(pool_size=2,padding='valid')(inp_3)
    out_4=merge([out_4,inp_4],mode='sum')

    out_5=block_type2(out_4,64*2)

    inp_5= Conv1D(64*3,1,padding='same')(inp_4)

    out_5=block_type2(out_5,64*3)
    #maxpooling5
    out_5=MaxPooling1D(pool_size=2,padding='valid')(out_5)
    inp_5=MaxPooling1D(pool_size=2,padding='valid')(inp_5)
    out_5=merge([out_5,inp_5],mode='sum')

    out_6=block_type2(out_5,64*3)
    out_6=block_type2(out_6,64*3)
    #maxpooling 6
    out_6=MaxPooling1D(pool_size=2,padding='valid')(out_6)
    inp_6=MaxPooling1D(pool_size=2,padding='valid')(inp_5)
    out_6=merge([out_6,inp_6],mode='sum')

    out_7=block_type2(out_6,64*3)

    inp_7= Conv1D(64*4,1,padding='same')(inp_6)
    out_7=block_type2(out_7,64*4)
    #maxpooling 7
    out_7=MaxPooling1D(pool_size=2,padding='valid')(out_7)
    inp_7=MaxPooling1D(pool_size=2,padding='valid')(inp_7)
    out_7=merge([out_7,inp_7],mode='sum')

    #out_8=block_type2(out_7,64*4)
    #out_8=block_type2(out_8,64*4)
    #out_8=MaxPooling1D(pool_size=2,padding='valid')(out_8)
    #inp_8=MaxPooling1D(pool_size=2,padding='valid')(inp_7)
    #out_8=merge([out_8,inp_8],mode='sum')
    #out_8=block_type2(out_8,64*4)

    out_final = BatchNormalization()(out_7)
    out_final= Activation('relu')(out_final)

    #out_final=Dense(14)(out_final)
    out_final=Flatten()(out_final)
    out_final=Dense(signal_size)(out_final)
    out_final= Activation('softmax')(out_final)
    model = Model(inp,out_final)
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))   # initialize Tensorboard
    model.compile(optimizer='adam',loss='mse')
    return model, tensorboard


def main():
    # X, y = data(normalize=True)
    fs = 125 # 125Hz
    period = 10 # 10s
    signal_size = fs * period # 10 stands for 10s of signal
    model, tensorboard = modelGenerator(signal_size)
    model.save('CNN_model.h5')
    # model.fit(X, y,
    #           batch_size=32,
    #           validation_split=0.1,
    #           epochs=10,
    #           callbacks=[tensorboard])  # train the model, 10 epochs with callback for tensorboard

if __name__ == '__main__':
    main()


