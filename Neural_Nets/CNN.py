import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from dataGenerator import data # dataGenerator library file generates data for model
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle
import os
import numpy as np
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

name = "ecg-cnn-{}".format(int(time.time()))

def generateModel():
    model = Sequential()
    model.add(Conv1D(256, 3, input_shape=(X_train.shape[1:])))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(256, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(256, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(256, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    # model.add(Dense(128))
    # model.add(Activation('relu'))

    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    tensorboard = TensorBoard(log_dir="logs/{}".format(name))  # initialize Tensorboard

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(loss='mse',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model, tensorboard

def main():
    model, tensorboard = generateModel()

    class_weight = {0.: 1,
                    1.: 5.277}

    model.fit(X_train, y_train,
              batch_size=128,
              epochs=10,
              validation_data=(X_test, y_test),
              class_weight=class_weight,
              callbacks=[tensorboard])

if __name__ == '__main__':
    main()
