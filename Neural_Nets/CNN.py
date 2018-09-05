import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from dataGenerator import data # dataGenerator library file generates data for model
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import time

name = "ecg-cnn-{}".format(int(time.time()))

def generate_Model(optimizer='adam', loss='categorical_crossentropy'):
    model.add(Conv1D(256, 3, input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))

    model.add(Conv1D(256, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))


    tensorboard = TensorBoard(log_dir="logs/{}".format(name))   # initialize Tensorboard

    model.compile(optimizer=optimizer,  # Good default optimizer to start with
                  loss=loss,
                  # how will we calculate our "error." Neural network aims to minimize loss.
                  metrics=['accuracy'])  # what to track
    return model, tensorboard

def main():
    X, y = data(normalize=True)
    model, tensorboard = generateModel()
    model.fit(X, y,
              batch_size=32,
              validation_split=0.1,
              epochs=10,
              callbacks=[tensorboard])  # train the model, 10 epochs with callback for tensorboard

if __name__ == '__main__':
    main()
