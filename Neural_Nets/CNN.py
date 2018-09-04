import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from dataGenerator import data # dataGenerator library file generates data for model
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import time

name = "ecg-cnn-{}".format(int(time.time()))

def generate_Model(optimizer='adam', loss='categorical_crossentropy', activation=tf.nn.relu):
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


    model.compile(optimizer=optimizer,  # Good default optimizer to start with
                  loss=loss,
                  # how will we calculate our "error." Neural network aims to minimize loss.
                  metrics=['accuracy'])  # what to track
    return model

def main():
    X, y = data(normalize=True)
    model = generateModel()
    model.fit(x_train, y_train, epochs=10)  # train the model, 10 epochs
    val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
    print(val_loss)  # model's loss (error)
    print(val_acc)  # model's accuracy


if __name__ == '__main__':
    main()
