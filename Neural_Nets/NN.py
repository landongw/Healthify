import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import os

path = 'E:\local-repo\Healthify'
os.chdir(path)

def generateModel(optimizer='adam', loss='categorical_crossentropy'):

    model = tf.keras.models.Sequential()  # a basic feed-forward model
    model.add(tf.keras.layers.Flatten())  # takes our 1x1250 and returns it
    # model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu))  # a simple fully-connected layer, 1024 units, relu activation
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))  # a simple fully-connected layer, 512 units, relu activation
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))  # a simple fully-connected layer, 256 units, relu activation
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # a simple fully-connected layer, 256 units, relu activation
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))  # our output layer. 3 units for 3 classes(1: Healthy objects, 2: MI Objects, 3: Other Heart diseases objects). Softmax for probability distribution
    name = "ecg-nn-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir="logs/{}".format(name))   # initialize Tensorboard
    model.compile(optimizer=optimizer,  # Good default optimizer to start with
                  loss=loss,
                  # how will we calculate our "error." Neural network aims to minimize loss.
                  metrics=['accuracy'])  # what to track
    return model, tensorboard

def main():
    f = open('X.pickle','rb')
    X = pickle.load(f)
    f = open('y.pickle','rb')
    y = pickle.load(f)
    model, tensorboard = generateModel()
    model.fit(X, y, batch_size=32, epochs=25, validation_split=0.1, callbacks=[tensorboard])  # train the model, 25 epochs
    model.save('CNN_model.h5')


if __name__ == '__main__':
    main()
