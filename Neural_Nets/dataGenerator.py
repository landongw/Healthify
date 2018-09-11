import numpy as np  # we all know what np is :)
import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import csv
import os
import pickle

path = 'E:\local-repo\Healthify'
os.chdir(path)
number_samples = (19643 - 622)  # all samples - n/a samples
fs = 100  # 100Hz
T = 3  # 10s
signalSize = fs * T  # 1000 number per sample

def data(normalize=True):
    with open('db.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        data = np.zeros((number_samples, 1 + signalSize))
        y = np.zeros((number_samples, 1))
        X = np.zeros((number_samples, signalSize))
        index = 0
        for row in readCSV:
            if ('n/a' in row[0]):
                continue
            elif ('Myocardial' in row[0]):
                data[index, 0] = 0
            elif ('Healthy' in row[0]):
                data[index, 0] = 1
            else:
                data[index, 0] = 0

            data[index, 1:] = row[1:-1]
            index += 1
        slice = round(0.8*number_samples)
        train = data[:slice, :]
        test = data[slice:, :]
        np.random.shuffle(train)
        np.random.shuffle(test)
        y_train = train[:, :1]
        X_train = train[:, 1:]
        y_test = test[:, :1]
        X_test = test[:, 1:]
    if (normalize == True):
        X_test = tf.keras.utils.normalize(X_test, axis=1)  # scales data between 0 and 1
        X_train = tf.keras.utils.normalize(X_train, axis=1)  # scales data between 0 and 1
    # print(y_test[1:1000])

    with open('X_train.pickle', 'wb') as f:
        pickle.dump(X_train, f)
    with open('y_train.pickle', 'wb') as f:
        pickle.dump(y_train, f)
    with open('X_test.pickle', 'wb') as f:
        pickle.dump(X_test, f)
    with open('y_test.pickle', 'wb') as f:
        pickle.dump(y_test, f)


data(normalize=True)

