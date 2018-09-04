import numpy as np  # we all know what np is :)
import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
from preproces import pre_data # preprocess library return the preprocessed data

def data(normalize):

    preData = pre_data()




    if (normalize == True):
        X = tf.keras.utils.normalize(X, axis=1)  # scales data between 0 and 1
        y = tf.keras.utils.normalize(y, axis=1)  # scales data between 0 and 1

    return X, y


