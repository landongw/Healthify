from keras.models import load_model
import numpy as np
import tensorflow as tf
import os
import pickle

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

path = 'C:\Healthify\Healthify'
os.chdir(path)

f = open('X_test.pickle', 'rb')
X_test = pickle.load(f)
f = open('y_test.pickle', 'rb')
y_test = pickle.load(f)

X_test = np.array(X_test).reshape(-1, 300, 1)
y_test = np.array(y_test).reshape(-1, 1)

model = load_model('CNN_model.h5')
predicted = model.predict(X_test)

average = np.average(predicted)

predicted = (predicted > average)

# for i in range(len(predicted)):
#     print(predicted[i], y_test[i])

confusionMatrice = np.zeros((2, 2))
for i in range(len(predicted)):
    if (predicted[i, 0] == True):
        if (y_test[i, 0] == 1):
            confusionMatrice[0, 0] += 1
        else:
            confusionMatrice[0, 1] += 1
    else:
        if (y_test[i, 0] == 1):
            confusionMatrice[1, 0] += 1
        else:
            confusionMatrice[1, 1] += 1

print(confusionMatrice)


