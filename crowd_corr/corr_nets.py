import numpy as np
#import theano
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import SGD

from keras.utils.vis_utils import plot_model

import random

import itertools
import time

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex
  
(x_train, y_train_num), (x_test, y_test_num) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255
y_train = np_utils.to_categorical(y_train_num, 10)
y_test = np_utils.to_categorical(y_test_num, 10)

num_units = [5, 10, 15, 25, 50, 100]
activations = ['relu', 'sigmoid', 'tanh']
learning_rate = [0.1, 1, 10]
batch_size = [10, 100]
epochs = [10, 25, 50]

hyperparameters = itertools.product(num_units, activations, learning_rate, batch_size, epochs)
hp_list = np.zeros(len(num_units)*len(activations)*len(learning_rate)*len(batch_size)*len(epochs),
                  dtype=[('units', 'int'), ('actv','S10'), ('lr', 'float'), ('batch', 'int'), ('epochs', 'int')])
for i,j in enumerate(hyperparameters):
  hp_list[i]=j
sample_size = 1000
num_teams = 100
model_avg = np.zeros((num_teams, 4, 50))
param_array = np.zeros(num_teams, dtype=[('units', 'int'), ('actv','S10'), ('lr', 'float'), ('batch', 'int'), ('epochs', 'int')])
counts=np.ones(hp_list.shape[0])
probs=softmax(counts)
probs_array = np.zeros((num_teams+1, probs.shape[0]))
probs_array[0]=probs
alpha = 1.5 #Parameter to alter probabilities
for k in range(0,num_teams):
  n_set = np.arange(len(y_train))
  sample = np.random.choice(a=n_set, size=sample_size)
  x_subset = x_train[sample]
  y_subset = y_train[sample]
  params = np.random.choice(hp_list, p=probs)
  param_array[k] = params
  params[1] = params[1]
  print "Training model", k+1
  print "Hidden Units:", params[0], "Activation:", params[1].decode(), "Learn. rate:", params[2]
  print "Batch size:", params[3], "Epochs:", params[4]
  model = Sequential()
  model.add(Flatten())
  model.add(Dense(params[0], input_dim = 784, activation=params[1].decode()))
  model.add(Dense(10, activation='softmax'))

  sgd=SGD(lr=params[2])
  model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
  history = model.fit(x_subset, y_subset,
                    batch_size=params[3], epochs=params[4], verbose=1, validation_data=(x_test, y_test))
  if history.history['acc'][-1] >= 0.8:
    loc = np.where(hp_list == params)
    counts[loc] += alpha
    probs = softmax(counts) #softmax ensures that the sum of the probabilities is always 1
  probs_array[k+1] = probs
  for i in range(params[4]):
    model_avg[k][0][i] = history.history['acc'][i]
    model_avg[k][1][i] = history.history['val_acc'][i]
    model_avg[k][2][i] = history.history['loss'][i]
    model_avg[k][3][i] = history.history['val_loss'][i]

np.save('corr_array5.npy', model_avg)
np.save('hp_corr5.npy', param_array)
np.save('prob_corr5.npy', probs_array)
