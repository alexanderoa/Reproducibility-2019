import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

n_set = np.arange(len(y_train))
sample = np.random.choice(a=n_set, size=1000)
x_subset = x_train[sample]
y_subset = y_train[sample]
np.save('rn_sample1k.npy', sample)
def random_search(n_iter, search_space):
  par_test = np.zeros((n_iter, 2))
  acc_test = np.zeros(n_iter)
  for i in range(n_iter):
    params = search_space[np.random.randint(search_space.shape[0])]
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(15, input_dim = 784, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    sgd=SGD(lr=params[0])
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    print("Training model", i)
    print("Learning rate:", params[0])
    print("Batch size:", params[1])
    history = model.fit(x_subset, y_subset, 
          batch_size=int(params[1]), epochs=10, verbose=2, validation_data = (x_test, y_test))
    par_test[i] = params
    acc_test[i] = history.history['acc'][-1]
  if par_test[np.where(acc_test==acc_test.max())].shape != (2,):
    return acc_test.max(), par_test[np.where(acc_test==acc_test.max())][0]
  return acc_test.max(), par_test[np.where(acc_test==acc_test.max())]  

bounds1=np.array([[0.001, 25]])
bounds2=np.array([[1, 250]])
X1 = np.arange(bounds1[:, 0], bounds1[:, 1], 0.001).reshape(-1, 1)
X2 = np.arange(bounds2[:, 0], bounds2[:, 1], 1).reshape(-1, 1)
space=np.array(np.meshgrid(X1, X2)).T.reshape(-1,2)
search_data = np.zeros(30)
par_data = np.zeros((30, 2))
acc_data = np.zeros((30, 2))
for i in range(30):
  search_data[i], par_data[i] = random_search(20, space)
  model = Sequential()
  model.add(Flatten())
  model.add(Dense(15, input_dim = 784, activation='sigmoid'))
  model.add(Dense(10, activation='softmax'))
  sgd=SGD(lr=par_data[i][0])
  model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
  print("Training final model", i)
  history = model.fit(x_subset, y_subset, 
          batch_size=int(par_data[i][1]), epochs=10, verbose=0, validation_data = (x_test, y_test))
  tf.reset_default_graph()
  acc_data[i][0] = history.history['acc'][-1]
  acc_data[i][1] = history.history['val_acc'][-1]

np.save('acc_data1k.npy', acc_data)
np.save('par_data1k.npy', par_data)
np.save('search_data1k.npy', search_data)
