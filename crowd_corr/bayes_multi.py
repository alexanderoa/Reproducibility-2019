import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
import itertools
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# %matplotlib inline
n_set = np.arange(len(y_train))
sample = np.random.choice(a=n_set, size=1000)
x_subset = x_train[sample]
y_subset = y_train[sample]

def f(params):
    print "Learning rate:", params[0]
    print "Batch size:", params[1]
    model = Sequential()

    model.add(Flatten())
    model.add(Dense(15, input_dim = 784, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    sgd=SGD(lr=params[0])
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

    history = model.fit(x_subset, y_subset, 
          batch_size=int(params[1]), epochs=10, verbose=0, validation_data = (x_test, y_test))
    return -history.history['val_acc'][-1]

from scipy.stats import norm

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    ''' Computes the EI at points X based on existing samples X_sample and Y_sample using a Gaussian process surrogate model. Args: X: Points at which EI shall be computed (m x d). X_sample: Sample locations (n x d). Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples. xi: Exploitation-exploration trade-off parameter. Returns: Expected improvements at points X. '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

from scipy.optimize import minimize

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    ''' Proposes the next sampling point by optimizing the acquisition function. Args: acquisition: Acquisition function. X_sample: Sample locations (n x d). Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples. Returns: Location of the acquisition function maximum. '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
    
    # Find the best optimum by starting from n_restart different random points.
    X1 = np.arange(bounds1[:, 0], bounds1[:, 1], 0.01).reshape(-1, 1)
    X2 = np.arange(bounds2[:, 0], bounds2[:, 1]).reshape(-1, 1)
    X=np.array(np.meshgrid(X1, X2)).T.reshape(-1,2)
    n_set = np.arange(X.shape[0])
    sample = np.random.choice(n_set, 25)
    for x0 in X[sample]:
        res = minimize(min_obj, x0=(x0[0], x0[1]), bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x.reshape(-1, 1)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
def bayesopt_multi(n_iter, X_init, Y_init, bounds1, bounds2, X1, X2):
  # Gaussian process with Mat??rn kernel as surrogate model
  m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
  gpr = GaussianProcessRegressor(kernel=m52, alpha=0.02**2)
  # Initialize samples
  X_sample = X_init
  Y_sample = Y_init
  best_acc = np.zeros(n_iter+1)
  best_acc[0] = -1 * Y_init.min()
  # Number of iterations

  for i in range(n_iter):
      print("Training model", i+1)
      # Update Gaussian process with existing samples
      gpr.fit(X_sample, Y_sample)

      # Obtain next sampling point from the acquisition function (expected_improvement)
      X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, ((bounds1[:, 0][0], bounds1[:, 1][0]), (bounds2[:, 0][0], bounds2[:, 1][0])))
    
      # Obtain next noisy sample from the objective function
      Y_next = f(X_next)
    
      # Add sample to previous samples
      X_sample = np.column_stack((X_sample.T, X_next)).T
      Y_sample = np.vstack((Y_sample, np.array(Y_next).reshape(-1,1)))
      best_acc[i+1] = -1 * Y_sample.min()
  return gpr, X_sample, Y_sample

acc_ch = np.zeros((3,10))
acc_rn = np.zeros((3, 10))

for i in range(10):
  print "Trial", i
  bounds1 = np.array([[0.001, 2]])
  bounds2 = np.array([[1, 32]])
  X1 = np.arange(bounds1[:, 0], bounds1[:, 1], 0.001).reshape(-1, 1)
  X2 = np.arange(bounds2[:, 0], bounds2[:, 1], 1).reshape(-1, 1)
  x1=np.array(np.meshgrid(X1, X2)).T.reshape(-1,2)
  X_init = x1[np.random.randint(x1.shape[0], size=2)]
  y1 = f(X_init[0])
  y2 = f(X_init[1])
  Y_init = np.array([[y1],[y2]])
  print "First team..."
  gpr1, x1_sample, y1_sample = bayesopt_multi(10, X_init, Y_init, bounds1, bounds2, X1, X2)

  bounds1 = np.array([[2, 25]])
  bounds2 = np.array([[32, 250]])
  X1 = np.arange(bounds1[:, 0], bounds1[:, 1], 0.1).reshape(-1, 1)
  X2 = np.arange(bounds2[:, 0], bounds2[:, 1], 1).reshape(-1, 1)
  x2=np.array(np.meshgrid(X1, X2)).T.reshape(-1,2)
  X_init = x2[np.random.randint(x2.shape[0], size=2)]
  y1 = f(X_init[0])
  y2 = f(X_init[1])
  Y_init = np.array([[y1],[y2]])
  print "Second team..."
  gpr2, x2_sample, y2_sample = bayesopt_multi(10, X_init, Y_init, bounds1, bounds2, X1, X2)
  
  mu1 = gpr1.predict(x1)
  mu1[mu1 < -1] = 0
  mu2 = gpr2.predict(x2)
  mu2[mu2 < -1] = 0

  minimum = mu1.min() if mu1.min() <= mu2.min() else mu2.min()

  params = x1[np.where(mu1==mu1.min())[0][0]] if mu1.min() <= mu2.min() else x2[np.where(mu2==mu2.min())[0][0]]
  
  print "Third team..."
  print "Learning rate:", params[0]
  print "Batch size:", params[1] 

  model = Sequential()

  model.add(Flatten())
  model.add(Dense(15, input_dim = 784, activation='sigmoid'))
  model.add(Dense(10, activation='softmax'))
  sgd=SGD(lr=params[0])
  model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

  history = model.fit(x_subset, y_subset, 
          batch_size=int(params[1]), epochs=10, verbose=1, validation_data = (x_test, y_test))
  
  acc_ch[0,i] = history.history['acc'][-1]
  acc_ch[1,i] = history.history['val_acc'][-1]
  acc_ch[2,i] = minimum
  
  choice = np.random.randint(2)+1
  params = x1[np.where(mu1==mu1.min())[0][0]] if choice==1 else x2[np.where(mu2==mu2.min())[0][0]]
  minimum = mu1.min() if choice==1 else mu2.min()
  
  model = Sequential()
  model.add(Flatten())
  model.add(Dense(15, input_dim = 784, activation='sigmoid'))
  model.add(Dense(10, activation='softmax'))
  sgd=SGD(lr=params[0])
  model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

  history = model.fit(x_subset, y_subset, 
          batch_size=int(params[1]), epochs=10, verbose=1, validation_data = (x_test, y_test))
  
  acc_rn[0,i] = history.history['acc'][-1]
  acc_rn[1,i] = history.history['val_acc'][-1]
  acc_rn[2,i] = minimum

np.save('choose_best1k_1.npy', acc_ch)
np.save('choose_random1k_1.npy', acc_rn)
