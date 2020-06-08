import copy
import numpy as np
import pandas as pd
from sklearn import datasets

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

n_samples = 2000

def load_test_data(variable_type = 'real'):

  X = np.random.randint(1, 5, n_samples)
  y = np.array([1 if x % 2 else 0 for x in X]) # y = 1 for even values, y = 0 for odd values

  # make 2D
  X = X.reshape(X.shape[0], -1)
  y = y.reshape(y.shape[0], -1)

  X_train = X[ : n_samples // 2]
  X_test = X[n_samples // 2 : ]
  y_train = y[ : n_samples // 2]
  y_test = y[n_samples // 2 : ]


  data_frame_non_hot = pd.DataFrame(
      np.concatenate((
        np.concatenate((y_train, X_train), axis = 1), # importantly, labels have to go first, else Dataset.__init__ messes up kurz column names
        np.concatenate((y_test, X_test), axis = 1), # importantly, labels have to go first, else Dataset.__init__ messes up kurz column names
      ),
      axis = 0,
    ),
    columns=['label', 'x0']
  )
  return data_frame_non_hot.astype('float64')


# def processDataAccordingToGraph(data):
#   # We assume the model below
#   # X_0 := U_0 \\ annual salary
#   # X_1 := X_0 / 5 +  U_1 \\
#   # U_0 ~ Poisson(salary_lambda) * salary_multiplier
#   # U_1 ~ Poisson(balance_lambda) * balance_multiplier
#   # data = copy.deepcopy(data)
#   data[:,0] = data[:,0]
#   data[:,1] = data[:,1] + data[:,0] * 3 / 10.
#   return data

