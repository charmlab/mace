import copy
import numpy as np
import pandas as pd
from sklearn import datasets

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

n_samples = 25000

def load_twomoon_data(variable_type = 'real'):

  # https://rohitmidha23.github.io/Neural-Network-Decision-Boundary/
  X, y = datasets.make_moons(n_samples=n_samples, noise=0.1, random_state=0)
  if variable_type == 'integer':
    X = np.round(4 * X)
  y = y.reshape(-1,1)
  # X = processDataAccordingToGraph(X) # TODO...

  X_train = X[ : n_samples // 2, :]
  X_test = X[n_samples // 2 : , :]
  y_train = y[ : n_samples // 2, :]
  y_test = y[n_samples // 2 : , :]


  data_frame_non_hot = pd.DataFrame(
      np.concatenate((
        np.concatenate((y_train, X_train), axis = 1), # importantly, labels have to go first, else Dataset.__init__ messes up kurz column names
        np.concatenate((y_test, X_test), axis = 1), # importantly, labels have to go first, else Dataset.__init__ messes up kurz column names
      ),
      axis = 0,
    ),
    columns=['label', 'x0', 'x1']
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

