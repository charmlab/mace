import copy
import numpy as np
import pandas as pd

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

salary_lambda = 10
salary_multiplier = 10000
balance_multiplier = 2500
mortgage_cutoff = -225000
n = 2000

w = np.array([[1], [5]])
b = mortgage_cutoff


def load_mortgage_data():

  U_0 = (np.random.poisson(salary_lambda, (n, 1)) + np.random.normal(0,1, (n,1))) * salary_multiplier
  U_1 = np.random.normal(0, 1, (n, 1)) * balance_multiplier
  U_1 = 500 * np.round(U_1 / 500)
  X = np.concatenate((U_0, U_1), axis=1).astype(float)
  X = processDataAccordingToGraph(X)
  # Shuffle just in case the random generator from Poisson
  # distributions gets skewed as more samples are generated
  np.random.shuffle(X) # must happen before we assign labels
  y = (np.sign(np.sign(np.dot(X, w) + b) + 1e-6) + 1) / 2 # add 1e-3 to prevent label 0.5

  X_train = X[ : n // 2, :]
  X_test = X[n // 2 : , :]
  y_train = y[ : n // 2, :]
  y_test = y[n // 2 : , :]

  X_test[0,:] = np.array([[75000, 25000]])

  # return w, b, X_train, y_train, X_test, y_test
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


def processDataAccordingToGraph(data):
  # We assume the model below
  # X_0 := U_0 \\ annual salary
  # X_1 := X_0 / 5 +  U_1 \\
  # U_0 ~ Poisson(salary_lambda) * salary_multiplier
  # U_1 ~ Poisson(balance_lambda) * balance_multiplier
  # data = copy.deepcopy(data)
  data[:,0] = data[:,0]
  data[:,1] = data[:,1] + data[:,0] * 3 / 10.
  return data

