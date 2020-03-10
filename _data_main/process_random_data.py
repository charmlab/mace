import copy
import numpy as np
import pandas as pd

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)


mu_x, sigma_x = 0, 1 # mean and standard deviation for data
mu_w, sigma_w = 0, 1 # mean and standard deviation for weights
n = 1000
d = 3

def getExperimentParams():
  w = np.random.normal(mu_w, sigma_w, (d, 1))

  X = np.random.normal(mu_x, sigma_x, (n, d))
  X = processDataAccordingToGraph(X)
  np.random.shuffle(X)
  y = (np.sign(np.dot(X, w)) + 1) / 2

  X_train = X[ : n // 2, :]
  X_test = X[n // 2 : , :]
  y_train = y[ : n // 2, :]
  y_test = y[n // 2 : , :]

  b = 0
  return w, b, X_train, y_train, X_test, y_test


def processDataAccordingToGraph(data):
  # We assume the model below
  # X_1 := U_1 \\
  # X_2 := X_1 + 1 + U_2 \\
  # X_3 := (X_1 - 1) / 4 + np.sqrt{3} * X_2 + U_3
  # U_i ~ \forall ~ i \in [3] \sim \mathcal{N}(0,1)
  data = copy.deepcopy(data)
  data[:,0] = data[:,0]
  data[:,1] += data[:,0] + np.ones((n))
  data[:,2] += (data[:,0] - 1)/4 + np.sqrt(3) * data[:,1]
  return data


def load_random_data():
  w, b, X_train, y_train, X_test, y_test = getExperimentParams()
  # print('w:\n', w)
  # print('X_test[0:5]:\n', X_test[0:5])
  data_frame_non_hot = pd.DataFrame(
      np.concatenate((
        np.concatenate((y_train, X_train), axis = 1), # importantly, labels have to go first, else Dataset.__init__ messes up kurz column names
        np.concatenate((y_test, X_test), axis = 1), # importantly, labels have to go first, else Dataset.__init__ messes up kurz column names
      ),
      axis = 0,
    ),
    columns=['label', 'x0', 'x1', 'x2']
  )
  return data_frame_non_hot.astype('float64')


def load_random_model():
  w, b, X_train, y_train, X_test, y_test = getExperimentParams()
  return w, b


from pysmt.shortcuts import *
from pysmt.typing import *

def getRandomCausalConsistencyConstraints(model_symbols, factual_sample):
  a = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        model_symbols['interventional']['x0']['symbol'],
        factual_sample['x0'],
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      model_symbols['counterfactual']['x0']['symbol'],
      model_symbols['interventional']['x0']['symbol'],
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      model_symbols['counterfactual']['x0']['symbol'],
      factual_sample['x0'],
    ),
  )

  b = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        model_symbols['interventional']['x1']['symbol'],
        factual_sample['x1'],
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      model_symbols['counterfactual']['x1']['symbol'],
      model_symbols['interventional']['x1']['symbol'],
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      model_symbols['counterfactual']['x1']['symbol'],
      Plus(
        factual_sample['x1'],
        Minus(
          model_symbols['counterfactual']['x0']['symbol'],
          factual_sample['x0'],
        )
      )
    ),
  )

  c = Ite(
    Not( # if YES intervened
      EqualsOrIff(
        model_symbols['interventional']['x2']['symbol'],
        factual_sample['x2'],
      )
    ),
    EqualsOrIff( # set value of X^CF to the intervened value
      model_symbols['counterfactual']['x2']['symbol'],
      model_symbols['interventional']['x2']['symbol'],
    ),
    EqualsOrIff( # else, set value of X^CF to (8) from paper
      model_symbols['counterfactual']['x2']['symbol'],
      Plus(
        factual_sample['x2'],
        Minus(
          Plus(
            Times(
              Minus(
                model_symbols['counterfactual']['x0']['symbol'],
                Real(1)
              ),
              Real(0.25)
            ),
            Times(
              model_symbols['counterfactual']['x1']['symbol'],
              Real(float(np.sqrt(3)))
            )
          ),
          Plus(
            Times(
              Minus(
                factual_sample['x0'],
                Real(1)
              ),
              Real(0.25)
            ),
            Times(
              factual_sample['x1'],
              Real(float(np.sqrt(3)))
            )
          ),
        )
      )
    ),
  )

  return And([a,b,c])
