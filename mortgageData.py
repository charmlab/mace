import copy
import numpy as np
np.random.seed(54321)

salary_lambda = 8
salary_multiplier = 10000
balance_lambda = 3
balance_multiplier = 2500
mortgage_cutoff = -225000

n = 2000

def getExperimentParams():
  w = np.array([[1], [5]])

  X_0 = np.random.poisson(salary_lambda, (n, 1)) * salary_multiplier
  X_1 = np.random.poisson(balance_lambda, (n, 1)) * balance_multiplier
  X = np.concatenate((X_0, X_1), axis=1).astype(float)
  X = processDataAccordingToGraph(X)
  # Shuffle just in case the random generator from Poisson
  # distributions gets skewed as more samples are generated
  np.random.shuffle(X) # must happen before we assign labels
  y = (np.sign(np.dot(X, w) + mortgage_cutoff + 1e-3) + 1) / 2 # add 1e-3 to prevent label 0.5

  X_train = X[ : n // 2, :]
  X_test = X[n // 2 : , :]
  y_train = y[ : n // 2, :]
  y_test = y[n // 2 : , :]

  X_test[0,:] = np.array([[75000, 25000]])

  return w, X_train, y_train, X_test, y_test

def processDataAccordingToGraph(data):
  # We assume the model below
  # X_0 := U_0 \\ annual salary
  # X_1 := X_0 / 5 +  U_1 \\
  # U_0 ~ Poisson(salary_lambda) * salary_multiplier
  # U_1 ~ Poisson(balance_lambda) * balance_multiplier
  data = copy.deepcopy(data)
  data[:,0] = data[:,0]
  data[:,1] += data[:,0] * 3 / 10.
  return data





