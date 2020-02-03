import copy
import numpy as np
np.random.seed(54321)

salary_lambda = 6
salary_multiplier = 10000
balance_lambda = 3
balance_multiplier = 100
mortgage_cutoff = -225000

n = 1000

def getExperimentParams():
  w = np.array([[1], [5]])

  X_train_0 = np.random.poisson(salary_lambda, (n,1)) * salary_multiplier
  X_train_1 = np.random.poisson(balance_lambda, (n,1)) * balance_multiplier
  X_train = np.concatenate((X_train_0, X_train_1), axis=1).astype(float)
  X_train = processDataAccordingToGraph(X_train)
  y_train = (np.sign(np.dot(X_train, w) + mortgage_cutoff) + 1) / 2

  X_test_sample = np.array([[75000, 25000]])
  X_test_0 = np.random.poisson(salary_lambda, (n - 1, 1)) * salary_multiplier
  X_test_1 = np.random.poisson(balance_lambda, (n - 1, 1)) * balance_multiplier
  X_test = np.concatenate((X_test_0, X_test_1), axis=1).astype(float)
  X_test = processDataAccordingToGraph(X_test)
  X_test = np.concatenate((X_test_sample, X_test), axis=0).astype(float)
  # X_test = processDataAccordingToGraph(X_test)
  y_test = (np.sign(np.dot(X_test, w) - mortgage_cutoff) + 1) / 2

  return w, X_train, y_train, X_test, y_test

def processDataAccordingToGraph(data):
  # We assume the model below
  # X_0 := U_0 \\ annual salary
  # X_1 := X_0 / 5 +  U_1 \\
  # U_0 ~ Poisson(salary_lambda) * salary_multiplier
  # U_1 ~ Poisson(balance_lambda) * balance_multiplier
  data = copy.deepcopy(data)
  data[:,0] = data[:,0]
  data[:,1] += data[:,0] / 5.
  return data
