import copy
import numpy as np
import pandas as pd

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

from debug import ipsh

mu_x, sigma_x = 0, 1 # mean and standard deviation for data
mu_w, sigma_w = 0, 1 # mean and standard deviation for weights
n = 10000
d = 3

w = np.random.normal(mu_w, sigma_w, (d, 1))
# b = 0 # see below.

def load_random_data(scm_class = 'nonlinear'):

  X = np.random.normal(mu_x, sigma_x, (n, d))
  X = processDataAccordingToGraph(X, scm_class)
  np.random.shuffle(X)
  # to create a more balanced dataset, do not set b to 0.
  b = - np.mean(np.dot(X, w))
  y = (np.sign(np.sign(np.dot(X, w) + b) + 1e-6) + 1) / 2 # add 1e-3 to prevent label 0.5

  X_train = X[ : n // 2, :]
  X_test = X[n // 2 : , :]
  y_train = y[ : n // 2, :]
  y_test = y[n // 2 : , :]

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


def processDataAccordingToGraph(data, scm_class = 'nonlinear'):
  if scm_class == 'linear':
    # We assume the model below
    # X_1 := U_1 \\
    # X_2 := X_1 + 1 + U_2 \\
    # X_3 := (X_1 - 1) / 4 + np.sqrt{3} * X_2 + U_3
    # U_i ~ \forall ~ i \in [3] \sim \mathcal{N}(0,1)
    data = copy.deepcopy(data)
    data[:,0] = data[:,0]
    data[:,1] += data[:,0] + np.ones((n))
    data[:,2] += (data[:,0] - 1)/4 + np.sqrt(3) * data[:,1]
  elif scm_class == 'nonlinear':
    # We assume the model below
    data = copy.deepcopy(data)
    data[:,0] = data[:,0]
    data[:,1] += data[:,0] + np.ones((n))
    data[:,2] += np.sqrt(3) * data[:,0] * np.power(data[:,1], 2)
  return data


# import numpy as np
# import pandas as pd

# import loadData

# from random import seed
# RANDOM_SEED = 54321
# seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
# np.random.seed(RANDOM_SEED)

# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso

# dataset_obj = loadData.loadDataset('random', return_one_hot = False, load_from_cache = False)
# df = dataset_obj.data_frame_kurz

# # See Figure 3 in paper

# # node: x1     parents: {x0}
# X_train = df[['x0']]
# y_train = df[['x1']]
# model_pretrain = LinearRegression()
# # model_pretrain = Lasso()
# model_trained = model_pretrain.fit(X_train, y_train)
# print(model_trained.coef_)
# print(model_trained.intercept_)

# # node: x2     parents: {x1,  x2}
# X_train = df[['x0', 'x1']]
# y_train = df[['x2']]
# model_pretrain = LinearRegression()
# model_trained = model_pretrain.fit(X_train, y_train)
# print(model_trained.coef_)
# print(model_trained.intercept_)
# #
