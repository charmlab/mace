import numpy as np
np.random.seed(54321)

mu_x, sigma_x = 0, 1 # mean and standard deviation for data
mu_w, sigma_w = 0, 1 # mean and standard deviation for weights
n = 1000
d = 3

def getExperimentParams():
  w = np.random.normal(mu_w, sigma_w, (d,1))

  X_train = np.random.normal(mu_x, sigma_x, (n,d))
  y_train = (np.sign(np.dot(X_train, w)) + 1) / 2

  X_test = np.random.normal(mu_x, sigma_x, (n,d))
  y_test = (np.sign(np.dot(X_test, w)) + 1) / 2

  return w, X_train, y_train, X_test, y_test

def ell2(a, b):
  np.linalg.norm(a - b, 2)

# Ok found the issue(s) in my code;
# 1) [solved?] seg fault if I remove print statements
# 2) assert throwing error; Counterfactual prediction does not match sklearn prediction.
# 3) ordering of my test samples vs Patrick's
# 4) why is 0-norm wierd?
