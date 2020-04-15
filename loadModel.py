def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn # to ignore all warnings.

import sys
import pickle
import numpy as np

import utils
import loadData

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

# TODO: change to be like _data_main below, and make python module
# this answer https://stackoverflow.com/a/50474562 and others
try:
  import treeUtils
except:
  print('[ENV WARNING] fair_adult_data not available')

SIMPLIFY_TREES = False


# def convertSklearnDtypeToPytorch(input_obj):
#   # return input_obj
#   # isinstance(input_obj, (np.ndarray)) # TODO complete
#   return np.around(input_obj, 4).astype('float32')

@utils.Memoize
def loadModelForDataset(model_class, dataset_string, experiment_folder_name = None):

  if experiment_folder_name != None:
    log_file = open(f'{experiment_folder_name}/log_training.txt','w')
  else:
    log_file = sys.stdout

  if not (model_class in {'lr', 'mlp', 'tree', 'forest'}):
    raise Exception(f'{model_class} not supported.')

  if not (dataset_string in {'random', 'mortgage', 'twomoon', 'german', 'credit', 'compass', 'adult'}):
    raise Exception(f'{dataset_string} not supported.')

  dataset_obj = loadData.loadDataset(dataset_string, return_one_hot = True, load_from_cache = True, debug_flag = False)
  X_train, X_test, y_train, y_test = loadData.getTrainTestData(dataset_obj, RANDOM_SEED, standardize_data = False)
  feature_names = dataset_obj.getInputAttributeNames('kurz') # easier to read (nothing to do with one-hot vs non-hit!)


  if model_class == 'tree':
    model_pretrain = DecisionTreeClassifier()
  elif model_class == 'forest':
    model_pretrain = RandomForestClassifier()
  elif model_class == 'lr':
    # IMPORTANT: The default solver changed from ‘liblinear’ to ‘lbfgs’ in 0.22. Results may differ slightly from paper.
    model_pretrain = LogisticRegression(penalty='l2') # default
  elif model_class == 'mlp':
    model_pretrain = MLPClassifier(hidden_layer_sizes = (10, 10))


  print('[INFO] Training `{}` on {:,} samples (%{:.2f} of {:,} samples)...'.format(model_class, X_train.shape[0], 100 * X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]), X_train.shape[0] + X_test.shape[0]), file=log_file)
  model_trained = model_pretrain.fit(X_train, y_train)
  print('\tTraining accuracy: %{:.2f}'.format(accuracy_score(y_train, model_trained.predict(X_train)) * 100), file=log_file)
  print('\tTesting accuracy: %{:.2f}'.format(accuracy_score(y_test, model_trained.predict(X_test)) * 100), file=log_file)
  print('[INFO] done.\n', file=log_file)

  # # (PERHAPS?) need to do this to match pytorch's dtypes for proper comparison
  # if model_class == 'lr':

  #   # the returned model_trained for 'mortgage' has 100% train & test accuracy.
  #   if dataset_string == 'mortgage':
  #     mortgage_cutoff = -225000
  #     w = np.array([[1], [5]]).T
  #     b = np.ones(1) * mortgage_cutoff
  #   elif dataset_string in {'random', 'twomoon', 'german', 'credit', 'compass', 'adult'}:
  #     w = model_trained.coef_
  #     b = model_trained.intercept_

  #   model_trained.coef_ = convertSklearnDtypeToPytorch(w)
  #   model_trained.intercept_ = convertSklearnDtypeToPytorch(b)

  # elif model_class == 'mlp':

  #   for i in range(len(model_trained.coefs_)):
  #     model_trained.coefs_[i] = convertSklearnDtypeToPytorch(model_trained.coefs_[i])
  #     model_trained.intercepts_[i] = convertSklearnDtypeToPytorch(model_trained.intercepts_[i])

  if model_class == 'tree':
    if SIMPLIFY_TREES:
      print('[INFO] Simplifying decision tree...', end = '', file=log_file)
      model_trained.tree_ = treeUtils.simplifyDecisionTree(model_trained, False)
      print('\tdone.', file=log_file)
    treeUtils.saveTreeVisualization(model_trained, model_class, '', X_test, feature_names, experiment_folder_name)
  elif model_class == 'forest':
    for tree_idx in range(len(model_trained.estimators_)):
      if SIMPLIFY_TREES:
        print(f'[INFO] Simplifying decision tree (#{tree_idx + 1}/{len(model_trained.estimators_)})...', end = '', file=log_file)
        model_trained.estimators_[tree_idx].tree_ = treeUtils.simplifyDecisionTree(model_trained.estimators_[tree_idx], False)
        print('\tdone.', file=log_file)
      treeUtils.saveTreeVisualization(model_trained.estimators_[tree_idx], model_class, f'tree{tree_idx}', X_test, feature_names, experiment_folder_name)

  pickle.dump(model_trained, open(f'{experiment_folder_name}/_model_trained', 'wb'))
  return model_trained









