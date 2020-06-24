def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn # to ignore all warnings.

import sys
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
  print('[ENV WARNING] treeUtils not available')

SIMPLIFY_TREES = False

@utils.Memoize
def loadModelForDataset(model_class, dataset_string, scm_class = None, experiment_folder_name = None):

  log_file = sys.stdout if experiment_folder_name == None else open(f'{experiment_folder_name}/log_training.txt','w')

  if not (model_class in {'lr', 'mlp', 'tree', 'forest'}):
    raise Exception(f'{model_class} not supported.')

  if not (dataset_string in {'synthetic', 'mortgage', 'twomoon', 'german', 'credit', 'compass', 'adult', 'test'}):
    raise Exception(f'{dataset_string} not supported.')

  dataset_obj = loadData.loadDataset(dataset_string, return_one_hot = True, load_from_cache = False, meta_param = scm_class)
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  X_all = pd.concat([X_train, X_test], axis = 0)
  y_all = pd.concat([y_train, y_test], axis = 0)
  assert sum(y_all) / len(y_all) == 0.5, 'Expected class balance should be 50/50%.'
  feature_names = dataset_obj.getInputAttributeNames('kurz') # easier to read (nothing to do with one-hot vs non-hit!)

  if model_class == 'tree':
    model_pretrain = DecisionTreeClassifier()
  elif model_class == 'forest':
    model_pretrain = RandomForestClassifier()
  elif model_class == 'lr':
    # IMPORTANT: The default solver changed from ‘liblinear’ to ‘lbfgs’ in 0.22;
    #            therefore, results may differ slightly from paper.
    model_pretrain = LogisticRegression() # default penalty='l2', i.e., ridge
  elif model_class == 'mlp':
    model_pretrain = MLPClassifier(hidden_layer_sizes = (10, 10))

  tmp_text = f'[INFO] Training `{model_class}` on {X_train.shape[0]:,} samples ' + \
    f'(%{100 * X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]):.2f} ' + \
    f'of {X_train.shape[0] + X_test.shape[0]:,} samples)...'
  print(tmp_text)
  print(tmp_text, file=log_file)
  model_trained = model_pretrain.fit(X_train, y_train)
  print(f'\tTraining accuracy: %{accuracy_score(y_train, model_trained.predict(X_train)) * 100:.2f}', file=log_file)
  print(f'\tTesting accuracy: %{accuracy_score(y_test, model_trained.predict(X_test)) * 100:.2f}', file=log_file)
  print(f'\tTraining accuracy: %{accuracy_score(y_train, model_trained.predict(X_train)) * 100:.2f}')
  print(f'\tTesting accuracy: %{accuracy_score(y_test, model_trained.predict(X_test)) * 100:.2f}')
  print('[INFO] done.\n', file=log_file)
  print('[INFO] done.\n')
  assert accuracy_score(y_train, model_trained.predict(X_train)) > 0.70

  classifier_obj = model_trained
  visualizeDatasetAndFixedModel(dataset_obj, classifier_obj, experiment_folder_name)

  if model_class == 'tree':
    if SIMPLIFY_TREES:
      print('[INFO] Simplifying decision tree...', end = '', file=log_file)
      model_trained.tree_ = treeUtils.simplifyDecisionTree(model_trained, False)
      print('\tdone.', file=log_file)
    # treeUtils.saveTreeVisualization(model_trained, model_class, '', X_test, feature_names, experiment_folder_name)
  elif model_class == 'forest':
    for tree_idx in range(len(model_trained.estimators_)):
      if SIMPLIFY_TREES:
        print(f'[INFO] Simplifying decision tree (#{tree_idx + 1}/{len(model_trained.estimators_)})...', end = '', file=log_file)
        model_trained.estimators_[tree_idx].tree_ = treeUtils.simplifyDecisionTree(model_trained.estimators_[tree_idx], False)
        print('\tdone.', file=log_file)
      # treeUtils.saveTreeVisualization(model_trained.estimators_[tree_idx], model_class, f'tree{tree_idx}', X_test, feature_names, experiment_folder_name)

  if experiment_folder_name:
    pickle.dump(model_trained, open(f'{experiment_folder_name}/_model_trained', 'wb'))

  return model_trained


def scatterDataset(dataset_obj, classifier_obj, ax):
  assert len(dataset_obj.getInputAttributeNames()) <= 3
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  X_train_numpy = X_train.to_numpy()
  X_test_numpy = X_test.to_numpy()
  y_train = y_train.to_numpy()
  y_test = y_test.to_numpy()
  number_of_samples_to_plot = min(200, X_train_numpy.shape[0], X_test_numpy.shape[0])
  for idx in range(number_of_samples_to_plot):
    color_train = 'black' if y_train[idx] == 1 else 'magenta'
    color_test = 'black' if y_test[idx] == 1 else 'magenta'
    if X_train.shape[1] == 2:
      ax.scatter(X_train_numpy[idx, 0], X_train_numpy[idx, 1], marker='s', color=color_train, alpha=0.2, s=10)
      ax.scatter(X_test_numpy[idx, 0], X_test_numpy[idx, 1], marker='o', color=color_test, alpha=0.2, s=15)
    elif X_train.shape[1] == 3:
      ax.scatter(X_train_numpy[idx, 0], X_train_numpy[idx, 1], X_train_numpy[idx, 2], marker='s', color=color_train, alpha=0.2, s=10)
      ax.scatter(X_test_numpy[idx, 0], X_test_numpy[idx, 1], X_test_numpy[idx, 2], marker='o', color=color_test, alpha=0.2, s=15)


def scatterDecisionBoundary(dataset_obj, classifier_obj, ax):

  if len(dataset_obj.getInputAttributeNames()) == 2:

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    X = np.linspace(ax.get_xlim()[0] - x_range / 10, ax.get_xlim()[1] + x_range / 10, 1000)
    Y = np.linspace(ax.get_ylim()[0] - y_range / 10, ax.get_ylim()[1] + y_range / 10, 1000)
    X, Y = np.meshgrid(X, Y)
    Xp = X.ravel()
    Yp = Y.ravel()

    # if normalized_fixed_model is False:
    #   labels = classifier_obj.predict(np.c_[Xp, Yp])
    # else:
    #   Xp = (Xp - dataset_obj.attributes_kurz['x0'].lower_bound) / \
    #        (dataset_obj.attributes_kurz['x0'].upper_bound - dataset_obj.attributes_kurz['x0'].lower_bound)
    #   Yp = (Yp - dataset_obj.attributes_kurz['x1'].lower_bound) / \
    #        (dataset_obj.attributes_kurz['x1'].upper_bound - dataset_obj.attributes_kurz['x1'].lower_bound)
    #   labels = classifier_obj.predict(np.c_[Xp, Yp])
    labels = classifier_obj.predict(np.c_[Xp, Yp])
    Z = labels.reshape(X.shape)

    cmap = plt.get_cmap('Paired')
    ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5)

  elif len(dataset_obj.getInputAttributeNames()) == 3:

    fixed_model_w = classifier_obj.coef_
    fixed_model_b = classifier_obj.intercept_

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    X = np.linspace(ax.get_xlim()[0] - x_range / 10, ax.get_xlim()[1] + x_range / 10, 10)
    Y = np.linspace(ax.get_ylim()[0] - y_range / 10, ax.get_ylim()[1] + y_range / 10, 10)
    X, Y = np.meshgrid(X, Y)
    Z = - (fixed_model_w[0][0] * X + fixed_model_w[0][1] * Y + fixed_model_b) / fixed_model_w[0][2]

    surf = ax.plot_wireframe(X, Y, Z, alpha=0.3)



def visualizeDatasetAndFixedModel(dataset_obj, classifier_obj, experiment_folder_name):

  if not len(dataset_obj.getInputAttributeNames()) <= 3:
    return

  fig = plt.figure()
  if len(dataset_obj.getInputAttributeNames()) == 2:
    ax = plt.subplot()
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid()
  elif len(dataset_obj.getInputAttributeNames()) == 3:
    ax = plt.subplot(1, 1, 1, projection = '3d')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.view_init(elev=10, azim=-20)

  scatterDataset(dataset_obj, classifier_obj, ax)
  scatterDecisionBoundary(dataset_obj, classifier_obj, ax)

  ax.set_title(f'{dataset_obj.dataset_name}')
  ax.grid(True)

  # plt.show()
  plt.savefig(f'{experiment_folder_name}/_dataset_and_model.pdf')
  plt.close()







