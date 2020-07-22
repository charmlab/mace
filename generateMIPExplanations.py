import sys
import time
import copy
import pickle
import numpy as np
import pandas as pd
import normalizedDistance
import torch
import gurobipy as grb

from modelConversion import *
from pysmt.shortcuts import *
from pysmt.typing import *
from pprint import pprint

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from network_linear_approximation import LinearizedNetwork
from mip_solver import MIPNetwork
from applyMIPConstraints import *

from debug import ipsh

from random import seed
RANDOM_SEED = 1122334455
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

# DEBUG_FLAG = True
DEBUG_FLAG = False


def getTorchFromSklearn(sklearn_model, input_dim, no_relu=False):
  model_width = 10  # TODO make more dynamic later and move to separate function
  if sklearn_model.hidden_layer_sizes == model_width:
    n_hidden_layers = 1
  else:
    n_hidden_layers = len(sklearn_model.hidden_layer_sizes)

  if n_hidden_layers == 1:
    assert sklearn_model.hidden_layer_sizes == (model_width)
    if no_relu:
      torch_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, 1))
    else:
      torch_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, 1),
        torch.nn.ReLU())

  elif n_hidden_layers == 2:
    assert sklearn_model.hidden_layer_sizes == (model_width, model_width)
    if no_relu:
      torch_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, 1))
    else:
      torch_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, 1),
        torch.nn.ReLU())

  elif n_hidden_layers == 3:
    assert sklearn_model.hidden_layer_sizes == (model_width, model_width, model_width)
    if no_relu:
      torch_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, 1))
    else:
      torch_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, 1),
        torch.nn.ReLU())

  elif n_hidden_layers == 4:
    assert sklearn_model.hidden_layer_sizes == (model_width, model_width, model_width, model_width)
    if no_relu:
      torch_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, 1))
    else:
      torch_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, model_width),
        torch.nn.ReLU(),
        torch.nn.Linear(model_width, 1),
        torch.nn.ReLU())

  else:
    raise Exception("MLP model not supported")

  for i in range(n_hidden_layers + 1):
    torch_model[2*i].weight = torch.nn.Parameter(torch.FloatTensor(sklearn_model.coefs_[i].astype('float64')).t(),
                                                 requires_grad=False)
  for i in range(n_hidden_layers + 1):
    torch_model[2*i].bias = torch.nn.Parameter(torch.FloatTensor(sklearn_model.intercepts_[i].astype('float64')),
                                             requires_grad=False)

  return torch_model


def solveMIP(sklearn_model, dataset_obj, factual_sample, norm_type, norm_lower, norm_upper):
  assert isinstance(sklearn_model, MLPClassifier), "Only MLP model supports the linear relaxation."
  input_dim = len(dataset_obj.getInputAttributeNames('kurz'))

  # First, translate sklearn model to PyTorch model
  torch_model = getTorchFromSklearn(sklearn_model, input_dim, no_relu=True)

  # Now create a linearized network
  layers = [module for module in torch_model.modules() if type(module) != torch.nn.Sequential]
  mip_net = MIPNetwork(layers)

  # Get input domains
  domains = np.zeros((input_dim, 2), dtype=np.float32)
  for i, attr_name_kurz in enumerate(dataset_obj.getInputAttributeNames('kurz')):
    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
    domains[i][0] = attr_obj.lower_bound
    domains[i][1] = attr_obj.upper_bound
  domains = torch.from_numpy(domains)

  # Setup MIP model and check bounds feasibility w.r.t. distance formula
  feasible = mip_net.setup_model(domains, factual_sample, dataset_obj, norm_type, norm_lower, norm_upper,
                                 sym_bounds=False, dist_as_constr=not('obj' in norm_type), bounds='opt')
  if not feasible:
    return False, None

  # Solve the MIP
  solved, sol, _ = mip_net.solve(domains, factual_sample)

  return solved, sol


def findClosestCounterfactualSample(mip_model, model_trained, model_vars, dataset_obj, factual_sample, norm_type, approach_string, epsilon, log_file):

  def getCenterNormThresholdInRange(lower_bound, upper_bound):
    return (lower_bound + upper_bound) / 2

  def assertPrediction(dict_sample, model_trained, dataset_obj):
    vectorized_sample = []
    for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
      vectorized_sample.append(dict_sample[attr_name_kurz])

    sklearn_prediction = int(model_trained.predict([vectorized_sample])[0])
    pysmt_prediction = int(dict_sample['y'])
    factual_prediction = int(factual_sample['y'])

    # IMPORTANT: sometimes, MACE does such a good job, that the counterfactual
    #            ends up super close to (if not on) the decision boundary; here
    #            the label is underfined which causes inconsistency errors
    #            between pysmt and sklearn. We skip the assert at such points.
    class_predict_proba = model_trained.predict_proba([vectorized_sample])[0]
    # print(class_predict_proba)
    if np.abs(class_predict_proba[0] - class_predict_proba[1]) < 1e-8:
      return

    if isinstance(model_trained, LogisticRegression):
      if np.dot(model_trained.coef_, vectorized_sample) + model_trained.intercept_ < 1e-10:
        return

    assert sklearn_prediction == pysmt_prediction, f'Pysmt prediction does not match sklearn prediction. \n{dict_sample} \n{factual_sample}'
    assert sklearn_prediction != factual_prediction, 'Counterfactual and factual samples have the same prediction.'


  counterfactuals = [] # list of tuples (samples, distances)
  # In case no counterfactuals are found (this could happen for a variety of
  # reasons, perhaps due to non-plausibility), return a template counterfactual
  counterfactuals.append({
    'counterfactual_sample': {},
    'counterfactual_distance': np.infty,
    'interventional_sample': {},
    'interventional_distance': np.infty,
    'time': np.infty,
    'norm_type': norm_type})

  norm_type = norm_type + '_obj'

  if isinstance(model_trained, MLPClassifier):
    # TODO: remove if and make generic
    solved, counterfactual_sample = solveMIP(model_trained, dataset_obj, factual_sample, norm_type, 0, 0)
    assert solved is True
  else:
    # mip_model.setParam('FeasibilityTol', 1e-9)
    # mip_model.setParam('OptimalityTol', 1e-9)
    # mip_model.setParam('IntFeassTol', 1e-9)
    applyPlausibilityConstrs(mip_model, dataset_obj)
    mip_model.update()
    applyDistanceConstrs(mip_model, dataset_obj, factual_sample, norm_type)
    mip_model.update()
    applyTrainedModelConstrs(mip_model, model_vars, model_trained)
    mip_model.update()
    mip_model.addConstr(model_vars['output']['y']['var'] == 1 - int(factual_sample['y']))
    mip_model.setObjective(mip_model.getVarByName('normalized_distance'), grb.GRB.MINIMIZE)
    mip_model.update()
    mip_model.reset()
    print("about to optimize")
    mip_model.optimize()
    print("did optimize")
    if mip_model.status is grb.GRB.INFEASIBLE:
      print("infeasible")
      print(mip_model.computeIIS())
      exit(0)
    assert mip_model.status is grb.GRB.OPTIMAL, f"Model status is not optimal but: {mip_model.status}"
    # Get the input that gives the optimal solution.
    counterfactual_sample = {}
    for feature_name in model_vars['counterfactual'].keys():
      var = model_vars['counterfactual'][feature_name]['var']
      counterfactual_sample[feature_name] = var.x
    counterfactual_sample['y'] = model_vars['output']['y']['var'].x
    # print("logit out found: ", mip_model.getVarByName('logit_out').x)
    # print("optimal distance: ", mip_model.getVarByName('normalized_distance').x)

  norm_type = norm_type.replace('_obj', '')

  # Assert samples have correct prediction label according to sklearn model
  assertPrediction(counterfactual_sample, model_trained, dataset_obj)
  counterfactual_distance = normalizedDistance.getDistanceBetweenSamples(
    factual_sample,
    counterfactual_sample,
    norm_type,
    dataset_obj)

  counterfactuals.append({
    'counterfactual_sample': counterfactual_sample,
    'counterfactual_distance': counterfactual_distance,
    'time': None,
    'norm_type': norm_type})

  closest_counterfactual_sample = sorted(counterfactuals, key=lambda x: x['counterfactual_distance'])[0]
  # closest_interventional_sample = sorted(counterfactuals, key = lambda x: x['interventional_distance'])[0]
  closest_interventional_sample = None

  return counterfactuals, closest_counterfactual_sample, closest_interventional_sample


def getPrettyStringForSampleDictionary(sample, dataset_obj):

  if len(sample.keys()) == 0 :
    return 'No sample found.'

  key_value_pairs_with_x_in_key = {}
  key_value_pairs_with_y_in_key = {}
  for key, value in sample.items():
    if key in dataset_obj.getInputAttributeNames('kurz'):
      key_value_pairs_with_x_in_key[key] = value
    elif key in dataset_obj.getOutputAttributeNames('kurz'):
      key_value_pairs_with_y_in_key[key] = value
    else:
      raise Exception('Sample keys may only be `x` or `y`.')

  assert \
    len(key_value_pairs_with_y_in_key.keys()) == 1, \
    f'expecting only 1 output variables, got {len(key_value_pairs_with_y_in_key.keys())}'

  all_key_value_pairs = []
  for key, value in sorted(key_value_pairs_with_x_in_key.items(), key = lambda x: int(x[0][1:].split('_')[0])):
    all_key_value_pairs.append(f'{key} : {value}')
  all_key_value_pairs.append(f"{'y'}: {key_value_pairs_with_y_in_key['y']}")

  return f"{{{', '.join(all_key_value_pairs)}}}"


def genExp(
  explanation_file_name,
  model_trained,
  dataset_obj,
  factual_sample,
  norm_type,
  approach_string,
  epsilon):

  # # ONLY TO BE USED FOR TEST PURPOSES ON MORTGAGE DATASET
  # factual_sample = {'x0': 75000, 'x1': 25000, 'y': False}

  if 'mace' not in approach_string and 'mint' not in approach_string:
    raise Exception(f'`{approach_string}` not recognized as valid approach string; expected `mint` or `mace`.')

  if DEBUG_FLAG:
    log_file = sys.stdout
  else:
    log_file = open(explanation_file_name, 'w')

  # Initial model
  mip_model = grb.Model()
  mip_model.setParam('OutputFlag', False)
  mip_model.setParam('Threads', 1)

  # Initial params
  model_vars = {
    'counterfactual': {},
    'interventional': {},
    'output': {'y': {'var': mip_model.addVar(obj=0, vtype=grb.GRB.BINARY, name='y')}}
  }

  # Populate model_vars['counterfactual'] using the
  # parameters saved during training
  for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
    lower_bound = attr_obj.lower_bound
    upper_bound = attr_obj.upper_bound
    # print(f'\n attr_name_kurz: {attr_name_kurz} \t\t lower_bound: {lower_bound} \t upper_bound: {upper_bound}', file = log_file)
    if attr_name_kurz not in dataset_obj.getInputAttributeNames('kurz'):
      continue # do not overwrite the output
    if attr_obj.attr_type == 'numeric-real':
      model_vars['counterfactual'][attr_name_kurz] = {
        'var': mip_model.addVar(lb=float(lower_bound), ub=float(upper_bound), obj=0,
                                 vtype=grb.GRB.CONTINUOUS, name=attr_name_kurz),
        'lower_bound': Real(float(lower_bound)),
        'upper_bound': Real(float(upper_bound))
      }
    elif attr_obj.attr_type == 'numeric-int': # refer to loadData.VALID_ATTRIBUTE_TYPES
      model_vars['counterfactual'][attr_name_kurz] = {
        'var': mip_model.addVar(lb=lower_bound, ub=upper_bound, obj=0,
                                 vtype=grb.GRB.INTEGER, name=attr_name_kurz),
        'lower_bound': Real(float(lower_bound)),
        'upper_bound': Real(float(upper_bound))
      }
    elif attr_obj.attr_type == 'binary' or 'cat' in attr_obj.attr_type or 'ord' in attr_obj.attr_type:
      model_vars['counterfactual'][attr_name_kurz] = {
        'var': mip_model.addVar(lb=lower_bound, ub=upper_bound, obj=0,
                                 vtype=grb.GRB.BINARY, name=attr_name_kurz),
        'lower_bound': Real(float(lower_bound)),
        'upper_bound': Real(float(upper_bound))
      }
    else:
      raise Exception(f"Variable type {attr_obj.attr_type} not defined.")

  mip_model.update()
  print('\n\n==============================================\n\n', file = log_file)
  print('Model Variables:', file = log_file)
  pprint(model_vars, log_file)

  # factual_sample['y'] = False
  start_time = time.time()

  # find closest counterfactual sample from this negative sample
  all_counterfactuals, closest_counterfactual_sample, closest_interventional_sample = findClosestCounterfactualSample(
    mip_model,
    model_trained,
    model_vars,
    dataset_obj,
    factual_sample,
    norm_type,
    approach_string,
    epsilon,
    log_file
  )

  end_time = time.time()

  print('\n', file = log_file)
  print(f"Factual sample: \t\t {getPrettyStringForSampleDictionary(factual_sample, dataset_obj)}", file = log_file)

  print(f"Nearest counterfactual sample:\t {getPrettyStringForSampleDictionary(closest_counterfactual_sample['counterfactual_sample'], dataset_obj)} (verified)", file = log_file)
  print(f"Minimum counterfactual distance: {closest_counterfactual_sample['counterfactual_distance']:.6f}", file = log_file)

  return {
    'fac_sample': factual_sample,
    'cfe_found': True,
    'cfe_plausible': True,
    'cfe_time': end_time - start_time,
    'cfe_sample': closest_counterfactual_sample['counterfactual_sample'],
    'cfe_distance': closest_counterfactual_sample['counterfactual_distance'],
    # 'all_counterfactuals': all_counterfactuals
  }
