import sys
import time
import copy
import pickle
import numpy as np
import pandas as pd
import normalizedDistance
import torch

from modelConversion import *
from pysmt.shortcuts import *
from pysmt.typing import *
from pprint import pprint

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Functions that were not different from MACE_SAT will be directly used:
from generateSATExplanations import getPlausibilityFormula, getDiversityFormulaUpdate, getCounterfactualFormula

from network_linear_approximation import LinearizedNetwork
from mip_solver import MIPNetwork

from debug import ipsh

from random import seed
RANDOM_SEED = 1122334455
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

# DEBUG_FLAG = True
DEBUG_FLAG = False


def getModelFormula(model_symbols, model_trained, mlp_lbs=None, mlp_ubs=None):
  if isinstance(model_trained, DecisionTreeClassifier):
    model2formula = lambda a,b : tree2formula(a,b)
  elif isinstance(model_trained, LogisticRegression):
    model2formula = lambda a,b : lr2formula(a,b)
  elif isinstance(model_trained, RandomForestClassifier):
    model2formula = lambda a,b : forest2formula(a,b)
  elif isinstance(model_trained, MLPClassifier):
    # model2formula = lambda a,b : mlp2formula(a,b)
    return mlp2formula(model_trained, model_symbols, mlp_lbs, mlp_ubs)

  return model2formula(
    model_trained,
    model_symbols)


def getDistanceFormula(model_symbols, dataset_obj, factual_sample, norm_type, approach_string, norm_lower_threshold, norm_upper_threshold):

  if 'mace' in approach_string:
    variable_to_compute_distance_on = 'counterfactual'
  elif 'mint' in approach_string:
    variable_to_compute_distance_on = 'interventional'


  def getAbsoluteDifference(symbol_1, symbol_2):
    return Ite(
      GE(Minus(ToReal(symbol_1), ToReal(symbol_2)), Real(0)),
      Minus(ToReal(symbol_1), ToReal(symbol_2)),
      Minus(ToReal(symbol_2), ToReal(symbol_1))
    )

  # normalize this feature's distance by dividing the absolute difference by the
  # range of the variable (only applies for non-hot variables)
  normalized_absolute_distances = []
  normalized_squared_distances = []

  # IMPORTANT CHANGE IN CODE (Feb 04, 2020): prior to today, actionable/mutable
  # features overlapped. Now that we have introduced 3 types of variables
  # (actionable and mutable, non-actionable but mutable, immutable and non-actionable),
  # we must re-write the distance function to depent on all mutable features only,
  # while before we wrote distance as a function over actionable/mutable features.

  mutable_attributes = dataset_obj.getMutableAttributeNames('kurz')
  one_hot_attributes = dataset_obj.getOneHotAttributesNames('kurz')
  non_hot_attributes = dataset_obj.getNonHotAttributesNames('kurz')

  # 1. mutable & non-hot
  for attr_name_kurz in np.intersect1d(mutable_attributes, non_hot_attributes):
    normalized_absolute_distances.append(
      Div(
        ToReal(
          getAbsoluteDifference(
            model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol'],
            factual_sample[attr_name_kurz]
          )
        ),
        # Real(1)
        ToReal(
          model_symbols[variable_to_compute_distance_on][attr_name_kurz]['upper_bound'] -
          model_symbols[variable_to_compute_distance_on][attr_name_kurz]['lower_bound']
        )
      )
    )
    normalized_squared_distances.append(
      Pow(
        Div(
          ToReal(
            Minus(
              model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol'],
              factual_sample[attr_name_kurz]
            )
          ),
          # Real(1)
          ToReal(
            model_symbols[variable_to_compute_distance_on][attr_name_kurz]['upper_bound'] -
            model_symbols[variable_to_compute_distance_on][attr_name_kurz]['lower_bound']
          )
        ),
        Real(2)
      )
    )

  # 2. mutable & integer-based & one-hot
  already_considered = []
  for attr_name_kurz in np.intersect1d(mutable_attributes, one_hot_attributes):
    if attr_name_kurz not in already_considered:
      siblings_kurz = dataset_obj.getSiblingsFor(attr_name_kurz)
      if 'cat' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
        # this can also be implemented as the abs value of sum of a difference
        # in each attribute, divided by 2
        normalized_absolute_distances.append(
          Ite(
            And([
              EqualsOrIff(
                model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol'],
                factual_sample[attr_name_kurz]
              )
              for attr_name_kurz in siblings_kurz
            ]),
            Real(0),
            Real(1)
          )
        )
        # TODO: What about this? might be cheaper than Ite.
        # normalized_absolute_distances.append(
        #   Minus(
        #     Real(1),
        #     ToReal(And([
        #       EqualsOrIff(
        #         model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol'],
        #         factual_sample[attr_name_kurz]
        #       )
        #       for attr_name_kurz in siblings_kurz
        #     ]))
        #   )
        # )

        # As the distance is 0 or 1 in this case, the 2nd power is same as itself
        normalized_squared_distances.append(normalized_absolute_distances[-1])

      elif 'ord' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
        normalized_absolute_distances.append(
          Div(
            ToReal(
              getAbsoluteDifference(
                Plus([
                  model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol']
                  for attr_name_kurz in siblings_kurz
                ]),
                Plus([
                  factual_sample[attr_name_kurz]
                  for attr_name_kurz in siblings_kurz
                ]),
              )
            ),
            Real(len(siblings_kurz))
          )
        )

        normalized_squared_distances.append(
          Pow(
            Div(
              ToReal(
                Minus(
                  Plus([
                    model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol']
                    for attr_name_kurz in siblings_kurz
                  ]),
                  Plus([
                    factual_sample[attr_name_kurz]
                    for attr_name_kurz in siblings_kurz
                  ]),
                )
              ),
              Real(len(siblings_kurz))
            ),
            Real(2)
          )
        )
      else:
        raise Exception(f'{attr_name_kurz} must include either `cat` or `ord`.')
      already_considered.extend(siblings_kurz)

  # # 3. compute normalized squared distances
  # # pysmt.exceptions.SolverReturnedUnknownResultError
  # normalized_squared_distances = [
  #   # Times(distance, distance)
  #   Pow(distance, Int(2))
  #   for distance in normalized_absolute_distances
  # ]
  # # TODO: deprecate?
  # # def getSquaredifference(symbol_1, symbol_2):
  # #   return Times(
  # #     ToReal(Minus(ToReal(symbol_1), ToReal(symbol_2))),
  # #     ToReal(Minus(ToReal(symbol_2), ToReal(symbol_1)))
  # #   )


  # 4. sum up over everything allowed...
  # We use 1 / len(normalized_absolute_distances) below because we only consider
  # those attributes that are mutable, and for each sibling-group (ord, cat)
  # we only consider 1 entry in the normalized_absolute_distances
  if norm_type == 'zero_norm':
    normalized_distance = Times(
        Real(1 / len(normalized_absolute_distances)),
        Plus([
          Ite(
            Equals(elem, Real(0)),
            Real(0),
            Real(1)
          ) for elem in normalized_absolute_distances
        ])
      )
    distance_formula = And(
      GE(
        normalized_distance,
        Real(norm_lower_threshold)
      ),
      LE(
        normalized_distance,
        Real(norm_upper_threshold)
      )
    )
  elif norm_type == 'one_norm':
    normalized_distance = Times(
        Real(1 / len(normalized_absolute_distances)),
        ToReal(Plus(normalized_absolute_distances))
      )
    distance_formula = And(
      GE(
        normalized_distance,
        Real(norm_lower_threshold)
      ),
      LE(
        normalized_distance,
        Real(norm_upper_threshold)
      )
    )
  elif norm_type == 'two_norm':
    normalized_distance = Times(
        Real(1 / len(normalized_squared_distances)),
        ToReal(Plus(normalized_squared_distances))
      )
    distance_formula = And(
      GE(
        normalized_distance,
        Pow(
          Real(norm_lower_threshold),
          Int(2)
        )
      ),
      LE(
        normalized_distance,
        Pow(
          Real(norm_upper_threshold),
          Int(2)
        )
      )
    )
  elif norm_type == 'infty_norm':
    normalized_distance = Times(
        Real(1 / len(normalized_absolute_distances)),
        ToReal(Max(normalized_absolute_distances))
      )
    distance_formula = And(
      GE(
        normalized_distance,
        Real(norm_lower_threshold)
      ),
      LE(
        normalized_distance,
        Real(norm_upper_threshold)
      )
    )
  else:
    raise Exception(f'{norm_type} not recognized as a valid `norm_type`.')

  return distance_formula



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

# TODO: Gurobi can also make use of past computations
def getNetworkBounds(sklearn_model, dataset_obj, factual_sample, norm_type, norm_lower, norm_upper):
  assert isinstance(sklearn_model, MLPClassifier), "Only MLP model supports the linear relaxation."
  input_dim = len(dataset_obj.getInputAttributeNames('kurz'))

  # First, translate sklearn model to PyTorch model
  torch_model = getTorchFromSklearn(sklearn_model, input_dim)

  # Now create a linearized network
  layers = [module for module in torch_model.modules() if type(module) != torch.nn.Sequential]
  lin_net = LinearizedNetwork(layers)

  # Get input domains
  domains = np.zeros((input_dim, 2), dtype=np.float32)
  for i, attr_name_kurz in enumerate(dataset_obj.getInputAttributeNames('kurz')):
    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
    domains[i][0] = attr_obj.lower_bound
    domains[i][1] = attr_obj.upper_bound
  domains = torch.from_numpy(domains)

  # Get lower and upper bounds on all neuron values
  #TODO check factualsample.values() order
  feasible = lin_net.define_linear_approximation(domains, factual_sample, dataset_obj, norm_type, norm_lower, norm_upper)
  if not feasible:
    return False, None, None

  # cnt = 0
  # # print("lower bounds:--------------")
  # for i, layer_bounds in enumerate(lin_net.lower_bounds):
  #   # print(layer_bounds)
  #   if i %2 == 0 and i > 0:
  #     for bnd in layer_bounds:
  #       if bnd > 0:
  #         cnt += 1
  #
  # # print("upper bounds:--------------")
  # for i, layer_bounds in enumerate(lin_net.upper_bounds):
  #   # print(layer_bounds)
  #   if i%2 == 0 and i > 0:
  #     for bnd in layer_bounds:
  #       if bnd == 0:
  #         cnt += 1
  #
  # print("num of ReLUs with fixed state: ", cnt)

  return True, lin_net.lower_bounds, lin_net.upper_bounds



def findClosestCounterfactualSample(model_trained, model_symbols, dataset_obj, factual_sample, norm_type, approach_string, epsilon, log_file):

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

    assert sklearn_prediction == pysmt_prediction, f'Pysmt prediction does not match sklearn prediction. \n{dict_sample} \n{factual_sample}'
    assert sklearn_prediction != factual_prediction, 'Counterfactual and factual samples have the same prediction.'

  assert isinstance(model_trained, MLPClassifier), 'This approach is only for MLPs.'
  iters = 1
  max_iters = 100
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

  # Convert to pysmt_sample so factual symbols can be used in formulae
  factual_pysmt_sample = getPySMTSampleFromDictSample(factual_sample, dataset_obj)

  norm_lower_bound = 0
  norm_upper_bound = 1
  curr_norm_threshold = getCenterNormThresholdInRange(norm_lower_bound, norm_upper_bound)

  # Get and merge all constraints
  print('Constructing initial formulas: model, counterfactual, distance, plausibility, diversity\t\t', end = '', file = log_file)
  model_formula = getModelFormula(model_symbols, model_trained)
  counterfactual_formula = getCounterfactualFormula(model_symbols, factual_pysmt_sample)
  plausibility_formula = getPlausibilityFormula(model_symbols, dataset_obj, factual_pysmt_sample, approach_string)
  distance_formula = getDistanceFormula(model_symbols, dataset_obj, factual_pysmt_sample, norm_type, approach_string, norm_lower_bound, curr_norm_threshold)
  diversity_formula = TRUE() # simply initialize and modify later as new counterfactuals come in
  print('done.', file = log_file)

  print('Solving (not searching) for closest counterfactual using various distance thresholds...', file = log_file)

  solver_name = "z3"

  #############################################
  ###### Reverse BS (Exponential Growth) ######
  #############################################

  reverse_norm_threshold = epsilon
  solved, rev_bs_model = False, None
  iteration_start_time, iteration_end_time = 0, 0

  while (not solved):
    lower = reverse_norm_threshold/2.0 if reverse_norm_threshold != epsilon else 0.0

    feasible, mlp_lbs, mlp_ubs = getNetworkBounds(model_trained, dataset_obj, factual_sample, norm_type, lower, reverse_norm_threshold)
    if not feasible:
      reverse_norm_threshold *= 2.0
      continue
    model_formula = getModelFormula(model_symbols, model_trained, mlp_lbs, mlp_ubs)
    distance_formula = getDistanceFormula(model_symbols, dataset_obj, factual_pysmt_sample, norm_type,
                                          approach_string, lower, reverse_norm_threshold)
    # distance_formula = distance_formula.simplify()
    formula = And(  # works for both initial iteration and all subsequent iterations
      model_formula,
      counterfactual_formula,
      plausibility_formula,
      distance_formula,
      diversity_formula,
    )
    # formula = formula.simplify()

    with Solver(name=solver_name) as solver:
      solver.add_assertion(formula)
      try:
        iteration_start_time = time.time()
        solved = solver.solve()
        iteration_end_time = time.time()
      except SolverReturnedUnknownResultError: # Might happen in the two-norm case
        counterfactuals = [counterfactuals[0]]
        iters = max_iters # To prevent the normal BS to start afterwards
        break
      if solved:
        rev_bs_model = solver.get_model()
      else:
        if 'two_norm' not in norm_type: # There is no logic to prove with two-norm...
          # assert is_sat(And(plausibility_formula, distance_formula, diversity_formula), solver_name=solver_name)
          f = Implies(And(plausibility_formula, distance_formula, diversity_formula),
                      And(model_formula, Not(counterfactual_formula)))
          f = f.simplify()
          assert is_sat(f, solver_name=solver_name), 'no solution found (SMT issue).'

    if not solved:
      reverse_norm_threshold *= 2.0

  # The upper bound on distance
  norm_upper_bound = reverse_norm_threshold
  # The lower bound on distance
  norm_lower_bound = 0 if reverse_norm_threshold == epsilon else reverse_norm_threshold / 2.0

  curr_norm_threshold = (norm_lower_bound + norm_upper_bound) / 2.0
  distance_formula = getDistanceFormula(model_symbols, dataset_obj, factual_pysmt_sample, norm_type, approach_string,
                                        norm_lower_bound, curr_norm_threshold)
  first_iter = True

  #############################################
  ################ Normal BS ##################
  #############################################

  while iters < max_iters and norm_upper_bound - norm_lower_bound >= epsilon:

    print(f'\tIteration #{iters:03d}: testing norm threshold {curr_norm_threshold:.6f} in range [{norm_lower_bound:.6f}, {norm_upper_bound:.6f}]...\t', end = '', file = log_file)
    iters = iters + 1

    with Solver(name=solver_name) as solver:

      if not first_iter: # In the first iter, only the CFE from the exponential part will be saved.

        feasible, mlp_lbs, mlp_ubs = getNetworkBounds(model_trained, dataset_obj, factual_sample, norm_type,
                                            norm_lower_bound, curr_norm_threshold)
        if feasible:
          model_formula = getModelFormula(model_symbols, model_trained, mlp_lbs, mlp_ubs)
          # model_formula = getModelFormula(model_symbols, model_trained)

          formula = And(  # works for both initial iteration and all subsequent iterations
            model_formula,
            counterfactual_formula,
            plausibility_formula,
            distance_formula,
            diversity_formula,
          )
          # formula = formula.simplify()


          solver.add_assertion(formula)
          try:
            iteration_start_time = time.time()
            solved = solver.solve()
            iteration_end_time = time.time()
          except SolverReturnedUnknownResultError:  # Might happen in the two-norm case
            counterfactuals = [counterfactuals[0]]
            break
        else:
          solved = False

      else:
        assert solved, 'last iter of reverse BS must have solved the formula!'
        assert rev_bs_model is not None, 'last iter of reverse BS must have solved the formula!'

      if solved: # joint formula is satisfiable
        if first_iter is True:
          model = rev_bs_model
          first_iter = False
        else:
          model = solver.get_model()
        print('solution exists & found.', file = log_file)
        counterfactual_pysmt_sample = {}
        interventional_pysmt_sample = {}
        for (symbol_key, symbol_value) in model:
          # symbol_key may be 'x#', {'p0#', 'p1#'}, 'w#', or 'y'
          tmp = str(symbol_key)
          if 'counterfactual' in str(symbol_key):
            tmp = tmp[:-15]
            if tmp in dataset_obj.getInputOutputAttributeNames('kurz'):
              counterfactual_pysmt_sample[tmp] = symbol_value
          elif 'interventional' in str(symbol_key):
            tmp = tmp[:-15]
            if tmp in dataset_obj.getInputOutputAttributeNames('kurz'):
              interventional_pysmt_sample[tmp] = symbol_value
          elif tmp in dataset_obj.getInputOutputAttributeNames('kurz'): # for y variable
            counterfactual_pysmt_sample[tmp] = symbol_value
            interventional_pysmt_sample[tmp] = symbol_value

        # Convert back from pysmt_sample to dict_sample to compute distance and save
        counterfactual_sample  = getDictSampleFromPySMTSample(
          counterfactual_pysmt_sample,
          dataset_obj)
        interventional_sample  = getDictSampleFromPySMTSample(
          interventional_pysmt_sample,
          dataset_obj)

        # Assert samples have correct prediction label according to sklearn model
        assertPrediction(counterfactual_sample, model_trained, dataset_obj)
        # of course, there is no need to assertPrediction on the interventional_sample

        counterfactual_distance = normalizedDistance.getDistanceBetweenSamples(
          factual_sample,
          counterfactual_sample,
          norm_type,
          dataset_obj)
        interventional_distance = normalizedDistance.getDistanceBetweenSamples(
          factual_sample,
          interventional_sample,
          norm_type,
          dataset_obj)
        counterfactual_time = iteration_end_time - iteration_start_time
        counterfactuals.append({
          'counterfactual_sample': counterfactual_sample,
          'counterfactual_distance': counterfactual_distance,
          'interventional_sample': interventional_sample,
          'interventional_distance': interventional_distance,
          'time': counterfactual_time,
          'norm_type': norm_type})

        # Update diversity and distance formulas now that we have found a solution
        # TODO: I think the line below should be removed, because in successive
        #       reductions of delta, we should be able to re-use previous CFs
        # diversity_formula = And(diversity_formula, getDiversityFormulaUpdate(model))

        # IMPORTANT: something odd happens somtimes if use vanilla binary search;
        #            On the first iteration, with [0, 1] bounds, we may see a CF at
        #            d = 0.22. When we update the bounds to [0, 0.5] bounds,  we
        #            sometimes surprisingly see a new CF at distance 0.24. We optimize
        #            the binary search to solve this.
        norm_lower_bound = norm_lower_bound
        # norm_upper_bound = curr_norm_threshold
        if 'mace' in approach_string:
          norm_upper_bound = float(counterfactual_distance + epsilon / 100) # not float64
        elif 'mint' in approach_string:
          norm_upper_bound = float(interventional_distance + epsilon / 100) # not float64
        curr_norm_threshold = getCenterNormThresholdInRange(norm_lower_bound, norm_upper_bound)
        distance_formula = getDistanceFormula(model_symbols, dataset_obj, factual_pysmt_sample, norm_type, approach_string, norm_lower_bound, curr_norm_threshold)

      else: # no solution found in the assigned norm range --> update range and try again

        # assert is_sat(And(plausibility_formula, distance_formula, diversity_formula), solver_name=solver_name)
        if feasible:
          if 'two_norm' not in norm_type:  # There is no logic to prove with two-norm...
            f = Implies(And(plausibility_formula, distance_formula, diversity_formula),
                        And(model_formula, Not(counterfactual_formula)))
            assert is_sat(f, solver_name=solver_name), 'no solution found (SMT issue).'
        print('no solution exists.', file = log_file)
        norm_lower_bound = curr_norm_threshold
        norm_upper_bound = norm_upper_bound
        curr_norm_threshold = getCenterNormThresholdInRange(norm_lower_bound, norm_upper_bound)
        distance_formula = getDistanceFormula(model_symbols, dataset_obj, factual_pysmt_sample, norm_type, approach_string, norm_lower_bound, curr_norm_threshold)


  # IMPORTANT: there may be many more at this same distance! OR NONE! (what?? 2020.02.19)
  closest_counterfactual_sample = sorted(counterfactuals, key = lambda x: x['counterfactual_distance'])[0]
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


def getPySMTSampleFromDictSample(dict_sample, dataset_obj):
  pysmt_sample = {}
  for attr_name_kurz in dataset_obj.getInputOutputAttributeNames('kurz'):
    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
    if attr_name_kurz not in dataset_obj.getInputAttributeNames('kurz'):
      pysmt_sample[attr_name_kurz] = Bool(dict_sample[attr_name_kurz])
    elif attr_obj.attr_type == 'numeric-real':
      pysmt_sample[attr_name_kurz] = Real(float(dict_sample[attr_name_kurz]))
    else: # refer to loadData.VALID_ATTRIBUTE_TYPES
      pysmt_sample[attr_name_kurz] = Int(int(dict_sample[attr_name_kurz]))
  return pysmt_sample


def getDictSampleFromPySMTSample(pysmt_sample, dataset_obj):
  dict_sample = {}
  for attr_name_kurz in dataset_obj.getInputOutputAttributeNames('kurz'):
    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
    try:
      if attr_name_kurz not in dataset_obj.getInputAttributeNames('kurz'):
        dict_sample[attr_name_kurz] = bool(str(pysmt_sample[attr_name_kurz]) == 'True')
      elif attr_obj.attr_type == 'numeric-real':
        dict_sample[attr_name_kurz] = float(eval(str(pysmt_sample[attr_name_kurz])))
      else: # refer to loadData.VALID_ATTRIBUTE_TYPES
        dict_sample[attr_name_kurz] = int(str(pysmt_sample[attr_name_kurz]))
    except:
      raise Exception(f'Failed to read value from pysmt sample. Debug me manually.')
  return dict_sample


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

  start_time = time.time()

  # Initial params
  model_symbols = {
    'counterfactual': {},
    'interventional': {},
    'output': {'y': {'symbol': Symbol('y', BOOL)}}
  }

  # Populate model_symbols['counterfactual'/'interventional'] using the
  # parameters saved during training
  for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
    lower_bound = attr_obj.lower_bound
    upper_bound = attr_obj.upper_bound
    # print(f'\n attr_name_kurz: {attr_name_kurz} \t\t lower_bound: {lower_bound} \t upper_bound: {upper_bound}', file = log_file)
    if attr_name_kurz not in dataset_obj.getInputAttributeNames('kurz'):
      continue # do not overwrite the output
    if attr_obj.attr_type == 'numeric-real':
      model_symbols['counterfactual'][attr_name_kurz] = {
        'symbol': Symbol(attr_name_kurz + '_counterfactual', REAL),
        'lower_bound': Real(float(lower_bound)),
        'upper_bound': Real(float(upper_bound))
      }
      model_symbols['interventional'][attr_name_kurz] = {
        'symbol': Symbol(attr_name_kurz + '_interventional', REAL),
        'lower_bound': Real(float(lower_bound)),
        'upper_bound': Real(float(upper_bound))
      }
    else: # refer to loadData.VALID_ATTRIBUTE_TYPES
      model_symbols['counterfactual'][attr_name_kurz] = {
        'symbol': Symbol(attr_name_kurz + '_counterfactual', INT),
        'lower_bound': Int(int(lower_bound)),
        'upper_bound': Int(int(upper_bound))
      }
      model_symbols['interventional'][attr_name_kurz] = {
        'symbol': Symbol(attr_name_kurz + '_interventional', INT),
        'lower_bound': Int(int(lower_bound)),
        'upper_bound': Int(int(upper_bound))
      }
  print('\n\n==============================================\n\n', file = log_file)
  print('Model Symbols:', file = log_file)
  pprint(model_symbols, log_file)

  # factual_sample['y'] = False

  # find closest counterfactual sample from this negative sample
  all_counterfactuals, closest_counterfactual_sample, closest_interventional_sample = findClosestCounterfactualSample(
    model_trained,
    model_symbols,
    dataset_obj,
    factual_sample,
    norm_type,
    approach_string,
    epsilon,
    log_file
  )

  print('\n', file = log_file)
  print(f"Factual sample: \t\t {getPrettyStringForSampleDictionary(factual_sample, dataset_obj)}", file = log_file)
  if 'mace' in approach_string:
    print(f"Nearest counterfactual sample:\t {getPrettyStringForSampleDictionary(closest_counterfactual_sample['counterfactual_sample'], dataset_obj)} (verified)", file = log_file)
    print(f"Minimum counterfactual distance: {closest_counterfactual_sample['counterfactual_distance']:.6f}", file = log_file)
  elif 'mint' in approach_string:
    print(f"Nearest interventional sample:\t {getPrettyStringForSampleDictionary(closest_interventional_sample['interventional_sample'], dataset_obj)}", file = log_file)
    print(f"Nearest resulting CF sample:\t {getPrettyStringForSampleDictionary(closest_interventional_sample['counterfactual_sample'], dataset_obj)} (verified)", file = log_file)
    print(f"Minimum interventional distance: {closest_interventional_sample['interventional_distance']:.6f}", file = log_file)
    print(f"Minimum resulting CF distance:\t {closest_interventional_sample['counterfactual_distance']:.6f}", file = log_file)

  end_time = time.time()

  if 'mace' in approach_string:
    return {
      'fac_sample': factual_sample,
      'cfe_found': True,
      'cfe_plausible': True,
      'cfe_time': end_time - start_time,
      'cfe_sample': closest_counterfactual_sample['counterfactual_sample'],
      'cfe_distance': closest_counterfactual_sample['counterfactual_distance'],
      # 'all_counterfactuals': all_counterfactuals
    }
  elif 'mint' in approach_string:
    action_set = {}
    for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
      if factual_sample[attr_name_kurz] != closest_interventional_sample['interventional_sample'][attr_name_kurz]:
        action_set[attr_name_kurz] = closest_interventional_sample['interventional_sample'][attr_name_kurz]
    return {
      'fac_sample': factual_sample,
      'scf_found': True,
      'scf_plausible': True,
      'scf_time': end_time - start_time,
      'scf_sample': closest_interventional_sample['counterfactual_sample'],
      'scf_distance': closest_interventional_sample['counterfactual_distance'],
      # 'int_sample': closest_interventional_sample['interventional_sample'],
      # 'int_distance': closest_interventional_sample['interventional_distance'],
      # 'action_set': action_set,
      'int_set': action_set,
      'int_cost': closest_interventional_sample['interventional_distance'],
      # 'all_counterfactuals': all_counterfactuals
    }
