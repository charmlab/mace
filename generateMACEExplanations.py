import time
import copy
import pickle
import numpy as np
import pandas as pd
import normalizedDistance

from modelConversion import *
from pysmt.shortcuts import *
from pysmt.typing import *
from pprint import pprint

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from random import seed
RANDOM_SEED = 1122334455
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)


def getModelFormula(model_symbols, model_trained):
  if isinstance(model_trained, DecisionTreeClassifier):
    model2formula = lambda a,b : tree2formula(a,b)
  elif isinstance(model_trained, LogisticRegression):
    model2formula = lambda a,b : lr2formula(a,b)
  elif isinstance(model_trained, RandomForestClassifier):
    model2formula = lambda a,b : forest2formula(a,b)
  elif isinstance(model_trained, MLPClassifier):
    model2formula = lambda a,b : mlp2formula(a,b)

  return model2formula(
    model_trained,
    model_symbols)


def getCounterfactualFormula(model_symbols, factual_sample):
  return EqualsOrIff(
    model_symbols['output']['y']['symbol'],
    Not(factual_sample['y'])
  ) # meaning we want the decision to be flipped.


def getDistanceFormula(model_symbols, dataset_obj, factual_sample, norm_type, norm_threshold):

  def getAbsoluteDifference(symbol_1, symbol_2):
    return Ite(
      GE(Minus(ToReal(symbol_1), ToReal(symbol_2)), Real(0)),
      Minus(ToReal(symbol_1), ToReal(symbol_2)),
      Minus(ToReal(symbol_2), ToReal(symbol_1))
    )

  # TODO: deprecate?
  # def getSquaredifference(symbol_1, symbol_2):
  #   return Times(
  #     ToReal(Minus(ToReal(symbol_1), ToReal(symbol_2))),
  #     ToReal(Minus(ToReal(symbol_2), ToReal(symbol_1)))
  #   )

  # normalize this feature's distance by dividing the absolute difference by the
  # range of the variable (only applies for non-hot variables)
  normalized_absolute_distances = []
  normalized_squared_distances = []

  actionable_attributes = dataset_obj.getActionableAttributeNames('kurz')
  one_hot_attributes = dataset_obj.getOneHotAttributesNames('kurz')
  non_hot_attributes = dataset_obj.getNonHotAttributesNames('kurz')

  # 1. actionable & non-hot
  for attr_name_kurz in np.intersect1d(actionable_attributes, non_hot_attributes):
    normalized_absolute_distances.append(
      Div(
        ToReal(
          getAbsoluteDifference(
            model_symbols['input'][attr_name_kurz]['symbol'],
            factual_sample[attr_name_kurz]
          )
        ),
        Real(1)
        # ToReal(
        #   model_symbols['input'][attr_name_kurz]['upper_bound'] -
        #   model_symbols['input'][attr_name_kurz]['lower_bound']
        # )
      )
    )

  # 2. actionable & integer-based & one-hot
  already_considered = []
  for attr_name_kurz in np.intersect1d(actionable_attributes, one_hot_attributes):
    if attr_name_kurz not in already_considered:
      siblings_kurz = dataset_obj.getSiblingsFor(attr_name_kurz)
      if 'cat' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
        # this can also be implemented as the abs value of sum of a difference
        # in each attribute, divided by 2
        normalized_absolute_distances.append(
          Ite(
            And([
              EqualsOrIff(
                model_symbols['input'][attr_name_kurz]['symbol'],
                factual_sample[attr_name_kurz]
              )
              for attr_name_kurz in siblings_kurz
            ]),
            Real(0),
            Real(1)
          )
        )
      elif 'ord' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
        normalized_absolute_distances.append(
          Div(
            ToReal(
              getAbsoluteDifference(
                Plus([
                  model_symbols['input'][attr_name_kurz]['symbol']
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
        # this can also be implemented as below:
        # normalized_absolute_distances.append(
        #   Div(
        #     ToReal(
        #       Plus([
        #         Ite(
        #           EqualsOrIff(
        #             model_symbols['input'][attr_name_kurz]['symbol'],
        #             factual_sample[attr_name_kurz]
        #           ),
        #           Real(0),
        #           Real(1)
        #         )
        #         for attr_name_kurz in siblings_kurz
        #       ])
        #     ),
        #     Real(len(siblings_kurz))
        #   )
        # )
      else:
        raise Exception(f'{attr_name_kurz} must include either `cat` or `ord`.')
      already_considered.extend(siblings_kurz)

  # 3. compute normalized squared distances
  # pysmt.exceptions.SolverReturnedUnknownResultError
  normalized_squared_distances = [
    # Times(distance, distance)
    Pow(distance, Real(2))
    for distance in normalized_absolute_distances
  ]

  # 4. sum up over everything allowed...
  # We use 1 / len(normalized_absolute_distances) below because we only consider
  # those attributes that are actionable, and for each sibling-group (ord, cat)
  # we only consider 1 entry in the normalized_absolute_distances
  if norm_type == 'zero_norm':
    distance_formula = LE(
      Times(
        Real(1 / len(normalized_absolute_distances)),
        Plus([
          Ite(
            Equals(elem, Real(0)),
            Real(0),
            Real(1)
          ) for elem in normalized_absolute_distances
        ])
      ),
      Real(norm_threshold)
    )
  elif norm_type == 'one_norm':
    distance_formula = LE(
      Times(
        Real(1 / len(normalized_absolute_distances)),
        ToReal(Plus(normalized_absolute_distances))
      ),
      Real(norm_threshold)
    )
  elif norm_type == 'two_norm':
    distance_formula = LE(
      Pow(
        Times(
          Real(1 / len(normalized_squared_distances)),
          ToReal(Plus(normalized_squared_distances))
        ),
      Real(0.5)
      ),
      Real(norm_threshold)
    )
  elif norm_type == 'infty_norm':
    distance_formula = LE(
      Times(
        Real(1 / len(normalized_absolute_distances)),
        ToReal(Max(normalized_absolute_distances))
      ),
      Real(norm_threshold)
    )
  else:
    raise Exception(f'{norm_type} not recognized as a valid `norm_type`.')

  return distance_formula


def getPlausibilityFormula(model_symbols, dataset_obj, factual_sample):
  # here is where the user specifies the range of variables, and which
  # variables are actionable or not, etc.
  range_plausibility = And([
    And(
      GE(model_symbols['input'][attr_name_kurz]['symbol'], model_symbols['input'][attr_name_kurz]['lower_bound']),
      LE(model_symbols['input'][attr_name_kurz]['symbol'], model_symbols['input'][attr_name_kurz]['upper_bound'])
    )
    for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz')
  ])

  onehot_categorical_plausibility = TRUE() # plausibility of categorical (sum = 1)
  onehot_ordinal_plausibility = TRUE() # plausibility ordinal (x3 >= x2 & x2 >= x1)

  if dataset_obj.is_one_hot:

    dict_of_siblings_kurz = dataset_obj.getDictOfSiblings('kurz')

    for parent_name_kurz in dict_of_siblings_kurz['cat'].keys():

      onehot_categorical_plausibility = And(
        onehot_categorical_plausibility,
        And(
          EqualsOrIff(
            Plus([
              model_symbols['input'][attr_name_kurz]['symbol']
              for attr_name_kurz in dict_of_siblings_kurz['cat'][parent_name_kurz]
            ]),
            Int(1)
          )
        )
      )

    for parent_name_kurz in dict_of_siblings_kurz['ord'].keys():

      onehot_ordinal_plausibility = And(
        onehot_ordinal_plausibility,
        And([
          GE(
            ToReal(model_symbols['input'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx]]['symbol']),
            ToReal(model_symbols['input'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx + 1]]['symbol'])
          )
          for symbol_idx in range(len(dict_of_siblings_kurz['ord'][parent_name_kurz]) - 1) # already sorted
        ])
      )

      # # Also implemented as the following logic, stating that
      # # if x_j == 1, all x_i == 1 for i < j
      # # Friendly reminder that for ordinal variables, x_0 is always 1
      # onehot_ordinal_plausibility = And([
      #   Ite(
      #     EqualsOrIff(
      #       ToReal(model_symbols['input'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx_ahead]]['symbol']),
      #       Real(1)
      #     ),
      #     And([
      #       EqualsOrIff(
      #         ToReal(model_symbols['input'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx_behind]]['symbol']),
      #         Real(1)
      #       )
      #       for symbol_idx_behind in range(symbol_idx_ahead)
      #     ]),
      #     TRUE()
      #   )
      #   for symbol_idx_ahead in range(1, len(dict_of_siblings_kurz['ord'][parent_name_kurz])) # already sorted
      # ])

  # plausibility of non-actionable values
  actionability_plausibility = []
  for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
    if attr_obj.actionability == 'none':
      actionability_plausibility.append(EqualsOrIff(
        model_symbols['input'][attr_name_kurz]['symbol'],
        factual_sample[attr_name_kurz]
      ))
    elif attr_obj.actionability == 'same-or-increase':
      actionability_plausibility.append(GE(
        model_symbols['input'][attr_name_kurz]['symbol'],
        factual_sample[attr_name_kurz]
      ))
    elif attr_obj.actionability == 'same-or-decrease':
      actionability_plausibility.append(LE(
        model_symbols['input'][attr_name_kurz]['symbol'],
        factual_sample[attr_name_kurz]
      ))
  actionability_plausibility = And(actionability_plausibility)


  return And(
    # range_plausibility,
    onehot_categorical_plausibility,
    onehot_ordinal_plausibility,
    actionability_plausibility
  )


def getDiversityFormulaUpdate(model):
  return Not(
    And([
      EqualsOrIff(
        symbol_key,
        symbol_value
      )
      for (symbol_key, symbol_value) in model
    ])
  )


def findClosestCounterfactualSample(model_trained, model_symbols, dataset_obj, factual_sample, norm_type, epsilon, log_file):

  def getCenterNormThresholdInRange(lower_bound, upper_bound):
    return (lower_bound + upper_bound) / 2

  def assertPrediction(dict_sample, model_trained, dataset_obj):
    vectorized_sample = []
    for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
      vectorized_sample.append(dict_sample[attr_name_kurz])

    prediction = int(str(dict_sample['y']) == 'True')
    assert int(model_trained.predict([vectorized_sample])[0]) == prediction, 'Counterfactual prediction does not match sklearn prediction.'

  # IMPORTANT TODO: Convert to pysmt_sample
  factual_pysmt_sample = getPySMTSampleFromDictSample(factual_sample, dataset_obj)

  norm_lower_bound = 0
  norm_upper_bound = 1
  curr_norm_threshold = getCenterNormThresholdInRange(norm_lower_bound, norm_upper_bound)

  # Get and merge all constraints
  print('Constructing initial formulas: model, counterfactual, distance, plausibility, diversity\t\t', end = '', file=log_file)
  model_formula = getModelFormula(model_symbols, model_trained)
  counterfactual_formula = getCounterfactualFormula(model_symbols, factual_pysmt_sample)
  plausibility_formula = getPlausibilityFormula(model_symbols, dataset_obj, factual_pysmt_sample)
  distance_formula = getDistanceFormula(model_symbols, dataset_obj, factual_pysmt_sample, norm_type, curr_norm_threshold)
  diversity_formula = TRUE() # simply initialize and modify later as new counterfactuals come in
  print('done.', file=log_file)

  iters = 1
  max_iters = 100
  counterfactuals = [] # list of tuples (samples, distances)
  counterfactuals.append({'sample': {}, 'distance': np.infty, 'time': np.infty, 'norm_type': None}) # in case no counterfactuals are found.
  explanation_counter = 0

  print('Solving (not searching) for closest counterfactual using various distance thresholds...', file=log_file)

  while iters < max_iters and norm_upper_bound - norm_lower_bound >= epsilon:

    print(f'\tIteration #{iters:03d}: testing norm threshold {curr_norm_threshold:.6f} in range [{norm_lower_bound:.6f}, {norm_upper_bound:.6f}]...\t', end = '', file=log_file)
    iters = iters + 1

    formula = And( # works for both initial iteration and all subsequent iterations
      model_formula,
      counterfactual_formula,
      plausibility_formula,
      distance_formula,
      diversity_formula,
    )
    iteration_start_time = time.time()
    model = get_model(formula)
    iteration_end_time = time.time()

    if model: # joint formula is satisfiable

      print('solution exists & found.', file=log_file)
      explanation_counter = explanation_counter + 1
      counterfactual_pysmt_sample = {}
      for (symbol_key, symbol_value) in model:
        if str(symbol_key) in dataset_obj.getInputOutputAttributeNames('kurz'): # symbol_key may be 'x#', {'p0#', 'p1#'}, 'w#', or 'y'
          counterfactual_pysmt_sample[str(symbol_key)] = symbol_value
      # IMPORTANT TODO: Convert back to dict_sample
      counterfactual_sample  = getDictSampleFromPySMTSample(counterfactual_pysmt_sample, dataset_obj)

      # assert samples have correct prediction label according to sklearn model
      assertPrediction(counterfactual_sample, model_trained, dataset_obj)

      counterfactual_distance = normalizedDistance.getDistanceBetweenSamples(factual_sample, counterfactual_sample, norm_type, dataset_obj)
      counterfactual_time = iteration_end_time - iteration_start_time
      counterfactuals.append({
        'sample': counterfactual_sample,
        'distance': counterfactual_distance,
        'time': counterfactual_time,
        'norm_type': norm_type})

      # Update diversity and distance formulas now that we have found a solution
      # TODO: I think the line below should be removed, because in successive
      #       reductions of delta, we should be able to re-use previous CFs
      # diversity_formula = And(diversity_formula, getDiversityFormulaUpdate(model))

      # ipsh()
      norm_lower_bound = norm_lower_bound
      # IMPORTANT something odd happens somtimes if use vanilla binary search;
      # On the first iteration, with [0, 1] bounds, we may see a CF at d = 0.22.
      # When we update the bounds to [0, 0.5] bounds, surprisingly we sometimes
      # see a new CF at distance 0.24. We optimize the binary search to solve this.
      # norm_upper_bound = curr_norm_threshold
      norm_upper_bound = float(counterfactual_distance + epsilon / 100) # not float64
      curr_norm_threshold = getCenterNormThresholdInRange(norm_lower_bound, norm_upper_bound)
      distance_formula = getDistanceFormula(model_symbols, dataset_obj, factual_pysmt_sample, norm_type, curr_norm_threshold)

    else: # no solution found in the assigned norm range --> update range and try again

      neg_formula = Not(formula)
      neg_model = get_model(neg_formula)
      if neg_model:
        print('no solution exists.', file=log_file)
        # ipsh()
        norm_lower_bound = curr_norm_threshold
        norm_upper_bound = norm_upper_bound
        curr_norm_threshold = getCenterNormThresholdInRange(norm_lower_bound, norm_upper_bound)
        distance_formula = getDistanceFormula(model_symbols, dataset_obj, factual_pysmt_sample, norm_type, curr_norm_threshold)
      else:
        print('no solution found (SMT issue).', file=log_file)
        quit()
        break

  sorted_counterfactuals = sorted(counterfactuals, key = lambda x: x['distance'])
  closest_counterfactual_sample = sorted_counterfactuals[0] # IMPORTANT: there may be many more at this same distance! OR NONE!

  return counterfactuals, closest_counterfactual_sample


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
    if not attr_obj.is_input:
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
    if not attr_obj.is_input:
      dict_sample[attr_name_kurz] = bool(str(pysmt_sample[attr_name_kurz]))
    elif attr_obj.attr_type == 'numeric-real':
      dict_sample[attr_name_kurz] = float(eval(str(pysmt_sample[attr_name_kurz])))
    else: # refer to loadData.VALID_ATTRIBUTE_TYPES
      dict_sample[attr_name_kurz] = int(str(pysmt_sample[attr_name_kurz]))
  return dict_sample


def genExp(
  explanation_file_name,
  model_trained,
  dataset_obj,
  factual_sample,
  norm_type,
  epsilon):

  start_time = time.time()

  log_file = open(explanation_file_name,'w')

  # Initial params
  model_symbols = {
    'input': {},
    'output': {'y': {'symbol': Symbol('y', BOOL)}}
  }

  # Populate model_symbols['input'] using the parameters saved during training
  for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
    lower_bound = attr_obj.lower_bound
    upper_bound = attr_obj.upper_bound
    if not attr_obj.is_input:
      continue # do not overwrite the output
    if attr_obj.attr_type == 'numeric-real':
      model_symbols['input'][attr_name_kurz] = {
        'symbol': Symbol(attr_name_kurz, REAL),
        'lower_bound': Real(float(lower_bound)),
        'upper_bound': Real(float(upper_bound))
      }
    else: # refer to loadData.VALID_ATTRIBUTE_TYPES
      model_symbols['input'][attr_name_kurz] = {
        'symbol': Symbol(attr_name_kurz, INT),
        'lower_bound': Int(int(lower_bound)),
        'upper_bound': Int(int(upper_bound))
      }
  print('\n\n====================================================================================================================\n\n', file=log_file)
  print('Model Symbols:', file=log_file)
  pprint(model_symbols, log_file)

  factual_sample['y'] = False

  # find closest counterfactual sample from this negative sample
  all_counterfactuals, closest_counterfactual_sample = findClosestCounterfactualSample(
    model_trained,
    model_symbols,
    dataset_obj,
    factual_sample,
    norm_type,
    epsilon,
    log_file
  )

  print('\n', file=log_file)
  print(f"Factual sample: \t\t {getPrettyStringForSampleDictionary(factual_sample, dataset_obj)}", file=log_file)
  print(f"Best counterfactual sample: \t {getPrettyStringForSampleDictionary(closest_counterfactual_sample['sample'], dataset_obj)} (verified)", file=log_file)
  print('', file=log_file)
  print(f"Minimum counterfactual distance (by solving the formula):\t {closest_counterfactual_sample['distance']:.6f}", file=log_file)

  end_time = time.time()

  return {
    'factual_sample': factual_sample,
    'counterfactual_sample': closest_counterfactual_sample['sample'],
    'counterfactual_found': True,
    'counterfactual_plausible': True,
    'counterfactual_distance': closest_counterfactual_sample['distance'],
    'counterfactual_time': end_time - start_time,
    'all_counterfactuals': all_counterfactuals
  }




# TODO:
# 1. [outer loop] binary search to find the minimum distance
# 2. [counterfactual constraint] write code / use open-source code to automatically convert a trained program --> SSA form --> WPC form
# 3. [plausibility constraint] extend to allow for immutable and conditionally immutable variables
# 4. [distance constraint] extend to combinations of distances (0-norm, 1-norm, inf-norm)
# 5. [diversity constraint] extend to allow for more complex variants, as described in the overleaf document

# We expect something like the following:
# model_symbols = {
#   'input': {
#     'x0': {'symbol': Symbol('x0', INT), 'lower_bound': Int(0), 'upper_bound': Int(1)},
#     'x1': {'symbol': Symbol('x1', INT), 'lower_bound': Int(0), 'upper_bound': Int(1)},
#     'x2': {'symbol': Symbol('x2', INT), 'lower_bound': Int(1), 'upper_bound': Int(4)},
#     'x3': {'symbol': Symbol('x3', INT), 'lower_bound': Int(0), 'upper_bound': Int(3)},
#   },
#   'output': {
#     'y': {'symbol': Symbol('y', BOOL)},
#   }
# }
# factual_sample = {
#   'x0': Int(int(neg_row[0])),
#   'x1': Int(int(neg_row[1])),
#   'x2': Int(int(neg_row[2])),
#   'x3': Int(int(neg_row[3])),
#   'y': FALSE()
# }
# observable_sample = {
#   'x0': Int(int(pos_row[0])),
#   'x1': Int(int(pos_row[1])),
#   'x2': Int(int(pos_row[2])),
#   'x3': Int(int(pos_row[3])),
#   'y': TRUE()
# }



















