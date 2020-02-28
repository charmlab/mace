import os
import time
import numpy as np
import pandas as pd
import normalizedDistance

from pprint import pprint
from recourse.builder import RecourseBuilder
from recourse.builder import ActionSet

def genExp(model_trained, factual_sample, norm_type, dataset_obj):

  start_time = time.time()

  # SIMPLE HACK!!
  # ActionSet() construction demands that all variables have a range to them. In
  # the case of one-hot ordinal variables (e.g., x2_ord_0, x2_ord_1, x2_ord_2)),
  # the first sub-category (i.e., x2_ord_0) will always have range(1,1), failing
  # the requirements of ActionSet(). Therefore, we add a DUMMY ROW to the data-
  # frame, which is a copy of another row (so not to throw off the range of other
  # attributes), but setting a 0 value to all _ord_ variables. (might as well do
  # this for all _cat_ variables as well).
  tmp_df = dataset_obj.data_frame_kurz
  sample_row = tmp_df.loc[0].to_dict()
  for attr_name_kurz in dataset_obj.getOneHotAttributesNames('kurz'):
    sample_row[attr_name_kurz] = 0
  tmp_df = tmp_df.append(pd.Series(sample_row), ignore_index=True)

  df = tmp_df
  X = df.loc[:, df.columns != 'y']

  # Enforce binary, categorical (including ordinal) variables only take on 2 values
  custom_bounds = {attr_name_kurz: (0, 100, 'p') for attr_name_kurz in np.union1d(
    dataset_obj.getOneHotAttributesNames('kurz'),
    dataset_obj.getBinaryAttributeNames('kurz')
  )}
  action_set = ActionSet(X = X, custom_bounds = custom_bounds)
  # action_set['x1'].mutable = False # x1 = 'Race'
  # In the current implementation, we only supports any/none actionability
  for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
    if attr_obj.actionability == 'none':
      action_set[attr_name_kurz].mutable = False
    elif attr_obj.actionability == 'any':
      continue # do nothing
    else:
      raise ValueError(f'Actionable Recourse does not support actionability type {attr_obj.actionability}')

  # Enforce search over integer-based grid for integer-based variables
  for attr_name_kurz in np.union1d(
    dataset_obj.getIntegerBasedAttributeNames('kurz'),
    dataset_obj.getBinaryAttributeNames('kurz'),
  ):
    action_set[attr_name_kurz].step_type = "absolute"
    action_set[attr_name_kurz].step_size = 1

  coefficients = model_trained.coef_[0]
  intercept = model_trained.intercept_[0]

  if norm_type == 'one_norm':
    mip_cost_type = 'total'
  elif norm_type == 'infty_norm':
    mip_cost_type = 'max'
  else:
    raise ValueError(f'Actionable Recourse does not support norm_type {norm_type}')

  factual_sample_values = list(factual_sample.values())
  # p = .8
  rb = RecourseBuilder(
        optimizer="cplex",
        coefficients=coefficients,
        intercept=intercept, # - (np.log(p / (1. - p))),
        action_set=action_set,
        x=factual_sample_values,
        mip_cost_type=mip_cost_type
  )

  output = rb.fit()
  counterfactual_sample_values = np.add(factual_sample_values, output['actions'])
  counterfactual_sample = dict(zip(factual_sample.keys(), counterfactual_sample_values))

  # factual_sample['y'] = False
  # counterfactual_sample['y'] = True
  counterfactual_sample['y'] = not factual_sample['y']
  counterfactual_plausible = True

  # IMPORTANT: no need to check for integer-based / binary-based plausibility,
  # because those were set above when we said step_type = absolute! Just round!
  for attr_name_kurz in np.union1d(
    dataset_obj.getOneHotAttributesNames('kurz'),
    dataset_obj.getBinaryAttributeNames('kurz')
  ):
    try:
      assert np.isclose(
        counterfactual_sample[attr_name_kurz],
        np.round(counterfactual_sample[attr_name_kurz])
      )
      counterfactual_sample[attr_name_kurz] = np.round(counterfactual_sample[attr_name_kurz])
    except:
      distance = -1
      counterfactual_plausible = False
      # return counterfactual_sample, distance

  # Perform plausibility-data-type check! Remember, all ordinal variables
  # have already been converted to categorical variables. It is important now
  # to check that 1 (and only 1) sub-category is activated in the resulting
  # counterfactual sample.
  already_considered = []
  for attr_name_kurz in dataset_obj.getOneHotAttributesNames('kurz'):
    if attr_name_kurz not in already_considered:
      siblings_kurz = dataset_obj.getSiblingsFor(attr_name_kurz)
      activations_for_category = [
        counterfactual_sample[attr_name_kurz] for attr_name_kurz in siblings_kurz
      ]
      sum_activations_for_category = np.sum(activations_for_category)
      if 'cat' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
        if sum_activations_for_category == 1:
          continue
        else:
          # print('not plausible, fixing..', end='')
          # TODO: don't actually return early! Instead see the actual distance,
          # fingers crossed that we can say that not only is their method giving
          # counterfactuals are larger distances, but in a lot of cases they are
          # not data-type plausible
          # INSTEAD, do below:
          # Convert to correct categorical/ordinal activations so we can
          # compute the distance using already written function.
          #     Turns out we need to do nothing, because the distance between
          #     [0,1,0] and anything other than itself, (e.g., [1,1,0] or [1,0,1])
          #     is always 1 :)
          # continue
          distance = -1
          counterfactual_plausible = False
          # return counterfactual_sample, distance
      elif 'ord' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
        # TODO: assert activations are in order...
        # if not, repeat as above...
        for idx in range(int(sum_activations_for_category)):
          if activations_for_category[idx] != 1:
            # Convert to correct categorical/ordinal activations so we can
            # compute the distance using already written function.
            # Find the max index of 1 in the array, and set everything before that to 1
            # print('not plausible, fixing..', end='')
            # max_index_of_1 = np.where(np.array(activations_for_category) == 1)[0][-1]
            # for idx2 in range(max_index_of_1 + 1):
            #   counterfactual_sample[siblings_kurz[idx2]] = 1
            # break
            distance = -1
            counterfactual_plausible = False
            # return counterfactual_sample, distance
      else:
        raise Exception(f'{attr_name_kurz} must include either `cat` or `ord`.')
      already_considered.extend(siblings_kurz)

  # TODO: convert to correct categorical/ordinal activations so we can compute

  # distance = output['cost'] # TODO: this must change / be verified!???
  distance = normalizedDistance.getDistanceBetweenSamples(
    factual_sample,
    counterfactual_sample,
    norm_type,
    dataset_obj
  )


  # # TODO: post-feasibible needed???? NO
  # # make plausible by rounding all non-numeric-real attributes to
  # # nearest value in range
  # for idx, elem in enumerate(es_instance):
  #     attr_name_kurz = dataset_obj.getInputAttributeNames('kurz')[idx]
  #     attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
  #     if attr_obj.attr_type != 'numeric-real':
  #         # round() might give a value that is NOT in plausible.
  #         # instead find the nearest plausible value
  #         es_instance[idx] = min(
  #             list(range(int(attr_obj.lower_bound), int(attr_obj.upper_bound) + 1)),
  #             key = lambda x : abs(x - es_instance[idx])
  #         )

  end_time = time.time()

  return {
    'factual_sample': factual_sample,
    'cfe_sample': counterfactual_sample,
    'cfe_found': True, # TODO?
    'cfe_plausible': counterfactual_plausible,
    'cfe_distance': distance,
    'cfe_time': end_time - start_time,
  }

