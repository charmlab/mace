import copy
import pickle
import numpy as np
import pandas as pd
from pprint import pprint

from random import seed
RANDOM_SEED = 1122334455
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

def getDistanceBetweenSamples(sample_1, sample_2, norm_type, dataset_obj):

  # IMPORTANT: sample1 and sample2 are dict_samples, not pysmt_samples
  # (of course, this is by design, otherwise it would break abstraction)
  assert(sample_1.keys() == sample_2.keys())
  assert(sorted(sample_1.keys()) == sorted(dataset_obj.getInputOutputAttributeNames('kurz')))

  # normalize this feature's distance by dividing the absolute difference by the
  # range of the variable (only applies for non-hot variables)
  normalized_absolute_distances = []

  mutable_attributes = dataset_obj.getMutableAttributeNames('kurz')
  one_hot_attributes = dataset_obj.getOneHotAttributesNames('kurz')
  non_hot_attributes = dataset_obj.getNonHotAttributesNames('kurz')

  # 1. mutable & non-hot
  for attr_name_kurz in np.intersect1d(mutable_attributes, non_hot_attributes):
    normalized_absolute_distances.append(
      # note: float() works for both integer-based and real-based attributes, no
      # need to separate them out
      abs(
        sample_1[attr_name_kurz] -
        sample_2[attr_name_kurz]
      ) /
      (
        dataset_obj.attributes_kurz[attr_name_kurz].upper_bound -
        dataset_obj.attributes_kurz[attr_name_kurz].lower_bound
      )
    )

  # 2. mutable & one-hot
  already_considered = []
  for attr_name_kurz in np.intersect1d(mutable_attributes, one_hot_attributes):
    if attr_name_kurz not in already_considered:
      siblings_kurz = dataset_obj.getSiblingsFor(attr_name_kurz)
      if 'cat' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
        sub_sample_1 = [sample_1[x] for x in siblings_kurz]
        sub_sample_2 = [sample_2[x] for x in siblings_kurz]
        normalized_absolute_distances.append(
          1 - int(np.array_equal(sub_sample_1, sub_sample_2))
        )
      elif 'ord' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
        sub_sample_1 = [sample_1[x] for x in siblings_kurz]
        sub_sample_2 = [sample_2[x] for x in siblings_kurz]
        normalized_absolute_distances.append(
          np.sum(np.abs(np.subtract(sub_sample_1, sub_sample_2))) / len(sub_sample_1)
        )
      else:
        raise Exception(f'{attr_name_kurz} must include either `cat` or `ord`.')
      already_considered.extend(siblings_kurz)

  # 3. compute normalized squared distances
  normalized_squared_distances = [
    distance ** 2
    for distance in normalized_absolute_distances
  ]

  # 4. sum up over everything allowed...
  zero_norm_distance = 1 / len(normalized_absolute_distances) * np.count_nonzero(normalized_absolute_distances)
  one_norm_distance = 1 / len(normalized_absolute_distances) * sum(normalized_absolute_distances)
  two_norm_distance = np.sqrt(1 / len(normalized_squared_distances) * sum(normalized_squared_distances)) # note the sqrt(1/len) guarantees dist \in [0,1]
  infty_norm_distance = 1 / len(normalized_absolute_distances) * max(normalized_absolute_distances)

  if norm_type == 'zero_norm':
    return zero_norm_distance
  elif norm_type == 'one_norm':
    return one_norm_distance
  elif norm_type == 'two_norm':
    return two_norm_distance
  elif norm_type == 'infty_norm':
    return infty_norm_distance
  else:
    raise Exception(f'{norm_type} not recognized as a valid `norm_type`.')
  # TODO: implement combinations of distances
