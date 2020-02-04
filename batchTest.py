import os
import copy
import pickle
import argparse
import numpy as np
import pandas as pd

from pprint import pprint
from datetime import datetime
from sklearn.model_selection import train_test_split

import loadData
import modelTraining

from debug import ipsh

try:
  import generateMACEExplanations
except:
  print('[ENV WARNING] activate virtualenv to allow for testing MACE')
import generateMOExplanations
import generateFTExplanations
try:
  import generateARExplanations
except:
  print('[ENV WARNING] deactivate virtualenv to allow for testing Actionable Recourse')


from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)


def getBalancedDataFrame(dataset_obj):
  balanced_data_frame = copy.deepcopy(dataset_obj.data_frame_kurz)

  # get input and output columns
  all_data_frame_cols = balanced_data_frame.columns.values

  input_cols = [x for x in all_data_frame_cols if 'y' not in x.lower()]
  output_col = [x for x in all_data_frame_cols if 'y' in x.lower()][0]

  # assert only two classes in label (maybe relax later??)
  assert np.array_equal(
    np.unique(balanced_data_frame[output_col]),
    np.array([0, 1]) # only allowing {0, 1} labels
  )

  # get balanced dataframe (take minimum of the count, then round down to nearest 250)
  unique_values_and_count = balanced_data_frame[output_col].value_counts()
  number_of_subsamples_in_each_class = unique_values_and_count.min() // 250 * 250
  balanced_data_frame = pd.concat([
      balanced_data_frame[balanced_data_frame.loc[:,output_col] == 0].sample(number_of_subsamples_in_each_class, random_state = RANDOM_SEED),
      balanced_data_frame[balanced_data_frame.loc[:,output_col] == 1].sample(number_of_subsamples_in_each_class, random_state = RANDOM_SEED),
  ]).sample(frac = 1, random_state = RANDOM_SEED)

  return balanced_data_frame, input_cols, output_col


def generateExplanations(
  approach_string,
  explanation_file_name,
  model_trained,
  dataset_obj,
  factual_sample,
  norm_type_string,
  potential_observable_samples,
  standard_deviations):

  if 'MACE' in approach_string: # 'MACE_counterfactual':

    tmp_index = approach_string.find('eps')
    epsilon_string = approach_string[tmp_index + 4 : tmp_index + 8]
    epsilon = float(epsilon_string)
    return generateMACEExplanations.genExp(
      explanation_file_name,
      model_trained,
      dataset_obj,
      factual_sample,
      norm_type_string,
      epsilon
    )

  elif approach_string == 'MO': # 'minimum_observable':

    return generateMOExplanations.genExp(
      explanation_file_name,
      dataset_obj,
      factual_sample,
      potential_observable_samples,
      norm_type_string
    )

  elif approach_string == 'FT': # 'feature_tweaking':

    possible_labels = [0, 1]
    desired_label = 1
    epsilon = .5
    return generateFTExplanations.genExp(
      model_trained,
      factual_sample,
      possible_labels,
      desired_label,
      epsilon,
      norm_type_string,
      dataset_obj,
      standard_deviations,
      False
    )

  elif approach_string == 'PFT': # 'plausible_feature_tweaking':

    possible_labels = [0, 1]
    desired_label = 1
    epsilon = .5
    return generateFTExplanations.genExp(
      model_trained,
      factual_sample,
      possible_labels,
      desired_label,
      epsilon,
      norm_type_string,
      dataset_obj,
      standard_deviations,
      True
    )

  elif approach_string == 'AR': # 'actionable_recourse':

    return generateARExplanations.genExp(
      model_trained,
      factual_sample,
      norm_type_string,
      dataset_obj
    )

  else:

    raise Exception(f'{approach_string} not recognized as a valid `approach_string`.')


def runExperiments(dataset_values, model_class_values, norm_values, approaches_values, batch_number, neg_sample_count):

  for dataset_string in dataset_values:

    print(f'\n\nExperimenting with dataset_string = `{dataset_string}`')

    for model_class_string in model_class_values:

      print(f'\tExperimenting with model_class_string = `{model_class_string}`')

      for norm_type_string in norm_values:

        print(f'\t\tExperimenting with norm_type_string = `{norm_type_string}`')

        for approach_string in approaches_values:

          print(f'\t\t\tExperimenting with approach_string = `{approach_string}`')

          if model_class_string in {'tree', 'forest'}:
            one_hot = False
          elif model_class_string in {'lr', 'mlp'}:
            # if dataset_string != 'random' and dataset_string != 'mortgage': # and dataset_string != 'german':
            if dataset_string != 'random' and dataset_string != 'mortgage' and dataset_string != 'german':
              one_hot = True
            else:
              one_hot = False
          else:
            raise Exception(f'{model_class_string} not recognized as a valid `model_class_string`.')

          # prepare experiment folder
          experiment_name = f'{dataset_string}__{model_class_string}__{norm_type_string}__{approach_string}__batch{batch_number}__samples{neg_sample_count}'
          experiment_folder_name = f"_experiments/{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}__{experiment_name}"
          explanation_folder_name = f'{experiment_folder_name}/__explanation_log'
          minimum_distance_folder_name = f'{experiment_folder_name}/__minimum_distances'
          os.mkdir(f'{experiment_folder_name}')
          os.mkdir(f'{explanation_folder_name}')
          os.mkdir(f'{minimum_distance_folder_name}')
          log_file = open(f'{experiment_folder_name}/log_experiment.txt','w')

          # save some files
          dataset_obj = loadData.loadDataset(dataset_string, return_one_hot = one_hot)
          pickle.dump(dataset_obj, open(f'{experiment_folder_name}/_dataset_obj', 'wb'))

          # TODO: deprecate this code and make it similar to other datasets; we
          #       did it this way, specifically for mortgage dataset, as we wanted
          #       to test a specific test example with salary / bank balance, and
          #       we had added this specific example to the top of the test set.
          if dataset_string == 'random':

            data_train = dataset_obj.data_frame_kurz.iloc[0:1000]
            data_test = dataset_obj.data_frame_kurz.iloc[1000:2000]

            X_train = data_train[['x0', 'x1', 'x2']]
            y_train = data_train[['y']]
            X_test = data_test[['x0', 'x1', 'x2']]
            y_test = data_test[['y']]

          elif dataset_string == 'mortgage':

            data_train = dataset_obj.data_frame_kurz.iloc[0:1000]
            data_test = dataset_obj.data_frame_kurz.iloc[1000:2000]

            X_train = data_train[['x0', 'x1']]
            y_train = data_train[['y']]
            X_test = data_test[['x0', 'x1']]
            y_test = data_test[['y']]

          else:
            # construct a balanced dataframe;
            #     training portion used to train model
            #     training & testing portions used to compute counterfactuals
            balanced_data_frame, input_cols, output_col = getBalancedDataFrame(dataset_obj)

            # get train / test splits
            all_data = balanced_data_frame.loc[:,input_cols]
            all_true_labels = balanced_data_frame.loc[:,output_col]
            X_train, X_test, y_train, y_test = train_test_split(all_data, all_true_labels, train_size=.7, random_state = RANDOM_SEED)

          feature_names = dataset_obj.getInputAttributeNames('kurz') # easier to read (nothing to do with one-hot vs non-hit!)
          standard_deviations = list(X_train.std())

          # train the model
          model_trained = modelTraining.trainAndSaveModels(
            experiment_folder_name,
            model_class_string,
            X_train,
            X_test,
            y_train,
            y_test,
            feature_names
          )

          # get the negatively predicted samples (only test set)
          X_test_pred_labels = model_trained.predict(X_test)
          # ipsh()
          neg_pred_data_df = X_test.iloc[X_test_pred_labels == 0]
          pos_pred_data_df = X_test.iloc[X_test_pred_labels == 1]

          # neg_pred_data_df = neg_pred_data_df[0 : neg_sample_count] # choose only a subset to compare
          batch_start_index = batch_number * neg_sample_count
          batch_end_index = (batch_number + 1) * neg_sample_count
          neg_pred_data_df = neg_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
          pos_pred_data_df = pos_pred_data_df[0 : -1] # choose ALL to compare (critical for minimum_observable method!)

          # convert to dictionary for easier enumeration (iteration)
          neg_pred_data_dict = neg_pred_data_df.T.to_dict()
          pos_pred_data_dict = pos_pred_data_df.T.to_dict()

          # loop through the negative samples (to be saved as part of the same file of minimum distances)
          explanation_counter = 1
          all_minimum_distances = {}
          for factual_sample_index, factual_sample in neg_pred_data_dict.items():

            print(
              '\t\t\t\t'
              f'Generating explanation for\t'
              f'batch #{batch_number}\t'
              f'sample #{explanation_counter}/{len(neg_pred_data_dict.keys())}\t'
              f'(sample index {factual_sample_index}): ', end = '') # , file=log_file)
            explanation_counter = explanation_counter + 1

            explanation_file_name = f'{explanation_folder_name}/sample_{factual_sample_index}.txt'
            potential_observable_samples = pos_pred_data_dict
            explanation_object = generateExplanations(
              approach_string,
              explanation_file_name,
              model_trained,
              dataset_obj,
              factual_sample,
              norm_type_string,
              potential_observable_samples, # used solely for minimum_observable method
              standard_deviations, # used solely for feature_tweaking method
            )

            print(
              f'\tcf_found: {explanation_object["counterfactual_found"]}'
              f'\tcf_plausible: {explanation_object["counterfactual_plausible"]}'
              f'\tcf_distance: {explanation_object["counterfactual_distance"]:.4f}'
              f'\tcf_time: {explanation_object["counterfactual_time"]:.4f}'
            ) # , file=log_file)

            all_minimum_distances[f'sample_{factual_sample_index}'] = explanation_object
            # {
            #   'factual_sample': factual_sample,
            #   'counterfactual_sample': counterfactual_sample,
            #   'distance': distance,
            # }

          pickle.dump(all_minimum_distances, open(f'{experiment_folder_name}/_minimum_distances', 'wb'))
          pprint(all_minimum_distances, open(f'{experiment_folder_name}/minimum_distances.txt', 'w'))


if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument(
      '-d', '--dataset',
      nargs = '+',
      type = str,
      default = 'compass',
      help = 'Name of dataset to train model on: compass, credit, adult')

  parser.add_argument(
      '-m', '--model_class',
      nargs = '+',
      type = str,
      default = 'tree',
      help = 'Model class that will learn data: tree, forest, lr, mlp')

  parser.add_argument(
      '-n', '--norm_type',
      nargs = '+',
      type = str,
      default = 'zero_norm',
      help = 'Norm used to evaluate distance to counterfactual: zero_norm, one_norm, two_norm, infty_norm')

  parser.add_argument(
      '-a', '--approach',
      nargs = '+',
      type = str,
      default = 'MACE_eps_1e-5',
      help = 'Approach used to generate counterfactual: MACE_eps_1e-5, MO, FT, ES, AR.')

  parser.add_argument(
      '-b', '--batch_number',
      type = int,
      default = -1,
      help = 'If b = b, s = s, compute explanations for samples in range( b * s, (b + 1) * s )).')

  parser.add_argument(
      '-s', '--neg_sample_count',
      type = int,
      default = 5,
      help = 'Number of samples seeking explanations.')


  # parsing the args
  args = parser.parse_args()

  if 'FT' in args.approach or 'PFT' in args.approach:
    assert len(args.model_class) == 1, 'FeatureTweaking approach only works with forests.'
    assert \
      args.model_class[0] == 'tree' or args.model_class[0] == 'forest', \
      'FeatureTweaking approach only works with forests.'
  elif 'AR' in args.approach:
    assert len(args.model_class) == 1, 'actionableRecourse approach only works with larger.'
    assert args.model_class[0] == 'lr', 'actionableRecourse approach only works with larger.'

  runExperiments(
    args.dataset,
    args.model_class,
    args.norm_type,
    args.approach,
    args.batch_number,
    args.neg_sample_count)










