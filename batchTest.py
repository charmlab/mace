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
  import generateSATExplanations
except:
  print('[ENV WARNING] activate virtualenv to allow for testing MACE or MINT')
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


def getEpsilonInString(approach_string):
  tmp_index = approach_string.find('eps')
  epsilon_string = approach_string[tmp_index + 4 : tmp_index + 8]
  return float(epsilon_string)


def generateExplanations(
  approach_string,
  explanation_file_name,
  model_trained,
  dataset_obj,
  factual_sample,
  norm_type_string,
  observable_data_dict,
  standard_deviations):

  if 'MACE' in approach_string: # 'MACE_counterfactual':

    return generateSATExplanations.genExp(
      explanation_file_name,
      model_trained,
      dataset_obj,
      factual_sample,
      norm_type_string,
      'mace',
      getEpsilonInString(approach_string)
    )

  elif 'MINT' in approach_string: # 'MINT_counterfactual':

    return generateSATExplanations.genExp(
      explanation_file_name,
      model_trained,
      dataset_obj,
      factual_sample,
      norm_type_string,
      'mint',
      getEpsilonInString(approach_string)
    )

  elif approach_string == 'MO': # 'minimum_observable':

    return generateMOExplanations.genExp(
      explanation_file_name,
      dataset_obj,
      factual_sample,
      observable_data_dict,
      norm_type_string
    )

  elif approach_string == 'FT': # 'feature_tweaking':

    possible_labels = [0, 1]
    epsilon = .5
    perform_while_plausibility = False
    return generateFTExplanations.genExp(
      model_trained,
      factual_sample,
      possible_labels,
      epsilon,
      norm_type_string,
      dataset_obj,
      standard_deviations,
      perform_while_plausibility
    )

  elif approach_string == 'PFT': # 'plausible_feature_tweaking':

    possible_labels = [0, 1]
    epsilon = .5
    perform_while_plausibility = True
    return generateFTExplanations.genExp(
      model_trained,
      factual_sample,
      possible_labels,
      epsilon,
      norm_type_string,
      dataset_obj,
      standard_deviations,
      perform_while_plausibility
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


def runExperiments(dataset_values, model_class_values, norm_values, approaches_values, batch_number, sample_count, gen_cf_for, process_id):

  for dataset_string in dataset_values:

    print(f'\n\nExperimenting with dataset_string = `{dataset_string}`')

    for model_class_string in model_class_values:

      print(f'\tExperimenting with model_class_string = `{model_class_string}`')

      for norm_type_string in norm_values:

        print(f'\t\tExperimenting with norm_type_string = `{norm_type_string}`')

        for approach_string in approaches_values:

          print(f'\t\t\tExperimenting with approach_string = `{approach_string}`')

          if norm_type_string == 'two_norm':
            raise Exception(f'{norm_type_string} not supported.')

          if model_class_string in {'tree', 'forest'}:
            one_hot = False
          elif model_class_string in {'lr', 'mlp'}:
            one_hot = True
          else:
            raise Exception(f'{model_class_string} not recognized as a valid `model_class_string`.')

          # prepare experiment folder
          experiment_name = f'{dataset_string}__{model_class_string}__{norm_type_string}__{approach_string}__batch{batch_number}__samples{sample_count}__pid{process_id}'
          experiment_folder_name = f"_experiments/{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}__{experiment_name}"
          explanation_folder_name = f'{experiment_folder_name}/__explanation_log'
          minimum_distance_folder_name = f'{experiment_folder_name}/__minimum_distances'
          os.mkdir(f'{experiment_folder_name}')
          os.mkdir(f'{explanation_folder_name}')
          os.mkdir(f'{minimum_distance_folder_name}')
          log_file = open(f'{experiment_folder_name}/log_experiment.txt','w')

          # save some files
          dataset_obj = loadData.loadDataset(dataset_string, return_one_hot = one_hot, load_from_cache = True, debug_flag = False)
          pickle.dump(dataset_obj, open(f'{experiment_folder_name}/_dataset_obj', 'wb'))

          # construct a balanced dataframe w/ equal number of {0,1} labels;
          #     training portion used to train model
          #     testing portion used to compute counterfactuals
          balanced_data_frame, input_cols, output_col = loadData.getBalancedDataFrame(dataset_obj)

          # get train / test splits
          all_data = balanced_data_frame.loc[:,input_cols]
          all_true_labels = balanced_data_frame.loc[:,output_col]
          X_train, X_test, y_train, y_test = train_test_split(
            all_data,
            all_true_labels,
            train_size=.7,
            random_state = RANDOM_SEED)

          feature_names = dataset_obj.getInputAttributeNames('kurz') # easier to read (nothing to do with one-hot vs non-hit!)
          standard_deviations = list(X_train.std())

          # train the model
          model_trained = modelTraining.trainAndSaveModels(
            experiment_folder_name,
            model_class_string,
            dataset_string,
            X_train,
            X_test,
            y_train,
            y_test,
            feature_names
          )

          # get the predicted labels (only test set)
          # X_test = pd.concat([X_train, X_test]) # ONLY ACTIVATE THIS WHEN TEST SET IS NOT LARGE ENOUGH TO GEN' MODEL RECON DATASET
          X_test_pred_labels = model_trained.predict(X_test)

          all_pred_data_df = X_test
          # IMPORTANT: note that 'y' is actually 'pred_y', not 'true_y'
          all_pred_data_df['y'] = X_test_pred_labels
          neg_pred_data_df = all_pred_data_df.where(all_pred_data_df['y'] == 0).dropna()
          pos_pred_data_df = all_pred_data_df.where(all_pred_data_df['y'] == 1).dropna()

          batch_start_index = batch_number * sample_count
          batch_end_index = (batch_number + 1) * sample_count

          # generate counterfactuals for {only negative, negative & positive} samples
          if gen_cf_for == 'neg_only':
            iterate_over_data_df = neg_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
            observable_data_df = pos_pred_data_df
          elif gen_cf_for == 'pos_only':
            iterate_over_data_df = pos_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
            observable_data_df = neg_pred_data_df
          elif gen_cf_for == 'neg_and_pos':
            iterate_over_data_df = all_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
            observable_data_df = all_pred_data_df
          else:
            raise Exception(f'{gen_cf_for} not recognized as a valid `gen_cf_for`.')

          # convert to dictionary for easier enumeration (iteration)
          iterate_over_data_dict = iterate_over_data_df.T.to_dict()
          observable_data_dict = observable_data_df.T.to_dict()

          # loop through samples for which we desire a counterfactual,
          # (to be saved as part of the same file of minimum distances)
          explanation_counter = 1
          all_minimum_distances = {}
          for factual_sample_index, factual_sample in iterate_over_data_dict.items():

            factual_sample['y'] = bool(factual_sample['y'])

            print(
              '\t\t\t\t'
              f'Generating explanation for\t'
              f'batch #{batch_number}\t'
              f'sample #{explanation_counter}/{len(iterate_over_data_dict.keys())}\t'
              f'(sample index {factual_sample_index}): ', end = '') # , file=log_file)
            explanation_counter = explanation_counter + 1
            explanation_file_name = f'{explanation_folder_name}/sample_{factual_sample_index}.txt'

            explanation_object = generateExplanations(
              approach_string,
              explanation_file_name,
              model_trained,
              dataset_obj,
              factual_sample,
              norm_type_string,
              observable_data_dict, # used solely for minimum_observable method
              standard_deviations, # used solely for feature_tweaking method
            )

            if 'MACE' in approach_string:
              print(
                f'\t- cfe_found: {explanation_object["cfe_found"]} -'
                f'\t- cfe_plaus: {explanation_object["cfe_plausible"]} -'
                f'\t- cfe_time: {explanation_object["cfe_time"]:.4f} -'
                f'\t- int_dist: N/A -'
                f'\t- cfe_dist: {explanation_object["cfe_distance"]:.4f} -'
              ) # , file=log_file)
            elif 'MINT' in approach_string:
              print(
                f'\t- scf_found: {explanation_object["scf_found"]} -'
                f'\t- scf_plaus: {explanation_object["scf_plausible"]} -'
                f'\t- scf_time: {explanation_object["scf_time"]:.4f} -'
                f'\t- int_dist: {explanation_object["int_distance"]:.4f} -'
                f'\t- scf_dist: {explanation_object["scf_distance"]:.4f} -'
              ) # , file=log_file)

            all_minimum_distances[f'sample_{factual_sample_index}'] = explanation_object

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
      help = 'Norm used to evaluate distance to counterfactual: zero_norm, one_norm, infty_norm') # two_norm

  parser.add_argument(
      '-a', '--approach',
      nargs = '+',
      type = str,
      default = 'MACE_eps_1e-5',
      help = 'Approach used to generate counterfactual: MACE_eps_1e-3, MINT_eps_1e-3, MO, FT, AR.') # ES

  parser.add_argument(
      '-b', '--batch_number',
      type = int,
      default = -1,
      help = 'If b = b, s = s, compute explanations for samples in range( b * s, (b + 1) * s )).')

  parser.add_argument(
      '-s', '--sample_count',
      type = int,
      default = 5,
      help = 'Number of samples seeking explanations.')

  parser.add_argument(
      '-g', '--gen_cf_for',
      type = str,
      default = 'neg_only',
      help = 'Decide whether to generate counterfactuals for negative pred samples only, or for both negative and positive pred samples.')

  parser.add_argument(
      '-p', '--process_id',
      type = str,
      default = '0',
      help = 'When running parallel tests on the cluster, process_id guarantees (in addition to time stamped experiment folder) that experiments do not conflict.')


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
    args.sample_count,
    args.gen_cf_for,
    args.process_id)










