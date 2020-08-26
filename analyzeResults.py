import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# sns.set(style="darkgrid")
sns.set_context("paper")

from pprint import pprint

from debug import ipsh



def gatherAndSaveDistances():

  # parent_folders = [
  #   '/Volumes/amir/dev/mace/_experiments/__2019.07.30__merged_unconstrained_MO_PFT_AR',
  #   '/Volumes/amir/dev/mace/_experiments/__2019.07.30__merged_unconstrained_MACE_eps_1e-1',
  #   '/Volumes/amir/dev/mace/_experiments/__2019.09.19__merged_unconstrained_MACE_eps_1e-3',
  #   '/Volumes/amir/dev/mace/_experiments/__2019.09.19__merged_unconstrained_MACE_eps_1e-5'
  # ]

  # parent_folders = [
  #   '/Volumes/amir/dev/mace/_experiments/__2019.07.30__merged_unconstrained_MO_PFT_AR',
  #   '/Volumes/amir/dev/mace/_experiments/__2019.09.29__merged_unconstrained_MACE_eps_1e-2__tree_forest_lr',
  #   '/Volumes/amir/dev/mace/_experiments/__2019.09.29__merged_unconstrained_MACE_eps_1e-3__tree_forest_lr',
  #   '/Volumes/amir/dev/mace/_experiments/__2019.09.29__merged_unconstrained_MACE_eps_1e-5__tree_forest_lr'
  # ]

  # parent_folders = [
  #   '/Volumes/amir/dev/mace/_experiments/__2019.07.30__merged_constrained_MO_AR__lr',
  #   '/Volumes/amir/dev/mace/_experiments/__2019.09.20__merged_constrained_MACE_eps_1e-3__tree_forest_lr'
  # ]
  # year = '2019'

  parent_folders = [
    './_experiments'
    # '/Users/a6karimi/dev/mace/_experiments/__merged_german-lr-one_norm-MACE_eps_1e-3'
  ]
  year = '2020'

  all_child_folders = []
  for parent_folder in parent_folders:
    child_folders = os.listdir(parent_folder)
    child_folders = [x for x in child_folders if year in x and x[0] != '.'] # remove .DS_Store, etc.
    child_folders = [os.path.join(parent_folder, x) for x in child_folders]
    all_child_folders.extend(child_folders) # happens in place

  # DATASET_VALUES = ['adult', 'credit', 'compass']
  # MODEL_CLASS_VALUES = ['tree', 'forest', 'lr'] # , 'mlp']
  # NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
  # APPROACHES_VALUES = ['MACE_eps_1e-2', 'MACE_eps_1e-3', 'MACE_eps_1e-5', 'MO', 'PFT', 'AR']

  DATASET_VALUES = ['compass', 'credit', 'adult']
  # MODEL_CLASS_VALUES = ['lr', 'tree', 'forest', 'mlp2x10']
  MODEL_CLASS_VALUES = ['mlp2x10']
  NORM_VALUES = ['zero_norm', 'one_norm', 'two_norm', 'infty_norm']
  APPROACHES_VALUES = ['MACE_MIP_OBJ_eps_1e-3', 'MACE_MIP_EXP_eps_1e-3', 'MACE_MIP_SAT_eps_1e-3', 'MACE_SAT_eps_1e-3']

  # all_counter = 72 + 18 + 6 # (without the unneccessary FT folders for LR and MLP)
  # assert len(all_child_folders) == all_counter, 'missing, or too many experiment folders'
  all_counter = len(DATASET_VALUES) * len(MODEL_CLASS_VALUES) * len(NORM_VALUES) * len(APPROACHES_VALUES)

  df_all_distances = pd.DataFrame({ \
    'dataset': [], \
    'model': [], \
    'norm': [], \
    'approach': [], \
    # 'approach_param': [], \
    'factual sample index': [], \
    'counterfactual found': [], \
    'counterfactual plausible': [], \
    'counterfactual distance': [], \
    'counterfactual time': [], \
    'all counterfactual distances': [], \
    'all counterfactual times': [], \
    'changed age': [], \
    'changed gender': [], \
    'changed race': [], \
    # 'changed attributes': [], \
    'age constant': [], \
    'age increased': [], \
    'age decreased': [], \
    # 'interventional distance': [], \
  })

  print('Loading and merging all distance files.')

  counter = 0

  for dataset_string in DATASET_VALUES:

    for model_class_string in MODEL_CLASS_VALUES:

      for norm_type_string in NORM_VALUES:

        for approach_string in APPROACHES_VALUES:

          if approach_string == 'PFT':
            if model_class_string != 'tree' and model_class_string != 'forest':
              continue
          elif approach_string == 'AR':
            if model_class_string != 'lr':
              continue

          counter = counter + 1

          matching_child_folders = [
            x for x in all_child_folders if
            f'__{dataset_string}__' in x.split('/')[-1] and
            f'__{model_class_string}__' in x.split('/')[-1] and
            f'__{norm_type_string}__' in x.split('/')[-1] and
            f'__{approach_string}' in x.split('/')[-1]
          ]

          # if approach_string == 'MACE_eps_1e-5':
          #   tmp_index = minimum_distance_file_path.find('eps')
          #   epsilon_string = minimum_distance_file_path[tmp_index + 4 : tmp_index + 8]
          #   approach_param = float(epsilon_string)
          # else:
          #   approach_param = -1

          # Find results folder
          try:
            assert len(matching_child_folders) == 1, f'Expecting only 1 folder, but we found {len(matching_child_folders)}.'
            matching_child_folder = matching_child_folders[0]
            minimum_distance_file_path = os.path.join(matching_child_folder, '_minimum_distances')
          except:
            print(f'\t[{counter} / (max {all_counter})] Cannot find folder for {dataset_string}-{model_class_string}-{norm_type_string}-{approach_string}')
            continue

          # Find results file
          try:
            assert os.path.isfile(minimum_distance_file_path)
            print(f'\t[{counter} / (max {all_counter})] Successfully found folder {matching_child_folder.split("/")[-1]}, found min dist file, ', end = '')
            minimum_distance_file = pickle.load(open(minimum_distance_file_path, 'rb'))
            print(f'adding {len(minimum_distance_file.keys())} distances.')
          except:
            print(f'Cannot find file {minimum_distance_file_path}')
            continue

          # Add results to global results data frame
          # try:
          for key in minimum_distance_file.keys():
            factual_sample = minimum_distance_file[key]['fac_sample']
            counterfactual_sample = minimum_distance_file[key]['cfe_sample']
            changed_age = False
            changed_gender = False
            changed_race = False
            # if dataset_string == 'adult':
            #   if not np.isclose(factual_sample['x0'], counterfactual_sample['x0']):
            #     changed_gender = True
            #   if not np.isclose(factual_sample['x1'], counterfactual_sample['x1']):
            #     changed_age = True

            # elif dataset_string == 'credit':
            #   if not np.isclose(factual_sample['x0'], counterfactual_sample['x0']):
            #     changed_gender = True
            #   if model_class_string == 'tree' or model_class_string == 'forest': # non-hot
            #     if not np.isclose(factual_sample['x2'], counterfactual_sample['x2']):
            #       changed_age = True
            #   else: # one-hot
            #     if not np.isclose(factual_sample['x2_ord_0'], counterfactual_sample['x2_ord_0']) or \
            #        not np.isclose(factual_sample['x2_ord_1'], counterfactual_sample['x2_ord_1']) or \
            #        not np.isclose(factual_sample['x2_ord_2'], counterfactual_sample['x2_ord_2']) or \
            #        not np.isclose(factual_sample['x2_ord_3'], counterfactual_sample['x2_ord_3']):
            #       changed_age = True
            # elif dataset_string == 'compass':
            #   if model_class_string == 'tree' or model_class_string == 'forest': # non-hot
            #     if not np.isclose(factual_sample['x0'], counterfactual_sample['x0']):
            #       changed_age = True
            #   else: # one-hot
            #     if not np.isclose(factual_sample['x0_ord_0'], counterfactual_sample['x0_ord_0']) or \
            #        not np.isclose(factual_sample['x0_ord_1'], counterfactual_sample['x0_ord_1']) or \
            #        not np.isclose(factual_sample['x0_ord_2'], counterfactual_sample['x0_ord_2']):
            #       changed_age = True
            #   if not np.isclose(factual_sample['x1'], counterfactual_sample['x1']):
            #     changed_race = True
            #   if not np.isclose(factual_sample['x2'], counterfactual_sample['x2']):
            #     changed_gender = True

            # changed_attributes = []
            # for attr in factual_sample.keys():
            #   if not isinstance(factual_sample[attr], float):
            #     print(attr)
            #     print(f'factual_sample[attr]: {factual_sample}')
            #     print(f'counterfactual_sample[attr]: {counterfactual_sample}')
            #   if not np.isclose(factual_sample[attr], counterfactual_sample[attr]):
            #     changed_attributes.append(attr)

            age_constant = False
            age_increased = False
            age_decreased = False
            # if dataset_string == 'adult':
            #   if factual_sample['x1'] < counterfactual_sample['x1']:
            #     age_increased = True
            #   elif factual_sample['x1'] == counterfactual_sample['x1']:
            #     age_constant = True
            #   elif factual_sample['x1'] > counterfactual_sample['x1']:
            #     age_decreased = True

            # append rows

            if 'MACE' in approach_string and 'all_counterfactuals' in minimum_distance_file.keys():
              all_counterfactual_distances = list(map(lambda x: x['counterfactual_distance'], minimum_distance_file[key]['all_counterfactuals']))
              all_counterfactual_times = list(map(lambda x: x['time'], minimum_distance_file[key]['all_counterfactuals']))
            else:
              all_counterfactual_distances = []
              all_counterfactual_times = []

            d = minimum_distance_file[key]['cfe_distance']

            # MACE_SAT and MACE_MIP_SAT do have perfect coverage but they simply don't terminate on some samples.
            if (d == np.infty or not(0.0 <= d <= 1.0)) and not('MACE_SAT' in approach_string or 'MACE_MIP_SAT' in approach_string or 'dice' in approach_string):
              raise Exception("Not perfect coverage.")

            df_all_distances = df_all_distances.append({
              'dataset': dataset_string,
              'model': model_class_string,
              'norm': norm_type_string,
              'approach': approach_string,
              # 'approach_param': approach_param,
              'factual sample index': key,
              'counterfactual found': (minimum_distance_file[key]['cfe_found'] and minimum_distance_file[key]['cfe_distance'] != np.infty),
              'counterfactual plausible': (minimum_distance_file[key]['cfe_plausible'] and minimum_distance_file[key]['cfe_distance'] != np.infty),
              'counterfactual distance': minimum_distance_file[key]['cfe_distance'],
              'counterfactual time': minimum_distance_file[key]['cfe_time'],
              'all counterfactual distances': all_counterfactual_distances,
              'all counterfactual times': all_counterfactual_times,
              'changed age': changed_age,
              'changed gender': changed_gender,
              'changed race': changed_race,
              # 'changed attributes': changed_attributes,
              'age constant': age_constant,
              'age increased': age_increased,
              'age decreased': age_decreased,
              # 'interventional distance': minimum_distance_file[key]['interventional_distance'],
            }, ignore_index =  True)
  # ipsh()
          # except:
          #   print(f'Problem with adding row in data frame.')


  print('Processing merged distance files.')

  print('Saving merged distance files.')

  pickle.dump(df_all_distances, open(f'_results/df_all_distances', 'wb'))


def gatherAndSaveDiversities():

  parent_folders = [
    './_experiments'
    # '/Users/a6karimi/dev/mace/_experiments/__merged_german-lr-one_norm-MACE_eps_1e-3'
  ]
  year = '2020'

  all_child_folders = []
  for parent_folder in parent_folders:
    child_folders = os.listdir(parent_folder)
    child_folders = [x for x in child_folders if year in x and x[0] != '.'] # remove .DS_Store, etc.
    child_folders = [os.path.join(parent_folder, x) for x in child_folders]
    all_child_folders.extend(child_folders) # happens in place

  DATASET_VALUES = ['compass']
  MODEL_CLASS_VALUES = ['mlp2x10']
  NORM_VALUES = ['one_norm']
  APPROACHES_VALUES = ['MACE_MIP_OBJ_DIVERSE_eps_1e-3', 'dice']
  K_CFES = ['2', '3', '4', '5', '6', '7', '8', '9', '10'] # Number of diverse CFs

  # all_counter = 72 + 18 + 6 # (without the unneccessary FT folders for LR and MLP)
  # assert len(all_child_folders) == all_counter, 'missing, or too many experiment folders'
  all_counter = len(DATASET_VALUES) * len(MODEL_CLASS_VALUES) * len(NORM_VALUES) * len(APPROACHES_VALUES)

  df_all_distances = pd.DataFrame({ \
    'dataset': [], \
    'model': [], \
    'norm': [], \
    'approach': [], \
    # 'approach_param': [], \
    'factual sample index': [], \
    'counterfactual found': [], \
    'counterfactual plausible': [], \
    'counterfactual distance': [], \
    'counterfactual time': [], \
    'all counterfactual distances': [], \
    'all counterfactual times': [], \
    'mean diversity': [], \
    'mean proximity': [], \
    'num of cfs': [], \
    'changed age': [], \
    'changed gender': [], \
    'changed race': [], \
    'age constant': [], \
    'age increased': [], \
    'age decreased': [], \
  })

  print('Loading and merging all distance files.')

  counter = 0

  for dataset_string in DATASET_VALUES:

    for model_class_string in MODEL_CLASS_VALUES:

      for norm_type_string in NORM_VALUES:

        for approach_string in APPROACHES_VALUES:

          for k_cfe_string in K_CFES:

            counter = counter + 1

            matching_child_folders = [
              x for x in all_child_folders if
              f'__{dataset_string}__' in x.split('/')[-1] and
              f'__{model_class_string}__' in x.split('/')[-1] and
              f'__{norm_type_string}__' in x.split('/')[-1] and
              f'__{approach_string}' in x.split('/')[-1] and
              f'__cfs{k_cfe_string}__' in x.split('/')[-1]
            ]

            # Find results folder
            try:
              assert len(matching_child_folders) == 1, f'Expecting only 1 folder, but we found {len(matching_child_folders)}.'
              matching_child_folder = matching_child_folders[0]
              minimum_distance_file_path = os.path.join(matching_child_folder, '_minimum_distances')
            except:
              print(f'\t[{counter} / (max {all_counter})] Cannot find folder for {dataset_string}-{model_class_string}-{norm_type_string}-{approach_string}')
              continue

            # Find results file
            try:
              assert os.path.isfile(minimum_distance_file_path)
              print(f'\t[{counter} / (max {all_counter})] Successfully found folder {matching_child_folder.split("/")[-1]}, found min dist file, ', end = '')
              minimum_distance_file = pickle.load(open(minimum_distance_file_path, 'rb'))
              print(f'adding {len(minimum_distance_file.keys())} distances.')
            except:
              print(f'Cannot find file {minimum_distance_file_path}')
              continue

            # Add results to global results data frame
            # try:
            for key in minimum_distance_file.keys():
              factual_sample = minimum_distance_file[key]['fac_sample']
              counterfactual_sample = minimum_distance_file[key]['cfe_sample']
              changed_age = False
              changed_gender = False
              changed_race = False

              age_constant = False
              age_increased = False
              age_decreased = False

              if 'MACE' in approach_string and 'all_counterfactuals' in minimum_distance_file.keys():
                all_counterfactual_distances = list(map(lambda x: x['counterfactual_distance'], minimum_distance_file[key]['all_counterfactuals']))
                all_counterfactual_times = list(map(lambda x: x['time'], minimum_distance_file[key]['all_counterfactuals']))
              else:
                all_counterfactual_distances = []
                all_counterfactual_times = []

              d = minimum_distance_file[key]['cfe_distance']

              # MACE_SAT and MACE_MIP_SAT do have perfect coverage but they simply don't terminate on some samples.
              if (d == np.infty or not(0.0 <= d <= 1.0)) and not('MACE_SAT' in approach_string or 'MACE_MIP_SAT' in approach_string or 'dice' in approach_string):
                raise Exception("Not perfect coverage.")

              df_all_distances = df_all_distances.append({
                'dataset': dataset_string,
                'model': model_class_string,
                'norm': norm_type_string,
                'approach': approach_string,
                # 'approach_param': approach_param,
                'factual sample index': key,
                'counterfactual found': (minimum_distance_file[key]['cfe_found'] and minimum_distance_file[key]['cfe_distance'] != np.infty),
                'counterfactual plausible': (minimum_distance_file[key]['cfe_plausible'] and minimum_distance_file[key]['cfe_distance'] != np.infty),
                'counterfactual distance': minimum_distance_file[key]['cfe_distance'],
                'counterfactual time': minimum_distance_file[key]['cfe_time'],
                'all counterfactual distances': all_counterfactual_distances,
                'all counterfactual times': all_counterfactual_times,
                'mean diversity': minimum_distance_file[key]['mean_diversity'],
                'mean proximity': minimum_distance_file[key]['mean_proximity'],
                'num of cfs': minimum_distance_file[key]['num_cfs'],
                'changed age': changed_age,
                'changed gender': changed_gender,
                'changed race': changed_race,
                # 'changed attributes': changed_attributes,
                'age constant': age_constant,
                'age increased': age_increased,
                'age decreased': age_decreased,
                # 'interventional distance': minimum_distance_file[key]['interventional_distance'],
              }, ignore_index =  True)


  print('Processing merged distance files.')

  print('Saving merged distance files.')

  pickle.dump(df_all_distances, open(f'_results/df_all_diversities', 'wb'))

def gatherAndSaveDistanceTimeTradeoffData():

  # unconstrained
  # DATASET_VALUES = ['adult', 'credit', 'compass']
  # MODEL_CLASS_VALUES = ['tree', 'forest', 'lr'] # , 'mlp']
  # NORM_VALUES = ['one_norm']
  # APPROACHES_VALUES = ['MO', 'PFT', 'AR', 'MACE_eps_1e-2', 'MACE_eps_1e-3', 'MACE_eps_1e-5']
  DATASET_VALUES = ['credit', 'adult']
  MODEL_CLASS_VALUES = ['mlp2x10', 'tree', 'lr']
  NORM_VALUES = ['zero_norm', 'one_norm', 'two_norm', 'infty_norm']
  APPROACHES_VALUES = ['MACE_MIP_OBJ_eps_1e-3', 'MACE_MIP_EXP_eps_1e-3']

  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()

  tmp_df = pd.DataFrame({ \
    'factual_sample_index': [], \
    'dataset': [], \
    'model': [], \
    'norm': [], \
    'approach': [], \
    'iteration': [], \
    'distance': [], \
    'time': [], \
  })

  counter = 1
  total_counter = len(DATASET_VALUES) * len(MODEL_CLASS_VALUES) * len(NORM_VALUES) * len(APPROACHES_VALUES)
  for model_class_string in MODEL_CLASS_VALUES:

    for norm_type_string in NORM_VALUES:

      for dataset_string in DATASET_VALUES:

        for approach_string in APPROACHES_VALUES:

          df = df_all_distances.where(
            (df_all_distances['dataset'] == dataset_string) &
            (df_all_distances['model'] == model_class_string) &
            (df_all_distances['norm'] == norm_type_string) &
            (df_all_distances['approach'] == approach_string),
          ).dropna()

          print(f'[INFO] (#{counter} / {total_counter}) Processing {dataset_string}-{model_class_string}-{norm_type_string}-{approach_string}...')
          counter = counter + 1

          if df.shape[0]: # if any tests exist for this setup

            if 'MACE' in approach_string:

              # max_iterations_over_all_factual_samples
              max_iterations = max(list(map(lambda x : len(x), df_all_distances['all counterfactual times'])))
              # print(f'max_iterations: {max_iterations}')

              for index, row in df.iterrows():

                all_counterfactual_distances = row['all counterfactual distances'][1:] # remove the first elem (np.infty)
                all_counterfactual_times = row['all counterfactual times'][1:] # remove the first elem (np.infty)
                assert len(all_counterfactual_distances) == len(all_counterfactual_times)
                # IMPORTANT: keep repeating last elem of array so that all factual
                # samples have the same number of iterations (this is important
                # for later when we take the average for any iteration; we do not
                # want the plot to come down-to-the-right, then go up last minute
                # Importantly, the repeating of last element should be done prior
                # to cumsum. max_iterations - len(array) - 1 (-1 because we remove
                # the first elem (np.infty))
                # all_counterfactual_distances.extend([all_counterfactual_distances[-1]] * (max_iterations - len(all_counterfactual_distances) - 1))
                # all_counterfactual_times.extend([all_counterfactual_times[-1]] * (max_iterations - len(all_counterfactual_times) - 1))
                # Now (and only after the 2 lines above), perform cumulation sum
                cum_counterfactual_times = np.cumsum(all_counterfactual_times)

                for iteration_counter in range(len(all_counterfactual_distances)):

                  tmp_df = tmp_df.append({
                    'factual_sample_index': row['factual sample index'],
                    'dataset': dataset_string,
                    'model': model_class_string,
                    'norm': norm_type_string,
                    'approach': approach_string,
                    'iteration': int(iteration_counter) + 1,
                    'distance': all_counterfactual_distances[iteration_counter],
                    'time': cum_counterfactual_times[iteration_counter],
                  }, ignore_index =  True)

            else:

              for index, row in df.iterrows():

                for iteration_counter in range(15):

                  tmp_df = tmp_df.append({
                    'factual_sample_index': row['factual sample index'],
                    'dataset': dataset_string,
                    'model': model_class_string,
                    'norm': norm_type_string,
                    'approach': approach_string,
                    'iteration': int(iteration_counter) + 1,
                    'distance': row['counterfactual distance'],
                    'time': row['counterfactual time'],
                  }, ignore_index =  True)

  pickle.dump(tmp_df, open(f'_results/df_all_distance_vs_time', 'wb'))


def latexify(fig_width=None, fig_height=None, columns=1, largeFonts=False, font_scale=1):
  """Set up matplotlib's RC params for LaTeX plotting.
  Call this before plotting a figure.

  Parameters
  ----------
  fig_width : float, optional, inches
  fig_height : float,  optional, inches
  columns : {1, 2}
  """

  # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

  # Width and max height in inches for IEEE journals taken from
  # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

  assert(columns in [1, 2])

  if fig_width is None:
      fig_width = 3.39 if columns == 1 else 6.9  # width in inches

  if fig_height is None:
      golden_mean = (np.sqrt(5) - 1.0) / 2.0    # Aesthetic ratio
      fig_height = fig_width * golden_mean  # height in inches

  MAX_HEIGHT_INCHES = 8.0
  if fig_height > MAX_HEIGHT_INCHES:
      print("WARNING: fig_height too large:" + fig_height +
            "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
      fig_height = MAX_HEIGHT_INCHES

  params = {'backend': 'ps',
            'text.latex.preamble': ['\\usepackage{gensymb}'],
            # fontsize for x and y labels (was 10)
            'axes.labelsize': font_scale * 10 if largeFonts else font_scale * 7,
            'axes.titlesize': font_scale * 10 if largeFonts else font_scale * 7,
            'font.size': font_scale * 10 if largeFonts else font_scale * 7,  # was 10
            'legend.fontsize': font_scale * 10 if largeFonts else font_scale * 7,  # was 10
            'xtick.labelsize': font_scale * 10 if largeFonts else font_scale * 7,
            'ytick.labelsize': font_scale * 10 if largeFonts else font_scale * 7,
            'text.usetex': True,
            'figure.figsize': [fig_width, fig_height],
            'font.family': 'serif',
            'xtick.minor.size': 0.5,
            'xtick.major.pad': 1.5,
            'xtick.major.size': 1,
            'ytick.minor.size': 0.5,
            'ytick.major.pad': 1.5,
            'ytick.major.size': 1,
            'lines.linewidth': 1.5,
            'lines.markersize': 0.1,
            'hatch.linewidth': 0.5
            }

  matplotlib.rcParams.update(params)
  plt.rcParams.update(params)


def analyzeRelativeDistances():
  # DATASET_VALUES = ['adult', 'credit', 'compass']
  # MODEL_CLASS_VALUES = ['tree', 'forest', 'lr', 'mlp']
  # NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
  # # APPROACHES_VALUES = ['MACE_eps_1e-1', 'MACE_eps_1e-3', 'MACE_eps_1e-5', 'MO', 'PFT', 'AR']
  # # APPROACHES_VALUES = ['MACE_eps_1e-3', 'MACE_eps_1e-5', 'MO', 'PFT', 'AR']
  # # APPROACHES_VALUES = ['MACE_eps_1e-5', 'MO', 'PFT', 'AR']
  # APPROACHES_VALUES = ['MACE_eps_1e-2', 'MO', 'PFT', 'AR']
  # # mace_baseline = 'MACE_eps_1e-5'
  # mace_baseline = 'MACE_eps_1e-2'

  DATASET_VALUES = ['credit', 'adult']
  MODEL_CLASS_VALUES = ['mlp2x10', 'tree', 'lr']
  NORM_VALUES = ['zero_norm', 'one_norm', 'two_norm', 'infty_norm']
  APPROACHES_VALUES = ['MACE_MIP_OBJ_eps_1e-3', 'MACE_MIP_EXP_eps_1e-3']

  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()

  print('Analyzing merged distance files.')

  df = df_all_distances

  MIN_SAMPLES_REQUIRED = 0

  for model_class_string in MODEL_CLASS_VALUES:
    for dataset_string in DATASET_VALUES:
      for norm_type_string in NORM_VALUES:

        # speficic dataset, speficic model, speficic norm, all approaches
        df = df_all_distances.where(
          (df_all_distances['dataset'] == dataset_string) &
          (df_all_distances['model'] == model_class_string) &
          (df_all_distances['norm'] == norm_type_string),
          # inplace = True
        ).dropna()

        if df.shape[0]: # if any tests exist for this setup

          tmp_string = f'{dataset_string}-{model_class_string}-{norm_type_string}'

          # for each approach, get the index of factual samples for which counterfactuals were computed
          factual_sample_index_per_approach = {}
          for approach_string in APPROACHES_VALUES:
            factual_sample_index_per_approach[approach_string] = \
              np.unique(df.where(df['approach'] == approach_string).dropna()['factual sample index'])

          # # MACE works in all scenarios
          # assert \
          #   len(factual_sample_index_per_approach['MACE_eps_1e-5']) >= MIN_SAMPLES_REQUIRED, \
          #   f'Expecting at least {MIN_SAMPLES_REQUIRED} samples for MACE, got {len(factual_sample_index_per_approach["MACE"])} ({tmp_string})'

          # # MO works in all scenarios
          # assert \
          #   len(factual_sample_index_per_approach['MO']) >= MIN_SAMPLES_REQUIRED, \
          #   f'Expecting at least {MIN_SAMPLES_REQUIRED} samples for MO, got {len(factual_sample_index_per_approach["MO"])} ({tmp_string})'

          # # TODO: FT works in all scenarios, except for adult tree??????
          # if model_class_string == 'tree' or model_class_string == 'forest':
          #   assert \
          #     len(factual_sample_index_per_approach['PFT']) >= MIN_SAMPLES_REQUIRED, \
          #     f'Expecting at least {MIN_SAMPLES_REQUIRED} samples for PFT, got {len(factual_sample_index_per_approach["PFT"])} ({tmp_string})'

          # # AR works in all scenarios, except for zero-norm
          # if model_class_string == 'lr':
          #   if norm_type_string != 'zero_norm':
          #     assert \
          #       len(factual_sample_index_per_approach['AR']) >= MIN_SAMPLES_REQUIRED, \
          #       f'Expecting at least {MIN_SAMPLES_REQUIRED} samples for AR, got {len(factual_sample_index_per_approach["AR"])} ({tmp_string})'

          # remove keys that don't have any factual sample indices
          tmp = factual_sample_index_per_approach
          tmp = dict((key, value) for (key, value) in tmp.items() if len(tmp[key]) > 0)
          factual_sample_index_per_approach = tmp
          # for key in factual_sample_index_per_approach.keys():
          #   print(f'key: {key}, num factual samples: {len(factual_sample_index_per_approach[key])}')

          # compute 1 - d_MACE / d_{MO, FT, ...}
          all_but_mace_approaches = list(np.setdiff1d(
            np.array(list(factual_sample_index_per_approach.keys())),
            np.array(mace_baseline)
          ))
          factual_sample_index_intersect = []
          for approach_string in all_but_mace_approaches:
            factual_sample_index_intersect = np.intersect1d(
              factual_sample_index_per_approach[mace_baseline],
              factual_sample_index_per_approach[approach_string]
            )
            assert len(factual_sample_index_intersect) >= MIN_SAMPLES_REQUIRED, f'Expecting at least {MIN_SAMPLES_REQUIRED} intersecting samples between MACE and {approach_string}'
            distance_reduction_list = []
            for factual_sample_index in factual_sample_index_intersect:
              sample_mace = df.where(
                (df['approach'] == mace_baseline) &
                (df['factual sample index'] == factual_sample_index)
              ).dropna().T.to_dict()
              assert len(sample_mace.keys()) == 1, f'Expecting only 1 sample with index {factual_sample_index} for approach {approach_string}'
              sample_other = df.where(
                (df['approach'] == approach_string) &
                (df['factual sample index'] == factual_sample_index)
              ).dropna().T.to_dict()
              assert len(sample_other.keys()) == 1, f'Expecting only 1 sample with index {factual_sample_index} for approach {approach_string}'
              minimum_distance_mace = sample_mace[list(sample_mace.keys())[0]]['counterfactual distance']
              minimum_distance_other = sample_other[list(sample_other.keys())[0]]['counterfactual distance']
              distance_reduction_list.append(1 - minimum_distance_mace / minimum_distance_other)
            tmp_mean = np.mean(np.array(distance_reduction_list)) * 100
            tmp_std = np.std(np.array(distance_reduction_list)) * 100
            print(f'\t Distance reduction for {dataset_string} {model_class_string} {norm_type_string} (1 - d_MACE / d_{approach_string}) = \t {tmp_mean:.2f} +/- {tmp_std:.2f} \t (N = {len(distance_reduction_list)})')


def analyzeAverageDistanceRunTimeCoverage():
  # DATASET_VALUES = ['adult', 'credit', 'compass']
  # MODEL_CLASS_VALUES = ['tree', 'forest', 'lr'] # , 'mlp']
  # NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
  # # APPROACHES_VALUES = ['MACE_eps_1e-3', 'MACE_eps_1e-5', 'MO', 'PFT', 'AR']
  # APPROACHES_VALUES = ['MACE_eps_1e-2', 'MACE_eps_1e-3', 'MACE_eps_1e-5']

  DATASET_VALUES = ['credit', 'adult']
  MODEL_CLASS_VALUES = ['mlp2x10', 'tree', 'lr']
  NORM_VALUES = ['zero_norm', 'one_norm', 'two_norm', 'infty_norm']
  APPROACHES_VALUES = ['MACE_MIP_OBJ_eps_1e-3', 'MACE_MIP_EXP_eps_1e-3']

  # DATASET_VALUES = ['adult', 'credit', 'compass']
  # MODEL_CLASS_VALUES = ['mlp']
  # NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
  # APPROACHES_VALUES = ['MACE_eps_1e-3', 'MACE_eps_1e-5']

  # APPROACHES_VALUES = ['MACE_eps_1e-3', 'MACE_eps_1e-5', 'MO'] # COVERAGE = %100 ALWAYS
  # APPROACHES_VALUES = ['PFT', 'AR']
  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  # DO NOT INCLUDE THE LINES BELOW!!!!!!!!!!!!!!!!!!!! WHY??? B/c we want to count statistics below
  # df_all_distances = df_all_distances.where(
  #   (df_all_distances['counterfactual found'] == True) &
  #   (df_all_distances['counterfactual plausible'] == True)
  # ).dropna()
  for model_class_string in MODEL_CLASS_VALUES:
    for approach_string in APPROACHES_VALUES:
      for dataset_string in DATASET_VALUES:
        for norm_type_string in NORM_VALUES:
          df = df_all_distances.where(
            (df_all_distances['dataset'] == dataset_string) &
            (df_all_distances['model'] == model_class_string) &
            (df_all_distances['norm'] == norm_type_string) &
            (df_all_distances['approach'] == approach_string),
          ).dropna()
          if df.shape[0]: # if any tests exist for this setup
            found_and_plausible = df.where((df['counterfactual found'] == True) & (df['counterfactual plausible'] == True))
            found_and_not_plausible = df.where((df['counterfactual found'] == True) & (df['counterfactual plausible'] == False))
            not_found = df.where(df['counterfactual found'] == False)
            count_found_and_plausible = found_and_plausible.dropna().shape[0]
            count_found_and_not_plausible = found_and_not_plausible.dropna().shape[0]
            count_not_found = not_found.dropna().shape[0]
            assert df.shape[0] == \
              count_found_and_plausible + \
              count_found_and_not_plausible + \
              count_not_found
            average_distance = found_and_plausible['counterfactual distance'].mean() # this is NOT a good way to compare methods! see analyzeRelativeDistances() instead, as it compares ratio of distances for the same samples!
            std_distance = found_and_plausible['counterfactual distance'].std()
            average_run_time = found_and_plausible['counterfactual time'].mean()
            std_run_time = found_and_plausible['counterfactual time'].std()
            coverage = count_found_and_plausible / df.shape[0] * 100
            print(f'{model_class_string}-{approach_string}-{dataset_string}-{norm_type_string} ({count_found_and_plausible} plausible samples found):')
            print(f'\tAvg distance: {average_distance:.2f} +/- {std_distance:.2f}')
            print(f'\tAvg run-time: {average_run_time:.2f} +/- {std_run_time:.2f} seconds')
            print(f'\tCoverage: %{coverage}')


# TODO: make updates to this function based on plotAllDistancesAppendix
# def plotDistancesMainBody():
#   DATASET_VALUES = ['adult', 'credit', 'compass']
#   MODEL_CLASS_VALUES = ['lr']
#   NORM_VALUES = ['one_norm', 'infty_norm']
#   APPROACHES_VALUES = ['MACE_eps_1e-1', 'MACE_eps_1e-3', 'MACE_eps_1e-5', 'MO', 'AR']
#   # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
#   df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
#   df_all_distances = df_all_distances.where(
#     (df_all_distances['counterfactual found'] == True) &
#     (df_all_distances['counterfactual plausible'] == True)
#   ).dropna()

#   # change norms for plotting
#   df_all_distances = df_all_distances.where(df_all_distances['norm'] != 'zero_norm').dropna()

#   df_all_distances['norm'] = df_all_distances['norm'].map({
#     'zero_norm': r'$\ell_0$',
#     'one_norm': r'$\ell_1$',
#     'infty_norm': r'$\ell_\infty$',
#   })

#   df_all_distances['dataset'] = df_all_distances['dataset'].map({
#     'adult': 'Adult',
#     'credit': 'Credit',
#     'compass': 'COMPAS',
#   })

#   df_all_distances['approach'] = df_all_distances['approach'].map({
#     'MACE_eps_1e-1': r'MACE ($\epsilon = 10^{-1}$)',
#     'MACE_eps_1e-3': r'MACE ($\epsilon = 10^{-3}$)',
#     'MACE_eps_1e-5': r'MACE ($\epsilon = 10^{-5}$)',
#     'MO': 'MO',
#     'PFT': 'PFT',
#     'AR': 'AR',
#   })

#   print('Plotting merged distance files.')

#   for model_string in MODEL_CLASS_VALUES:

#     model_specific_df = df_all_distances.where(df_all_distances['model'] == model_string).dropna()

#     if model_string == 'tree' or model_string == 'forest':
#       hue_order = [r'MACE ($\epsilon = 10^{-1}$)', r'MACE ($\epsilon = 10^{-3}$)', r'MACE ($\epsilon = 10^{-5}$)', 'MO', 'PFT']
#     elif model_string == 'lr':
#       hue_order = [r'MACE ($\epsilon = 10^{-1}$)', r'MACE ($\epsilon = 10^{-3}$)', r'MACE ($\epsilon = 10^{-5}$)', 'MO', 'AR']
#     elif model_string == 'mlp':
#       hue_order = [r'MACE ($\epsilon = 10^{-1}$)', r'MACE ($\epsilon = 10^{-3}$)', r'MACE ($\epsilon = 10^{-5}$)', 'MO']

#     latexify(1.5 * 6, 6, font_scale = 1.2)
#     ax = sns.catplot(
#       x = 'dataset',
#       y = 'counterfactual distance',
#       hue = 'approach',
#       hue_order = hue_order,
#       col = 'norm',
#       data = model_specific_df,
#       kind = 'box',
#       height = 2.5,
#       aspect = 1,
#       palette = sns.color_palette("muted", 5),
#       sharey = False,
#       whis = np.inf,
#     )
#     ax.set(ylim=(0,None))
#     ax.set_axis_labels("", r"Distance $\delta$ to" + "\nNearest Counterfactual")
#     ax.set_titles('{col_name}')
#     ax.set_xlabels() # remove "dataset" on the x-axis
#     ax.savefig(f'_results/distances_{model_string}_main_body.png', dpi = 400)


def plotAllDistancesAppendix():
  time_or_distance = 'distance'

  MODEL_CLASS_VALUES = ['lr', 'tree', 'forest']
  # MODEL_CLASS_VALUES = ['mlp2x10']
  NORM_VALUES = ['zero_norm', 'one_norm', 'two_norm', 'infty_norm']
  DATASET_VALUES = ['compass', 'credit', 'adult']
  # APPROACHES_VALUES = ['MACE_MIP_OBJ_eps_1e-3', 'MACE_MIP_EXP_eps_1e-3', 'MACE_MIP_SAT_eps_1e-3', 'MACE_SAT_eps_1e-3']
  APPROACHES_VALUES = ['MACE_MIP_OBJ_eps_1e-3', 'MACE_MIP_EXP_eps_1e-3', 'MACE_SAT_eps_1e-3']

  # tmp_constrained = 'constrained'
  tmp_constrained = 'unconstrained'
  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = pickle.load(open(f'_results/df_all_distances_otherThanMLP', 'rb'))

  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()

  # get the intersection of all approaches... (as some do not have perfect coverage)
  filtered_df_all_distance = None
  n_samples = {}
  for model in MODEL_CLASS_VALUES:
    n_samples[model] = {}
    for i, norm in enumerate(NORM_VALUES):
      n_samples[model][i] = {}
      for dataset in DATASET_VALUES:
        specific_df = df_all_distances.where(
          (df_all_distances['model'] == model) &
          (df_all_distances['norm'] == norm) &
          (df_all_distances['dataset'] == dataset)
        ).dropna()

        n_approaches = len(specific_df['approach'].unique())
        fac_sample_occurances = specific_df['factual sample index'].value_counts()
        for fac_sample in fac_sample_occurances.keys():
          if fac_sample_occurances[fac_sample] < n_approaches:
            specific_df = specific_df.where(specific_df['factual sample index'] != fac_sample).dropna()

        if len(specific_df) > 0:
          assert len(specific_df) % n_approaches == 0
          n_common_samples = int(len(specific_df) / n_approaches)
        else:
          n_common_samples = 0

        if dataset == 'adult':
          n_samples[model][i]['Adult'] = n_common_samples
        elif dataset == 'credit':
          n_samples[model][i]['Credit'] = n_common_samples
        elif dataset == 'compass':
          n_samples[model][i]['COMPAS'] = n_common_samples
        else:
          raise Exception(f"{dataset} not a recognized dataset.")

        if filtered_df_all_distance is None:
          filtered_df_all_distance = specific_df
        else:
          filtered_df_all_distance = pd.concat([filtered_df_all_distance, specific_df])


  # change norms for plotting??????
  # df_all_distances = df_all_distances.where(df_all_distances['norm'] != 'zero_norm').dropna()
  df_all_distances = filtered_df_all_distance

  df_all_distances['norm'] = df_all_distances['norm'].map({
    'zero_norm': r'$\ell_0$',
    'one_norm': r'$\ell_1$',
    'two_norm': r'$\ell_2$',
    'infty_norm': r'$\ell_\infty$',
  })

  df_all_distances['dataset'] = df_all_distances['dataset'].map({
    'adult': 'Adult',
    'credit': 'Credit',
    'compass': 'COMPAS',
  })

  df_all_distances['approach'] = df_all_distances['approach'].map({
    # 'MACE_eps_1e-1': r'MACE ($\epsilon = 10^{-1}$)',
    # 'MACE_eps_1e-2': r'MACE ($\epsilon = 10^{-2}$)',
    'MACE_MIP_EXP_eps_1e-3': r'MIP\_EXP ($\epsilon = 10^{-3}$)',
    'MACE_MIP_OBJ_eps_1e-3': r'MIP\_OBJ ($\epsilon = 10^{-3}$)',
    'MACE_SAT_eps_1e-3': r'SAT ($\epsilon = 10^{-3}$)',
    'MACE_MIP_SAT_eps_1e-3': r'MIP\_SAT ($\epsilon = 10^{-3}$)',
    'dice': 'DiCE',
    # 'MACE_eps_1e-3': r'MACE ($\epsilon = 10^{-3}$)',
    # 'MACE_eps_1e-5': r'MACE ($\epsilon = 10^{-5}$)',
    # 'MO': 'MO',
    # 'PFT': 'PFT',
    # 'AR': 'AR',
  })

  print(f'Plotting merged {time_or_distance} files.')

  for model_string in MODEL_CLASS_VALUES:

    model_specific_df = df_all_distances.where(df_all_distances['model'] == model_string).dropna()

    # hue_order = [r'MIP\_OBJ ($\epsilon = 10^{-3}$)', r'MIP\_EXP ($\epsilon = 10^{-3}$)',
    #              r'MIP\_SAT ($\epsilon = 10^{-3}$)', r'SAT ($\epsilon = 10^{-3}$)']
    hue_order = [r'MIP\_OBJ ($\epsilon = 10^{-3}$)', r'MIP\_EXP ($\epsilon = 10^{-3}$)',
                 r'SAT ($\epsilon = 10^{-3}$)']
    # hue_order = [r'MIP\_OBJ ($\epsilon = 10^{-3}$)', 'DiCE']
    # hue_order = [r'MACE ($\epsilon = 10^{-3}$)']
    # if model_string == 'tree' or model_string == 'forest':
    #   hue_order.extend(['MO', 'PFT'])
    # elif model_string == 'lr':
    #   hue_order.extend(['MO', 'AR'])
    # elif model_string == 'mlp':
    #   hue_order.extend(['MO'])

    latexify(1.5 * 6, 6, font_scale = 1.2)
    sns.set_style("whitegrid")

    ax = sns.catplot(
      x = 'dataset',
      y = f'counterfactual {time_or_distance}',
      hue = 'approach',
      hue_order = hue_order,
      col = 'norm',
      data = model_specific_df,
      kind = 'box',

      # kind = 'violin',
      # kind = 'swarm',
      height = 3.5,
      aspect = .9,
      palette = sns.color_palette("muted", 5),
      sharey = False,
      whis = np.inf,
      legend_out = False,
    )
    # ax.legend(loc = 'lower left', ncol = 1, fancybox = True, shadow = True, fontsize = 'small')
    # ax.fig.get_children()[-1].set_bbox_to_anchor((1.1, 0.5, 0, 0))
    ax.fig.get_axes()[0].legend().remove()

    # write the number of samples under dataset name
    # for i in range(len(ax.fig.axes)):
    #   labels = []
    #   for dataset_text in ax.fig.axes[i].get_xticklabels():
    #     labels.append(dataset_text._text + f'\n(n = {n_samples[model_string][i][dataset_text._text]})')
    #   ax.fig.axes[i].set_xticklabels(labels)

    if time_or_distance == 'distance':
      ax.fig.get_axes()[0].legend(loc='upper right', fancybox=True, shadow=True, fontsize='small')
      ax.set(ylim=(0, None))
      ax.set_axis_labels("", r"Distance $\delta$ to" + "\nNearest Counterfactual")
    else:
      ax.fig.get_axes()[2].legend(loc='upper right', fancybox=True, shadow=True, fontsize='small')
      ax.set_axis_labels("", r"Time in seconds to find" + "\nNearest Counterfactual")
      # ticks = [-4, -3, -2, -1, 0, 1, 2, 3]
      # labels = [i for i in ticks]
      ax.set(yscale='log')
    ax.set_titles('{col_name}')
    ax.set_xlabels() # remove "dataset" on the x-axis
    ax.savefig(f'_results/all_{time_or_distance}__{model_string}.png', dpi = 400)


def plotScalibility():

  DATASET_VALUES = ['compass', 'credit', 'adult']
  MODEL_CLASS_VALUES = []
  for i in range(1, 21):
    MODEL_CLASS_VALUES.append(f'mlp{i}x10')
  # for i in range(20, 401, 20):
  #   MODEL_CLASS_VALUES.append(f'mlp2x{i}')
  NORM_VALUES = ['one_norm']
  APPROACHES_VALUES = ['MACE_MIP_OBJ_eps_1e-3', 'MACE_MIP_EXP_eps_1e-3', 'MACE_MIP_SAT_eps_1e-3', 'MACE_SAT_eps_1e-3']

  # tmp_constrained = 'constrained'
  tmp_constrained = 'unconstrained'
  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible

  df_all_distances = pickle.load(open(f'_results/df_all_distances_scalibility', 'rb'))

  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()

  # get the intersection of all approaches... (as some do not have perfect coverage)
  filtered_df_all_distance = None
  n_samples = {}
  for model in MODEL_CLASS_VALUES:
    n_samples[model] = {}
    for i, norm in enumerate(NORM_VALUES):
      n_samples[model][i] = {}
      for dataset in DATASET_VALUES:
        specific_df = df_all_distances.where(
          (df_all_distances['model'] == model) &
          (df_all_distances['norm'] == norm) &
          (df_all_distances['dataset'] == dataset)
        ).dropna()

        n_approaches = len(specific_df['approach'].unique())
        fac_sample_occurances = specific_df['factual sample index'].value_counts()
        for fac_sample in fac_sample_occurances.keys():
          if fac_sample_occurances[fac_sample] < n_approaches:
            specific_df = specific_df.where(specific_df['factual sample index'] != fac_sample).dropna()

        if len(specific_df) > 0:
          assert len(specific_df) % n_approaches == 0
          n_common_samples = int(len(specific_df) / n_approaches)
        else:
          n_common_samples = 0

        if dataset == 'adult':
          n_samples[model][i]['Adult'] = n_common_samples
        elif dataset == 'credit':
          n_samples[model][i]['Credit'] = n_common_samples
        elif dataset == 'compass':
          n_samples[model][i]['COMPAS'] = n_common_samples
        else:
          raise Exception(f"{dataset} not a recognized dataset.")

        if filtered_df_all_distance is None:
          filtered_df_all_distance = specific_df
        else:
          filtered_df_all_distance = pd.concat([filtered_df_all_distance, specific_df])


  # change norms for plotting??????
  # df_all_distances = df_all_distances.where(df_all_distances['norm'] != 'zero_norm').dropna()
  df_all_distances = filtered_df_all_distance

  df_all_distances['norm'] = df_all_distances['norm'].map({
    'zero_norm': r'$\ell_0$',
    'one_norm': r'$\ell_1$',
    'two_norm': r'$\ell_2$',
    'infty_norm': r'$\ell_\infty$',
  })

  df_all_distances['dataset'] = df_all_distances['dataset'].map({
    'adult': 'Adult',
    'credit': 'Credit',
    'compass': 'COMPAS',
  })

  df_all_distances['approach'] = df_all_distances['approach'].map({
    'MACE_MIP_EXP_eps_1e-3': r'MIP_EXP ($\epsilon = 10^{-3}$)',
    'MACE_MIP_OBJ_eps_1e-3': r'MIP_OBJ ($\epsilon = 10^{-3}$)',
    'MACE_SAT_eps_1e-3': r'SAT ($\epsilon = 10^{-3}$)',
    'MACE_MIP_SAT_eps_1e-3': r'MIP_SAT ($\epsilon = 10^{-3}$)',
    'dice': 'DiCE',
  })

  print(f'Plotting merged files.')

  fig, axs = plt.subplots(1, len(DATASET_VALUES), figsize=(16, 6))

  for i, dataset_string in enumerate(df_all_distances['dataset'].unique()):

    dataset_specific_df = df_all_distances.where(df_all_distances['dataset'] == dataset_string).dropna()

    for approach_string in dataset_specific_df['approach'].unique():

      approach_specific_df = dataset_specific_df.where(dataset_specific_df['approach'] == approach_string).dropna()

      mean_runtimes, std_runtimes, labels = [], [], []
      for mlp_type in MODEL_CLASS_VALUES:
        mean_runtimes.append(
          approach_specific_df.where(approach_specific_df['model'] == mlp_type).dropna()['counterfactual time'].mean())
        std_runtimes.append(
          approach_specific_df.where(approach_specific_df['model'] == mlp_type).dropna()['counterfactual time'].std())
        labels.append(mlp_type.split('x')[0].replace('mlp', ''))
        # labels.append(n_samples[mlp_type][0][dataset_string])

      est = np.array(mean_runtimes)
      sd = np.array(std_runtimes)
      ax = axs[i]
      ax.scatter(np.arange(len(mean_runtimes)), mean_runtimes)
      ax.plot(np.arange(len(mean_runtimes)), mean_runtimes, label=approach_string)
      # ax.fill_between(np.arange(len(mean_runtimes)), est-sd, est+sd, alpha=0.2)
      ax.set_xticks(np.arange(len(mean_runtimes)))
      ax.set_xticklabels(labels, rotation=65)
      ax.set_title(f'{dataset_string}')
      ax.set_xlabel('MLP depth')

    axs[i].grid()
    axs[i].set_yscale("log")
    axs[0].set_ylabel("Runtime in seconds")
    axs[0].legend()

  # plt.show()
  fig.tight_layout()
  plt.savefig(f'_results/scalibility.png', bboc_inches='tight', pad_inches=0, dpi=400)


def plotDiversity():

  DATASET_VALUES = ['compass']
  MODEL_CLASS_VALUES = ['mlp2x10']
  NORM_VALUES = ['one_norm']
  APPROACHES_VALUES = ['MACE_MIP_OBJ_DIVERSE_eps_1e-3', 'dice']
  K_CFES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
  KEY_TO_PLOT = 'counterfactual time'

  df_all_distances = pickle.load(open(f'_results/df_all_diversities', 'rb'))

  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()

  # get the intersection of all approaches... (as some do not have perfect coverage)
  filtered_df_all_distance = None
  n_samples = {}
  for model in MODEL_CLASS_VALUES:
    n_samples[model] = {}
    for i, norm in enumerate(NORM_VALUES):
      n_samples[model][i] = {}
      for dataset in DATASET_VALUES:
        n_samples[model][i][dataset] = {}
        for k_cfe in K_CFES:

          specific_df = df_all_distances.where(
            (df_all_distances['model'] == model) &
            (df_all_distances['norm'] == norm) &
            (df_all_distances['dataset'] == dataset) &
            (df_all_distances['num of cfs'] == k_cfe)
          ).dropna()

          n_approaches = len(specific_df['approach'].unique())
          fac_sample_occurances = specific_df['factual sample index'].value_counts()
          for fac_sample in fac_sample_occurances.keys():
            if fac_sample_occurances[fac_sample] < n_approaches:
              specific_df = specific_df.where(specific_df['factual sample index'] != fac_sample).dropna()

          if len(specific_df) > 0:
            assert len(specific_df) % n_approaches == 0
            n_common_samples = int(len(specific_df) / n_approaches)
          else:
            n_common_samples = 0

          n_samples[model][i][dataset][k_cfe] = n_common_samples

          if filtered_df_all_distance is None:
            filtered_df_all_distance = specific_df
          else:
            filtered_df_all_distance = pd.concat([filtered_df_all_distance, specific_df])


  # change norms for plotting??????
  # df_all_distances = df_all_distances.where(df_all_distances['norm'] != 'zero_norm').dropna()
  df_all_distances = filtered_df_all_distance

  df_all_distances['norm'] = df_all_distances['norm'].map({
    'zero_norm': r'$\ell_0$',
    'one_norm': r'$\ell_1$',
    'two_norm': r'$\ell_2$',
    'infty_norm': r'$\ell_\infty$',
  })

  df_all_distances['dataset'] = df_all_distances['dataset'].map({
    'adult': 'Adult',
    'credit': 'Credit',
    'compass': 'COMPAS',
  })

  df_all_distances['approach'] = df_all_distances['approach'].map({
    'MACE_MIP_OBJ_DIVERSE_eps_1e-3': r'MIP_DIVERSE ($\epsilon = 10^{-3}$)',
    'dice': 'DiCE',
  })

  markers = {
    'MIP_DIVERSE ($\epsilon = 10^{-3}$)' : 'D',
    'DiCE': 's'
  }

  print(f'Plotting merged files.')

  fig, axs = plt.subplots(1, len(DATASET_VALUES), figsize=(6, 6))

  for model in MODEL_CLASS_VALUES:
    for norm in df_all_distances['norm'].unique():
      for i, dataset_string in enumerate(df_all_distances['dataset'].unique()):
        ax = axs[i] if len(DATASET_VALUES) > 1 else axs
        for approach_string in df_all_distances['approach'].unique():

          specific_df = df_all_distances.where(
            (df_all_distances['model'] == model) &
            (df_all_distances['norm'] == norm) &
            (df_all_distances['dataset'] == dataset_string) &
            (df_all_distances['approach'] == approach_string)
          ).dropna()

          feature_to_plot, labels = [], []
          for k_cfe in K_CFES:
            feature_to_plot.append(specific_df.where(specific_df['num of cfs'] == k_cfe).dropna()[KEY_TO_PLOT].mean())
            labels.append(f'{k_cfe} CFs')

          ax.scatter(np.arange(len(feature_to_plot)), feature_to_plot)
          ax.plot(np.arange(len(feature_to_plot)), feature_to_plot, label=approach_string, marker=markers[approach_string], markersize=8)
          ax.set_xticks(np.arange(len(feature_to_plot)))
          ax.set_xticklabels(labels, rotation=65)
          ax.set_title(f'{dataset_string}')
          # ax.set_xlabel('')

        ax.grid()
        if 'time' in KEY_TO_PLOT:
          ax.set_yscale("log")
        ax.set_ylabel(KEY_TO_PLOT)
        ax.legend()

  # plt.show()
  fig.tight_layout()
  plt.savefig(f'_results/diversity_{KEY_TO_PLOT}.png', bboc_inches='tight', pad_inches=0, dpi=400)


def plotAvgDistanceRunTimeCoverageTradeoffAgainstIterations():
  DATASET_VALUES = ['credit', 'adult']
  MODEL_CLASS_VALUES = ['mlp2x10','tree','lr']
  NORM_VALUES = ['zero_norm', 'one_norm', 'two_norm', 'infty_norm']
  APPROACHES_VALUES = ['MACE_MIP_OBJ_eps_1e-3', 'MACE_MIP_EXP_eps_1e-3']

  # tmp_constrained = 'constrained'
  tmp_constrained = 'unconstrained'
  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df_all_distance_vs_time = pickle.load(open(f'_results/df_all_distance_vs_time', 'rb'))

  # df_all_distance_vs_time = df_all_distance_vs_time.where(df_all_distance_vs_time['iteration'] <= 10).dropna()


  df_all_distances['counterfactual found and plausible'] = df_all_distances.apply(
    lambda row : row['counterfactual found'] and row['counterfactual plausible'],
    axis = 1
  )

  # df_all_distance_vs_time['norm'] = df_all_distance_vs_time['norm'].map({
  #   'zero_norm': r'$\ell_0$',
  #   'one_norm': r'$\ell_1$',
  #   'infty_norm': r'$\ell_\infty$',
  # })

  df_all_distances['dataset'] = df_all_distances['dataset'].map({
    'adult': 'Adult',
    'credit': 'Credit',
    'compass': 'COMPAS',
  })

  df_all_distance_vs_time['dataset'] = df_all_distance_vs_time['dataset'].map({
    'adult': 'Adult',
    'credit': 'Credit',
    'compass': 'COMPAS',
  })

  df_all_distances['approach'] = df_all_distances['approach'].map({
    # 'MACE_eps_1e-1': r'MACE ($\epsilon = 10^{-1}$)',
    'MACE_eps_1e-2': r'MACE ($\epsilon = 10^{-2}$)',
    'MACE_eps_1e-3': r'MACE ($\epsilon = 10^{-3}$)',
    'MACE_eps_1e-5': r'MACE ($\epsilon = 10^{-5}$)',
    'MO': 'MO',
    'PFT': 'PFT',
    'AR': 'AR',
  })

  df_all_distance_vs_time['approach'] = df_all_distance_vs_time['approach'].map({
    # 'MACE_eps_1e-1': r'MACE ($\epsilon = 10^{-1}$)',
    'MACE_eps_1e-2': r'MACE ($\epsilon = 10^{-2}$)',
    'MACE_eps_1e-3': r'MACE ($\epsilon = 10^{-3}$)',
    'MACE_eps_1e-5': r'MACE ($\epsilon = 10^{-5}$)',
    'MO': 'MO',
    'PFT': 'PFT',
    'AR': 'AR',
  })

  for model_class_string in MODEL_CLASS_VALUES:

    dataset_order = ['Adult', 'Credit', 'COMPAS']

    approach_order = [r'MACE ($\epsilon = 10^{-2}$)', r'MACE ($\epsilon = 10^{-3}$)', r'MACE ($\epsilon = 10^{-5}$)']
    # approach_order = [r'MACE ($\epsilon = 10^{-3}$)', r'MACE ($\epsilon = 10^{-5}$)']
    # approach_order = [r'MACE ($\epsilon = 10^{-3}$)']
    if model_class_string == 'tree' or model_class_string == 'forest':
      approach_order.extend(['MO', 'PFT'])
    elif model_class_string == 'lr':
      approach_order.extend(['MO', 'AR'])
    elif model_class_string == 'mlp':
      approach_order.extend(['MO'])

    for norm_type_string in NORM_VALUES:

      print(f'[INFO] Processing {model_class_string}-{norm_type_string}...')

      tmp_df = df_all_distance_vs_time.where(
        # (df_all_distance_vs_time['dataset'] == 'credit') &
        (df_all_distance_vs_time['model'] == model_class_string) &
        # (df_all_distance_vs_time['approach'] == 'AR') &
        (df_all_distance_vs_time['norm'] == norm_type_string), # &
      ).dropna()

      tmp_df_2 = df_all_distances.where(
        # (df_all_distances['dataset'] == 'credit') &
        (df_all_distances['model'] == model_class_string) &
        # (df_all_distances['approach'] == 'AR') &
        (df_all_distances['norm'] == norm_type_string), # &
      ).dropna()

      # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
      sns.set_style("whitegrid")
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

      ax1.set(yscale="log")
      ax2.set(yscale="log")
      sns.lineplot(
        x = "iteration",
        y = "time",
        style = 'dataset',
        style_order = dataset_order,
        hue = "approach",
        hue_order = approach_order,
        markers = False,
        dashes = True,
        data = tmp_df,
        legend = False,
        ax = ax1)
      sns.lineplot(
        x = "iteration",
        y = "distance",
        style = 'dataset',
        style_order = dataset_order,
        hue = "approach",
        hue_order = approach_order,
        markers = False,
        dashes = True,
        data = tmp_df,
        legend = 'full',
        ax = ax2)
      # sns.barplot(
      #   x = 'dataset',
      #   y = 'counterfactual found and plausible',
      #   hue = 'approach',
      #   hue_order = approach_order,
      #   data = tmp_df_2,
      #   ax = ax3)

      # ax1.set(ylim = (0, 60))
      # ax2.set(ylim = (0, 0.5))
      # ax2.legend(loc = 'upper center', bbox_to_anchor = (-.1, 1.15), ncol = 5, fancybox = True, shadow = True)
      # ax2.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
      # ax1.legend(loc = 'lower right', ncol = 2, fancybox = True, shadow = True, fontsize = 'small')
      # ax2.legend(loc = 'upper right', ncol = 2, fancybox = True, shadow = True, fontsize = 'small')
      ax2.legend(loc = 'lower left', ncol = 2, fancybox = True, shadow = True, fontsize = 'small')
      # ax3.legend(loc = 'lower center', ncol = 1, fancybox = True, shadow = True, fontsize = 'small')


      ax1.set_xlabel(r"# Calls to SAT Solver - $O(\log(1 / \epsilon))$")
      ax1.set_ylabel(r"Time $\tau$ to compute" + "\nNearest Counterfactual")
      ax2.set_xlabel(r"# Calls to SAT Solver - $O(\log(1 / \epsilon))$")
      ax2.set_ylabel(r"Distance $\delta$ to" + "\nNearest Counterfactual")
      # ax3.set_xlabel('') # remove "dataset" on the x-axis
      # ax3.set_ylabel(r"Coverage $\Omega$")

      fig.tight_layout()
      fig.savefig(f'_results/{tmp_constrained}__avg_tradeoff__{model_class_string}_{norm_type_string}.png', dpi = 300)

      # tmp_df = tmp_df.sample(10000)
      # tmp_df['time'] = tmp_df['time'].apply(lambda x: np.floor(x * .5) / .5)
      # fig, ax = plt.subplots(figsize=(8, 8))
      # # ax.set(xscale="log", yscale="log")
      # sns.lineplot(
      #   x = "time",
      #   y = "distance",
      #   hue = 'dataset',
      #   style = "approach",
      #   markers = True,
      #   dashes = True,
      #   data = tmp_df,
      #   legend = 'brief',
      #   ax = ax)
      # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)
      # # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
      # # ax1.set(ylim=(0, 60))
      # # ax2.set(ylim=(0, 0.5))
      # fig.savefig(f'_results/test_distance_vs_time__{model_class_string}_{norm_type_string}.png', dpi = 400)


      # OTHER

      # g = sns.FacetGrid(tmp_df, col="dataset", hue="approach", margin_titles=True)
      # g.map(plt.scatter, "time", "distance", alpha=.4)
      # g.add_legend()
      # g.savefig(f'_results/distance_vs_time__{model_class_string}_{norm_type_string}_scatter.png', dpi = 400)

      # ipsh()
      # fig, ax = plt.subplots(figsize=(8, 8))
      # ax.set(xscale="log", yscale="log")
      # # ax.set_aspect("equal")
      # ar = tmp_df.query("approach == 'AR'")
      # mo = tmp_df.query("approach == 'MO'")
      # mace = tmp_df.query("approach == 'MACE_eps_1e-5'")
      # ax = sns.kdeplot(ar.time, ar.distance, cmap = "Greens", shade = True, shade_lowest = False)
      # ax = sns.kdeplot(mo.time, mo.distance, cmap = "Reds", shade = True, shade_lowest = False)
      # ax = sns.kdeplot(mace.time, mace.distance, cmap = "Blues", shade = True, shade_lowest = False)
      # fig.savefig(f'_results/test_{model_class_string}_{norm_type_string}.png', dpi = 400)


      # tmp_df = tmp_df.sample(1000)
      # fig, ax = plt.subplots(figsize=(8, 8))
      # ax.set(xscale="log", yscale="log")
      # # ax.set_aspect("equal")
      # ar = tmp_df.query("approach == 'AR'")
      # mo = tmp_df.query("approach == 'MO'")
      # mace = tmp_df.query("approach == 'MACE_eps_1e-5'")
      # ax = sns.scatterplot(ar.time, ar.distance, cmap = "Greens")
      # ax = sns.scatterplot(mo.time, mo.distance, cmap = "Reds")
      # ax = sns.scatterplot(mace.time, mace.distance, cmap = "Blues")
      # fig.savefig(f'_results/test_{model_class_string}_{norm_type_string}_scatter.png', dpi = 400)


def compareMACEandMINT():
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df = df_all_distances
  counterfactual_distances = np.array(df['counterfactual distance'])
  counterfactual_distances[counterfactual_distances > 1] = 1
  counterfactual_distances = counterfactual_distances[counterfactual_distances != 0]
  interventional_distances = np.array(df['interventional distance'])
  interventional_distances[interventional_distances > 1] = 1
  interventional_distances = interventional_distances[interventional_distances != 0]
  mean_distance_ratio = np.mean(counterfactual_distances / interventional_distances)
  std_distance_ratio = np.std(counterfactual_distances / interventional_distances)
  print(f'MACE / MINT distances: {mean_distance_ratio:.4f} +/- {std_distance_ratio:.4f}')

if __name__ == '__main__':
  # gatherAndSaveDistances()
  # gatherAndSaveDiversities()
  # compareMACEandMINT()
  # gatherAndSaveDistanceTimeTradeoffData()

  # analyzeRelativeDistances()
  # analyzeAverageDistanceRunTimeCoverage()

  # plotDistancesMainBody()
  # plotAllDistancesAppendix()
  # plotScalibility()
  plotDiversity()
  # plotAvgDistanceRunTimeCoverageTradeoffAgainstIterations()


  # measureEffectOfRaceCompass()
  # measureSensitiveAttributeChange()
  # DEPRECATED # measureEffectOfAgeCompass()
  # measureEffectOfAgeAdultPart1()
  # measureEffectOfAgeAdultPart2()
  # measureEffectOfAgeAdultPart3()
































