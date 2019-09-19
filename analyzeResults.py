import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from pprint import pprint

from debug import ipsh

parent_folders = [
  # '/Volumes/amir/dev/mace/_experiments/_may_20_all_results',
  # '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change',
  # '/Users/a6karimi/dev/mace/_results/_may_20_all_results_backup'
  '/Volumes/amir/dev/mace/_experiments/__merged'
  # '/Volumes/amir/dev/mace/_experiments/__merged_all_unconstrained_tests_epsilon_1e-1_2019.07.30'
  # '/Users/a6karimi/dev/mace/_results/__merged_all_unconstrained_tests_epsilon_1e-1_2019.07.30'
  # '/Users/a6karimi/dev/mace/_experiments/'
  # '/Users/a6karimi/dev/mace/_results/__merged'
]

all_child_folders = []
for parent_folder in parent_folders:
  child_folders = os.listdir(parent_folder)
  child_folders = [x for x in child_folders if '2019' in x and x[0] != '.'] # remove .DS_Store, etc.
  child_folders = [os.path.join(parent_folder, x) for x in child_folders]
  all_child_folders.extend(child_folders) # happens in place

DATASET_VALUES = ['adult', 'credit', 'compass']
MODEL_CLASS_VALUES = ['tree', 'forest', 'lr'] # MLP
NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
APPROACHES_VALUES = ['SAT']

# 48 tests (54 tests - PFT x Adult x {tree, forest})
# DATASET_VALUES = ['adult', 'credit', 'compass']
# MODEL_CLASS_VALUES = ['tree', 'forest']
# NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
# APPROACHES_VALUES = ['SAT', 'MO', 'PFT']

# 18 tests
# DATASET_VALUES = ['adult', 'credit', 'compass']
# MODEL_CLASS_VALUES = ['lr']
# NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
# APPROACHES_VALUES = ['SAT', 'MO']

# 6 tests
# DATASET_VALUES = ['adult', 'credit', 'compass']
# MODEL_CLASS_VALUES = ['lr']
# NORM_VALUES = ['one_norm', 'infty_norm']
# APPROACHES_VALUES = ['AR']

# 18 tests
# DATASET_VALUES = ['adult', 'credit', 'compass']
# MODEL_CLASS_VALUES = ['mlp']
# NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
# APPROACHES_VALUES = ['SAT', 'MO']


all_counter = 72 + 18  + 6 # (without the unneccessary FT folders for LR and MLP)
# assert len(all_child_folders) == all_counter, 'missing, or too many experiment folders'

def gatherAndSaveDistances():
  df_all_distances = pd.DataFrame({ \
    'dataset': [], \
    'model': [], \
    'norm': [], \
    'approach': [], \
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

          try:
            assert len(matching_child_folders) == 1, f'Expecting only 1 folder, but we found {len(matching_child_folders)}.'
            matching_child_folder = matching_child_folders[0]
            minimum_distance_file_path = os.path.join(matching_child_folder, '_minimum_distances')
          except:
            print(f'\t[{counter} / {all_counter}] Cannot find folder for {dataset_string}-{model_class_string}-{norm_type_string}-{approach_string}')
            continue

          try:
            assert os.path.isfile(minimum_distance_file_path)
            print(f'\t[{counter} / {all_counter}] Successfully found folder {matching_child_folder.split("/")[-1]}, found min dist file, ', end = '')
            minimum_distance_file = pickle.load(open(minimum_distance_file_path, 'rb'))
            print(f'adding {len(minimum_distance_file.keys())} distances.')
          except:
            print(f'Cannot find file {minimum_distance_file_path}')

          # try:
          for key in minimum_distance_file.keys():
            factual_sample = minimum_distance_file[key]['factual_sample']
            counterfactual_sample = minimum_distance_file[key]['counterfactual_sample']
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

            if approach_string == 'SAT':
              all_counterfactual_distances = list(map(lambda x: x['distance'], minimum_distance_file[key]['all_counterfactuals']))
              all_counterfactual_times = list(map(lambda x: x['time'], minimum_distance_file[key]['all_counterfactuals']))
            else:
              all_counterfactual_distances = []
              all_counterfactual_times = []

            df_all_distances = df_all_distances.append({
              'dataset': dataset_string,
              'model': model_class_string,
              'norm': norm_type_string,
              'approach': approach_string,
              'factual sample index': key,
              'counterfactual found': minimum_distance_file[key]['counterfactual_found'],
              'counterfactual plausible': minimum_distance_file[key]['counterfactual_plausible'],
              'counterfactual distance': minimum_distance_file[key]['counterfactual_distance'],
              'counterfactual time': minimum_distance_file[key]['counterfactual_time'],
              'all counterfactual distances': all_counterfactual_distances,
              'all counterfactual times': all_counterfactual_times,
              'changed age': changed_age,
              'changed gender': changed_gender,
              'changed race': changed_race,
              # 'changed attributes': changed_attributes,
              'age constant': age_constant,
              'age increased': age_increased,
              'age decreased': age_decreased,
            }, ignore_index =  True)
  # ipsh()
          # except:
          #   print(f'Problem with adding row in data frame.')


  print('Processing merged distance files.')


  # TODO: Maybe move this to after saving so you can implement coverage here....
  # TODO: Also, all methods (except ours which seg-faults) should have 500 rows
  #       and each model-dataset pair should have identical indices.
  # for dataset_string in DATASET_VALUES:
  #   for model_class_string in MODEL_CLASS_VALUES:
  #     for norm_type_string in NORM_VALUES:
  #       for approach_string in APPROACHES_VALUES:
  #         df = df_all_distances.where(
  #           (df_all_distances['dataset'] == dataset_string) &
  #           (df_all_distances['model'] == model_class_string) &
  #           (df_all_distances['norm'] == norm_type_string) &
  #           (df_all_distances['approach'] == approach_string),
  #         ).dropna()
  #         # ipsh()
  #         found_and_plausible = df.where((df['counterfactual found'] == True) & (df['counterfactual plausible'] == True)).dropna().shape[0]
  #         found_and_not_plausible = df.where((df['counterfactual found'] == True) & (df['counterfactual plausible'] == False)).dropna().shape[0]
  #         not_found = df.where(df['counterfactual found'] == False).dropna().shape[0]
  #         assert df.shape[0] == \
  #           found_and_plausible + \
  #           found_and_not_plausible + \
  #           not_found
  #         print(f'{dataset_string}-{model_class_string}-{norm_type_string}-{approach_string}:'.ljust(40), end = '')
  #         print(f'found_and_plausible: {found_and_plausible}, found_and_not_plausible: {found_and_not_plausible}, not_found: {not_found}')

  print('Saving merged distance files.')

  pickle.dump(df_all_distances, open(f'_results/df_all_distances', 'wb'))


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


def plotDistancesMainBody():
  DATASET_VALUES = ['adult', 'credit', 'compass']
  MODEL_CLASS_VALUES = ['lr']
  NORM_VALUES = ['one_norm', 'infty_norm']
  APPROACHES_VALUES = ['SAT', 'MO', 'AR']
  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()

  # change norms for plotting
  df_all_distances = df_all_distances.where(df_all_distances['norm'] != 'zero_norm').dropna()

  df_all_distances['norm'] = df_all_distances['norm'].map({
    'zero_norm': r'$\ell_0$',
    'one_norm': r'$\ell_1$',
    'infty_norm': r'$\ell_\infty$',
  })

  df_all_distances['dataset'] = df_all_distances['dataset'].map({
    'adult': 'Adult',
    'credit': 'Credit',
    'compass': 'COMPAS',
  })

  print('Plotting merged distance files.')

  for model_string in MODEL_CLASS_VALUES:

    model_specific_df = df_all_distances.where(df_all_distances['model'] == model_string).dropna()

    if model_string == 'tree' or model_string == 'forest':
      hue_order = ['SAT', 'MO', 'PFT']
    elif model_string == 'lr':
      hue_order = ['SAT', 'MO', 'AR']
    else:
      hue_order = ['SAT', 'MO']

    latexify(1.5 * 6, 6, font_scale = 1.2)
    ax = sns.catplot(
      x = 'dataset',
      y = 'counterfactual distance',
      hue = 'approach',
      hue_order = hue_order,
      col = 'norm',
      data = model_specific_df,
      kind = 'box',
      height = 2.5,
      aspect = 1,
      palette = sns.color_palette("muted", 3),
      sharey = False,
      whis = np.inf,
    )
    ax.set(ylim=(0,None))
    ax.set_axis_labels("", r"Distance $\delta$ to" + "\nNearest Counterfactual")
    ax.set_titles('{col_name}')
    ax.set_xlabels() # remove "dataset" on the x-axis
    ax.savefig(f'_results/distances_{model_string}_main_body.png', dpi = 400)


def plotDistancesAppendix():
  DATASET_VALUES = ['adult', 'credit', 'compass']
  MODEL_CLASS_VALUES = ['tree', 'forest', 'lr', 'mlp']
  NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
  APPROACHES_VALUES = ['SAT', 'MO', 'PFT', 'AR']
  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()

  # change norms for plotting
  # df_all_distances = df_all_distances.where(df_all_distances['norm'] != 'zero_norm').dropna()

  df_all_distances['norm'] = df_all_distances['norm'].map({
    'zero_norm': r'$\ell_0$',
    'one_norm': r'$\ell_1$',
    'infty_norm': r'$\ell_\infty$',
  })

  df_all_distances['dataset'] = df_all_distances['dataset'].map({
    'adult': 'Adult',
    'credit': 'Credit',
    'compass': 'COMPAS',
  })

  print('Plotting merged distance files.')

  for model_string in MODEL_CLASS_VALUES:

    model_specific_df = df_all_distances.where(df_all_distances['model'] == model_string).dropna()

    if model_string == 'tree' or model_string == 'forest':
      hue_order = ['SAT', 'MO', 'PFT']
    elif model_string == 'lr':
      hue_order = ['SAT', 'MO', 'AR']
    else:
      hue_order = ['SAT', 'MO']

    latexify(1.5 * 6, 6, font_scale = 1.2)
    ax = sns.catplot(
      x = 'dataset',
      y = 'counterfactual distance',
      hue = 'approach',
      hue_order = hue_order,
      col = 'norm',
      data = model_specific_df,
      kind = 'box',
      height = 3.5,
      aspect = .9,
      palette = sns.color_palette("muted", 3),
      sharey = False,
      whis = np.inf,
    )
    ax.set(ylim=(0,None))
    ax.set_axis_labels("", r"Distance $\delta$ to" + "\nNearest Counterfactual")
    ax.set_titles('{col_name}')
    ax.set_xlabels() # remove "dataset" on the x-axis
    ax.savefig(f'_results/distances_{model_string}_appendix.png', dpi = 400)


def measureSensitiveAttributeChange():
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()
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
            age_changed = df.where(df['changed age'] == True).dropna()
            age_not_changed = df.where(df['changed age'] == False).dropna()
            gender_changed = df.where(df['changed gender'] == True).dropna()
            gender_not_changed = df.where(df['changed gender'] == False).dropna()
            race_changed = df.where(df['changed race'] == True).dropna()
            race_not_changed = df.where(df['changed race'] == False).dropna()

            sensitive_changed = df.where(
              (df['changed age'] == True) |
              (df['changed gender'] == True) |
              (df['changed race'] == True)
            ).dropna()
            sensitive_not_changed = df.where(
              (df['changed age'] == False) &
              (df['changed gender'] == False) &
              (df['changed race'] == False)
            ).dropna()

            assert df.shape[0] == age_changed.shape[0] + age_not_changed.shape[0]
            assert df.shape[0] == gender_changed.shape[0] + gender_not_changed.shape[0]
            assert df.shape[0] == race_changed.shape[0] + race_not_changed.shape[0]
            assert df.shape[0] == sensitive_changed.shape[0] + sensitive_not_changed.shape[0]

            print(f'{dataset_string}-{model_class_string}-{norm_type_string}-{approach_string}:'.ljust(40), end = '')
            print(f'\t\tpercent age change: % {100 * age_changed.shape[0] / df.shape[0]:.2f}', end = '')
            print(f'\t\tpercent gender change: % {100 * gender_changed.shape[0] / df.shape[0]:.2f}', end = '')
            if dataset_string == 'compass':
              print(f'\t\tpercent race change: % {100 * race_changed.shape[0] / df.shape[0]:.2f}', end = '')
            print(f'\t\tsensitive_changed: {sensitive_changed.shape[0]}, \t sensitive_not_changed: {sensitive_not_changed.shape[0]}, \t percent: % {100 * sensitive_changed.shape[0] / df.shape[0]:.2f}', end = '\n')


def measureEffectOfAgeCompass():
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()
  model_class_string = 'lr'
  dataset_string = 'compass'
  norm_type_string = 'zero_norm'
  for approach_string in ['SAT', 'MO']:

    df = df_all_distances.where(
      (df_all_distances['dataset'] == dataset_string) &
      (df_all_distances['model'] == model_class_string) &
      (df_all_distances['norm'] == norm_type_string) &
      (df_all_distances['approach'] == approach_string),
    ).dropna()
    if df.shape[0]: # if any tests exist for this setup
      age_changed = df.where(df['changed age'] == True).dropna()
      age_not_changed = df.where(df['changed age'] == False).dropna()
      assert df.shape[0] == age_changed.shape[0] + age_not_changed.shape[0]

      print(f'\n\n{dataset_string}-{model_class_string}-{norm_type_string}-{approach_string}:'.ljust(40))

      # print(f'\tage_changed: {age_changed.shape[0]} samples')
      # uniques, counts = np.unique(age_changed["counterfactual distance"], return_counts = True)
      # for idx, elem in enumerate(uniques):
      #   print(f'\t\t{counts[idx]} samples at distance {uniques[idx]}')

      # print(f'\tage_not_changed: {age_not_changed.shape[0]} samples')
      # uniques, counts = np.unique(age_not_changed["counterfactual distance"], return_counts = True)
      # for idx, elem in enumerate(uniques):
      #   print(f'\t\t{counts[idx]} samples at distance {uniques[idx]}')

      print(f'\tage_changed: {age_changed.shape[0]} samples')
      for unique_distance in np.unique(age_changed["counterfactual distance"]):
        unique_distance_df = age_changed.where(age_changed['counterfactual distance'] == unique_distance).dropna()
        print(f'\t\t{unique_distance_df.shape[0]} samples at distance {unique_distance}')
        # unique_attr_changes, counts = np.unique(unique_distance_df["changed attributes"], return_counts = True)
        # for idx, unique_attr_change in enumerate(unique_attr_changes):
        #   print(f'\t\t\t{counts[idx]} samples changing attributes {unique_attr_change}')

      print(f'\tage_not_changed: {age_not_changed.shape[0]} samples')
      for unique_distance in np.unique(age_not_changed["counterfactual distance"]):
        unique_distance_df = age_not_changed.where(age_not_changed['counterfactual distance'] == unique_distance).dropna()
        print(f'\t\t{unique_distance_df.shape[0]} samples at distance {unique_distance}')
        # unique_attr_changes, counts = np.unique(unique_distance_df["changed attributes"], return_counts = True)
        # for idx, unique_attr_change in enumerate(unique_attr_changes):
        #   print(f'\t\t\t{counts[idx]} samples changing attributes {unique_attr_change}')

      print(f'\tpercent: % {100 * age_changed.shape[0] / df.shape[0]:.2f}', end = '\n')


def measureEffectOfRaceCompass():
  pairs = [
    (
      'compass-lr-one_norm-AR',
      '/Users/a6karimi/dev/mace/_experiments/2019.05.23_14.10.06__compass__lr__one_norm__AR__batch0__samples500/_minimum_distances',
      '/Users/a6karimi/dev/mace/_experiments/2019.05.23_14.10.40__compass__lr__one_norm__AR__batch0__samples500/_minimum_distances',
    ), \
    # ('compass-lr-infty_norm-AR', ), \
  ]
  print('\n\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n\n')
  for pair in pairs:
    unrestricted_file = pickle.load(open(pair[1], 'rb'))
    restricted_file = pickle.load(open(pair[2], 'rb'))
    increase_in_distance = []
    unrestricted_distances = []
    restricted_distances = []
    for factual_sample_index in unrestricted_file.keys():

      restricted_factual_sample = restricted_file[factual_sample_index]['factual_sample']
      unrestricted_factual_sample = unrestricted_file[factual_sample_index]['factual_sample']
      restricted_counterfactual_sample = restricted_file[factual_sample_index]['counterfactual_sample']
      unrestricted_counterfactual_sample = unrestricted_file[factual_sample_index]['counterfactual_sample']
      assert restricted_factual_sample == unrestricted_factual_sample

      # if it changes race
      if not np.isclose(unrestricted_factual_sample['x3'], unrestricted_counterfactual_sample['x3']):
        unrestricted_distance = unrestricted_counterfactual_sample['counterfactual_distance']
        restricted_distance = restricted_counterfactual_sample['counterfactual_distance']
        try:
          assert (restricted_distance >= unrestricted_distance) or np.isclose(restricted_distance, unrestricted_distance, 1e-3, 1e-3)
        except:
          print(f'\t Unexpected: \t\t restricted_distance - unrestricted_distance = {restricted_distance - unrestricted_distance}')
        increase_in_distance.append(restricted_distance / unrestricted_distance)
        unrestricted_distances.append(unrestricted_distance)
        restricted_distances.append(restricted_distance)

    print(f'{pair[0]}:')
    print(f'\t\tMean unrestricted distance = {np.mean(unrestricted_distances):.4f}')
    print(f'\t\tMean restricted distance = {np.mean(restricted_distances):.4f}')
    print(f'\t\tMean increase in distance (restricted / unrestricted) = {np.mean(increase_in_distance):.4f}')

      # factual_sample = restricted_factual_sample # or unrestricted_factual_sample
      # restricted_changed_attributes = []
      # unrestricted_changed_attributes = []
      # for attr in factual_sample.keys():
      #   if not np.isclose(factual_sample[attr], restricted_counterfactual_sample[attr]):
      #     restricted_changed_attributes.append((attr, factual_sample[attr], restricted_counterfactual_sample[attr]))
      #   if not np.isclose(factual_sample[attr], unrestricted_counterfactual_sample[attr]):
      #     unrestricted_changed_attributes.append((attr, factual_sample[attr], unrestricted_counterfactual_sample[attr]))

      # # restricted_changed_attributes.pop('y')
      # # unrestricted_changed_attributes.pop('y')

      # print(f'Sample: {factual_sample_index}')
      # print(f'\trestricted_changed_attributes: {restricted_changed_attributes}')
      # print(f'\tunrestricted_changed_attributes: {unrestricted_changed_attributes}')


def measureEffectOfAgeAdultPart1():
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()

  model_class_string = 'forest'
  dataset_string = 'adult'
  for approach_string in ['SAT', 'MO', 'PFT']:
    for norm_type_string in ['zero_norm', 'one_norm', 'infty_norm']:

      df = df_all_distances.where(
        (df_all_distances['dataset'] == dataset_string) &
        (df_all_distances['model'] == model_class_string) &
        (df_all_distances['norm'] == norm_type_string) &
        (df_all_distances['approach'] == approach_string),
      ).dropna()
      if df.shape[0]: # if any tests exist for this setup

        age_constant = df.where(df['age constant'] == True).dropna()
        age_increased = df.where(df['age increased'] == True).dropna()
        age_decreased = df.where(df['age decreased'] == True).dropna()
        assert df.shape[0] == age_constant.shape[0] + age_increased.shape[0] + age_decreased.shape[0]

        test_name = f'{dataset_string}-{model_class_string}-{norm_type_string}-{approach_string}'

        print(f'\n\n{test_name}:'.ljust(40))

        print(f'\tage_increased: {age_increased.shape[0]} samples \t % {100 * age_increased.shape[0] / df.shape[0]} \t average cost: {np.mean(age_increased["counterfactual distance"]):.4f}')
        # for unique_distance in np.unique(age_increased["counterfactual distance"]):
        #   unique_distance_df = age_increased.where(age_increased['counterfactual distance'] == unique_distance).dropna()
        #   print(f'\t\t{unique_distance_df.shape[0]} samples at distance {unique_distance}')
          # unique_attr_changes, counts = np.unique(unique_distance_df["changed attributes"], return_counts = True)
          # for idx, unique_attr_change in enumerate(unique_attr_changes):
          #   print(f'\t\t\t{counts[idx]} samples changing attributes {unique_attr_change}')

        print(f'\tage_constant: {age_constant.shape[0]} samples \t % {100 * age_constant.shape[0] / df.shape[0]} \t average cost: {np.mean(age_constant["counterfactual distance"]):.4f}')
        # for unique_distance in np.unique(age_constant["counterfactual distance"]):
        #   unique_distance_df = age_constant.where(age_constant['counterfactual distance'] == unique_distance).dropna()
        #   print(f'\t\t{unique_distance_df.shape[0]} samples at distance {unique_distance}')
          # unique_attr_changes, counts = np.unique(unique_distance_df["changed attributes"], return_counts = True)
          # for idx, unique_attr_change in enumerate(unique_attr_changes):
          #   print(f'\t\t\t{counts[idx]} samples changing attributes {unique_attr_change}')

        print(f'\tage_decreased: {age_decreased.shape[0]} samples \t % {100 * age_decreased.shape[0] / df.shape[0]} \t average cost: {np.mean(age_decreased["counterfactual distance"]):.4f}')
        # for unique_distance in np.unique(age_decreased["counterfactual distance"]):
        #   unique_distance_df = age_decreased.where(age_decreased['counterfactual distance'] == unique_distance).dropna()
        #   print(f'\t\t{unique_distance_df.shape[0]} samples at distance {unique_distance}')
          # unique_attr_changes, counts = np.unique(unique_distance_df["changed attributes"], return_counts = True)
          # for idx, unique_attr_change in enumerate(unique_attr_changes):
          #   print(f'\t\t\t{counts[idx]} samples changing attributes {unique_attr_change}')

        pickle.dump(age_increased, open(f'_results/{test_name}_age_increased_df', 'wb'))
        pickle.dump(age_decreased, open(f'_results/{test_name}_age_decreased_df', 'wb'))


def measureEffectOfAgeAdultPart2():

  # pairs = [
  #   ('adult-forest-zero_norm-SAT', '_results/adult-forest-zero_norm-SAT_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_21_restricted_results_no_age_reduction/2019.05.21_17.17.39__adult__forest__zero_norm__SAT/_minimum_distances'), \
  #   ('adult-forest-one_norm-SAT', '_results/adult-forest-one_norm-SAT_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_21_restricted_results_no_age_reduction/2019.05.21_17.37.32__adult__forest__one_norm__SAT/_minimum_distances'), \
  #   ('adult-forest-infty_norm-SAT', '_results/adult-forest-infty_norm-SAT_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_21_restricted_results_no_age_reduction/2019.05.21_17.46.25__adult__forest__infty_norm__SAT/_minimum_distances'), \
  #   ('adult-forest-zero_norm-MO', '_results/adult-forest-zero_norm-MO_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_21_restricted_results_no_age_reduction/2019.05.21_17.19.16__adult__forest__zero_norm__MO/_minimum_distances'), \
  #   ('adult-forest-one_norm-MO', '_results/adult-forest-one_norm-MO_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_21_restricted_results_no_age_reduction/2019.05.21_17.37.35__adult__forest__one_norm__MO/_minimum_distances'), \
  #   ('adult-forest-infty_norm-MO', '_results/adult-forest-infty_norm-MO_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_21_restricted_results_no_age_reduction/2019.05.21_17.54.09__adult__forest__infty_norm__MO/_minimum_distances'), \
  # ]
  pairs = [
    ('adult-forest-zero_norm-SAT', '_results/adult-forest-zero_norm-SAT_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_17.42.00__adult__forest__zero_norm__SAT/_minimum_distances'), \
    ('adult-forest-one_norm-SAT', '_results/adult-forest-one_norm-SAT_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_18.17.48__adult__forest__one_norm__SAT/_minimum_distances'), \
    ('adult-forest-infty_norm-SAT', '_results/adult-forest-infty_norm-SAT_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_18.36.35__adult__forest__infty_norm__SAT/_minimum_distances'), \
    ('adult-forest-zero_norm-MO', '_results/adult-forest-zero_norm-MO_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_18.00.02__adult__forest__zero_norm__MO/_minimum_distances'), \
    ('adult-forest-one_norm-MO', '_results/adult-forest-one_norm-MO_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_18.20.50__adult__forest__one_norm__MO/_minimum_distances'), \
    ('adult-forest-infty_norm-MO', '_results/adult-forest-infty_norm-MO_age_decreased_df', '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_18.32.33__adult__forest__infty_norm__MO/_minimum_distances'), \
  ]
  print('\n\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n\n')
  for pair in pairs:
    unrestricted_df = pickle.load(open(pair[1], 'rb'))
    restricted_file = pickle.load(open(pair[2], 'rb'))
    increase_in_distance = []
    unrestricted_distances = []
    restricted_distances = []
    for index, row in unrestricted_df.iterrows():
      factual_sample_index = row.loc['factual sample index']
      unrestricted_distance = row.loc['counterfactual distance']
      restricted_distance = restricted_file[factual_sample_index]['counterfactual_distance']
      try:
        assert (restricted_distance >= unrestricted_distance) or np.isclose(restricted_distance, unrestricted_distance, 1e-3, 1e-3)
      except:
        print(f'\t Unexpected: \t\t restricted_distance - unrestricted_distance = {restricted_distance - unrestricted_distance}')
      increase_in_distance.append(restricted_distance / unrestricted_distance)
      unrestricted_distances.append(unrestricted_distance)
      restricted_distances.append(restricted_distance)
    print(f'{pair[0]}:')
    print(f'\t\tMean unrestricted distance = {np.mean(unrestricted_distances):.4f}')
    print(f'\t\tMean restricted distance = {np.mean(restricted_distances):.4f}')
    print(f'\t\tMean increase in distance (restricted / unrestricted) = {np.mean(increase_in_distance):.4f}')


def measureEffectOfAgeAdultPart3():

  pairs = [(
    'adult-forest-zero_norm-SAT',
    '_results/adult-forest-zero_norm-SAT_age_decreased_df',
    '_results/adult-forest-zero_norm-SAT_age_increased_df',
    '/Volumes/amir/dev/mace/_experiments/_may_20_all_results/2019.05.20_17.44.50__adult__forest__zero_norm__SAT/_minimum_distances',
    '/Volumes/amir/dev/mace/_experiments/_may_22_restricted_results_no_age_change/2019.05.22_17.42.00__adult__forest__zero_norm__SAT/_minimum_distances'
  )]
  print('\n\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n\n')
  for pair in pairs:
    unrestricted_df1 = pickle.load(open(pair[1], 'rb'))
    unrestricted_df2 = pickle.load(open(pair[2], 'rb'))
    unrestricted_df = unrestricted_df1.append(unrestricted_df2, ignore_index=True)
    unrestricted_file = pickle.load(open(pair[3], 'rb'))
    restricted_file = pickle.load(open(pair[4], 'rb'))
    # assert unrestricted_df.shape[0] == 18 # %3.6 x 500
    for index, row in unrestricted_df.iterrows():

      factual_sample_index = row.loc['factual sample index']

      # unrestricted_distance = row.loc['counterfactual distance']
      # restricted_distance = restricted_file[factual_sample_index]['counterfactual_distance']
      # assert unrestricted_distance == restricted_distance

      restricted_factual_sample = restricted_file[factual_sample_index]['factual_sample']
      unrestricted_factual_sample = unrestricted_file[factual_sample_index]['factual_sample']
      restricted_counterfactual_sample = restricted_file[factual_sample_index]['counterfactual_sample']
      unrestricted_counterfactual_sample = unrestricted_file[factual_sample_index]['counterfactual_sample']
      assert restricted_factual_sample == unrestricted_factual_sample

      factual_sample = restricted_factual_sample # or unrestricted_factual_sample
      restricted_changed_attributes = []
      unrestricted_changed_attributes = []
      for attr in factual_sample.keys():
        if not np.isclose(factual_sample[attr], restricted_counterfactual_sample[attr]):
          restricted_changed_attributes.append((attr, factual_sample[attr], restricted_counterfactual_sample[attr]))
        if not np.isclose(factual_sample[attr], unrestricted_counterfactual_sample[attr]):
          unrestricted_changed_attributes.append((attr, factual_sample[attr], unrestricted_counterfactual_sample[attr]))

      # restricted_changed_attributes.pop('y')
      # unrestricted_changed_attributes.pop('y')

      print(f'Sample: {factual_sample_index}')
      print(f'\trestricted_changed_attributes: {restricted_changed_attributes}')
      print(f'\tunrestricted_changed_attributes: {unrestricted_changed_attributes}')


def analyzeDistances():
  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  df_all_distances = df_all_distances.where(
    (df_all_distances['counterfactual found'] == True) &
    (df_all_distances['counterfactual plausible'] == True)
  ).dropna()

  print('Analyzing merged distance files.')

  df = df_all_distances

  # MIN_SAMPLES_REQUIRED = 0

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

          # # SAT works in all scenarios
          # assert \
          #   len(factual_sample_index_per_approach['SAT']) >= MIN_SAMPLES_REQUIRED, \
          #   f'Expecting at least {MIN_SAMPLES_REQUIRED} samples for SAT, got {len(factual_sample_index_per_approach["SAT"])} ({tmp_string})'

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

          # compute 1 - d_SAT / d_{MO, FT, ...}
          all_but_sat_approaches = list(np.setdiff1d(
            np.array(list(factual_sample_index_per_approach.keys())),
            np.array('SAT')
          ))
          factual_sample_index_intersect = []
          for approach_string in all_but_sat_approaches:
            factual_sample_index_intersect = np.intersect1d(
              factual_sample_index_per_approach['SAT'],
              factual_sample_index_per_approach[approach_string]
            )
            assert len(factual_sample_index_intersect) >= MIN_SAMPLES_REQUIRED, f'Expecting at least {MIN_SAMPLES_REQUIRED} intersecting samples between SAT and {approach_string}'
            distance_reduction_list = []
            for factual_sample_index in factual_sample_index_intersect:
              sample_sat = df.where(
                (df['approach'] == 'SAT') &
                (df['factual sample index'] == factual_sample_index)
              ).dropna().T.to_dict()
              assert len(sample_sat.keys()) == 1, f'Expecting only 1 sample with index {factual_sample_index} for approach {approach_string}'
              sample_other = df.where(
                (df['approach'] == approach_string) &
                (df['factual sample index'] == factual_sample_index)
              ).dropna().T.to_dict()
              assert len(sample_other.keys()) == 1, f'Expecting only 1 sample with index {factual_sample_index} for approach {approach_string}'
              minimum_distance_sat = sample_sat[list(sample_sat.keys())[0]]['counterfactual distance']
              minimum_distance_other = sample_other[list(sample_other.keys())[0]]['counterfactual distance']
              distance_reduction_list.append(1 - minimum_distance_sat / minimum_distance_other)
            tmp_mean = np.mean(np.array(distance_reduction_list)) * 100
            tmp_std = np.std(np.array(distance_reduction_list)) * 100
            print(f'\t Distance reduction for {dataset_string} {model_class_string} {norm_type_string} (1 - d_SAT / d_{approach_string}) = \t {tmp_mean:.2f} +/- {tmp_std:.2f} \t (N = {len(distance_reduction_list)})')


def analyzeAverageDistanceRunTimeCoverage():
  DATASET_VALUES = ['adult', 'credit', 'compass']
  MODEL_CLASS_VALUES = ['tree', 'forest', 'lr', 'mlp']
  NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
  APPROACHES_VALUES = ['SAT', 'MO', 'PFT', 'AR']
  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  # DO NOT INCLUDE THE LINES BELOW!!!!!!!!!!!!!!!!!!!! WHY???
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
            average_distance = found_and_plausible['counterfactual distance'].mean() # this is NOT a good way to compare methods! see analyzeDistances() instead, as it compares ratio of distances for the same samples!
            std_distance = found_and_plausible['counterfactual distance'].std()
            average_run_time = found_and_plausible['counterfactual time'].mean()
            std_run_time = found_and_plausible['counterfactual time'].std()
            coverage = count_found_and_plausible / df.shape[0] * 100
            print(f'{model_class_string}-{approach_string}-{dataset_string}-{norm_type_string} ({count_found_and_plausible} plausible samples found):')
            print(f'\tAvg distance: {average_distance:.2f} +/- {std_distance:.2f}')
            print(f'\tAvg run-time: {average_run_time:.2f} +/- {std_run_time:.2f} seconds')
            print(f'\tCoverage: %{coverage}')


def plotDistanceTimeTradeofAgainstIterations():

  DATASET_VALUES = ['adult', 'credit', 'compass']
  MODEL_CLASS_VALUES = ['tree', 'forest', 'lr'] # MLP
  NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
  APPROACHES_VALUES = ['SAT']
  # Remove FeatureTweaking / ActionableRecourse distances that were unsuccessful or non-plausible
  df_all_distances = pickle.load(open(f'_results/df_all_distances', 'rb'))
  # DO NOT INCLUDE THE LINES BELOW!!!!!!!!!!!!!!!!!!!! WHY???
  # df_all_distances = df_all_distances.where(
  #   (df_all_distances['counterfactual found'] == True) &
  #   (df_all_distances['counterfactual plausible'] == True)
  # ).dropna()
  # for model_class_string in MODEL_CLASS_VALUES:
  #   for approach_string in APPROACHES_VALUES:
  #     for dataset_string in DATASET_VALUES:
  #       for norm_type_string in NORM_VALUES:
  #         df = df_all_distances.where(
  #           (df_all_distances['dataset'] == dataset_string) &
  #           (df_all_distances['model'] == model_class_string) &
  #           (df_all_distances['norm'] == norm_type_string) &
  #           (df_all_distances['approach'] == approach_string),
  #         ).dropna()
  #         # ipsh()
  #         if df.shape[0]: # if any tests exist for this setup
  #           # max_iterations = max(list(map(lambda x : len(x), df_all_distances['all counterfactual times'])))
  #           # for elem in df_all_distances['all counterfactual times'].count()
  #           tmp_df = pd.DataFrame({ \
  #             'factual_sample_index': [], \
  #             'iteration': [], \
  #             'distance': [], \
  #             'time': [], \
  #           })
  #           for index, row in df.iterrows():
  #             all_counterfactual_distances = row['all counterfactual distances'][1:] # remove the first elem (np.infty)
  #             all_counterfactual_times = row['all counterfactual times'][1:] # remove the first elem (np.infty)
  #             cum_counterfactual_times = np.cumsum(all_counterfactual_times)
  #             assert len(all_counterfactual_distances) == len(all_counterfactual_times)
  #             for iteration_counter in range(len(all_counterfactual_distances)):
  #               tmp_df = tmp_df.append({
  #                 'factual_sample_index': row['factual sample index'],
  #                 'iteration': int(iteration_counter),
  #                 'distance': all_counterfactual_distances[iteration_counter],
  #                 # 'time': all_counterfactual_times[iteration_counter],
  #                 'time': cum_counterfactual_times[iteration_counter],
  #               }, ignore_index =  True)
  #           fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)
  #           sns.lineplot(x="iteration", y="time", data=tmp_df, ax=ax1)
  #           sns.lineplot(x="iteration", y="distance", data=tmp_df, ax=ax2)
  #           # fig.set_title('model_class_string')
  #           fig.savefig(f'_results/distance_vs_time___{model_class_string}_{dataset_string}_{norm_type_string}.png', dpi = 400)
  #           # ax.get_figure().savefig(f'_results/time_{model_class_string}.png', dpi = 400)
  #           # ax.clf()
  #           # ax.get_figure().savefig(f'_results/distance_vs_time___{model_class_string}.png', dpi = 400)
  #           # ax.clf()


  # for model_class_string in MODEL_CLASS_VALUES:
  #   for approach_string in APPROACHES_VALUES:
  #     for dataset_string in DATASET_VALUES:

  #       tmp_df = pd.DataFrame({ \
  #         'factual_sample_index': [], \
  #         'dataset': [], \
  #         'model': [], \
  #         'norm': [], \
  #         'approach': [], \
  #         'iteration': [], \
  #         'distance': [], \
  #         'time': [], \
  #       })

  #       for norm_type_string in NORM_VALUES:
  #         df = df_all_distances.where(
  #           (df_all_distances['dataset'] == dataset_string) &
  #           (df_all_distances['model'] == model_class_string) &
  #           (df_all_distances['norm'] == norm_type_string) &
  #           (df_all_distances['approach'] == approach_string),
  #         ).dropna()

  #         if df.shape[0]: # if any tests exist for this setup
  #           # max_iterations = max(list(map(lambda x : len(x), df_all_distances['all counterfactual times'])))
  #           # for elem in df_all_distances['all counterfactual times'].count()

  #           for index, row in df.iterrows():
  #             all_counterfactual_distances = row['all counterfactual distances'][1:] # remove the first elem (np.infty)
  #             all_counterfactual_times = row['all counterfactual times'][1:] # remove the first elem (np.infty)
  #             cum_counterfactual_times = np.cumsum(all_counterfactual_times)
  #             assert len(all_counterfactual_distances) == len(all_counterfactual_times)
  #             for iteration_counter in range(len(all_counterfactual_distances)):
  #               tmp_df = tmp_df.append({
  #                 'factual_sample_index': row['factual sample index'],
  #                 'dataset': dataset_string,
  #                 'model': model_class_string,
  #                 'norm': norm_type_string,
  #                 'approach': approach_string,
  #                 'iteration': int(iteration_counter),
  #                 'distance': all_counterfactual_distances[iteration_counter],
  #                 # 'time': all_counterfactual_times[iteration_counter],
  #                 'time': cum_counterfactual_times[iteration_counter],
  #               }, ignore_index =  True)
  #       fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)
  #       sns.lineplot(x="iteration", y="time", hue='norm', data=tmp_df, ax=ax1)
  #       sns.lineplot(x="iteration", y="distance", hue='norm', data=tmp_df, ax=ax2)
  #       # fig.savefig(f'_results/distance_vs_time___{model_class_string}_{dataset_string}_{norm_type_string}.png', dpi = 400)
  #       fig.savefig(f'_results/distance_vs_time___{model_class_string}_{dataset_string}.png', dpi = 400)


  for model_class_string in MODEL_CLASS_VALUES:
    for approach_string in APPROACHES_VALUES:

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

      for dataset_string in DATASET_VALUES:

        # for norm_type_string in NORM_VALUES:
        norm_type_string = 'one_norm'
        df = df_all_distances.where(
          (df_all_distances['dataset'] == dataset_string) &
          (df_all_distances['model'] == model_class_string) &
          (df_all_distances['norm'] == norm_type_string) &
          (df_all_distances['approach'] == approach_string),
        ).dropna()

        if df.shape[0]: # if any tests exist for this setup
          # max_iterations = max(list(map(lambda x : len(x), df_all_distances['all counterfactual times'])))
          # for elem in df_all_distances['all counterfactual times'].count()

          for index, row in df.iterrows():
            all_counterfactual_distances = row['all counterfactual distances'][1:] # remove the first elem (np.infty)
            all_counterfactual_times = row['all counterfactual times'][1:] # remove the first elem (np.infty)
            cum_counterfactual_times = np.cumsum(all_counterfactual_times)
            assert len(all_counterfactual_distances) == len(all_counterfactual_times)
            for iteration_counter in range(len(all_counterfactual_distances)):
              tmp_df = tmp_df.append({
                'factual_sample_index': row['factual sample index'],
                'dataset': dataset_string,
                'model': model_class_string,
                'norm': norm_type_string,
                'approach': approach_string,
                'iteration': int(iteration_counter),
                'distance': all_counterfactual_distances[iteration_counter],
                # 'time': all_counterfactual_times[iteration_counter],
                'time': cum_counterfactual_times[iteration_counter],
              }, ignore_index =  True)
      fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)
      sns.lineplot(x="iteration", y="time", hue='dataset', data=tmp_df, ax=ax1)
      sns.lineplot(x="iteration", y="distance", hue='dataset', data=tmp_df, ax=ax2)
      ax1.set(ylim=(0, 60))
      ax2.set(ylim=(0, 1))
      # fig.savefig(f'_results/distance_vs_time___{model_class_string}_{dataset_string}_{norm_type_string}.png', dpi = 400)
      fig.savefig(f'_results/distance_vs_time___{model_class_string}.png', dpi = 400)



if __name__ == '__main__':
  # gatherAndSaveDistances()
  # measureEffectOfRaceCompass()
  # measureSensitiveAttributeChange()
  # DEPRECATED # measureEffectOfAgeCompass()
  # measureEffectOfAgeAdultPart1()
  # measureEffectOfAgeAdultPart2()
  # measureEffectOfAgeAdultPart3()
  # analyzeAverageDistanceRunTimeCoverage()
  # plotDistancesMainBody()
  # plotDistancesAppendix()
  # DONE # analyzeDistances()
  plotDistanceTimeTradeofAgainstIterations()

































