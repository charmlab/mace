import os
import pickle
import numpy as np
import pandas as pd

from pprint import pprint



parent_folder = '/Users/a6karimi/dev/mace/_may_17_test_02_all_mlp_results_final'

child_folders = [
  '2019.05.15_12.56.09__compass__mlp__zero_norm__SAT',
  '2019.05.15_13.00.18__compass__mlp__zero_norm__MO',
  '2019.05.15_13.01.57__compass__mlp__one_norm__SAT',
  '2019.05.15_13.02.56__compass__mlp__one_norm__MO',
  '2019.05.15_13.04.41__compass__mlp__infty_norm__SAT',
  '2019.05.15_13.05.09__compass__mlp__infty_norm__MO',
  '2019.05.15_13.05.25__credit__mlp__zero_norm__SAT',
  '2019.05.15_13.05.37__credit__mlp__zero_norm__MO',
  '2019.05.15_13.06.07__credit__mlp__one_norm__SAT',
  '2019.05.15_13.06.20__credit__mlp__one_norm__MO',
  '2019.05.15_13.06.29__credit__mlp__infty_norm__SAT',
  '2019.05.15_13.06.40__credit__mlp__infty_norm__MO',
  '2019.05.15_13.07.08__adult__mlp__zero_norm__SAT',
  '2019.05.15_13.07.19__adult__mlp__zero_norm__MO',
  '2019.05.15_14.21.12__adult__mlp__one_norm__SAT',
  '2019.05.15_14.28.34__adult__mlp__one_norm__MO',
  '2019.05.15_14.31.15__adult__mlp__infty_norm__SAT',
  '2019.05.15_14.40.41__adult__mlp__infty_norm__MO',
]

for idx, child_folder in enumerate(child_folders):

  print(f'[{idx} / {len(child_folders)}] Converting results for {child_folder}...')

  minimum_distances_file = os.path.join(parent_folder, child_folder, '_minimum_distances')
  minimum_distances = pickle.load(open(minimum_distances_file, 'rb'))

  for key in minimum_distances.keys():
    tmp = minimum_distances[key]
    tmp['counterfactual_distance'] = tmp.pop('distance')
    tmp['counterfactual_found'] = True
    tmp['counterfactual_plausible'] = True
    minimum_distances[key] = tmp

  pickle.dump(minimum_distances, open(minimum_distances_file, 'wb'))



