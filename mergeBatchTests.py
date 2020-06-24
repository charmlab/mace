import os
import glob
import pickle
from tqdm import tqdm
from pprint import pprint
from shutil import copyfile

from debug import ipsh

# DATASET_VALUES = ['adult', 'credit', 'compass']
# MODEL_CLASS_VALUES = ['tree', 'forest', 'lr'] # MLP
# # MODEL_CLASS_VALUES = ['mlp']
# NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
# # APPROACHES_VALUES = ['MACE_eps_1e-3', 'MACE_eps_1e-5']
# APPROACHES_VALUES = ['MACE_eps_1e-3']

# DATASET_VALUES = ['adult', 'credit', 'compass']
# MODEL_CLASS_VALUES = ['tree', 'forest', 'lr'] # , 'mlp']
# NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
# APPROACHES_VALUES = ['MACE_eps_1e-5'] # , 'MACE_eps_1e-3', 'MACE_eps_1e-5']
# # APPROACHES_VALUES = ['MACE_eps_1e-3']

# DATASET_VALUES = ['german']
# MODEL_CLASS_VALUES = ['lr']
# NORM_VALUES = ['one_norm']
# APPROACHES_VALUES = ['MACE_eps_1e-3']

# DATASET_VALUES = ['credit']
# MODEL_CLASS_VALUES = ['mlp']
# NORM_VALUES = ['one_norm']
# APPROACHES_VALUES = ['MACE_eps_1e-3']

# DATASET_VALUES = ['random', 'mortgage']
# MODEL_CLASS_VALUES = ['lr']
# NORM_VALUES = ['one_norm']
# APPROACHES_VALUES = ['MACE_eps_1e-3']

# DATASET_VALUES = ['random', 'mortgage', 'twomoon', 'credit']
# MODEL_CLASS_VALUES = ['lr', 'mlp']
# NORM_VALUES = ['two_norm']
# # APPROACHES_VALUES = ['MINT_eps_1e-3']
# APPROACHES_VALUES = ['MACE_eps_1e-3']

# DATASET_VALUES = ['twomoon'] #, 'twomoon', 'credit']
# MODEL_CLASS_VALUES = ['lr', 'mlp']
# NORM_VALUES = ['two_norm']
# # APPROACHES_VALUES = ['MACE_eps_1e-3']
# APPROACHES_VALUES = ['MACE_eps_1e-3', 'MINT_eps_1e-3']

DATASET_VALUES = ['twomoon'] #, 'twomoon', 'credit']
MODEL_CLASS_VALUES = ['lr', 'tree', 'mlp']
NORM_VALUES = ['two_norm']
APPROACHES_VALUES = ['MACE_eps_1e-3']
# APPROACHES_VALUES = ['MACE_eps_1e-5', 'MINT_eps_1e-5']



experiments_folder_path = '/Volumes/amir/dev/mace/_experiments/'
# experiments_folder_path = '/Users/a6karimi/dev/mace/_experiments/'
all_counter = len(DATASET_VALUES) * len(MODEL_CLASS_VALUES) * len(NORM_VALUES) * len(APPROACHES_VALUES)
counter = 0

for dataset_string in DATASET_VALUES:

  for model_class_string in MODEL_CLASS_VALUES:

    for norm_type_string in NORM_VALUES:

      for approach_string in APPROACHES_VALUES:

        counter = counter + 1

        specific_experiment_path = f'{dataset_string}__{model_class_string}__{norm_type_string}__{approach_string}'
        # specific_experiment_path = 'adult__mlp__zero_norm__MACE_eps_1e-5'

        print(f'\n[{counter} / {all_counter}] Merging together folders for {specific_experiment_path}')

        all_batch_folders = glob.glob(f'{experiments_folder_path}*{specific_experiment_path}*')
        sorted_all_batch_folders = sorted(all_batch_folders, key = lambda x : x.split('__')[-1])
        all_minimum_distances = {}
        folders_not_found = []
        for batch_folder in tqdm(sorted_all_batch_folders):
          batch_number_string = batch_folder.split('__')[-2]
          batch_minimum_distances_path = os.path.join(batch_folder, '_minimum_distances')
          try:
            assert os.path.isfile(batch_minimum_distances_path)
            batch_minimum_distances = pickle.load(open(batch_minimum_distances_path, 'rb'))
            all_minimum_distances = {**all_minimum_distances, **batch_minimum_distances}
            # print(f'\tAdding minimum distance file for {batch_number_string}')
            # print(len(all_minimum_distances.keys()))
          except:
            folders_not_found.append(batch_number_string)

        if len(folders_not_found):
          print(f'\tCannot find minimum distance file for {folders_not_found}')

        # create new folder
        random_batch_folder = sorted_all_batch_folders[0]
        new_folder_name = random_batch_folder.split('/')[-1]
        new_folder_name = '__'.join(new_folder_name.split('__')[:-2])
        # new_folder_path = os.path.join(experiments_folder_path, '__merged_MACE_eps_1e-5', new_folder_name)
        new_folder_path = os.path.join(experiments_folder_path, '__merged', new_folder_name)

        print(f'Creating new merged folder {new_folder_path}')
        os.makedirs(new_folder_path, exist_ok = False)
        files_to_copy = {'_dataset_obj', '_model_trained', 'log_experiment.txt', 'log_training.txt'}
        for file_name in files_to_copy:
          copyfile(
            os.path.join(random_batch_folder, file_name),
            os.path.join(new_folder_path, file_name)
          )

        pickle.dump(all_minimum_distances, open(f'{new_folder_path}/_minimum_distances', 'wb'))
        pprint(all_minimum_distances, open(f'{new_folder_path}/minimum_distances.txt', 'w'))

