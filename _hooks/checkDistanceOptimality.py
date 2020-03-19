import os
import glob
import pickle
import argparse
from tqdm import tqdm
from pprint import pprint
from shutil import copyfile

# (approximate) optimal distances for all the combination of the following setups
# has been saved in the optimal_distances pickle file
DATASET_VALUES = ['compass', 'credit', 'adult']
MODEL_CLASS_VALUES = ['tree', 'forest', 'lr', 'mlp']
NORM_VALUES = ['zero_norm', 'one_norm', 'infty_norm']
APPROACHES_VALUES = ['MACE_eps_1e-3']

experiments_folder_path = '_experiments/'

def get_all_new_minimum_distances(experiments_start_time):
    # create a dictionary with regards to the setup
    all_new_minimum_distances = {}
    for dataset_string in DATASET_VALUES:
        all_new_minimum_distances[dataset_string] = {}
        for model_class_string in MODEL_CLASS_VALUES:
            all_new_minimum_distances[dataset_string][model_class_string] = {}
            for norm_type_string in NORM_VALUES:
                all_new_minimum_distances[dataset_string][model_class_string][norm_type_string] = {}
                for approach_string in APPROACHES_VALUES:
                    all_new_minimum_distances[dataset_string][model_class_string][norm_type_string][approach_string] = {}

    # take the minimum distances from each experiment folder created after experiments_start_time
    for experiment_folder in glob.glob(f'{experiments_folder_path}*'):

        experiment_details = experiment_folder.split('__')

        if experiment_details[0].split('/')[-1] >= experiments_start_time:
            dataset_string = experiment_details[1]
            model_class_string = experiment_details[2]
            norm_type_string = experiment_details[3]
            approach_string = experiment_details[4]

            minimum_distances_path = os.path.join(experiment_folder, '_minimum_distances')

            try:
                assert os.path.isfile(minimum_distances_path)
                minimum_distances = pickle.load(open(minimum_distances_path, 'rb'))
                all_new_minimum_distances[dataset_string][model_class_string][norm_type_string][approach_string] = minimum_distances
            except:
                pass

    return all_new_minimum_distances


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-t', '--experiments_start_time',
        nargs = '+',
        type = str,
        default = '',
        help = 'The start date and time of the experiments to be cheked')

    args = parser.parse_args()
    experiments_start_time = args.experiments_start_time[0]

    # # read the previously-saved (approx) optimal distances
    all_old_minimum_distances = pickle.load(open('_hooks/old_minimum_distances', 'rb'))

    # get minimum distances computed by running the hooks (i.e. experiments after experiments_start_time)
    all_new_minimum_distances = get_all_new_minimum_distances(experiments_start_time)

    # check whether the computed minimum_distances are close enough to the optimal distances (+- epsilon)
    for dataset_string in DATASET_VALUES:

        for model_class_string in MODEL_CLASS_VALUES:

            for norm_type_string in NORM_VALUES:

                for approach_string in APPROACHES_VALUES:

                    cur_old_minimum_distances = all_old_minimum_distances[dataset_string][model_class_string][norm_type_string][approach_string]
                    cur_new_minimum_distances = all_new_minimum_distances[dataset_string][model_class_string][norm_type_string][approach_string]

                    if bool(cur_new_minimum_distances):

                        if approach_string == 'MACE_eps_1e-3':

                            assert bool(cur_old_minimum_distances)
                            for sample_idx in cur_new_minimum_distances.keys():
                                
                                if sample_idx in cur_old_minimum_distances.keys():
                                    if not abs(cur_new_minimum_distances[sample_idx]['cfe_distance'] - cur_old_minimum_distances[sample_idx]['cfe_distance']) <= 2e-3:
                                        print('\033[91m' + f'Error (distance to optimal > 2eps): {dataset_string}, {model_class_string}, {norm_type_string}, {approach_string}' + '\033[0m')
                                        exit(1)
                                else:
                                    raise Exception("No previously-saved optimal distance for sample " + sample_idx)
                        else:
                            raise Exception("No previously-saved optimal distance for this approach string")
                    else:
                        print('\033[93m' + f'no minimum distance (probably due to time-limit exceed): {dataset_string}, {model_class_string}, {norm_type_string}, {approach_string}' + '\033[0m')
                        continue
                    
                    print('\033[92m' + f'OK (distance to optimal <= 2eps): {dataset_string}, {model_class_string}, {norm_type_string}, {approach_string}' + '\033[0m')


    # The following line is for the one-time writing of optimal distances from experiments to the optimal_distances binary file
    # pickle.dump(all_new_minimum_distances, open(f'_hooks/optimal_distances', 'wb'))
