import os
import sys
import copy
import time
import pickle
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot

import torch
import argparse
import numpy as np
from pprint import pprint
from datetime import datetime

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

import loadData
import loadModel
import normalizedDistance

# from torch.utils.tensorboard import SummaryWriter

import dice_ml
from dice_ml.utils import helpers # helper functions

import tensorflow as tf
from tensorflow import keras


def getPrediction(sklearn_model, instance):
  prediction = sklearn_model.predict(np.array(list(instance.values())).reshape(1,-1))[0]
  # assert prediction in {-1, 1}, f'Expected prediction in {-1,1}; got {prediction}'
  assert prediction in {0, 1}, f'Expected prediction in {0,1}; got {prediction}'
  return prediction


def didFlip(sklearn_model, factual_instance, counter_instance):
  return getPrediction(sklearn_model, factual_instance) != getPrediction(sklearn_model, counter_instance)


def scatterPairOfInstances(sklearn_model, factual_instance, counter_instance, ax):
    assert len(factual_instance.keys()) == len(counter_instance.keys())
    if len(factual_instance.keys()) == 2:
        ax.scatter(
            factual_instance['x0'],
            factual_instance['x1'],
            marker='P',
            color='black',
            s=70)
        ax.scatter(
            counter_instance['x0'],
            counter_instance['x1'],
            marker='o',
            color='green' if didFlip(sklearn_model, factual_instance, counter_instance) else 'red',
            s=70)
        pyplot.plot(
            [factual_instance['x0'], counter_instance['x0']],
            [factual_instance['x1'], counter_instance['x1']],
            'b--'
        )
    elif len(factual_instance.keys()) == 3:
        ax.scatter(
            factual_instance['x1'],
            factual_instance['x2'],
            factual_instance['x3'],
            marker='P',
            color='black',
            s=70)
        ax.scatter(
            counter_instance['x1'],
            counter_instance['x2'],
            counter_instance['x3'],
            marker='o',
            color='green' if didFlip(sklearn_model, factual_instance, counter_instance) else 'red',
            s=70)
        pyplot.plot(
            [factual_instance['x1'], counter_instance['x1']],
            [factual_instance['x2'], counter_instance['x2']],
            [factual_instance['x3'], counter_instance['x3']],
            'b--'
        )


def scatterDataset(X_train, X_test, y_train, y_test, ax):
    X_train_numpy = X_train.to_numpy()
    X_test_numpy = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    number_of_samples_to_plot = min(200, X_train_numpy.shape[0], X_test_numpy.shape[0])
    for idx in range(number_of_samples_to_plot):
        color_train = 'black' if y_train[idx] == 1 else 'magenta'
        color_test = 'black' if y_test[idx] == 1 else 'magenta'
        if X_train.shape[1] == 2:
            ax.scatter(X_train_numpy[idx, 0], X_train_numpy[idx, 1], marker='s', color=color_train, alpha=0.2, s=10)
            ax.scatter(X_test_numpy[idx, 0], X_test_numpy[idx, 1], marker='o', color=color_test, alpha=0.2, s=15)
        elif X_train.shape[1] == 3:
            ax.scatter(X_train_numpy[idx, 0], X_train_numpy[idx, 1], X_train_numpy[idx, 2], marker='s', color=color_train, alpha=0.2, s=10)
            ax.scatter(X_test_numpy[idx, 0], X_test_numpy[idx, 1], X_test_numpy[idx, 2], marker='o', color=color_test, alpha=0.2, s=15)


def scatterDecisionBoundary(DATASET, MODEL_CLASS, sklearn_model, ax):
    if DATASET == 'random':
        fixed_model_w = sklearn_model.coef_
        fixed_model_b = sklearn_model.intercept_

        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        X = np.linspace(ax.get_xlim()[0] - x_range / 10, ax.get_xlim()[1] + x_range / 10, 10)
        Y = np.linspace(ax.get_ylim()[0] - y_range / 10, ax.get_ylim()[1] + y_range / 10, 10)
        X, Y = np.meshgrid(X, Y)
        Z = - (fixed_model_w[0][0] * X + fixed_model_w[0][1] * Y + fixed_model_b) / fixed_model_w[0][2]

        surf = ax.plot_wireframe(X, Y, Z, alpha=0.3)
    elif DATASET in {'mortgage', 'twomoon'}:
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        X = np.linspace(ax.get_xlim()[0] - x_range / 10, ax.get_xlim()[1] + x_range / 10, 1000)
        Y = np.linspace(ax.get_ylim()[0] - y_range / 10, ax.get_ylim()[1] + y_range / 10, 1000)
        X, Y = np.meshgrid(X, Y)

        labels = sklearn_model.predict(np.c_[X.ravel(), Y.ravel()])
        Z = labels.reshape(X.shape)

        cmap = pyplot.get_cmap('Paired')
        ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5)

def printAndVisualizeTestStatistics(
    setup_name,
    x_test_factual,
    x_test_counter,
    PLOTTING_FLAG,
    DATASET,
    MODEL_CLASS,
    sklearn_model,
    experiment_folder_name,
    X_train_df,
    X_test_df,
    y_train_df,
    y_test_df,
):

    if PLOTTING_FLAG:
        pyplot.clf()
        if DATASET == 'random':
            ax = pyplot.subplot(1, 1, 1, projection = '3d')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('x3')
            ax.view_init(elev=10, azim=-20)
        elif DATASET in {'mortgage', 'twomoon'}:
            ax = pyplot.subplot()
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.grid()

        for test_sample_idx in x_test_factual.keys():
            single_x_test_factual = x_test_factual[test_sample_idx]
            single_x_test_counter = x_test_counter[test_sample_idx]
            scatterPairOfInstances(
                sklearn_model,
                single_x_test_factual,
                single_x_test_counter,
                ax
            )

        scatterDataset(X_train_df, X_test_df, y_train_df, y_test_df, ax)
        scatterDecisionBoundary(DATASET, MODEL_CLASS, sklearn_model, ax)
        pyplot.title( \
            f'{setup_name}\n'\
        , fontsize=8)
        pyplot.draw()
        # pyplot.pause(0.0001)
        # pyplot.pause(0.05)
        # pyplot.show()
        pyplot.savefig(f'{experiment_folder_name}/plot.png')

def normalize(data_point, data_range):
    new_data_point = {}
    for key in data_point.keys():
        if key != 'y' and not('cat' in key):
            new_data_point[key] = (data_point[key] - data_range[key][0]) / (data_range[key][1] - data_range[key][0])
        else:
            new_data_point[key] = data_point[key]
    return new_data_point

def de_normalize(data_point, data_range):
    new_data_point = {}
    for key in data_point.keys():
        if key != 'y' and not('cat' in key):
            new_data_point[key] = (data_point[key] * (data_range[key][1] - data_range[key][0])) + data_range[key][0]
        else:
            new_data_point[key] = data_point[key]
    return new_data_point

def cat_to_dice(data_point):
    new_data_point = {}
    for key in data_point.keys():
        if not('cat' in key):
            new_data_point[key] = data_point[key]
        elif data_point[key] == 1.0:
            new_data_point[key.replace('_'+key.split('_')[-1], '')] = key
    return new_data_point

def cat_from_dice(data_point, dataset_obj):
    new_data_point = {}
    for key in data_point.keys():
        if not('cat' in key):
            new_data_point[key] = data_point[key]
        else:
            cat_choice = key + '_' + data_point[key]
            for sibling in dataset_obj.getSiblingsFor(cat_choice):
                if sibling == cat_choice:
                    new_data_point[sibling] = 1.0
                else:
                    new_data_point[sibling] = 0.0
    return new_data_point

def generateDiCEExplanations(APPROACH, DATASET, MODEL_CLASS, LEARNING_RATE, PROXIMITY_WEIGHT, PROCESS_ID, GEN_CF_FOR, SAMPLES):

    setup_name = f'{DATASET}__{MODEL_CLASS}__one_norm__{APPROACH}__lr{LEARNING_RATE}__pr{PROXIMITY_WEIGHT}'
    experiment_name = f'{setup_name}__pid{PROCESS_ID}'
    experiment_folder_name = f"_experiments/{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}__{experiment_name}"
    os.mkdir(experiment_folder_name)
    # writer = SummaryWriter(log_dir=experiment_folder_name)

    ################################################################################
    # Dataset processer + loader
    ################################################################################
    print(f'[INFO] Loading the `{DATASET}` dataset and training sklearn model...\t', end = '')
    dataset_obj = loadData.loadDataset(DATASET, return_one_hot = True, load_from_cache = False, debug_flag = False)

    # Get bounds now because after normalization all bounds will turn into [0,1]
    dice_dataset_features = {}
    already_considered = []
    for attr_name_kurz in dataset_obj.getInputAttributeNames():
        attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
        if attr_obj.attr_type == 'numeric-int' or attr_obj.attr_type == 'binary':
            dice_dataset_features[attr_name_kurz] = [int(attr_obj.lower_bound), int(attr_obj.upper_bound)]
        elif attr_obj.attr_type == 'sub-categorical' and attr_name_kurz in already_considered:
            continue
        elif attr_obj.attr_type == 'sub-categorical':
            dice_dataset_features[attr_name_kurz.replace('_'+attr_name_kurz.split('_')[-1], '')] = dataset_obj.getSiblingsFor(attr_name_kurz)
            already_considered += dataset_obj.getSiblingsFor(attr_name_kurz)
        else:
            raise Exception(f"Attribute type {attr_obj.attr_type} not supported by DiCE")

    # normalized data for the models
    X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = \
        dataset_obj.getTrainTestSplit(preprocessing='normalize')

    data_dim = X_train_normalized.shape[1]

    sklearn_model = loadModel.loadModelForDataset(MODEL_CLASS, DATASET, experiment_folder_name, preprocessing='normalize')

    # get the predicted labels (only test set)
    X_test_pred_labels = sklearn_model.predict(X_test_normalized)

    all_pred_data_df_normalized = X_test_normalized

    # IMPORTANT: note that 'y' is actually 'pred_y', not 'true_y'
    all_pred_data_df_normalized['y'] = X_test_pred_labels
    neg_pred_data_df_normalized = all_pred_data_df_normalized.where(all_pred_data_df_normalized['y'] == 0).dropna()
    pos_pred_data_df_normalized = all_pred_data_df_normalized.where(all_pred_data_df_normalized['y'] == 1).dropna()

    # generate counterfactuals for {only negative, negative & positive} samples
    if GEN_CF_FOR == 'neg_only':
        iterate_over_data_df_normalized = neg_pred_data_df_normalized[0: SAMPLES]  # choose only a subset to compare
        observable_data_df_normalized = pos_pred_data_df_normalized
    elif GEN_CF_FOR == 'pos_only':
        iterate_over_data_df_normalized = pos_pred_data_df_normalized[0: SAMPLES]  # choose only a subset to compare
        observable_data_df_normalized = neg_pred_data_df_normalized
    elif GEN_CF_FOR == 'neg_and_pos':
        iterate_over_data_df_normalized = all_pred_data_df_normalized[0: SAMPLES]  # choose only a subset to compare
        observable_data_df_normalized = all_pred_data_df_normalized
    else:
        raise Exception(f'{gen_cf_for} not recognized as a valid `gen_cf_for`.')

    # convert to dictionary for easier enumeration (iteration)
    iterate_over_data_dict_normalized = iterate_over_data_df_normalized.T.to_dict()
    observable_data_dict_normalized = observable_data_df_normalized.T.to_dict()

    print(f'done.')


    ################################################################################
    # Load fixed_model < f >
    ################################################################################
    print(f'[INFO] Loading the `{DATASET}` fixed predictor...\t', end = '')

    if MODEL_CLASS == 'mlp2x10':
        fixed_model_width = 10 # TODO make more dynamic later and move to separate function
        assert sklearn_model.hidden_layer_sizes == (fixed_model_width, fixed_model_width)

        # Keras model
        sess = tf.InteractiveSession()

        fixed_model = keras.Sequential()
        fixed_model.add(keras.layers.Dense(fixed_model_width, input_shape=(data_dim,), activation=tf.nn.relu))
        fixed_model.add(keras.layers.Dense(fixed_model_width, input_shape=(fixed_model_width,), activation=tf.nn.relu))
        fixed_model.add(keras.layers.Dense(1, input_shape=(fixed_model_width,), activation=tf.nn.sigmoid))

        fixed_model.layers[0].set_weights([sklearn_model.coefs_[0].astype('float64'), sklearn_model.intercepts_[0].astype('float64')])
        fixed_model.layers[1].set_weights([sklearn_model.coefs_[1].astype('float64'), sklearn_model.intercepts_[1].astype('float64')])
        fixed_model.layers[2].set_weights([sklearn_model.coefs_[2].astype('float64'), sklearn_model.intercepts_[2].astype('float64')])

        fixed_model.layers[0].trainable = False
        fixed_model.layers[1].trainable = False
        fixed_model.layers[2].trainable = False
    else:
        raise Exception(f"{MODEL_CLASS} not a recognized model class.")

    print(f'done.')

    ################################################################################
    # Preparing for DiCE
    ################################################################################

    dice_dataset = dice_ml.data.Data(features=dice_dataset_features, outcome_name='y')
    dice_model = dice_ml.model.Model(model=fixed_model, backend='TF1')

    # DiCE explanation instance
    exp = dice_ml.Dice(dice_dataset, dice_model)

    # iterate over samples for which we desire a counterfactual,
    # (to be saved as part of the same file of minimum distances)
    explanation_counter = 1
    all_minimum_distances, all_fcs, all_cfs = {}, {}, {}
    tot_dist_flipped, tot_flipped = 0, 0
    for factual_sample_index, factual_sample_normalized in iterate_over_data_dict_normalized.items():

        print('-'*80 + ' factual sample #' + str(factual_sample_index))
        init_label = factual_sample_normalized['y']
        del factual_sample_normalized['y']

        sklearn_prediction = int(sklearn_model.predict([list(factual_sample_normalized.values())])[0])
        fixed_model_output = tf.keras.backend.get_value(fixed_model(tf.convert_to_tensor([list(factual_sample_normalized.values())])))[0][0]

        assert bool(sklearn_prediction) == init_label
        assert bool(sklearn_prediction) == (fixed_model_output >= 0.5)


        if len(dataset_obj.getOneHotAttributesNames()) > 0:
            dice_sample = cat_to_dice(de_normalize(factual_sample_normalized, dice_dataset_features))
        else:
            dice_sample = de_normalize(factual_sample_normalized, dice_dataset_features)

        # Generate DiCE explanations
        start_time = time.time()
        dice_exp = exp.generate_counterfactuals(dice_sample,
                                                total_CFs=1, desired_class="opposite",
                                                algorithm="DiverseCF", proximity_weight=PROXIMITY_WEIGHT, diversity_weight=0.0,
                                                learning_rate=LEARNING_RATE)
        end_time = time.time()

        # Print counterfactual explanation
        if len(dice_exp.final_cfs_list) > 0:
            # dice_exp.visualize_as_list()
            cfe_label = (dice_exp.final_cfs_list[0][-1] >= 0.5)

            dice_counterfactual_sample_normalized = normalize(dice_exp.final_cfs_df.T.to_dict()[0], dice_dataset_features)
            if len(dataset_obj.getOneHotAttributesNames()) > 0:
                counterfactual_sample_normalized = cat_from_dice(dice_counterfactual_sample_normalized, dataset_obj)
            else:
                counterfactual_sample_normalized = dice_counterfactual_sample_normalized

            fixed_model_output_on_cfe = tf.keras.backend.get_value(
                fixed_model(tf.convert_to_tensor([list(counterfactual_sample_normalized.values())[:-1]])))[0][0]
            assert abs(fixed_model_output_on_cfe - dice_exp.final_cfs_list[0][
                -1] < 1e-2), "DiCE model prob output does not match fixed model output"

            factual_sample_normalized['y'] = init_label
            distance = normalizedDistance.getDistanceBetweenSamples(
                factual_sample_normalized,
                counterfactual_sample_normalized,
                'two_norm',
                dataset_obj  # is normalized thus [0, 1] bounds
            )

        else:
            raise Exception("CFE list is empty!")

        if cfe_label != init_label:
            tot_dist_flipped += distance
            tot_flipped += 1
            all_minimum_distances[f'sample_{factual_sample_index}'] = {
                'fac_sample': de_normalize(factual_sample_normalized, dice_dataset_features),
                'cfe_sample': de_normalize(counterfactual_sample_normalized, dice_dataset_features),
                'cfe_found': True,
                'cfe_plausible': True,
                'cfe_distance': distance,
                'cfe_time': end_time - start_time,
            }
            del counterfactual_sample_normalized['y']
            del factual_sample_normalized['y']
            all_cfs[factual_sample_index] = counterfactual_sample_normalized
            all_fcs[factual_sample_index] = factual_sample_normalized
        else:
            all_minimum_distances[f'sample_{factual_sample_index}'] = {
                'fac_sample': de_normalize(factual_sample_normalized, dice_dataset_features),
                'cfe_sample': de_normalize(counterfactual_sample_normalized, dice_dataset_features),
                'cfe_found': False,
                'cfe_plausible': False,
                'cfe_distance': np.infty,
                'cfe_time': end_time - start_time,
            }
            print("prediction proba: ", dice_exp.final_cfs_list[0][-1])
            print("CFE not found.")

    if len(dataset_obj.getOneHotAttributesNames()) == 0:
        printAndVisualizeTestStatistics( experiment_name, all_fcs, all_cfs, True, DATASET, MODEL_CLASS,
        sklearn_model, experiment_folder_name, X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized)

    pickle.dump(all_minimum_distances, open(f'{experiment_folder_name}/_minimum_distances', 'wb'))
    pprint(all_minimum_distances, open(f'{experiment_folder_name}/minimum_distances.txt', 'w'))

    avg_dist = tot_dist_flipped / tot_flipped
    avg_flip = tot_flipped / SAMPLES * 100
    # writer.add_scalars(f'metrics/avg_flip', {'avg_flip': avg_flip}, 0)
    # writer.add_scalars(f'metrics/avg_flip', {'avg_flip': avg_flip}, 1)
    # writer.add_scalars(f'metrics/avg_dist', {'avg_dist': avg_dist}, 0)
    # writer.add_scalars(f'metrics/avg_dist', {'avg_dist': avg_dist}, 1)
    # writer.flush()
    # writer.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-a', '--approach',
        type = str,
        default = 'dice',
        help = 'Approach used to generate counterfactual: dice')

    parser.add_argument(
        '-d', '--dataset',
        type = str,
        default = 'random',
        help = 'Name of dataset to train explanation model for')

    parser.add_argument(
        '-m', '--model_class',
        type = str,
        default = 'mlp',
        help = 'Model class that will learn data: mlp2x10')

    parser.add_argument(
        '-l', '--learning_rate',
        type = float,
        default = 0.05,
        help = 'learning_rate')

    parser.add_argument(
        '-w', '--proximity_weight',
        type=float,
        default=0.5,
        help='proximity weight for DiCE')

    parser.add_argument(
        '-s', '--samples',
        type=int,
        default=25,
        help='number of factual samples')

    parser.add_argument(
        '-g', '--gen_cf_for',
        type=str,
        default='neg_only',
        help='Decide whether to generate counterfactuals for negative pred samples only, or for both negative and positive pred samples.')

    parser.add_argument(
        '-p', '--process_id',
        type = str,
        default = '0',
        help = 'When running parallel tests on the cluster, process_id guarantees (in addition to time stamped experiment folder) that experiments do not conflict.')

    # parsing the args
    args = parser.parse_args()

    generateDiCEExplanations(args.approach,
                             args.dataset,
                             args.model_class,
                             args.learning_rate,
                             args.proximity_weight,
                             args.process_id,
                             args.gen_cf_for,
                             args.samples)
