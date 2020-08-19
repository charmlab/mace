import pickle
import glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

all_dists = []

N_SAMPLES = 500

def getDesiredKeyVals(desired_key, path, orders=None):
    all = pickle.load(open(path, 'rb'))
    desired = [i for i in range(len(orders))]
    for i, sample_idx in enumerate(orders.keys()):
        if sample_idx in all:
            desired[orders[sample_idx]] = all[sample_idx][desired_key]
        else:
            print(f"Sample not found for: {path}")
            # desired[orders[sample_idx]] = 1000
            # print("No key for this order!")

    all_dists.append(desired)
    return desired

def getPlottingOrder(path, desired_key, path2=None):
    all = pickle.load(open(path, 'rb'))
    if path2 is not None:
        all2 = pickle.load(open(path2, 'rb'))

    desired = []
    for sample_idx in all.keys():
        if path2 is not None:
            if sample_idx in all2:
                desired.append((all[sample_idx][desired_key], sample_idx))
        elif all[sample_idx]['cfe_found'] is True:
            desired.append((all[sample_idx][desired_key], sample_idx))
        if len(desired) == N_SAMPLES:
            break

    desired = sorted(desired)

    orders = {}
    for i in range(len(desired)):
        orders[desired[i][1]] = i

    return orders

def plotScatterDesiredKey(ax, label, path, orders, desired_key):
    desired = getDesiredKeyVals(desired_key, path, orders)
    ax.scatter(np.arange(len(desired)), desired)
    ax.plot(np.arange(len(desired)), desired, label=label)
    ax.set_xticks(np.arange(len(desired)))
    ax.set_xticklabels([list(orders.keys())[list(orders.values()).index(i)].split('_')[-1] for i in range(len(desired))], rotation=65)
    ax.legend()

if __name__ == "__main__":

    DATASET_VALUES = ['compass', 'credit', 'adult']
    MODEL_CLASS_VALUES = ['mlp2x10']
    NORM_VALUES = ['one_norm']
    APPROACHES_VALUES = ['MACE_MIP_OBJ_eps_1e-3', 'dice']

    KEY = 'cfe_distance'
    experiments_path = './_experiments/'

    for norm in NORM_VALUES:

        fig, axs = plt.subplots(len(DATASET_VALUES), len(MODEL_CLASS_VALUES), figsize=(20, 11))
        path = ''

        for i, dataset in enumerate(DATASET_VALUES):
            for j, model_type in enumerate(MODEL_CLASS_VALUES):
                if len(DATASET_VALUES) == 1 and len(MODEL_CLASS_VALUES) == 1:
                    ax = axs
                elif len(DATASET_VALUES) == 1:
                    ax = axs[j]
                elif len(MODEL_CLASS_VALUES) == 1:
                    ax = axs[i]
                else:
                    ax = axs[i, j]
                ax.grid()
                if 'time' in KEY:
                    ax.set_yscale("log")
                ax.set_ylabel(f"{model_type}")
                if j == 0:
                    ax.set_title(f"{dataset} dataset")
                paths = glob.glob(f'{experiments_path}/*{dataset}__{model_type}__{norm}__*/_minimum_distances')
                order_path = paths[0] if 'dice' in paths[0] else paths[1]
                orders = getPlottingOrder(order_path, KEY)
                ax.set_xlabel(f"{KEY.split('_')[-1]} on {len(orders)} samples")
                # for norm in NORM_VALUES:
                for approach in APPROACHES_VALUES:
                    path = glob.glob(f'{experiments_path}/*{dataset}__{model_type}__{norm}__{approach}*/_minimum_distances')
                    if len(path) == 0:
                        continue
                    assert len(path) == 1
                    path = path[0]
                    label = approach.replace('_eps_1e-3', '')
                    label = label.replace('MACE_', '')
                    plotScatterDesiredKey(ax, label, path, orders, KEY)


        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        # plt.show()

        plt.savefig(f'_results/{KEY}_{norm}.png', bboc_inches='tight', pad_inches=0)