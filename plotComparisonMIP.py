import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

all_dists = []

N_SAMPLES = 10

def getDesiredKeyVals(desired_key, path, orders=None):
    all = pickle.load(open(path, 'rb'))
    desired = [i for i in range(len(orders))]
    for i, sample_idx in enumerate(orders.keys()):
        curr = all[sample_idx][desired_key]
        if 'model-free' in path and desired_key == 'cfe_time':
            print('\033[93m' + f'handling model-free cfe_time by hand...' + '\033[0m')
            curr /= len(all)

        if orders is None:
            desired[i] = curr
        else:
            desired[orders[sample_idx]] = curr

    all_dists.append(desired)
    return desired

def getPlottingOrder(path, desired_key):
    all = pickle.load(open(path, 'rb'))

    desired = []
    for sample_idx in all.keys():
        if all[sample_idx]['cfe_found'] is True:
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
    mace_path = '_experiments/LP_one-norm/MACE__twomoon__mlp2x10__one_norm__MACE_eps_1e-5__batch0__samples10__pid0/_minimum_distances'
    oneSideDist_path = '_experiments/LP_one-norm/oneSideDist__twomoon__mlp2x10__one_norm__MACE_eps_1e-5__batch0__samples10__pid0/_minimum_distances'
    twoSideDist_path = '_experiments/LP_one-norm/twoSideDist__twomoon__mlp2x10__one_norm__MACE_eps_1e-5__batch0__samples10__pid0/_minimum_distances'
    keepDistLP_path = '_experiments/LP_one-norm/keepDistLP__twomoon__mlp2x10__one_norm__MACE_eps_1e-5__batch0__samples10__pid0/_minimum_distances'

    N_SAMPLES = 10

    fig = plt.figure(figsize=(16.0, 8.0))

    key = 'cfe_distance'
    ax1 = plt.subplot(1, 3, 1)
    runtime_orders = getPlottingOrder(keepDistLP_path, key)
    plotScatterDesiredKey(ax1, "MACE", mace_path, runtime_orders, key)
    plotScatterDesiredKey(ax1, "oneSideDist", oneSideDist_path, runtime_orders, key)
    plotScatterDesiredKey(ax1, "twoSideDist", twoSideDist_path, runtime_orders, key)
    plotScatterDesiredKey(ax1, "keepDistLP", keepDistLP_path, runtime_orders, key)
    ax1.set_xlabel("Time to find nearest counterfactual on 10 samples")
    ax1.set_ylabel("Time in seconds")
    ax1.set_title("MLP 2x10")
    ax1.set_yscale("log")
    # ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax1.set_yticks(np.arange(0, 1000, 50))
    ax1.grid()

    mace_path = mace_path.replace("mlp2x10", "mlp3x10")
    oneSideDist_path = oneSideDist_path.replace("mlp2x10", "mlp3x10")
    twoSideDist_path = twoSideDist_path.replace("mlp2x10", "mlp3x10")
    keepDistLP_path = keepDistLP_path.replace("mlp2x10", "mlp3x10")

    ax2 = plt.subplot(1, 3, 2)
    runtime_orders = getPlottingOrder(keepDistLP_path, key)
    plotScatterDesiredKey(ax2, "MACE", mace_path, runtime_orders, key)
    plotScatterDesiredKey(ax2, "oneSideDist", oneSideDist_path, runtime_orders, key)
    plotScatterDesiredKey(ax2, "twoSideDist", twoSideDist_path, runtime_orders, key)
    plotScatterDesiredKey(ax2, "keepDistLP", keepDistLP_path, runtime_orders, key)
    ax2.set_xlabel("Time to find nearest counterfactual on 10 samples")
    ax2.set_ylabel("Time in seconds")
    ax2.set_title("MLP 3x10")
    # ax2.set_yticks(np.arange(0, 1000, 50))
    ax2.set_yscale("log")
    ax2.grid()

    mace_path = mace_path.replace("mlp3x10", "mlp4x10")
    oneSideDist_path = oneSideDist_path.replace("mlp3x10", "mlp4x10")
    twoSideDist_path = twoSideDist_path.replace("mlp3x10", "mlp4x10")
    keepDistLP_path = keepDistLP_path.replace("mlp3x10", "mlp4x10")

    ax3 = plt.subplot(1, 3, 3)
    runtime_orders = getPlottingOrder(keepDistLP_path, key)
    plotScatterDesiredKey(ax3, "MACE", mace_path, runtime_orders, key)
    plotScatterDesiredKey(ax3, "oneSideDist", oneSideDist_path, runtime_orders, key)
    plotScatterDesiredKey(ax3, "twoSideDist", twoSideDist_path, runtime_orders, key)
    plotScatterDesiredKey(ax3, "keepDistLP", keepDistLP_path, runtime_orders, key)
    ax3.set_xlabel("Time to find nearest counterfactual on 10 samples")
    ax3.set_ylabel("Time in seconds")
    ax3.set_title("MLP 4x10")
    # ax2.set_yticks(np.arange(0, 1000, 50))
    ax3.set_yscale("log")
    ax3.grid()


    a = np.array(all_dists)
    print(abs(a[0]-a[1]))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()

    # plt.savefig('comp.png', bboc_inches='tight', pad_inches=0)