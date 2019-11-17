def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn # to ignore all warnings.

import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import argparse

# TODO: change to be like _data_main below, and make python module
# this answer https://stackoverflow.com/a/50474562 and others
import treeUtils
import modelConversion

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

SIMPLIFY_TREES = False


################################################################################
##                                                     Printing-related Methods
################################################################################

# def assertMatchingPredictionsOnTestData(model, model_class, X_test):
#     list_predictions = []
#     for a in range(20):
#         sample = X_test.iloc[a].tolist()
#         if model_class == 'tree':
#             print(model.predict_proba([sample]))
#             # list_predictions.append(abs(int(model.predict([sample])[0]) - predict_tree(*sample)))
#         elif model_class == 'forest':
#             list_predictions.append(abs(int(model.predict([sample])[0]) - predict_forest(*sample)))
#         elif model_class == 'lr':
#             print(model.predict_proba([sample]))
#             # list_predictions.append(abs(int(model.predict([sample])[0]) - predict_lr(sample)))
#         elif model_class == 'mlp':
#             list_predictions.append(abs(int(model.predict([sample])[0]) - predict_mlp(model, sample)))

#     assert(not any(list_predictions))

def trainAndSaveModels(experiment_folder_name, model_class, X_train, X_test, y_train, y_test, feature_names):

    log_file = open(f'{experiment_folder_name}/log_training.txt','w')

    if model_class == 'tree':
        # model_pretrain = DecisionTreeClassifier(min_samples_leaf = min(100, round(0.1 * X_train.shape[0])))
        model_pretrain = DecisionTreeClassifier()
    elif model_class == 'forest':
        # model_pretrain = RandomForestClassifier(min_samples_leaf = min(100, round(0.1 * X_train.shape[0])))
        model_pretrain = RandomForestClassifier()
    elif model_class == 'lr':
        model_pretrain = LogisticRegression()
    elif model_class == 'mlp':
        model_pretrain = MLPClassifier(hidden_layer_sizes = (10, 10))
        # model_pretrain = MLPClassifier() # = hidden_layer_sizes = (100, 100)
        # model_pretrain = MLPClassifier(hidden_layer_sizes = (100, 100))

    print('[INFO] Training `{}` on {:,} samples (%{:.2f} of {:,} samples)...'.format(model_class, X_train.shape[0], 100 * X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]), X_train.shape[0] + X_test.shape[0]), file=log_file)
    model_trained = model_pretrain.fit(X_train, y_train)
    print('\tTraining accuracy: %{:.2f}'.format(accuracy_score(y_train, model_trained.predict(X_train)) * 100), file=log_file)
    print('\tTesting accuracy: %{:.2f}'.format(accuracy_score(y_test, model_trained.predict(X_test)) * 100), file=log_file)
    print('[INFO] done.\n', file=log_file)

    if model_class == 'lr':
        w = np.array([[ 0.22397889, 0.7445909, -0.33426894]]).T # Hardcoded because haven't cached this value from experimentSetup.py!
        assert w.T.shape == model_trained.coef_.shape, f'Expecting equal size weight vector for new experiments no lr model.'
        model_trained.coef_ = w.T
        model_trained.intercept_ = np.zeros(1)

    if model_class == 'tree':
        tmp = 1
        # exec(modelConversion.tree2py(model_trained, feature_names))
        if SIMPLIFY_TREES:
            print('[INFO] Simplifying decision tree...', end = '', file=log_file)
            model_trained.tree_ = treeUtils.simplifyDecisionTree(model_trained, False)
            print('\tdone.', file=log_file)
        treeUtils.saveTreeVisualization(model_trained, model_class, '', X_test, feature_names, experiment_folder_name)
    elif model_class == 'forest':
        tmp = 1
        # exec(modelConversion.forest2py(model_trained, feature_names))
        for tree_idx in range(len(model_trained.estimators_)):
            if SIMPLIFY_TREES:
                print(f'[INFO] Simplifying decision tree (#{tree_idx + 1}/{len(model_trained.estimators_)})...', end = '', file=log_file)
                model_trained.estimators_[tree_idx].tree_ = treeUtils.simplifyDecisionTree(model_trained.estimators_[tree_idx], False)
                print('\tdone.', file=log_file)
            treeUtils.saveTreeVisualization(model_trained.estimators_[tree_idx], model_class, f'tree{tree_idx}', X_test, feature_names, experiment_folder_name)
    elif model_class == 'lr':
        tmp = 1
        # exec(modelConversion.lr2py(model_trained, feature_names))
    elif model_class == 'mlp':
        tmp = 1
        # exec(modelConversion.mlp2py(model_trained))

    pickle.dump(model_trained, open(f'{experiment_folder_name}/_model_trained', 'wb'))
    return model_trained

    # TODO: the line below will not run outside of if __name__ == '__main__'
    # assertMatchingPredictionsOnTestData(model_trained, model_class, X_test)
    # print('[INFO] Perfect prediction match (sklearn vs py-generated) for `{}`? \t [{}]\n'.format(model_class, not any(list_predictions)), file=log_file)


