import time
import copy
import numpy as np
import pandas as pd
import scipy.stats
import normalizedDistance

from pprint import pprint
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def search_path(estimator, class_labels, counterfactual_label):
    """
    return path index list containing [{leaf node id, inequality symbol, threshold, feature index}].
    estimator: decision tree
    maxj: the number of selected leaf nodes
    """
    """ select leaf nodes whose outcome is counterfactual_label """
    children_left = estimator.tree_.children_left  # information of left child node
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    # leaf nodes ID
    leaf_nodes = np.where(children_left == -1)[0]
    # outcomes of leaf nodes
    leaf_values = estimator.tree_.value[leaf_nodes].reshape(len(leaf_nodes), len(class_labels))
    # select the leaf nodes whose outcome is counterfactual_label
    # (select index of leaf node in tree, not in the previous leaf_node array!)
    leaf_nodes = leaf_nodes[np.where(leaf_values[:, counterfactual_label] != 0)[0]]
    """ search the path to the selected leaf node """
    paths = {}
    for leaf_node in leaf_nodes:
        """ correspond leaf node to left and right parents """
        child_node = leaf_node
        parent_node = -100  # initialize
        parents_left = []
        parents_right = []
        while (parent_node != 0):
            if (np.where(children_left == child_node)[0].shape == (0, )):
                parent_left = -1
                parent_right = np.where(children_right == child_node)[0][0]
                parent_node = parent_right
            elif (np.where(children_right == child_node)[0].shape == (0, )):
                parent_right = -1
                parent_left = np.where(children_left == child_node)[0][0]
                parent_node = parent_left
            parents_left.append(parent_left)
            parents_right.append(parent_right)
            """ for next step """
            child_node = parent_node
        # nodes dictionary containing left parents and right parents
        paths[leaf_node] = (parents_left, parents_right)

    path_info = {}
    for i in paths:
        node_ids = []  # node ids used in the current node
        # inequality symbols used in the current node
        inequality_symbols = []
        thresholds = []  # thretholds used in the current node
        features = []  # features used in the current node
        parents_left, parents_right = paths[i]
        for idx in range(len(parents_left)):
            if (parents_left[idx] != -1):
                """ the child node is the left child of the parent """
                node_id = parents_left[idx]  # node id
                node_ids.append(node_id)
                inequality_symbols.append(0)
                thresholds.append(threshold[node_id])
                features.append(feature[node_id])
            elif (parents_right[idx] != -1):
                """ the child node is the right child of the parent """
                node_id = parents_right[idx]
                node_ids.append(node_id)
                inequality_symbols.append(1)
                thresholds.append(threshold[node_id])
                features.append(feature[node_id])
            path_info[i] = {'node_id': node_ids,
                            'inequality_symbol': inequality_symbols,
                            'threshold': thresholds,
                            'feature': features}
    return path_info

def esatisfactory_instance(x, epsilon, path_info, standard_deviations):
    """
    return the epsilon satisfactory instance of x.
    """
    esatisfactory = copy.deepcopy(x)
    for i in range(len(path_info['feature'])):
        # feature index
        feature_idx = path_info['feature'][i]
        # threshold used in the current node
        threshold_value = path_info['threshold'][i]
        # inequality symbol
        inequality_symbol = path_info['inequality_symbol'][i]
        if inequality_symbol == 0:
            # esatisfactory[feature_idx] = threshold_value - epsilon
            esatisfactory[feature_idx] = threshold_value - epsilon * standard_deviations[feature_idx]
        elif inequality_symbol == 1:
            # esatisfactory[feature_idx] = threshold_value + epsilon
            esatisfactory[feature_idx] = threshold_value + epsilon * standard_deviations[feature_idx]
        else:
            print('something wrong')
    return esatisfactory

def genExp(model_trained, factual_sample, class_labels, epsilon, norm_type, dataset_obj, standard_deviations, perform_while_plausibility):
    """
    This function return the active feature tweaking vector.
    x: feature vector
    class_labels: list containing the all class labels
    counterfactual_label: the label which we want to transform the label of x to
    """

    """ initialize """
    start_time = time.time()
    x = factual_sample.copy()
    del x['y']
    x = np.array(list(x.values()))
    # factual_sample['y'] = False
    counterfactual_label = not factual_sample['y']

    # initialize output in case no solution is found
    closest_counterfactual_sample = dict(zip(factual_sample.keys(), [-1 for elem in factual_sample.values()]))
    closest_counterfactual_sample['y'] = counterfactual_label
    counterfactual_found = False
    closest_distance = 1000  # initialize cost (if no solution is found, this is returned)

    # We want to support forest and tree, and we keep in mind that a trained
    # tree does NOT perform the same as a trained forest with 1 tree!
    if isinstance(model_trained, DecisionTreeClassifier):
        # ensemble_classifier will in fact not be an ensemble, but only be a tree
        estimator = model_trained
        if estimator.predict(x.reshape(1, -1)) != counterfactual_label:
            paths_info = search_path(estimator, class_labels, counterfactual_label)
            for key in paths_info:
                """ generate epsilon-satisfactory instance """
                path_info = paths_info[key]
                es_instance = esatisfactory_instance(x, epsilon, path_info, standard_deviations)

                if perform_while_plausibility:
                    # make plausible by rounding all non-numeric-real attributes to
                    # nearest value in range
                    for idx, elem in enumerate(es_instance):
                        attr_name_kurz = dataset_obj.getInputAttributeNames('kurz')[idx]
                        attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
                        if attr_obj.attr_type != 'numeric-real':
                            # round() might give a value that is NOT in plausible.
                            # instead find the nearest plausible value
                            es_instance[idx] = min(
                                list(range(int(attr_obj.lower_bound), int(attr_obj.upper_bound) + 1)),
                                key = lambda x : abs(x - es_instance[idx])
                            )

                if estimator.predict(es_instance.reshape(1, -1)) == counterfactual_label:
                    counterfactual_sample = dict(zip(factual_sample.keys(), es_instance))
                    counterfactual_sample['y'] = counterfactual_label
                    distance = normalizedDistance.getDistanceBetweenSamples(
                        factual_sample,
                        counterfactual_sample,
                        norm_type,
                        dataset_obj
                    )
                    if distance < closest_distance:
                        closest_counterfactual_sample = counterfactual_sample
                        counterfactual_found = True
                        closest_distance = distance

    elif isinstance(model_trained, RandomForestClassifier):
        ensemble_classifier = model_trained
        for estimator in ensemble_classifier:
            if (ensemble_classifier.predict(x.reshape(1, -1)) == estimator.predict(x.reshape(1, -1))
                and estimator.predict(x.reshape(1, -1) != counterfactual_label)):
                paths_info = search_path(estimator, class_labels, counterfactual_label)
                for key in paths_info:
                    """ generate epsilon-satisfactory instance """
                    path_info = paths_info[key]
                    es_instance = esatisfactory_instance(x, epsilon, path_info, standard_deviations)

                    if perform_while_plausibility:
                        # make plausible by rounding all non-numeric-real attributes to
                        # nearest value in range
                        for idx, elem in enumerate(es_instance):
                            attr_name_kurz = dataset_obj.getInputAttributeNames('kurz')[idx]
                            attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
                            if attr_obj.attr_type != 'numeric-real':
                                # round() might give a value that is NOT in plausible.
                                # instead find the nearest plausible value
                                es_instance[idx] = min(
                                    list(range(int(attr_obj.lower_bound), int(attr_obj.upper_bound) + 1)),
                                    key = lambda x : abs(x - es_instance[idx])
                                )

                    if ensemble_classifier.predict(es_instance.reshape(1, -1)) == counterfactual_label:
                        counterfactual_sample = dict(zip(factual_sample.keys(), es_instance))
                        counterfactual_sample['y'] = counterfactual_label
                        distance = normalizedDistance.getDistanceBetweenSamples(
                            factual_sample,
                            counterfactual_sample,
                            norm_type,
                            dataset_obj
                        )
                        if distance < closest_distance:
                            closest_counterfactual_sample = counterfactual_sample
                            counterfactual_found = True
                            closest_distance = distance

    # better naming
    counterfactual_sample = closest_counterfactual_sample
    distance = closest_distance

    # Perform plausibility check on the nearest counterfactual found
    counterfactual_plausible = True
    for attr_name_kurz in dataset_obj.getInputOutputAttributeNames('kurz'):
        attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
        if attr_obj.attr_type != 'numeric-real':
            try:
                assert np.isclose(
                    counterfactual_sample[attr_name_kurz],
                    np.round(counterfactual_sample[attr_name_kurz])
                ), 'not satisfying plausibility (data-type)'
                counterfactual_sample[attr_name_kurz] = np.round(counterfactual_sample[attr_name_kurz])
                assert counterfactual_sample[attr_name_kurz] >= attr_obj.lower_bound, 'not satisfying plausibility (data-range)'
                assert counterfactual_sample[attr_name_kurz] <= attr_obj.upper_bound, 'not satisfying plausibility (data-range)'
            except:
                counterfactual_plausible = False
                # distance = 1000
                # return factual_sample, counterfactual_sample, distance

    end_time = time.time()

    return {
        'factual_sample': factual_sample,
        'cfe_sample': counterfactual_sample,
        'cfe_found': counterfactual_found,
        'cfe_plausible': counterfactual_plausible,
        'cfe_distance': distance,
        'cfe_time': end_time - start_time,
    }





