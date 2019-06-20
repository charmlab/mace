import os
import graphviz
import numpy as np
from sklearn.tree import _tree, export_graphviz


# Source:
# https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
# https://web.archive.org/web/20171005203850/http://www.kdnuggets.com/2017/05/simplifying-decision-tree-interpretation-decision-rules-python.html
# Note-to-self: merging leaves with the same classification decision doesn't seem to make sense; see final figure here: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
def simplifyDecisionTree(tree, debug_flag):
    tree_ = tree.tree_

    # while there exists a node whose left & right children
    #   1. are leaves
    #   2. share the same classification results
    # merge together into parent, making their parent a leaf
    parent_idx, left_child_idx, right_child_idx, children_class = mergeConditionHoldsTrue(tree_)

    if debug_flag > 0:
        print('[INFO] Simplifying decision tree...')

    while parent_idx > -1:

        if debug_flag > 1:
            print('\t[INFO] Merging children at index {} (left) and {} (right) into index {} (parent)...'.format(left_child_idx, right_child_idx, parent_idx), end='')

        tree_.feature[parent_idx] = -2
        tree_.threshold[parent_idx] = -2
        tree_.children_left[parent_idx] = -1
        tree_.children_right[parent_idx] = -1

        # Note, unfortunately we cannot update the attributes of tree_ (even if
        # we were to deep-copy it), and so all pruned nodes will remain. Instead
        # we will introduce another value (-3) to represent pruned nodes.
        tree_.feature[left_child_idx] = -3
        tree_.feature[right_child_idx] = -3
        tree_.threshold[left_child_idx] = -3
        tree_.threshold[right_child_idx] = -3
        tree_.children_left[left_child_idx] = -3
        tree_.children_left[right_child_idx] = -3
        tree_.children_right[left_child_idx] = -3
        tree_.children_right[right_child_idx] = -3

        parent_idx, left_child_idx, right_child_idx, children_class = mergeConditionHoldsTrue(tree_)

        if debug_flag > 1:
            print('done.')

    if debug_flag > 0:
        print('[INFO] done.\n')
    return tree_


def mergeConditionHoldsTrue(tree_):
    # Find pairs of leaves
    # Assert they share a parent
    # Assert they share a classification result
    for elem in getParentLeftRightTuples(tree_):
        parent_idx = elem[0]
        left_child_idx = elem[1]
        right_child_idx = elem[2]
        left_child_class = list(tree_.value[left_child_idx][0]).index(max(list(tree_.value[left_child_idx][0])))
        right_child_class = list(tree_.value[right_child_idx][0]).index(max(list(tree_.value[right_child_idx][0])))
        if left_child_class == right_child_class:
            children_class = left_child_class
            return parent_idx, left_child_idx, right_child_idx, children_class
    return -1, -1, -1, -1


def getParentLeftRightTuples(tree_):
    features_list = list(tree_.feature)
    tuples_list = []
    # Because we cannot delete pruned nodes, we must check for sequences
    # of [-2, m x -3, -2] within the main list, where m >= 0
    for i in range(len(features_list)):
        if features_list[i] == -2:
            for j in range(i+1, len(features_list)):
                if features_list[j] == -2:
                    parent_idx = i - 1
                    left_child_idx = i
                    right_child_idx = j
                    # print((parent_idx, left_child_idx, right_child_idx))
                    if isValidParentIdx(tree_, parent_idx, left_child_idx, right_child_idx):
                        tuples_list.append((parent_idx, left_child_idx, right_child_idx))
                    break
                elif features_list[j] == -3:
                    continue
                else:
                    break
    return tuples_list


def getAllSubIdx(x, y):
    sub_indices = []
    l1, l2 = len(x), len(y)
    for i in range(l1):
        if x[i:i+l2] == y:
            sub_indices.append(i)
    return sub_indices


def isValidParentIdx(tree_, parent_idx, left_child_idx, right_child_idx):
    return \
        tree_.children_left[parent_idx] == left_child_idx and \
        tree_.children_right[parent_idx] == right_child_idx


def saveTreeVisualization(model, model_class, sub_model_name, X_test, feature_names, save_folder_name):
    save_path = f'{save_folder_name}/{model_class}_{sub_model_name}_{X_test.shape[1]}_features'
    dot_data = export_graphviz(model, out_file=None,
        feature_names=feature_names,
        class_names=['0','1'],
        filled=True,
        rounded=True,
        special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(save_path)
    os.remove(save_path) # two files are outputted, one is extra
