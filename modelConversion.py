import graphviz
import numpy as np
from sklearn.tree import _tree, export_graphviz
from pysmt.shortcuts import *
from pysmt.typing import *


# # Hoare triple examples:
#     # https://www.cs.cmu.edu/~aldrich/courses/654-sp07/slides/7-hoare.pdf
#     # https://cs.stackexchange.com/questions/86936/finding-weakest-precondition

################################################################################
##                                                         Tree-related Methods
################################################################################

def tree2py(tree, feature_names, return_value = 'class_idx_max', tree_idx = ''):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
        for i in tree_.feature
    ]
    lines = list()
    lines.append('def predict_tree{}({}):'.format(tree_idx, ', '.join(feature_names)))
    #
    def recurse(node, depth):
        indent = '\t' * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            lines.append('{}if {} <= {}:'.format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            lines.append('{}else:'.format(indent))
            # lines.append('{}else: # if {} > {}'.format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            if return_value == 'class_idx_max':
                values = list(tree_.value[node][0])
                output = values.index(max(values))
                lines.append('{}output = {}'.format(indent, output))
            elif return_value == 'class_prob_array':
                prob_array = list(np.divide(tree_.value[node][0], np.sum(tree_.value[node][0])))
                lines.append('{}output = {}'.format(indent, prob_array))
    #
    recurse(0, 1)
    lines.append('')
    lines.append('\treturn output;')
    lines.append('')
    return '\n'.join(lines)


def tree2c(tree, feature_names, return_value = 'class_idx_max', tree_idx = ''):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
        for i in tree_.feature
    ]
    lines = list()
    lines.append('proc predict_tree{}({} : int) : int = {{'.format(tree_idx, ' : int, '.join(feature_names)))
    lines.append('')
    lines.append('\tvar output : int;')
    lines.append('')
    #
    def recurse(node, depth):
        indent = '\t' * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            lines.append('{}if ( {} <= {} ) then {{ '.format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            lines.append('{}}} else {{ '.format(indent))
            recurse(tree_.children_right[node], depth + 1)
            lines.append('{}}}'.format(indent, name, threshold))
        else:
            if return_value == 'class_idx_max':
                values = list(tree_.value[node][0])
                output = values.index(max(values))
                lines.append('{}output = {};'.format(indent, output))
            elif return_value == 'class_prob_array':
                prob_array = list(np.divide(tree_.value[node][0], np.sum(tree_.value[node][0])))
                lines.append('{}output = {};'.format(indent, prob_array))
    #
    recurse(0, 1)
    lines.append('')
    lines.append('\treturn output;')
    lines.append('')
    lines.append('}')
    lines.append('')
    return '\n'.join(lines)


def tree2formula(tree, model_symbols, return_value = 'class_idx_max', tree_idx = ''):
    tree_ = tree.tree_
    feature_names = list(model_symbols['counterfactual'].keys())
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
        for i in tree_.feature
    ]

    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = float(tree_.threshold[node])
            return Or(
                And(
                    LE(ToReal(model_symbols['counterfactual'][name]['symbol']), Real(threshold)),
                    recurse(tree_.children_left[node])
                ),
                And(
                    Not(LE(ToReal(model_symbols['counterfactual'][name]['symbol']), Real(threshold))),
                    recurse(tree_.children_right[node])
                )
            )
        else:
            if return_value == 'class_idx_max':
                values = list(tree_.value[node][0])
                output = bool(values.index(max(values)))
                return EqualsOrIff(model_symbols['output']['y']['symbol'], Bool(output))
            elif return_value == 'class_prob_array':
                prob_array = list(np.divide(tree_.value[node][0], np.sum(tree_.value[node][0])))
                return And(
                    EqualsOrIff(model_symbols['aux'][f'p0{tree_idx}']['symbol'], Real(float(prob_array[0]))),
                    EqualsOrIff(model_symbols['aux'][f'p1{tree_idx}']['symbol'], Real(float(prob_array[1])))
                )

    return recurse(0)


################################################################################
##                                                       Forest-related Methods
################################################################################

def forest2py(forest, feature_names):
    lines = []
    for tree_idx in range(len(forest.estimators_)):
        tree = forest.estimators_[tree_idx]
        lines.append(tree2py(tree, feature_names, return_value = 'class_prob_array', tree_idx = tree_idx))
        lines.append('')
        lines.append('')
    lines.append('def predict_forest({}):'.format(', '.join(feature_names)))
    lines.append('')
    lines.append('\tsum_prob = [0, 0]')
    for tree_idx in range(len(forest.estimators_)):
        lines.append('')
        lines.append('\tsum_prob[0] = sum_prob[0] + predict_tree{}({})[0]'.format(tree_idx, ', '.join(feature_names)))
        lines.append('\tsum_prob[1] = sum_prob[1] + predict_tree{}({})[1]'.format(tree_idx, ', '.join(feature_names)))
    lines.append('')
    lines.append('\tif sum_prob[0] >= sum_prob[1]:')
    lines.append('\t\toutput = 0')
    lines.append('\telse:')
    lines.append('\t\toutput = 1')
    lines.append('')
    lines.append('\treturn output')
    lines.append('')
    lines.append('')
    return '\n'.join(lines)


def forest2c(forest, feature_names):
    lines = []
    for tree_idx in range(len(forest.estimators_)):
        tree = forest.estimators_[tree_idx]
        lines.append(tree2c(tree, feature_names, return_value = 'class_prob_array', tree_idx = tree_idx))
        lines.append('')
        lines.append('')
    lines.append('proc predict_forest({} : int) : int = {{'.format(', '.join(feature_names)))
    lines.append('')
    lines.append('\tsum_prob = [0, 0] : int;')
    for tree_idx in range(len(forest.estimators_)):
        lines.append('')
        lines.append('\tsum_prob[0] = sum_prob[0] + predict_tree{}({})[0];'.format(tree_idx, ', '.join(feature_names)))
        lines.append('\tsum_prob[1] = sum_prob[1] + predict_tree{}({})[1];'.format(tree_idx, ', '.join(feature_names)))
    lines.append('')
    lines.append('\tif sum_prob[0] >= sum_prob[1] {{')
    lines.append('\t\toutput = 0;')
    lines.append('\t} else {')
    lines.append('\t\toutput = 1;')
    lines.append('\t}')
    lines.append('')
    lines.append('\treturn output;')
    lines.append('')
    lines.append('}')
    lines.append('')
    return '\n'.join(lines)


def forest2formula(forest, model_symbols):
    model_symbols['aux'] = {}
    for tree_idx in range(len(forest.estimators_)):
        model_symbols['aux'][f'p0{tree_idx}'] = {'symbol': Symbol(f'p0{tree_idx}', REAL)}
        model_symbols['aux'][f'p1{tree_idx}'] = {'symbol': Symbol(f'p1{tree_idx}', REAL)}

    tree_formulas = And([
        tree2formula(forest.estimators_[tree_idx], model_symbols, return_value = 'class_prob_array', tree_idx = tree_idx)
        for tree_idx in range(len(forest.estimators_))
    ])
    output_formula = Ite(
        GE(
            Plus([model_symbols['aux'][f'p0{tree_idx}']['symbol'] for tree_idx in range(len(forest.estimators_))]),
            Plus([model_symbols['aux'][f'p1{tree_idx}']['symbol'] for tree_idx in range(len(forest.estimators_))])
        ),
        EqualsOrIff(model_symbols['output']['y']['symbol'], FALSE()),
        EqualsOrIff(model_symbols['output']['y']['symbol'], TRUE()),
    )

    return And(
        tree_formulas,
        output_formula
    )


################################################################################
##                                          Logistic Regression-related Methods
################################################################################

def lr2py(model, feature_names):
    lines = list()
    lines.append('def predict_lr(x):')
    lines.append('\tw = [{}]'.format(', '.join(['{}'.format(w) for w in model.coef_[0]])))
    lines.append('\tscore = {}'.format(model.intercept_[0]))
    lines.append('\tfor i in range({}):'.format(model.coef_.shape[1]))
    lines.append('\t\tscore += x[i] * w[i]')
    lines.append('\tif score > 0:')
    lines.append('\t\treturn 1')
    lines.append('\treturn 0')
    return '\n'.join(lines)


def lr2c(model, feature_names):
    lines = list()
    lines.append('proc predict_lr({} : int) : int = {{'.format(' : int, '.join(feature_names)))
    lines.append('')
    lines.append('\tvar {}, score : int;'.format(', '.join(['w'+str(i) for i in range(len(feature_names))])))
    lines.append('\tvar output : int;')
    lines.append('')
    for i in range(len(feature_names)):
        lines.append('\tw{} = {};'.format(i, model.coef_[0][i]))
    lines.append('\tscore0 = {};'.format(model.intercept_[0]))
    lines.append('')
    for i in range(len(feature_names)):
        lines.append('\tscore{} = score{} + ({} * w{});'.format(i+1, i, feature_names[i], i))
    lines.append('')
    lines.append('\tif (score{} > 0) {{'.format(i+1))
    lines.append('\t\toutput = 1;')
    lines.append('\t} else {')
    lines.append('\t\toutput = 0;')
    lines.append('\t}')
    lines.append('')
    lines.append('\treturn output;')
    lines.append('')
    lines.append('}')
    lines.append('')
    return '\n'.join(lines)


def lr2formula(model, model_symbols):
    return Ite(
        # it turns out that sklearn's model.predict() classifies the following
        # as class -1, not +1: np.dot(coef_, sample) + intercept_ = 0
        # Therefore, below we should use GT, and not GE. We didn't encounter
        # this bug before, becuase of another bug in genMACEExplanations where
        # numeric-real variables were not being set to REAL. Therefore, it never
        # happened that np.dot(coef_, sample) + intercept_ would be exactly 0 lol!
        # UPDATE (2020.02.28 & 2020.06.24): turns out that:
        #                        for factual_sample['y'] = 1 for which we seek a negative CF --> GE works, GT fails
        #                        for factual_sample['y'] = 0 for which we seek a positive CF --> GT works, GE fails
        # ... which is due to 2e-16 values... therefore, I reluctantly added a
        # condition to assertPrediction in genSATExplanations.py. Resolved.
        GT(
            Plus(
                Real(float(model.intercept_[0])),
                Plus([
                    Times(
                        ToReal(model_symbols['counterfactual'][symbol_key]['symbol']),
                        Real(float(model.coef_[0][idx]))
                        )
                    for idx, symbol_key in enumerate(model_symbols['counterfactual'].keys())
                ])
            ),
            Real(0)),
        EqualsOrIff(model_symbols['output']['y']['symbol'], TRUE()),
        EqualsOrIff(model_symbols['output']['y']['symbol'], FALSE())
    )


################################################################################
##                                       Multi-Layer Perceptron-related Methods
################################################################################

def mlp2py(model):
    lines = list()
    lines.append('def predict_mlp(model, x):')
    lines.append('\tlayer_output = x')
    lines.append('\tfor layer_idx in range(len(model.coefs_)):')
    lines.append('\t\t#')
    lines.append('\t\tlayer_input_size = len(model.coefs_[layer_idx])')
    lines.append('\t\tif layer_idx != len(model.coefs_) - 1:')
    lines.append('\t\t\tlayer_output_size = len(model.coefs_[layer_idx + 1])')
    lines.append('\t\telse:')
    lines.append('\t\t\tlayer_output_size = model.n_outputs_')
    lines.append('\t\t#')
    lines.append('\t\tlayer_input = layer_output')
    lines.append('\t\tlayer_output = [0 for j in range(layer_output_size)]')
    lines.append('\t\t# i: indices of nodes in layer L')
    lines.append('\t\t# j: indices of nodes in layer L + 1')
    lines.append('\t\tfor j in range(layer_output_size):')
    lines.append('\t\t\tscore = model.intercepts_[layer_idx][j]')
    lines.append('\t\t\tfor i in range(layer_input_size):')
    lines.append('\t\t\t\tscore += layer_input[i] * model.coefs_[layer_idx][i][j]')
    lines.append('\t\t\tif score > 0: # relu operator')
    lines.append('\t\t\t\tlayer_output[j] = score')
    lines.append('\t\t\telse:')
    lines.append('\t\t\t\tlayer_output[j] = 0')
    lines.append('\tif layer_output[0] > 0:')
    lines.append('\t\treturn 1')
    lines.append('\treturn 0')
    return '\n'.join(lines)


def mlp2c(model, feature_names):

    num_layers = 2 + len(model.hidden_layer_sizes)
    layer_widths = []
    layer_widths.append(len(feature_names))
    layer_widths.extend(model.hidden_layer_sizes)
    layer_widths.append(model.n_outputs_)

    all_weight_variables = []
    for interlayer_idx in range(len(model.coefs_)):
        interlayer_weight_matrix = model.coefs_[interlayer_idx]
        for prev_layer_feature_idx in range(interlayer_weight_matrix.shape[0]):
            for curr_layer_feature_idx in range(interlayer_weight_matrix.shape[1]):
                all_weight_variables.append('w_{}_{}_{}'.format(interlayer_idx, prev_layer_feature_idx, curr_layer_feature_idx))

    all_feature_variables = []
    for layer_idx in range(len(layer_widths)):
        for feature_idx in range(layer_widths[layer_idx]):
            all_feature_variables.append('f_{}_{}'.format(layer_idx, feature_idx))

    lines = []
    lines.append('proc predict_mlp({} : int) : int = {{'.format(' : int, '.join(feature_names)))

    lines.append('')

    # lines.append('\tvar num_layers : int;')
    # lines.append('\tvar layer_widths : int list;')
    lines.append('\tvar {} : int;'.format(', '.join(all_weight_variables)))
    lines.append('\tvar {} : int;'.format(', '.join(all_feature_variables)))
    lines.append('\tvar output : int;')

    lines.append('')

    # lines.append('\tnum_layers = {};'.format(num_layers))
    # for i in range(len(layer_widths)):
    #     lines.append('\tlayer_widths = layer_widths ++ [{}]'.format(layer_widths[i]))

    for interlayer_idx in range(len(model.coefs_)):
        interlayer_weight_matrix = model.coefs_[interlayer_idx]
        for prev_layer_feature_idx in range(interlayer_weight_matrix.shape[0]):
            for curr_layer_feature_idx in range(interlayer_weight_matrix.shape[1]):
                lines.append('\tw_{}_{}_{} = {};'.format(interlayer_idx, prev_layer_feature_idx, curr_layer_feature_idx, interlayer_weight_matrix[prev_layer_feature_idx, curr_layer_feature_idx]))
    # TODO... why don't lines below work??
    # for layer_idx in range(len(layer_widths) - 1):
    #     for prev_layer_feature_idx in range(layer_widths[layer_idx]):
    #         for curr_layer_feature_idx in range(layer_widths[layer_idx] + 1):
    #             lines.append('\tw_{}_{}_{} = {};'.format(layer_idx, prev_layer_feature_idx, curr_layer_feature_idx, model.coefs_[layer_idx][prev_layer_feature_idx, curr_layer_feature_idx]))
    lines.append('')
    lines.append('\t%% value set to input to MLP')
    for feature_idx in range(layer_widths[0]):
        lines.append('\tf_{}_{} = {};'.format(0, feature_idx, feature_names[feature_idx]))
    lines.append('\t%% initial value set to node bias')
    for layer_idx in range(1, len(layer_widths)):
        for feature_idx in range(layer_widths[layer_idx]):
            lines.append('\tf_{}_{} = {};'.format(layer_idx, feature_idx, model.intercepts_[layer_idx - 1][feature_idx]))

    lines.append('')

    # TODO: s/int/float
    for layer_idx in range(1, len(layer_widths)):
        lines.append('')
        lines.append('\t%% Layer {}'.format(layer_idx))
        for curr_layer_feature_idx in range(layer_widths[layer_idx]):
            curr_layer_feature = 'f_{}_{}'.format(layer_idx, curr_layer_feature_idx)
            tmp_strings = []
            for prev_layer_feature_idx in range(layer_widths[layer_idx - 1]):
                prev_layer_feature = 'f_{}_{}'.format(layer_idx - 1, prev_layer_feature_idx)
                tmp_strings.append('({} * w_{}_{}_{})'.format(prev_layer_feature, layer_idx - 1, prev_layer_feature_idx, curr_layer_feature_idx))
            lines.append('\t{} = {} + {};'.format(curr_layer_feature, curr_layer_feature, ' + '.join(tmp_strings)))
            #
            lines.append('\tif ({} > 0) {{'.format(curr_layer_feature))
            lines.append('\t\t{} = {};'.format(curr_layer_feature, curr_layer_feature))
            lines.append('\t} else {')
            lines.append('\t\t{} = 0;'.format(curr_layer_feature))
            lines.append('\t}')
            #
            # lines.append('\t{} = ({} > 0) ? {} : 0'.format(curr_layer_feature, curr_layer_feature, curr_layer_feature))

    # using the final value set to curr_layer_feature (remember it's only 1 value because binary classification)
    lines.append('')
    lines.append('\toutput = {};'.format(curr_layer_feature)) # TODO: if > 1, return TRUE, else FALSE
    lines.append('\treturn output;')
    lines.append('')
    lines.append('}')
    lines.append('')
    return '\n'.join(lines)


def mlp2formula(model, model_symbols):

    model_symbols['aux'] = {}
    layer_widths = []
    for interlayer_idx in range(len(model.coefs_)):
        layer_widths.append(model.coefs_[interlayer_idx].shape[0])
    layer_widths.append(model.coefs_[-1].shape[1])


    for layer_idx in range(1, len(layer_widths)):
        for feature_idx in range(layer_widths[layer_idx]):
            feature_string_1 = 'f_{}_{}_pre_nonlin'.format(layer_idx, feature_idx)
            feature_string_2 = 'f_{}_{}_post_nonlin'.format(layer_idx, feature_idx)
            model_symbols['aux'][feature_string_1] = {'symbol': Symbol(feature_string_1, REAL)}
            model_symbols['aux'][feature_string_2] = {'symbol': Symbol(feature_string_2, REAL)}


    formula_assign_feature_values = []
    for layer_idx in range(1, len(layer_widths)):

        interlayer_weight_matrix = model.coefs_[layer_idx - 1]

        for curr_layer_feature_idx in range(layer_widths[layer_idx]):

            curr_layer_feature_string_1 = 'f_{}_{}_pre_nonlin'.format(layer_idx, curr_layer_feature_idx)
            curr_layer_feature_string_2 = 'f_{}_{}_post_nonlin'.format(layer_idx, curr_layer_feature_idx)
            bias_string = 'b_{}_{}'.format(layer_idx, curr_layer_feature_idx)

            inputs_to_curr_layer_feature = []
            inputs_to_curr_layer_feature.append(Real(float(model.intercepts_[layer_idx - 1][curr_layer_feature_idx])))

            for prev_layer_feature_idx in range(layer_widths[layer_idx - 1]):

                if layer_idx == 1:
                    input_layer_feature_string = list(model_symbols['counterfactual'].keys())[prev_layer_feature_idx]
                    weight_string = 'w_{}_{}_{}'.format(layer_idx - 1, prev_layer_feature_idx, curr_layer_feature_idx)
                    inputs_to_curr_layer_feature.append(
                        Times(
                            ToReal(model_symbols['counterfactual'][input_layer_feature_string]['symbol']),
                            Real(float(interlayer_weight_matrix[prev_layer_feature_idx, curr_layer_feature_idx]))
                        )
                    )
                else:
                    prev_layer_feature_string_2 = 'f_{}_{}_post_nonlin'.format(layer_idx - 1, prev_layer_feature_idx)
                    weight_string = 'w_{}_{}_{}'.format(layer_idx - 1, prev_layer_feature_idx, curr_layer_feature_idx)
                    inputs_to_curr_layer_feature.append(
                        Times(
                            model_symbols['aux'][prev_layer_feature_string_2]['symbol'],
                            Real(float(interlayer_weight_matrix[prev_layer_feature_idx, curr_layer_feature_idx]))
                        )
                    )

            formula_assign_feature_values.append(
                EqualsOrIff(
                    model_symbols['aux'][curr_layer_feature_string_1]['symbol'],
                    Plus(inputs_to_curr_layer_feature),
                )
            )

            formula_assign_feature_values.append(
                EqualsOrIff(
                    model_symbols['aux'][curr_layer_feature_string_2]['symbol'],
                    Ite(
                        GE(model_symbols['aux'][curr_layer_feature_string_1]['symbol'], Real(0)),
                        model_symbols['aux'][curr_layer_feature_string_1]['symbol'],
                        Real(0)
                    )
                )

            )

    final_layer_binary_feature_string = f'f_{len(layer_widths) - 1}_0_post_nonlin'
    output_formula = Ite(
        GT(
            model_symbols['aux'][final_layer_binary_feature_string]['symbol'],
            Real(0),
        ),
        EqualsOrIff(model_symbols['output']['y']['symbol'], TRUE()),
        EqualsOrIff(model_symbols['output']['y']['symbol'], FALSE()),
    )

    # Flatten before And() to get a & b & c, not (a & b) & c... maybe easier for solver.
    tmp = []
    for elem in formula_assign_feature_values:
        tmp.append(elem)
    tmp.append(output_formula)

    return And(tmp)








