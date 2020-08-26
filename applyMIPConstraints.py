import gurobipy as grb
import numpy as np

from modelConversion import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def applyTrainedModelConstrs(model, model_vars, model_trained):
    if isinstance(model_trained, DecisionTreeClassifier):
        tree2mip(model_trained, model, model_vars)
    elif isinstance(model_trained, LogisticRegression):
        lr2mip(model_trained, model, model_vars)
    elif isinstance(model_trained, RandomForestClassifier):
        forest2mip(model_trained, model, model_vars)
    else:
        raise Exception(f"Trained model type not recognized")

def applyDistanceConstrs(model: grb.Model, dataset_obj, factual_sample, norm_type, norm_lower=0, norm_upper=0, id=None):

    mutables = dataset_obj.getMutableAttributeNames('kurz')
    one_hots = dataset_obj.getOneHotAttributesNames('kurz')
    non_hots = dataset_obj.getNonHotAttributesNames('kurz')

    abs_diffs_normalized = []
    squ_diffs_normalized = []
    is_zeros = []

    # TODO: should these intermediate variables also have the same type as input vars?
    # TODO: variable bounds w.r.t. new bounds derived from distance? (for second bounded MIP network)

    # 1. mutable & non-hot
    for attr_name_kurz in np.intersect1d(mutables, non_hots):
        v = model.getVarByName(attr_name_kurz)
        lb = dataset_obj.attributes_kurz[attr_name_kurz].lower_bound
        ub = dataset_obj.attributes_kurz[attr_name_kurz].upper_bound

        if 'zero_norm' in norm_type:
            is_zero = model.addVar(obj=0, vtype=grb.GRB.BINARY, name=f'is_zero_{attr_name_kurz}_{id}')
            diff = model.addVar(lb=lb-ub, ub=ub-lb, obj=0,
                                    vtype=grb.GRB.CONTINUOUS, name=f'diff_{attr_name_kurz}_{id}')
            model.addConstr(diff == v - factual_sample[attr_name_kurz])
            model.addSOS(grb.GRB.SOS_TYPE1, [diff, is_zero])
            is_zeros.append(is_zero)
        else:
            diff_normalized = model.addVar(lb=-1.0, ub=1.0, obj=0,
                                                vtype=grb.GRB.CONTINUOUS, name=f'diff_{attr_name_kurz}_{id}')
            model.addConstr(
                diff_normalized == (v - factual_sample[attr_name_kurz]) / (ub - lb)
            )

        if 'one_norm' in norm_type or 'infty_norm' in norm_type:
            abs_diff_normalized = model.addVar(lb=0.0, ub=1.0, obj=0,
                                                    vtype=grb.GRB.CONTINUOUS, name=f'abs_{attr_name_kurz}_{id}')
            model.addConstr(
                abs_diff_normalized == grb.abs_(diff_normalized)
            )
            abs_diffs_normalized.append(abs_diff_normalized)

        elif 'two_norm' in norm_type:
            squ_diff_normalized = model.addVar(lb=0.0, ub=1.0, obj=0,
                                               vtype=grb.GRB.CONTINUOUS, name=f'squ{attr_name_kurz}_{id}')
            model.addConstr(
                squ_diff_normalized == diff_normalized * diff_normalized
            )
            squ_diffs_normalized.append(squ_diff_normalized)

    # 2. mutable & integer-based & one-hot
    already_considered = []
    for attr_name_kurz in np.intersect1d(mutables, one_hots):
        if attr_name_kurz not in already_considered:
            siblings_kurz = dataset_obj.getSiblingsFor(attr_name_kurz)
            if 'cat' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:

                diff_normalized = model.addVar(obj=0, vtype=grb.GRB.BINARY, name=f'diff_{attr_name_kurz}_{id}')

                elemwise_diffs = []
                for sib_name_kurz in siblings_kurz:
                    elem_diff = model.addVar(lb=-1.0, ub=1.0, obj=0, vtype=grb.GRB.INTEGER,
                                                   name=f'elem_diff_{attr_name_kurz}_{id}')
                    model.addConstr(elem_diff == model.getVarByName(sib_name_kurz) - factual_sample[sib_name_kurz])
                    elemwise_diffs.append(elem_diff)

                model.addConstr(
                    diff_normalized == grb.max_(elemwise_diffs)
                )
                if 'zero_norm' in norm_type:
                    is_zero = model.addVar(obj=0, vtype=grb.GRB.BINARY, name=f'is_zero_{attr_name_kurz}_{id}')
                    model.addConstr(is_zero == 1 - diff_normalized)
                    is_zeros.append(is_zero)
                else:
                    # It's either 0 or 1, so no need for grb.abs_()
                    abs_diffs_normalized.append(diff_normalized)
                    squ_diffs_normalized.append(diff_normalized)

            elif 'ord' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
                if 'zero_norm' in norm_type:
                    diff = model.addVar(lb=-len(siblings_kurz), ub=len(siblings_kurz), obj=0, vtype=grb.GRB.CONTINUOUS,
                                                   name=f'diff_{attr_name_kurz}_{id}')
                    is_zero = model.addVar(obj=0, vtype=grb.GRB.BINARY, name=f'is_zero_{attr_name_kurz}_{id}')
                    model.addConstr(
                        diff == (grb.quicksum(model.getVarByName(sib_name_kurz) for sib_name_kurz in siblings_kurz)
                                            -
                                            sum(factual_sample[sib_name_kurz] for sib_name_kurz in siblings_kurz))
                    )
                    model.addSOS(grb.GRB.SOS_TYPE1, [diff, is_zero])
                    is_zeros.append(is_zero)
                else:
                    diff_normalized = model.addVar(lb=-1.0, ub=1.0, obj=0, vtype=grb.GRB.CONTINUOUS,
                                                   name=f'diff_{attr_name_kurz}_{id}')
                    model.addConstr(
                        diff_normalized == (grb.quicksum(model.getVarByName(sib_name_kurz) for sib_name_kurz in siblings_kurz)
                                            -
                                            sum(factual_sample[sib_name_kurz] for sib_name_kurz in siblings_kurz))
                        /
                        len(siblings_kurz)
                    )
                if 'one_norm' in norm_type or 'infty_norm' in norm_type:
                    abs_diff_normalized = model.addVar(lb=0.0, ub=1.0, obj=0,
                                                            vtype=grb.GRB.CONTINUOUS, name=f'abs_{attr_name_kurz}_{id}')
                    model.addConstr(
                        abs_diff_normalized == grb.abs_(diff_normalized)
                    )
                    abs_diffs_normalized.append(abs_diff_normalized)
                elif 'two_norm' in norm_type:
                    squ_diff_normalized = model.addVar(lb=0.0, ub=1.0, obj=0,
                                                       vtype=grb.GRB.CONTINUOUS, name=f'squ_{attr_name_kurz}_{id}')
                    model.addConstr(
                        squ_diff_normalized == diff_normalized * diff_normalized
                    )
                    squ_diffs_normalized.append(squ_diff_normalized)

            else:
                raise Exception(f'{attr_name_kurz} must include either `cat` or `ord`.')
            already_considered.extend(siblings_kurz)

    dist_name = 'normalized_distance' if id is None else f'normalized_distance_{id}'
    normalized_distance = model.addVar(lb=0.0, ub=1.0, obj=0, vtype=grb.GRB.CONTINUOUS,
                                                    name=dist_name)
    if 'zero_norm' in norm_type:
        model.addConstr(
            normalized_distance
            ==
            (len(is_zeros) - grb.quicksum(is_zeros)) / len(is_zeros)
        )
    elif 'one_norm' in norm_type:
        model.addConstr(
            normalized_distance
            ==
            grb.quicksum(abs_diffs_normalized) / len(abs_diffs_normalized)
        )
    elif 'two_norm' in norm_type:
        model.addConstr(
            normalized_distance
            ==
            grb.quicksum(squ_diffs_normalized) / len(squ_diffs_normalized)
        )
    elif 'infty_norm' in norm_type:
        mx = model.addVar(lb=0.0, ub=1.0, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'max_unnormalized_{id}')
        model.addConstr(mx == grb.max_(abs_diffs_normalized))
        model.addConstr(
            normalized_distance
            ==
            mx / len(abs_diffs_normalized)
        )
    else:
        raise Exception(f"{norm_type} not a recognized norm type.")

    if 'obj' not in norm_type or id is not None:

        # For two_norm, we don't compute the squared root, instread, we square the thresholds
        if 'two_norm' in norm_type:
            norm_upper = norm_upper ** 2
            norm_lower = norm_lower ** 2

        if id is None:
            model.addConstr(normalized_distance <= norm_upper, name=f'dist_less_than_{id}')
            model.addConstr(normalized_distance >= norm_lower, name=f'dist_greater_than_{id}')
        else:
            # we are looking for Diverse CFs
            model.addConstr(normalized_distance >= norm_lower, name=f'dist_greater_than_{id}')


def applyPlausibilityConstrs(model, dataset_obj):
    # 1. Data range plausibility:
    #    Already met when defining input variables

    # 2. Data type plausibility

    dict_of_siblings_kurz = dataset_obj.getDictOfSiblings('kurz')
    for parent_name_kurz in dict_of_siblings_kurz['ord'].keys():
        model.addConstrs(
            model.getVarByName(dict_of_siblings_kurz['ord'][parent_name_kurz][name_idx])
            >=
            model.getVarByName(dict_of_siblings_kurz['ord'][parent_name_kurz][name_idx + 1])
            for name_idx in range(len(dict_of_siblings_kurz['ord'][parent_name_kurz]) - 1)
        )
        # print("\nAdding this constraint: ")
        # for name_idx in range(len(dict_of_siblings_kurz['ord'][parent_name_kurz])):
        #     print(dict_of_siblings_kurz['ord'][parent_name_kurz][name_idx] + ' >= ', end='')

    for parent_name_kurz in dict_of_siblings_kurz['cat'].keys():
        model.addConstr(
            grb.quicksum(
                model.getVarByName(dict_of_siblings_kurz['cat'][parent_name_kurz][name_idx])
                for name_idx in range(len(dict_of_siblings_kurz['cat'][parent_name_kurz]))
            )
            ==
            1
        )
        # print("\nAdding this constraint: ")
        # for name_idx in range(len(dict_of_siblings_kurz['cat'][parent_name_kurz])):
        #     print(dict_of_siblings_kurz['cat'][parent_name_kurz][name_idx] + ' + ', end='')

    # 3. Actionability + Mutability
    # 4. Causal Consistency

