import gurobipy as grb
import numpy as np


def applyDistanceConstrs(model, dataset_obj, factual_sample, norm_type, norm_lower, norm_upper):

    mutables = dataset_obj.getMutableAttributeNames('kurz')
    one_hots = dataset_obj.getOneHotAttributesNames('kurz')
    non_hots = dataset_obj.getNonHotAttributesNames('kurz')

    assert 'one_norm' in norm_type
    abs_diffs_normalized = []

    # TODO: should these intermediate variables also have the same type as input vars?

    # 1. mutable & non-hot
    for attr_name_kurz in np.intersect1d(mutables, non_hots):
        v = model.getVarByName(attr_name_kurz)
        lb = dataset_obj.attributes_kurz[attr_name_kurz].lower_bound
        ub = dataset_obj.attributes_kurz[attr_name_kurz].upper_bound

        diff_normalized = model.addVar(lb=-1.0, ub=1.0, obj=0,
                                            vtype=grb.GRB.CONTINUOUS, name=f'diff_{attr_name_kurz}')
        model.addConstr(
            diff_normalized == (v - factual_sample[attr_name_kurz]) / (ub - lb)
        )
        abs_diff_normalized = model.addVar(lb=0.0, ub=1.0, obj=0,
                                                vtype=grb.GRB.CONTINUOUS, name=f'abs_{attr_name_kurz}')
        model.addConstr(
            abs_diff_normalized == grb.abs_(diff_normalized)
        )
        abs_diffs_normalized.append(abs_diff_normalized)

    # 2. mutable & integer-based & one-hot
    already_considered = []
    for attr_name_kurz in np.intersect1d(mutables, one_hots):
        if attr_name_kurz not in already_considered:
            siblings_kurz = dataset_obj.getSiblingsFor(attr_name_kurz)
            if 'cat' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:

                diff_normalized = model.addVar(lb=-1.0, ub=1.0, obj=0, vtype=grb.GRB.CONTINUOUS,
                                                    name=f'diff_{attr_name_kurz}')
                model.addConstr(
                    diff_normalized == grb.max_(model.getVarByName(sib_name_kurz) - factual_sample[sib_name_kurz]
                                                for sib_name_kurz in siblings_kurz)
                )
                # It's either 0 or 1, so no need for grb.abs_()
                abs_diffs_normalized.append(diff_normalized)

            elif 'ord' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
                diff_normalized = model.addVar(lb=-1.0, ub=1.0, obj=0, vtype=grb.GRB.CONTINUOUS,
                                                    name=f'diff_{attr_name_kurz}')
                model.addConstr(
                    diff_normalized == (grb.quicksum(
                        model.getVarByName(sib_name_kurz) for sib_name_kurz in siblings_kurz
                    )
                                        -
                                        sum(factual_sample[sib_name_kurz] for sib_name_kurz in siblings_kurz))
                    /
                    len(siblings_kurz)
                )
                abs_diff_normalized = model.addVar(lb=0.0, ub=1.0, obj=0,
                                                        vtype=grb.GRB.CONTINUOUS, name=f'abs_{attr_name_kurz}')
                model.addConstr(
                    abs_diff_normalized == grb.abs_(diff_normalized)
                )
                abs_diffs_normalized.append(abs_diff_normalized)
            else:
                raise Exception(f'{attr_name_kurz} must include either `cat` or `ord`.')
            already_considered.extend(siblings_kurz)

    normalized_distance = model.addVar(lb=0.0, ub=1.0, obj=0, vtype=grb.GRB.CONTINUOUS,
                                                    name='normalized_distance')
    model.addConstr(
        normalized_distance ==
        grb.quicksum(abs_diffs_normalized) / len(abs_diffs_normalized)
    )

    if 'obj' not in norm_type:
        model.addConstr(normalized_distance <= norm_upper)
        if norm_lower != 0.0:
            model.addConstr(normalized_distance >= norm_lower)


def applyPlausibilityConstrs(model, dataset_obj):
    # 1. Data range plausibility:
    #    Already met when defining input variables

    # 2. Data type plausibility
    # TODO: add categorical

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

    # 3. Actionability + Mutability
    # 4. Causal Consistency

