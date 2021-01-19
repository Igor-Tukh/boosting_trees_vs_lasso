from typing import List

from monomial import Monomial
from tqdm import tqdm

import numpy as np

current_rules = []
monomials = []
rules = set([])
X = None


def _sift_monomials(ms):
    sifted_result = []
    for monomial in ms:
        sifted_result.append(monomial)
        for other_monomial in sifted_result[:-1]:
            if other_monomial == monomial:
                sifted_result = sifted_result[:-1]
                break
    return sifted_result


def add_monomial(m):
    global monomials
    if m in monomials:
        return
    if X is not None:
        exists = False
        for x in X:
            if m(x):
                exists = True
                break
        if not exists:
            return
    monomials.append(m)


def parse_monomials_from_decision_tree_ensemble(ensemble, debug_output=True, X_train=None):
    global X
    if X_train is not None:
        X = X_train
    result = []
    for ind, estimator in enumerate(tqdm(ensemble.estimators_)):
        if debug_output:
            print(f'Tree {ind}:')
        result.extend(parse_decision_tree(estimator[0], debug_output))
    # result = _sift_monomials(result)
    if debug_output:
        print('All parsed monomials:')
        for monomial in result:
            monomial.print_to_console()
    return result


def parse_decision_tree(decision_tree_estimator, debug_output=True) -> List[Monomial]:
    global current_rules
    current_rules = []
    result = _dfs(0,
                  decision_tree_estimator.tree_.children_left,
                  decision_tree_estimator.tree_.children_right,
                  decision_tree_estimator.tree_.feature,
                  decision_tree_estimator.tree_.threshold,
                  debug_output)
    return result


def _dfs(current_ind, left_children, right_children, feature, threshold, debug_output=True):
    global current_rules, monomials, rules
    result = []
    if left_children[current_ind] == right_children[current_ind]:
        if debug_output:
            print(f'Leaf {current_ind}', end=': ')
            for feature, threshold, optional in current_rules:
                if optional:
                    print(f'(1 - [x[{feature}] >= {threshold}])', end='')
                else:
                    print(f'[x[{feature}] >= {threshold}]', end='')
            print()
        monomials = []
        rules = []
        _generate_monomials(0, current_rules)
        if debug_output:
            print('Parsed monomials:')
            for monomial in monomials:
                monomial.print_to_console()
        return monomials
    current_rules.append((feature[current_ind], threshold[current_ind], True))
    result.extend(_dfs(left_children[current_ind], left_children, right_children, feature, threshold, debug_output))
    current_rules = current_rules[:-1]
    current_rules.append((feature[current_ind], threshold[current_ind], False))
    result.extend(_dfs(right_children[current_ind], left_children, right_children, feature, threshold, debug_output))
    current_rules = current_rules[:-1]
    return result


def _generate_monomials(current_ind, feature_threshold_optional):
    global rules, monomials
    if current_ind == len(feature_threshold_optional):
        new_monomial = Monomial()
        for feature, threshold in rules:
            new_monomial.add_split(feature, threshold)
        new_monomial.finalize()
        # monomials.append(new_monomial)годня 
        add_monomial(new_monomial)
        return
    if feature_threshold_optional[current_ind][2]:
        _generate_monomials(current_ind + 1, feature_threshold_optional)
    rules.append((feature_threshold_optional[current_ind][0], feature_threshold_optional[current_ind][1]))
    _generate_monomials(current_ind + 1, feature_threshold_optional)
    rules = rules[:-1]
