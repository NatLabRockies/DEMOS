import copy
import numpy as np

#################################
### NUMPY NESTED LOGIT ##########
#################################

def flatten_dict_values(dictionary):
    return {k: np.array(list(v.values())) for k, v in dictionary.items()}


def dict_to_array(d):
    x = [values for _, values in d.items()]
    return np.array(x)


def map_dict_values(dict_1, dict_2):
    new_dict = copy.deepcopy(dict_2)

    for key, value in dict_2.items():
        for k, v in value.items():
            try:
                new_dict[key][k] = dict_1[v]
            except KeyError:
                pass
    return new_dict


def softmax(x, scale=1):
    """
    Probability estimation for j alternatives and n observations.

    Parameters:
    ------------
    - x: array-like. Utility input array in the shape (j, n)
    - scale: int, optional - scaling factor for exp(scale * x). default = 1
    j is the number of alternatives, n is the number of observations.
    Returns:
    --------
    An array with the same shape of x.
    The result will sum to 1 along the 0 axis.
    """
    exp_utility = np.exp(scale * x)
    sum_exp_utility = np.sum(exp_utility, axis = 0, keepdims=True)
    probs = exp_utility / sum_exp_utility
    # assert probs.sum(axis = 1) ==  1 ## FIX ME: Find a better way to do this
    probs = np.nan_to_num(probs)
    return probs


def softplus(x, scale=1):
    """
    [Logsum Function] Maximum expected utility of x (along 0 axis)

    Parameters:
    ------------
    - x: array-like. Utility input array in the shape (j, n)
    - scale: int, optional - ccaling factor for exp(scale * x). default = 1
    j is the number of alternatives, n is the number of observations.
    Returns:
    --------
    An array of shape (1,n) with the maximum expected utility.
    """
    return (1/scale) * np.log(np.sum(np.exp(scale * x), axis = 0))


def nested_probabilities(x, nest_specs):
    '''
    Estimates the probabilities and logsum for a given nest structure

    Parameters:
    ------------
    - x: array-like. Utility input array in the shape (j,n)
    - nest_specs: dict. dictionary with nesting structure.

    Returns:
    --------
    (array) An array with the same shape of x. The result will
    sum to 1 along 0 axis, (array) An array of shape (1,n) with
    the maximum expected utility of the nest.
    '''
    is_leaf = isinstance(nest_specs['alternatives'], np.ndarray)

    if is_leaf:
        alternatives = nest_specs['alternatives'] - 1
        scale = nest_specs['coefficient']
        u = x[alternatives] # Select unscaled utilities of alternatives in the nest
        return softmax(u, scale), softplus(u, scale)

    else:
        softmax_, logsums_ = [], []
        scale = nest_specs['coefficient']

        for nest in nest_specs['alternatives']:
            s, l = nested_probabilities(x, nest)
            softmax_.append(s)
            logsums_.append(l)

        logsums_ =  np.vstack(logsums_)
        nest_probs = softmax(logsums_, scale)
        nest_logsum = softplus(logsums_, scale)

        probs = []
        for i in range(len(softmax_)):
            prob = nest_probs[i] * softmax_[i]
            probs.append(prob)

        return np.concatenate(probs, axis = 0), nest_logsum


def mode_choice_probabilities(params, specs, exp_vars, nest = None):

    w = map_dict_values(params, specs)
    w = flatten_dict_values(w)

    utils = {mode: np.dot(w[mode].T, exp_vars[mode]) for mode in w.keys()}
    utils = dict_to_array(utils)

    if nest is not None:
        probabilities, _ = nested_probabilities(utils, nest)
    else:
        probabilities = softmax(utils, scale=1)

    return probabilities


def mode_choice_logsums(params, specs, exp_vars, nest = None):

    w = map_dict_values(params, specs)
    w = flatten_dict_values(w)

    utils = {mode: np.dot(w[mode].T, exp_vars[mode]) for mode in w.keys()}
    utils = dict_to_array(utils)

    if nest is not None:
        _, logsums = nested_probabilities(utils, nest)
    else:
        logsums = softplus(utils, scale = 1)

    return logsums