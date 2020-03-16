from sklearn.datasets import make_gaussian_quantiles
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import operator
import prettytable


def bayes_classifier(x_vec, kdes):
    """
    Classifies an input sample into class w_j determined by
    maximizing the class conditional probability for p(x|w_j).

    Keyword arguments:
        x_vec: A dx1 dimensional numpy array representing the sample.
        kdes: List of the gausssian_kde (kernel density) estimates

    Returns a tuple ( p(x|w_j)_value, class label ).

    """
    p_vals = []
    for kde in kdes:
        p_vals.append(kde.evaluate(x_vec))
    max_index, max_value = max(enumerate(p_vals), key=operator.itemgetter(1))
    return (max_value, max_index + 1)


def empirical_error(data_set, classes, classifier_func, classifier_func_args):
    """
    Keyword arguments:
        data_set: 'n x d'- dimensional numpy array, class label in the last column.
        classes: List of the class labels.
        classifier_func: Function that returns the max argument from the discriminant function.
            evaluation and the class label as a tuple.
        classifier_func_args: List of arguments for the 'classifier_func'.

    Returns a tuple, consisting of a dictionary withthe classif. counts and the error.

    e.g., ( {1: {1: 321, 2: 5}, 2: {1: 0, 2: 317}}, 0.05)
    where keys are class labels, and values are sub-dicts counting for which class (key)
    how many samples where classified as such.

    """
    class_dict = {i: {j: 0 for j in classes} for i in classes}

    for cl in classes:
        for row in data_set[data_set[:, -1] == cl][:, :-1]:
            g = classifier_func(row, *classifier_func_args)
            class_dict[cl][g[1]] += 1

    correct = 0
    for i in classes:
        correct += class_dict[i][i]

    misclass = data_set.shape[0] - correct
    return (class_dict, misclass / data_set.shape[0])


