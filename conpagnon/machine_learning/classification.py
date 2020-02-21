#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:40:49 2017

@author: db242421 (dhaif.bekha@cea.fr)
"""

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.svm import LinearSVC
import matplotlib.pylab as plt
import seaborn as sns


def two_groups_classification(pooled_groups_connectivity_matrices, kinds, labels, n_splits, test_size,train_size, C=1.0,
                              dual=True, fit_intercept=True, loss='squared_hinge', max_iter=1000,
                              penalty='l2', random_state=0, scoring='accuracy'):
    """Perform binary classification on two classes using connectivity coefficient as features.


    Perform classification  on two labels classes using a support vector machine with a linear kernel function.

    Parameters
    ----------
    pooled_groups_connectivity_matrices : dict
         A multi-levels dictionnary organised as follow :
             - The first key is the different groupes to compare.
             - The second key is the different kinds you want to test
             for classification. The values are simply the stacked connectivity
             matrices as a ndarray of shape (..., number of regions, number of regions)
    kinds : list
        The different kind you want to test for classification.
    labels : numpy.array of shape (number of subjects)
        A one-dimensional numpy array of binary labels like 0 for the
        first class and one for the second class.
    n_splits : int
        The number of splitting operations performed by the cross validation
        scheme, StratifiedShuffleSplit.
    test_size : float
        Between 0 and 1, the proportion of the testing set. This is the
        complementary of the train_size : test_size = 1 - train_size.
    train_size : float
        Between 0 and 1, the proportion of the training set. This is the
        complementary of the test_size : train_size = 1 - test_size.
    C : float, optional
        Penalty parameter of the error term.
        Default is 1.0.
    dual : bool, optional
        Algoritm to solve the dual or primal formulation of the problems.
        When n_samples < n_features, prefer dual = True, and False otherwise.
    fit_intercept : bool, optional
        Fit an intercept in the model. If false, the data assumed to be centered.
        Default is True

    loss : str, optional
        The loss function you want to use. Choices are : 'hinge' or 'squared_hinge'.
        Default is 'squared_hinge'.
    max_iter : int, optional
        The maximum iteration you want to run
        Default is 1000.
    penalty : str, optional
        The type of penalty term you want to add in the model. Choices are
        the L1 norm 'l1' or L2 norm 'l2'.
        Default is 'l2'.
    scoring : str, optional
        The maners you want to evaluate the classification by computing
        a score.
        Default is 'accuracy'. See Notes.


    Returns
    -------
    output 1 : list
        List of mean scores for each kind.

    output 2 : dict
        Dictionnary of mean scores with each kind as keys.


    Raises
    ------
    ValueError : If the numbers of the different labels is less than two,
    raises a ValueError. Binary classification requires at least two different classes.

    Notes
    -----
    The classifier, and the Cross validation scheme are from the scikit learn
    library. I encourage the users to consult the scikit learn site cited
    in the references to further details concerning the way using support
    vector machines as classifiers, and the different ways of evaluating classification
    algorithm.

    References
    ----------
    The Scikit learn official documentation.
    [1] http://scikit-learn.org/stable/index.html


    """

    # Dictionary for saving the mean accuracy of classification
    mean_score_dict = {}

    # Check that they are two group to classify, i.e two different labels
    n_labels = len(np.unique(labels))
    if n_labels < 2:
        raise ValueError('At least classification require two classes, only {} label where found !'.format(n_labels))

    # liste qui va contenir la moyenne des scores de classification pour chaque kinds
    mean_scores = []
    # Si les labels sont au format de liste on les convertis en tableaux
    if isinstance(labels, str):
        labels = np.array(labels)

    # K-Fold cross validation schéma:
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state, train_size=train_size, test_size=test_size)

    # Boucle pour classer les deux groupes
    for kind in kinds:
        # Initialisation du classifieur:
        svc = LinearSVC(random_state=random_state, penalty=penalty, loss=loss, fit_intercept=fit_intercept, C=C,
                        dual = dual, max_iter = max_iter)
        # On estime les parametres du model et on calcules les scores pour chaque fold par cross-validation
        cv_scores = cross_val_score(estimator = svc, X = pooled_groups_connectivity_matrices[kind], y = labels, cv = cv, scoring = scoring)
        # On calcule la moyenne selon les k-fold et on empile dans mean_scores:
        mean_scores.append(cv_scores.mean())
        mean_score_dict[kind] = cv_scores.mean()

    return mean_scores, mean_score_dict
