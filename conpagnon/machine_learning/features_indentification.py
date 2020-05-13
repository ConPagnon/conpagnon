"""
This module enable the identification of discriminative brain
connections when performing classification between two groups
with connectivity coefficients as features.

The classification is performed with a Support Vector Machine (SVM)
algorithm with a linear kernel. The C constant is set to 1.

References
----------
.. [1] Bernard Ng, Gaël Varoquaux, Jean-Baptiste Poline, Michael D. Greicius,
       Bertrand Thirion, "Transport on Riemannian Manifold for Connectivity-based
       brain decoding", IEEE Transactions on Medical Imaging, 2015.

Author: Dhaif BEKHA.


# TODO: code a function permutation_svc to estimate the null distribution without
# TODO: bootstrapping, because when sample size is too small, bootstrap with replacement fail.
"""

import numpy as np
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
import time
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from sklearn.multiclass import OneVsRestClassifier
from conpagnon.utils.array_operation import array_rebuilder
from conpagnon.utils.folders_and_files_management import save_object
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn.plotting import plot_connectome
import os
from conpagnon.plotting.display import plot_matrix


def timer(start, end):
    """Print measured time between two point in the code

    Parameters
    ----------
    start: float
        The start of the measure
    end: float
        The end of the measure

    """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def bootstrap_svc_(vectorized_connectivity_matrices, class_labels, bootstrap_number):
    """Fit Support Vector Machine with linear kernel on bootstrap sample
    """

    # Support Vector Machine with a linear kernel
    svc = LinearSVC()
    bootstrap_weights = []
    # indices of subjects number to bootstrap
    indices = np.arange(vectorized_connectivity_matrices.shape[0])
    for b in range(bootstrap_number):
        bootstrap_indices = np.random.choice(indices, size=vectorized_connectivity_matrices.shape[0],
                                             replace=True)
        bootstrap_matrices = vectorized_connectivity_matrices[bootstrap_indices, ...]
        bootstrap_class_label = class_labels[bootstrap_indices]
        # Fit SVC on bootstrap sample
        svc.fit(bootstrap_matrices, bootstrap_class_label)
        # Append classification weight for the current bootstrap
        bootstrap_weights.append(svc.coef_[0, ...])
    bootstrap_weights = np.array(bootstrap_weights)
    # Compute the normalized mean over bootstrap
    bootstrap_normalized_mean = (bootstrap_weights.mean(axis=0))/(bootstrap_weights.std(axis=0))

    return bootstrap_normalized_mean


def bootstrap_classification(features, class_labels, boot_indices, C=1):
    """Perform classification on two binary class for each sample generated
    by bootstrap (with replacement).

    Parameters
    ----------
    features: numpy.ndarray, shape (n_samples, n_features)
        The connectivity matrices in a vectorized form, that is
        each row is a subjects and each column is a pair of regions. Only
        the lower part of connectivity matrices should be given.
    class_labels: numpy.ndarray, shape (n_samples, )
        The class labels of each subjects, permuted one time.
    boot_indices: numpy.ndarray, shape (n_samples, )
        The array containing the indices of bootstrapped
        subjects.

    Returns
    -------
    output: nunmpy.ndarray, shape (n_features, )
        The weight of the linear SVM estimated on the boostrap sample.
    """

    svc = LinearSVC(C=C)

    bootstrap_matrices = features[boot_indices, ...]
    bootstrap_class_label = class_labels[boot_indices]
    # Fit SVC on bootstrap sample
    svc.fit(bootstrap_matrices, bootstrap_class_label)
    # Weight of features for the bootstrap sample
    bootstrap_coefficients = svc.coef_[0, ...]

    return bootstrap_coefficients


def bootstrap_svc(features, class_labels,
                  bootstrap_array_indices,
                  n_cpus_bootstrap=1,
                  verbose=0,
                  backend='multiprocessing',
                  C=1):
    """Perform classification between two binary class on
    bootstrapped samples.

    Parameters
    ----------
    features: numpy.ndarray, shape (n_samples, n_features)
        The connectivity matrices in a vectorized form, that is
        each row is a subjects and each column is a pair of regions. Only
        the lower part of connectivity matrices should be given.
    class_labels: numpy.ndarray, shape (n_sample, )
        The class labels of each subjects.
    bootstrap_array_indices: numpy.ndarray, shape (n_bootstrap, n_features)
        A array containing the bootstrapped indices. Each row
        contain the indices to generate a bootstrapped sample.
    n_cpus_bootstrap: int, optional
        The number CPU to be used concurrently during computation on
        bootstrap sample.  Default is one, like a classical
        for loop over bootstrap sample.
    backend: str, optional
        The method used to execute concurrent task. This argument
        is passed to the Parallel function in the joblib package.
        Default is multiprocessing.
    verbose: int, optional
        The verbosity level during parallel computation. This
        argument is passed to Parallel function in the joblib package.

    Returns
    -------
    output: numpy.ndarray, shape (n_bootstrap, n_features)
        The array of estimated features weights, for each
        bootstrapped sample.
    """
    bootstrap_number = bootstrap_array_indices.shape[0]
    results_bootstrap = Parallel(
        n_jobs=n_cpus_bootstrap, verbose=verbose,
        backend=backend)(delayed(bootstrap_classification)(
                            features=features,
                            class_labels=class_labels,
                            boot_indices=bootstrap_array_indices[b, ...],
                            C=C)
                         for b in range(bootstrap_number))

    return np.array(results_bootstrap)


def permutation_bootstrap_svc(features, class_labels_perm,
                              bootstrap_array_perm,
                              n_permutations=1000,
                              n_cpus_bootstrap=1,
                              backend='multiprocessing',
                              verbose_bootstrap=0,
                              verbose_permutation=0,
                              C=1):
    """Perform classification on two binary class for each sample generated
    by bootstrap (with replacement) and permuted class labels vector.

    Parameters
    ----------
    features: numpy.ndarray, shape (n_samples, n_features)
        The connectivity matrices in a vectorized form, that is
        each row is a subjects and each column is a pair of regions. Only
        the lower part of connectivity matrices should be given.
    class_labels_perm: numpy.ndarray, shape (n_permutations, n_samples)
        The class labels array: each row contain the subjects labels
        permuted one time.
    n_permutations: int, optional
        The number of permutations. Default is 1000.
    bootstrap_array_perm:,numpy.ndarray, shape (n_permutations, n_bootstrap, n_samples)
        A array which contain a number of bootstrap array indices for each permutations.
    n_cpus_bootstrap: int, optional
        The number CPU to be used concurrently during computation
        on bootstrap sample. Default is one, like a classical
        for loop over bootstrap sample.
    backend: str, optional
        The method used to execute concurrent task. This argument
        is passed to the Parallel function in the joblib package.
        Default is multiprocessing.
    verbose_bootstrap: int, optional
        The verbosity level during parallel computation. This
        argument is passed to Parallel function in the joblib package.
    verbose_permutation: int, optional
        If equal to 1, print the progression of the permutations testing.
        Default is 1.

    Returns
    -------
    output: numpy.ndarray, shape (n_features, )
        The normalized features weights mean, estimated by classification
        with a linear SVM, over bootstrap sample
    """

    # Initialization of the normalized mean features weight over
    # N permutations
    null_distribution = np.zeros((n_permutations, features.shape[1]))

    tic_permutations = time.time()
    for n in range(n_permutations):
        if verbose_permutation == 1:
            print('Performing permutation number {} out of {}'.format(n, n_permutations))
        # Generate a bootstrap array indices for the current permutations
        # Perform the classification of each bootstrap sample, but with the labels shuffled
        bootstrap_weight_perm = bootstrap_svc(features=features,
                                              class_labels=class_labels_perm[n, ...],
                                              bootstrap_array_indices=bootstrap_array_perm[n, ...],
                                              verbose=verbose_bootstrap,
                                              backend=backend,
                                              n_cpus_bootstrap=n_cpus_bootstrap,
                                              C=C)
        # Compute the normalized mean of weight for the current permutation
        normalized_mean_weight_perm = bootstrap_weight_perm.mean(axis=0) / bootstrap_weight_perm.std(axis=0)
        # Save it in the null distribution array
        null_distribution[n, ...] = normalized_mean_weight_perm
    tac_permutations = time.time() - tic_permutations
    print('Elapsed time for {} permutations: {} min'.format(n_permutations, tac_permutations/60))

    return null_distribution


def one_against_all_classification(features, class_labels, boot_indices):
    """Perform multi-class classification problem on a bootstrapped
    sample with a one versus all strategy.

    Parameters
    ----------
    features: numpy.ndarray, shape (n_samples, n_features)
        The connectivity matrices in a vectorized form, that is
        each row is a subjects and each column is a pair of regions. Only
        the lower part of connectivity matrices should be given.
    class_labels: numpy.ndarray, shape (n_samples, )
        The class labels of each subjects, permuted one time.
    boot_indices: numpy.ndarray, shape (n_samples, )
        The array containing the indices of bootstrapped
        subjects.

    Returns
    -------
    output: nunmpy.ndarray, shape (n_features, )
        The weight of the linear SVM estimated on the boostrap sample.
    """
    svc = LinearSVC()
    one_vs_all = OneVsRestClassifier(estimator=svc)

    bootstrap_matrices = features[boot_indices, ...]
    bootstrap_class_label = class_labels[boot_indices]
    # Fit SVC on bootstrap sample using a one class vs all classifier
    one_vs_all.fit(bootstrap_matrices, bootstrap_class_label)
    # Weight of features for the bootstrap sample for each classes
    bootstrap_coefficients = one_vs_all.coef_

    return bootstrap_coefficients


def one_against_all_bootstrap(features, class_labels,
                              bootstrap_array_indices,
                              n_cpus_bootstrap=1,
                              verbose=0,
                              backend='multiprocessing'):
    """Perform classification between two binary class on
    bootstrapped samples.

    Parameters
    ----------
    features: numpy.ndarray, shape (n_samples, n_features)
        The connectivity matrices in a vectorized form, that is
        each row is a subjects and each column is a pair of regions. Only
        the lower part of connectivity matrices should be given.
    class_labels: numpy.ndarray, shape (n_sample, )
        The class labels of each subjects.
    bootstrap_array_indices: numpy.ndarray, shape (n_bootstrap, n_features)
        A array containing the bootstrapped indices. Each row
        contain the indices to generate a bootstrapped sample.
    n_cpus_bootstrap: int, optional
        The number CPU to be used concurrently during computation on
        bootstrap sample.  Default is one, like a classical
        for loop over bootstrap sample.
    backend: str, optional
        The method used to execute concurrent task. This argument
        is passed to the Parallel function in the joblib package.
        Default is multiprocessing.
    verbose: int, optional
        The verbosity level during parallel computation. This
        argument is passed to Parallel function in the joblib package.

    Returns
    -------
    output: numpy.ndarray, shape (n_bootstrap, n_features)
        The array of estimated features weights, for each
        bootstrapped sample.
    """
    bootstrap_number = bootstrap_array_indices.shape[0]
    results_bootstrap = Parallel(
        n_jobs=n_cpus_bootstrap, verbose=verbose,
        backend=backend)(delayed(one_against_all_classification)(
                            features=features,
                            class_labels=class_labels,
                            boot_indices=bootstrap_array_indices[b, ...])
                         for b in range(bootstrap_number))

    return np.array(results_bootstrap)


def one_against_all_permutation_bootstrap(features, class_labels_perm,
                                          bootstrap_array_perm,
                                          n_classes,
                                          n_permutations=1000,
                                          n_cpus_bootstrap=1,
                                          backend='multiprocessing',
                                          verbose_bootstrap=0,
                                          verbose_permutation=0):
    """Perform classification on two binary class for each sample generated
    by bootstrap (with replacement) and permuted class labels vector.

    Parameters
    ----------
    features: numpy.ndarray, shape (n_samples, n_features)
        The connectivity matrices in a vectorized form, that is
        each row is a subjects and each column is a pair of regions. Only
        the lower part of connectivity matrices should be given.
    class_labels_perm: numpy.ndarray, shape (n_permutations, n_samples)
        The class labels array: each row contain the subjects labels
        permuted one time.
    n_permutations: int, optional
        The number of permutations. Default is 1000.
    bootstrap_array_perm:,numpy.ndarray, shape (n_permutations, n_bootstrap, n_samples)
        A array which contain a number of bootstrap array indices for each permutations.
    n_cpus_bootstrap: int, optional
        The number CPU to be used concurrently during computation
        on bootstrap sample. Default is one, like a classical
        for loop over bootstrap sample.
    backend: str, optional
        The method used to execute concurrent task. This argument
        is passed to the Parallel function in the joblib package.
        Default is multiprocessing.
    verbose_bootstrap: int, optional
        The verbosity level during parallel computation. This
        argument is passed to Parallel function in the joblib package.
    verbose_permutation: int, optional
        If equal to 1, print the progression of the permutations testing.
        Default is 1.

    Returns
    -------
    output: numpy.ndarray, shape (n_features, )
        The normalized features weights mean, estimated by classification
        with a linear SVM, over bootstrap sample
    """

    null_distribution = np.zeros((n_permutations, n_classes, features.shape[1]))

    tic_permutations = time.time()
    for n in range(n_permutations):
        if verbose_permutation == 1:
            print('Performing permutation number {} out of {}'.format(n, n_permutations))
        # Generate a bootstrap array indices for the current permutations
        # Perform the classification of each bootstrap sample, but with the labels shuffled
        bootstrap_weight_perm = one_against_all_bootstrap(features=features,
                                              class_labels=class_labels_perm[n, ...],
                                              bootstrap_array_indices=bootstrap_array_perm[n, ...],
                                              verbose=verbose_bootstrap,
                                              backend=backend,
                                              n_cpus_bootstrap=n_cpus_bootstrap)
        # Compute the normalized mean of weight for the current permutation
        normalized_mean_weight_perm = bootstrap_weight_perm.mean(axis=0) / bootstrap_weight_perm.std(axis=0)
        # Save it in the null distribution array
        null_distribution[n, ...] = normalized_mean_weight_perm
    tac_permutations = time.time() - tic_permutations
    print('Elapsed time for {} permutations: {} min'.format(n_permutations, tac_permutations/60))

    return null_distribution


def k_largest_index_argsort(arr, k, reverse_order=False):
    """Returns the k+1 largest element indices in a an array

    Parameters
    ----------
    arr: numpy.ndarray
        A multi-dimensional array.
    k: int
        The number of largest elements indices to return.
    reverse_order: bool
        If True, the indices are returned from the largest to
        smallest element.

    Returns
    -------
        output: numpy.ndarray
            The array of the k+1 largest element indices.
            The shape is the same of the input array.

    """
    idx = np.argsort(arr.ravel())[:-k - 1:-1]
    if reverse_order:
        idx = idx[::-1]
    return np.column_stack(np.unravel_index(idx, arr.shape))


def k_smallest_index_argsort(arr, k, reverse_order=False):
    """Returns the k+1 smallest element indices in a an array

    Parameters
    ----------
    arr: numpy.ndarray
        A multi-dimensional array.
    k: int
        The number of smallest elements indices to return.
    reverse_order: bool
        If True, the indices are returned from the largest to
        smallest element.

    Returns
    -------
        output: numpy.ndarray
            The array of the k+1 smallest element indices.
            The shape is the same of the input array.
    """

    idx = np.argsort(arr.ravel())[:k + 1:1]
    if reverse_order:
        idx = idx[::-1]
    return np.column_stack(np.unravel_index(idx, arr.shape))


def remove_reversed_duplicates(iterable):
    # Create a set for already seen elements
    seen = set()
    for item in iterable:
        # Lists are mutable so we need tuples for the set-operations.
        tup = tuple(item)
        if tup not in seen:
            # If the tuple is not in the set append it in REVERSED order.
            seen.add(tup[::-1])
            # If you also want to remove normal duplicates uncomment the next line
            # seen.add(tup)
            yield item


def rank_top_features_weight(coefficients_array, top_features_number,
                             features_labels):
    """Establish a ranking of the most important features from the classifier, based
    on weights magnitude.

    Parameters
    ----------
    coefficients_array: numpy.ndarray, shape (n_features, n_features)
        The 2D array containing the features weights to rank.
    top_features_number: int
        The desired number of top features.
    features_labels: list
        The features labels list

    Returns
    -------
    output 1: numpy.ndarray, shape (n_top_features + 1, )
        The desired top features weights
    output 2: numpy.ndarray, shape (n_top_features +1, 2)
        The positions of each top features weights
    output 3: numpy.ndarray, shape (n_top_features +1, 2)
        The roi labels couple corresponding to top features.
    """

    # Find the k largest/smallest index couple coefficients in the array
    top_positive_coefficients_indices = k_largest_index_argsort(arr=coefficients_array, k=top_features_number,
                                                                reverse_order=True)

    top_negative_coefficients_indices = k_smallest_index_argsort(arr=coefficients_array, k=top_features_number,
                                                                 reverse_order=False)
    # Stack the k largest/smallest couple coefficient vector array
    top_coefficients_indices = np.vstack([top_negative_coefficients_indices,
                                          top_positive_coefficients_indices])

    # In the top coefficient stack, we have symmetric pair, due to symmetric connectivity
    # behavior, we only keep one pairs
    top_coefficients_indices_ = np.array(list(remove_reversed_duplicates(top_coefficients_indices)))

    # We construct an array of feature name based on top_coefficients_ couple index
    features_labels_couple = np.array([features_labels[top_coefficients_indices_[i]]
                                       for i in range(top_coefficients_indices_.shape[0])])

    # Fetch the weight in the normalized mean array based on indices stacked
    # in top_coefficients_indices
    top_weights = coefficients_array[top_coefficients_indices_[:, 0], top_coefficients_indices_[:, 1]]

    return top_weights, top_coefficients_indices_, features_labels_couple


def features_weights_max_t_correction(null_distribution_features_weights,
                                      normalized_mean_weight):
    """Compute features weight corrected p values with
    the estimated null distribution of normalized mean features weight.

    Parameters
    ----------
    null_distribution_features_weights: numpy.ndarray, shape (n_permutations, (n_features*(n_features - 1)/2))
        The estimated null distribution of normalized mean of  features weights
        estimated with class labels permutations, and bootstrap.
    normalized_mean_weight: numpy.ndarray, shape ( (n_features*(n_features - 1)/2), )
        The normalized mean of features weight estimated on bootstrapped sample.

    Returns
    -------
    output 1: numpy.ndarray, shape (
    """
    # Retrieve the number of permutation and features
    n_permutations = null_distribution_features_weights.shape[0]
    n_features = null_distribution_features_weights.shape[1]

    null_min_and_max_distribution = np.zeros((n_permutations, 2))

    # Find minimum and maximum weight in the normalized mean for each permutations
    null_min_and_max_distribution[:, 0], null_min_and_max_distribution[:, 1] = \
        null_distribution_features_weights.min(axis=1), \
        null_distribution_features_weights.max(axis=1)

    # Compare each edges weight from the true mean weight normalized distribution to the minimum and
    # maximum null estimated distribution.

    # null distribution for maximum and minimum normalized weight
    sorted_null_maximum_dist = sorted(null_min_and_max_distribution[:, 1])
    sorted_null_minimum_dist = sorted(null_min_and_max_distribution[:, 0])

    # p values array
    p_values_max = np.zeros(n_features)
    p_values_min = np.zeros(n_features)

    for feature in range(normalized_mean_weight.shape[0]):
        p_values_max[feature] = \
            (len(np.where(sorted_null_maximum_dist > normalized_mean_weight[feature])[0]) / (n_permutations + 1))
        p_values_min[feature] = \
            (len(np.where(sorted_null_minimum_dist < normalized_mean_weight[feature])[0]) / (n_permutations + 1))

    return sorted_null_maximum_dist, sorted_null_minimum_dist, p_values_max, p_values_min


def features_weights_parametric_correction(null_distribution_features_weights,
                                           normalized_mean_weight,
                                           method='fdr_bh',
                                           alpha=.05):
    """Parametric estimation of p-values for each features weight using the estimated
    null distribution and fitting it's mean and standard deviation with normal law.

    Parameters
    ----------
    null_distribution_features_weights: numpy.ndarray of shape (n_permutation,  (n_features*(n_features - 1)/2))
        The normalized mean features weights for each permutations.
    normalized_mean_weight: numpy.ndarray, shape ( (n_features*(n_features - 1)/2), )
        The estimated normalized mean features weights from bootstrapped samples.
    method: str, optional
        The correction method. There are multiple possible choices, please
        consults the statsmodels library. Default is the False Discovery Rate
        correction (FDR).
    alpha: float, optional
        The type I error rate threshold. Default is 0.05.

    Returns
    -------
    output: numpy.ndarray, shape ((n_features*(n_features - 1)/2), )
        The corrected p-values array.

    """

    # Compute the p-values in a parametric way
    # The mean normalized features weight over all permutation
    mean_normalized_weight_perm = null_distribution_features_weights.mean(axis=0)
    # the standard deviation of the mean normalized weight over all permutation
    std_normalized_weight_perm = null_distribution_features_weights.std(axis=0)
    # p values array initialisation
    p_values_array = np.zeros(normalized_mean_weight.shape[0])
    # We compute for each features weight the corresponding
    # p values.
    for j in range(normalized_mean_weight.shape[0]):
        p_values_array[j] = 2*min(1-norm.cdf(x=normalized_mean_weight[j],
                                             loc=mean_normalized_weight_perm[j],
                                             scale=std_normalized_weight_perm[j]),
                                  norm.cdf(x=normalized_mean_weight[j],
                                           loc=mean_normalized_weight_perm[j],
                                           scale=std_normalized_weight_perm[j]))
    # Account with multiple comparison
    reject, p_values_corrected, _, _ = multipletests(pvals=p_values_array,
                                                     alpha=alpha,
                                                     method=method)

    return p_values_corrected


def find_top_features(normalized_mean_weight_array, labels_regions, top_features_number=50):
    """Find the top features weight in the normalized mean weight array, and mask the other
    features weight outside the ranking.

    Parameters
    ----------
    normalized_mean_weight_array: numpy.ndarray shape (n_features,n_features)
        The array of each normalized mean feature weight, computed after bootstrapping.
    labels_regions: list
        The list of feature label.
    top_features_number: int
        The top features number to keep.

    Returns
    -------
    output 1: numpy.ndarray, shape(n_features, n_features)
        The normalized mean weight array containing the top features weight values
        and zero elsewhere.
    output 2: numpy.ndarray, shape(top_features_number + 1, )
        The top features weights.
    output 3: numpy.ndarray, shape(top_features_number + 1, 2)
        The indices of the top features in the normalized mean weights array.
    output 4: numpy.ndarray, shape(top_features_number + 1, 2)
        The labels of the top features in the normalized mean weights array.
    """

    n_nodes = normalized_mean_weight_array.shape[0]
    # Find the top features among the normalized mean weight distribution

    top_weights, top_coefficients_indices, top_weight_labels = rank_top_features_weight(
        coefficients_array=normalized_mean_weight_array,
        top_features_number=top_features_number,
        features_labels=labels_regions)

    # Plot the top features weight on glass brain
    top_weights_mask = np.zeros((n_nodes, n_nodes), dtype=bool)
    top_weights_mask[top_coefficients_indices[:, 0], top_coefficients_indices[:, 1]] = True
    normalized_mean_weight_array_top_features = np.multiply(normalized_mean_weight_array,
                                                            top_weights_mask)
    normalized_mean_weight_array_top_features += normalized_mean_weight_array_top_features.T

    return normalized_mean_weight_array_top_features, top_weights, top_coefficients_indices, top_weight_labels


def find_significant_features_indices(p_positive_features_significant,
                                      p_negative_features_significant,
                                      features_labels):
    """Return regions indices and corresponding labels for s
    surviving features after permutation testing, for both
    negative and positive features weight.

    Parameters
    ----------
    p_positive_features_significant: numpy.ndarray, shape (n_features, n_features)
        An array containing the weight for a associated significant p-values features for
        positive weights, and zero elsewhere.
    p_negative_features_significant: numpy.ndarray, shape (n_features, n_features)
        An array containing the weight for a associated significant p-values features for
        negative weights, and zero elsewhere.
    features_labels: numpy.ndarray, shape (n_features, )
        The features labels.

    Returns
    -------
    output 1: numpy.ndarray, shape (n_significant_features, 2)
        The indices array of significant positive weighted features.
    output 2: numpy.ndarray, shape (n_significant_features, 2)
        The indices array of significant negative weighted features.
    output 3: numpy.ndarray, shape (n_significant_features, 2)
        The labels array of significant positive weighted features.
    output 4: numpy.ndarray, shape (n_significant_features, 2)
        The labels array of significant negative weighted features.

    """

    significant_positive_features_indices = \
        np.array(list(remove_reversed_duplicates(np.array(np.where(p_positive_features_significant != 0)).T)))
    significant_positive_features_labels = \
        np.array([features_labels[significant_positive_features_indices[i]] for i
                  in range(significant_positive_features_indices.shape[0])])

    significant_negative_features_indices = \
        np.array(list(remove_reversed_duplicates(np.array(np.where(p_negative_features_significant != 0)).T)))

    significant_negative_features_labels = \
        np.array([features_labels[significant_negative_features_indices[i]] for i
                  in range(significant_negative_features_indices.shape[0])])

    return significant_positive_features_indices, significant_negative_features_indices, \
           significant_positive_features_labels, significant_negative_features_labels


def compute_weight_distribution(vectorized_connectivity_matrices,
                                bootstrap_number,
                                n_permutations,
                                class_labels,
                                C=1,
                                n_cpus_bootstrap=1,
                                verbose_bootstrap=1,
                                verbose_permutations=1):

    # Number of subjects
    n_subjects = vectorized_connectivity_matrices.shape[0]

    # Indices to bootstrap
    indices = np.arange(n_subjects)
    # Generate a matrix containing all bootstrapped indices
    bootstrap_matrix = np.random.choice(a=indices, size=(bootstrap_number, n_subjects),
                                        replace=True)

    # Generate a permuted class labels array
    class_labels_permutation_matrix = np.array([np.random.permutation(class_labels)
                                                for n in range(n_permutations)])

    bootstrap_array_perm = np.random.choice(a=indices,
                                            size=(n_permutations, bootstrap_number,
                                                  n_subjects),
                                            replace=True)

    # Sanity check fo bootstrapped sample class labels for each permutation
    print('Check Bootstrapped class labels for each permutation...')
    for n in range(n_permutations):
        for b in range(bootstrap_number):
            bootstrapped_permuted_labels = class_labels_permutation_matrix[n, bootstrap_array_perm[n, b, ...]]
            count_labels_occurence = len(np.unique(bootstrapped_permuted_labels))
            if count_labels_occurence == 2:
                pass
            else:
                print('For the permutation # {}, the bootstrapped samples # {} is ill...'.format(n, b))
                print(b)
                # We replace the problematic bootstrap
                new_bootstrap_indices = np.random.choice(a=indices, size=len(indices),
                                                         replace=True)
                bootstrap_array_perm[n, b, ...] = new_bootstrap_indices
                new_bootstrap_class_labels_permuted = class_labels_permutation_matrix[n, new_bootstrap_indices]
                class_labels_permutation_matrix[n, ...] = new_bootstrap_class_labels_permuted

    print('Verifying that bootstrapped sample labels classes contain two labels....')
    for n in range(n_permutations):
        for b in range(bootstrap_number):
            bootstrapped_permuted_labels = class_labels_permutation_matrix[n, bootstrap_array_perm[n, b, ...]]
            count_labels_occurence = len(np.unique(bootstrapped_permuted_labels))
            if count_labels_occurence < 2:
                raise ValueError('Sample size seems too small to generate clean bootstrapped class labels \n '
                                 'with at least two classes !')
            else:
                pass
    print('Done checking bootstrapped class labels for each permutation.')

    # True weight obtain by bootstrapping
    print('Performing classification on {} bootstrap sample...'.format(bootstrap_number))
    bootstrap_weight = bootstrap_svc(features=vectorized_connectivity_matrices,
                                     class_labels=class_labels,
                                     bootstrap_array_indices=bootstrap_matrix,
                                     n_cpus_bootstrap=n_cpus_bootstrap,
                                     verbose=verbose_bootstrap,
                                     C=C)
    # Estimation of null distribution of normalized mean weight
    null_distribution = permutation_bootstrap_svc(features=vectorized_connectivity_matrices,
                                                  class_labels_perm=class_labels_permutation_matrix,
                                                  bootstrap_array_perm=bootstrap_array_perm,
                                                  n_permutations=n_permutations,
                                                  n_cpus_bootstrap=n_cpus_bootstrap,
                                                  verbose_bootstrap=verbose_bootstrap,
                                                  verbose_permutation=verbose_permutations,
                                                  C=C)

    normalized_mean_weight = bootstrap_weight.mean(axis=0) / bootstrap_weight.std(axis=0)

    return normalized_mean_weight, null_distribution


def discriminative_brain_connection_identification(vectorized_connectivity_matrices, class_labels,
                                                   class_names, save_directory,
                                                   n_permutations, bootstrap_number,
                                                   features_labels, features_colors,
                                                   n_nodes,
                                                   atlas_nodes,
                                                   first_class_mean_matrix,
                                                   second_class_mean_matrix,
                                                   top_features_number=100,
                                                   correction='fdr_bh',
                                                   alpha=0.05,
                                                   n_cpus_bootstrap=1,
                                                   verbose_bootstrap=1,
                                                   verbose_permutations=1,
                                                   write_report=True,
                                                   node_size=15,
                                                   C=1):
    """Identify important connection when performing a binary classification task.
    """
    # Number of subjects
    n_subjects = vectorized_connectivity_matrices.shape[0]

    # report name for features visualisation and parameters text report
    report_filename = 'features_identification_' + class_names[0] + '_' + class_names[1] + '_' + str(alpha) + \
                      '_' + correction + '.pdf'
    text_report_filename = 'features_identification_' + class_names[0] + '_' + class_names[1] + '_' + str(alpha) + \
                           '_' + correction + '.txt'

    normalized_mean_weight, null_distribution = compute_weight_distribution(
        vectorized_connectivity_matrices=vectorized_connectivity_matrices,
        bootstrap_number=bootstrap_number,
        n_permutations=n_permutations,
        class_labels=class_labels,
        C=C,
        n_cpus_bootstrap=n_cpus_bootstrap,
        verbose_bootstrap=verbose_bootstrap,
        verbose_permutations=verbose_permutations
    )
    save_object(object_to_save=normalized_mean_weight, saving_directory=save_directory,
                filename='normalized_mean_weight_' + class_names[0] + '_' + class_names[1] + '.pkl')

    # Save the null distribution to avoid
    save_object(object_to_save=null_distribution, saving_directory=save_directory,
                filename='null_distribution_' + class_names[0] + '_' + class_names[1] + '.pkl')

    # Rebuild a symmetric array from normalized mean weight vector
    normalized_mean_weight_array = array_rebuilder(normalized_mean_weight,
                                                   'numeric', diagonal=np.zeros(n_nodes))
    # Find top features
    normalized_mean_weight_array_top_features, top_weights, top_coefficients_indices, top_weight_labels = \
        find_top_features(normalized_mean_weight_array=normalized_mean_weight_array,
                          labels_regions=features_labels)

    labels_str = []
    for i in range(top_weight_labels.shape[0]):
        labels_str.append(str(top_weight_labels[i]))

    if correction == 'max_t':
        # Corrected p values with the maximum statistic
        sorted_null_maximum_dist, sorted_null_minimum_dist, p_value_positive_weights, p_value_negative_weights = \
            features_weights_max_t_correction(null_distribution_features_weights=null_distribution,
                                              normalized_mean_weight=normalized_mean_weight)

        # Rebuild vectorized p values array
        p_max_values_array = array_rebuilder(vectorized_array=p_value_positive_weights,
                                             array_type='numeric',
                                             diagonal=np.ones(n_nodes))

        p_min_values_array = array_rebuilder(vectorized_array=p_value_negative_weights,
                                             array_type='numeric',
                                             diagonal=np.ones(n_nodes))

        # Find p-values under the alpha threshold
        p_negative_features_significant = np.array(p_min_values_array < alpha, dtype=int)
        p_positive_features_significant = np.array(p_max_values_array < alpha, dtype=int)

        significant_positive_features_indices, significant_negative_features_indices, \
        significant_positive_features_labels, significant_negative_features_labels = find_significant_features_indices(
            p_positive_features_significant=p_positive_features_significant,
            p_negative_features_significant=p_negative_features_significant,
            features_labels=features_labels)

        # Take the mean difference and masking all connection except the surviving ones for
        # surviving negative and positive features weight
        mean_difference_positive_mask = np.multiply(first_class_mean_matrix - second_class_mean_matrix,
                                                    p_positive_features_significant)
        mean_difference_negative_mask = np.multiply(first_class_mean_matrix - second_class_mean_matrix,
                                                    p_negative_features_significant)

        # Mask for all significant connection, for both,
        # positive and negative weight
        p_all_significant_features = p_positive_features_significant + p_negative_features_significant

        # Take the mean difference for the overall significant features mask
        mean_difference_all_significant_features_mask = np.multiply(
            first_class_mean_matrix - second_class_mean_matrix,
            p_all_significant_features)

        if write_report is True:

            with PdfPages(os.path.join(save_directory, report_filename)) as pdf:
                # Plot the estimated null distribution
                plt.figure(constrained_layout=True)
                plt.hist(sorted_null_maximum_dist, 'auto', histtype='bar', alpha=0.5,
                         edgecolor='black')
                # The five 5% extreme values among maximum distribution
                p95 = np.percentile(sorted_null_maximum_dist, q=95)
                plt.axvline(x=p95, color='black')
                plt.title('Null distribution of maximum normalized weight mean')
                pdf.savefig()

                plt.figure()
                plt.hist(sorted_null_minimum_dist, 'auto', histtype='bar',
                         alpha=0.5, edgecolor='black')
                # The five 5% extreme values among minimum distribution
                p5 = np.percentile(sorted_null_minimum_dist, q=5)
                plt.axvline(x=p5, color='black')
                plt.title('Null distribution of minimum normalized weight mean')
                pdf.savefig()

                plt.figure()
                plot_connectome(adjacency_matrix=normalized_mean_weight_array,
                                node_coords=atlas_nodes, colorbar=True,
                                title='Features weight', node_size=node_size,
                                node_color=features_colors)
                pdf.savefig()

                # plot the top weight in a histogram fashion
                fig = plt.figure(figsize=(15, 10), constrained_layout=True)

                weight_colors = ['blue' if weight < 0 else 'red' for weight in top_weights]
                plt.bar(np.arange(len(top_weights)), list(top_weights),
                        color=weight_colors,
                        edgecolor='black',
                        alpha=0.5)
                plt.xticks(np.arange(0, len(top_weights)), labels_str,
                           rotation=60,
                           ha='right')
                for label in range(len(plt.gca().get_xticklabels())):
                    plt.gca().get_xticklabels()[label].set_color(weight_colors[label])
                plt.xlabel('Features names')
                plt.ylabel('Features weights')
                plt.title('Top {} features ranking of normalized mean weight'.format(top_features_number))
                pdf.savefig()

                plt.figure()
                plot_connectome(adjacency_matrix=normalized_mean_weight_array_top_features,
                                node_coords=atlas_nodes,
                                colorbar=True,
                                title='Top {} features weight'.format(top_features_number),
                                node_size=node_size,
                                node_color=features_colors)
                pdf.savefig()

                if np.where(p_all_significant_features == 1)[0].size != 0:
                    # Plot on glass brain the significant positive features weight
                    plt.figure()
                    plot_connectome(adjacency_matrix=p_positive_features_significant,
                                    node_coords=atlas_nodes, colorbar=True,
                                    title='Significant positive weight', edge_cmap='Reds',
                                    node_size=node_size,
                                    node_color=features_colors)
                    pdf.savefig()

                    # Plot on glass brain the mean difference in connectivity between
                    # the two groups for surviving positive weight
                    plt.figure()
                    plot_connectome(adjacency_matrix=mean_difference_positive_mask,
                                    node_coords=atlas_nodes, colorbar=True,
                                    title='Difference in connectivity {} - {} (positive weights)'.format(
                                        class_names[0],
                                        class_names[1]),
                                    node_size=node_size,
                                    node_color=features_colors)
                    pdf.savefig()

                    # Plot on glass brain the significant negative features weight
                    plt.figure()
                    plot_connectome(adjacency_matrix=p_negative_features_significant,
                                    node_coords=atlas_nodes, colorbar=True,
                                    title='Significant negative weight', edge_cmap='Blues',
                                    node_size=node_size,
                                    node_color=features_colors)
                    pdf.savefig()

                    plt.figure()
                    plot_connectome(adjacency_matrix=mean_difference_negative_mask,
                                    node_coords=atlas_nodes, colorbar=True,
                                    title='Difference in connectivity {} - {} (negative weights)'.format(
                                        class_names[0],
                                        class_names[1]),
                                    node_size=node_size,
                                    node_color=features_colors)
                    pdf.savefig()

                    plt.figure()
                    plot_connectome(adjacency_matrix=mean_difference_all_significant_features_mask,
                                    node_coords=atlas_nodes, colorbar=True,
                                    title='Difference in connectivity {} - {} (all weights)'.format(
                                        class_names[0],
                                        class_names[1]),
                                    node_size=node_size,
                                    node_color=features_colors)
                    pdf.savefig()

                    # Matrix view of significant positive and negative weight
                    plt.figure()
                    plot_matrix(matrix=p_negative_features_significant,
                                labels_colors='auto', mpart='all',
                                colormap='Blues', linecolor='black',
                                title='Significant negative weight',
                                vertical_labels=features_labels,
                                horizontal_labels=features_labels)
                    pdf.savefig()

                    plt.figure()
                    plot_matrix(matrix=p_positive_features_significant,
                                labels_colors='auto', mpart='all',
                                colormap='Reds', linecolor='black',
                                title='Significant positive weight',
                                vertical_labels=features_labels,
                                horizontal_labels=features_labels)
                    pdf.savefig()

                    plt.close("all")
        else:
            pass

    else:

        # Perform another type of correction like FDR, ....
        p_values_corrected = features_weights_parametric_correction(
            null_distribution_features_weights=null_distribution,
            normalized_mean_weight=normalized_mean_weight,
            method=correction)

        p_values_corrected_array = array_rebuilder(vectorized_array=p_values_corrected,
                                                   array_type='numeric',
                                                   diagonal=np.ones(n_nodes))

        # Find p values under alpha threshold for negative and positive weight features
        p_negative_features_significant = np.array(
            (p_values_corrected_array < alpha) & (normalized_mean_weight_array < 0),
            dtype=int)
        p_positive_features_significant = np.array(
            (p_values_corrected_array < alpha) & (normalized_mean_weight_array > 0),
            dtype=int)

        significant_positive_features_indices, significant_negative_features_indices, \
        significant_positive_features_labels, significant_negative_features_labels = find_significant_features_indices(
            p_positive_features_significant=p_positive_features_significant,
            p_negative_features_significant=p_negative_features_significant,
            features_labels=features_labels)

        # Take the mean difference and masking all connection except the surviving ones for
        # surviving negative and positive features weight
        mean_difference_positive_mask = np.multiply(first_class_mean_matrix - second_class_mean_matrix,
                                                    p_positive_features_significant)
        mean_difference_negative_mask = np.multiply(first_class_mean_matrix - second_class_mean_matrix,
                                                    p_negative_features_significant)

        # Mask for all significant connection, for both,
        # positive and negative weight
        p_all_significant_features = p_positive_features_significant + p_negative_features_significant

        # Take the mean difference for the overall significant features mask
        mean_difference_all_significant_features_mask = np.multiply(
            first_class_mean_matrix - second_class_mean_matrix,
            p_all_significant_features)
        if write_report is True:

            # plot the top weight in a histogram fashion
            with PdfPages(os.path.join(save_directory, report_filename)) as pdf:

                plt.figure()
                plot_connectome(adjacency_matrix=normalized_mean_weight_array,
                                node_coords=atlas_nodes, colorbar=True,
                                title='Features weight',
                                node_size=node_size,
                                node_color=features_colors)
                pdf.savefig()

                fig = plt.figure(figsize=(15, 10), constrained_layout=True)
                weight_colors = ['blue' if weight < 0 else 'red' for weight in top_weights]
                plt.bar(np.arange(len(top_weights)), list(top_weights),
                        color=weight_colors,
                        edgecolor='black',
                        alpha=0.5)
                plt.xticks(np.arange(0, len(top_weights)), labels_str,
                           rotation=60,
                           ha='right')
                for label in range(len(plt.gca().get_xticklabels())):
                    plt.gca().get_xticklabels()[label].set_color(weight_colors[label])
                plt.xlabel('Features names')
                plt.ylabel('Features weights')
                plt.title('Top {} features ranking of normalized mean weight'.format(top_features_number))
                pdf.savefig()

                plt.figure()
                plot_connectome(adjacency_matrix=normalized_mean_weight_array_top_features,
                                node_coords=atlas_nodes, colorbar=True,
                                title='Top {} features weight'.format(top_features_number),
                                node_size=node_size,
                                node_color=features_colors)
                pdf.savefig()
                if np.where(p_all_significant_features == 1)[0].size != 0:
                    # Plot on glass brain the significant positive features weight
                    plt.figure()
                    plot_connectome(adjacency_matrix=p_positive_features_significant,
                                    node_coords=atlas_nodes, colorbar=True,
                                    title='Significant positive weight after {} correction'.format(correction),
                                    edge_cmap='Reds',
                                    node_size=node_size,
                                    node_color=features_colors)

                    pdf.savefig()

                    # the two groups for surviving positive weight
                    plt.figure()
                    plot_connectome(adjacency_matrix=mean_difference_positive_mask,
                                    node_coords=atlas_nodes, colorbar=True,
                                    title='Difference in connectivity {} - {} (positive weights)'.format(
                                        class_names[0],
                                        class_names[1]),
                                    node_size=node_size,
                                    node_color=features_colors)
                    pdf.savefig()

                    # Plot on glass brain the significant negative features weight
                    plt.figure()
                    plot_connectome(adjacency_matrix=p_negative_features_significant,
                                    node_coords=atlas_nodes, colorbar=True,
                                    title='Significant negative weight after {} correction'.format(correction),
                                    edge_cmap='Blues',
                                    node_size=node_size,
                                    node_color=features_colors)
                    pdf.savefig()

                    plt.figure()
                    plot_connectome(adjacency_matrix=mean_difference_negative_mask,
                                    node_coords=atlas_nodes, colorbar=True,
                                    title='Mean difference in connectivity {} - {} (negative weights)'.format(
                                        class_names[0],
                                        class_names[1]),
                                    node_size=node_size,
                                    node_color=features_colors)
                    pdf.savefig()

                    plt.figure()
                    plot_connectome(adjacency_matrix=mean_difference_all_significant_features_mask,
                                    node_coords=atlas_nodes, colorbar=True,
                                    title='Difference in connectivity {} - {} (all weights)'.format(
                                        class_names[0],
                                        class_names[1]),
                                    node_size=node_size,
                                    node_color=features_colors)
                    pdf.savefig()

                    # Matrix view of significant positive and negative weight
                    plt.figure()
                    plot_matrix(matrix=p_negative_features_significant,
                                labels_colors=features_colors, mpart='all',
                                colormap='Blues', linecolor='black',
                                title='Significant negative weight after {} correction'.format(correction),
                                vertical_labels=features_labels, horizontal_labels=features_labels)
                    pdf.savefig()

                    plt.figure()
                    plot_matrix(matrix=p_positive_features_significant,
                                labels_colors=features_colors,
                                mpart='all',
                                colormap='Reds', linecolor='black',
                                title='Significant positive weight after {} correction'.format(correction),
                                vertical_labels=features_labels, horizontal_labels=features_labels)
                    pdf.savefig()

                    plt.close("all")
        else:
            pass

    # Write a small report in a text file
    with open(os.path.join(save_directory, text_report_filename), 'w') as output_results:
        # Write parameters
        output_results.write('------------ Parameters ------------')
        output_results.write('\n')
        output_results.write('Number of subjects: {}'.format(n_subjects))
        output_results.write('\n')
        output_results.write('Groups: {}'.format(class_names))
        output_results.write('\n')
        output_results.write('Groups labels: {}'.format(class_labels))
        output_results.write('\n')
        output_results.write('Bootstrap number: {}'.format(bootstrap_number))
        output_results.write('\n')
        output_results.write('Number of permutations: {}'.format(n_permutations))
        output_results.write('\n')
        output_results.write('Alpha threshold: {}'.format(alpha))
        output_results.write('\n')
        output_results.write('Regularization parameters C: {}'.format(C))
        output_results.write('\n')
        output_results.write('\n')
        output_results.write('------------ Results ------------')
        output_results.write('\n')
        output_results.write('------------ Discriminative connections for negative features weight ------------')
        output_results.write('\n')
        # Write the labels of features with negative weight identified
        for negative_feature in range(len(significant_negative_features_labels)):
            output_results.write('\n')
            output_results.write('{} <-> {}, (indices: {} <-> {})'.format(
                significant_negative_features_labels[negative_feature][0],
                significant_negative_features_labels[negative_feature][1],
                significant_negative_features_indices[negative_feature][0],
                significant_negative_features_indices[negative_feature][1]))

        output_results.write('\n')
        output_results.write('\n')
        output_results.write('------------ Discriminative connections for positive features weight ------------')
        output_results.write('\n')
        # Write the labels of features with negative weight identified
        for positive_feature in range(len(significant_positive_features_labels)):
            output_results.write('\n')
            output_results.write('{} <-> {}, (indices: {} <-> {})'.format(
                significant_positive_features_labels[positive_feature][0],
                significant_positive_features_labels[positive_feature][1],
                significant_positive_features_indices[positive_feature][0],
                significant_positive_features_indices[positive_feature][1]))

    return normalized_mean_weight, null_distribution, p_all_significant_features











