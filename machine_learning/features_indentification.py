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
"""

import numpy as np
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
import time
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


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


def bootstrap_classification(features, class_labels, boot_indices):
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

    svc = LinearSVC()

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
        backend=backend)(delayed(bootstrap_classification)(
                            features=features,
                            class_labels=class_labels,
                            boot_indices=bootstrap_array_indices[b, ...])
                         for b in range(bootstrap_number))

    return np.array(results_bootstrap)


def permutation_bootstrap_svc(features, class_labels_perm,
                              n_permutations=1000,
                              bootstrap_number=500,
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
    bootstrap_number: int, optional
        The number of bootstrap sample to generate. Defaults is 500.
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
    # indices of subjects for bootstrapping
    indices = np.arange(features.shape[0])

    tic_permutations = time.time()
    for n in range(n_permutations):
        if verbose_permutation == 1:
            print('Performing permutation number {} out of {}'.format(n, n_permutations))
        # Generate a bootstrap array indices for the current permutations
        bootstrap_indices_perm = np.random.choice(a=indices, size=(bootstrap_number, features.shape[0]),
                                                  replace=True)
        # Perform the classification of each bootstrap sample, but with the labels shuffled
        bootstrap_weight_perm = bootstrap_svc(features=features,
                                              class_labels=class_labels_perm[n, ...],
                                              bootstrap_array_indices=bootstrap_indices_perm,
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
    null_distribution_features_weights: numpy.ndarray, shape (n_permutations, n_features)
        The estimated null distribution of normalized mean of  features weights
        estimated with class labels permutations, and bootstrap.
    normalized_mean_weight: numpy.ndarray, shape (n_features, )
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
