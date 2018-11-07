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
                              bootstrap_array_perm,
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
                                              n_cpus_bootstrap=n_cpus_bootstrap)
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





