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
    by bootstrap (with replacement) and class labels permuted one time.

    """

    svc = LinearSVC()

    bootstrap_matrices = features[boot_indices, ...]
    bootstrap_class_label = class_labels[boot_indices]
    # Fit SVC on bootstrap sample
    svc.fit(bootstrap_matrices, bootstrap_class_label)
    # Weight of features for the bootstrap sample
    bootstrap_coefficients = svc.coef_[0, ...]

    return bootstrap_coefficients


def bootstrap_svc(features, class_labels, bootstrap_array_indices, bootstrap_number, n_cpus_bootstrap=1,
                  verbose=0, backend='multiprocessing'):

    results_bootstrap = Parallel(n_jobs=n_cpus_bootstrap, verbose=verbose,
                                 backend=backend)(delayed(bootstrap_classification)(
                                    features=features,
                                    class_labels=class_labels,
                                    boot_indices=bootstrap_array_indices[b, ...]) for b in range(bootstrap_number))

    return np.array(results_bootstrap)


def permutation_bootstrap_svc(features, class_labels_perm, indices, bootstrap_number=500,
                              n_cpus_bootstrap=1, verbose=0, backend='multiprocessing'):
    """Perform classification on two binary class for each sample generated
    by bootstrap (with replacement) and permuted class labels vector.
    """

    # Number of samples
    n_subjects = features.shape[0]
    # Permute the class labels
    # Build a array shape (n_bootstrap, n_subjects) containing bootstrap indices
    bootstrap_matrix_perm = np.random.choice(a=indices, size=(bootstrap_number, n_subjects), replace=True)
    # Perform classification on each bootstrap samples, but with labels permuted
    bootstrap_weight_perm = bootstrap_svc(features=features, class_labels=class_labels_perm,
                                          bootstrap_array_indices=bootstrap_matrix_perm,
                                          bootstrap_number=bootstrap_number, n_cpus_bootstrap=n_cpus_bootstrap,
                                          verbose=verbose, backend=backend)

    return bootstrap_weight_perm


def null_distribution_classifier_weight(features, class_labels_perm_matrix, indices, bootstrap_number=100,
                                        n_permutations=500, n_cpus_permutations=1, n_cpus_bootstrap=1,
                                        verbose_bootstrap=1, verbose_permutations=1, joblib_tmp_folder='/tmp'):

    results_permutations_bootstrap = \
        Parallel(n_jobs=n_cpus_permutations, backend="multiprocessing",
                 verbose=verbose_permutations, temp_folder=joblib_tmp_folder)(delayed(permutation_bootstrap_svc)(
                    features=features,
                    class_labels_perm=class_labels_perm_matrix[n, ...],
                    indices=indices,
                    bootstrap_number=bootstrap_number,
                    n_cpus_bootstrap=n_cpus_bootstrap,
                    verbose=verbose_bootstrap) for n in range(n_permutations))

    # The results is an array of shape (n_permutations, n_bootstrap, n_features), we compute
    # mean over bootstrap sample for each permutations
    results_permutations_bootstrap = np.array(results_permutations_bootstrap)
    null_distribution = results_permutations_bootstrap.mean(axis=1)/results_permutations_bootstrap.std(axis=1)

    return null_distribution


def k_largest_index_argsort(a, k, reverse_order=False):
    """Returns the k+1 largest element in a an array
    """
    idx = np.argsort(a.ravel())[:-k - 1:-1]
    if reverse_order:
        idx = idx[::-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def k_smallest_index_argsort(a, k, reverse_order=False):
    idx = np.argsort(a.ravel())[:k + 1:1]
    if reverse_order:
        idx = idx[::-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


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
    top_positive_coefficients_indices = k_largest_index_argsort(a=coefficients_array,
                                                                k=top_features_number,
                                                                reverse_order=True)

    top_negative_coefficients_indices = k_smallest_index_argsort(a=coefficients_array,
                                                                 k=top_features_number,
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
