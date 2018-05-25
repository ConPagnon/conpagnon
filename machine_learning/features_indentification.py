import numpy as np
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed


def bootstrap_SVC(vectorized_connectivity_matrices, class_labels, bootstrap_number):
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
                                        n_permutations=500, n_cpus_permutations=1, n_cpus_bootstrap=1):

    results_permutations_bootstrap = \
        Parallel(n_jobs=n_cpus_permutations, verbose=100, backend="multiprocessing")(delayed(permutation_bootstrap_svc)(
            features=features,
            class_labels_perm=class_labels_perm_matrix[n, ...],
            indices=indices,
            bootstrap_number=bootstrap_number,
            n_cpus_bootstrap=n_cpus_bootstrap) for n in range(n_permutations))

    return np.array(results_permutations_bootstrap)