import numpy as np
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
# todo: make helper function: one for the bootstraping which take a set of connectivity matrices and labels
# todo: and same thing with the permutations testing !
# todo: Joblib: generate first B bootstrap indices and then fit on every on them in a parallel way ?


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


def test_one_boot(features, class_labels, boot_indices):
    svc = LinearSVC()

    bootstrap_matrices = features[boot_indices, ...]
    bootstrap_class_label = class_labels[boot_indices]
    # Fit SVC on bootstrap sample
    svc.fit(bootstrap_matrices, bootstrap_class_label)
    # Weight of features for the bootstrap sample
    bootstrap_coefficients = svc.coef_[0, ...]

    return bootstrap_coefficients


def bootstrap_svc(vectorized_connectivity_matrices, class_labels, indices, n_subjects,
                  bootstrap_number, n_cpus=1):

    results_bootstrap = Parallel(n_jobs=n_cpus, verbose=1, backend="multiprocessing")(delayed(test_one_boot)(
        features=vectorized_connectivity_matrices,
        class_labels=class_labels,
        boot_indices=np.random.choice(a=indices, size=n_subjects, replace=True)) for b in range(bootstrap_number))

    return np.array(results_bootstrap)


