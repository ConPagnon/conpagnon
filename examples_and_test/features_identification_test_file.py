from utils.folders_and_files_management import load_object
import os
import numpy as np
from nilearn.connectome import sym_matrix_to_vec
from machine_learning.features_indentification import bootstrap_svc, \
    permutation_bootstrap_svc
import psutil
import pyximport; pyximport.install()
import time

# Load connectivity matrices
data_folder = '/media/db242421/db242421_data/ConPagnon_data/patient_controls/dictionary'
connectivity_dictionary_name = 'raw_subjects_connectivity_matrices.pkl'
subjects_connectivity_matrices = load_object(os.path.join(data_folder, connectivity_dictionary_name))
class_names = ['controls', 'patients']
vectorized_connectivity_matrices = sym_matrix_to_vec(
    np.array([subjects_connectivity_matrices[class_name][s]['correlation'] for class_name
              in class_names for s in subjects_connectivity_matrices[class_name].keys()]), discard_diagonal=True)
# Labels vectors
class_labels = np.hstack((np.zeros(len(subjects_connectivity_matrices[class_names[0]].keys())),
                          np.ones(len(subjects_connectivity_matrices[class_names[1]].keys()))))

# Number of Bootstrap (with replacement)
bootstrap_number = 500

# Number of permutation
n_permutations = 5000

# Number of subjects
n_subjects = vectorized_connectivity_matrices.shape[0]

# Indices to bootstrap
indices = np.arange(n_subjects)
# Generate a matrix containing all bootstraped indices
bootstrap_matrix = np.random.choice(a=indices, size=(bootstrap_number, n_subjects), replace=True)
n_physical = psutil.cpu_count(logical=False)
n_cpu_with_logical = psutil.cpu_count(logical=True)

class_labels_permutation_matrix = np.array([np.random.permutation(class_labels) for n in range(n_permutations)])
# Null distribution for minimum and maximum weight of classifier at each permutation
null_extremum_distribution = np.zeros((n_permutations, 2))

if __name__ == '__main__':
    # True bootstrap weight
    tic_bootstrap = time.time()
    print('Performing classification on {} bootstrap sample...'.format(bootstrap_number))
    bootstrap_weight = bootstrap_svc(features=vectorized_connectivity_matrices, class_labels=class_labels,
                                     bootstrap_array_indices=bootstrap_matrix, bootstrap_number=bootstrap_number,
                                     n_cpus_bootstrap=n_physical, verbose=1)
    tac_bootstrap = time.time()
    t_bootstrap = tac_bootstrap - tic_bootstrap

    normalized_mean_weight = bootstrap_weight.mean(axis=0)/bootstrap_weight.std(axis=0)

    # Try with a classical for loop
    null_distribution = np.zeros((n_permutations, vectorized_connectivity_matrices.shape[1]))
    tic_permutations = time.time()
    for n in range(n_permutations):
        print('Performing permutation number {} out of {}'.format(n, n_permutations))
        # Perform the classification of each bootstrap sample, but with the labels shuffled
        bootstrap_weight_perm = permutation_bootstrap_svc(features=vectorized_connectivity_matrices,
                                                          class_labels_perm=class_labels_permutation_matrix[n, ...],
                                                          indices=indices,
                                                          bootstrap_number=bootstrap_number,
                                                          n_cpus_bootstrap=n_physical,
                                                          verbose=1)
        # Compute the normalized mean of weight for the current permutation
        normalized_mean_weight_perm = bootstrap_weight_perm.mean(axis=0) / bootstrap_weight_perm.std(axis=0)
        # Save it in the null distribution array
        null_distribution[n, ...] = normalized_mean_weight_perm
    tac_permutations = time.time() - tic_permutations

    # Find minimum and maximum weight in the normalized mean for each permutations
    null_extremum_distribution[:, 0], null_extremum_distribution[:, 1] = \
        null_distribution.min(axis=1), null_distribution.max(axis=1)

    # Compare each edges weight from the true mean weight normalized distribution to the minimum and
    # maximum null estimated distribution.

    # null distribution for maximum and minimum normalized weight
    sorted_null_maximum_dist = sorted(null_extremum_distribution[:, 1])
    sorted_null_minimum_dist = sorted(null_extremum_distribution[:, 0])

    # p values array
    p_values_max = np.zeros(vectorized_connectivity_matrices.shape[1])
    p_values_min = np.zeros(vectorized_connectivity_matrices.shape[1])

    for feature in range(normalized_mean_weight.shape[0]):
        p_values_max[feature] = \
            (len(np.where(sorted_null_maximum_dist > normalized_mean_weight[feature])[0]) / (n_permutations + 1))
        p_values_min[feature] = \
            (len(np.where(sorted_null_minimum_dist < normalized_mean_weight[feature])[0]) / (n_permutations + 1))






