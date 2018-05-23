from utils.folders_and_files_management import load_object
import os
import numpy as np
from nilearn.connectome import sym_matrix_to_vec
from machine_learning.features_indentification import bootstrap_svc, \
    null_distribution_classifier_weight
import psutil
import pyximport; pyximport.install()
from machine_learning.cythonized_version import features_indentification_cython
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
n_permutations = 10000

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
    bootstrap_weight = bootstrap_svc(features=vectorized_connectivity_matrices, class_labels=class_labels,
                                     bootstrap_array_indices=bootstrap_matrix, bootstrap_number=bootstrap_number,
                                     n_cpus_bootstrap=n_physical)

    normalized_mean_weight = bootstrap_weight.mean(axis=0)/bootstrap_weight.std(axis=0)
    tic = time.time()
    null_distribution = null_distribution_classifier_weight(features=vectorized_connectivity_matrices,
                                                            class_labels_perm_matrix=class_labels_permutation_matrix,
                                                            indices=indices, bootstrap_number=bootstrap_number,
                                                            n_permutations=n_permutations,
                                                            n_cpus_permutations=10,
                                                            n_cpus_bootstrap=18)
    tac = time.time()
    T = tac - tic

    # Compute normalized mean over bootstrap sample for each permutation
    normalized_mean_permutations =\
        np.array([(null_distribution[n, ...].mean(axis=0)/null_distribution[n, ...].std(axis=0))
                  for n in range(n_permutations)])

    # Find minimum and maximum weight in the normalized mean for each permutations
    null_extremum_distribution[:, 0], null_extremum_distribution[:, 1] = \
        normalized_mean_permutations.min(axis=1), normalized_mean_permutations.max(axis=1)





