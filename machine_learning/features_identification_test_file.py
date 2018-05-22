from utils.folders_and_files_management import load_object
import os
import numpy as np
from nilearn.connectome import sym_matrix_to_vec
from machine_learning.features_indentification import bootstrap_svc, permutation_bootstrap_svc, \
    null_distribution_classifier_weight
import psutil

# Load connectivity matrices
data_folder = 'D:\\text_output_11042018\\dictionary'
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
bootstrap_number = 100

# Number of permutation
n_permutations = 10

# Number of subjects
n_subjects = vectorized_connectivity_matrices.shape[0]

# Indices to bootstrap
indices = np.arange(n_subjects)
# Generate a matrix containing all bootstraped indices
bootstrap_matrix = np.random.choice(a=indices, size=(bootstrap_number, n_subjects), replace=True)
n_physical = psutil.cpu_count(logical=False)
n_cpu_with_logical = psutil.cpu_count(logical=True)

if __name__ == '__main__':
    # True bootstrap weight
    bootstrap_weight = bootstrap_svc(features=vectorized_connectivity_matrices, class_labels=class_labels,
                                     indices=bootstrap_matrix, bootstrap_number=bootstrap_number,
                                     n_cpus=n_cpu_with_logical)

    class_labels_perm_bootstrap_weight = permutation_bootstrap_svc(
        features=vectorized_connectivity_matrices,
        class_labels=class_labels, indices=indices,
        bootstrap_number=bootstrap_number, n_cpus=n_cpu_with_logical)

    null_distribution = null_distribution_classifier_weight(features=vectorized_connectivity_matrices,
                                                            class_labels=class_labels,
                                                            indices=indices,
                                                            bootstrap_number=bootstrap_number,
                                                            n_permutations=n_permutations,
                                                            n_cpus=n_cpu_with_logical)





