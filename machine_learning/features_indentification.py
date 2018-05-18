from utils.folders_and_files_management import load_object
import os
import numpy as np
from sklearn.model_selection._split import ShuffleSplit
from nilearn.connectome import sym_matrix_to_vec
from sklearn.svm import LinearSVC

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
bootstrap_number = 200

# Number of permutation
n_permutations = 200


# SVC initialization
svc = LinearSVC()

coef_ = []
for train_index, test_index in cross_validation_iterator.split(vectorized_connectivity_matrices, class_labels):
    svc.fit(X=vectorized_connectivity_matrices[train_index], y=class_labels[train_index])
    coef_.append(svc.coef_)


