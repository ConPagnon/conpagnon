from scipy.io import savemat
from utils.folders_and_files_management import load_object
import os
import numpy as np
"""
Convert some python object to matlab (.mat)
objects.

Author: Dhaif BEKHA.
"""

# Load subject connectivity matrices
patients_connectivity_matrices = load_object(
    full_path_to_object='times_series_individual_atlases.pkl'
                        'raw_subjects_connectivity_matrices.pkl')['patients']
controls_connectivity_matrices = load_object(
    full_path_to_object='/media/db242421/db242421_data/ConPagnon_data/patient_controls/dictionary/'
                        'raw_subjects_connectivity_matrices.pkl')['controls']

mean_patients_matrices = load_object('/media/db242421/db242421_data/ConPagnon_data/patient_controls/dictionary/'
                                     'mean_connectivity_matrices_patients_controls.pkl')['patients']
mean_controls_matrices = load_object('/media/db242421/db242421_data/ConPagnon_data/patient_controls/dictionary/'
                                     'mean_connectivity_matrices_patients_controls.pkl')['controls']
# save for each subject the connectivity matrices for the desired metric
metric = 'correlation'
save_patients_dir = '/media/db242421/db242421_data/Bac_a_sable/Test_BCT/patients_mat'
save_controls_dir = '/media/db242421/db242421_data/Bac_a_sable/Test_BCT/controls_mat'

# Save patients matrices, fill the diagonal with zeros
for subject in patients_connectivity_matrices.keys():
    subject_dict = {}
    np.fill_diagonal(patients_connectivity_matrices[subject][metric], 0)
    subject_dict[subject] = patients_connectivity_matrices[subject][metric]
    savemat(os.path.join(save_patients_dir, subject + '.mat'), subject_dict)

# Save patients matrices, fill the diagonal with zeros
for subject in controls_connectivity_matrices.keys():
    subject_dict = {}
    np.fill_diagonal(controls_connectivity_matrices[subject][metric], 0)
    subject_dict[subject] = controls_connectivity_matrices[subject][metric]
    savemat(os.path.join(save_controls_dir, subject + '.mat'), subject_dict)


# Save mean patients mean matrices
mean_patients_dict = {}
np.fill_diagonal(mean_patients_matrices[metric], 0)
mean_patients_dict['patients'] = mean_patients_matrices[metric]
savemat('/media/db242421/db242421_data/Bac_a_sable/Test_BCT/mean_patients.mat', mean_patients_dict)

# Save mean controls matrices
mean_controls_dict = {}
np.fill_diagonal(mean_controls_matrices[metric], 0)
mean_controls_dict['controls'] = mean_controls_matrices[metric]
savemat('/media/db242421/db242421_data/Bac_a_sable/Test_BCT/mean_controls.mat', mean_controls_dict)
