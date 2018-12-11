from utils.folders_and_files_management import load_object
import os
import numpy as np
from data_handling import atlas
from nilearn.connectome import sym_matrix_to_vec
from machine_learning.features_indentification import discriminative_brain_connection_identification

# Atlas set up
atlas_folder = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference'
atlas_name = 'atlas4D_2.nii'
labels_text_file = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference/atlas4D_2_labels.csv'
colors = ['navy', 'sienna', 'orange', 'orchid', 'indianred', 'olive',
          'goldenrod', 'turquoise', 'darkslategray', 'limegreen', 'black',
          'lightpink']
# Number of regions in each user defined networks
networks = [2, 10, 2, 6, 10, 2, 8, 6, 8, 8, 6, 4]
# Atlas path
# Read labels regions files
atlas_nodes, labels_regions, labels_colors, n_nodes = atlas.fetch_atlas(
    atlas_folder=atlas_folder,
    atlas_name=atlas_name,
    network_regions_number=networks,
    colors_labels=colors,
    labels=labels_text_file,
    normalize_colors=True)

# Load connectivity matrices
data_folder = '/media/db242421/db242421_data/ConPagnon_data/language_study_ANOVA_ACM_controls_new_figures/dictionary'
connectivity_dictionary_name = 'z_fisher_transform_subjects_connectivity_matrices.pkl'
subjects_connectivity_matrices = load_object(os.path.join(data_folder,
                                                          connectivity_dictionary_name))
subjects_connectivity_matrices['patients'] = {**subjects_connectivity_matrices['PAL'],
                                              **subjects_connectivity_matrices['PNL']}
del subjects_connectivity_matrices['PAL']
del subjects_connectivity_matrices['PNL']
class_names = ['patients', 'TDC']
metric = 'tangent'

# Vectorize the connectivity for classification
vectorized_connectivity_matrices = sym_matrix_to_vec(
    np.array([subjects_connectivity_matrices[class_name][s][metric] for class_name
              in class_names for s in subjects_connectivity_matrices[class_name].keys()]),
    discard_diagonal=True)

# Stacked the 2D array of connectivity matrices for each subjects
stacked_connectivity_matrices = np.array([subjects_connectivity_matrices[class_name][s][metric]
                                          for class_name in class_names
                                          for s in subjects_connectivity_matrices[class_name].keys()])

# Compute mean connectivity matrices for each class
first_class_mean_matrix = np.array([subjects_connectivity_matrices[class_names[0]][s][metric] for s in
                                    subjects_connectivity_matrices[class_names[0]].keys()]).mean(axis=0)
second_class_mean_matrix = np.array([subjects_connectivity_matrices[class_names[1]][s][metric] for s in
                                     subjects_connectivity_matrices[class_names[1]].keys()]).mean(axis=0)

save_directory = '/media/db242421/db242421_data/ConPagnon_data/language_study_ANOVA_ACM_controls_new_figures' \
                 '/discriminative_connection_identification'

# Labels vectors
class_labels = np.hstack((1*np.ones(len(subjects_connectivity_matrices[class_names[0]].keys())),
                          -1*np.ones(len(subjects_connectivity_matrices[class_names[1]].keys()))))

classifier_weights, weight_null_distribution, p_values_corrected = \
    discriminative_brain_connection_identification(
        vectorized_connectivity_matrices=vectorized_connectivity_matrices,
        class_labels=class_labels,
        class_names=class_names,
        save_directory=save_directory,
        n_permutations=1000,
        bootstrap_number=500,
        features_labels=labels_regions,
        features_colors=labels_colors,
        n_nodes=n_nodes,
        atlas_nodes=atlas_nodes,
        first_class_mean_matrix=first_class_mean_matrix,
        second_class_mean_matrix=second_class_mean_matrix,
        n_cpus_bootstrap=16,
        write_report=True,
        correction='fdr_bh',
        C=1)
