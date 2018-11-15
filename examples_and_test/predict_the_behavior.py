import importlib
from data_handling import atlas, data_architecture, dictionary_operations
from utils import folders_and_files_management
from utils.array_operation import array_rebuilder
from computing import compute_connectivity_matrices as ccm
from machine_learning import classification
from connectivity_statistics import parametric_tests
from plotting import display
from data_handling import data_management
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, LeaveOneOut, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# Reload all module
importlib.reload(data_management)
importlib.reload(atlas)
importlib.reload(display)
importlib.reload(parametric_tests)
importlib.reload(ccm)
importlib.reload(folders_and_files_management)
importlib.reload(classification)
importlib.reload(data_architecture)
importlib.reload(dictionary_operations)

# Some path
root_directory = "/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/network_classification"
save_in = '/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/network_classification'
kinds = ['correlation', 'partial correlation', 'tangent']
network_name = ['DMN', 'Auditory', 'Executive',
                         'Language', 'Basal_Ganglia', 'MTL',
                         'Salience', 'Sensorimotor', 'Visuospatial',
                         'Primary_Visual', 'Precuneus', 'Secondary_Visual']
subjects_connectivity_matrices = folders_and_files_management.load_object(
    full_path_to_object=os.path.join(root_directory,
                                     "dictionary/z_fisher_transform_subjects_connectivity_matrices.pkl")
)
groups = ['controls', 'non_impaired_language']
# Load the behavior data
behavior_data = pd.read_csv(os.path.join(root_directory, 'behavioral_data.csv'))
behavior_data = data_management.shift_index_column(panda_dataframe=behavior_data,
                                                   columns_to_index=['subjects'])
# Load the atlas data
atlas_excel_file = '/media/db242421/db242421_data/atlas_AVCnn/atlas_version2.xlsx'
sheetname = 'complete_atlas'

atlas_information = pd.read_excel(atlas_excel_file, sheetname=sheetname)
# Choose the connectivity measure for the analysis
kind = 'tangent'
# Homotopic roi couple position in the connectivity matrices.
homotopic_roi_indices = np.array([
    (1, 0), (2, 3), (4, 5), (6, 7), (8, 11), (9, 10), (13, 12), (14, 15), (16, 17), (18, 19), (20, 25),
    (21, 26), (22, 29), (23, 28), (24, 27), (30, 31), (32, 33), (35, 34), (36, 37), (38, 39), (44, 40),
    (41, 45), (42, 43), (46, 49), (47, 48), (50, 53), (53, 54), (54, 57), (55, 56), (58, 61), (59, 60),
    (62, 63), (64, 65), (66, 67), (68, 69), (70, 71)])

# Extract homotopic connectivity coefficient for each subject
homotopic_connectivity = np.array(
    [subjects_connectivity_matrices[group][subject]['tangent'][homotopic_roi_indices[:, 0], homotopic_roi_indices[:, 1]]
        for group in groups for subject in list(subjects_connectivity_matrices[group].keys())])

# Extract homotopic connectivity coefficient for each network
intra_network_connectivity_dict, network_dict, network_labels_list, network_label_colors = \
    ccm.intra_network_functional_connectivity(
        subjects_individual_matrices_dictionnary=subjects_connectivity_matrices,
        groupes=groups, kinds=kinds,
        atlas_file=atlas_excel_file,
        sheetname=sheetname,
        roi_indices_column_name='atlas4D index',
        network_column_name='network',
        color_of_network_column='Color')
homotopic_intra_network_connectivity = dict.fromkeys(network_name)
for network in network_name:
    # Pick the 4D index of the roi in the network
    network_roi_ind = network_dict[network]['dataframe']['atlas4D index']
    # Extract connectivity coefficient couple corresponding to homotopic regions in the network
    network_homotopic_couple_ind = np.array([couple for couple in homotopic_roi_indices if (couple[0] or couple[1])
                                             in network_roi_ind])
    homotopic_intra_network_connectivity[network] = np.array(
    [subjects_connectivity_matrices[group][subject]['tangent'][network_homotopic_couple_ind[:, 0],
                                                               network_homotopic_couple_ind[:, 1]]
        for group in groups for subject in list(subjects_connectivity_matrices[group].keys())])

features_labels = np.hstack((1*np.ones(len(subjects_connectivity_matrices[groups[0]].keys())),
                             2*np.ones(len(subjects_connectivity_matrices[groups[1]].keys()))))

mean_scores = []
svc = LinearSVC()
sss = StratifiedShuffleSplit(n_splits=10000)
loo = LeaveOneOut()
lr = LogisticRegression()
# Final mean accuracy scores will be stored in a dictionary
save_network_parameters = dict.fromkeys(network_name)
for network in network_name:
    svc = LinearSVC()
    sss = StratifiedShuffleSplit(n_splits=10000)

    features = homotopic_intra_network_connectivity[network]
    search_C = GridSearchCV(estimator=svc, param_grid={'C': np.linspace(start=0.00001, stop=1, num=100)},
                            scoring='accuracy', cv=sss, n_jobs=16, verbose=1)

    if __name__ == '__main__':
        search_C.fit(X=features, y=features_labels)

    print("Classification between {} and {} with a L2 linear SVM achieved the best score of {} % accuracy "
          "with C={}".format(groups[0], groups[1],
                             search_C.best_score_*100, search_C.best_params_['C']))

    C = search_C.best_params_['C']
    cv_scores = cross_val_score(estimator=LinearSVC(C=C), X=features,
                                y=features_labels, cv=sss,
                                scoring='accuracy', n_jobs=16,
                                verbose=1)
    print('mean accuracy {} % +- {} %'.format(cv_scores.mean()*100, cv_scores.std()*100))
    save_network_parameters[network] = {'n_rois': features.shape[1],
                                        'Best C': C,
                                        'Accuracy': cv_scores.mean()*100,
                                        'Std': cv_scores.std()*100}

folders_and_files_management.save_object(
    object_to_save=save_network_parameters,
    saving_directory=os.path.join(save_in,
                                  'non_impaired_language_controls/Network classification'),
    filename='best_params_and_accuracy_networks_intra_homotopic_until_1.pkl')

folders_and_files_management.save_object(
    object_to_save=homotopic_intra_network_connectivity,
    saving_directory=os.path.join(save_in,
                                  'non_impaired_language_controls/Network classification'),
    filename='homotopic_networks_dictionary.pkl'
)

folders_and_files_management.save_object(
    object_to_save=features_labels,
    saving_directory=os.path.join(save_in,
                                  'non_impaired_language_controls/Network classification'),
    filename='homotopic_networks_features_labels.pkl'
)


# Identify important brain connection in each network
# Atlas set up
atlas_folder = '/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/network_classification/atlas_AVCnn'
atlas_name = 'atlas4D_2.nii'
labels_text_file = '/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/network_classification/' \
                   'atlas_AVCnn/atlas4D_2_labels.csv'
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

# Load dictionary containing the best C parameters
best_parameters = folders_and_files_management.load_object(
    full_path_to_object=os.path.join(
        save_in,
        'best_params_and_accuracy_networks_intra_homotopic.pkl')
)
from nilearn.connectome import vec_to_sym_matrix
from machine_learning.features_indentification import compute_weight_distribution, \
    features_weights_parametric_correction, features_weights_max_t_correction
connectivity_matrices = homotopic_intra_network_connectivity
network_name = ['Auditory', 'Basal_Ganglia', 'DMN',
                'Executive', 'Language', 'Precuneus',
                'MTL', 'Primary_Visual',
                'Secondary_Visual',
                'Sensorimotor', 'Salience', 'Visuospatial']
network_classification_results = dict.fromkeys(network_name)
correction = 'max_t'
for network in network_name:
    print('Identify important connection for the {} network'.format(network))
    # Compute mean network matrices
    first_class_mean_matrix = vec_to_sym_matrix(
        vec=np.array([intra_network_connectivity_dict[groups[0]][s][kind][network]['network array']
                      for s in list(intra_network_connectivity_dict[groups[0]].keys())]),
        diagonal= np.zeros((len(list(intra_network_connectivity_dict[groups[0]].keys())),
                            network_dict[network]['number of rois']))).mean(axis=0)
    second_class_mean_matrix = vec_to_sym_matrix(
        vec=np.array([intra_network_connectivity_dict[groups[1]][s][kind][network]['network array']
                      for s in list(intra_network_connectivity_dict[groups[1]].keys())]),
        diagonal= np.zeros((len(list(intra_network_connectivity_dict[groups[1]].keys())),
                            network_dict[network]['number of rois']))).mean(axis=0)

    vectorized_connectivity_matrices = homotopic_intra_network_connectivity[network]
    class_labels = features_labels

    network_classification_weight, network_weight_distribution = compute_weight_distribution(
        vectorized_connectivity_matrices=vectorized_connectivity_matrices,
        bootstrap_number=500,
        n_permutations=10000,
        class_labels=class_labels,
        C=best_parameters[network]['Best C'],
        n_cpus_bootstrap=16,
        verbose_permutations=0,
        verbose_bootstrap=0
    )

    #  network_p_values_corrected = features_weights_parametric_correction(
    #     null_distribution_features_weights=network_weight_distribution,
    #    normalized_mean_weight=network_classification_weight,
    #    method=correction,
    #    alpha=.05
    # )
    sorted_null_maximum_dist, sorted_null_minimum_dist, p_values_max, p_values_min = \
        features_weights_max_t_correction(null_distribution_features_weights=network_weight_distribution,
                                          normalized_mean_weight=network_classification_weight)
    # network_classification_results[network] = {'network_features_weight': network_classification_weight,
    #                                           'network_weight_distribution': network_weight_distribution,
    #                                           'p_values_corrected': network_p_values_corrected}
    network_classification_results[network] = {'sorted_null_maximum_dist': sorted_null_maximum_dist,
                                               'sorted_null_minimum_dist': sorted_null_minimum_dist,
                                               'p_values_max': p_values_max,
                                               'p_values_min': p_values_min}

folders_and_files_management.save_object(
    object_to_save=network_classification_results,
    saving_directory=save_in,
    filename=correction + '_intra_homotopic_connection_identification.pkl'
)

# Plot results for intra network homotopic connection for each network
