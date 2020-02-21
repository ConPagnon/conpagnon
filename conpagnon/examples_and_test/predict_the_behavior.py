import importlib
from conpagnon.data_handling import data_architecture, dictionary_operations, atlas, data_management
from conpagnon.utils import folders_and_files_management
from conpagnon.utils.array_operation import array_rebuilder
from conpagnon.computing import compute_connectivity_matrices as ccm
from conpagnon.machine_learning import classification
from conpagnon.connectivity_statistics import parametric_tests
from conpagnon.plotting import display
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, LeaveOneOut, GridSearchCV
from sklearn.svm import LinearSVC
from nilearn.plotting import plot_connectome
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from conpagnon.data_handling.dictionary_operations import groupby_factor_connectivity_matrices
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
root_directory = "/media/db242421/db242421_data/ConPagnon_data/language_study_ANOVA_ACM_controls"
save_in = '/media/db242421/db242421_data/ConPagnon_data/language_study_ANOVA_ACM_controls/' \
          'classification/left_lesioned_patients_controls/Network classification'
kinds = ['correlation', 'partial correlation', 'tangent']
network_name = ['DMN', 'Executive',
                 'Language', 'MTL',
                 'Salience', 'Sensorimotor', 'Visuospatial',
                 'Primary_Visual', 'Secondary_Visual']

# Load the behavior data
behavior_data = pd.read_csv(os.path.join(root_directory, 'behavioral_data.csv'))
behavior_data = data_management.shift_index_column(panda_dataframe=behavior_data,
                                                   columns_to_index=['subjects'])

subjects_connectivity_matrices = folders_and_files_management.load_object(
    full_path_to_object=os.path.join(root_directory,
                                     "dictionary/z_fisher_transform_subjects_connectivity_matrices.pkl")
)
# Group by language profil and lesion side
t, p, k = groupby_factor_connectivity_matrices(
    population_data_file=os.path.join(root_directory, 'behavioral_data.xlsx'),
    sheetname='Middle Cerebral Artery+controls',
    subjects_connectivity_matrices_dictionnary=subjects_connectivity_matrices,
    groupes=['non_impaired_language', 'impaired_language'], factors=['Lesion'])
# rename the new key
subjects_connectivity_matrices['left_lesioned_patients'] = t[('G')]
# Dump old key
del subjects_connectivity_matrices['non_impaired_language']
del subjects_connectivity_matrices['impaired_language']

groups = ['controls', 'left_lesioned_patients']


features_labels = np.hstack((1*np.ones(len(subjects_connectivity_matrices[groups[0]].keys())),
                             2*np.ones(len(subjects_connectivity_matrices[groups[1]].keys()))))

mean_subjects_connectivity_matrices = ccm.group_mean_connectivity(
    subjects_connectivity_matrices=subjects_connectivity_matrices,
    kinds=kinds)

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
    [subjects_connectivity_matrices[group][subject]['tangent'][homotopic_roi_indices[:, 0],
                                                               homotopic_roi_indices[:, 1]]
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
    homotopic_intra_network_connectivity[network + '_indices'] = network_homotopic_couple_ind

# svc = LinearSVC()
# sss = StratifiedShuffleSplit(n_splits=10000)
# Final mean accuracy scores will be stored in a dictionary
save_parameters = dict.fromkeys(network_name)

for network in network_name:
    svc = LinearSVC()
    sss = StratifiedShuffleSplit(n_splits=10000)

    features = np.array([intra_network_connectivity_dict[group][s][kind][network]['network array']
                         for group in groups for s in list(intra_network_connectivity_dict[group].keys())])

    # features = homotopic_connectivity
    search_C = GridSearchCV(estimator=svc, param_grid={'C': np.linspace(start=1e-5, stop=10, num=100)},
                            scoring='accuracy', cv=sss, n_jobs=20, verbose=1)

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
    save_parameters[network] = {
                        'Best C': C,
                        'Accuracy': cv_scores.mean()*100,
                        'Std': cv_scores.std()*100}

folders_and_files_management.save_object(
    object_to_save=save_parameters,
    saving_directory=save_in,
    filename='best_params_and_accuracy_all_intranetwork_connection.pkl')

# folders_and_files_management.save_object(
#    object_to_save=homotopic_intra_network_connectivity,
#    saving_directory=save_in,
#    filename='homotopic_networks_dictionary.pkl'
# )

folders_and_files_management.save_object(
    object_to_save=features_labels,
    saving_directory=save_in,
    filename='homotopic_networks_features_labels.pkl'
)


# Identify important brain connection in each network
# Atlas set up
atlas_folder = '/media/db242421/db242421_data/atlas_AVCnn'
atlas_name = 'atlas4D_2.nii'
labels_text_file = '/media/db242421/db242421_data/atlas_AVCnn/atlas4D_2_labels.csv'
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
        'best_params_and_accuracy_all_intra_network_connection.pkl')
)
from conpagnon.machine_learning import compute_weight_distribution, \
    features_weights_parametric_correction

network_name = ['DMN',
                'Executive', 'Language',
                'MTL', 'Primary_Visual',
                'Secondary_Visual',
                'Sensorimotor', 'Salience', 'Visuospatial']
#classification_results = dict.fromkeys(network_name)
correction = 'fdr_bh'
classification_results = dict.fromkeys(network_name)
for network in network_name:

    print('Identify important connection for the {} network'.format(network))

    vectorized_connectivity_matrices = \
        np.array([intra_network_connectivity_dict[group][s][kind][network]['network array']
                  for group in groups for s in list(intra_network_connectivity_dict[group].keys())])

# vectorized_connectivity_matrices = homotopic_connectivity

    class_labels = features_labels

    classification_weight, weight_distribution = compute_weight_distribution(
        vectorized_connectivity_matrices=vectorized_connectivity_matrices,
        bootstrap_number=500,
        n_permutations=1000,
        class_labels=class_labels,
        C=best_parameters[network]['Best C'],
        n_cpus_bootstrap=18,
        verbose_permutations=1,
        verbose_bootstrap=0
    )

    p_values_corrected = features_weights_parametric_correction(
       null_distribution_features_weights=weight_distribution,
       normalized_mean_weight=classification_weight,
       method=correction,
       alpha=.05
  )
    # sorted_null_maximum_dist, sorted_null_minimum_dist, p_values_max, p_values_min = \
    #   features_weights_max_t_correction(null_distribution_features_weights=weight_distribution,
    #                                     normalized_mean_weight=classification_weight)

    classification_results[network] = {'network_features_weight': classification_weight,
                                               'network_weight_distribution': weight_distribution,
                                               'p_values_corrected': p_values_corrected}
   # classification_results = {'sorted_null_maximum_dist': sorted_null_maximum_dist,
    #                          'sorted_null_minimum_dist': sorted_null_minimum_dist,
    #                          'p_values_max': p_values_max,
    #                          'p_values_min': p_values_min}

#classification_results = {
#    'features_weight': classification_weight,
#    'weight_distribution': weight_distribution,
#    'p_values_corrected': p_values_corrected}

folders_and_files_management.save_object(
    object_to_save=classification_results,
    saving_directory=save_in,
    filename=correction + '_all_homotopic_connection_identification.pkl'
)

# Plot results for intra network homotopic connection for each network
homotopic_results_fdr = folders_and_files_management.load_object(
    full_path_to_object=os.path.join(save_in, 'fdr_bh_intra_homotopic_connection_identification.pkl')
)
with PdfPages(os.path.join(save_in, 'fdr_corrected_intra_network_homotopic_classification.pdf')) as pdf:

    for network in list(homotopic_results_fdr.keys()):
        # Search p significant
        network_p = homotopic_results_fdr[network]['p_values_corrected']
        significant_p = np.where(network_p < 0.05)[0]
        if network_p[significant_p]:
            # Retrieve indices for the significant connection
            network_connection_indices = \
                homotopic_intra_network_connectivity[network + '_indices'][significant_p, :]
            # Retrieve roi coordinates for the current network
            network_rois_coordinates = network_dict[network]['dataframe'][['x', 'y', 'z']]
            # Retrieve roi colors for the current network
            network_colors = labels_colors[network_rois_coordinates.index]
            # Take the mean connectivity difference between the two group for the
            # significant connection
            first_group_mean_matrix = mean_subjects_connectivity_matrices[groups[0]][kind]
            second_group_mean_matrix = mean_subjects_connectivity_matrices[groups[1]][kind]
            mean_difference = second_group_mean_matrix - first_group_mean_matrix
            network_mean_difference_matrix_tmp = mean_difference[network_rois_coordinates.index, :]
            network_mean_difference_matrix =  network_mean_difference_matrix_tmp[:, network_rois_coordinates.index]
            network_adjacency_matrix = np.zeros((network_rois_coordinates.shape[0],
                                                 network_rois_coordinates.shape[0]))
            # Fill network adjacency matrix
            for i in range(network_connection_indices.shape[0]):
                print(i)
                i_position = np.where(network_connection_indices[i][0] == network_rois_coordinates.index)[0][0]
                j_position = np.where(network_connection_indices[i][1] == network_rois_coordinates.index)[0][0]
                for j in range(network_connection_indices.shape[1]):
                    network_adjacency_matrix[i_position, j_position] = network_mean_difference_matrix[i_position,
                                                                                                      j_position]
            network_adjacency_matrix = network_adjacency_matrix + network_adjacency_matrix.T
            # Plot connectome
            plt.figure()
            plot_connectome(adjacency_matrix=network_adjacency_matrix,
                            node_coords=network_rois_coordinates,
                            node_color=network_colors,
                            node_size=20,
                            title='{} ({} - {})'.format(network,
                                                        groups[1], groups[0]),
                            colorbar=True)
            pdf.savefig()
            plt.show()

# plot results for the identification of connection for ALL homotopic
# connection
all_homotopic_adjacency_matrix = np.zeros((n_nodes, n_nodes))
#significant_p_values_indices = np.hstack((np.where(classification_results['p_values_min'] < 0.05)[0],
#                                          np.where(classification_results['p_values_max'] < 0.05)[0]))
significant_p_values_indices = np.where(classification_results['p_values_corrected'] < 0.05)[0]
# Build Homotopic adjacency matrix

significant_labels_indices = homotopic_roi_indices[significant_p_values_indices ]
for i in range(significant_labels_indices.shape[0]):
    all_homotopic_adjacency_matrix[significant_labels_indices[i][0],
                                   significant_labels_indices[i][1]] = \
        mean_subjects_connectivity_matrices[groups[1]]['tangent'][
            significant_labels_indices[i][0], significant_labels_indices[i][1]] - \
        mean_subjects_connectivity_matrices[groups[0]]['tangent'][
            significant_labels_indices[i][0], significant_labels_indices[i][1]]
all_homotopic_adjacency_matrix += all_homotopic_adjacency_matrix.T
with PdfPages(os.path.join(save_in, correction + '_all_homotopic_identification.pdf')) as pdf:
    plt.figure()
    plot_connectome(adjacency_matrix=all_homotopic_adjacency_matrix,
                    node_coords=atlas_nodes,
                    node_color=labels_colors,
                    node_size=20,
                    title='Surviving homotopic connection ({} - {})'.format(groups[1],
                                                                            groups[0]))
    pdf.savefig()
    plt.show()


significant_labels =[(labels_regions[significant_labels_indices[i][0]],
                      labels_regions[significant_labels_indices[i][1]])
                     for i in range(significant_labels_indices.shape[0])]
folders_and_files_management.save_object(
    object_to_save=significant_labels,
    saving_directory=save_in,
    filename=correction + '_all_homotopic_significant_features_labels.pkl'
)
from conpagnon.machine_learning import remove_reversed_duplicates
# Plot classification for intra network connectivity
intra_network_classification = folders_and_files_management.load_object(
    full_path_to_object=os.path.join(save_in, 'intra_network/fdr_bh_all_intra_network_connection_identification.pkl')
)

with PdfPages(os.path.join(save_in, 'fdr_corrected_intra_network_classification.pdf')) as pdf:

    for network in list(intra_network_classification.keys()):
        # Search p significant
        network_p = intra_network_classification[network]['p_values_corrected']
        significant_p = np.where(network_p < 0.05)[0]
        if network_p[significant_p].shape[0] != 0:
            network_labels = list(network_dict[network]['dataframe']['anatomical label'])
            network_n_rois = network_dict[network]['number of rois']
            network_coord = network_dict[network]['dataframe'][['x', 'y', 'z']]
            network_colors = labels_colors[network_dict[network]['dataframe']['atlas4D index']]
            # Rebuild the network adjacency matrices:
            network_p_array = array_rebuilder(network_p,
                                              array_type='numeric',
                                              diagonal=np.ones(network_n_rois))
            network_p_indices = np.where(network_p_array < 0.05)
            significant_labels_indices = np.array(list(
                remove_reversed_duplicates(np.array(np.where(network_p_array < 0.05)).T)))
            significant_labels = [(network_labels[significant_labels_indices[i][0]],
                                   network_labels[significant_labels_indices[i][1]])
                                  for i in range(significant_labels_indices.shape[0])]
            intra_network_classification[network]['significant features labels'] = significant_labels
            # Take mean difference in connectivity
            network_mean_difference = \
                np.array([array_rebuilder(
                    intra_network_connectivity_dict[groups[1]][s][kind][network]['network array'],
                    array_type='numeric',
                    diagonal=intra_network_connectivity_dict[groups[1]][s][kind][network]['network diagonal array'])
                    for s in list(intra_network_connectivity_dict[groups[1]].keys())]).mean(axis=0) - \
                np.array([array_rebuilder(intra_network_connectivity_dict[groups[0]][s][kind][network]['network array'],
                                          array_type='numeric',
                                          diagonal=intra_network_connectivity_dict[groups[0]][s][kind][network][
                                              'network diagonal array'])
                          for s in list(intra_network_connectivity_dict[groups[0]].keys())]).mean(axis=0)
            # Build network adjacency matrix
            network_adjacency_matrix = np.zeros((network_n_rois, network_n_rois))
            network_adjacency_matrix[network_p_indices[0], network_p_indices[1]] = \
                network_mean_difference[network_p_indices[0], network_p_indices[1]]
            # plot connectome
            plt.figure()
            plot_connectome(adjacency_matrix=network_adjacency_matrix,
                            node_coords=network_coord,
                            node_color=network_colors,
                            node_size=20,
                            title='{} ({} - {})'.format(network,
                                                        groups[1], groups[0]),
                            colorbar=True)
            pdf.savefig()
            plt.show()

folders_and_files_management.save_object(
    object_to_save=intra_network_classification,
    saving_directory=save_in,
    filename=correction + '_all_intra_network_connection_identification.pkl'
)

# Prediction of continuous language test with leave one out cross
# validation
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge

list_of_scores = ['uni_deno', 'plu_deno', 'uni_rep', 'plu_rep',
                  'empan', 'phono', 'elision_i',
                  'invers', 'ajout', 'elision_f', 'morpho',
                  'listea', 'listeb', 'topo',
                  'voc1', 'voc2', 'voc1_ebauche', 'voc2_ebauche',
                  'abstrait_diff', 'abstrait_pos', 'lex1', 'lex2',
                  'pc1_language']
subjects_list = list(subjects_connectivity_matrices[groups[1]].keys())
# Load features
from nilearn.connectome import sym_matrix_to_vec
from sklearn.model_selection import cross_val_predict

# Build subjects array

# For intra-network, based on intra network classification
subjects_features_all_networks = dict.fromkeys(network_name)
for network in network_name:
    network_p = intra_network_classification[network]['p_values_corrected']
    significant_p = np.where(network_p < 0.05)[0]
    if network_p[significant_p].shape[0] != 0:
        vectorized_connectivity_matrices = \
            np.array([intra_network_connectivity_dict[groups[1]][s][kind][network]['network array']
                      for s in subjects_list])
        # Build features array for the current network, based on surviving
        # connection in the classification task
        subjects_network_features = np.zeros((len(subjects_list), significant_p.shape[0]))
        subjects_network_features[:, ...] = vectorized_connectivity_matrices[:, significant_p]
        subjects_features_all_networks[network] = subjects_network_features
    else:
        del subjects_features_all_networks[network]


subjects_features = sym_matrix_to_vec(np.array([subjects_connectivity_matrices[groups[1]][s][kind]
                                                for s in subjects_list]),
                                      discard_diagonal=True)
subjects_data = behavior_data[behavior_data['Lesion'] == 'G'].reindex(subjects_list)

# Search a good regularization parameters for each scores
from statsmodels.api import add_constant, OLS
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
results_prediction = dict.fromkeys(list(subjects_features_all_networks.keys()))
for network in list(subjects_features_all_networks.keys()):
    results_prediction[network] = dict.fromkeys(list_of_scores)
    for score in list_of_scores:

        sc = StandardScaler()
        subjects_scores = sc.fit_transform(np.array(subjects_data[score]).reshape(-1, 1))
        # subjects_scores = np.array(subjects_data[score])
        svr = LinearSVR()
        ridge = Ridge()
        loo = LeaveOneOut()
        squared_error = GridSearchCV(estimator=LinearSVR(),
                                     param_grid={'C': np.linspace(start=1e-6,
                                                                  stop=1e6,
                                                                  num=5000)},
                                     scoring=make_scorer(mean_squared_error,
                                                         greater_is_better=False),
                                     n_jobs=12,
                                     verbose=1,
                                     cv=LeaveOneOut())
        squared_error.fit(X=subjects_features_all_networks[network], y=subjects_scores.ravel())

        predicted = cross_val_predict(estimator=LinearSVR(C=squared_error.best_params_['C']),
                                      X=subjects_features_all_networks[network],
                                      y=subjects_scores.ravel(),
                                      n_jobs=4,
                                      cv=LeaveOneOut())
        # TODO: erase score keys in the sub-dictionary !! Useless
        results_prediction[network][score] = {
                                           {'score_true': subjects_scores,
                                            'score_predicted': predicted,
                                            'r2': pearsonr(x=subjects_scores.ravel(),
                                                           y=predicted)[0]**2,
                                            'best_parameters': squared_error.best_params_}}
        # Perform linear regression
        X = add_constant(subjects_scores)
        reg = OLS(predicted, X)
        results = reg.fit()
        # Append Adjusted R squared between predicted and true values
        results_prediction[network][score]['r_squared'] = results.rsquared

folders_and_files_management.save_object(object_to_save=results_prediction,
                                         saving_directory=os.path.join(root_directory, 'score_prediction'),
                                         filename='LinearSVR_score_prediction_networks.pkl')
# Find best network predictors
all_r2 = dict.fromkeys(network_name)
