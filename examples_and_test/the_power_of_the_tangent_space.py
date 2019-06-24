"""
 Created by db242421 at 13/12/18

 """
import importlib
from data_handling import atlas, data_architecture, dictionary_operations
from utils import folders_and_files_management
from computing import compute_connectivity_matrices as ccm
from machine_learning import classification
from connectivity_statistics import parametric_tests
from plotting import display
from matplotlib.backends import backend_pdf
from data_handling import data_management
import os
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure, vec_to_sym_matrix

from nilearn.plotting import plot_connectome

import networkx as nx

from nilearn import plotting

from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedShuffleSplit, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, SelectFpr, mutual_info_regression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from machine_learning.CPM_method import predictors_selection_correlation
from sklearn.svm import SVR, LinearSVC
from sklearn.linear_model import Ridge
from joblib import delayed, Parallel
from sklearn.decomposition import PCA
from machine_learning.scores_predictions import predict_scores, vcorrcoef
from machine_learning import scores_predictions
from statsmodels.api import OLS, add_constant
from sklearn import decomposition
from plotting.display import plot_matrix
import statsmodels.api as sm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.datasets import samples_generator

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

# Groups name to include in the study
groups = ['patients_r', 'TDC']
# The root fmri data directory containing all the fmri files directories
root_fmri_data_directory = \
    '/media/db242421/db242421_data/ConPagnon_data/fmri_images'
# Check all functional files existence
folders_and_files_management.check_directories_existence(
    root_directory=root_fmri_data_directory,
    directories_list=groups)

# Full path, including extension, to the text file containing
# all the subject identifiers.
subjects_ID_data_path = \
    '/media/db242421/db242421_data/ConPagnon_data/text_data/acm_patients_and_controls_all.txt'

organised_data = data_architecture.fetch_data(
    subjects_id_data_path=subjects_ID_data_path,
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groups,
    individual_counfounds_directory=None
)

# kinds
kinds = ['tangent']

# Nilearn cache directory
nilearn_cache_directory = '/media/db242421/db242421_data/ConPagnon/nilearn_cache'

# Repetition time between time points in the fmri acquisition, usually 2.4s or 2.5s
repetition_time = 2.5

# Load behavioral scores
behavioral_scores = data_management.read_excel_file(
    excel_file_path='/media/db242421/db242421_data/ConPagnon_data/regression_data/'
                    'Resting State AVCnn_cohort data.xlsx',
    sheetname='Middle Cerebral Artery+controls')
behavioral_scores = behavioral_scores.set_index(["NIP"])

times_series = ccm.time_series_extraction(
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groups,
    subjects_id_data_path=subjects_ID_data_path,
    reference_atlas=os.path.join(atlas_folder, atlas_name),
    group_data=organised_data,
    repetition_time=repetition_time,
    nilearn_cache_directory=nilearn_cache_directory
)

# try to select a sub group directly from the time series
# dictionary
groups_by_factor, population_df_by_factor, factor_keys = \
    dictionary_operations.groupby_factor_connectivity_matrices(
        population_data_file='/media/db242421/db242421_data/ConPagnon_data/regression_data/'
                             'Resting State AVCnn_cohort data2.xlsx', sheetname='Middle Cerebral Artery+controls',
        subjects_connectivity_matrices_dictionnary=times_series, groupes=['patients_r'], factors=['Lesion',
                                                                                                  'Parole'],
        drop_subjects_list=None)

times_series_ = {'reference_group': groups_by_factor[('G', 'N')],
                 'group_to_project': groups_by_factor[('D', 'N')]}

reference_group = list(times_series_['reference_group'].keys())
subjects_to_project = list(times_series_['group_to_project'].keys())

# Stack the times series for each group
reference_group_time_series = np.array([times_series_['reference_group'][s]['time_series']
                                        for s in reference_group])
group_to_project_time_series = np.array([times_series_['group_to_project'][s]['time_series']
                                         for s in subjects_to_project])

# Number of bootstrap
m = 10000
size_subset_reference_group = 10
alpha = 0.01

from computing.compute_connectivity_matrices import tangent_space_projection

tangent_space_projection_dict = tangent_space_projection(
    reference_group=reference_group_time_series,
    group_to_project=group_to_project_time_series,
    bootstrap_number=m,
    bootstrap_size=size_subset_reference_group,
    output_directory="/media/db242421/db242421_data/ConPagnon_data/tangent_space/"
                     "test_fonction2",
    verif_null=True,
    statistic='t',
    correction_method="bonferroni",
    alpha=alpha)

# Retrieves import results from the dictionary

# Corrected p values for each projected subject
p_values_corrected = tangent_space_projection_dict['p_values_corrected']
# Tangent Connectivity matrices for each projected subject
group_to_project_tangent_matrices = tangent_space_projection_dict['group_to_project_tangent_matrices']
# Reference group mean correlation matrices
reference_group_tangent_mean = tangent_space_projection_dict['reference_group_tangent_mean']
# output statistic for each projected subject
group_to_project_stats = tangent_space_projection_dict['group_to_project_stats']

# Count the number of time a node appear across subjects_to_project
significant_hit_per_nodes = vec_to_sym_matrix(np.sum(p_values_corrected < alpha, axis=0),
                                              diagonal=np.zeros(n_nodes))
significant_hit_per_nodes_g = nx.from_numpy_array(significant_hit_per_nodes)
significant_node_degree = np.array([val for (node, val) in significant_hit_per_nodes_g.degree()])*10
empty_adjacency_matrix = np.zeros(shape=(n_nodes, n_nodes))

plt.figure()
plot_connectome(adjacency_matrix=empty_adjacency_matrix,
                node_coords=atlas_nodes,
                node_color=labels_colors,
                node_size=significant_node_degree,
                title='Common nodes across subjects_to_project')
plt.show()


with backend_pdf.PdfPages('/media/db242421/db242421_data/ConPagnon_data/tangent_space/test_fonction2/'
                          'projected_group_stats_t_bonf_0.01_LG_ParoleTypique_LD_Paroletypique.pdf') as pdf:
    for subject in range(len(subjects_to_project)):
        if subjects_to_project[subject] in behavioral_scores.index:

            # compute node degree for each subject
            # based on the surviving connection
            group_to_project_significant_edges = vec_to_sym_matrix(p_values_corrected[subject, ...] < alpha,
                                                          diagonal=np.zeros(n_nodes))
            patient_adjacency_matrices = nx.from_numpy_array(group_to_project_significant_edges)
            degrees = np.array([val for (node, val) in patient_adjacency_matrices.degree()])*40

            # plot corrected connection
            if np.unique(group_to_project_significant_edges).size == 1:
                plt.figure()
                plot_connectome(
                    adjacency_matrix=empty_adjacency_matrix,
                    node_coords=atlas_nodes,
                    node_color=labels_colors,
                    title='{}, Lesion: {}, Language: {}, Speech: {}, PC1: {}'.format(
                        subjects_to_project[subject][0:5],
                        behavioral_scores.loc[subjects_to_project[subject]]['Lesion'],
                        behavioral_scores.loc[subjects_to_project[subject]]['langage_clinique'],
                        behavioral_scores.loc[subjects_to_project[subject]]['Parole'],
                        round(behavioral_scores.loc[subjects_to_project[subject]]['pc1_language'], 3)),
                    colorbar=False,
                    node_kwargs={'edgecolor': 'black', 'alpha': 1})
                pdf.savefig()
                plt.show()
            else:

                plt.figure()
                plot_connectome(

                    adjacency_matrix=vec_to_sym_matrix(np.multiply(p_values_corrected[subject, ...] < alpha,
                                                                   group_to_project_tangent_matrices[subject, ...]),
                                                       diagonal=np.zeros(n_nodes)),
                    node_coords=atlas_nodes,
                    node_color=labels_colors,
                    title='{}, Lesion: {}, Language: {}, Speech: {}, PC1: {}'.format(
                        subjects_to_project[subject][0:5],
                        behavioral_scores.loc[subjects_to_project[subject]]['Lesion'],
                        behavioral_scores.loc[subjects_to_project[subject]]['langage_clinique'],
                        behavioral_scores.loc[subjects_to_project[subject]]['Parole'],
                        round(behavioral_scores.loc[subjects_to_project[subject]]['pc1_language'], 3)),
                    colorbar=True,
                    node_size=degrees,
                    node_kwargs={'edgecolor': 'black', 'alpha': 1},
                    edge_threshold=None)
                pdf.savefig()
                plt.show()

                plt.figure()
                view = plotting.view_connectome(
                    adjacency_matrix=vec_to_sym_matrix(np.multiply(p_values_corrected[subject, ...] < alpha,
                                                                   group_to_project_stats[subject, ...]),
                                                       diagonal=np.zeros(n_nodes)),
                    coords=atlas_nodes,
                    cmap='bwr')
                view.save_as_html(os.path.join('/media/db242421/db242421_data/ConPagnon_data/tangent_space',
                                               subjects_to_project[subject] + '_t_score.html'))
    # Plot tangent mean controls connectome
    plt.figure()
    plot_connectome(adjacency_matrix=reference_group_tangent_mean, node_coords=atlas_nodes,
                    node_color=labels_colors, edge_threshold='80%', title='{} tangent mean'.format("reference group"),
                    colorbar=True)
    pdf.savefig()
    plt.show()




#### Prediction ####
sc = StandardScaler()
absolute_max_distance = np.max(np.abs(group_to_project_tangent_matrices), axis=1)
X = sm.add_constant(absolute_max_distance)
y = np.array(behavioral_scores.loc[subjects_to_project]['pc1_language_zscores'])
dist_tangent = sm.OLS(y, X).fit()
dist_tangent.summary()

# SVM classification
labels = np.array([0 if behavioral_scores.loc[s]['Lesion'] == 'G' else 1 for s in subjects_to_project])
sss = StratifiedShuffleSplit(n_splits=10000)
accuracy = cross_val_score(estimator=LinearSVC(C=1), X=group_to_project_tangent_matrices, y=labels,
                           cv=LeaveOneOut(), n_jobs=6)
print('Accuracy: {} % +- {} %'.format(round(np.array(accuracy).mean(), 2)*100,
                                      round(np.array(accuracy).std(), 2)*100))

atypical = group_to_project_tangent_matrices[np.where(labels == 1)[0], ...]
typical = group_to_project_tangent_matrices[np.where(labels == 0)[0], ...]

# Prediction of raw NEEL score based on functional connectivity
list_of_scores = ['uni_deno', 'plu_deno', 'uni_rep', 'plu_rep',
                  'empan', 'phono', 'elision_i',
                  'invers', 'ajout', 'elision_f', 'morpho',
                  'listea', 'listeb', 'topo',
                  'voc1', 'voc2', 'voc1_ebauche', 'voc2_ebauche',
                  'abstrait_diff', 'abstrait_pos', 'lex1', 'lex2',
                  'pc1_language', 'pc2_language']

# Scaled the raw behavioral scores
sc_score = StandardScaler()
all_scores_array = np.array(behavioral_scores.loc[subjects_to_project][list_of_scores])
all_scaled_scores = sc_score.fit_transform(all_scores_array)

# Scaled the features if needed
sc_features = StandardScaler()
maxScaler = MaxAbsScaler()
patients_tangent_matrices_sc = maxScaler.fit_transform(group_to_project_tangent_matrices)

# PCA on connectivity matrices
pca = decomposition.PCA(n_components=0.95)
pca.fit(patients_tangent_matrices_sc)
patients_tangent_matrices_pca = pca.transform(patients_tangent_matrices_sc)

# Select features on raw connectivity matrices, based on a mask
# computed with the scaled matrices with the maximum
maximum_features_mask = np.where(np.abs(patients_tangent_matrices_sc) == 1)
patients_tangent_matrices_masked = \
    [group_to_project_tangent_matrices[p, np.where(np.abs(patients_tangent_matrices_sc[p, ...]) == 1)]
     for p in range(len(subjects_to_project))]

# Selection parameter, if alpha equal to 1, all features are selected
alpha = 1
r2_scores = dict.fromkeys(list_of_scores)
# Optimisation of C parameter for each score
c_grid = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 1, 10, 50, 100, 200, 300, 500, 1000, 5000, 10000, 100000, 1000000]
alpha_grid = [1, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000, 3000,
              4000, 5000, 8000, 10000, 100000, 1000000]
# Optimisation of number of features selected by SelectKBest
k_grid = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
          1700, 1800, 1900, 2000, 2100, 2400, 2556]

best_c = dict.fromkeys(list_of_scores)
save_directory = '/media/db242421/db242421_data/ConPagnon_data/tangent_space/' \
                 'prediction_evaluation_pipeline_K_best_Ridge'

import matplotlib
colorsList = [(1, 0, 0),(0,0,1)]
CustomCmap = matplotlib.colors.ListedColormap(colorsList)

if __name__ == '__main__':
    for score in range(len(list_of_scores)):
        print('Computing selection of features and '
              'prediction for {} test....'.format(list_of_scores[score]))
        # Search on a grid, the best K, and the best C parameter minimizing the
        # mean squared error
        K_best = SelectKBest(f_regression)
        pipeline = Pipeline([('K_best_features', K_best), ('ridge', Ridge())])
        grid_search = GridSearchCV(pipeline, {'K_best_features__k': k_grid,
                                              'ridge__alpha': alpha_grid},
                                   cv=LeaveOneOut(),
                                   scoring='neg_mean_squared_error',
                                   n_jobs=10,
                                   verbose=1)
        grid_search.fit(group_to_project_tangent_matrices, all_scaled_scores[..., score])

        # Reduce the dimensionality of connectivity matrices with the best
        # K selection
        best_K = grid_search.best_params_['K_best_features__k']
        K_best_reducer = SelectKBest(f_regression, k=best_K)
        patients_tangent_matrices_KBest = K_best_reducer.fit_transform(group_to_project_tangent_matrices,
                                                                       all_scaled_scores[..., score])
        # Select the best C parameters from previous selection
        best_alpha = grid_search.best_params_['ridge__alpha']
        best_c[list_of_scores[score]] = {'alpha': best_alpha}

        # score prediction
        r2, score_pred, score_pred_weights, _ = \
            predict_scores(connectivity_matrices=patients_tangent_matrices_KBest,
                           raw_score=all_scores_array[:, score],
                           alpha=alpha,
                           alpha_ridge=best_c[list_of_scores[score]]['alpha'],
                           estimator='ridge')

        r2_scores[list_of_scores[score]] = {
            'r2': r2,
            'predicted scores': np.array(score_pred),
            'true scores': all_scaled_scores[:, score],
            'features weights': score_pred_weights,
            'features weights brain space': K_best_reducer.inverse_transform(
                np.squeeze(np.array(score_pred_weights))),
            'selected features indices': K_best_reducer.get_support(),
            'alpha': best_c[list_of_scores[score]]['alpha'],
            'best K': best_K,
            'correlation_score_connectivity': vcorrcoef(X=group_to_project_tangent_matrices.T,
                                                        y=all_scaled_scores[:, score])[0]}

        print('Prediction done for {} test'.format(list_of_scores[score]))

    with backend_pdf.PdfPages(os.path.join(
            save_directory,
            'prediction_scores_raw_matrices_f_regression_Ridge_linear_connectome_V3.pdf')) \
            as pdf:
        for score in range(len(list_of_scores)):
            r2_score = r2_scores[list_of_scores[score]]['r2']
            score_prediction_weights = vec_to_sym_matrix(
                np.mean(r2_scores[list_of_scores[score]]['features weights brain space'], axis=0),
                diagonal=np.zeros(n_nodes))
            correlation_score_connectivity = r2_scores[list_of_scores[score]]['correlation_score_connectivity']
            selected_features_indices = np.array(r2_scores[list_of_scores[score]]['selected features indices'],
                                                 dtype=int)
            masked_correlation_score_connectivity = vec_to_sym_matrix(np.multiply(correlation_score_connectivity,
                                                                selected_features_indices),
                                                                      diagonal=np.zeros(n_nodes))
            masked_correlation_score_connectivity[masked_correlation_score_connectivity > 0] = 1
            masked_correlation_score_connectivity[masked_correlation_score_connectivity < 0] = -1
            score_prediction_weights_ = np.multiply(masked_correlation_score_connectivity,
                                                    np.abs(score_prediction_weights))

            number_of_connections = r2_scores[list_of_scores[score]]['best K']

            node_size = (0.5*np.sum(np.abs(score_prediction_weights), axis=0))*200
            plt.figure()
            plot_connectome(
                adjacency_matrix=masked_correlation_score_connectivity,
                node_coords=atlas_nodes, node_color=labels_colors,
                title='score: {}, r2: {}, K: {}'.format(list_of_scores[score],
                                                        r2_scores[list_of_scores[score]]['r2'][0],
                                                        number_of_connections),
                colorbar=True,
                node_kwargs={'edgecolor': 'black', 'alpha': 1}, edge_cmap=CustomCmap,
                node_size=node_size)
            pdf.savefig()
            plt.show()

    with backend_pdf.PdfPages(os.path.join(save_directory,
                                           'prediction_scores_raw_matrices_regression_Ridge_linear_.pdf')) \
            as pdf:
        for score in list_of_scores:

            plt.figure()
            plt.plot(np.array(r2_scores[score]['true scores']),
                     np.array(r2_scores[score]['predicted scores']), 'o')
            plt.title('score: {}, r2: {}'.format(score, r2_scores[score]['r2'][0]))
            plt.xlabel('True scores')
            plt.ylabel('Predicted scores')
            pdf.savefig()
            plt.show()
    # Save the dictionary containing the results of the prediction
    # for each score
    folders_and_files_management.save_object(
        r2_scores,
        os.path.join(save_directory, 'prediction_scores_raw_connectivity_matrices_Ridge_K_best_pipeline.pkl')
       )

    # TODO permuation testing !!!!!!
    n_permutations = 10001
    permutations = [np.random.choice(np.arange(0, len(subjects_to_project), 1), replace=False, size=len(subjects_to_project))
                    for n in range(n_permutations)]
    permutations[0] = np.arange(0, len(subjects_to_project), 1)
    r2_null_distribution = dict.fromkeys(list_of_scores)
    for score in list_of_scores:
        print('Prediction for {} test'.format(score))
        prediction_scores_permutation = Parallel(
            n_jobs=14, verbose=1000,
            backend='multiprocessing',
            temp_folder='/media/db242421/db242421_data/ConPagnon_data/tangent_space/joblib_dir')(delayed(predict_scores)(
                connectivity_matrices=patients_tangent_matrices_KBest,
                raw_score=behavioral_scores.loc[subjects_to_project][score][b],
                scoring='neg_mean_squared_error',
                c_grid=c_grid,
                scale_predictors=False,
                alpha=1,
                n_jobs=20) for b in permutations)

        r2_null = np.array([prediction_scores_permutation[n][0] for n in range(n_permutations)])[1:, 0]
        true_r2 = prediction_scores_permutation[0][0]
        true_score_weight = prediction_scores_permutation[0][2]
        true_score_selected_weight_indices = prediction_scores_permutation[0][3]
        common_features = set.intersection(*map(set, true_score_selected_weight_indices))
        r2_null_distribution[score] = {'r2_null': r2_null,
                                       'p': np.sum(r2_null > true_r2) / n_permutations,
                                       'r2': true_r2[0],
                                       'true_score_weights': true_score_weight,
                                       'true_score_selected_features_indices': true_score_selected_weight_indices,
                                       'common_selected_features_indices': common_features}

# Distribution plot for each score
with backend_pdf.PdfPages('/media/db242421/db242421_data/ConPagnon_data/tangent_space/'
                          'r2_null_dist_pca_svr_non_scaled_features.pdf') as pdf:

    for score in list_of_scores:
        plt.figure()
        plt.hist(r2_null_distribution[score]['r2_null'], bins='auto', edgecolor='black')
        plt.axvline(x=r2_null_distribution[score]['r2'], color='black')
        plt.title('R2 null distribution for {} test, p = {}'.format(score, r2_null_distribution[score]['p']))
        pdf.savefig()
        plt.show()


# Illustration , tangent matrices of subjects_to_project
with backend_pdf.PdfPages('/media/db242421/db242421_data/ConPagnon_data/tangent_space/'
                          'patient_matrices_max_abs_scaled.pdf') as pdf:

    for subject in range(len(subjects_to_project)):
        plt.figure()
        plot_matrix(matrix=vec_to_sym_matrix(patients_tangent_matrices_sc[subject, ...],
                                             diagonal=np.zeros(n_nodes)),
                    labels_colors=labels_colors, horizontal_labels=labels_regions,
                    vertical_labels=labels_regions,
                    title='{}'.format(subjects_to_project[subject]))
        pdf.savefig()
        plt.show()

# Illustration of prediction of scores
r2_scores_results = folders_and_files_management.load_object(
    full_path_to_object='/media/db242421/db242421_data/ConPagnon_data/tangent_space/'
                        'predictions_evaluation_f_regression/'
                        'svr_linear_prediction_scores_raw_connectivity_matrices_200_best_features_f_regression.pkl'
)
for score in range(len(list_of_scores)):
    score_prediction_weights = r2_scores_results[list_of_scores[score]]['features weights brain space']
    r2_score = r2_scores_results[list_of_scores[score]]['r2']
    node_contribution = 4
    plt.figure()
    plot_connectome(
        adjacency_matrix=r2_score * vec_to_sym_matrix(np.mean(score_prediction_weights, axis=0),
                                                      diagonal=np.zeros(n_nodes)),
        node_coords=atlas_nodes, node_color=labels_colors,
        title='score: {}, r2: {}'.format(list_of_scores[score], r2_scores_results[list_of_scores[score]]['r2'][0]),
        colorbar=True)
    plt.show()

