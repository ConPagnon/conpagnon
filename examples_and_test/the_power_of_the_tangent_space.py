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

import numpy as np
import matplotlib.pyplot as plt
from nilearn.connectome import vec_to_sym_matrix
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
from computing.compute_connectivity_matrices import tangent_space_projection
from nilearn.image import load_img
import nibabel as nb
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
atlas_folder = '/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/AVCnn_Dhaif/ConPagnon_data/atlas/atlas_reference'
atlas_name = 'atlas4D_2.nii'
labels_text_file = '/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/AVCnn_Dhaif/ConPagnon_data/atlas/' \
                   'atlas_reference/atlas4D_2_labels.csv'
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
    '/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/AVCnn_Dhaif/ConPagnon_data/fmri_images'
# Check all functional files existence
folders_and_files_management.check_directories_existence(
    root_directory=root_fmri_data_directory,
    directories_list=groups)

# Full path, including extension, to the text file containing
# all the subject identifiers.
subjects_ID_data_path = \
    '/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/AVCnn_Dhaif/ConPagnon_data/text_data/' \
    'acm_patients_and_controls_all.txt'

organised_data = data_architecture.fetch_data(
    subjects_id_data_path=subjects_ID_data_path,
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groups,
    individual_counfounds_directory=None
)

# kinds
kinds = ['tangent']

# Nilearn cache directory
nilearn_cache_directory = '/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/AVCnn_Dhaif/nilearn_cache'

# Repetition time between time points in the fmri acquisition, usually 2.4s or 2.5s
repetition_time = 2.5

# Load behavioral scores
behavioral_scores = data_management.read_excel_file(
    excel_file_path='/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/AVCnn_Dhaif/ConPagnon_data/regression_data/'
                    'regression_data_2.xlsx',
    sheetname='cohort_functional_data', subjects_column_name='subjects')
# behavioral_scores = behavioral_scores.set_index(["subjects"])

times_series = ccm.time_series_extraction(
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groups,
    subjects_id_data_path=subjects_ID_data_path,
    reference_atlas=os.path.join(atlas_folder, atlas_name),
    group_data=organised_data,
    repetition_time=repetition_time,
    nilearn_cache_directory=nilearn_cache_directory
)

# Output directory
output_dir = "/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/AVCnn_Dhaif/ConPagnon_data/" \
             "avcnn_language_score_prediction"
report_name = "controls_reference_LG_patients_projected_t_stat.pdf"

# Directory containing all lesion files
lesion_dir = "/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/AVCnn_Dhaif/ConPagnon_data/all_lesion_cleaned"

# Name of the projected group
projected_group_name = "patients"

# Name of the reference group
reference_group_name = "controls"

# select a sub group directly from the time series
# dictionary
groups_by_factor, population_df_by_factor, factor_keys = \
    dictionary_operations.groupby_factor_connectivity_matrices(
        population_data_file='/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/AVCnn_Dhaif/ConPagnon_data/regression_data/'
                             'regression_data_2.xlsx',
        sheetname='cohort_functional_data',
        subjects_connectivity_matrices_dictionnary=times_series,
        groupes=['patients_r'],
        factors=['Groupe', 'Lesion'],
        drop_subjects_list=None)

times_series_ = {'reference_group':  times_series['TDC'],
                 'group_to_project': groups_by_factor[('P', 'G')]}

reference_group = list(times_series_['reference_group'].keys())
subjects_to_project = list(times_series_['group_to_project'].keys())

# Stack the times series for each group
reference_group_time_series = np.array([times_series_['reference_group'][s]['time_series']
                                        for s in reference_group])
group_to_project_time_series = np.array([times_series_['group_to_project'][s]['time_series']
                                         for s in subjects_to_project])

# Number of bootstrap
m = 10000
size_subset_reference_group = 15
alpha = 0.05


tangent_space_projection_dict = tangent_space_projection(
    reference_group=reference_group_time_series,
    group_to_project=group_to_project_time_series,
    bootstrap_number=m,
    bootstrap_size=size_subset_reference_group,
    output_directory=output_dir,
    verif_null=True,
    statistic='t',
    correction_method="fdr_bh",
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

# Initialize a empty adjacency matrix
empty_adjacency_matrix = np.zeros(shape=(n_nodes, n_nodes))

# Lesions overlay
lesion_affine = load_img(os.path.join(lesion_dir, os.listdir(lesion_dir)[0])).affine
overlay_data = np.array([load_img(os.path.join(lesion_dir, lesion_img)).get_data()
                         for lesion_img in os.listdir(lesion_dir)]).sum(axis=0)
plt.figure()
plotting.plot_stat_map(nb.Nifti1Image(overlay_data, affine=lesion_affine),
                       output_file=os.path.join(output_dir, "lesions_overlay.pdf"),
                       title="Lesion overlay")
plt.show()


with backend_pdf.PdfPages(os.path.join(output_dir, report_name)) as pdf:
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

    # Plot tangent mean controls connectome
    plt.figure()
    plot_connectome(adjacency_matrix=reference_group_tangent_mean,
                    node_coords=atlas_nodes,
                    node_color=labels_colors, edge_threshold='80%',
                    title='{} tangent mean'.format(reference_group_name),
                    colorbar=True)
    pdf.savefig()
    plt.show()

# Quantification of t value
positive_connectivity_difference_from_reference = group_to_project_tangent_matrices > 0
negative_connectivity_difference_from_reference = group_to_project_tangent_matrices < 0
# Plot histogram counting for each node the number of time it is superior of inferior to
# the reference group
positive_connectivity_difference_from_reference_m = vec_to_sym_matrix(
    positive_connectivity_difference_from_reference.sum(axis=0), diagonal=np.zeros(n_nodes))
negative_connectivity_difference_from_reference_m = vec_to_sym_matrix(
    negative_connectivity_difference_from_reference.sum(axis=0), diagonal=np.zeros(n_nodes))

# SVM classification
labels = np.array([0 if behavioral_scores.loc[s]['langage_clinique'] == 'N' else 1 for s in subjects_to_project])
sss = StratifiedShuffleSplit(n_splits=10000)
accuracy = cross_val_score(estimator=LinearSVC(C=1), X=group_to_project_tangent_matrices, y=labels,
                           cv=sss, n_jobs=2)
print('Accuracy: {} % +- {} %'.format(round(np.array(accuracy).mean(), 2)*100,
                                      round(np.array(accuracy).std(), 2)*100))

atypical = group_to_project_tangent_matrices[np.where(labels == 1)[0], ...]
typical = group_to_project_tangent_matrices[np.where(labels == 0)[0], ...]

# Prediction of language score based on functional connectivity
list_of_scores = ['parole_zscore', 'lexExp_zscore', 'syntaxExp_zscore',
                  'lexComp_zscore', 'syntaxComp_zscore']

# Scaled the raw behavioral scores
sc_score = StandardScaler(with_mean=False, with_std=False)
all_scores_array = np.array(behavioral_scores.loc[subjects_to_project][list_of_scores])
all_scaled_scores = sc_score.fit_transform(all_scores_array)

# Scaled the features if needed
sc_features = StandardScaler()
maxScaler = MaxAbsScaler()
patients_tangent_matrices_sc = maxScaler.fit_transform(group_to_project_tangent_matrices)

# Selection parameter, if alpha equal to 1, all features are selected
alpha = 1
r2_scores = dict.fromkeys(list_of_scores)
# Optimisation of C parameter for each score
c_grid = [1]
alpha_grid = [1, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000, 3000,
              4000, 5000, 8000, 10000, 100000, 1000000]
# Optimisation of number of features selected by SelectKBest
k_grid = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
          1700, 1800, 1900, 2000, 2100, 2400, 2556]

best_c = dict.fromkeys(list_of_scores)
save_directory = output_dir

import matplotlib
from scipy.stats import pearsonr
colorsList = [(1, 0, 0),(0,0,1)]
CustomCmap = matplotlib.colors.ListedColormap(colorsList)

from machine_learning import scores_predictions
importlib.reload(scores_predictions)

# Permutation testing parameters
n_permutations = 10000
r2_null_distribution = dict.fromkeys(list_of_scores)

if __name__ == '__main__':
    for score in range(len(list_of_scores)):
        print('Computing selection of features and '
              'prediction for {} test....'.format(list_of_scores[score]))
        # Search on a grid, the best K, and the best C parameter minimizing the
        # mean squared error
        K_best = SelectKBest(f_regression)
        pipeline = Pipeline([('K_best_features', K_best), ('svr', SVR(kernel='linear'))])
        grid_search = GridSearchCV(pipeline, {'K_best_features__k': k_grid,
                                              'svr__C': c_grid},
                                   cv=LeaveOneOut(),
                                   scoring='neg_mean_squared_error',
                                   n_jobs=4,
                                   verbose=1)
        grid_search.fit(group_to_project_tangent_matrices, all_scaled_scores[..., score])

        # Reduce the dimensionality of connectivity matrices with the best
        # K selection
        best_K = grid_search.best_params_['K_best_features__k']
        K_best_reducer = SelectKBest(f_regression, k=best_K)
        patients_tangent_matrices_KBest = K_best_reducer.fit_transform(group_to_project_tangent_matrices,
                                                                       all_scaled_scores[..., score])
        # Select the best C parameters from previous selection
        best_parameter = grid_search.best_params_['svr__C']
        best_c[list_of_scores[score]] = {'C': best_parameter}

        # score prediction
        r2, score_pred, score_pred_weights, _ = \
            predict_scores(connectivity_matrices=patients_tangent_matrices_KBest,
                           raw_score=all_scores_array[:, score],
                           alpha=alpha,
                           C=best_c[list_of_scores[score]]['C'],
                           estimator='svr', with_mean=False, with_std=False)

        r2_scores[list_of_scores[score]] = {
            'r2': r2,
            'predicted scores': np.array(score_pred),
            'true scores': all_scaled_scores[:, score],
            'features weights': score_pred_weights,
            'features weights brain space': K_best_reducer.inverse_transform(np.squeeze(np.array(score_pred_weights))),
            'selected features indices': K_best_reducer.get_support(),
            'C': best_c[list_of_scores[score]]['C'],
            'best K': best_K,
            'correlation_score_connectivity': vcorrcoef(X=group_to_project_tangent_matrices.T,
                                                        y=all_scaled_scores[:, score])[0]}

        print('Prediction done for {} test'.format(list_of_scores[score]))

        print('Perform permutation testing on r2 for {}'.format(list_of_scores[score]))

        # Permutation testing on the r2
        permutations = [np.random.choice(np.arange(0, len(subjects_to_project), 1), replace=False,
                                         size=len(subjects_to_project))
                        for n in range(n_permutations)]
        permutations[0] = np.arange(0, len(subjects_to_project), 1)

        prediction_scores_permutation = Parallel(
            n_jobs=4, verbose=1000)(delayed(predict_scores)(
                connectivity_matrices=patients_tangent_matrices_KBest,
                raw_score=behavioral_scores.loc[subjects_to_project][list_of_scores[score]][b],
                alpha=alpha,
                C=best_c[list_of_scores[score]]['C'],
                estimator='svr', with_mean=False, with_std=False) for b in permutations)

        r2_null = np.array([prediction_scores_permutation[n] for n in range(n_permutations)])[1:, 0]
        true_r2 = prediction_scores_permutation[0][0]
        true_score_weight = prediction_scores_permutation[0][2]
        true_score_selected_weight_indices = prediction_scores_permutation[0][3]
        common_features = set.intersection(*map(set, true_score_selected_weight_indices))
        r2_null_distribution[list_of_scores[score]] = {
            'r2_null': r2_null,
            'p': (np.sum(r2_null > true_r2) + 1) / (n_permutations + 1),
            'r2': true_r2,
            'true_score_weights': true_score_weight,
            'true_score_selected_features_indices': true_score_selected_weight_indices,
            'common_selected_features_indices': common_features}

        print('Done permutation testing on r2 for {}'.format(list_of_scores[score]))

    # Save the dictionary containing the results of the prediction
    # for each score
    folders_and_files_management.save_object(
        r2_scores,
        filename='prediction_scores_raw_connectivity_matrices_LinearSVR_K_best_pipeline.pkl',
        saving_directory=output_dir
       )

    with backend_pdf.PdfPages(os.path.join(
            save_directory,
            'prediction_scores_raw_matrices_K_best_SVR_linear_connectome.pdf')) \
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

            node_size = (0.5*np.sum(np.abs(score_prediction_weights), axis=0))*50
            plt.figure()
            plot_connectome(
                adjacency_matrix=masked_correlation_score_connectivity,
                node_coords=atlas_nodes, node_color=labels_colors,
                title='score: {}, r2: {}, K: {}'.format(list_of_scores[score],
                                                        r2_scores[list_of_scores[score]]['r2'],
                                                        number_of_connections),
                colorbar=True,
                node_kwargs={'edgecolor': 'black', 'alpha': 1}, edge_cmap=CustomCmap,
                node_size=node_size)
            pdf.savefig()
            plt.show()

    with backend_pdf.PdfPages(os.path.join(save_directory,
                                           'prediction_scores_raw_matrices_r_pred_vsr_true_LinearSVR_K_best.pdf')) \
            as pdf:
        for score in list_of_scores:

            plt.figure()
            plt.plot(np.array(r2_scores[score]['true scores']),
                     np.array(r2_scores[score]['predicted scores']), 'o')
            plt.title('score: {}, r2: {}'.format(score, r2_scores[score]['r2']))
            plt.xlabel('True scores')
            plt.ylabel('Predicted scores')
            pdf.savefig()
            plt.show()


# Distribution plot for each score
with backend_pdf.PdfPages(os.path.join(output_dir, 'r2_null_Kbest_SVR.pdf')) as pdf:

    for score in list_of_scores:
        plt.figure()
        plt.hist(r2_null_distribution[score]['r2_null'], bins='auto', edgecolor='black')
        plt.axvline(x=r2_null_distribution[score]['r2'], color='black')
        plt.title('R2 null distribution for {} test, r2={}, p = {}'.format(score, r2_null_distribution[score]['r2'],
                                                                           r2_null_distribution[score]['p']))
        pdf.savefig()
        plt.show()


# Illustration , tangent matrices of subjects_to_project
with backend_pdf.PdfPages('/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/AVCnn_Dhaif/ConPagnon_data/tangent_space/'
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
    full_path_to_object='/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/AVCnn_Dhaif/ConPagnon_data/tangent_space/'
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
        title='score: {}, r2: {}'.format(list_of_scores[score], r2_scores_results[list_of_scores[score]]['r2']),
        colorbar=True)
    plt.show()

