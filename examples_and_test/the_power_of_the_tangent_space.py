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
from nilearn.connectome import ConnectivityMeasure, vec_to_sym_matrix
from statsmodels.stats.multitest import multipletests
from nilearn.plotting import plot_connectome
from scipy.stats import zmap, pearsonr
import networkx as nx
from scipy.stats import ttest_1samp
from nilearn import plotting
from grakel import GraphKernel, datasets, Graph
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

times_series = ccm.time_series_extraction(
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groups,
    subjects_id_data_path=subjects_ID_data_path,
    reference_atlas=os.path.join(atlas_folder, atlas_name),
    group_data=organised_data,
    repetition_time=repetition_time,
    nilearn_cache_directory=nilearn_cache_directory
)

controls = list(times_series['TDC'].keys())
patients = list(times_series['patients_r'].keys())

# Order patients by language performance: worst to best
# patients = ['sub24_ed110159', 'sub20_hd120032', 'sub08_jl110342', 'sub26_as110192', 'sub44_av130474',
#            'sub21_yg120001', 'sub32_mp130025', 'sub14_rs120006', 'sub10_dl120547', 'sub07_lc110496',
#            'sub13_vl110480', 'sub04_rc110343', 'sub39_ya130305', 'sub34_jc130100', 'sub30_zp130008',
#            'sub25_ec110149', 'sub41_sa130332', 'sub43_mc130373', 'sub38_mv130274', 'sub17_eb120007',
#            'sub12_ab110489', 'sub35_gc130101', 'sub37_la130266', 'sub23_lf120459', 'sub06_ml110125']

# Stack the times series for each group
controls_time_series = np.array([times_series['TDC'][s]['time_series'] for s in controls])
patients_time_series = np.array([times_series['patients_r'][s]['time_series'] for s in patients])

# Number of bootstrap
m = 10000
size_subset_controls = 15

# generate null distribution with the controls group
# by leave one out
one_patients_time_series = patients_time_series[0, ...]
null_distribution = []

indices = np.arange(0, len(controls), step=1)
bootstrap_matrix = np.array([np.random.choice(
    indices,
    size=size_subset_controls,
    replace=False) for b in range(m)])

for b in range(m):
    print('Bootstrap # {}'.format(b))
    bootstrap_controls = bootstrap_matrix[b, :]
    # We chose one controls among the bootstrap sample
    left_out_subject = bootstrap_controls[0]
    subset_controls = bootstrap_controls[1:]
    # Compute mean matrices, and connectivity matrices
    # in the tangent on the subset of controls without
    # the leftout subject.
    connectivity_measure = ConnectivityMeasure(kind='tangent',
                                               vectorize=True,
                                               discard_diagonal=True)
    controls_subset_matrices = \
        connectivity_measure.fit_transform(X=controls_time_series[subset_controls, ...])
    controls_subset_mean = connectivity_measure.mean_
    # Project at the previously computed mean
    # the leftout control subject
    leftout_control = \
        connectivity_measure.transform(
            X=[controls_time_series[left_out_subject, ...]])[0]

    # Compute a z-score between the left out controls tangent connectivity
    # and the resampled controls group
    z_null = zmap(scores=leftout_control, compare=controls_subset_matrices)
    # Another statistic: one sample t test, controls - the left out controls
    # as contrast
    t_null = -1.0*ttest_1samp(a=controls_subset_matrices, popmean=leftout_control)[0]

    # Store the results of the test, as null distribution
    null_distribution.append(t_null)

null_distribution_array = np.array(null_distribution)

# plot of some null distribution randomly chosen
n_rois = 10
random_roi = np.random.choice(a=np.arange(start=0, stop=null_distribution_array.shape[1], step=1),
                              size=n_rois)
with backend_pdf.PdfPages('/media/db242421/db242421_data/ConPagnon_data/tangent_space/null_distribution_tangent.pdf') \
        as pdf:

    for i in range(n_rois):
        plt.subplot(5, 2, i+1)
        plt.hist(x=null_distribution_array[random_roi[i], ...], bins='auto', edgecolor='black')
        plt.title('# {}'.format(random_roi[i]))
        plt.subplots_adjust(hspace=1, wspace=.01)
    pdf.savefig()
    plt.show()


# Compute tangent connectivity matrices on the whole controls group
controls_connectivity = ConnectivityMeasure(kind='tangent',
                                            vectorize=True,
                                            discard_diagonal=True)
controls_tangent_matrices = controls_connectivity.fit_transform(X=controls_time_series)
# Compute the tangent mean for the controls group
controls_tangent_mean = controls_connectivity.mean_

# Project each patient's correlation matrices in the
# tangent space, using the controls group tangent mean
patients_tangent_matrices = controls_connectivity.transform(X=patients_time_series)
folders_and_files_management.save_object(
    object_to_save=patients_tangent_matrices,
    saving_directory='/media/db242421/db242421_data/ConPagnon_data/tangent_space',
    filename='patients_tangent_matrices')
# Test for each patient, the difference in connectivity regarding
# the controls group
patients_z_scores = zmap(scores=patients_tangent_matrices, compare=controls_tangent_matrices)
patients_t_scores = np.array([-1.0*ttest_1samp(a=controls_tangent_matrices,
                                               popmean=patients_tangent_matrices[p, ...])[0]
                              for p in range(patients_tangent_matrices.shape[0])])

p_values = np.empty(shape=patients_z_scores.shape)
for patient in range(patients_z_scores.shape[0]):
    for i in range(patients_z_scores.shape[1]):
        # with the z-score, or the t-score
        p_values[patient, i] = \
            np.sum(np.abs(null_distribution_array[:, i]) > np.abs(patients_t_scores[patient, i])) / m
        #    np.sum(np.abs(null_distribution_array[:, i]) > np.abs(patients_z_scores[patient, i])) / m

# correct the p values for each patients
p_values_corrected = np.empty(shape=patients_z_scores.shape)
for patient in range(patients_z_scores.shape[0]):
    p_values_corrected[patient, ...] = multipletests(pvals=p_values[patient, ...],
                                                     method='bonferroni',
                                                     alpha=0.05)[1]

# Count the number of time a node appear across patients
significant_hit_per_nodes = vec_to_sym_matrix(np.sum(p_values_corrected < 0.05, axis=0),
                                              diagonal=np.zeros(n_nodes))
significant_hit_per_nodes_g = nx.from_numpy_array(significant_hit_per_nodes)
significant_node_degree = np.array([val for (node, val) in significant_hit_per_nodes_g.degree()])*10
empty_adjacency_matrix = np.zeros(shape=(n_nodes, n_nodes))

plt.figure()
plot_connectome(adjacency_matrix=empty_adjacency_matrix,
                node_coords=atlas_nodes,
                node_color=labels_colors,
                node_size=significant_node_degree,
                title='Common nodes across patients')
plt.show()

# Load behavioral scores
behavioral_scores = data_management.read_excel_file(
    excel_file_path='/media/db242421/db242421_data/ConPagnon_data/regression_data/'
                    'Resting State AVCnn_cohort data.xlsx',
    sheetname='Middle Cerebral Artery+controls')

with backend_pdf.PdfPages('/media/db242421/db242421_data/ConPagnon_data/tangent_space/'
                          'patient_t_score_bonf_dsigma__.pdf') as pdf:
    for patient in range(len(patients)):
        if patients[patient] in behavioral_scores.index:

            # compute node degree for each subject
            # based on the surviving connection
            patient_significant_edges = vec_to_sym_matrix(p_values_corrected[patient, ...] < 0.05,
                                                          diagonal=np.zeros(n_nodes))
            patient_adjacency_matrices = nx.from_numpy_array(patient_significant_edges)
            degrees = np.array([val for (node, val) in patient_adjacency_matrices.degree()])*40

            # plot corrected connection
            plt.figure()
            plot_connectome(

                adjacency_matrix=vec_to_sym_matrix(np.multiply(p_values_corrected[patient, ...] < 0.05,
                                                               patients_tangent_matrices[patient, ...]),
                                                   diagonal=np.zeros(n_nodes)),
                node_coords=atlas_nodes,
                node_color=labels_colors,
                title='{}, Lesion: {}, Language: {}, Speech: {}, PC1: {}'.format(
                    patients[patient][0:5],
                    behavioral_scores.loc[patients[patient]]['Lesion'],
                    behavioral_scores.loc[patients[patient]]['langage_clinique'],
                    behavioral_scores.loc[patients[patient]]['Parole'],
                    round(behavioral_scores.loc[patients[patient]]['pc1_language'], 3)),
                colorbar=True,
                node_size=degrees,
                node_kwargs={'edgecolor': 'black', 'alpha': 1},
                edge_threshold=None)
            pdf.savefig()
            plt.show()

            plt.figure()
            view = plotting.view_connectome(
                adjacency_matrix=vec_to_sym_matrix(np.multiply(p_values_corrected[patient, ...] < 0.05,
                                                               patients_t_scores[patient, ...]),
                                                   diagonal=np.zeros(n_nodes)),
                coords=atlas_nodes,
                cmap='bwr')
            view.save_as_html(os.path.join('/media/db242421/db242421_data/ConPagnon_data/tangent_space',
                                           patients[patient] + '_t_score.html'))
    # Plot tangent mean controls connectome
    plt.figure()
    plot_connectome(adjacency_matrix=controls_tangent_mean, node_coords=atlas_nodes,
                    node_color=labels_colors, edge_threshold='80%', title='Controls tangent mean',
                    colorbar=True)
    pdf.savefig()
    plt.show()


sc = StandardScaler()
absolute_max_distance = np.max(np.abs(patients_tangent_matrices), axis=1)
X = sm.add_constant(absolute_max_distance)
y = np.array(behavioral_scores.loc[patients]['pc1_language_zscores'])
dist_tangent = sm.OLS(y, X).fit()
dist_tangent.summary()


plt.figure()
plt.plot(np.max(np.abs(patients_tangent_matrices), axis=1),
         behavioral_scores.loc[patients]['pc1_language_zscores'], 'o')
plt.xlabel('Absolute max distance to controls')
plt.ylabel('PC1 language (mean performance score)')
plt.show()

all_adjacency_matrix = [vec_to_sym_matrix(np.array(p_values_corrected[patient, ...] < 0.05, dtype=int),
                                          diagonal=np.zeros(n_nodes)) for patient in range(len(patients))]
all_edge_weight_matrix = [vec_to_sym_matrix(np.multiply(np.array(p_values_corrected[patient, ...] < 0.05, dtype=int),
                                                        patients_tangent_matrices[patient, ...]),
                                            diagonal=np.zeros(n_nodes)) for patient in range(len(patients))]


all_patients_graph = [nx.from_numpy_array(all_edge_weight_matrix[p]) for p in range(len(patients))]

# For each patients measure the similarity of the graph versus
# the other
node_dictionary = {node_number: labels_regions[node_number] for node_number in range(n_nodes)}
all_graph_for_grakel = [[[all_adjacency_matrix[p], node_dictionary]] for p in range(len(patients))]

# Build Graph representation suitable for most of the Grakel algorithm
my_graphs = [Graph(list(all_patients_graph[i].edges()), node_labels=node_dictionary,
                edge_labels={edge: all_edge_weight_matrix[i][edge] for edge in all_patients_graph[i].edges()})
             for i in range(len(patients))]

# without weight on edges, simply adjacency matrix, and node labels.
my_graphs2 = [Graph(all_adjacency_matrix[i], node_labels=node_dictionary) for i in range(len(patients))]


kernel = GraphKernel(kernel={'name': 'neighborhood_hash'}, normalize=True)
result_kernel = kernel.fit_transform(my_graphs2)

plt.figure()
plot_matrix(matrix=result_kernel, labels_colors='auto', mpart='all',
            horizontal_labels=patients, vertical_labels=patients)
plt.show()
# Classification based on the previous computed kernel
svc = SVC(kernel='precomputed')


graph_array = np.array(my_graphs2)

kernel = GraphKernel(kernel={'name': 'shortest_path'}, normalize=True)
nh = kernel.fit_transform(X=my_graphs2)

for patient in range(len(patients)):

    print('Request subject: {}'.format(patients[patient]))
    print('--------------')
    most_similar = np.argsort(nh[patient, ...])[-2]
    print('Most similar subject: {}'.format(patients[most_similar]))
    print('\n')
    with backend_pdf.PdfPages('/media/db242421/db242421_data/ConPagnon_data/tangent_space/'
                          'most_similar_subjects_pairs.pdf') as pdf:

        patient_significant_edges = vec_to_sym_matrix(p_values_corrected[patient, ...] < 0.05,
                                                      diagonal=np.zeros(n_nodes))
        patient_adjacency_matrices = nx.from_numpy_array(patient_significant_edges)
        degrees = np.array([val for (node, val) in patient_adjacency_matrices.degree()]) * 40

        most_similar_patient_significant_edges = vec_to_sym_matrix(p_values_corrected[most_similar, ...] < 0.05,
                                                      diagonal=np.zeros(n_nodes))
        most_similar_patient_adjacency_matrices = nx.from_numpy_array( most_similar_patient_significant_edges)
        most_similar_degrees = np.array([val for (node, val) in most_similar_patient_adjacency_matrices.degree()])*40

        # plot corrected connection
        plt.figure()
        plot_connectome(

            adjacency_matrix=vec_to_sym_matrix(np.multiply(p_values_corrected[patient, ...] < 0.05,
                                                           patients_tangent_matrices[patient, ...]),
                                               diagonal=np.zeros(n_nodes)),
            node_coords=atlas_nodes,
            node_color=labels_colors,
            title='{}, Lesion: {}, Language: {}, Speech: {}, PC1: {}'.format(
                patients[patient][0:5],
                behavioral_scores.loc[patients[patient]]['Lesion'],
                behavioral_scores.loc[patients[patient]]['langage_clinique'],
                behavioral_scores.loc[patients[patient]]['Parole'],
                round(behavioral_scores.loc[patients[patient]]['pc1_language'], 3)),
            colorbar=True,
            node_size=degrees,
            node_kwargs={'edgecolor': 'black', 'alpha': 1},
            edge_threshold=None)
     #   pdf.savefig()
        plt.show()
        plt.figure()
        plot_connectome(

            adjacency_matrix=vec_to_sym_matrix(np.multiply(p_values_corrected[most_similar, ...] < 0.05,
                                                           patients_tangent_matrices[most_similar, ...]),
                                               diagonal=np.zeros(n_nodes)),
            node_coords=atlas_nodes,
            node_color=labels_colors,
            title='{}, Lesion: {}, Language: {}, Speech: {}, PC1: {}'.format(
                patients[most_similar][0:5],
                behavioral_scores.loc[patients[most_similar]]['Lesion'],
                behavioral_scores.loc[patients[most_similar]]['langage_clinique'],
                behavioral_scores.loc[patients[most_similar]]['Parole'],
                round(behavioral_scores.loc[patients[most_similar]]['pc1_language'], 3)),
            colorbar=True,
            node_size=most_similar_degrees,
            node_kwargs={'edgecolor': 'black', 'alpha': 1},
            edge_threshold=None)
     #   pdf.savefig()
        plt.show()

# SVM classification
labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
sss = StratifiedShuffleSplit(n_splits=10000)
accuracy = cross_val_score(estimator=LinearSVC(C=1), X=patients_tangent_matrices, y=labels,
                           cv=sss, n_jobs=6)
print('Accuracy: {} % +- {} %'.format(round(np.array(accuracy).mean(), 2)*100,
                                      round(np.array(accuracy).std(), 2)*100))

atypical = patients_tangent_matrices[np.where(labels == 1)[0], ...]
typical = patients_tangent_matrices[np.where(labels == 0)[0], ...]

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
all_scores_array = np.array(behavioral_scores.loc[patients][list_of_scores])
all_scaled_scores = sc_score.fit_transform(all_scores_array)

# Scaled the features if needed
sc_features = StandardScaler()
maxScaler = MaxAbsScaler()
patients_tangent_matrices_sc = maxScaler.fit_transform(patients_tangent_matrices)

# PCA on connectivity matrices
pca = decomposition.PCA(n_components=0.95)
pca.fit(patients_tangent_matrices_sc)
patients_tangent_matrices_pca = pca.transform(patients_tangent_matrices_sc)

# Select features on raw connectivity matrices, based on a mask
# computed with the scaled matrices with the maximum
maximum_features_mask = np.where(np.abs(patients_tangent_matrices_sc) == 1)
patients_tangent_matrices_masked = \
    [patients_tangent_matrices[p, np.where(np.abs(patients_tangent_matrices_sc[p, ...]) == 1)]
     for p in range(len(patients))]

# Selection parameter, if alpha equal to 1, all features are selected
alpha = 1
r2_scores = dict.fromkeys(list_of_scores)
# Optimisation of C parameter for each score
c_grid = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 1, 10, 50, 100, 200, 300, 500, 1000, 5000, 10000, 100000, 1000000]
# Optimisation of number of features selected by SelectKBest
k_grid = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
          1700, 1800, 1900, 2000, 2100, 2400, 2556]
# TODO: Try PCA features reduction
best_c = dict.fromkeys(list_of_scores)
if __name__ == '__main__':

    for score in range(len(list_of_scores)):
        print('Computing selection of features and '
              'prediction for {} test....'.format(list_of_scores[score]))
        # Search on a grid, the best K, and the best C parameter minimizing the
        # mean squared error
        K_best = SelectKBest(f_regression)
        pipeline = Pipeline([('K_best_features__k', K_best), ('svr', SVR(kernel='linear'))])
        grid_search = GridSearchCV(pipeline, {'K_best_features__k': k_grid,
                                              'svr__C': c_grid},
                                   cv=LeaveOneOut(),
                                   scoring='neg_mean_squared_error',
                                   n_jobs=14,
                                   verbose=1)
        grid_search.fit(patients_tangent_matrices, all_scaled_scores[..., score])

        # Reduce the dimensionality of connectivity matrices with the best
        # K selection
        best_K = grid_search.best_params_['K_best_features__k']
        K_best_reducer = SelectKBest(f_regression, k=best_K)
        patients_tangent_matrices_KBest = K_best_reducer.fit_transform(patients_tangent_matrices,
                                                                       all_scaled_scores[..., score])
        # Select the best C parameters from previous selection
        best_C = grid_search.best_params_['svr__C']
        best_c[list_of_scores[score]] = {'C': best_C}

        # score prediction
        r2, score_pred, score_pred_weights, score_selected_features = \
            predict_scores(connectivity_matrices=patients_tangent_matrices_KBest,
                           raw_score=all_scores_array[:, score],
                           c_grid=c_grid,
                           scoring='neg_mean_squared_error',
                           alpha=alpha,
                           optimize_regularization=False,
                           C=best_c[list_of_scores[score]]['C'],
                           n_jobs=14)
        r2_scores[list_of_scores[score]] = {'r2': r2,
                                            'predicted scores': np.array(score_pred),
                                            'true scores': all_scaled_scores[:, score],
                                            'features weights': score_pred_weights,
                                            'features weights brain space': K_best_reducer.inverse_transform(
                                                np.squeeze(np.array(score_pred_weights))),
                                            'selected features indices': score_selected_features,
                                            'C': best_c[list_of_scores[score]]['C'],
                                            'best K': best_K}

        print('Prediction done for {} test'.format(list_of_scores[score]))
    r2_scores = folders_and_files_management.load_object(
        '/media/db242421/db242421_data/ConPagnon_data/tangent_space/'
        'prediction_evaluation_pipeline_K_best_SVR/svr_linear_prediction_scores_raw_connectivity_matrices_f_regression_pipeline.pkl'
    )
    with backend_pdf.PdfPages('/media/db242421/db242421_data/ConPagnon_data/tangent_space/'
                              'prediction_evaluation_pipeline_K_best_SVR/'
                              'prediction_scores_raw_matrices_f_regression_SVR_linear_connectome_v2.pdf') \
            as pdf:
        for score in range(len(list_of_scores)):
            r2_score = r2_scores[list_of_scores[score]]['r2']
            score_prediction_weights = r2_score*vec_to_sym_matrix(
                np.mean(r2_scores[list_of_scores[score]]['features weights brain space'], axis=0),
                diagonal=np.zeros(n_nodes))

            number_of_connections = r2_scores[list_of_scores[score]]['best K']
            node_size = (0.5*np.sum(np.abs(score_prediction_weights), axis=0))*100
            plt.figure()
            plot_connectome(
                adjacency_matrix=score_prediction_weights,
                node_coords=atlas_nodes, node_color=labels_colors,
                title='score: {}, r2: {}, K: {}'.format(list_of_scores[score],
                                                        r2_scores[list_of_scores[score]]['r2'][0],
                                                        number_of_connections),
                colorbar=True,
                node_kwargs={'edgecolor': 'black', 'alpha': 1})
            pdf.savefig()
            plt.show()

    with backend_pdf.PdfPages(('/media/db242421/db242421_data/ConPagnon_data/tangent_space/'
                               'prediction_evaluation_pipeline_SVR/'
                               'prediction_scores_raw_matrices_regression_SVR_linear_.pdf')) \
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
        '/media/db242421/db242421_data/ConPagnon_data/tangent_space/',
        'svr_linear_prediction_scores_raw_connectivity_matrices_PCA_pipeline.pkl')

    # TODO permuation testing !!!!!!
    n_permutations = 10001
    permutations = [np.random.choice(np.arange(0, len(patients), 1), replace=False, size=len(patients))
                    for n in range(n_permutations)]
    permutations[0] = np.arange(0, len(patients), 1)
    r2_null_distribution = dict.fromkeys(list_of_scores)
    for score in list_of_scores:
        print('Prediction for {} test'.format(score))
        prediction_scores_permutation = Parallel(
            n_jobs=14, verbose=1000,
            backend='multiprocessing',
            temp_folder='/media/db242421/db242421_data/ConPagnon_data/tangent_space/joblib_dir')(delayed(predict_scores)(
                connectivity_matrices=patients_tangent_matrices_KBest,
                raw_score=behavioral_scores.loc[patients][score][b],
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


# Illustration , tangent matrices of patients
with backend_pdf.PdfPages('/media/db242421/db242421_data/ConPagnon_data/tangent_space/'
                          'patient_matrices_max_abs_scaled.pdf') as pdf:

    for patient in range(len(patients)):
        plt.figure()
        plot_matrix(matrix=vec_to_sym_matrix(patients_tangent_matrices_sc[patient, ...],
                                             diagonal=np.zeros(n_nodes)),
                    labels_colors=labels_colors, horizontal_labels=labels_regions,
                    vertical_labels=labels_regions,
                    title='{}'.format(patients[patient]))
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
