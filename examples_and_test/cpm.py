from data_handling import atlas, data_management
from utils.folders_and_files_management import load_object
import numpy as np
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
from patsy import dmatrix
from data_handling import dictionary_operations
from machine_learning.CPM_method import predict_behavior
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_pdf import PdfPages
from utils.folders_and_files_management import save_object
from nilearn.plotting import plot_connectome
import psutil
from plotting.display import plot_matrix
import networkx as nx
import os
from joblib import Parallel, delayed

# Atlas set up
atlas_folder = '/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/AICHA_test/references_img'
atlas_name = 'AICHA4D.nii'
monAtlas = atlas.Atlas(path=atlas_folder,
                       name=atlas_name)
# Atlas path
atlas_path = monAtlas.fetch_atlas()
# Read labels regions files
labels_text_file = '/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/AICHA_test/references_img/AICHA4D_labels.csv'
labels_regions = monAtlas.GetLabels(labels_text_file)
# User defined colors for labels ROIs regions
# Transformation of string colors list to an RGB color array,
# all colors ranging between 0 and 1.
labels_colors = monAtlas.RandomNodesLabelsColors()
atlas_nodes = monAtlas.GetCenterOfMass()
# Fetch number of nodes in the parcellation
n_nodes = monAtlas.GetRegionNumbers()

# Load raw and Z-fisher transform matrix
subjects_connectivity_matrices = load_object(
    full_path_to_object='/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/AICHA_test/'
                        'aicha_connectivity_matrices.pkl')
Z_subjects_connectivity_matrices = load_object(
    full_path_to_object='/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/dictionary/'
                        'z_fisher_transform_subjects_connectivity_matrices.pkl')
# Load behavioral data file
regression_data_file = data_management.read_excel_file(
    excel_file_path='/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/regression_data/regression_data.xlsx',
    sheetname='cohort_functional_data')

# Type of subjects connectivity matrices
subjects_matrices = subjects_connectivity_matrices

# Select a subset of patients
# Compute the connectivity matrices dictionary with factor as keys.
group_by_factor_subjects_connectivity, population_df_by_factor, factor_keys, =\
    dictionary_operations.groupby_factor_connectivity_matrices(
        population_data_file='/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/regression_data/regression_data.xlsx',
        sheetname='cohort_functional_data',
        subjects_connectivity_matrices_dictionnary=subjects_matrices,
        groupes=['patients'], factors=['Lesion'], drop_subjects_list=['sub40_np130304'])

subjects_matrices = {}
subjects_matrices['patients'] = group_by_factor_subjects_connectivity['G']

# Fetch patients matrices, and one behavioral score
kind = 'tangent'
patients_subjects_ids = list(subjects_matrices['patients'].keys())
# Patients matrices stack
patients_connectivity_matrices = np.array([subjects_matrices['patients'][s][kind] for
                                           s in patients_subjects_ids])

# Behavioral score
behavioral_scores = regression_data_file['language_score'].loc[patients_subjects_ids]
# Vectorized connectivity matrices of shape (n_samples, n_features)
vectorized_connectivity_matrices = sym_matrix_to_vec(patients_connectivity_matrices, discard_diagonal=True)

# Build confounding variable
confounding_variables = ['Sexe', 'lesion_normalized']
confounding_variables_data = regression_data_file[confounding_variables].loc[patients_subjects_ids]
# Encode the confounding variable in an array
confounding_variables_matrix = dmatrix(formula_like='+'.join(confounding_variables), data=confounding_variables_data,
                                       return_type='dataframe').drop(['Intercept'], axis=1)

add_predictive_variables = confounding_variables_matrix


n_subjects = vectorized_connectivity_matrices.shape[0]
# Features selection by leave one out cross validation scheme
# Clean behavioral data
drop_subject_in_data = ['sub40_np130304']
try:
    regression_data_file = regression_data_file.drop(drop_subject_in_data)
except:
    pass

n_core_physical = psutil.cpu_count(logical=False)
n_core_phys_and_log = psutil.cpu_count(logical=True)

saving_directory = '/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/AICHA_test/CPM_AICHA'


# Study the impact of threshold: choose the best threshold
threshold = np.arange(0.001, 0.05, 0.0001)
r_pos_ = []
r_neg_ = []
threshold_result = Parallel(n_jobs=n_core_physical, verbose=1)(delayed(predict_behavior)(
    vectorized_connectivity_matrices=vectorized_connectivity_matrices,
    behavioral_scores=np.array(behavioral_scores),
    selection_predictor_method='correlation',
    significance_selection_threshold=threshold[t],
    confounding_variables_matrix=None,
    add_predictive_variables=add_predictive_variables,
    verbose=0) for t in range(len(threshold)))

for t in range(len(threshold)):
    r_pos_.append(threshold_result[t][0])
    r_neg_.append(threshold_result[t][1])

r_pos_ = np.array(r_pos_)
r_neg_ = np.array(r_neg_)

# Find the optimum threshold for positive and negative features
threshold_positive_features = round(threshold[np.where(r_pos_ == r_pos_.max())[0][0]], 4)
threshold_negative_features = round(threshold[np.where(r_neg_ == r_neg_.max())[0][0]], 4)

# plot the evolution of correlation of prediction versus the threshold
with PdfPages(os.path.join(saving_directory, 'threshold_effect.pdf')) as pdf:
    plt.figure()
    plt.plot(threshold, r_pos_, 'r', 'o')
    plt.plot(threshold, r_neg_, 'b', 'o')
    plt.axvline(x=threshold_negative_features, color='blue')
    plt.axvline(x=threshold_positive_features, color='red')
    plt.xlabel('p-value threshold')
    plt.ylabel('Correlation between true and predicted scores')
    plt.title('Evolution of correlation between true and predicted score \n')
    plt.legend(['Regions pairs with positive correlation to behavior',
                'Regions pairs with negative correlation to behavior'])
    pdf.savefig()
    plt.show()


print('Maximum correlation between predicted and true scores for edges with '
      'positive correlation with behavior at p = {}, r_positive = {}'.format(threshold_positive_features,
                                                                             r_pos_.max()))
print('Maximum correlation between predicted and true scores for edges with '
      'negative correlation with behavior at p = {}, r_negative = {}'.format(threshold_negative_features,
                                                                             r_neg_.max()))


# Build an array containing the requested number of score permutations, shape (n_permutations, n_subjects)
n_permutations = 10000
behavioral_scores_permutation_matrix = np.array([np.random.permutation(behavioral_scores)
                                                 for n in range(n_permutations)])
significance_selection_threshold = [threshold_positive_features, threshold_negative_features]


if __name__ == '__main__':
    for thresh in significance_selection_threshold:
        # report analysis filename
        filename = 'LG_gender_lesion_' + str(thresh) + '.pdf'

        # Compute true correlation between predicted scores and true scores for
        # positive and negative features
        tic = time.time()
        (True_R_positive, True_R_negative, all_positive_features, all_negatives_features) = predict_behavior(
            vectorized_connectivity_matrices=vectorized_connectivity_matrices,
            behavioral_scores=np.array(behavioral_scores),
            selection_predictor_method='correlation',
            significance_selection_threshold=thresh,
            confounding_variables_matrix=None,
            add_predictive_variables=add_predictive_variables,
            verbose=0)
        tac = time.time()
        T = tac - tic

        # Permutation test
        tic_ = time.time()
        results_perm = Parallel(n_jobs=n_core_physical, verbose=1, backend="multiprocessing")(delayed(predict_behavior)(
            vectorized_connectivity_matrices=vectorized_connectivity_matrices,
            behavioral_scores=behavioral_scores_permutation_matrix[n_perm, ...],
            selection_predictor_method='correlation',
            significance_selection_threshold=thresh,
            confounding_variables_matrix=None,
            add_predictive_variables=add_predictive_variables,
            verbose=0) for n_perm in range(n_permutations))
        tac_ = time.time()
        T_ = tac_ - tic_

        null_distribution = np.array([[results_perm[i][0], results_perm[i][1]] for i in range(n_permutations)])

        # Compute p-value
        sorted_positive_null_distribution = sorted(null_distribution[:, 0])
        sorted_negative_null_distribution = sorted(null_distribution[:, 1])

        p_positive = (len(np.where(sorted_positive_null_distribution > True_R_positive)[0]) / (n_permutations + 1))
        p_negative = (len(np.where(sorted_negative_null_distribution > True_R_negative)[0]) / (n_permutations + 1))

        print('For positive model, R_positive = {}'.format(True_R_positive))
        print('For negative model, R_negative = {}'.format(True_R_negative))

        # Save null distribution in pickle format
        save_object(object_to_save=null_distribution,
                    saving_directory=saving_directory,
                    filename='estimated_null_distribution_' + str(thresh) + '.pkl')

        # plot on glass brain common negative/positive feature common across all cross validation
        # iterations
        positive_features_arrays = vec_to_sym_matrix(np.array(all_positive_features),
                                                     diagonal=np.zeros((n_subjects, n_nodes)))
        negative_features_arrays = vec_to_sym_matrix(np.array(all_negatives_features),
                                                     diagonal=np.zeros((n_subjects, n_nodes)))

        # Find intersection node by summing all edges across subjects
        positive_sum_mask = positive_features_arrays.sum(axis=0)
        negative_sum_mask = negative_features_arrays.sum(axis=0)

        positive_sum_mask[positive_sum_mask != n_subjects] = 0
        negative_sum_mask[negative_sum_mask != n_subjects] = 0

        # Save negative and positive common features array
        save_object(object_to_save=positive_sum_mask,
                    saving_directory=saving_directory,
                    filename='positive_common_features_' + str(thresh) + '.pkl')

        save_object(object_to_save=negative_sum_mask,
                    saving_directory=saving_directory,
                    filename='negative_common_features_' + str(thresh) + '.pkl')

        # Plot of histogram
        with PdfPages(os.path.join(saving_directory, filename)) as pdf:
            plt.figure()
            plt.hist(sorted_positive_null_distribution, 'auto', histtype='bar', normed=True, alpha=0.5,
                     edgecolor='black')
            plt.title('Null distribution of correlation for positive features model \n'
                      'R_pos = {}, p_pos = {}'.format(True_R_positive, p_positive))
            R_positive_thresh = np.percentile(sorted_positive_null_distribution, q=95)
            plt.axvline(x=True_R_positive, color='red')
            plt.axvline(x=R_positive_thresh, color='black')
            plt.legend(['True predicted correlation', '95% threshold correlation'])
            pdf.savefig()
            # plt.show()

            plt.figure()
            plt.hist(sorted_negative_null_distribution, 'auto', histtype='bar', normed=True, alpha=0.5,
                     edgecolor='black')
            plt.title('Null distribution of correlation for negative features model \n'
                      'R_neg = {}, p_neg = {}'.format(True_R_negative, p_negative))
            R_negative_thresh = np.percentile(sorted_negative_null_distribution, q=95)
            plt.axvline(x=True_R_negative, color='blue')
            plt.axvline(x=R_negative_thresh, color='black')
            plt.legend(['True predicted correlation', '95% threshold correlation'])
            pdf.savefig()
            # plt.show()

            # plot mask on glass brain
            plt.figure()
            plot_connectome(adjacency_matrix=positive_sum_mask, node_coords=atlas_nodes,
                            edge_cmap='Reds',
                            title='Edges with positive correlation to behavior',
                            node_size=10)
            pdf.savefig()
            # plt.show()

            # plot mask on glass brain
            plt.figure()
            plot_connectome(adjacency_matrix=negative_sum_mask, node_coords=atlas_nodes,
                            edge_cmap='Blues',
                            title='Edges with negative correlation to behavior',
                            node_size=10)
            pdf.savefig()
            # plt.show()

            # Plot matrix of common negative and positive features
            plt.figure()
            plot_matrix(matrix=positive_sum_mask, mpart='lower',
                        colormap='Reds', horizontal_labels=labels_regions, vertical_labels=labels_regions,
                        linecolor='black', title='Common edges with positive correlation with behavior')
            pdf.savefig()
            # plt.show()

            plt.figure()
            plot_matrix(matrix=negative_sum_mask, mpart='lower',
                        colormap='Blues', horizontal_labels=labels_regions, vertical_labels=labels_regions,
                        linecolor='black', title='Common edges with negative correlation with behavior')
            pdf.savefig()
            # plt.show()

            # Generate a graph objects from adjacency matrix
            positive_features_g = nx.from_numpy_array(A=positive_sum_mask)
            positive_features_edges = positive_features_g.edges()
            positive_features_non_zeros_g = nx.Graph(positive_features_edges)
            positive_features_nodes = positive_features_non_zeros_g.nodes()
            positives_nodes_labels_dict = dict({node: labels_regions[node] for node in positive_features_nodes})
            positives_nodes_labels_colors = dict({node: labels_colors[node] for node in positive_features_nodes})

            plt.figure()
            nx.draw(positive_features_non_zeros_g, labels=positives_nodes_labels_dict,
                    with_labels=True, node_size=10, font_size=5)
            plt.title('Edges with positive correlation with behavior')
            pdf.savefig()
            # plt.show()

            # Generate a graph objects from adjacency matrix
            negative_features_g = nx.from_numpy_array(A=negative_sum_mask)
            negative_features_edges = negative_features_g.edges()
            negative_features_non_zeros_g = nx.Graph(negative_features_edges)
            negative_features_nodes = negative_features_non_zeros_g.nodes()
            negatives_nodes_labels_dict = dict({node: labels_regions[node] for node in negative_features_nodes})
            negatives_nodes_labels_colors = dict({node: labels_colors[node] for node in negative_features_nodes})

            plt.figure()
            nx.draw(negative_features_non_zeros_g, labels=negatives_nodes_labels_dict,
                    with_labels=True, node_size=10, font_size=5)
            plt.title('Edges with negative correlation with behavior')
            pdf.savefig()
            # plt.show()
