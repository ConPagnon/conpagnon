from data_handling import atlas, data_management
from utils.folders_and_files_management import load_object
import numpy as np
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
from patsy import dmatrix
from data_handling import dictionary_operations
from machine_learning.CPM_method import predict_behavior
from scipy.io import savemat
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_pdf import PdfPages
from utils.folders_and_files_management import save_object
from nilearn.plotting import plot_connectome
import pyximport; pyximport.install()
from machine_learning.cythonized_version import CPM_method
import psutil
from plotting.display import plot_matrix
import networkx as nx
import os

# Atlas set up
atlas_folder = '/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/atlas_reference'
atlas_name = 'atlas4D_2.nii'
monAtlas = atlas.Atlas(path=atlas_folder,
                       name=atlas_name)
# Atlas path
atlas_path = monAtlas.fetch_atlas()
# Read labels regions files
labels_text_file = '/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/atlas_reference/atlas4D_2_labels.csv'
labels_regions = monAtlas.GetLabels(labels_text_file)
# User defined colors for labels ROIs regions
colors = ['navy', 'sienna', 'orange', 'orchid', 'indianred', 'olive',
          'goldenrod', 'turquoise', 'darkslategray', 'limegreen', 'black',
          'lightpink']
# Number of regions in each user defined networks
networks = [2, 10, 2, 6, 10, 2, 8, 6, 8, 8, 6, 4]
# Transformation of string colors list to an RGB color array,
# all colors ranging between 0 and 1.
labels_colors = (1./255)*monAtlas.UserLabelsColors(networks=networks,
                                                   colors=colors)
# Fetch nodes coordinates
atlas_nodes = monAtlas.GetCenterOfMass()
# Fetch number of nodes in the parcellation
n_nodes = monAtlas.GetRegionNumbers()

# Load raw and Z-fisher transform matrix
subjects_connectivity_matrices = load_object(
    full_path_to_object='/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/dictionary/'
                        'raw_subjects_connectivity_matrices.pkl')
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
subjects_matrices['patients'] = group_by_factor_subjects_connectivity['D']

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
significance_selection_threshold = 0.003

n_subjects = vectorized_connectivity_matrices.shape[0]
# Features selection by leave one out cross validation scheme
# Clean behavioral data
drop_subject_in_data = ['sub40_np130304']
try:
    regression_data_file = regression_data_file.drop(drop_subject_in_data)
except:
    pass

saving_directory = '/home/db242421/CPM_results_23_05_2018/LD_gender_lesion'
filename = 'LD_gender_lesion_' + str(significance_selection_threshold) + '.pdf'

# Save the matrices for matlab utilisation
# Transpose the shape to (n_features, n_features, n_subjects)
#patients_connectivity_matrices_t = np.transpose(patients_connectivity_matrices, (1, 2, 0))
# Put ones on the diagonal
#for i in range(patients_connectivity_matrices_t.shape[2]):
#    np.fill_diagonal(patients_connectivity_matrices_t[..., i], 1)
# Save the matrix in .mat format
#patients_matrices_dict = {'patients_matrices': patients_connectivity_matrices_t}
#savemat('/media/db242421/db242421_data/CPM_matlab_ver/patients_LesionFlip_mat.mat', patients_matrices_dict)
# Save gender in .mat format
#gender_dict = {'gender': np.array(confounding_variables_data['Sexe'])}
#savemat('/media/db242421/db242421_data/CPM_matlab_ver/gender_LesionFlip_mat.mat', gender_dict)
# Save lesion normalized
#lesion_dict = {'lesion': np.array(confounding_variables_data['lesion_normalized'])}
#savemat('/media/db242421/db242421_data/CPM_matlab_ver/lesion_LesionFlip_mat.mat', lesion_dict)
# Save behavior
#behavior_dict = {'behavior': np.array(behavioral_scores)}
# savemat('/media/db242421/db242421_data/CPM_matlab_ver/behavior_LesionFlip_mat.mat', behavior_dict)

tic = time.time()
(True_R_positive, True_R_negative, all_positive_features, all_negatives_features) = predict_behavior(
    vectorized_connectivity_matrices=vectorized_connectivity_matrices,
    behavioral_scores=np.array(behavioral_scores),
    selection_predictor_method='correlation',
    significance_selection_threshold=significance_selection_threshold,
    confounding_variables_matrix=None,
    add_predictive_variables=add_predictive_variables,
    verbose=0)
tac = time.time()
T = tac - tic

n_permutations = 10
behavioral_scores_permutation_matrix = np.array([np.random.permutation(behavioral_scores)
                                                 for n in range(n_permutations)])

n_core_physical = psutil.cpu_count(logical=False)
n_core_phys_and_log = psutil.cpu_count(logical=True)

if __name__ == '__main__':
    # Permutation test
    tic = time.time()
    results_perm = Parallel(n_jobs=10, verbose=1, backend="multiprocessing")(delayed(predict_behavior)(
        vectorized_connectivity_matrices=vectorized_connectivity_matrices,
        behavioral_scores=behavioral_scores_permutation_matrix[n_perm, ...],
        selection_predictor_method='correlation',
        significance_selection_threshold=significance_selection_threshold,
        confounding_variables_matrix=None,
        add_predictive_variables=add_predictive_variables,
        verbose=0) for n_perm in range(n_permutations))
    tac = time.time()
    T = tac - tic

    null_distribution = np.array([[results_perm[i][0], results_perm[i][1]] for i in range(n_permutations)])

    # Compute p-value
    sorted_positive_null_distribution = sorted(null_distribution[:, 0])
    sorted_negative_null_distribution = sorted(null_distribution[:, 1])

    p_positive = (len(np.where(sorted_positive_null_distribution > True_R_positive)[0]) / (n_permutations + 1))
    p_negative = (len(np.where(sorted_negative_null_distribution > True_R_negative)[0])/ (n_permutations + 1))

    print('For positive model, R_positive = {}'.format(True_R_positive))
    print('For negative model, R_negative = {}'.format(True_R_negative))

    # Save null distribution in pickle format
    save_object(object_to_save=null_distribution,
                saving_directory=saving_directory,
                filename='estimated_null_distribution.pkl')

    # plot on glass brain common negative/positive feature common accros all cross validation
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
                filename='positive_common_features.pkl')

    save_object(object_to_save=negative_sum_mask,
                saving_directory=saving_directory,
                filename='negative_common_features.pkl')

    # Plot of histogram
    with PdfPages(os.path.join(saving_directory, filename)) as pdf:
        plt.figure()
        plt.hist(sorted_positive_null_distribution, 'auto', histtype='bar', normed=True, alpha=0.5, edgecolor='black')
        plt.title('Null distribution of correlation for positive features model \n'
                  'R_pos = {}, p_pos = {}'.format(True_R_positive, p_positive))
        R_positive_thresh = np.percentile(sorted_positive_null_distribution, q=95)
        plt.axvline(x=True_R_positive, color='red')
        plt.axvline(x=R_positive_thresh, color='black')
        plt.legend(['True predicted correlation', '95% threshold correlation'])
        pdf.savefig()
        plt.show()

        plt.figure()
        plt.hist(sorted_negative_null_distribution, 'auto', histtype='bar', normed=True, alpha=0.5, edgecolor='black')
        plt.title('Null distribution of correlation for negative features model \n'
                  'R_neg = {}, p_neg = {}'.format(True_R_negative, p_negative))
        R_negative_thresh = np.percentile(sorted_negative_null_distribution, q=95)
        plt.axvline(x=True_R_negative, color='blue')
        plt.axvline(x=R_negative_thresh, color='black')
        plt.legend(['True predicted correlation', '95% threshold correlation'])
        pdf.savefig()
        plt.show()

        # plot mask on glass brain
        plt.figure()
        plot_connectome(adjacency_matrix=positive_sum_mask, node_coords=atlas_nodes,
                        node_color=labels_colors,edge_cmap='Reds',
                        title='Edges with positive correlation to behavior')
        pdf.savefig()
        plt.show()

        # plot mask on glass brain
        plt.figure()
        plot_connectome(adjacency_matrix=negative_sum_mask, node_coords=atlas_nodes,
                        node_color=labels_colors,edge_cmap='Blues',
                        title='Edges with negative correlation to behavior')
        pdf.savefig()
        plt.show()

        # Plot matrix of common negative and positive features
        plt.figure()
        plot_matrix(matrix=positive_sum_mask, labels_colors=labels_colors, mpart='lower',
                    colormap='Reds', horizontal_labels=labels_regions, vertical_labels=labels_regions,
                    linecolor='black', title='Common edges with positive correlation with behavior')
        pdf.savefig()
        plt.show()

        plt.figure()
        plot_matrix(matrix=negative_sum_mask, labels_colors=labels_colors, mpart='lower',
                    colormap='Blues', horizontal_labels=labels_regions, vertical_labels=labels_regions,
                    linecolor='black', title='Common edges with negative correlation with behavior')
        pdf.savefig()
        plt.show()

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
        plt.show()
        
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
        plt.show()
