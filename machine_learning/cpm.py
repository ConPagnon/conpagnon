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
# Atlas set up
atlas_folder = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference'
atlas_name = 'atlas4D_2.nii'
monAtlas = atlas.Atlas(path=atlas_folder,
                       name=atlas_name)
# Atlas path
atlas_path = monAtlas.fetch_atlas()
# Read labels regions files
labels_text_file = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference/atlas4D_2_labels.csv'
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
    full_path_to_object='/media/db242421/db242421_data/ConPagnon_data/CPM/dictionary/'
                        'raw_subjects_connectivity_matrices.pkl')
Z_subjects_connectivity_matrices = load_object(
    full_path_to_object='/media/db242421/db242421_data/ConPagnon_data/CPM/dictionary/'
                        'z_fisher_transform_subjects_connectivity_matrices.pkl')
# Load behavioral data file
regression_data_file = data_management.read_excel_file(
    excel_file_path='/media/db242421/db242421_data/ConPagnon_data/regression_data/regression_data.xlsx',
    sheetname='cohort_functional_data')

# Type of subjects connectivity matrices
subjects_matrices = subjects_connectivity_matrices

# Select a subset of patients
# Compute the connectivity matrices dictionary with factor as keys.
#group_by_factor_subjects_connectivity, population_df_by_factor, factor_keys, =\
#    dictionary_operations.groupby_factor_connectivity_matrices(
#        population_data_file='/media/db242421/db242421_data/ConPagnon_data/regression_data/regression_data.xlsx',
#        sheetname='cohort_functional_data',
#        subjects_connectivity_matrices_dictionnary=subjects_matrices,
#        groupes=['patients'], factors=['Lesion'], drop_subjects_list=['sub40_np130304'])

#subjects_matrices = {}
#subjects_matrices['patients'] = group_by_factor_subjects_connectivity['D']

# Fetch patients matrices, and one behavioral score
kind = 'tangent'
patients_subjects_ids = list(subjects_matrices['LesionFlip'].keys())
# Patients matrices stack
patients_connectivity_matrices = np.array([subjects_matrices['LesionFlip'][s][kind] for
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
significance_selection_threshold = 0.02

n_subjects = vectorized_connectivity_matrices.shape[0]
# Features selection by leave one out cross validation scheme
# Clean behavioral data
drop_subject_in_data = ['sub40_np130304']
try:
    regression_data_file = regression_data_file.drop(drop_subject_in_data)
except:
    pass

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
    add_predictive_variables=None,
    verbose=0)
tac = time.time()
T = tac - tic
# with a for loop
#tic = time.time()
#null_distribution_array = np.zeros((n_permutations, 2))
#for i in range(n_permutations):
#    behavioral_scores_perm = np.random.permutation(behavioral_scores)
#    R_positive_perm, R_negative_perm = predict_behavior(
#        vectorized_connectivity_matrices=vectorized_connectivity_matrices,
#        behavioral_scores=behavioral_scores_perm,
#        selection_predictor_method='correlation',
#        significance_selection_threshold=significance_selection_threshold,
#        confounding_variables_matrix=None,
#        add_predictive_variables=None,
#        verbose=0)
#    null_distribution_array[:, 0] = R_positive_perm
#    null_distribution_array[:, 1] = R_negative_perm
#tac = time.time()
#T = tac - tic

if __name__ == '__main__':
    # Permutation test
    n_permutations = 10000

    tic = time.time()
    results_perm = Parallel(n_jobs=26, verbose=10, backend="multiprocessing")(delayed(predict_behavior)(
        vectorized_connectivity_matrices=vectorized_connectivity_matrices,
        behavioral_scores=np.random.permutation(behavioral_scores),
        selection_predictor_method='correlation',
        significance_selection_threshold=significance_selection_threshold,
        confounding_variables_matrix=None,
        add_predictive_variables=add_predictive_variables,
        verbose=0) for n_perm in range(n_permutations))
    tac = time.time()
    T = tac - tic

    null_distribution = np.array([[results_perm[i][0],results_perm[i][1]] for i in range(n_permutations)])

    # Compute p-value
    sorted_positive_null_distribution = sorted(null_distribution[:, 0])
    sorted_negative_null_distribution = sorted(null_distribution[:, 1])

    p_positive = (len(np.where(sorted_positive_null_distribution > True_R_positive)[0]) / (n_permutations + 1))
    p_negative = (len(np.where(sorted_negative_null_distribution > True_R_negative)[0])/ (n_permutations + 1))

    print('For positive model, R_positive = {}'.format(True_R_positive))
    print('For negative model, R_negative = {}'.format(True_R_negative))

    # Save null distribution in pickle format
    save_object(object_to_save=null_distribution,
                saving_directory='/media/db242421/db242421_data/ConPagnon_data/CPM_results/LD_patients',
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
    # Plot of histogram
    with PdfPages('/media/db242421/db242421_data/ConPagnon_data/CPM_results/LD_patients/LD_patients.pdf') as pdf:
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
