"""
This module is designed to study the link between
functional connectivity and behaviour. The algorithm
named connectome predictive modelling is adapted from
[1].

.. [1] Using connectome-based predictive modeling to
predict individual behavior from brain connectivity, Shen et al.

author: Dhaif BEKHA.
# TODO: Break the code into small part !!!!
# TODO: Use GLM with both negative and positive values as predictor of behavioral variable
# TODO: When the predicted value is computed, add possibility to add other variable in the model
# TODO: Compute a network visualisation of the results with networkx for example
# TODO: Estimate model efficience with other metrics such as MSE, instead of simple correlation
# TODO: Estimate p-value of model power with permutations statistic
"""
from data_handling import atlas, data_management
from utils.folders_and_files_management import load_object
import numpy as np
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
from patsy import dmatrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from sklearn.model_selection import LeaveOneOut
from scipy import stats
from nilearn.plotting import plot_connectome
from data_handling import dictionary_operations
import pandas as pd
import seaborn as sns
from machine_learning.CPM_method import predictors_selection_linear_model, fit_model_on_training_set, \
    compute_summary_subjects_summary_values, predictors_selection_correlation, predictor_selection_pcorrelation
from scipy.io import savemat
from connectivity_statistics.parametric_tests import partial_corr
from scipy.stats import t

# Atlas set up
atlas_folder = 'D:\\atlas_AVCnn'
atlas_name = 'atlas4D_2.nii'
monAtlas = atlas.Atlas(path=atlas_folder,
                       name=atlas_name)
# Atlas path
atlas_path = monAtlas.fetch_atlas()
# Read labels regions files
labels_text_file = 'D:\\atlas_AVCnn\\atlas4D_2_labels.csv'
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
    full_path_to_object='D:\\text_output_11042018\\dictionary'
                        '\\raw_subjects_connectivity_matrices.pkl')
Z_subjects_connectivity_matrices = load_object(
    full_path_to_object='D:\\text_output_11042018\dictionary'
                        '\\z_fisher_transform_subjects_connectivity_matrices.pkl')
# Load behavioral data file
regression_data_file = data_management.read_excel_file(
    excel_file_path='D:\\regression_data\\regression_data.xlsx',
    sheetname='cohort_functional_data')

# Type of subjects connectivity matrices
subjects_matrices = subjects_connectivity_matrices

# Select a subset of patients
# Compute the connectivity matrices dictionary with factor as keys.
group_by_factor_subjects_connectivity, population_df_by_factor, factor_keys, =\
    dictionary_operations.groupby_factor_connectivity_matrices(
        population_data_file='D:\\regression_data\\regression_data.xlsx',
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
confounding_variables = ['lesion_normalized', 'Sexe']
confounding_variables_data = regression_data_file[confounding_variables].loc[patients_subjects_ids]
# Encode the confounding variable in an array
confounding_variables_matrix = dmatrix(formula_like='+'.join(confounding_variables), data=confounding_variables_data,
                                       return_type='dataframe').drop(['Intercept'], axis=1)

# Features selection by leave one out cross validation scheme
# Clean behavioral data
drop_subject_in_data = ['sub40_np130304']
try:
    regression_data_file = regression_data_file.drop(drop_subject_in_data)
except:
    pass

# Initialize leave one out object
leave_one_out_generator = LeaveOneOut()

# Selection connectivity threshold
significance_selection_threshold = 0.01

# Initialization of behavior prediction vector
behavior_prediction_positive_edges = np.zeros(len(patients_subjects_ids))
behavior_prediction_negative_edges = np.zeros(len(patients_subjects_ids))

# Choose method to relate connectivity to behavior (predictor selection)
selection_predictor_method = 'partial correlation'

# Save the matrices for matlab utilisation
# Transpose the shape to (n_features, n_features, n_subjects)
# patients_connectivity_matrices_t = np.transpose(patients_connectivity_matrices, (1,2,0))
# Put ones on the diagonal
# for i in range(patients_connectivity_matrices_t.shape[2]):
    # np.fill_diagonal(patients_connectivity_matrices_t[..., i], 1)
# Save the matrix in .mat format
# patients_matrices_dict = {'patients_matrices': patients_connectivity_matrices_t}
# savemat('/media/db242421/db242421_data/CPM_matlab_ver/patients_LG_mat.mat', patients_matrices_dict)
# Save gender in .mat format
# gender_dict = {'gender': np.array(confounding_variables_data['Sexe'])}
# savemat('/media/db242421/db242421_data/CPM_matlab_ver/gender_LG_mat.mat', gender_dict)
# Save lesion normalized
# lesion_dict = {'lesion': np.array(confounding_variables_data['lesion_normalized'])}
# savemat('/media/db242421/db242421_data/CPM_matlab_ver/lesion_LG_mat.mat', lesion_dict)
# Save behavior
# behavior_dict = {'behavior': np.array(behavioral_scores)}
# savemat('/media/db242421/db242421_data/CPM_matlab_ver/behavior_LG_mat.mat', behavior_dict)


for train_index, test_index in leave_one_out_generator.split(vectorized_connectivity_matrices):
    print('Train on {}'.format([patients_subjects_ids[i] for i in train_index]))
    print('Test on {}'.format([patients_subjects_ids[i] for i in test_index]))
    # For each iteration, split the patients matrices array in train and
    # test set using leave one out cross validation
    patients_train_set, leave_one_out_patients = \
        vectorized_connectivity_matrices[train_index], vectorized_connectivity_matrices[test_index]

    # Training set behavioral scores
    training_set_behavioral_score_ = np.zeros((patients_train_set.shape[0], 1))
    training_set_behavioral_score_[:, 0] = behavioral_scores[train_index]

    # Test subject behavioral score
    test_subject_behavioral_score = behavioral_scores[test_index]

    # The confounding variables, stored in an array for the training set
    training_confound_variable_matrix = confounding_variables_matrix.loc[[patients_subjects_ids[i]
                                                                          for i in train_index]]

    if selection_predictor_method == 'linear_model':

        # Correlation of each edge to the behavioral score for training set
        R_mat, P_mat = predictors_selection_linear_model(
            training_connectivity_matrices=patients_train_set,
            training_confound_variable_matrix=training_confound_variable_matrix,
            training_set_behavioral_score=training_set_behavioral_score_)

        # Compute summary values for both positive and negative edges model
        negative_edges_mask, positive_edges_mask, negative_edges_summary_values, positive_edges_summary_values =\
            compute_summary_subjects_summary_values(
                training_connectivity_matrices=patients_train_set,
                significance_selection_threshold=significance_selection_threshold,
                R_mat=R_mat, P_mat=P_mat)

    elif selection_predictor_method == 'correlation':

        R_mat, P_mat = predictors_selection_correlation(training_connectivity_matrices=patients_train_set,
                                                        training_set_behavioral_scores=training_set_behavioral_score_)

        negative_edges_mask, positive_edges_mask, negative_edges_summary_values, positive_edges_summary_values =\
            compute_summary_subjects_summary_values(
                training_connectivity_matrices=patients_train_set,
                significance_selection_threshold=significance_selection_threshold,
                R_mat=R_mat, P_mat=P_mat)

    elif selection_predictor_method == 'partial correlation':

        R_mat, P_mat = predictor_selection_pcorrelation(
            training_connectivity_matrices=patients_train_set,
            training_set_behavioral_scores=training_set_behavioral_score_,
            training_set_confounding_variables=training_confound_variable_matrix)

        negative_edges_mask, positive_edges_mask, negative_edges_summary_values, positive_edges_summary_values =\
            compute_summary_subjects_summary_values(
                training_connectivity_matrices=patients_train_set,
                significance_selection_threshold=significance_selection_threshold,
                R_mat=R_mat, P_mat=P_mat)
    else:
        raise ValueError('Selection method not understood')

    # Reshape the matrices to get indices of regions for plotting purpose
    negatives_edges_matrix = vec_to_sym_matrix(vec=negative_edges_mask,
                                               diagonal=np.zeros(n_nodes))
    positive_edges_matrix = vec_to_sym_matrix(vec=positive_edges_mask,
                                              diagonal=np.zeros(n_nodes))

    # Fit a linear model on the training set
    positive_edge_model_fit, negative_edge_model_fit = fit_model_on_training_set(
        negative_edges_summary_values=negative_edges_summary_values,
        positive_edges_summary_values=positive_edges_summary_values,
        training_set_behavioral_score=training_set_behavioral_score_)

    # Test the positive edges model on the left out subject
    test_subject_positive_edges_summary = np.sum(np.multiply(leave_one_out_patients[0, :], positive_edges_mask))
    test_subject_negative_edges_summary = np.sum(np.multiply(leave_one_out_patients[0, :], negative_edges_mask))

    # Fit the model of on the left out subject
    behavior_prediction_negative_edges[test_index] = \
        negative_edge_model_fit.params[1]*test_subject_negative_edges_summary + \
        negative_edge_model_fit.params[0]

    behavior_prediction_positive_edges[test_index] = \
        positive_edge_model_fit.params[1]*test_subject_positive_edges_summary + positive_edge_model_fit.params[0]

# Compare prediction and true behavioral score
R_predict_negative_model, P_predict_negative_model = \
    stats.pearsonr(x=behavior_prediction_negative_edges,
                   y=np.array(behavioral_scores))

R_predict_positive_model,  P_predict_positive_model = \
    stats.pearsonr(x=np.array(behavioral_scores),
                   y=behavior_prediction_positive_edges)



def predict_behavior(vectorized_connectivity_matrices, behavioral_scores,
                     selection_predictor_method='correlation',
                     significance_selection_threshold=0.01, **kwargs):
    """Predicting behavior scores wih a linear model

    """

    for train_index, test_index in leave_one_out_generator.split(vectorized_connectivity_matrices):
        print('Train on {}'.format([patients_subjects_ids[i] for i in train_index]))
        print('Test on {}'.format([patients_subjects_ids[i] for i in test_index]))
        # For each iteration, split the patients matrices array in train and
        # test set using leave one out cross validation
        patients_train_set, leave_one_out_patients = \
            vectorized_connectivity_matrices[train_index], vectorized_connectivity_matrices[test_index]

        # Training set behavioral scores
        training_set_behavivoral_score_ = np.zeros((patients_train_set.shape[0], 1))
        training_set_behavioral_score_[:, 0] = behavioral_scores[train_index]

        # Test subject behavioral score
        test_subject_behavioral_score = behavioral_scores[test_index]

        # The confounding variables, stored in an array for the training set
        training_confound_variable_matrix = confounding_variables_matrix.loc[[patients_subjects_ids[i]
                                                                              for i in train_index]]

