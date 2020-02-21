import numpy as np
from patsy import dmatrix
from sklearn.model_selection import LeaveOneOut
from scipy import stats
import pandas as pd
from conpagnon.machine_learning.CPM_method import predictors_selection_linear_model, fit_model_on_training_set, \
    compute_summary_subjects_summary_values, predictors_selection_correlation, predictor_selection_pcorrelation


def predict_behavior(vectorized_connectivity_matrices, behavioral_scores,
                     selection_predictor_method='correlation',
                     significance_selection_threshold=0.01,
                     confounding_variables=None, confounding_variables_kwarg=None):
    """The Connectome Predictive Modelling pipeline. This function select the predictors,
    train/test a linear model on the selected predictors following a Leave One Out cross
    validation scheme.

    Parameters
    ----------
    vectorized_connectivity_matrices: numpy.array of shape (n_subjects, n_features)
        The stack of the vectorized (lower or upper triangle of the connectivity matrices)
        connectivity matrices. Be careful, the matrices should be stack in the same order
        as the vector of scores to predict !
    behavioral_scores: numpy.array of shape (n_subject, 1)
        The vector of scores to predict. The scores should be in the same
        order as the vectorized connectivity matrices stack.
    selection_predictor_method: str, optional
        The predictors selection method. By default, a correlation between
        each connectivity coefficient and scores is computed, and the resulted
        correlation matrices is threshold at a type I error rate equal to 0.01.
        Other selection are available: 'linear_model', 'partial correlation'.
    significance_selection_threshold: float, optional
        The significance threshold during the selection procedure. By default,
        set to 0.01.
    confounding_variables: list, optional
        A list of the possible confounding variables you might
        want to add, during the selection procedure only.
    confounding_variables_kwarg: dict, optional
        A dictionary with a field called 'file_path'. This field
        should contains the full path to a file containing as
        many columns as confounding variable.

    Returns
    -------
    output 1: float
        The correlation coefficient between the predicted and true scores
        from the positively correlated set of features.
    output 2: float
        he correlation coefficient between the predicted and true scores
        from the negatively correlated set of features.

    """
    # Initialize leave one out object
    leave_one_out_generator = LeaveOneOut()

    # Initialization of behavior prediction vector
    behavior_prediction_positive_edges = np.zeros(len(vectorized_connectivity_matrices.shape[0]))
    behavior_prediction_negative_edges = np.zeros(len(vectorized_connectivity_matrices.shape[0]))

    # Date preprocessing
    if confounding_variables is not None:
        # read confounding variable file
        if confounding_variables_kwarg['file_path'].endswith('csv', 'txt'):
            # Read the text file, and fetch the columns corresponding to the confounding variables
            confounding_variables_data = pd.read_csv(confounding_variables_kwarg['file_path'])[confounding_variables]
            # Construct the design matrix containing the confound variables
            confounding_variables_matrix = dmatrix(formula_like='+'.join(confounding_variables),
                                                   data=confounding_variables_data,
                                                   return_type='dataframe').drop(['Intercept'], axis=1)

        elif confounding_variables_kwarg['file_path'].endswith('xlsx'):
            confounding_variables_data = pd.read_excel(confounding_variables_kwarg['file_path'])[confounding_variables]
            # Construct the design matrix containing the confound variables
            confounding_variables_matrix = dmatrix(formula_like='+'.join(confounding_variables),
                                                   data=confounding_variables_data,
                                                   return_type='dataframe').drop(['Intercept'], axis=1)

        else:
            raise ValueError('Datafile extension unrecognized')

    for train_index, test_index in leave_one_out_generator.split(vectorized_connectivity_matrices):
        # For each iteration, split the patients matrices array in train and
        # test set using leave one out cross validation
        patients_train_set, leave_one_out_patients = \
            vectorized_connectivity_matrices[train_index], vectorized_connectivity_matrices[test_index]

        # Training set behavioral scores
        training_set_behavioral_score_ = np.zeros((patients_train_set.shape[0], 1))
        training_set_behavioral_score_[:, 0] = behavioral_scores[train_index]

        # The confounding variables, stored in an array for the training set
        training_confound_variable_matrix = confounding_variables_matrix.iloc[train_index]

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

            R_mat, P_mat = \
                predictors_selection_correlation(training_connectivity_matrices=patients_train_set,
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
    R_predict_negative_model, _ = \
        stats.pearsonr(x=behavior_prediction_negative_edges,
                       y=np.array(behavioral_scores))

    R_predict_positive_model, _ = \
        stats.pearsonr(x=np.array(behavioral_scores),
                       y=behavior_prediction_positive_edges)

    return R_predict_positive_model, R_predict_negative_model


