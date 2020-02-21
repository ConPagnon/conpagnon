"""
This module is designed to study the link between
functional connectivity and behaviour. The algorithm
named connectome predictive modelling is adapted from
[1].

.. [1] Using connectome-based predictive modeling to
predict individual behavior from brain connectivity, Shen et al.

author: Dhaif BEKHA.
"""
import numpy as np
from conpagnon.pylearn_mulm import mulm
from scipy import stats
from scipy.stats import t
import statsmodels.api as sm
from conpagnon.connectivity_statistics.parametric_tests import partial_corr
from sklearn.model_selection import LeaveOneOut

"""
This module contain useful function for the connectome predictive modelling algorithm.


Author: Dhaif BEKHA.
"""


def predictors_selection_linear_model(training_connectivity_matrices,
                                      training_confound_variable_matrix,
                                      training_set_behavioral_score):
    """Relate each edges of subjects connectivity matrices in the training set
    with a behavioral scores using a linear model


    """
    # Fit linear model at each edge
    n_edge = training_connectivity_matrices.shape[1]
    t_stats = np.zeros(n_edge)
    p_vals = np.zeros(n_edge)
    df_ = np.zeros(n_edge)
    for i in range(n_edge):
        X_train_edge = np.c_[np.ones(training_connectivity_matrices.shape[0]),
                             training_connectivity_matrices[:, i],
                             np.array(training_confound_variable_matrix)]
        # Fit a linear model
        training_set_model = mulm.MUOLS(training_set_behavioral_score, X_train_edge)
        contrasts = np.identity(X_train_edge.shape[1])
        t_value, p_value, df = training_set_model.fit().t_test(contrasts, pval=True, two_tailed=True)
        df_[i] = df[1]

        # Append t value for the principal predictor of interest, i.e the connectivity coefficient
        t_stats[i] = t_value[1, :]
        p_vals[i] = p_value[1, :]

    # For each edges convert the t statistic of the linear model in correlation
    # coefficient value
    R_mat = np.sign(t_stats) * np.sqrt(t_stats**2 / (df_[0] + t_stats**2))
    P_mat = 2*t.sf(np.abs(t_stats), df_[0])

    return R_mat, P_mat


def predictors_selection_correlation(training_connectivity_matrices, 
                                     training_set_behavioral_scores):
    R_mat = np.zeros(training_connectivity_matrices.shape[1])
    P_mat = np.zeros(training_connectivity_matrices.shape[1])

    for i in range(training_connectivity_matrices.shape[1]):
        # Simple correlation between each edges and behavior
        R_mat[i], P_mat[i] = stats.pearsonr(x=training_set_behavioral_scores[:, 0],
                                            y=training_connectivity_matrices[:, i])

    return R_mat, P_mat


def predictor_selection_pcorrelation(training_connectivity_matrices,
                                     training_set_behavioral_scores,
                                     training_set_confounding_variables):
    # Matrix which will contain the correlation of each edge to behavior, and the corresponding
    # p values
    R_mat = np.zeros(training_connectivity_matrices.shape[1])

    # Construct temporary array to contain the connectivity, behavior and
    # other variable to regress
    for i in range(training_connectivity_matrices.shape[1]):
        R_ = partial_corr(np.c_[training_connectivity_matrices[:, i],
                                training_set_behavioral_scores[:, 0],
                                training_set_confounding_variables])
        R_mat[i] = R_[0, 1]

    df = training_connectivity_matrices.shape[0] - 2 - training_set_confounding_variables.shape[1]
    # Compute p values manually: convert R values in t statistic
    t_mat = (np.sqrt(df) * np.abs(R_mat)) / (np.sqrt(np.ones(R_mat.shape[0]) - R_mat ** 2))
    # Compute two tailed p value
    P_mat = t.sf(np.abs(t_mat), df) * 2

    return R_mat, P_mat


def compute_summary_subjects_summary_values(training_connectivity_matrices,
                                            significance_selection_threshold,
                                            R_mat, P_mat):
    # Positive and Negative correlation indices, under the selection threshold
    negative_edges_indices = np.nonzero((R_mat < 0) & (P_mat < significance_selection_threshold))
    positives_edges_indices = np.nonzero((R_mat > 0) & (P_mat < significance_selection_threshold))

    negative_edges_mask = np.zeros(training_connectivity_matrices.shape[1])
    positive_edges_mask = np.zeros(training_connectivity_matrices.shape[1])

    # Fill the corresponding indices with 1 if indices exist, zero elsewhere
    negative_edges_mask[negative_edges_indices] = 1
    positive_edges_mask[positives_edges_indices] = 1

    # Get the sum off all edges in the mask
    negative_edges_summary_values = np.zeros(training_connectivity_matrices.shape[0])
    positive_edges_summary_values = np.zeros(training_connectivity_matrices.shape[0])

    for i in range(training_connectivity_matrices.shape[0]):
        negative_edges_summary_values[i] = np.sum(np.multiply(negative_edges_mask,
                                                              training_connectivity_matrices[i, ...]))
        positive_edges_summary_values[i] = np.sum(np.multiply(positive_edges_mask,
                                                              training_connectivity_matrices[i, ...]))

    return negative_edges_mask, positive_edges_mask, negative_edges_summary_values, \
           positive_edges_summary_values


def fit_model_on_training_set(negative_edges_summary_values, positive_edges_summary_values,
                              training_set_behavioral_score, add_predictive_variables=None):
    """Fit a linear model on the training set for positive and negative model, with behavioral score
    as variable.

    Parameters
    ----------
    negative_edges_summary_values: numpy.array, shape(n_subjects_in_training_set, )
        The sum of predictors for each subject in the training set
        having a negative correlation with the behavioral scores.
    positive_edges_summary_values: numpy.array, shape(n_subjects_in_training_set, )
        The sum of predictors for each subject in the training set
        having a positive correlation with the behavioral scores.
    training_set_behavioral_score: numpy.array, shape (n_subjects_in_training_set, 1)
        The behavioral scores of the training set.
    add_predictive_variables: pandas.DataFrame, optional
        If not None, a pandas DataFrame of shape (n_subjects_in_training_set, n_variables) should
        be given. The variables wil be used in the predictive model.

    """
    # Fit a linear model on the training set, for negative and positive edges summary values
    design_matrix_negative_edges = sm.add_constant(negative_edges_summary_values)

    design_matrix_positive_edges = sm.add_constant(positive_edges_summary_values)

    # if the user add additional to the predictive model we concatenate
    # the additional variables matrix to the existing one containing the intercept
    # and negatives/positive edges summary values.
    if add_predictive_variables is not None:

        design_matrix_negative_edges = np.c_[design_matrix_negative_edges, add_predictive_variables]
        design_matrix_positive_edges = np.c_[design_matrix_positive_edges, add_predictive_variables]
    else:
        design_matrix_negative_edges = design_matrix_negative_edges
        design_matrix_positive_edges = design_matrix_positive_edges

    # Fit positive edges model
    positive_edges_model = sm.OLS(training_set_behavioral_score, design_matrix_positive_edges)
    positive_edge_model_fit = positive_edges_model.fit()

    # Fit negative edges model
    negative_edges_model = sm.OLS(training_set_behavioral_score, design_matrix_negative_edges)
    negative_edge_model_fit = negative_edges_model.fit()

    return positive_edge_model_fit, negative_edge_model_fit


def predict_behavior(vectorized_connectivity_matrices, behavioral_scores,
                     selection_predictor_method='correlation',
                     significance_selection_threshold=0.01,
                     confounding_variables_matrix=None,
                     add_predictive_variables=None,
                     verbose=0):
    """Predict behavior from matrix connectivity with a simple linear model.

    Parameters
    ----------
    vectorized_connectivity_matrices: numpy.array, shape (n_sample, n_features)
        The subjects connectivity matrices in a vectorized form, i.e, you must
        give a array of shape (n_subjects, 0.5*n_features*(n_features-1)).
    behavioral_scores: pandas.DataFrame or numpy.array of shape(n_subjects, 1)
        The behavioral scores in a pandas DataFrame or numpy array form. Off course
        the behavioral scores should ordered in the same order of subjects matrices.
    selection_predictor_method: str, optional
        The selection method of predictors. When relate behavior and functional connectivity
        multiple choices are possible: simple correlation, partial correlation or linear_model.
        Default is correlation. Note that partial correlation or linear model should give
        the same results.
    significance_selection_threshold: float, optional
        The threshold level for the p-value resulting of the selection of predictors
        step. Default is 0.01.
    confounding_variables_matrix: pandas.DataFrame of shape (n_subjects, n_variables), optional
        A dataframe  of shape (n_subjects, n_variables) containing the confounding/controlling
        variable which might be used in the selection predictors step when selection method is
        partial correlation/linear regression. Defaults is None.
    add_predictive_variables: pandas.DataFrame shape (n_subjects_in_training_set, n_variables) or None, optional
        If not None, additional variables will be fitted in the predictive model, besides negative
        and positive summary features.
    verbose: int, optional
        If verbose equal to 0 nothing is printed.

    Returns
    -------
    output 1: float
        The correlation coefficient for the positive features model.
    output 2: float
        The correlation coefficient for the negative features model.




    """
    # Initialize leave one out object
    leave_one_out_generator = LeaveOneOut()

    # Initialization of behavior prediction vector
    behavior_prediction_positive_edges = np.zeros(vectorized_connectivity_matrices.shape[0])
    behavior_prediction_negative_edges = np.zeros(vectorized_connectivity_matrices.shape[0])

    all_selected_positive_features = []
    all_selected_negative_features = []

    for train_index, test_index in leave_one_out_generator.split(vectorized_connectivity_matrices):
        if verbose:
            print('Train on subjects # {}'.format(train_index))
            print('Test on subject # {}'.format(test_index))
        # For each iteration, split the patients matrices array in train and
        # test set using leave one out cross validation
        patients_train_set, leave_one_out_patients = \
            vectorized_connectivity_matrices[train_index], vectorized_connectivity_matrices[test_index]

        # Training set behavioral scores
        training_set_behavioral_score_ = np.zeros((patients_train_set.shape[0], 1))
        training_set_behavioral_score_[:, 0] = behavioral_scores[train_index]

        # The confounding variables, stored in an array for the training set
        if confounding_variables_matrix is not None:
            training_confound_variable_matrix = confounding_variables_matrix.iloc[train_index]
        else:
            training_confound_variable_matrix = None

        # The additional variable for the predictive model
        if add_predictive_variables is not None:
            training_additional_predictive_variables = add_predictive_variables.iloc[train_index]
            test_subject_additional_predictive_variables = add_predictive_variables.iloc[test_index]
        else:
            training_additional_predictive_variables = None
            test_subject_additional_predictive_variables = None

        if selection_predictor_method == 'linear_model':

            # Correlation of each edge to the behavioral score for training set
            R_mat, P_mat = predictors_selection_linear_model(
                training_connectivity_matrices=patients_train_set,
                training_confound_variable_matrix=training_confound_variable_matrix,
                training_set_behavioral_score=training_set_behavioral_score_)

            # Compute summary values for both positive and negative edges model
            negative_edges_mask, positive_edges_mask, negative_edges_summary_values, positive_edges_summary_values = \
                compute_summary_subjects_summary_values(
                    training_connectivity_matrices=patients_train_set,
                    significance_selection_threshold=significance_selection_threshold,
                    R_mat=R_mat, P_mat=P_mat)

            all_selected_negative_features.append(negative_edges_mask)
            all_selected_positive_features.append(positive_edges_mask)

        elif selection_predictor_method == 'correlation':

            R_mat, P_mat = \
                predictors_selection_correlation(training_connectivity_matrices=patients_train_set,
                                                 training_set_behavioral_scores=training_set_behavioral_score_)

            negative_edges_mask, positive_edges_mask, negative_edges_summary_values, positive_edges_summary_values = \
                compute_summary_subjects_summary_values(
                    training_connectivity_matrices=patients_train_set,
                    significance_selection_threshold=significance_selection_threshold,
                    R_mat=R_mat, P_mat=P_mat)

            all_selected_negative_features.append(negative_edges_mask)
            all_selected_positive_features.append(positive_edges_mask)

        elif selection_predictor_method == 'partial correlation':
            if confounding_variables_matrix is None:
                raise ValueError('Confound variables is empty !')
            else:
                R_mat, P_mat = predictor_selection_pcorrelation(
                    training_connectivity_matrices=patients_train_set,
                    training_set_behavioral_scores=training_set_behavioral_score_,
                    training_set_confounding_variables=training_confound_variable_matrix)

                negative_edges_mask, positive_edges_mask, negative_edges_summary_values, \
                    positive_edges_summary_values = \
                    compute_summary_subjects_summary_values(
                        training_connectivity_matrices=patients_train_set,
                        significance_selection_threshold=significance_selection_threshold,
                        R_mat=R_mat, P_mat=P_mat)

                all_selected_negative_features.append(negative_edges_mask)
                all_selected_positive_features.append(positive_edges_mask)
        else:
            raise ValueError('Selection method not understood')

        # Fit a linear model on the training set
        positive_edge_model_fit, negative_edge_model_fit = fit_model_on_training_set(
            negative_edges_summary_values=negative_edges_summary_values,
            positive_edges_summary_values=positive_edges_summary_values,
            training_set_behavioral_score=training_set_behavioral_score_,
            add_predictive_variables=training_additional_predictive_variables)

        # Compute summary statistic for the left out subject
        test_subject_positive_edges_summary = np.sum(np.multiply(leave_one_out_patients[0, :],
                                                                 positive_edges_mask))
        test_subject_negative_edges_summary = np.sum(np.multiply(leave_one_out_patients[0, :],
                                                                 negative_edges_mask))

        # Add additional predictive variables for the test subject
        if add_predictive_variables is not None:
            test_subject_variable_positive_edges = np.c_[np.ones(1), test_subject_positive_edges_summary,
                                                         test_subject_additional_predictive_variables]
            test_subject_variable_negative_edges = np.c_[np.ones(1), test_subject_negative_edges_summary,
                                                         test_subject_additional_predictive_variables]

        else:
            test_subject_variable_positive_edges = np.c_[np.ones(1), test_subject_positive_edges_summary]
            test_subject_variable_negative_edges = np.c_[np.ones(1), test_subject_negative_edges_summary]

        # Fit the model of on the left out subject
        behavior_prediction_negative_edges[test_index] = \
            negative_edge_model_fit.predict(test_subject_variable_negative_edges)

        behavior_prediction_positive_edges[test_index] = \
            positive_edge_model_fit.predict(test_subject_variable_positive_edges)

    # Compare prediction and true behavioral score
    R_predict_negative_model, _ = \
        stats.pearsonr(x=behavior_prediction_negative_edges,
                       y=np.array(behavioral_scores))

    R_predict_positive_model, _ = \
        stats.pearsonr(x=np.array(behavioral_scores),
                       y=behavior_prediction_positive_edges)

    return (R_predict_positive_model, R_predict_negative_model, all_selected_positive_features,
            all_selected_negative_features)
