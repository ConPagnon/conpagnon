import numpy as np
from pylearn_mulm import mulm
from scipy.stats import t
import statsmodels.api as sm

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
        X_train_edge = np.c_[np.ones(training_connectivity_matrices.shape[0]), training_connectivity_matrices[:, i],
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

    return negative_edges_mask, positive_edges_mask, negative_edges_summary_values, positive_edges_summary_values


def fit_model_on_training_set(negative_edges_summary_values, positive_edges_summary_values,
                              training_set_behavioral_score):
    """Fit a linear model on the training set for positive and negative model, with behavioral score
    as variable.

    """
    # Fit a linear model on the training set, for negative and positive edges summary values
    design_matrix_negative_edges = sm.add_constant(negative_edges_summary_values)
    # design_matrix_negative_edges = np.c_[design_matrix_negative_edges, np.array(training_set_extra_variables)]

    design_matrix_positive_edges = sm.add_constant(positive_edges_summary_values)
    # design_matrix_positive_edges = np.c_[design_matrix_positive_edges, np.array(training_set_extra_variables)]

    # Fit positive edges model
    positive_edges_model = sm.OLS(training_set_behavioral_score, design_matrix_positive_edges)
    positive_edge_model_fit = positive_edges_model.fit()

    # Fit negative edges model
    negative_edges_model = sm.OLS(training_set_behavioral_score, design_matrix_negative_edges)
    negative_edge_model_fit = negative_edges_model.fit()

    return positive_edge_model_fit, negative_edge_model_fit
