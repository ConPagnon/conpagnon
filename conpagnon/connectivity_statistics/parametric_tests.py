import importlib
from conpagnon.utils import pre_preprocessing, folders_and_files_management
from conpagnon.utils import array_operation
from conpagnon.utils.array_operation import vectorizer
from scipy.stats import mstats, norm, ttest_ind, pearsonr, ttest_rel
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
import statsmodels.api as sm
import warnings
from conpagnon.computing import compute_connectivity_matrices as ccm
from patsy import dmatrix, dmatrices
import pandas as pd
from conpagnon.pylearn_mulm import mulm
import os
import errno
import copy
from scipy import linalg
import statsmodels.formula.api as smf
importlib.reload(pre_preprocessing)
importlib.reload(array_operation)
importlib.reload(ccm)
importlib.reload(mulm)

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:17:21 2017
@author: db242421 (dhaif.bekha@cea.fr)
Parametric test used in a resting state group connectivity analysis.
"""


def two_samples_t_test(subjects_connectivity_matrices_dictionnary, groupes, kinds, contrast,
                       preprocessing_method='fisher',
                       alpha=.05, multicomp_method='fdr_bh'):
    """Perform two samples t-test on connectivity matrices to detect group differences in connectivity
    using different kinds.

    The t-test account for discarded rois you might want to exclude in the analysis.

    Parameters
    ----------
    subjects_connectivity_matrices_dictionnary : dict
        A multi-levels dictionnary organised as follow :
            - The first keys levels is the different groupes in the study.
            - The second keys levels is the subjects IDs
            - The third levels is the different kind matrices
            for each subjects, a 'discarded_rois' key for the
            discarded rois array index, a 'masked_array' key containing
            the array of Boolean of True for the discarded_rois index, and False
            elsewhere.

    groupes : list
        The list of the two groups to detect group differences.

    kinds : list
        The list of metrics you want to perform the group comparison.
        Choices are : 'correlation', 'covariances', 'tangent', 'partial correlation',
        'precision'.
    preprocessing_method : string, optional
        The type of preprocessing methods to apply of connectivity coefficients
        of type 'correlation', 'partial correlation', 'covariances', 'precision'.
        Choices are : 'fisher'.
    contrast : list, optional
        The contrast you want to compute in the t-test. Default is [1.0, -1.0]
        to compute mean(groupes[0]) - mean(groupes[1]).
        The other contrast is [-1.0, 1.0] for mean(groupes[1]) - mean(groupes[0]).
    alpha : float, optional
        The false positive proportion, commonly named the alpha level.
        Default is 0.05.
    multicomp_method : str, optional
        The inference method for accounting the multiple comparison problems.
        Default is the classic False Discovery Rate (FDR) proposed by
        Benjamini & Hochberg, see Notes.


    Returns
    -------
    output : dict
        A dictionnary containing multiple keys :
            - 'tstatistic' : The raw statistic t-map for the chosen contrast
            2D numpy array of shape (number of regions, number of regions).

            - 'uncorrected pvalues' : The raw pvalues, 2D numpy array of shape
            (number of regions, number of regions).

            - 'corrected pvalues' : The corrected pvalues with the chosen methods.
            2D numpy array of shape (number of regions, number of regions).

            - 'significant edges' : The significant t-values after masking
            for the non significant pvalues at alpha level. 2D numpy array of shape
            (number of regions, number of regions).

            - 'significant pvalues' : The significant pvalues at level alpha.
            2D numpy array of shape (number of regions, number of regions)

            - 'significant mean effect' : The differences of mean connectivity
            between the two groups according to the chosen contrast, and non-significant
            connexion are mask at alpha level. 2D numpy array of shape
            (number of regions, number of regions).

    Raises
    ------
    ValueError : If the number in groupes is strictly less than 2 raise
    a ValueError, print a warning and take the two first groupes in the list
    else.

    ValueError : If the contrast is an unrecognized contrast is entered.

    See Also
    --------

    compute_connectivity_matrices.individual_connectivity_matrices :
        This is the function which compute the connectivity matrices for
        the chosen kinds, and return a structured dictionnary with can be used
        for the argument `subjects_connectivity_matrices_dictionnary`.

    pre_preprocessing.fisher_transform :
        Compute the Fisher Transform of connectivity coefficient

    Notes
    -----
    The two sample t-test I perform here account for masked array using
    statistical test for masked array in the Scipy packages. That is,
    if in the subjects masked_array, some regions are masked, they will be
    discarded when performing the t-test. If the mask contain only False
    value, no region are discarded.

    All the metric are symmetric, therefore I only compute the t-test
    for the lower part of the matrices of shape n_columns * (n_columns + 1) /2

    The correction for the multiple comparison is applied to the resulting
    t-values, and entire matrices are reconstructed after and resulting shape
    is (number of regions, number of regions)

    Different method for accounting the multiple comparison problems exist,
    please refers to the corresponding function in the statsmodels library :
    statsmodels.sandbox.stats.multicomp.multipletests.

    For the tangent kinds, keep in mind that the mean effect is computed
    in the tangent space only, that is, not in the same space as classical
    metrics like covariances, correlation of partial correlation.

    """

    stacked_matrices = pre_preprocessing.stacked_connectivity_matrices(subjects_connectivity_matrices=
                                                                       subjects_connectivity_matrices_dictionnary,
                                                                       kinds=kinds)

    # Initialise a dictionnary for saving the t_statistic, and the uncorrected and corrected p-value
    t_test_dictionnary = dict.fromkeys(kinds)

    # Check if they are at least two samples:
    n_groupes = len(groupes)

    if n_groupes < 2:
        raise ValueError('Two samples t-test requires at '
                         'least two samples...only found {} group'.format(n_groupes))
    elif n_groupes > 2:
        warnings.warn('{} groups was found ! Only the first two groups will be use '
                      'for the two sample t-test: {} and {}'.format(n_groupes,
                                                                    groupes[0],
                                                                    groupes[1]))

    if contrast == [1.0, -1.0]:
        print('Computing two sample t-test for kinds {} and contrast {} - {}'.format(kinds, groupes[0], groupes[1]))
    elif contrast == [-1.0, 1.0]:

        print('Computing two sample t-test for kinds {} and contrast {} - {}'.format(kinds, groupes[1], groupes[0]))
    else:
        raise ValueError('Unrecognized contrast, only [1.0,1.0] or [-1.0,-1.0] are accepted. '
                         'You enter {}'.format(contrast))

    if 'tangent' in kinds:
        warnings.warn('I think using a two sample t-test in this fashion on tangent '
                      'space should be interpreted carefully !')

    # Fisher transform coefficients
    for kind in kinds:
        if preprocessing_method == 'fisher':
            if kind != 'tangent':
                X = pre_preprocessing.fisher_transform(symmetric_array=stacked_matrices[groupes[0]][kind])
                Y = pre_preprocessing.fisher_transform(symmetric_array=stacked_matrices[groupes[1]][kind])
            elif kind == 'tangent':
                X = stacked_matrices[groupes[0]][kind]
                Y = stacked_matrices[groupes[1]][kind]
        elif preprocessing_method == 'already_preprocessed':
            X = stacked_matrices[groupes[0]][kind]
            Y = stacked_matrices[groupes[1]][kind]
        else:
            raise ValueError('Unrecognized pre-processing method')

        # Vectorize the connectivity matrices, discarding the diagonal
        X_vectorize = sym_matrix_to_vec(symmetric=X)
        Y_vectorize = sym_matrix_to_vec(symmetric=Y)
        # Vectorize the corresponding boolean array
        vec_X_mask = array_operation.vectorize_boolean_mask(
            symmetric_boolean_mask=stacked_matrices[groupes[0]]['masked_array'])
        vec_Y_mask = array_operation.vectorize_boolean_mask(
            symmetric_boolean_mask=stacked_matrices[groupes[1]]['masked_array'])

        # Create a numpy masked array structure for the two sample
        X_ = np.ma.array(data=X_vectorize, mask=vec_X_mask)
        Y_ = np.ma.array(data=Y_vectorize, mask=vec_Y_mask)

        # Finally perform a two sample t-test for masked array along the first
        # dimension according contrast, accounting for discarded rois
        if contrast == [1.0, -1.0]:
            t_stats_vec, pvalues_vec = mstats.ttest_ind(a=X_, b=Y_, axis=0)
        elif contrast == [-1.0, 1.0]:
            t_stats_vec, pvalues_vec = mstats.ttest_ind(a=Y_, b=X_, axis=0)

        # Replace nan values in pvalues_vec by 1 to avoid failure in correction method
        pvalues_vec[np.isnan(pvalues_vec)] = 1.

        # Correction of pvalues, reject of null hypotheses below alpha level.
        reject_, pvalues_vec_corrected, _, _ = multipletests(pvals=pvalues_vec.data,
                                                             alpha=alpha,
                                                             method=multicomp_method)

        # Computing the effect: difference between mean according to the contrast vector:
        groupes_mean_matrices = ccm.group_mean_connectivity(
            subjects_connectivity_matrices=subjects_connectivity_matrices_dictionnary,
            kinds=kinds)

        mean_X = groupes_mean_matrices[groupes[0]][kind]
        mean_Y = groupes_mean_matrices[groupes[1]][kind]

        if contrast == [1.0, -1.0]:
            mean_effect = mean_X - mean_Y
        elif contrast == [-1.0, 1.0]:

            mean_effect = mean_Y - mean_X

        # Reconstruction of statistic for each kind
        t_stats_matrix = vec_to_sym_matrix(vec=t_stats_vec)
        pvalues_uncorrected_matrix = vec_to_sym_matrix(vec=pvalues_vec)
        pvalues_corrected_matrix = vec_to_sym_matrix(vec=pvalues_vec_corrected)

        # Fill diagonal with p = 1 for corrected pvalues
        np.fill_diagonal(pvalues_corrected_matrix, 1)

        # Reconstruction of significant brain connection and significant pvalues
        reject_boolean_matrix = vec_to_sym_matrix(vec=reject_)
        # Fill diagonal with 0 in the reconstructed numerical rejection array
        np.fill_diagonal(reject_boolean_matrix, 0)

        significant_edges = np.multiply(t_stats_matrix, reject_boolean_matrix)
        significant_pvalues = np.multiply(pvalues_corrected_matrix, reject_boolean_matrix)
        significant_mean_effect = np.multiply(mean_effect, reject_boolean_matrix)
        uncorrected_mean_effect = np.multiply(mean_effect, pvalues_uncorrected_matrix < alpha)

        # Save the t_stats_matrix, the uncorrected p values matrix, the corrected pvalues matrix in the dictionary.
        t_test_dictionnary[kind] = {'tstatistic': t_stats_matrix,
                                    'uncorrected pvalues': pvalues_uncorrected_matrix,
                                    'corrected pvalues': pvalues_corrected_matrix,
                                    'significant edges': significant_edges,
                                    'significant pvalues': significant_pvalues,
                                    'significant mean effect': significant_mean_effect,
                                    'total mean effect': mean_effect,
                                    'uncorrected mean effect': uncorrected_mean_effect,
                                    'tested_contrast': contrast}

    return t_test_dictionnary


def linear_regression(connectivity_data, data, formula, NA_action,
                      kind, subjects_to_drop=None, sheetname=0, save_regression_directory=None,
                      contrasts='Id', compute_pvalues=True, pvalues_tail='two_tailed',
                      alpha=0.05, pvals_correction_method=['fdr_bh'], nperms_maxT=10000,
                      vectorize=False, discard_diagonal=False):
    # TODO: add a way to select column containing the subjects ID, and
    # TODO: shift it to be the index of the DataFrame
    """Fit a linear model on connectivity coefficients across subjects.

    Parameters
    ----------
    connectivity_data: dict
        The connectivity matrix organised in a dictionary. The subject identifier
        as keys, and metric as values, i.e the matrix which contain the connectivity coefficient.
        As metric are symmetric, matrix as to be vectorize in a 1D array of shape n_features.
    formula: string
        The model, in a R fashion.
    data: string
        The full path to the xlsx data file, containing all the dependant variables
        in the model you want to estimate.
    NA_action: string
        Directive for handling missing data in the xlsx file. Choices are :
        'drop': subject will discarded in the analysis, 'raise': raise an
        error if missing data is present.
    sheetname: int
        The position in your excel file of the sheet of interest.
    subjects_to_drop: list, optional
        List of subjects you want to discard in the analysis. If None, all the
        row in the dataframe are kept. Default is None.
    kind: string
        The metric, present in the provided connectivity data you want to perform analysis.
    save_regression_directory: string
        The full path to a directory for saving the regression results
    contrasts: string or numpy.array of shape (n_features,n_features) optional
        The contrast vector for infering the regression coefficients.
        Default is 'Id', all regressors are tested.
    compute_pvalues: bool, optional
        If True pvalues are computed. Default is True
    pvalues_tail: string, optional
        If 'two_tailed', a two-sided t-test is computed. If 'one_tailed' a one
        tailed t-test is computed.
    alpha: float, optional
        The Type error rate. Corrected p-values above alpha will be discarded.
        Default is 0.05.
    pvals_correction_method: string, optional
        The method for accounting for the multiple comparison problems.
        Choices are among the statsmodels library : {'bonferroni', 'sidak', 'holm-sidak', 'holm',
        'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'}, and the
        'maxT' method in the mulm library. Default is 'fdr_bh'.
    nperms_maxT: int, optional
        If maximum statistic correction is chosen, the number of permutations. Default is 10000.

    Returns
    -------
    output 1: dict
        The regression results, with the regressors variable as keys, and
        corrected and corrected p-values matrix, significant t-values matrix,
        t-values matrix under the alpha uncorrected threshold.
    output 2: numpy.array of shape (n_samples, n_features)
        The design matrix of the analysis.
    output 3: numpy.array of shape (n_samples, q), q: number of model to fit.
        The independent variable matrix with all the multiple outcome to fit.

    """
    # Reading the data
    df = pd.read_excel(data, sheet_name=sheetname)
    # Set the index to the subjects identifier column
    df = df.set_index(['subjects'])
    
    # Drop the subjects we want to discard :
    df_c = df.copy()
    if subjects_to_drop is not None:
        df = df_c.drop(labels=subjects_to_drop)
    else:
        df = df_c
    
    # Build the design matrix according the dataframe and regression model.
    X_df = dmatrix(formula_like=formula, data=df, return_type='dataframe',
                   NA_action=NA_action)

    # Stacked vectorized connectivity matrices in the same order of subjects
    # index list of the DESIGN MATRIX, because of missing data, not all subjects
    # will be in the analysis.
    # All the subjects present in the excel file
    general_regression_subjects_list = X_df.index
    # Intersection of subjects to perform regression and the general list because
    # of possible dropped NA values.
    regression_subjects_list = \
        list(set(connectivity_data.keys()).intersection(general_regression_subjects_list))
    y = np.array([connectivity_data[subject][kind] for subject in regression_subjects_list])
    
    if vectorize:
        y = sym_matrix_to_vec(y, discard_diagonal=discard_diagonal)
    else:
        y = y
    
    # Conversion of X_df into a classic numpy array
    X_df = X_df.loc[regression_subjects_list]
    X = np.array(X_df.loc[regression_subjects_list])
    
    # Setting the contrast vector
    if contrasts == 'Id':
        contrasts = np.identity(X.shape[1])
    else:
        contrasts = contrasts
    
    # Mass univariate testing using MUOLS library
    mod = mulm.MUOLS(Y=y, X=X).fit()
    raw_tvals, raw_pvals, dfree = mod.t_test(contrasts=contrasts,
                                             pval=compute_pvalues,
                                             two_tailed=pvalues_tail)
    
    # Compute prediction of the models
    y_prediction = mod.predict(X=X)
    
    # Replace nan values in pvalues_vec by 1 to avoid failure in correction method
    raw_pvals[np.isnan(raw_pvals)] = 1.
    
    # Initialize boolean mask for rejected H0 hypothesis, i.e for corrected pvalues < alpha.
    reject_ = np.zeros(raw_pvals.shape, dtype='bool')
    # Initialize array to save corrected pvalues
    pvalues_vec_corrected = np.zeros(raw_pvals.shape)
    
    # Save the results dictionary per correction method
    correction_method_regression_results = dict.fromkeys(pvals_correction_method)
    
    # Saving the results of the regression
    # Creation of the directory containing the regression results
    if save_regression_directory is not None:
        try:
            os.makedirs(save_regression_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
    # Choose multiple comparison correction method among the statsmodels library
    for corr_method in pvals_correction_method:
        
        # Listing the dependent variable, in the order of the output of
        # the regression analysis
        covariable_name = X_df.columns
        # Save a dictionnary with the principal
        regression_results = dict.fromkeys(covariable_name)
        
        if corr_method in ['fdr_bh', 'bonferroni']:
            
            # Flatten the array of raw p-values
            raw_p_shape = raw_pvals.shape
            # Call fdr correction, and reshape the results to (n_variable, n_features)
            pvalues_vec_corrected = multipletests(pvals=raw_pvals.flatten(),
                                                   method=corr_method, alpha=alpha)[1].reshape(raw_p_shape)
            
            reject_ = multipletests(pvals=raw_pvals.flatten(),
                                                   method=corr_method, alpha=alpha)[0].reshape(raw_p_shape)
        
        elif corr_method == 'maxT':
            # Inference with the maximum statistic method :
            _, pvalues_vec_corrected, _, null_distribution = \
            mod.t_test_maxT(contrasts=contrasts,
                            nperms=nperms_maxT)
            # Manual construction for the reject_ mask:
            reject_ = pvalues_vec_corrected < alpha
        
        else:
            raise ValueError('Unrecognized correction method ! \n'
                             'Please refer to the docstring function.')
        
        # Reconstruction in a matrix structure of the different output
        
        # Reconstruction of boolean reject_ mask : it's a binary mask
        reject_m_ = vec_to_sym_matrix(reject_)
        # Reconstruction of raw t values
        raw_tvals_m = vec_to_sym_matrix(raw_tvals)
        # Reconstruction of raw p values
        raw_pvals_m = vec_to_sym_matrix(raw_pvals)
        # Reconstruction of corrected p values
        corrected_pvals_m = vec_to_sym_matrix(pvalues_vec_corrected)
        # Construction of masked raw t values according to corrected p values
        # under alpha threshold
        significant_tvals_m = np.multiply(raw_tvals_m, reject_m_)

        for i in range(len(X_df.columns)):
            regression_results[covariable_name[i]] = \
                {'raw pvalues': raw_pvals_m[i, :, :],
                 'raw tvalues': raw_tvals_m[i, :, :],
                 'corrected pvalues': corrected_pvals_m[i, :, :],
                 'significant tvalues': significant_tvals_m[i, :, :]}

        correction_method_regression_results[corr_method] = {'results': regression_results}
    
        if corr_method == 'maxT':    
            correction_method_regression_results['maximum T null distribution'] = null_distribution

    # saving the dictionary
    if save_regression_directory is not None:
        folders_and_files_management.save_object(correction_method_regression_results,
                                                 save_regression_directory,
                                                 'regression_results.pkl')

    return correction_method_regression_results, X_df, y, y_prediction, regression_subjects_list


def functional_connectivity_distribution_estimation(functional_connectivity_estimate):
    # TODO : When functional_connectivity_estimate contain nan, the estimation fail,
    # TODO: I have to add a case for data contain non finite value
    """Estimates the mean and standard deviation of functional connectivity distribution assuming a Gaussian behavior.

    Parameters
    ----------
    functional_connectivity_estimate: numpy.array, shape(n_features, n_features)
    or shape 0.5*n_features*(n_features + 1)
        A functional connectivity matrices, if a 2D array is provided it will vectorized discarding the diagonal.

    Returns
    -------
    output 1: numpy.array shape 0.5*n_features*(n_features + 1)
        The vectorized functional connectivity array.
    output 2: float
        The estimated mean of the data.
    output 2: float
        The estimated standard deviation of the data.

    See Also
    --------
    scipy.stats.norm :
        This function from the scipy library is used here,
        to estimate the mean and the standard deviation of the data.
    pre_preprocessing.fisher_transform :
        To ensure a normal behavior of the connectivity coefficient,
        this function apply a classical Fisher transform to the data.

    Notes
    -----
    We assume here that the functional connectivity your're dealing with have a **Gaussian** behavior,
    and therefore can be describe properly with two parameters: the mean and the standard deviation.
    To ensure a Gaussian behavior transformation of the connectivity coefficient should be used like
    Fisher transform for correlation or partial correlation for example.

    """

    # Check the dimension of the passed functional matrices
    functional_connectivity_shape = functional_connectivity_estimate.shape
    # If it is a 2D matrix we vectorize it, discarding the diagonal to have a 1D distribution array
    if len(functional_connectivity_shape) == 2:
        vectorized_functional_connectivity_estimate = sym_matrix_to_vec(functional_connectivity_estimate,
                                                                        discard_diagonal=True)
    else:
        vectorized_functional_connectivity_estimate = functional_connectivity_estimate

    # We estimate the parameters of the distribution assuming it is Gaussian
    mean_data, std_data = norm.fit(vectorized_functional_connectivity_estimate)

    return vectorized_functional_connectivity_estimate, mean_data, std_data


def mean_functional_connectivity_distribution_estimation(mean_groups_connectivity_matrices):
    """Estimates for the mean connectivity matrices for each group, the mean and standard deviation
    assuming gaussian distribution

    Parameters
    ----------
    A multi-levels dictionnary organised as follow :
        - The first keys levels is the different groups in the study.
        - The second keys levels is the mean connectivity matrices for
        the different kinds. They are array of shape (number of regions , number of regions).

    Returns
    -------
    output : dict
        A dictionnary organised as follow:
            - The first keys levels is the different groups in the study.
            - The second keys levels is the kinds present in the provided dictionnary.
            - The third levels keys contain the estimated mean, the estimated standard deviation,
            and the vectorized array of connectivity coefficients.

    Notes
    -----
    Apply a Z-fisher transformation to the input matrices can be useful to improve and ensure a normal
    behavior of the data.

    """

    parameters_functional_connectivity_estimation = dict.fromkeys(mean_groups_connectivity_matrices)

    for groupe in mean_groups_connectivity_matrices.keys():
        parameters_functional_connectivity_estimation[groupe] = dict.fromkeys(mean_groups_connectivity_matrices[groupe])
        for kind in mean_groups_connectivity_matrices[groupe].keys():
            mean_kind_matrices = mean_groups_connectivity_matrices[groupe][kind]
            # Estimating the mean and standard deviation for the distribution
            vectorized_kind_matrices, kind_mean_data, kind_std_data = functional_connectivity_distribution_estimation(mean_kind_matrices)
            # Fill the dictionnary saving the mean and standard deviation for each kind, and the vectorized connectivity matrices
            parameters_functional_connectivity_estimation[groupe][kind] = {'mean': kind_mean_data, 'std': kind_std_data,
                                                                           'vectorized connectivity': vectorized_kind_matrices}

    return parameters_functional_connectivity_estimation


def two_sample_t_test_on_mean_connectivity(mean_matrix_for_each_subjects, kinds, groupes, contrast):
    """Performs a two sample t-test between the two groupes in the study on the mean matrix of each subjects.

    Parameters
    ----------
    mean_matrix_for_each_subjects: dict
        A dictionnary containing, at the first level, the groups as keys, and the kinds as values.
        Inside each each kind, a dictionnary containing the subjects identifier as values, and
        the mean functional connectivity of that subject as values.
    kinds: list
        The list of kinds to test.
    groupes: list
        The list of the two groupes in the study.
    contrast: list of int.
        A list of int, to precise the contrast between the two group. If contrast
        is set to the vector [1.0, -1.0], the computed t values is base on the contrast
        groupes[0] - groupes[1], and if contrast is [-1.0, 1.0], it's the contrary.

    Returns
    -------
    output: dict
        A dictionnary containing each kinds as keys, and the t-statistic and p-values as
        values for each kinds.

    Notes
    -----
    The p-values is based on a two-sided statistic.


    """

    t_test_dictionnary_functional_score = dict.fromkeys(kinds)
    for kind in kinds:

        # Stacked the functional score of the first group, the first in the list of groupes
        x = np.array([mean_matrix_for_each_subjects[groupes[0]][kind][s] for s in mean_matrix_for_each_subjects[groupes[0]][kind].keys()])
        # Stack the functional score of the second group, the second in the list of groupes
        y = np.array([mean_matrix_for_each_subjects[groupes[1]][kind][s] for s in mean_matrix_for_each_subjects[groupes[1]][kind].keys()])

        # Depending on the contrast
        if contrast == [1.0, -1.0]:
            t_stat_, uncorrected_pvalues = ttest_ind(x, y)
        elif contrast == [-1.0, 1.0]:
            t_stat_, uncorrected_pvalues = ttest_ind(y, x)
        else:
            raise ValueError('Unrecognized contrast !')

        # Fill a dictionnary with the t-statistic and the p-values
        t_test_dictionnary_functional_score[kind] = {'tstatistic': t_stat_, 'uncorrected pvalues': uncorrected_pvalues}

    return t_test_dictionnary_functional_score


def distribution_estimation_mean_subjects_connectivity(mean_matrix_for_each_subjects, groupes, kinds):
    """Provides an estimation of mean and standard deviation of the distribution for the two groupes
    under study, of the mean matrix of each subjects.

    Parameters
    ----------
    mean_matrix_for_each_subjects: dict
        A dictionnary containing, at the first level, the groups as keys, and the kinds as values.
        Inside each each kind, a dictionnary containing the subjects identifier as values, and
        the mean functional connectivity of that subject as values.
    kinds: list
        The list of kinds to test.
    groupes: list
        The list of the two groupes in the study.

    Returns
    -------
    output: dict
        A dictionnary with the groupes as keys, and for each group the estimated standard deviation,
        the estimated mean of the distribution, and the array of shape (n_subject, )
        of the mean matrix of each subjects.


    """
    # Gaussian estimation of the distribution of mean connectivity for each subjects
    estimation_dict = dict.fromkeys(groupes)
    for groupe in groupes:
        estimation_dict[groupe] = dict.fromkeys(kinds)
        for kind in kinds:
            # Fetch the mean connectivity for each subject in one numerical array
            mean_connectivity_vec = np.array(
                [mean_matrix_for_each_subjects[groupe][kind][s]
                 for s in mean_matrix_for_each_subjects[groupe][kind].keys()])
            # Estimate the mean and standard deviation of the data with a Gaussian assumption
            # Fill the dictionnary with the parameters estimate for the current group and kind
            estimation_dict[groupe][kind] = {
                'std': functional_connectivity_distribution_estimation(mean_connectivity_vec)[2],
                'mean': functional_connectivity_distribution_estimation(mean_connectivity_vec)[1],
                'mean connectivity': mean_connectivity_vec}

    return estimation_dict


def intra_network_two_samples_t_test(intra_network_connectivity_dictionary, groupes, kinds,
                                     contrast, network_labels_list, alpha=.05,
                                     p_value_correction_method='fdr_bh', assume_equal_var=True,
                                     nan_policy='omit', paired=False):
    """Test the difference of intra network connectivity between the groups under the study.

    Parameters
    ----------
    intra_network_connectivity_dictionary: dict
        A subjects connectivity dictionnary, with groupes in the study as the first levels of keys,
        the kinds in the study as the second levels of keys, and the intra network connectivity for each
        network in the study.
    groupes: list
        The list of the groups under the study.
    kinds: list
        The list of kinds in the study.
    contrast: list
        The contrast vector use for the t-test. Choices are [1.0, -1.0], or [-1.0, 1.0].
    network_labels_list: list
        The list of the name of the different network.
    alpha: float, optional
        The type I error rate threshold. For p-value under alpha, the null hypothesis can be rejected.
        Default os 0.05.
    p_value_correction_method: string, optional
        The correction method accounting for the multiple comparison problem.
        Default is 'fdr_bh', the traditional False Discovery Rate correction from Benjamini & Hochberg.
    assume_equal_var: bool, optional
        If False, the Welch t-test is perform accounting for different variances between the tested sample.
    nan_policy: string, optional
        Behavior regarding possible missing data (nan values). Default is 'omit'.

    Returns
    -------
    output: dict
        A dictionnary structure containing the t-test result for each network : the t statistic value, the corrected and
        uncorrected p values, the intra network connectivity array of shape (number of subject, ) for
        each group, and the contrast vector.
        .
    # TODO: add an argument to precise the field name in the dictionnary containing the network strength.
    """
    intra_network_strength_t_test = dict.fromkeys(kinds)
    # For some network, like Auditory network in subject 40 in patients, the strength is a masked value which will we
    # convert to nan. So we force to omit nan value in the t-test function call.
    for kind in kinds:

        intra_network_strength_t_test[kind] = dict.fromkeys(network_labels_list)
        # List to stack the uncorrected p value for each network
        all_network_uncorrected_p_values = []
        for network in network_labels_list:
            # Fetch the intra-network strength for each subject in the first group
            x = np.array([intra_network_connectivity_dictionary[groupes[0]][subject][kind][network]['network connectivity strength']
                          for subject in intra_network_connectivity_dictionary[groupes[0]].keys()])
            # Fetch the intra-network strength for each subject in the second group
            y = np.array([intra_network_connectivity_dictionary[groupes[1]][subject][kind][network]['network connectivity strength']
                          for subject in intra_network_connectivity_dictionary[groupes[1]].keys()])
            # Perform two sample t-test according to contrast vector
            if contrast == [1.0, -1.0]:
                if paired is False:
                    network_t_statistic, network_p_values_uncorrected = ttest_ind(x, y,
                                                                                  nan_policy=nan_policy,
                                                                                  equal_var=assume_equal_var)
                else:
                    network_t_statistic, network_p_values_uncorrected = ttest_rel(x, y,
                                                                                  nan_policy=nan_policy,
                                                                                  )
            elif contrast == [-1.0, 1.0]:
                if paired is False:
                    network_t_statistic, network_p_values_uncorrected = ttest_ind(y, x,
                                                                                  nan_policy=nan_policy,
                                                                                  equal_var=assume_equal_var)
                else:
                    network_t_statistic, network_p_values_uncorrected = ttest_rel(y, x,
                                                                                  nan_policy=nan_policy,
                                                                                  )
            else:
                raise ValueError('Unrecognized contrast !')

            # Fill the dictionary to save the result
            intra_network_strength_t_test[kind][network] = {'t statistic': network_t_statistic,
                                                            'uncorrected p values': network_p_values_uncorrected,
                                                            groupes[0]: x,
                                                            groupes[1]: y,
                                                            'contrast': contrast,
                                                            }
            # Stack the uncorrected network p value in the order of network_labels_list
            all_network_uncorrected_p_values.append(network_p_values_uncorrected)

        # Correction of p-values, for each kinds correct for the number of test i.e the number of network
        all_network_p_values = np.array(all_network_uncorrected_p_values)
        # Correct the p value with the chosen method
        reject_boolean_mask, corrected_pvalues, _, _ = \
            multipletests(pvals=all_network_p_values,
                          method=p_value_correction_method,
                          alpha=alpha)

        # Fill the dictionnary, appending a new key containing the corrected p values
        for network in network_labels_list:
            intra_network_strength_t_test[kind][network]['corrected pvalues'] = \
                corrected_pvalues[network_labels_list.index(network)]

    return intra_network_strength_t_test


def inter_network_two_sample_t_test(subjects_inter_network_connectivity_matrices, groupes,
                                    kinds, contrast,
                                    network_label_list, alpha=.05,
                                    p_value_correction_method='fdr_bh',
                                    assuming_equal_var=True,
                                    nan_policy='omit'):
    """Test the difference of connectivity between network performing a two sample t test.

    Parameters
    ----------
    subjects_inter_network_connectivity_matrices: dict
        A subjects connectivity dictionnary, with groupes in the study as the first levels of keys,
        the kinds in the study as the second levels of keys, and the inter network connectivity
        matrices as values, of shape (number of networks, number of networks)
    groupes: list
        The list of the groups under the study.
    kinds: list
        The list of kinds in the study.
    contrast: list
        The contrast vector use for the t-test. Choices are [1.0, -1.0], or [-1.0, 1.0].
    network_label_list: list
        The list of the name of the different network.
    alpha: float, optional
        The type I error rate threshold. For p-value under alpha, the null hypothesis can be rejected.
        Default is 0.05.
    p_value_correction_method: string, optional
        The correction method accounting for the multiple comparison problem.
        Default is 'fdr_bh', the traditional False Discovery Rate correction from
        Benjamini & Hochberg.
    assuming_equal_var: bool, optional
        If False, the Welch t-test is perform accounting for different variances between
        the tested sample.
    nan_policy: string, optional
        Behavior regarding possible missing data (nan values). Default is 'omit'.

    Returns
    -------
    output: dict
        A dictionnary structure containing the t-test result : the raw t statistic array, the corrected and
        uncorrected p values array, the masked t statistic array for corrected p values under
        the alpha threshold.

    """
    inter_network_t_test_result = dict.fromkeys(kinds)

    n_network = len(network_label_list)

    # Fetch the group connectivity, and compute a t-test for each kind.
    for kind in kinds:
        # Fetch and vectorize the subjects connectivity matrices
        x = np.array([vectorizer(numpy_array=subjects_inter_network_connectivity_matrices[groupes[0]][s][kind],
                                 discard_diagonal=True)[1]
                      for s in subjects_inter_network_connectivity_matrices[groupes[0]].keys()])

        y = np.array([vectorizer(numpy_array=subjects_inter_network_connectivity_matrices[groupes[1]][s][kind],
                                 discard_diagonal=True)[1]
                      for s in subjects_inter_network_connectivity_matrices[groupes[1]].keys()])
        if contrast == [1.0, -1.0]:
            t_statistic, p_values_uncorrected = ttest_ind(x, y, axis=0, equal_var=assuming_equal_var,
                                                          nan_policy=nan_policy)
        elif contrast == [-1.0, 1.0]:
            t_statistic, p_values_uncorrected = ttest_ind(y, x, axis=0, equal_var=assuming_equal_var,
                                                          nan_policy=nan_policy)
        else:
            raise ValueError('Unrecognized contrast !')

        # Correction of p values
        vec_reject_mask, vec_corrected_p_values, _, _, = multipletests(pvals=p_values_uncorrected,
                                                                       alpha=alpha,
                                                                       method=p_value_correction_method)

        # Reconstruction of boolean reject_ mask : it's a binary mask
        reject_m_ = vec_to_sym_matrix(vec_reject_mask, np.bool_(np.ones(n_network)))
        # Reconstruction of raw t values
        raw_tvals_m = vec_to_sym_matrix(t_statistic, np.zeros(n_network))
        # Reconstruction of raw p values
        raw_pvals_m = vec_to_sym_matrix(p_values_uncorrected, np.ones(n_network))
        # Reconstruction of corrected p values
        corrected_pvals_m = vec_to_sym_matrix(vec_corrected_p_values, np.ones(n_network))
        # Construction of masked raw t values according to corrected p values under alpha threshold
        significant_tvals_m = np.multiply(raw_tvals_m, reject_m_)

        inter_network_t_test_result[kind] = {'raw t statistic': raw_tvals_m,
                                             'uncorrected p values': raw_pvals_m,
                                             'corrected p values': corrected_pvals_m,
                                             'significant t values': significant_tvals_m,
                                             'contrast': contrast}

    return inter_network_t_test_result


def two_sample_t_test_(connectivity_dictionnary_, groupes, kinds, field, contrast,
                       assume_equal_var=True,
                       nan_policy='omit',
                       paired=False):
    """Perform a simple two sample t test between two sets
    of connectivity matrices.

    Parameters
    ---------
    connectivity_dictionnary_: dict
        A dictionnary which contain some connectivity of interest, and associated measure,
        in a ConPagnon subjects connectivity matrices like.
    groupes: list
        The list of groupes under in the study.
    kinds: list
        The list of kinds in the study
    field: string
        The field name in the dictionary containing the connectivity coefficient array.
        It should be a 1D vector of shape (number of subject, ).
    contrast: list
        The contrast vector. Choices are [1.0, -1.0], or [-1.0, 1.0].
    assume_equal_var: bool, optional
        If False, a Welch t-test assuming different variances between the two group.
    nan_policy: string, optional
        Behavior regarding the missing value in the tested data. Default is 'omit'

    Returns
    -------
    output: dict
        A dictionnary with the raw t statistic, the used contrast vector,
        and the **uncorrected** p values.

    """

    t_test_result = dict.fromkeys(kinds)
    # Fetch the group connectivity, and compute a t-test for each kind.
    for kind in kinds:
        x = connectivity_dictionnary_[groupes[0]][kind][field]
        y = connectivity_dictionnary_[groupes[1]][kind][field]
        if contrast == [1.0, -1.0]:
            if paired is False:

                t_statistic, p_values_uncorrected = ttest_ind(x, y, equal_var=assume_equal_var,
                                                              nan_policy=nan_policy)
            else:
                t_statistic, p_values_uncorrected = ttest_rel(x, y, nan_policy=nan_policy)
        elif contrast == [-1.0, 1.0]:
            if paired is False:

                t_statistic, p_values_uncorrected = ttest_ind(y, x, equal_var=assume_equal_var,
                                                              nan_policy=nan_policy)
            else:
                t_statistic, p_values_uncorrected = ttest_rel(y, x, nan_policy=nan_policy)
        else:
            raise ValueError('Unrecognized contrast !')
        t_test_result[kind] = {'t_statistic': t_statistic,
                               'uncorrected p value': p_values_uncorrected,
                               'contrast': contrast}

    return t_test_result


def regress_confounds(vectorize_subjects_connectivity, confound_dictionary, groupes, kinds,
                      data,
                      sheetname,
                      NA_action='drop'):
    # TODO: add a way to select column containing the subjects ID, and
    # TODO: shift it to be the index of the DataFrame
    """Regress confound on connectivity matrices

    Parameters
    ----------
    vectorize_subjects_connectivity: dict
        The subject connectivity matrices dictionary, with vectorized matrices,
        WITHOUT the diagonal.
    confound_dictionary: dict
        The nested dictionary containing for each group and kind: a field
        named 'confounds' containing a list of confounds. A second field named
        'subjects to drop' containing a list of subjects identifier to drop, None
        if you want to pick all of the subjects.
    groupes: list
        The list of group on which you want to regress confounds.
    kinds: list
        The list of kind
    data: str
        The full path, including extension to the excel file containing
        the confound for each subjects. This will be read by pandas.
        Note that the index of the resulting dataframe must be the
        subjects identifiers.
    sheetname: str
        The sheet name if the excel file containing the confound for each
        subjects of each groups.
    NA_action: str, optional
        Behavior regarding the missing values. If 'drop', the entire
        row is deleted from the design matrix.
    """
    # Reading the data excel file
    df = pd.read_excel(data, sheet_name=sheetname)

    # Copy of the vectorized connectivity matrices dictionary
    # to override the matrices
    regressed_subjects_connectivity_dictionary = copy.deepcopy(vectorize_subjects_connectivity)
    regression_by_kind = dict.fromkeys(groupes, dict.fromkeys(kinds))
    # Save the design matrix, and the stack of matrices before and after
    # regressing
    regression_data_dictionary = dict.fromkeys(groupes)

    for group in groupes:
        # Read the confound and subject to drop for
        # each group
        # List of confounds
        group_confounds = confound_dictionary[group]['confounds']
        # List of subject to drop
        group_subjects_to_drop = confound_dictionary[group]['subjects to drop']
        # List of subjects present in the group dictionary
        group_subjects_list = list(vectorize_subjects_connectivity[group].keys())

        # Copy of original dataframe to avoid
        # side effect :
        df_c = df.copy()
        # Fetch the sub-dataframe containing the subjects
        # in the dictionary
        df_group = df_c.loc[group_subjects_list]
        if group_subjects_to_drop is not None:
            df_group = df_group.drop(labels=group_subjects_to_drop)
        else:
            df_group = df_group

        # Build the formula: regress the confound without
        # Intercept term
        group_formula = '+'.join(group_confounds)

        # Build the design matrix according the dataframe and regression model.
        X_df = dmatrix(formula_like=group_formula, data=df_group,
                       return_type='dataframe',
                       NA_action=NA_action)
        # Drop intercept
        X_df = X_df.drop('Intercept', axis=1)

        # Conversion of design dataframe into numpy array
        # for QR decomposition
        X = np.array(X_df)

        regression_data_dictionary[group] = {'design matrix': X_df}

        for kind in kinds:
            # stack the vectorized array into a numpy array of shape (n_subjects, n_features)
            y = np.array([vectorize_subjects_connectivity[group][subject][kind]
                          for subject in group_subjects_list])

            # check first dimension of X and y, it should be the same
            if y.shape[0] != X.shape[0]:
                raise ValueError('Matrices are not aligned ! First '
                                 'dimension of y is {} and first dimension of X is {}'.format(y.shape[0],
                                                                                              X.shape[0]))

            # QR decomposition of X matrix
            Q, R, _ = linalg.qr(X, mode='economic', pivoting=True)
            # Improve numerical stability
            Q = Q[:, np.abs(np.diag(R)) > np.finfo(np.float).eps * 100.]

            # Regress the confounds in X on y
            y_regressed = y - Q.dot(Q.T).dot(y)

            # Override the connectivity matrices for the current group and kind
            for subject in group_subjects_list:
                regressed_subjects_connectivity_dictionary[group][subject][kind] = \
                    y_regressed[group_subjects_list.index(subject), ...]

            regression_by_kind[group][kind] = {'original matrices': y, 'after regression': y_regressed}

        regression_data_dictionary[group].update(regression_by_kind)

    return regressed_subjects_connectivity_dictionary, regression_data_dictionary


def design_matrix_builder(dataframe, formula, return_type='dataframe'):
    """Build a design matrix based on a dataframe

    Parameters
    ----------
    dataframe: pandas.DataFrame
        A pandas dataframe containing the data.
    formula: str
        The formula written in R style, with the variable
        to explain in the left side of the ~, and the explanatory
        variables in the right side.
    return_type: str, optional
        Return type of the response variable, and design matrix.
        Default is dataframe. Other choices are: matrix.

    Returns
    -------
    output1: pandas.DataFrame
        A dataframe, of shape (n_observations, ). This is
        the variable to explain
    output2: pandas.DataFrame
        The design matrix, of shape (n_observation, n_explanatory_variables +1)
    """
    # Build the design matrix and return the response variable
    # and design matrix
    y, X = dmatrices(formula, data=dataframe, return_type=return_type)

    return y, X


def ols_regression(y, X):
    """Fit a linear model with ordinary least square regression
    from statmodels library

    Parameters
    ----------
    y: array-like
        The variable to explain of shape (n_observations, )
    X: array-like
        The design matrix, of shape (n_obervations, n_regressors) or
        (n_obervations + 1, n_regressors) if intercept is added in the model.

    Returns
    -------
    output: statsmodels.regression.linear_model.RegressionResultsWrapper
        A statsmodels regression object containing the fit of the model.
    """

    # Model initialization with OLS method
    ols_model = sm.OLS(endog=y, exog=X)
    # Fit of the model
    ols_model_fit = ols_model.fit()

    return ols_model_fit


def ols_regression_formula(formula, data):
    """Fit a linear model with a formula API in R style.

    :param formula:
    :param data:
    :return:
    """
    ols_model_fit = smf.ols(formula=formula, data=data).fit()

    return ols_model_fit


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """

    C = np.asarray(C)
    # Add a columns of one to have the same behavior of matlab partialcorr function
    C = np.c_[C, np.ones(C.shape[0])]
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr, _ = pearsonr(res_i, res_j)
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr
