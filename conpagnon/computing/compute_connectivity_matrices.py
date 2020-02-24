#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:02:19 2017

ComPagnon 2.0

@author: Dhaif BEKHA

"""
from conpagnon.data_handling.data_architecture import read_text_data_file, create_group_dictionnary
from conpagnon.data_handling import atlas
import numpy as np
from nilearn import input_data
from nilearn.connectome import ConnectivityMeasure, vec_to_sym_matrix
from conpagnon.utils.array_operation import masked_arrays_mean, check_2d, vectorizer
from copy import deepcopy
from conpagnon.utils import array_operation
import webcolors
import itertools
from math import sqrt
from conpagnon.data_handling.data_management import remove_duplicate
from scipy.stats import zmap, ttest_1samp
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from conpagnon.utils.folders_and_files_management import save_object
import os
from statsmodels.stats.multitest import multipletests


def create_connectivity_mask(time_series_dictionary, groupes):
    """Create boolean mask for each subjects accounting
    for discarded roi if they exist.

    Parameters
    ----------
    time_series_dictionary: dict
        The time series dictionary structured as follow:
            - The first keys levels is the different groupes.
            - The second keys levels is the subjects IDs
            - The third levels is two keys : 'time_series' containing the
            subject time series in an array of shape (number of regions, number of time points)
            A key 'discarded_rois' containing an array of the index of ROIs
            where the corresponding labels is 'void'. If no void labels is detected,
            then the array is empty.
    groupes: list
        The list of groups in the study.

    Returns
    -------
    output: dict
        The same time series dictionary with a field called 'masked_array' for
        each subjects. A masked array is a boolean mask accounting for discarded
        rois, True for a discarded rois, and False elsewhere.

    Notes
    -----
    If 'discarded_rois' is empty the mask is False at every position
    in the array.
    Almost all the operation in ConPagnon accounts for discarded rois with
    masked array via the dedicated module in NumPy.

    """
    for group in groupes:
        subject_in_group = list(time_series_dictionary[group].keys())
        for subject in subject_in_group:
            # If a field discarded_rois exist:
            number_of_regions = time_series_dictionary[group][subject]['time_series'].shape[1]
            # mask computation
            subject_mask = np.zeros((number_of_regions, number_of_regions), dtype='bool')
            if 'discarded_rois' in time_series_dictionary[group][subject].keys():
                # We compute a mask according the position of discarded roi
                # number of regions
                empty_roi_position = time_series_dictionary[group][subject]['discarded_rois']
                # Replace by True the boolean value position corresponding to discarded rois
                # If empty_roi_position is not empty
                if empty_roi_position.size > 0:
                    for discarded_roi in empty_roi_position:
                        # True for the entire rows corresponding to the current empty roi
                        subject_mask[:, discarded_roi] = True
                        # True for the entire columns corresponding to the current empty roi
                        subject_mask[discarded_roi, :] = True
            else:
                # We compute a mask, with False everywhere, because all the test
                # is performed with masked array in general.
                subject_mask = subject_mask

            # Create a new field
            time_series_dictionary[group][subject]['masked_array'] = subject_mask

    return time_series_dictionary


def time_series_extraction_with_individual_atlases(root_fmri_data_directory, groupes, subjects_id_data_path, group_data,
                                                   repetition_time, low_pass_filtering=None, high_pass_filtering=None,
                                                   detrend_signal=True, standardize_signal=True, smooth_signal=None,
                                                   resampling_target='data', memory_level=1,
                                                   nilearn_cache_directory=None,
                                                   ):
    """Times series extractions for each subjects with an individual atlas.
    
    This function extract time series for each subjects according predefined regions in individual atlases.
    
    Parameters
    ----------
    
    root_fmri_data_directory : str
        The full path of a root directory containing one or numerous
        sub-directories where functional images are.
        
    groupes : list
        The list of groups of interest, i.e the name of the
        sub-directories containing the functional images you want 
        to study.
        
    subjects_id_data_path : str
        The full path to the data file containing the subjects IDs.
        
    group_data : dict
        A multi-levels dictionary structured as follow :
            - The first keys level is the different groups
            to study.
            - The second level keys is the subjects IDs for
            all subjects in each groupes.
            - The third level keys contain multiple field : 
            'functional_file' contain the full path to the subject
            fmri file, 'atlas_file' contain the full path to the 
            subject atlas, 'label_file' the full path to the subject
            atlas label, 'confounds_file' the full path to the subject
            confounds file if they exist, a empty list if not.
    
    repetition_time : float
        The repetition time in second, i.e the time between two volumes
        in a fmri image.
        
    low_pass_filtering : float or None, optional
        The low pass frequency in Hz cut-off for filtering the times series.
        Default is None.
    
    high_pass_filtering : float or None, optional
        The high pass frequency in Hz cut-off for filtering the time series.
        Default is None
        
    detrend_signal : bool, optional
        Detrend the time series removing the first order moment to the
        time series, i.e removing the mean signals to each time series.
        Default if True.
        
    standardize_signal : bool, optional
        Set the times series to unit variance
        Default is True.
        
    smooth_signal : float or None, optional
        The full-width half maximum in millimeters of a Gaussian
        spatial smoothing to apply to the time series.
        Default is None.
        
    resampling_target : str
        Gives the reference image which the source image image
        will be resample. 
        Choices are  : {“mask”, “maps”, “data”, None}. 
        Default is 'data'.
        
    memory_level : int, optional
        Caching parameters of functions. Default is 1.
        
    nilearn_cache_directory : str or None
        The full path which will contain the folder used to cache
        the regions extractions. If None, no cache is performing.
        Default is None.
    
    
    Returns
    -------

    output : dict
        A dictionnary structured as follow :
            - The first keys levels is the different groupes. 
            - The second keys levels is the subjects IDs
            - The third levels is two keys : 'time_series' containing the
            subject time series in an array of shape (number of regions, number of time points) 
            A key 'discarded_rois' containing an array of the index of ROIs 
            where the corresponding labels is 'void'. If no void labels is detected,
            then the array is empty.
            
    See Also
    --------
    
    data_architecture.fetch_data_with_individual_atlases : This function
    returned the organised dictionnary with all the information needed for the
    time series extraction with individual atlases. This is 
    simply the argument `group_data`.
    
    Notes
    -----
    
    The times series extraction is based on functions contain is the Nilearn
    packages. I encourage the users to consult the docstring of the following
    function for the detailed mechanism of signal extraction : nilearn.signal.clean,
    nilearn.input_data.NiftiMapsMasker.
    
    The subjects IDs file, whatever the format, should not contain any header.
    It's should have a row column of ID for each subjects.
    
    Remembers that the discarded rois are defined according to their labels
    which must be declared as 'void' in the subject atlas labels files. 
    
    References
    ----------
    
    The Nilearn official documentation on Github :
    [1] http://nilearn.github.io/index.html
    
    
    
    """
    
    # Dictionnary initialisation for each group, and subject as keys.
    times_series_dictionnary = create_group_dictionnary(subjects_id_data_path=subjects_id_data_path, groupes=groupes,
                                                        root_fmri_data_directory=root_fmri_data_directory)

    for groupe in group_data.keys():
        # For each each subject found in the group_data dictionnary, which fetch all the path for important file.
        for subject in group_data[groupe].keys():
            # We fetch the individual atlas of the subject
            subject_atlas = group_data[groupe][subject]['atlas_file'][0]
            # Fetch of discarded rois index, reading the subject atlas labels file, and looking for a 'void' label.
            subject_atlas_labels = read_text_data_file(file_path=group_data[groupe][subject]['label_file'][0])
            subject_void_rois = np.where(subject_atlas_labels[0] == 'void')[0]
            # Fetch of the subject functional file.
            subject_fmri = group_data[groupe][subject]['functional_file'][0]
            # Fetch the file containing the signal regressors, if it exist
            if group_data[groupe][subject]['counfounds_file']:
                subject_confounds = group_data[groupe][subject]['counfounds_file'][0]
            else:
                subject_confounds = None

            # Set the computation parameters for time series, via the NifTiMapsMasker class
            subject_masker = input_data.NiftiMapsMasker(
                maps_img=subject_atlas, resampling_target=resampling_target,
                smoothing_fwhm=smooth_signal, high_pass=high_pass_filtering,
                low_pass=low_pass_filtering, detrend=detrend_signal, standardize=standardize_signal,
                t_r=repetition_time, memory=nilearn_cache_directory, memory_level=memory_level)
            
            # We extract the subject time series, for each regions defined in the atlas, if confounds was found,
            # we regress them
            subject_time_series = subject_masker.fit_transform(imgs=subject_fmri, confounds=subject_confounds)
            
            # We fill the dictionnary for each and subjects, with the
            # time series array of shape (time point, number of regions),
            # and the list of index of discarded rois.
            times_series_dictionnary[groupe][subject] = {'time_series': subject_time_series,
                                                         'discarded_rois': subject_void_rois}

    times_series_dictionnary = create_connectivity_mask(time_series_dictionary=times_series_dictionnary,
                                                        groupes=groupes)

    return times_series_dictionnary


def time_series_extraction(root_fmri_data_directory, groupes, subjects_id_data_path,
                           reference_atlas, group_data, repetition_time,
                           low_pass_filtering=None, high_pass_filtering=None,
                           detrend_signal=True, standardize_signal=True,
                           smooth_signal=None, resampling_target='data', memory_level=1,
                           nilearn_cache_directory=None):
    
    """Times series extractions for each subjects on a common atlas.
    
    This function extract time series for each subjects according predefined regions on one common atlas.
    
    Parameters
    ----------
    root_fmri_data_directory : str
        The full path of a root directory containing one or numerous
        sub-directories where functional images are.
    groupes : list
        The list of groups of interest, i.e the name of the
        sub-directories containing the functional images you want 
        to study.
    subjects_id_data_path : str
        The full path to the data file containing the subjects IDs.
    group_data : dict
        A multi-levels dictionnary structured as follow :
            - The first keys level is the different groups
            to study.
            - The second level keys is the subjects IDs for
            all subjects in each groupes.
            - The third level keys contain multiple field : 
            'functional_file' contain the full path to the subject
            fmri file, 'atlas_file' contain the full path to the 
            subject atlas, 'label_file' the full path to the subject
            atlas label, 'confounds_file' the full path to the subject
            confounds file if they exist, a empty list if not.
    reference_atlas : str
        The full path to the reference atlas which will be used
        to extract signals from regions.
    repetition_time : float
        The repetition time in second, i.e the time between two volumes
        in a fmri image.
    low_pass_filtering : float or None, optional
        The low pass frequency in Hz cut-off for filtering the times series.
        Default is None.
    high_pass_filtering : float or None, optional
        The high pass frequency in Hz cut-off for filtering the time series.
        Default is None
    detrend_signal : bool, optional
        Detrend the time series removing the first order moment to the
        time series, i.e removing the mean signals to each time series.
        Default if True.
    standardize_signal : bool, optional
        Set the times series to unit variance
        Default is True.
    smooth_signal : float or None, optional
        The full-width half maximum in millimeters of a Gaussian
        spatial smoothing to apply to the time series.
        Default is None.
    resampling_target : str
        Gives the reference image which the source image image
        will be resample. 
        Choices are  : {“mask”, “maps”, “data”, None}. 
        Default is 'data'.
    memory_level : int, optional
        Caching parameters of functions. Default is 1.
    nilearn_cache_directory : str or None
        The full path which will contain the folder used to cache
        the regions extractions. If None, no cache is performing.
        Default is None.

    Returns
    -------
    output : dict
        A dictionnary structured as follow :
            - The first keys levels is the different groupes. 
            - The second keys levels is the subjects IDs
            - The third levels is two keys : 'time_series' containing the
            subject time series in an array of shape (number of regions, number of time points) 

    See Also
    --------
    data_architecture.fetch_data : This function
    returned the organised dictionnary with all the information needed for the
    time series extraction on a common atlas. This is 
    simply the argument `group_data`.
    
    Notes
    -----
    The times series extraction is based on functions contain is the Nilearn
    packages. I encourage the users to consult the docstring of the following
    function for the detailed mechanism of signal extraction : nilearn.signal.clean,
    nilearn.input_data.NiftiMapsMasker.
    
    The subjects IDs file, whatever the format, should not contain any header.
    It's should have a row column of ID for each subjects.
        
    References
    ----------
    The Nilearn official documentation on Github :
    [1] http://nilearn.github.io/index.html
    """

    # Dictionnary initialisation for time series, for each group, and for each subjects.
    times_series_dictionnary = create_group_dictionnary(subjects_id_data_path=subjects_id_data_path,
                                                        groupes=groupes,
                                                        root_fmri_data_directory=root_fmri_data_directory)

    for groupe in group_data.keys():
        # For each subject in the group_data dictionnary
        for subject in group_data[groupe].keys():
            # Fetch the subject functional file path
            subject_fmri = group_data[groupe][subject]['functional_file'][0]
            # Fetch the subject regressors file path, if it exist
            if group_data[groupe][subject]['counfounds_file']:
                subject_confounds = group_data[groupe][subject]['counfounds_file'][0]
            else:
                subject_confounds = None
            
            # Set the computation parameters for time series, via the NifTiMapsMasker class
            subject_masker = input_data.NiftiMapsMasker(maps_img=reference_atlas, resampling_target=resampling_target,
                                                        smoothing_fwhm=smooth_signal, high_pass=high_pass_filtering,
                                                        low_pass=low_pass_filtering, detrend=detrend_signal,
                                                        standardize=standardize_signal, t_r=repetition_time,
                                                        memory=nilearn_cache_directory, memory_level=memory_level)
            
            # We extract the subject time series, for each regions defined in the atlas, if confounds was found,
            # we regress them
            subject_time_series = subject_masker.fit_transform(imgs=subject_fmri, confounds=subject_confounds)
            
            # We fill the time series array of shape (times points, number of regions) for each subject in each group in
            # time series dictionnary
            times_series_dictionnary[groupe][subject] = {'time_series': subject_time_series}

    times_series_dictionnary = create_connectivity_mask(time_series_dictionary=times_series_dictionnary,
                                                        groupes=groupes)

    return times_series_dictionnary


def individual_connectivity_matrices(time_series_dictionary, kinds, covariance_estimator, vectorize=False,
                                     discarding_diagonal=False, z_fisher_transform=False):
    """Compute the connectivity matrices for groups of subjects
    
    This function computes connectivity matrices for different metrics.

    Parameters
    ----------
    time_series_dictionary : dict
        A multi-levels dictionnary organised as follow :
            - The first keys levels is the different groupes in the study. 
            - The second keys levels is the subjects IDs
            - The third levels is two keys : 'time_series' containing the
            subject time series in an array of shape (number of regions, number of time points) 
            A key 'discarded_rois' containing an array of the index of ROIs 
            where the corresponding labels is 'void'. If no void labels is detected,
            then the array should be empty.
    kinds : list
        List of the different metrics you want to compute the 
        connectivity matrices. Choices are 'tangent', 'correlation',
        'partial correlation', 'covariance', 'precision'.
    covariance_estimator : estimator object
        All the kinds  are based on derivation of covariances matrices. You need
        to precise the estimator, see Notes.
    vectorize : bool, optional
        If True, the connectivity matrices are reshape into 1D arrays of 
        the vectorized lower part of the matrices. Useful for classification,
        regression...
        Default is False.
    z_fisher_transform: bool, optional
        If True, the z fisher transform is apply to all
        the connectivity matrices. Default is False
    discarding_diagonal: bool, optional
        If True, the diagonal is discarded in
        the vectorization process. Default is False.

    Returns
    -------
    output : dict
        A multi-levels dictionnary organised as follow :
            - The first keys levels is the different groupes in the study. 
            - The second keys levels is the subjects IDs
            - The third levels contain multiple keys : multiple
            kind keys containing the corresponding kind matrix.
            a 'discarded_rois' key containing the index of discarded
            rois.
            Finally you should find a 'masked_array' key : This key 
            contains a array of Boolean, shape (numbers of regions, numbers of regions)
            where the value are True for the index in 'discarded_rois' array, and False
            elsewhere. See Note for further details. The diagonal of the
            matrix are saved for each kind too in a dedicated field.
            
    See Also
    --------
    time_series_extraction, time_series_extraction_with_individual_atlases :
        These are the functions which extract the time series according atlas
        regions and returned a structured dictionnary. 
        This is simply the argument `time_series_dictionary`.
        
    Notes 
    -----
    Covariances estimator are estimators compute in the scikit-learn library.
    Multiple estimator can be found in the module sklearn.covariance, popular
    choices are the Ledoit-Wolf estimator, or the OAS estimator.
    In the output dictionnary, each subjects have a masked array of boolean. The
    masked will be useful when computing the mean connectivity matrices, we will
    account the discarded rois in the derivation. For the statistical test you might
    want perform, it will be useful too, to discarded those rois. A True value is
    a masked roi, False elsewhere.
    For the tangent kind, the derivation of individual matrices need to be
    made on the POOLED GROUPS which is performed here.
    The derivation of connectivity matrices are based on Nilearn functions. I
    encourage the user to read the following docstring of important functions :
    nilearn.connectome.ConnectivityMeasure
    
    References
    ----------
    For the use of tangent :
        .. [1] G. Varoquaux et al. “Detection of brain functional-connectivity
        difference in post-stroke patients using group-level covariance modeling", MICCAI 2010
    """

    saving_subjects_connectivity_matrices_dictionary = dict.fromkeys(list(time_series_dictionary.keys()))

    for groupe in time_series_dictionary.keys():
        subjects_list = time_series_dictionary[groupe].keys()
        saving_subjects_connectivity_matrices_dictionary[groupe] = dict.fromkeys(subjects_list)
        for subject in subjects_list:
            subject_time_series = [time_series_dictionary[groupe][subject]['time_series']]
            saving_subjects_connectivity_matrices_dictionary[groupe][subject] = dict.fromkeys(kinds)

            # operation on the boolean mask for each subject
            subject_mask = time_series_dictionary[groupe][subject]['masked_array']
            if vectorize:
                if discarding_diagonal:
                    subject_mask_diagonal_, subject_mask_ = \
                        vectorizer(numpy_array=time_series_dictionary[groupe][subject]['masked_array'],
                                   discard_diagonal=discarding_diagonal,
                                   array_type='boolean')
                    saving_subjects_connectivity_matrices_dictionary[groupe][subject]['masked_array'] = \
                        subject_mask_
                    saving_subjects_connectivity_matrices_dictionary[groupe][subject][
                        'diagonal_mask'] = subject_mask_diagonal_
                else:
                    # discarding_diagonal is False
                    subject_mask_diagonal_, subject_mask_ = \
                        vectorizer(numpy_array=subject_mask,
                                   discard_diagonal=discarding_diagonal,
                                   array_type='boolean')
                    saving_subjects_connectivity_matrices_dictionary[groupe][subject]['masked_array'] = \
                        subject_mask_
                    saving_subjects_connectivity_matrices_dictionary[groupe][subject][
                        'diagonal_mask'] = subject_mask_diagonal_
            else:
                saving_subjects_connectivity_matrices_dictionary[groupe][subject]['masked_array'] = subject_mask
                saving_subjects_connectivity_matrices_dictionary[groupe][subject]['diagonal_mask'] = \
                    np.diag(subject_mask)

            # Fetch the index of discarded rois
            if 'discarded_rois' in time_series_dictionary[groupe][subject].keys():
                saving_subjects_connectivity_matrices_dictionary[groupe][subject]['discarded_rois'] = \
                    time_series_dictionary[groupe][subject]['discarded_rois']

            for kind in kinds:
                # We compute connectivity matrices except for the tangent kind, which need the pooled groups
                if kind != 'tangent':
                    kind_measure = ConnectivityMeasure(kind=kind, cov_estimator=covariance_estimator)
                    # the current subject connectivity matrices
                    subject_kind_matrice = kind_measure.fit_transform(subject_time_series)
                    # We fill the subject dictionnary, accounting for vectorization option
                    if vectorize:
                        # if discard_diagonal is True
                        if discarding_diagonal:
                            subject_kind_matrice_diagonal_, subject_kind_matrice_ = \
                                vectorizer(numpy_array=subject_kind_matrice[0, ...],
                                           discard_diagonal=discarding_diagonal,
                                           array_type='numeric')
                        else:
                            subject_kind_matrice_diagonal_, subject_kind_matrice_ = \
                                vectorizer(numpy_array=subject_kind_matrice[0, ...], discard_diagonal=False,
                                           array_type='numeric')
                        if z_fisher_transform is True:
                            # Apply a fisher transform, i.e arctanh to each connectivity coefficient
                            undefined_value = np.invert(subject_kind_matrice_ == 1)
                            subject_kind_matrice_ = np.arctanh(subject_kind_matrice_, where=undefined_value)
                            # Fill with nan, the undefined value after transformation
                            subject_kind_matrice_[np.invert(undefined_value)] = np.nan
                        else:
                            subject_kind_matrice_ = subject_kind_matrice_

                        saving_subjects_connectivity_matrices_dictionary[groupe][subject][kind] = \
                            subject_kind_matrice_[:]
                        # Save the diagonal in this case:
                        saving_subjects_connectivity_matrices_dictionary[groupe][subject][kind + '_diagonal'] = \
                            subject_kind_matrice_diagonal_
                    else:
                        # If we do not vectorize the connectivity matrix
                        if z_fisher_transform is True:
                            # Apply a fisher transform, i.e arctanh to each connectivity coefficient
                            # For r = 1, in the diagonal, arctanh is not defined.
                            # We compute a mask where False value are ignored
                            undefined_value = np.invert(subject_kind_matrice[0, ...] == 1)
                            subject_kind_matrice = np.arctanh(subject_kind_matrice, where=undefined_value)
                            subject_kind_matrice[0, np.invert(undefined_value)] = np.nan
                        else:
                            subject_kind_matrice = subject_kind_matrice

                        saving_subjects_connectivity_matrices_dictionary[groupe][subject][kind] = \
                            subject_kind_matrice[0, ...]

    if 'tangent' in kinds:
        # We call pooled_groups_connectivity on the tangent space, returning along the tangent matrices
        # the labels subjects in the same order of tangent connectivity matrices:
        tangent_matrix, subjects_order = pooled_groups_connectivity(
            time_series_dictionary=time_series_dictionary,
            kinds=['tangent'],
            covariance_estimator=covariance_estimator,
            vectorize=False)

        for groupe in time_series_dictionary.keys():
            group_subject_list = time_series_dictionary[groupe].keys()
            for subject in group_subject_list:
                subjects_order_list = list(subjects_order)
                sub_index = subjects_order_list.index(subject)
                if vectorize:
                    if discarding_diagonal:
                        diagonal_matrice_tangent, corresponding_tangent_matrice = \
                            vectorizer(numpy_array=tangent_matrix['tangent'][sub_index, ...],
                                       discard_diagonal=discarding_diagonal,
                                       array_type='numeric')
                    else:
                        diagonal_matrice_tangent, corresponding_tangent_matrice = \
                            vectorizer(numpy_array=tangent_matrix['tangent'][sub_index, ...],
                                       discard_diagonal=False,
                                       array_type='numeric')
                    saving_subjects_connectivity_matrices_dictionary[groupe][subject]['tangent'] = \
                        corresponding_tangent_matrice
                    saving_subjects_connectivity_matrices_dictionary[groupe][subject]['tangent' + '_diagonal'] = \
                        diagonal_matrice_tangent
                else:
                    corresponding_tangent_matrice = tangent_matrix['tangent'][sub_index, ...]
                    saving_subjects_connectivity_matrices_dictionary[groupe][subject]['tangent'] = \
                        corresponding_tangent_matrice

    return saving_subjects_connectivity_matrices_dictionary


def pooled_groups_tangent_mean(time_series_dictionary, covariance_estimator):
    """Compute the geometric mean of covariances connectivity matrices for the tangent kind.
    
    The geometric mean is the point in the symmetric manifold matrices spaces 
    where the tangent space is defined. Therefore it's make sense for the pooled
    groups only.
    
    Parameters
    ----------
    time_series_dictionary : dict
        A multi-levels dictionnary organised as follow :
            - The first keys levels is the different groupes in the study. 
            - The second keys levels is the subjects IDs
            - The third levels is two keys : 'time_series' containing the
            subject time series in an array of shape (number of regions, number of time points) 
            A key 'discarded_rois' containing an array of the index of ROIs 
            where the corresponding labels is 'void'. If no void labels is detected,
            then the array should be empty.
    covariance_estimator : estimator object
    All the kinds  are based on derivation of covariances matrices. You need
    to precise the estimator, see Notes.
    
    Returns 
    -------
    output : numpy.array of shape (n_features, n_features)
        The geometric mean a the pooled group.
        
    See Also
    --------
    time_series_extraction, time_series_extraction_with_individual_atlases :
    These are the functions which extract the time series according atlas
    regions and returned a structured dictionnary. This is simply the argument
    `time_series_dictionary`.
    
    Notes
    -----
    Covariances estimator are estimators compute in the scikit-learn library.
    Multiple estimator can be found in the module sklearn.covariance, popular
    choices are the Ledoit-Wolf estimator, or the OAS estimator.
    For now, we doesnt account for discarded rois for the derivation of the
    geometric mean.
    """

    tangent_measure = ConnectivityMeasure(kind='tangent', cov_estimator=covariance_estimator)
    group_stacked_time_series = []

    for groupe in time_series_dictionary.keys():
        for subject in time_series_dictionary[groupe].keys():
            # appending subject time series
            group_stacked_time_series.append(time_series_dictionary[groupe][subject]['time_series'])

    # Projecting in tangent space the whole cohort of subjects at the geometric mean
    _ = tangent_measure.fit_transform(group_stacked_time_series)
    # Retrieve the geometric mean
    tangent_mean_ = tangent_measure.mean_

    return tangent_mean_


def group_mean_connectivity(subjects_connectivity_matrices, kinds, axis=0):
    """Compute the mean connectivity matrices for each kind accounting for masked rois.
        Read the notes for the tangent space !
    
    Parameters
    ----------
    subjects_connectivity_matrices : dict
         A multi-levels dictionnary organised as follow :
             - The first keys levels is the different groupes in the study. 
            - The second keys levels is the subjects IDs
            - The third levels is the different kind matrices
            for each subjects, a 'discarded_rois' key for the
            discarded rois array index, a 'masked_array' key containing
            the array of Boolean of True for the discarded_rois index, and False
            elsewhere.
    kinds : list
        List of kinds you want the mean connectivity. Choices
        are 'correlation', 'tangent', 'covariances', 'precision', 'partial 
        correlation'. Off course, the kind should be in the subjects_connectivity_matrices
        dictionnary.
    axis : int, optional
        The axis you want to compute the mean, the subjects axis. Default is 0.
    
    Returns 
    -------
    output : dict
        A multi-levels dictionnary organised as follow :
            - The first keys levels is the different groups in the study.
            - The second keys levels is the mean connectivity matrices for
            the different kinds. They are array of shape (number of regions , number of regions)
            if vectorize is False, and shape (n_columns * (n_columns + 1) /2) else.
            
    See Also
    --------
    individual_connectivity_matrices : These function returned a organised
    dictionnary containing the connectivity matrices for different kinds. This is
    simply the argument `subjects_connectivity_matrices`
    
    Notes
    -----
    When computing the mean, we account for the 'discarded_rois' entries. That mean 
    when the value is True in the masked_array, we discard the rois for the corresponding
    subject in the derivation of the mean.
    When I compute the mean in the tangent space, it's a arithmetic mean. This
    mean matrix is in the tangent space, that is NOT in the same space as correlation or partial correlation matrix.
    Be careful with the interpretation !!
    That said, the tangent space is defined at ONE point in the manifold of symmetric matrices.
    This point is the geometric mean for the POOLED groups if multiple group are studied !
    """

    group_mean_connectivity_matrices = dict.fromkeys(list(subjects_connectivity_matrices.keys()))
    for groupe in subjects_connectivity_matrices.keys():
        # Dictionnary initialisation : for each group, we generate the list of kinds as keys
        group_mean_connectivity_matrices[groupe] = dict.fromkeys(kinds)
        # Subject list of the current group
        subjects_list = subjects_connectivity_matrices[groupe].keys()
        for kind in kinds:
            # We stack the subjects connectivity matrices for the group and current kind
            kind_stack_matrices = np.array([subjects_connectivity_matrices[groupe][subject][kind]
                                            for subject in subjects_list])
            # We also stack the subjects boolean array
            mask_array_stack = np.array([subjects_connectivity_matrices[groupe][subject]['masked_array']
                                         for subject in subjects_list])
            # We compute the mean for each kind, accounting for the discarded rois,
            # via the function call masked_array_mean
            group_mean_connectivity_matrices[groupe][kind] = masked_arrays_mean(arrays=kind_stack_matrices,
                                                                                masked_array=mask_array_stack,
                                                                                axis=axis)

    return group_mean_connectivity_matrices


def pooled_groups_connectivity(time_series_dictionary, kinds, covariance_estimator, vectorize):
    """Compute connectivity matrices of pooled groups.
    
    This function simply stack the all the times series of each group in one pooled groups and compute 
    the connectivity matrices  on the whole group. When computing the tangent kind this function is
    called.
    
    Parameters 
    ----------
    time_series_dictionary : dict
    A multi-levels dictionnary organised as follow :
        - The first keys levels is the different groupes in the study. 
        - The second keys levels is the subjects IDs
        - The third levels is two keys : 'time_series' containing the
        subject time series in an array of shape (number of regions, number of time points) 
        A key 'discarded_rois' containing an array of the index of ROIs 
        where the corresponding labels is 'void'. If no void labels is detected,
        then the array should be empty.
    kinds : list
        List of the different metrics you want to compute the 
        connectivity matrices. Choices are 'tangent', 'correlation',
        'partial correlation', 'covariance', 'precision'.
    covariance_estimator : estimator object
        All the kinds  are based on derivation of covariances matrices. You need
        to precise the estimator, see Notes.
    vectorize : bool
        If True, the connectivity matrices are reshape into 1D arrays of 
        the vectorized lower part of the matrices. Useful for classification,
        regression...
        Diagonal are discarded.
    
    Returns
    -------
    output 1 : dict
        A multi-levels dictionnary organised as follow :
            - The first level keys is the different kinds
            - The second levels is simply a ndimensional array of connectivity
            matrices of shape (number of subjects, number of regions, number of regions)
            if vectorize is False, and shape (n_columns * (n_columns + 1) /2) else
    output 2 : list
        The list of subject IDs, in the order of time series computation.
    
    Notes
    -----
    Covariances estimator are estimators compute in the scikit-learn library.
    Multiple estimator can be found in the module sklearn.covariance, popular
    choices are the Ledoit-Wolf estimator, or the OAS estimator.
    """
    group_stacked_time_series = []
    for groupe in time_series_dictionary.keys():
        subject_list = time_series_dictionary[groupe].keys()
        for subject in subject_list:
            # Appending a tuple: time series and the corresponding labels of the subjects
            group_stacked_time_series.append((time_series_dictionary[groupe][subject]['time_series'], subject))
    
    # We compute the connectivity matrices for each subject in the pooled group
    pooled_connectivity_matrices_dictionary = dict.fromkeys(kinds)
    stacked_couple_subject_time_series = np.array(group_stacked_time_series)
    group_stacked_time_series = stacked_couple_subject_time_series[:, 0]
    subject_labels = stacked_couple_subject_time_series[:, 1]
    for kind in kinds:
        connectivity_measure = ConnectivityMeasure(kind=kind,
                                                   cov_estimator=covariance_estimator,
                                                   vectorize=vectorize)
        pooled_connectivity_matrices_dictionary[kind] = \
            connectivity_measure.fit_transform(group_stacked_time_series)
        
    return pooled_connectivity_matrices_dictionary, subject_labels


def extract_sub_connectivity_matrices(subjects_connectivity_matrices, kinds,
                                      regions_index,
                                      vectorize=False,
                                      discard_diagonal=False):
    """Extract sub matrices given region index.

    Parameters
    ----------
    subjects_connectivity_matrices: dict
        The dictionnary containing for each subjects, for each group and kind, the
        connectivity matrices, of shape (n_features, n_features), and also
        the boolean mask array indicating the discarded roi.
    kinds: list
        The list of kinds.
    regions_index: list, or 1D numpy.array of shape (number of indices, )
        The list of index of regions you want to extract the connectivity. The
        region index also correspond to the index of ROIs in the 4D reference
        atlas of the study.
    vectorize: bool, optional
        If True, the extracted sub-matrices are vectorized, keeping
        the diagonal of the matrix, and the corresponding boolean
        mask.
    discard_diagonal: bool, optional
        If True, the diagonal is discard when the extracted connectivity matrices are vectorized.

    Returns
    -------
    output: dict
        A dictionnary containing for each group and kinds, for each subject
        the extract sub-matrice, and the corresponding boolean mask.
    """

    # Copy of the subjects connectivity matrices dictionnary
    subjects_connectivity_matrices_copy = deepcopy(subjects_connectivity_matrices)

    for groupe in subjects_connectivity_matrices_copy.keys():
        for subject in subjects_connectivity_matrices_copy[groupe].keys():
            for kind in kinds:
                # Sub extraction of connectivity matrices according the provided region_index array
                row_sliced = subjects_connectivity_matrices_copy[groupe][subject][kind][regions_index, :]
                subjects_connectivity_matrices_copy[groupe][subject][kind] = row_sliced[:, regions_index]
                if vectorize:
                    _, subjects_connectivity_matrices_copy[groupe][subject][kind] = \
                        vectorizer(numpy_array=row_sliced[:, regions_index],
                                   discard_diagonal=discard_diagonal,
                                   array_type='numeric')

            # Sub extraction of corresponding masked array
            row_mask_sliced = subjects_connectivity_matrices_copy[groupe][subject]['masked_array'][regions_index, :]
            subjects_connectivity_matrices_copy[groupe][subject]['masked_array'] = row_mask_sliced[:, regions_index]
            if vectorize:
                _, subjects_connectivity_matrices_copy[groupe][subject]['masked_array'] = vectorizer(
                    numpy_array=row_mask_sliced[:, regions_index],
                    discard_diagonal=discard_diagonal,
                    array_type='boolean')

    # Return the subjects sub connectivity matrices
    subjects_sub_connectivity_matrices = subjects_connectivity_matrices_copy

    return subjects_sub_connectivity_matrices


def subjects_mean_connectivity_(subjects_individual_matrices_dictionnary,
                                connectivity_coefficient_position, kinds,
                                groupes):
    """Compute for each subjects, the mean connectivity for some connectivity coefficient in the general subjects
    connectivity matrices.

    Parameters
    ----------
    subjects_individual_matrices_dictionnary: dict
        The subjects connectivity dictionnary containing connectivity matrices
        and corresponding mask array for discarded rois, for each group.
    connectivity_coefficient_position: numpy.array of shape (number of rois, row_index, column_index)
        The array containing the position in the connectivity matrices of the rois you want
        to extract the connectivity coefficients.
    kinds: list
        The list of kinds.
    groupes: list
        The list of the two group in the study.

    Returns
    -------
    output: dict
        A dictionnary containing for each subject in group, a masked array structure containing
        the numerical array of the connectivity coefficient of interest, along with the boolean
        mask accounting for discarded rois. The second key, contain the mean of the extracted
        connectivity coefficient for each subject, also accounting for discarded rois it they
        exist.

    Notes
    -----
    The subjects connectivity matrices shouldn't be vectorized, the shape should
    be (n_features, n_features).
    """
    
    # Indices of the connectivity coefficient in terms of row and column index.
    row_index = connectivity_coefficient_position[:, 0]
    column_index = connectivity_coefficient_position[:, 1]

    # Extract from the subject connectivity dictionnary the connectivity coefficients

    # Dictionnary initialisation to store the regions connectivity coefficients, and the mask value for further
    # accounting for discarded rois
    connectivity_of_interest = dict.fromkeys(list(subjects_individual_matrices_dictionnary.keys()))
    for groupe in groupes:
        subjects_list = subjects_individual_matrices_dictionnary[groupe].keys()
        connectivity_of_interest[groupe] = dict.fromkeys(list(subjects_list))
        for subject in subjects_list:
            connectivity_of_interest[groupe][subject] = dict.fromkeys(kinds)
            for kind in kinds:
                # Check is subject matrice is two dimensional
                check_2d(numpy_array=subjects_individual_matrices_dictionnary[groupe][subject][kind])
                # Check is the corresponding mask is two dimensional
                check_2d(numpy_array=subjects_individual_matrices_dictionnary[groupe][subject]['masked_array'])

                # Extract connectivity coefficient based on the list of indices of row and column index
                subjects_connectivity_of_interest = \
                    subjects_individual_matrices_dictionnary[groupe][subject][kind][row_index,
                                                                                    column_index]
                # Extract corresponding mask values
                subjects_masked_array_connectivity_of_interest = \
                    subjects_individual_matrices_dictionnary[groupe][subject]['masked_array'][row_index,
                                                                                              column_index]
                # Build a masked array structure for further computation accounting for discarded roi
                subjects_connectivity_of_interest_ma = np.ma.array(data=subjects_connectivity_of_interest,
                                                                   mask=subjects_masked_array_connectivity_of_interest)
                # Compute the mean connectivity for the current subject, accounting for discarded rois
                mean_subject_connectivity_of_interest = subjects_connectivity_of_interest_ma.mean()
                # Fill the dictionnary, saving the subject masked array structure and
                # mean connectivity of extracted coefficient.
                connectivity_of_interest[groupe][subject][kind] = \
                    {'connectivity masked array': subjects_connectivity_of_interest_ma,
                     'mean connectivity': mean_subject_connectivity_of_interest}

    return connectivity_of_interest


def intra_network_functional_connectivity(subjects_individual_matrices_dictionnary,
                                          groupes, kinds, atlas_file,
                                          network_column_name,
                                          sheetname,
                                          roi_indices_column_name, color_of_network_column):
    """Compute for each subjects, the intra network connectivity for each network in the study.

    Parameters
    ----------
    subjects_individual_matrices_dictionnary: dict
        The subjects connectivity dictionnary containing connectivity matrices
        and corresponding mask array for discarded rois, for each group.
    groupes: list
        The list of groups in the study
    kinds: list
        The list of kinds in the study
    atlas_file: string
        The full path to an excel file containing information on the atlas
    network_column_name: string
        The name of the column in the excel file containing the network label for each roi
    roi_indices_column_name: string
        The name of the column in the excel file containing the index of each ROI in the 4D atlas file.
    sheetname: string
        The name of the active sheet in the excel file
    color_of_network_column: string
        The name of the columns containing for each roi, the corresponding network color.

    Returns
    -------
    output 1: dict
        A dictionnary structure containing for each subject and for each network: the network connectivity,
        the vectorized array of coefficients of the network without the diagonal, the corresponding vectorized
        mask array accounting for discarded rois, the diagonal of the mask array, the masked array structure
        of the network, and finally the number of rois in the network.
    output 2: dict
        A dictionnary network, containing information fetch from the atlas excel file of the atlas.
    output 3: list
        The network label list.
    output 4: array of shape (number of network, 3)
        The array containing the color in the normalized RGB space, in the order of the network label list.

    Notes
    -----
    The intra connectivity is simply defined as the mean, for each network,
    of the coefficient belonging to the network. Because connectivity metrics are symmetric,
    we only taking the vectorize part of the network connectivity matrices. We account for discarded roi,
    as we compute the mean on a numpy masked array structure, that is the vectorized array
    along with the vectorized boolean mask for the current network.

    References
    ----------
    This intra-network composite scores is used in the following references:

    .. [1] M. Brier, "Loss of Intranetwork and Internetwork Resting State Functional Connections with Alzheimer's
       Disease Progression" The Journal of Neuroscience, 2012.
    .. [2] P. Wang, "Aberrant intra- and inter-network connectivity architectures in Alzheimer's disease and mild
       cognitive impairment", Nature Publishing Group, 2015.

    """

    network_dict = atlas.fetch_atlas_functional_network(atlas_excel_file=atlas_file,
                                                        sheetname=sheetname,
                                                        network_column_name=network_column_name)
    network_labels_list = list(network_dict.keys())
    # The list of network are NOT in the same order of the atlas in the analysis, 
    # we have to fetch the color according to network labels list.
    
    network_label_color_name = [list(set(network_dict[n]['dataframe'][color_of_network_column]))[0]
                                for n in network_labels_list]
    network_label_colors = (1/255)*np.array([webcolors.name_to_rgb(network_label_color_name[i])
                                             for i in range(len(network_label_color_name))])
    
    # For each subject in each group : the network intra-connectivity
    # is defined as the mean of the connectivity coefficient inside
    # the network.

    # Intra network dictionnary initialisation
    intra_network_connectivity_dict = dict.fromkeys(groupes)
    for groupe in groupes:
        subjects_list = subjects_individual_matrices_dictionnary[groupe].keys()
        intra_network_connectivity_dict[groupe] = dict.fromkeys(subjects_list)
        for subject in subjects_list:
            intra_network_connectivity_dict[groupe][subject] = dict.fromkeys(kinds)
            for kind in kinds:
                intra_network_connectivity_dict[groupe][subject][kind] = dict.fromkeys(network_labels_list)

    # Initialisation of the list containing the label of network which have at least two regions
    valid_network_list = []
    # Initialisation of the list containing the color of the label network which have at least two regions
    valid_network_color_list = []
    for network in network_labels_list:

        # Fetch the indices of the network roi in the 4D atlas file, it will the same indices to slice in the subject
        # connectivity matrices
        network_roi_4d_indices = network_dict[network]['dataframe'][roi_indices_column_name]
        # Fetch the number of rois in the network
        n_roi_in_network = network_dict[network]['number of rois']
        # Computation make sense only they are at least two regions in the current network
        if n_roi_in_network >= 2:
            # It is a valid network, we append it's label and color for plotting purpose in the t-test
            valid_network_list.append(network)
            valid_network_color_list.append(network_label_colors[network_labels_list.index(network)])
            # Extract the sub matrix corresponding to the current network for all kinds
            network_connectivity_matrix = extract_sub_connectivity_matrices(
                subjects_connectivity_matrices=subjects_individual_matrices_dictionnary,
                kinds=kinds, regions_index=network_roi_4d_indices, vectorize=False)
            # For Each group, each kind and subject:
            # sum the lower diagonal of each network, accounting for the discarded ROIs,
            # if they are some in the network
            for groupe in groupes:
                subjects_list = network_connectivity_matrix[groupe].keys()
                for subject in subjects_list:
                    for kind in kinds:
                        # Vectorize the network connectivity matrix
                        diag_network, vectorize_network = array_operation.vectorizer(
                            numpy_array=network_connectivity_matrix[groupe][subject][kind],
                            discard_diagonal=True,
                            array_type='numeric')
                        # Vectorize the boolean mask
                        diag_network_mask, vectorize_network_mask = array_operation.vectorizer(
                            numpy_array=network_connectivity_matrix[groupe][subject]['masked_array'],
                            discard_diagonal=True,
                            array_type='boolean')
                        # Construct the masked array structure for the current network
                        network_masked_array = np.ma.array(data=vectorize_network, mask=vectorize_network_mask)
                        # Compute the mean for the network, accounting for discarded
                        # rois via the numpy masked array structure
                        network_mean = network_masked_array.mean()
                        # Fill the intra network dictionnary
                        intra_network_connectivity_dict[groupe][subject][kind][network] = {
                            'network connectivity strength': network_mean,
                            'network array': vectorize_network,
                            'network diagonal array': diag_network,
                            'network mask': vectorize_network_mask,
                            'network diagonal masked array': diag_network_mask,
                            'number of rois in network': n_roi_in_network}

    return intra_network_connectivity_dict, network_dict, valid_network_list,  valid_network_color_list


def inter_network_subjects_connectivity_matrices(subjects_individual_matrices_dictionnary, groupes,
                                                 kinds, atlas_file, sheetname,
                                                 network_column_name, roi_indices_column_name):
    """Compute for each subjects, the inter network connectivity matrices

    Parameters
    ----------
    subjects_individual_matrices_dictionnary: dict
        The subjects connectivity matrices dictionnary for each groupes, and kind in the study.
    groupes: list
        The list of groups in the study
    kinds: list
        The list of kinds in the study
    atlas_file: string
        The full path to an excel file containing information on the atlas
    network_column_name: string
        The name of the column in the excel file containing the network label for each roi
    roi_indices_column_name
        The name of the column in the excel file containing the index of each ROI in the 4D atlas file.
    sheetname: string
        The name of the active sheet in the excel file

    Returns
    -------
    output: dict
        The subjects inter network connectivity matrices dictionnary, for each group and kinds. Each matrices
        should have shape (number of network, number of network).

    Notes
    -----
    The inter-network connectivity is simply defined as the mean of all possible
    connection between a network pair. We also account for the possible discarded rois
    in one or both the network by computing the mean on a masked array structure, that is,
    the inter-network coefficient of interest, along with the corresponding value of the masked
    array.

    References
    ----------
    This inter-network composite score is used in the following references :

    .. [1] M. Brier, "Loss of Intranetwork and Internetwork Resting State Functional Connections with Alzheimer's
       Disease Progression" The Journal of Neuroscience, 2012.
    .. [2] P. Wang, "Aberrant intra- and inter-network connectivity architectures in Alzheimer's disease and mild
       cognitive impairment", Nature Publishing Group, 2015.

    """

    # Fetch all the network information from a excel file
    network_dict = atlas.fetch_atlas_functional_network(atlas_excel_file=atlas_file,
                                                        sheetname=sheetname,
                                                        network_column_name=network_column_name)
    # The list of network name
    network_labels_list = list(network_dict.keys())

    # Find all possible pair of network
    network_possible_pairs = list(itertools.combinations(network_labels_list, 2))
    subjects_inter_network_connectivity_matrices = dict.fromkeys(groupes)
    n_network = len(network_labels_list)

    # Build the network labels order from network possible
    # pair list for plotting purpose
    network_possible_pairs_array = np.array(network_possible_pairs)
    network_labels_order = remove_duplicate(seq=list(network_possible_pairs_array[:, 1]))
    network_labels_order.insert(0, network_possible_pairs[0][0])

    # Initialize a dictionnary for each network
    # Extract the inter network connectivity matrix
    for groupe in groupes:
        subjects_list = list(subjects_individual_matrices_dictionnary[groupe].keys())
        subjects_inter_network_connectivity_matrices[groupe] = dict.fromkeys(subjects_list)
        for subject in subjects_list:
            subjects_inter_network_connectivity_matrices[groupe][subject] = dict.fromkeys(kinds)
            for kind in kinds:
                subjects_inter_network_connectivity_matrices[groupe][subject][kind] = \
                    dict.fromkeys(network_possible_pairs)
                all_inter_network_strength = []
                for pair_of_network in network_possible_pairs:
                    inter_network_coefficient_to_sum = []
                    inter_network_coefficient_to_sum_mask = []
                    # Fetch the roi index in 4D atlas of the first network
                    # in the pair via the network dictionary, and number of rois
                    first_network_roi_index = network_dict[pair_of_network[0]]['dataframe'][roi_indices_column_name]
                    # Fetch the roi index in 4D atlas of the second network in the pair
                    second_network_roi_index = network_dict[pair_of_network[1]]['dataframe'][roi_indices_column_name]
                    # Loop over the pair of coefficient between the pair of network
                    for j in first_network_roi_index:
                        for i in second_network_roi_index:
                            # Append the coefficient to sum of all possible inter network coefficient pair
                            inter_network_coefficient_to_sum.append(
                                subjects_individual_matrices_dictionnary[groupe][subject][kind][i, j])
                            # Append the corresponding values in the masked array accounting for discarded rois.
                            inter_network_coefficient_to_sum_mask.append(
                                subjects_individual_matrices_dictionnary[groupe][subject]['masked_array'][i, j])
    
                    # Construct the masked array structure
                    # TODO: ABS or not Abs?
                    inter_network_pair_masked_array = np.ma.array(data=inter_network_coefficient_to_sum,
                                                                  mask=inter_network_coefficient_to_sum_mask)
                    # Average of the masked array structure, this is the inter network strength
                    inter_network_strength = inter_network_pair_masked_array.mean()
    
                    all_inter_network_strength.append(inter_network_strength)
    
                    subjects_inter_network_connectivity_matrices[groupe][subject][kind][pair_of_network] = {
                        'strength': inter_network_strength,
                        'connectivity coefficient': inter_network_pair_masked_array.data,
                        'masked array': inter_network_pair_masked_array.mask
                                                                                                            }
    
                subjects_inter_network_connectivity_matrices[groupe][subject][kind] = \
                    vec_to_sym_matrix(np.array(all_inter_network_strength), diagonal=np.ones(n_network)/sqrt(2))

    return subjects_inter_network_connectivity_matrices, network_labels_order


def mean_of_flatten_connectivity_matrices(subjects_individual_matrices_dictionary, groupes, kinds):
    """Return the flat mean connectivity for each subjects.

    Parameters
    ----------
    subjects_individual_matrices_dictionary: dict
        The subjects connectivity dictionnary containing connectivity matrices
        and corresponding mask array for discarded rois, for each group.
    groupes: list
        The list of groupes in the study
    kinds: list
        The list of kinds in the study

    Returns
    -------
    output: dict
        A dictionnary containing for each subject in group, a masked array structure containing
        the numerical array of the connectivity coefficient of interest, along with the boolean
        mask accounting for discarded rois. The second key, contain the mean of the extracted
        connectivity coefficient for each subject, also accounting for discarded rois it they
        exist.

    Notes
    -----
    The subjects connectivity matrices shouldn't be vectorized, the shape should
    be (n_features, n_features).
    """
    connectivity_of_interest = dict.fromkeys(list(subjects_individual_matrices_dictionary.keys()))
    for groupe in groupes:
        subjects_list = subjects_individual_matrices_dictionary[groupe].keys()
        connectivity_of_interest[groupe] = dict.fromkeys(list(subjects_list))
        for subject in subjects_list:
            connectivity_of_interest[groupe][subject] = dict.fromkeys(kinds)
            for kind in kinds:
                # Vectorize the array for the current kind
                diag_connectivity, vectorize_connectivity = array_operation.vectorizer(
                    numpy_array=subjects_individual_matrices_dictionary[groupe][subject][kind],
                    discard_diagonal=True,
                    array_type='numeric')
                # Vectorize the corresponding boolean mask
                diag_mask, vectorize_mask = array_operation.vectorizer(
                    numpy_array=subjects_individual_matrices_dictionary[groupe][subject]['masked_array'],
                    discard_diagonal=True,
                    array_type='boolean')
                # Compute the masked mean accounting for discarded ROIs
                vectorize_connectivity_masked = np.ma.masked_array(data=vectorize_connectivity,
                                                                   mask=vectorize_mask)
                flat_mean_subject_connectivity = vectorize_connectivity_masked.mean()
                # Fill the dictionnary with the mean connectivity of interest for the current subjects
                connectivity_of_interest[groupe][subject][kind] = {
                    'connectivity masked array': vectorize_connectivity_masked,
                    'mean connectivity': flat_mean_subject_connectivity}

    return connectivity_of_interest


def tangent_space_projection(reference_group, group_to_project, bootstrap_number,
                             bootstrap_size, output_directory, verif_null=True,
                             correction_method="bonferroni",
                             alpha=0.05, statistic="t"):
    """Project a group of time series to a space tangent to a reference connectivity matrix.
    The reference matrix is derived from a reference times series group. Then, each edges of
    the projected matrices is tested regarding the reference group through a one sample t-test
    or a simple z-score. The null distribution of edges is derived with bootstrap.
    Please, take a look to the reference paper for further a deeper understanding of the
    method.

    Parameters
    ----------
    reference_group: numpy.array
        The stack of the vectorized connectivity matrices of shape
        (n_subjects, n_features). This is the reference point where
        the tangent space will be computed. All tangent connectivity
        matrices will be derived regarding this reference set of matrices.
    group_to_project: numpy.array
        The stack of the vectorized connectivity matrices of shape
        (n_subjects, n_features). Each connectivity matrices will be
        projected in the tangent space, at the reference matrix of the
        reference group.
    bootstrap_number: int
        The null distribution of each edges is derived trough bootstrapping
        of the reference set of matrices. Usually around 500.
    bootstrap_size: int
        The size of the bootstrapped sample, i.e the number of
        subjects to pick from the reference set.
    output_directory: str
        The full path of the directory for saving the results
    verif_null: bool, optional
        If True, generate a report choosing a set of 20 random
        rois and plot the null distributions in those rois. True
        by defaults
    correction_method: str, optional
        The correction method used for to correct for
        multiple comparison. The default is the classical
        bonferroni method. You can choose among all the correction
        proposed by the statsmodels library.
    alpha: float, optional
        The type I error rate used for the multiple comparison
        correction. Set to .05 by defaults.
    statistic: str, optional
        The statistic used, a one sample t-test by default. Other choices
        are: z, for a simple z-score.

    Returns
    -------
    output: dict
        A dictionary with the following fields:
            - null_distribution: The array containing
            the estimated null distribution for each edges.
            - p_values_corrected: The p values, adjusted
            after the multiple comparison correction
            - reference_group_tangent_mean: The reference
            matrix, derived from the mean of the matrices
            from the reference group. This is the point
            where all matrices are projected.
            - reference_group_tangent_matrices: The connectivity
            matrices in the tangent space for the reference
            group.
            - group_to_project_tangent_matrices: The
            connectivity matrices in the tangent space
            for the projected group
            - group_to_project_stats: The chosen
            statistic for each subject, each edges
            in the projected set of subjects.

    References
    ----------

    .. [1] G. Varoquaux et al. “Detection of brain functional-connectivity
        difference in post-stroke patients using group-level covariance modeling", MICCAI 2010
    """
    # Initialize a list containing the null distribution
    null_distribution = []

    # Generate a bootstrap matrix of indices
    # before the loop.
    indices = np.arange(0, reference_group.shape[0], step=1)
    bootstrap_matrix = np.array([np.random.choice(
        indices,
        size=bootstrap_size,
        replace=False) for b in range(bootstrap_number)])

    # Loop over the bootstrap indices
    for b in range(bootstrap_number):
        print('Bootstrap # {}'.format(b))
        bootstrap_reference_group = bootstrap_matrix[b, :]
        # We chose one subject of the reference group
        # among the bootstrap sample
        left_out_subject = bootstrap_reference_group[0]
        subset_reference_group = bootstrap_reference_group[1:]
        # Compute mean matrices, and connectivity matrices
        # in the tangent on the subset of subject from the
        # reference group without the leftout subject.
        connectivity_measure = ConnectivityMeasure(kind='tangent',
                                                   vectorize=True,
                                                   discard_diagonal=True)
        reference_group_subset_matrices = \
            connectivity_measure.fit_transform(X=reference_group[subset_reference_group, ...])
        reference_group_subset_mean = connectivity_measure.mean_
        # Project at the previously computed mean
        # the leftout control subject
        leftout_subject = \
            connectivity_measure.transform(
                X=[reference_group[left_out_subject, ...]])[0]

        # Compute a z-score between the left out controls tangent connectivity
        # and the resampled controls group
        if statistic == "z":

            stat_null = zmap(scores=leftout_subject, compare=reference_group_subset_matrices)
        elif statistic == "t":

            # Another statistic: one sample t test, controls - the left out controls
            # as contrast
            stat_null = -1.0 * ttest_1samp(a=reference_group_subset_matrices, popmean=leftout_subject)[0]
        else:
            raise ValueError("Statistic type unknown. Type are z or t statistic, "
                             "you entered {}".format(statistic))

        # Store the results of the test, as null distribution
        null_distribution.append(stat_null)

    null_distribution_array = np.array(null_distribution)
    if verif_null:
        # plot of some null distribution randomly chosen
        n_rois = 20
        random_roi = np.random.choice(a=np.arange(start=0, stop=null_distribution_array.shape[1], step=1),
                                      size=n_rois)
        with PdfPages(os.path.join(output_directory, "null_distribution_random_roi.pdf")) \
                as pdf:

            for i in range(n_rois):
                plt.subplot(10, 2, i + 1)
                plt.hist(x=null_distribution_array[..., random_roi[i]], bins='auto', edgecolor='black')
                plt.title('# {}'.format(random_roi[i]))
                plt.subplots_adjust(hspace=1, wspace=.01)
            pdf.savefig()
            plt.show()

    # Compute tangent connectivity matrices on the whole
    # reference group
    reference_group_connectivity = ConnectivityMeasure(kind='tangent',
                                                       vectorize=True,
                                                       discard_diagonal=True)

    reference_group_tangent_matrices = reference_group_connectivity.fit_transform(X=reference_group)
    # Compute the tangent mean for the controls group
    reference_group_tangent_mean = reference_group_connectivity.mean_

    # Project each subject's matrices belonging to
    # group to project in the tangent space
    group_to_project_tangent_matrices = reference_group_connectivity.transform(
        X=group_to_project)
    save_object(
        object_to_save=group_to_project_tangent_matrices,
        saving_directory=output_directory,
        filename='projected_group_connectivity_matrices')

    # Test for each projected subject, and for each brain connection
    # if there exist a significant connectivity difference between the
    # a subject and the reference group.
    if statistic == "z":
        # Z-statistic between a projected subject and the whole
        # reference group
        group_to_project_scores = zmap(scores=group_to_project_tangent_matrices,
                                       compare=reference_group_tangent_matrices)
    elif statistic == "t":
        # A one sample t-test between the reference group and a projected subject
        group_to_project_scores = np.array([-1.0 * ttest_1samp(
            a=reference_group_tangent_matrices, popmean=group_to_project_tangent_matrices[p, ...])[0]
                                            for p in range(group_to_project_tangent_matrices.shape[0])])
    else:
        raise ValueError("Statistic type unknown. Type are z or t statistic, "
                         "you entered {}".format(statistic))

    # Compute the raw p value based on the previously
    # computed null distribution, for each subject.
    p_values = np.empty(shape=group_to_project_scores.shape)
    for subject in range(group_to_project_scores.shape[0]):
        for i in range(group_to_project_scores.shape[1]):
            # with the z-score, or the t-score
            p_values[subject, i] = \
                (np.sum(np.abs(null_distribution_array[:, i]) > np.abs(group_to_project_scores[subject, i])) + 1) / \
                (bootstrap_number + 1)

    # correct the p values for each subject
    # in the group to project
    p_values_corrected = np.empty(shape=group_to_project_scores.shape)
    for subject in range(group_to_project_scores.shape[0]):
        p_values_corrected[subject, ...] = multipletests(pvals=p_values[subject, ...],
                                                         method=correction_method,
                                                         alpha=alpha)[1]

    return {"null_distribution": null_distribution_array, "p_values_corrected": p_values_corrected,
            "reference_group_tangent_mean": reference_group_tangent_mean,
            "reference_group_tangent_matrices": reference_group_tangent_matrices,
            "group_to_project_tangent_matrices": group_to_project_tangent_matrices,
            "group_to_project_stats": group_to_project_scores}
