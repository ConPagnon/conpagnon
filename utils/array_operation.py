#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:58:24 2017

@author: Dhaif BEKHA


Useful operation on arrays.
"""

import numpy as np
from numpy import ma
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
import re
from nilearn.image import concat_imgs
import nibabel as nb
import glob
import os
from math import sqrt


def masked_void_rois_connectivity_matrix(subject_kinds_connectivity, kinds,
                                         discard_diagonal=False,
                                         vectorize=False):

    """Compute a boolean mask array for discarded rois 
    
    Parameters
    ----------
    subject_kinds_connectivity: dict
        The individual subject dictionnary containing as keys:
            - The different kinds
            - A keys called 'discarded_rois' for ROIs to discard during analysis
            
    kinds : list 
        List of present kinds in the subjects connectivity matrices
        dictionnary.

    discard_diagonal: bool, optional
        If True, the diagonal are discarded in the vectorization
        process. Default is False, the diagonal is kept.

    vectorize: bool, optional
        If True, the returned mask is vectorized. Default
        is False

    Returns
    -------
    output : numpy.array, shape (number of regions, number of regions)
        A boolean array : True when rois are considered discarded, False elsewhere
        See notes concerning the discarded roi
        
    See Also
    --------
    compute_connectivity_matrices.individual_connectivity_matrices :
        This is the function which compute the connectivity matrices for
        the chosen kinds, and return a structured dictionnary containing
        all the connectivity matrices for each subjects and kinds
        
    Notes
    -----
    A discarded rois, is a ROI where the corresponding labels is 'void' in the
    individual atlas of a given subject. This information is automatically gathered 
    when you extract time series with an individual atlas for each subjects. We account
    for discarded rois when computing arithmetic mean, or performing t-test for
    examples by excluding them during the computation.
    
    Excluding the discarded rois, is possible via the numpy.ma structure which
    build a numpy masked array that is a couple of a numerical array and a boolean
    array where True are masked values, and False are un-masked values.
    
    Please refer to the numpy documentation for further details.
    """
    # We fetch the number of regions
    # Get the shape of the matrices
    matrix_shape = subject_kinds_connectivity[kinds[0]].shape

    # We fetch the index of discarded rois
    subject_empty_roi = subject_kinds_connectivity['discarded_rois']
    
    # Masked array: True for discarded rois, else elsewhere.

    # If we have a 2D subjects connectivity matrices, we initialize a 2D boolean mask
    if not vectorize:
        numbers_of_regions = matrix_shape[0]
        mask = np.invert(ma.make_mask(np.ones((numbers_of_regions, numbers_of_regions))))
        if subject_empty_roi.size:
            for empty_roi in subject_empty_roi:
                # True for the entire rows corresponding to the current empty roi
                mask[:, empty_roi] = True
                # True for the entire columns corresponding to the current empty roi
                mask[empty_roi, :] = True
        # Compute the diagonal of the mask
        diag_mask = np.diagonal(mask)
    else:
        if discard_diagonal:
            c = len(subject_kinds_connectivity[kinds[0]])
            numbers_of_regions = int(((sqrt(8 * c + 1) - 1.) / 2) + 1)
            # Initialise a vectorized mask
            mask = np.invert(ma.make_mask(np.ones(numbers_of_regions)))
            # Reconstruct the mask to a matrix
            _ , mask_m = vectorizer(numpy_array=mask, discard_diagonal=discard_diagonal,
                                  array_type='boolean')
            # Fill the corresponding empty ROI with True value
            if subject_empty_roi.size:
                for empty_roi in subject_empty_roi:
                    # True for the entire rows corresponding to the current empty roi
                    mask_m[:, empty_roi] = True
                    # True for the entire columns corresponding to the current empty roi
                    mask_m[empty_roi, :] = True

            # Re-vectorized the matrix boolean mask discarding
            # the diagonal
            diag_mask, mask = vectorizer(numpy_array=mask_m,
                                         discard_diagonal=discard_diagonal,
                                         array_type='boolean')
        else:
            c = len(subject_kinds_connectivity[kinds[0]])
            numbers_of_regions = int(((sqrt(8 * c + 1) - 1.) / 2))
            # Initialise a vectorized mask
            mask = np.invert(ma.make_mask(np.ones(numbers_of_regions)))
            # Reconstruct the mask to a matrix
            _, mask_m = vectorizer(numpy_array=mask, discard_diagonal=False,
                                   array_type='boolean')
            # Fill the corresponding empty ROI with True value
            if subject_empty_roi.size:
                for empty_roi in subject_empty_roi:
                    # True for the entire rows corresponding to the current empty roi
                    mask_m[:, empty_roi] = True
                    # True for the entire columns corresponding to the current empty roi
                    mask_m[empty_roi, :] = True
            # If discard_diagonal is False, we vectorized
            # the mask keeping the diagonal
            diag_mask, mask = vectorizer(numpy_array=mask_m,
                                         discard_diagonal=False,
                                         array_type='boolean')

    return mask, diag_mask


def append_masks(subjects_connectivity_dictionnary, kinds,
                 discard_diagonal, vectorize):

    """Append a new key called 'masked_array' in the subjects connectivity
    matrices dictionnary and stock as values a boolean mask array accounting
    for discarded rois.
    
    Parameters
    ----------
    subjects_connectivity_dictionnary : dict
        A multi-levels dictionnary organised as follow :
            - The first keys levels is the different groupes in the study. 
            - The second keys levels is the subjects IDs
            - The third levels is the different kind matrices
            for each subjects, a 'discarded_rois' key for the
            discarded rois array index.
    
    kinds : list
        List of present kinds in the subjects connectivity matrices
        dictionnary.

    discard_diagonal: bool
        If True, the mask diagonal is discarded in the
        vectorization process.
        This argument is passed to masked_void_rois_connectivity_matrix

    vectorize: bool
        If True, the returned mask is vectorized.
        This argument is passed to masked_void_rois_connectivity_matrix

    Returns
    -------
    output : dict
        The modified subjects connectivity matrices with a new key for
        each subjects containing the boolean mask array for the discarded ROIs.
        A new keys is added too, which contain the diagonal array of the mask.

        
    Notes
    -----
    If the discarded rois keys do not exist, that is no labels 'void' is found 
    in the atlas labels file of each subjects, then all roi are included in the
    analysis, and we append a boolean mask array of False values.
    
    
    """
    for groupe in subjects_connectivity_dictionnary.keys():
        subjects_list = subjects_connectivity_dictionnary[groupe].keys()
        for subject in subjects_list:
            # Entries in subject dictionnary, for the empty rois field.
            subject_kind_connectivity_matrice = subjects_connectivity_dictionnary[groupe][subject]
            # Creation of the mask array for void rois based on the field 'empty_rois'
            if 'discarded_rois' in subjects_connectivity_dictionnary[groupe][subject].keys():
                if vectorize:
                    if discard_diagonal:
                        subject_mask, subject_diag_mask = masked_void_rois_connectivity_matrix(
                            subject_kinds_connectivity=subject_kind_connectivity_matrice,
                            kinds=kinds, discard_diagonal=discard_diagonal,
                            vectorize=vectorize)
                    else:
                        subject_mask, subject_diag_mask = masked_void_rois_connectivity_matrix(
                            subject_kinds_connectivity=subject_kind_connectivity_matrice,
                            kinds=kinds, discard_diagonal=False,
                            vectorize=vectorize)

                    # We create new keys containing the masked array.
                    subjects_connectivity_dictionnary[groupe][subject]['masked_array'] = subject_mask
                    # And the corresponding diagonal
                    subjects_connectivity_dictionnary[groupe][subject]['diagonal_masked_array'] = subject_diag_mask
                else:
                    subject_mask, subject_diag_mask = masked_void_rois_connectivity_matrix(
                        subject_kinds_connectivity=subject_kind_connectivity_matrice,
                        kinds=kinds, discard_diagonal=False,
                        vectorize=False)

                    # We create new keys containing the masked array.
                    subjects_connectivity_dictionnary[groupe][subject]['masked_array'] = subject_mask
                    subjects_connectivity_dictionnary[groupe][subject]['diagonal_masked_array'] = subject_diag_mask
            else:
                subject_mask = np.invert(ma.make_mask(np.ones(
                    subjects_connectivity_dictionnary[groupe][subject][kinds[0]].shape)))
                if vectorize:
                    if discard_diagonal:
                        subject_diag_mask, subject_mask = vectorizer(numpy_array=subject_mask,
                                                                     discard_diagonal=discard_diagonal,
                                                                     array_type='boolean')
                    else:
                        subject_diag_mask, subject_mask = vectorizer(numpy_array=subject_mask,
                                                                     discard_diagonal=False,
                                                                     array_type='boolean')

                    subjects_connectivity_dictionnary[groupe][subject]['masked_array'] = subject_mask
                    subjects_connectivity_dictionnary[groupe][subject]['diagonal_masked_array'] = subject_diag_mask
                else:
                    subject_mask = subject_mask
                    subjects_connectivity_dictionnary[groupe][subject]['masked_array'] = subject_mask

    new_subjects_connectivity_dictionnary = subjects_connectivity_dictionnary
    
    return new_subjects_connectivity_dictionnary


def masked_arrays_mean(arrays, masked_array, axis):
    """Compute the arithmetic mean of a ndimensional numpy masked array.
    
    Parameters
    ----------
    arrays : ndimensional numpy.array, shape (..., number of regions, number of regions)
        The ndimensional array you want to compute the mean.
        
    masked_array : ndimensional numpy.array, shape (..., numbers_of_regions, numbers_of_regions)
        The ndimensional boolean masked accounting for the values you want to discard.
        True for masked values, False elsewhere.
    
    axis : int
        The direction for computing the mean.
    
    Returns
    -------
    output : numpy.array shape (number of regions, number of regions)
        The mean array excluding some value in the direction along axis.
    
    See Also
    --------
    masked_void_rois_connectivity_matrix :
        This function compute the boolean mask for accounting
        the discarded rois.
    
    """
    # Initialization of masked array: a couple of a classical numpy array along with
    # a boolean array of the same shape
    mean_array_masked = np.ma.array(data=arrays, mask=masked_array)
    if masked_array.shape != arrays.shape:
        raise ValueError('Array dimension is {}, but the '
                         'corresponding mask is shape {}'.format(arrays.shape, masked_array.shape))
    
    mean_array = mean_array_masked.mean(axis=axis).data
    
    return mean_array


def mask_array_std(arrays, masked_array, axis):
    """Compute the standard deviation of a ndimensional numpy array.
    
    Parameters
    ----------
    arrays : ndimensional numpy.array, shape (number of subjects, number of regions, number of regions)
        The ndimensional array you want to compute the mean.
        
    masked_array : ndimensional numpy.array, shape (..., numbers_of_regions, numbers_of_regions)
        The ndimensional boolean masked accounting for the values you want to discard.
        True for masked values, False elsewhere.
    
    axis : int
        The direction for computing the standard deviation.
    
    Returns
    -------
    output : numpy.array shape (number of regions, number of regions)
        The standard deviation array excluding some value in the direction along axis.
    
    
    See Also
    --------
    masked_void_rois_connectivity_matrix :
        This function compute the boolean mask for accounting
        the discarded rois.
    
    """
    
    # Masked array initialization with numpy masked methods.
    std_array_masked = np.ma.array(data=arrays, mask=masked_array)
    if masked_array.shape != arrays.shape:
        raise ValueError('The numerical array have shape {}, but '
                         'the corresponding mask is shape {} !'.format(arrays.shape,
                                                                       masked_array.shape))

    std_array = std_array_masked.std(axis=axis)
    
    return std_array


def vectorize_boolean_mask(symmetric_boolean_mask, discard_diagonal=False):
    """Vectorize the lower part of symmetric boolean mask. 
    
    Parameters
    ----------
    symmetric_boolean_mask : numpy.array, shape(..., numbers of regions, numbers of regions)
        A ndimensional boolean numpy array.
        
    discard_diagonal : bool, optional
        If True, the values of the diagonal are not returned.
        Default is False.
        
    Returns
    -------
    output : numpy.ndarray
        The flatten ndimensional boolean mask of shape (..., n_features * (n_features + 1) / 2) 
        if discard_diagonal is False and (..., (n_features - 1) * n_features / 2) otherwise.
        
    See Also
    --------
    nilearn.connectome.sym_matrix_to_vec, vec_to_sym_matrix :
        Useful Nilearn function to vectorize and reconstruct 
        symmetric array like the connectivity matrices.
    
    vec_to_matrix_boolean_mask :
        This function reconstruct a flatten boolean ndimensional array.
        
    """
    # Convert the boolean mask to a numerical array
    numerical_mask = np.asanyarray(symmetric_boolean_mask, dtype=int)

    # vectorize the mask
    symmetric_boolean_mask_vectorized = sym_matrix_to_vec(symmetric=numerical_mask, discard_diagonal=discard_diagonal)
    
    if not discard_diagonal:
        symmetric_boolean_mask_vectorized = np.asanyarray(symmetric_boolean_mask_vectorized, dtype=bool)

    return symmetric_boolean_mask_vectorized


def vec_to_matrix_boolean_mask(vectorized_mask, diagonal=None):
    """Reconstruct a ndim symmetrical boolean mask array giving a boolean vectorized mask.
    
    Parameters
    ----------
    vectorized_mask : numpy.ndarray, shape(..., number of regions, number of regions)
        The vectorized boolean mask.    
    
    
    diagonal : numpy.array of shape(number of regions), optional
        The diagonal to reconstruct if you discard the diagonal when
        you first vectorize the ndimensional mask array
        
    Returns
    -------
    output : numpy.ndarray
        The reconstruct ndimensional array of shape (..., numbers of regions, numbers of regions)
        
    See Also
    --------
    nilearn.connectome.sym_matrix_to_vec, vec_to_sym_matrix :
        Useful Nilearn function to vectorize and reconstruct 
        symmetric array like the connectivity matrices.
    
    vectorize_boolean_mask :
        This function flatten a 
        boolean ndimensional array.
    """
    # mask reconstruction
    numerical_mask = vec_to_sym_matrix(vec=vectorized_mask, diagonal=diagonal)
    
    # Convert numerical mask to the boolean mask:
    symmetric_boolean_mask_reconstructed = np.asanyarray(numerical_mask, dtype=bool)

    return symmetric_boolean_mask_reconstructed


def vectorizer(numpy_array, discard_diagonal=False, array_type='numeric'):
    """Vectorize a numerical or boolean numpy array

    Parameters
    ----------
    numpy_array: numpy.array, shape (n_features, n_features)
        A two-dimensional numerical, or boolean data type array.

    discard_diagonal: bool, optional
        If True, the lower triangle of the array is flatten without the diagonal. Default is False.

    array_type: string, optional
        Precise the array type, choices are {'numeric', boolean'}, default is 'numeric'.

    Returns
    -------
    output 1: numpy.array, shape(n_features)
        The diagonal of the entered array.
    output 2: numpy.array, shape( (n_features*(n_features+1))/2 )
    is discard_diagonal is True, (n_features*(n_features-1))/2 else.
        The vectorized array.

    """

    # Retrieve the array diagonal
    array_diagonal = np.diag(numpy_array)

    # if it's a numerical array
    if array_type == 'numeric':
        vectorized_array = sym_matrix_to_vec(numpy_array, discard_diagonal=discard_diagonal)
    elif array_type == 'boolean':

        # two case: first discard diagonal is False
        if discard_diagonal is False:
            vectorized_array = vectorize_boolean_mask(symmetric_boolean_mask=numpy_array, discard_diagonal=False)
        else:
            vectorized_array = vectorize_boolean_mask(symmetric_boolean_mask=numpy_array, discard_diagonal=True)
            # vectorized_array is numerical, we force it to boolean type
            vectorized_array = np.asanyarray(vectorized_array, dtype=bool)
    else:
        raise ValueError('Unrecognized array type, possible choices are: \n '
                         'numeric or boolean, and you entered:{}'.format(array_type))

    return array_diagonal, vectorized_array


def array_rebuilder(vectorized_array, array_type, diagonal=None):
    """Reconstruct a square array assuming a it is symmetric

    Parameters
    ----------
    vectorized_array: numpy.array
        The one dimensional array you want to rebuild.
    diagonal: numpy.array, optional
        The one dimensional array containing the diagonal of
        the non-vectorized array.
        If None, we assume that the diagonal was kept in the
        vectorization process.
    array_type: str
        The type of the array.
        Choices are: numeric, bool.

    Returns
    -------
    output 1: numpy.array
        The reconstructed array, shape (n_features, n_features).

    See Also
    --------
    vectorizer:
        This function vectorized a two dimensional numeric
        or boolean array
    nilearn.connectome.vec_to_sym_matrix:
        This function reconstruct a symmetric array from a one
        dimensional array.
    """

    if array_type == 'numeric':
        # We divide by sqrt(2), preserving the norm if the user
        # give a diagonal
        if diagonal is not None:
            rebuild_array = vec_to_sym_matrix(vec=vectorized_array,
                                              diagonal=(1/sqrt(2))*diagonal)
        else:
            # If the user doesn't give a diagonal we assume it
            # was vectorized without discarding the diagonal

            # rebuild the array
            rebuild_array = vec_to_sym_matrix(vec=vectorized_array,
                                              diagonal=None)

    elif array_type == 'bool':
        if diagonal is not None:
            # If the user give the boolean diagonal
            # Check if the diagonal is a boolean
            if np.issubdtype(diagonal.dtype, np.bool):

                rebuild_array = np.array(vec_to_sym_matrix(vec=vectorized_array,
                                                           diagonal=diagonal),
                                         dtype='bool')
            else:
                raise TypeError('Diagonal should be type bool, and you give type {}'.
                                format(np.dtype(diagonal)))
        else:
            # If the user doesn't give a diagonal we assume
            # the mask was vectorized with the diagonal

            # We rebuild the boolean mask
            rebuild_array = np.array(vec_to_sym_matrix(vec=vectorized_array,
                                                       diagonal=None), dtype='bool')

    else:
        raise ValueError('Array type not understood, choices are: numeric or bool')

    return rebuild_array


def concatenate_imgs_in_order(imgs_directory, index_roi_number, filename,
                              save_directory):
    """Concatenate 3D NifTi images into a single 4D NifTi files, given a user order.

    Parameters
    ----------
    imgs_directory: string
        The directory containing the list of images. Each file, must contain a number
        identifying the file.
    index_roi_number: list of int
        The list of file number in the order you want to ordered
        the images before concatenation
    filename: string
        The final name of the 4D file containing the images.
    save_directory: string
        The directory where to save the 4D generated 4D file.

    Returns
    -------
    output: list
        Return the list of the pull path to roi files,
        in the order of concatenation. It should exactly follow the order you provide
        via the variable `index_roi_number`.

    See Also:
    --------
    nilearn.image.concat_img: This function concatenate a list of 3D NifTi images.
        We use this function to concatenate the images.

    Notes
    -----
    Each file you want to concatenate, must have a pattern to identifying them. Indeed,
    we use it to search the entire path corresponding to the file, and append it in a list
    containing all the path to the files in the right order before concatenation. The pattern
    is simply a number in the file name.

    """
    unordered_roi_list = glob.glob(os.path.join(imgs_directory, '*.nii'))

    # Fetch the correspond fmri file
    ordered_roi_list = []
    for roi in index_roi_number:
        # Search the string containing the roi_index we want
        regex = re.compile(".*(" 'roi_' + str(roi) + '.nii' ").*")
        roi_path = [m.group(0) for l in unordered_roi_list for m in [regex.search(l)] if m]
        # Append the right path
        ordered_roi_list.append(roi_path[0])

    # Concatenate the image following the order in ordered_roi_list
    img4d = concat_imgs(ordered_roi_list)

    # Save the new image
    nb.save(img=img4d, filename=os.path.join(save_directory, filename))

    return ordered_roi_list


def check_2d(numpy_array):
    """Check if an array is two dimensional.

    Parameters
    ----------
    numpy_array: numpy.array of any shape.

    Raises
    ------
    ValueError : if the passed array is NOT two dimensional, a Value Error
    is raised.

    """

    if numpy_array.ndim != 2:
        raise ValueError('Expected two dimensional array but '
                         'passed array shape is {}'.format(numpy_array.shape))


def truncate(f, n):
    """Truncates/pads a float f to n decimal places without rounding

    """

    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])
