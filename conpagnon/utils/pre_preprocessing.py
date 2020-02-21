#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:04:32 2017

@author: db242421 (dhaif.bekha@cea.fr)


Useful pre-processing step before doing statistical test.
"""

from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
import numpy as np


def fisher_transform(symmetric_array):
    """Return the Fisher transform of all coefficient in a symmetric array.
    
    Parameters
    ----------
    symmetric_array : numpy.ndarray, shape(..., number of regions, numbers of regions)
        A symmetric numpy array, typically correlation, or partial correlation
        subject connectivity matrices.
    
    Returns
    -------
    output : numpy.ndarray, shape(..., number of regions, numbers of regions)
        The fisher transform of the ndimensional array.
        
    Notes
    -----
    The fisher transform is classically computed according to the 
    formula :
        
        .. math::
            z = arctanh(r)
    
    For **r = 1**, the Fisher transform is not defined, typically on the diagonal
    of connectivity matrices. Then we fill diagonal value with np.nan values.
    
    
    """

    # Discarding diagonal which is one for correlation, and partial correlation, not defined in Z-Fisher transformation
    vectorize_array = sym_matrix_to_vec(symmetric = symmetric_array, discard_diagonal=False)
    
    # Apply Fisher transform, i.e the bijection of hyperbolic tangent:
    fisher_vectorize_array = np.arctanh(vectorize_array)

    # Reconstruct the symmetric matrix:
    fisher_transform_array = vec_to_sym_matrix(vec = fisher_vectorize_array, diagonal=None)
    
    # Number of dimension of fisher transform array
    fisher_array_ndim = fisher_transform_array.ndim
    
    # if ndim = 2, put NaN in the diagonal:
    if fisher_array_ndim == 2:
        
        np.fill_diagonal(a = fisher_transform_array, val = np.nan)
    # if ndim > 2, we consider fisher_transform_array as ndimensional array with shape (..., n_features, n_features),
    # we put nan in the diagonal of each array, along the first dimension
    elif fisher_array_ndim > 2:
        for i in range(fisher_transform_array.shape[0]):
            np.fill_diagonal(a = fisher_transform_array[i,:,:], val = np.nan)

    return fisher_transform_array


def stacked_connectivity_matrices(subjects_connectivity_matrices, kinds):
    """Stacked the connectivity matrices for each group and for each kinds.
    
    If a masked array exist, i.e a 'masked_array' key in each subjects dictionnary, we
    stacked them too.
    
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
        List of kinds you are interrested in.
        
    Returns
    -------
    ouput : dict
        A dictionnary with two keys : kinds and masked_array. The values
        is respectively the stacked matrices for one kind in a ndimensional
        array of shape (..., numbers of regions, numbers of regions), and the
        ndimensional boolean mask array of the same shape (..., numbers of regions,
        numbers of regions).
    
    """
    #Dictionnary which will contain the stack of connectivity matrices for each groups and kinds.
    stacked_connectivity_dictionnary = dict.fromkeys(subjects_connectivity_matrices)
    for groupe in subjects_connectivity_matrices.keys():
        #Dictionnary initialisation with all kinds as entry
        stacked_connectivity_dictionnary[groupe] = dict.fromkeys(kinds)
        #subjects list for the groupe
        subjects_list = subjects_connectivity_matrices[groupe].keys()
        for kind in kinds:
            #We stack the matrices for the current kind
            kind_groupe_stacked_matrices = [subjects_connectivity_matrices[groupe][subject][kind] for subject in subjects_list]
            #We detect if the keys named 'masked_array' existe to create a numpy masked array strucutre.
            kind_masked_array = [ subjects_connectivity_matrices[groupe][subject]['masked_array'] for subject in subjects_list if 'masked_array' in subjects_connectivity_matrices[groupe][subject].keys() ]
            
            #We fill a dictionnary with kind connectivity as keys, and stacked connectivity matrices, with masked array if they exist as values
            stacked_connectivity_dictionnary[groupe][kind] = np.array(kind_groupe_stacked_matrices)
        
            stacked_connectivity_dictionnary[groupe]['masked_array'] = np.array(kind_masked_array)
            
    return stacked_connectivity_dictionnary
