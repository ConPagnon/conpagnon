import pandas as pd
from sklearn.utils import shuffle
import random
from computing import compute_connectivity_matrices as ccm
from utils.array_operation import array_rebuilder
from math import sqrt
import numpy as np
"""
This module contain useful operation on subjects connectivity matrices dictionnary.

Author: Dhaif BEKHA.

"""

# TODO: create a function for merging dictionnary key easily


def groupby_factor_connectivity_matrices(population_data_file, sheetname,
                                         subjects_connectivity_matrices_dictionnary,
                                         groupes, factors,
                                         drop_subjects_list=None):
    """Group by attribute the subjects connectivity matrices.

    """
    # Read the excel containing information of interest
    population_text_data = pd.read_excel(population_data_file, sheet_name=sheetname)

    # Drop subjects if needed
    if drop_subjects_list is not None:
        population_text_data = population_text_data.drop(drop_subjects_list)
    else:
        population_text_data = population_text_data

    # Using pandas, group the dataframe by the factor list entered and store
    # it in a dictionnary
    population_data_by_factor = population_text_data.groupby(factors).groups
    # Store all the keys, i.e all the possible factor pairs.
    factors_keys = list(population_data_by_factor.keys())

    # Create a subjects connectivity matrices dictionnary, with factor keys as first level
    # keys, instead of groupes list
    group_by_factor_subjects_connectivity_matrices = dict.fromkeys(factors_keys)
    # Stack the connectivity matrices
    stacked_matrices = {s: subjects_connectivity_matrices_dictionnary[groupe][s] for groupe in groupes
                        for s in subjects_connectivity_matrices_dictionnary[groupe].keys()}
    # Fill the dictionnary, with the corresponding subject level dictionary for each
    # factor key pair
    for factor in factors_keys:
        # subjects list ID for the current factor pair key
        subject_for_this_factor = list(population_data_by_factor[factor])
        group_by_factor_subjects_connectivity_matrices[factor] = dict.fromkeys(subject_for_this_factor)
        for s in subject_for_this_factor:
            group_by_factor_subjects_connectivity_matrices[factor][s] = stacked_matrices[s]

    return group_by_factor_subjects_connectivity_matrices, population_data_by_factor, factors_keys


def random_draw_of_connectivity_matrices(subjects_connectivity_dictionary, groupe, n_matrices,
                                         subjects_id_list=None, random_state=None, extract_kwargs=None):
    """Randomly pick N connectivity matrices from a subjects connectivity dictionnary.

    Parameters
    ----------
    subjects_connectivity_dictionary: dict
        The subjects dictionnary containing connectivity matrices
    groupe: str
        The group in which you want pick the matrices
    n_matrices: int
        The number of connectivity matrices you want
        to randomly choose
    subjects_id_list: list, optional
        The subjects identifiers list in which you
        want to choose matrices. If None, random matrices
        are picked in the entire group. Default is None.
    random_state: int, optional
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.
    extract_kwargs: dict, optional
        A dictionnary of argument passed to extract_sub_connectivity_matrices
        function. Default is None

    Returns
    -------
    output 1: dict
        The connectivity matrices dictionary, with subjects chosen
        randomly.
    output 2: list
        The  list of randomly chosen subjects identifier.

    """
    random_dictionary = {}
    # If a subjects list ids is entered
    if subjects_id_list is not None:
        subjects_id = list(set(subjects_id_list))
    else:
        # We take all the subjects ids
        subjects_id = list(set(subjects_connectivity_dictionary[groupe].keys()))

    # Shuffle the subjects id list, without impacting the original array
    shuffle_subjects_id = shuffle(subjects_id, random_state=random_state)
    # Randomly pick N subjects
    random_draw_subjects_id = random.sample(shuffle_subjects_id, n_matrices)
    # Extract the corresponding sub-dictionnary
    random_subjects_dictionary = {k: subjects_connectivity_dictionary[groupe][k]
                                  for k in random_draw_subjects_id}

    random_dictionary[groupe] = random_subjects_dictionary
    # if we want specific sub-matrices we extract them
    # extract sub connectivity matrices function
    if extract_kwargs is not None:
        random_dictionary = ccm.extract_sub_connectivity_matrices(
            subjects_connectivity_matrices=random_dictionary,
            kinds=extract_kwargs['kinds'],
            regions_index=extract_kwargs['regions_index'],
            vectorize=extract_kwargs['vectorize'],
            discard_diagonal=extract_kwargs['discard_diagonal'])

    return random_dictionary, random_draw_subjects_id


def merge_dictionary(dict_list, new_key=None):
    """Merge a list of dictionary

    Parameters
    ----------
    new_key: str, optional
        The key of the new merged dictionary. If None, the
        dictionaries in the list are simply merged together.
        Default is None
    dict_list: list
        A list of the dictionary to be merged

    Returns
    -------
    output: dict
        A dictionnary with one key, and merged dictionary
        as value.

    Notes
    -----
    Note that all the dictionnary you want to merge must have
    different keys.
    """
    merged_dictionary = dict()
    dictionary = {}
    for d in dict_list:
        merged_dictionary.update(d)
    if new_key is not None:
        dictionary[new_key] = merged_dictionary
    else:
        dictionary = merged_dictionary
    return dictionary


def stack_subjects_connectivity_matrices(subjects_connectivity_dictionary, groupes, kinds):
    """Re-arrange the subjects connectivity dictionary to return a stack version per group
    and kind.

    :param subjects_connectivity_dictionary:
    :param groupes:
    :param kinds:
    :return:
    """
    # Initialize dictionary
    stack_connectivity_dictionary = dict.fromkeys(groupes)

    for group in groupes:
        subjects_list = subjects_connectivity_dictionary[group].keys()
        stack_connectivity_dictionary[group] = dict.fromkeys(kinds)
        for kind in kinds:
            group_stacked_connectivity = [(s, subjects_connectivity_dictionary[group][s][kind])
                                          for s in subjects_list]
            group_stacked_mask = [(s, subjects_connectivity_dictionary[group][s]['masked_array'])
                                  for s in subjects_list]

            stack_connectivity_dictionary[group][kind] = {
                'connectivity matrices': group_stacked_connectivity,
                'masked_array': group_stacked_mask}

    return stack_connectivity_dictionary


def rebuild_subject_connectivity_matrices(subjects_connectivity_dictionary, groupes, kinds,
                                          diagonal_is_there=False):
    """Given the subject connectivity dictionary, the matrix are rebuild from the vectorized
    one.


    :param subjects_connectivity_dictionary:
    :param groupes:
    :param kinds:
    :param diagonal_is_there:
    :return:
    """
    for group in groupes:
        subjects_in_group = list(subjects_connectivity_dictionary[group].keys())
        for kind in kinds:
            for subject in subjects_in_group:
                # If the diagonal is in the dictionary
                if diagonal_is_there:
                    # Fetch the diagonal of the connectivity matrices
                    subject_kind_diagonal = subjects_connectivity_dictionary[group][subject][kind+'_diagonal']
                    # Fetch the diagonal of the corresponding mask
                    subject_kind_mask_diagonal = \
                        subjects_connectivity_dictionary[group][subject]['diagonal_mask']
                    # Rebuild the connectivity matrix and the mask by
                    # override the present corresponding field
                    subjects_connectivity_dictionary[group][subject][kind] = array_rebuilder(
                        vectorized_array=subjects_connectivity_dictionary[group][subject][kind],
                        diagonal=subject_kind_diagonal, array_type='numeric'
                    )
                    subjects_connectivity_dictionary[group][subject]['masked_array'] = array_rebuilder(
                        vectorized_array=subjects_connectivity_dictionary[group][subject]['masked_array'],
                        array_type='bool', diagonal=subject_kind_mask_diagonal
                    )
                else:
                    # We generate a diagonal of zeros for connectivity matrices,
                    # and a diagonal of True on the diagonal.

                    # Number of coefficient under the diagonal
                    c = len(subjects_connectivity_dictionary[group][subject][kind])
                    # Number of regions: resolve the quadratic equation in number of regions give:
                    n_regions = ((sqrt(8 * c + 1) - 1.) / 2) + 1
                    # Generate zeros diagonal
                    zeros_diag = np.zeros(int(n_regions))
                    # Generate True diagonal
                    true_diag = np.ones(int(n_regions), dtype='bool')
                    # Rebuild the array
                    # override the present corresponding field
                    subjects_connectivity_dictionary[group][subject][kind] = array_rebuilder(
                        vectorized_array=subjects_connectivity_dictionary[group][subject][kind],
                        diagonal=zeros_diag, array_type='numeric'
                    )
                    subjects_connectivity_dictionary[group][subject]['masked_array'] = array_rebuilder(
                        vectorized_array=subjects_connectivity_dictionary[group][subject]['masked_array'],
                        array_type='bool', diagonal=true_diag)

    return subjects_connectivity_dictionary
