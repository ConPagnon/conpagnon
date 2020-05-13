import pandas as pd
from sklearn.utils import shuffle
import random
from conpagnon.computing import compute_connectivity_matrices as ccm
from conpagnon.utils.array_operation import array_rebuilder
import copy
"""
This module contain useful operation on subjects connectivity matrices dictionnary.

Author: Dhaif BEKHA.
"""

# TODO: create a function for merging dictionnary key easily


def groupby_factor_connectivity_matrices(population_data_file, sheetname,
                                         subjects_connectivity_matrices_dictionnary,
                                         groupes, factors,
                                         drop_subjects_list=None,
                                         index_col=0):
    """Group by attribute the subjects connectivity matrices.
    # TODO: 18/09/2019: I added index_col to precise the index of the column
    # TODO: to be considered as the index of the whole dataframe.
    # TODO: Side Note: this function work with a time series dictionary too. !!
    # TODO: Refractoring of subjects_connectivity_matrices_dictionary to subjects_dictionary.
    """
    # Read the excel containing information of interest
    population_text_data = pd.read_excel(population_data_file, sheet_name=sheetname, index_col=index_col)

    # Drop subjects if needed
    if drop_subjects_list is not None:
        population_text_data = population_text_data.drop(drop_subjects_list)
    else:
        population_text_data = population_text_data

    all_subjects_list = []
    for groupe in groupes:
        for s in subjects_connectivity_matrices_dictionnary[groupe].keys():
            all_subjects_list.append(s)
    population_text_data = population_text_data.loc[population_text_data.index.intersection(all_subjects_list)]

    # Using pandas, group the dataframe by the factor list entered and store
    # it in a dictionary
    population_data_by_factor = population_text_data.groupby(factors).groups
    # Store all the keys, i.e all the possible factor pairs.
    factors_keys = list(population_data_by_factor.keys())

    # Create a subjects connectivity matrices dictionary, with factor keys as first level
    # keys, instead of groupes list
    group_by_factor_subjects_connectivity_matrices = dict.fromkeys(factors_keys)
    # Stack the connectivity matrices
    stacked_matrices = {s: subjects_connectivity_matrices_dictionnary[groupe][s] for groupe in groupes
                        for s in subjects_connectivity_matrices_dictionnary[groupe].keys()}
    # Fill the dictionary, with the corresponding subject level dictionary for each
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
    """Randomly pick N connectivity matrices from a subjects connectivity dictionary.

    Parameters
    ----------
    subjects_connectivity_dictionary: dict
        The subjects dictionary containing connectivity matrices
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
                                          diagonal_were_kept=False):
    """Given the subject connectivity dictionary, the matrix are rebuild from the vectorized
    one.

    Parameters
    ----------
    subjects_connectivity_dictionary: dict
        The subjects connectivity dictionary
    groupes: list
        The list of groups to rebuild the subjects matrices.
    kinds: list
        The list of kinds to rebuild.
    diagonal_were_kept: bool, optional
        If True, the reconstructed matrix, will have
        the diagonal store in the kind diagonal field of
        the dictionary, and the mask diagonal field for
        the mask.
        If False, the reconstructed matrix will have
        a zeros diagonal, and a True diagonal for the
        mask.

    Returns
    -------
    output 1: dict
        The reconstructed subjects connectivity
        matrices. All the matrices have now
        shape (number_of_regions, number_of_regions).

    Notes
    -----
    If in the input dictionary, the matrices and corresponding
    mask where vectorized with the diagonal kept, the argument
    `diagonal_is_there` must be set to False. A dimension
    error will be raises otherwise.

    """
    # Copy of the original dictionary to avoid side effect
    subjects_connectivity_dictionary_ = copy.deepcopy(subjects_connectivity_dictionary)

    for group in groupes:
        subjects_in_group = list(subjects_connectivity_dictionary_[group].keys())
        # First, we rebuild the vectorized mask
        for subject in subjects_in_group:
            vectorized_subject_mask = \
                subjects_connectivity_dictionary_[group][subject]['masked_array']
            # If the diagonal where kept in the vectorization process
            if diagonal_were_kept:
                # We simply rebuild the boolean mask
                rebuild_subject_mask = array_rebuilder(
                    vectorized_array=vectorized_subject_mask,
                    array_type='bool', diagonal=None)
            else:
                # We fetch the mask diagonal in the corresponding
                # subject dictionary
                subject_kind_mask_diagonal = \
                    subjects_connectivity_dictionary_[group][subject]['diagonal_mask']
                rebuild_subject_mask = array_rebuilder(
                    vectorized_array=vectorized_subject_mask,
                    array_type='bool', diagonal=subject_kind_mask_diagonal)
            # Now, iterate over the kind to rebuild the subject connectivity
            # matrices
            for kind in kinds:
                # Fetch the vectorized connectivity matrice of the current subject
                vectorized_subject_matrix = \
                    subjects_connectivity_dictionary_[group][subject][kind]
                # If the diagonal where kept in the vectorization process
                if diagonal_were_kept:
                    # Rebuild the connectivity matrices
                    rebuild_subject_matrix = array_rebuilder(
                        vectorized_array=vectorized_subject_matrix,
                        diagonal=None, array_type='numeric'
                    )
                else:
                    # Fetch the kind diagonal
                    subject_kind_diagonal = subjects_connectivity_dictionary_[group][subject][kind + '_diagonal']
                    # Rebuild the array
                    # override the present corresponding field
                    rebuild_subject_matrix = array_rebuilder(
                        vectorized_array=vectorized_subject_matrix,
                        diagonal=subject_kind_diagonal, array_type='numeric'
                    )

                subjects_connectivity_dictionary_[group][subject][kind] = rebuild_subject_matrix
                subjects_connectivity_dictionary_[group][subject]['masked_array'] = rebuild_subject_mask

    return subjects_connectivity_dictionary_
