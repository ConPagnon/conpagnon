import csv
import os
import pandas as pd
import statsmodels
import numpy as np
from conpagnon.utils.folders_and_files_management import create_directory
from functools import reduce
"""Text file management, output results, modify
 and append information to text files

Author: Dhaif BEKHA.
"""


def read_csv(csv_file, delimiter=','):
    """Read a CSV file and return a panda.DataFrame

     Parameters
     ----------
     csv_file: str
        The full path to the CSV file to read
     delimiter: str
        The separator use in the CSV file

    """
    # Read the csv file
    data = pd.read_csv(csv_file, delimiter=delimiter)

    return data


def csv_from_dictionary(subjects_dictionary, groupes, kinds, field_to_write,
                        header, csv_filename, output_directory, delimiter=','):
    """Write a csv file from a subjects dictionary.

    Parameters
    ----------
    subjects_dictionary: dict
        A dictionnary with the same structure as a
        subjects connectivity matrices dictionary
    groupes: list
        The list of groups to write
    kinds: list
        The list of kind to write
    field_to_write: str
        The field containing the value to write
        for each subject.
    header: list
        The header of the CSV file, in a list of
        column name
    csv_filename: str
        The end of CSV filename with the extension
    output_directory: str
        The full path to a directory for saving
        the CSV file.
    delimiter: str, optional
        The delimiter between columns.
        Default is a comma.

    """
    # Write the corresponding field for each subject
    # in each groups and kinds.
    for group in groupes:
        for kind in kinds:
            output_csv = os.path.join(
                output_directory, group + '_' + kind + '_' + csv_filename)
            with open(output_csv, 'w') as csv_file:
                # Initialize a writer object
                writer = csv.writer(csv_file, delimiter=delimiter)
                # The first row is the header
                writer.writerow(header)
                # Write for each subject, the corresponding connectivity value
                # field
                for subject, subject_sub_dictionary in subjects_dictionary[group].items():
                    writer.writerow([subject, subject_sub_dictionary[kind][field_to_write]])


def csv_from_intra_network_dictionary(subjects_dictionary, groupes, kinds, network_labels_list,
                                      field_to_write, output_directory, csv_prefix, delimiter=',',
                                      ):
    """Write csv file from the intra-network connectivity dictionary structure.

    """
    for group in groupes:
        for kind in kinds:
            for network in network_labels_list:
                # Create the output directory
                header = ['subjects', 'intra_' + network + '_connectivity']
                create_directory(directory=os.path.join(
                    output_directory, kind, network), erase_previous=False)
                # CSV filename
                csv_filename = group + '_' + csv_prefix + '_' + network + '_connectivity.csv'
                output_csv = os.path.join(
                    output_directory, kind, network, csv_filename)
                with open(output_csv, 'w') as csv_file:
                    # Initialize a writer object
                    writer = csv.writer(csv_file, delimiter=delimiter)
                    # The first row is the header
                    writer.writerow(header)
                    # Write for each subject, the corresponding connectivity value
                    # field
                    for subject, subject_sub_dictionary in subjects_dictionary[group].items():
                        writer.writerow(
                            [subject, subject_sub_dictionary[kind][network][field_to_write]])


def dataframe_to_csv(dataframe, path, delimiter=',', index=False):
    """Create and write a CSV file from a DataFrame

    """
    dataframe.to_csv(path, sep=delimiter, index=index)


def read_excel_file(excel_file_path, sheetname,
                    subjects_column_name):
    """Read a excel document

    Parameters
    ----------
    excel_file_path: str
        Full path to the excel document
    sheetname: str
        The sheetname to read in the excel document
    subjects_column_name: str
        The column name containing the subjects identifiers.

    Returns
    -------
    output: pandas.DataFrame
        A panda DataFrame, indexed by subject name.
    """

    data = pd.read_excel(io=excel_file_path, sheet_name=sheetname)
    # Shift the index
    data = data.set_index(subjects_column_name)

    return data


def shift_index_column(panda_dataframe, columns_to_index):
    """Shift the index column of a pandas DataFrame

    Parameters
    ----------
    panda_dataframe: pandas.DataFrame
        A pandas dataframe.
    columns_to_index: list
        Column label or list of column labels / arrays

    Returns
    -------
    output: pandas.DataFrame
        A new pandas DataFrame with the shifted columns as index.
    """
    new_data = panda_dataframe.set_index(columns_to_index)

    return new_data


def concatenate_dataframes(list_of_dataframes, axis=0):
    """Concatenate a list of pandas DataFrame

    """
    new_data = pd.concat(list_of_dataframes, axis=axis)

    return new_data


def merge_by_index(dataframe1, dataframe2, left_index=True, right_index=True):
    """Merge two dataframes based on the index concordances

    Parameters
    ----------
    dataframe1: pandas.DataFrame
        A panda dataframe
    dataframe2: pandas.DataFrame
        A panda dataframe
    left_index: bool, optional
        If True, the merge operation is based on the left index
    right_index: bool, optional
        If True, the merge operation is based on the right index

    Returns
    -------
    output: pandas.DataFrame
        The merged dataframe.

    Notes
    -----
    If `left_index` and `right_index` are both True
    the merge  is based on the intersection of both dataframe,
    i.e a missing index in one of the dataframe will be deleted in
    the final dataframe.
    """
    merged_dataframe = pd.merge(dataframe1, dataframe2, left_index=left_index,
                                right_index=right_index)

    return merged_dataframe


def write_ols_results(ols_fit, design_matrix, response_variable, output_dir, model_name,
                      design_matrix_index_name=None):
    """Write OLS result, along with the design matrix and the variable to explain.
    """
    # Check if we have a statmodels OLS object
    if not isinstance(ols_fit, statsmodels.regression.linear_model.RegressionResultsWrapper):
        raise TypeError('expected a statsmodels regression results wrapper '
                        'but got type {} instead'.format(type(ols_fit)))

    header = ['variables', 'coefficients', 'std_error', 't', 'p_value', 'conf_inf', 'conf_sup']
    # Construction of DataFrame based on the fit result of the model
    # we transpose it to get a column structure
    ols_results_dataframe = pd.DataFrame([ols_fit.params, ols_fit.bse, ols_fit.tvalues,
                                          ols_fit.pvalues, ols_fit.conf_int()[0],
                                          ols_fit.conf_int()[1]]).T
    ols_results_dataframe = ols_results_dataframe.reset_index()
    ols_results_dataframe.columns = header
    # We write the result in a CSV file
    with open(os.path.join(output_dir, model_name + '_parameters.csv'), 'w') as csvfile:
        # Save the results: coefficients, t-values, p-values, standard error,
        # and confidence interval
        ols_results_dataframe.to_csv(csvfile, index=False)

    with open(os.path.join(output_dir, model_name + '_qualitity_fit.csv'), 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Save r squared, adjusted r squared, degrees of freedom
        csv_writer.writerow(['r_squared', 'adj_r_squared', 'n_obs', 'df_model', 'df_resid'])
        csv_writer.writerow([ols_fit.rsquared, ols_fit.rsquared_adj, ols_fit.nobs, ols_fit.df_model,
                             ols_fit.df_resid])

    with open(os.path.join(output_dir, model_name + '_design_matrix.csv'), 'w') as csvfile:
        # Save the design matrix, and the response variable in the
        # same dataframe
        data = concatenate_dataframes([design_matrix, response_variable], axis=1)
        if design_matrix_index_name is not None:
            data.index.name = design_matrix_index_name

        data.to_csv(csvfile, index=True)

    # save the prediction for plotting purpose: numpy will be enough
    np.savetxt(os.path.join(output_dir, model_name + '_prediction.csv'), np.c_[ols_fit.predict()],
               header='prediction', comments='')


def group_by_factors(dataframe, list_of_factors, return_type='list_of_dataframe'):
    """Group by factors present in a dataframe

    Parameters
    ----------
    dataframe: pandas.DataFrame
        A pandas dataframe.
    list_of_factors: list
        The list of factors, i.e columns name in the dataframe,
        you want to group by.
    return_type: str
        The output format, choices are `list_of_dataframe` or
        `dictionary`. If the former, a list of dataframe is returned
        of length equal to the number of groups, if the latter a dictionary
        with groups name as keys and corresponding dataframe as values is returned.
        Default is `list_of_dataframe`.

    Returns
    -------
    output:
        A list or dictionary of the corresponding dataframe group by attribute.
    """
    # Group by the list of factors
    grouped_dataframe = dataframe.groupby(list_of_factors)
    # Get the groups keys name
    groups_names = grouped_dataframe.groups.keys()
    # Depending on the wanted return_type, construct
    # a dictionary of a list
    if return_type is 'list_of_dataframe':
        # Initialize a list containing the dataframe
        list_of_dataframes = []
        for group in groups_names:
            list_of_dataframes.append(grouped_dataframe.get_group(name=group))

        dataframe_by_group = list_of_dataframes
    elif return_type is 'dictionary':
        # Initialize a dictionary containing groups name
        # as keys, and the corresponding group dataframe as values
        grouped_dataframe_dictionary = {group_name: grouped_dataframe.get_group(name=group_name) for
                                        group_name in groups_names}

        dataframe_by_group = grouped_dataframe_dictionary
    else:
        raise ValueError('return type not understood. Choices are dictionary, list_of_dataframe'
                         'and you enter {}'.format(return_type))

    return dataframe_by_group


def dictionary_to_csv(dictionary, output_dir, output_filename):
    """Write dictionary couple (key, value) in a CSV file
    """

    with open(os.path.join(output_dir, output_filename), 'w') as f:
        w = csv.writer(f)
        w.writerows(dictionary.items())


def merge_list_dataframes(list_dataframes):
    """Merge a list of dataframes
    """
    df = reduce(lambda df1, df2: pd.merge(df1, df2, left_index=True, right_index=True),
                list_dataframes)

    return df


def remove_duplicate(seq):
    """Remove duplicate in a sequence of items while keeping
    the order.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def _flatten(values):
    if isinstance(values, np.ndarray):
        yield values.flatten()
    else:
        for value in values:
            yield from _flatten(value)


def _unflatten(flat_values, prototype, offset):
    if isinstance(prototype, np.ndarray):
        shape = prototype.shape
        new_offset = offset + np.product(shape)
        value = flat_values[offset:new_offset].reshape(shape)
        return value, new_offset
    else:
        result = []
        for value in prototype:
            value, offset = _unflatten(flat_values, value, offset)
            result.append(value)
        return result, offset


def flatten(values):
    """Flatten a list of numpy ND-array

    Parameters
    ----------
    values: list
        A list of numpy array, with same or different dimensions.

    Returns
    -------
    output: numpy.array
        A flat array (one dimensional array) containing all
        the values in the same order of the list of array.

    """
    return np.concatenate(list(_flatten(values)))


def unflatten(flat_values, prototype):
    """Unflatten a one dimension array of values to the
    original list of array.

    Parameters
    ----------
    flat_values: numpy.ndarray
        The numpy array containing the values.
    prototype: list
        The original list of numpy array.

    Returns
    -------
    output: list
        A list of array with the same structure as prototype.
    """
    result, offset = _unflatten(flat_values, prototype, 0)
    assert(offset == len(flat_values))
    return result
