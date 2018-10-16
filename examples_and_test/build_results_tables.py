import importlib
from data_handling import atlas, data_architecture, dictionary_operations
from utils import folders_and_files_management
from computing import compute_connectivity_matrices as ccm
from machine_learning import classification
from connectivity_statistics import parametric_tests
from plotting import display
from data_handling import data_management
import os
import pandas as pd

# Reload all module
importlib.reload(data_management)
importlib.reload(atlas)
importlib.reload(display)
importlib.reload(parametric_tests)
importlib.reload(ccm)
importlib.reload(folders_and_files_management)
importlib.reload(classification)
importlib.reload(data_architecture)
importlib.reload(dictionary_operations)

"""
Build results table from NAIS statistical analysis
Author: Dhaif BEKHA (dhaif.bekha@cea.fr) 

"""

# Results directory
results_directory = '/media/db242421/db242421_data/ConPagnon_data/patients_behavior_ACM/' \
                    'wisc_irp_zscores/regression_analysis/tangent'
summary_results_directory = '/media/db242421/db242421_data/ConPagnon_data/patients_behavior_ACM/' \
                            'wisc_irp_zscores/regression_analysis/tangent/summary'
# models to read
models = ['intra_homotopic']
whole_brain_models = ['mean_connectivity', 'mean_homotopic', 'mean_contralesional',
                      'mean_ipsilesional']

# Variable to read
variable_to_read = ['wisc_irp_zscores', 'Sexe[T.M]', 'lesion_normalized']

network_name = ['DMN', 'Executive', 'Language', 'MTL', 'Primary_Visual',
                'Salience', 'Secondary_Visual', 'Sensorimotor', 'Visuospatial',
                'Basal_Ganglia', 'Auditory', 'Precuneus']

# Network level model
for variable in variable_to_read:
    for model in models:
        all_network_df = []
        for network in network_name:
            # parameters file
            parameters_file = pd.read_csv(os.path.join(results_directory, network,
                                                       model + '_parameters.csv'))
            # Fetch interesting value
            parameters_file = data_management.shift_index_column(panda_dataframe=parameters_file,
                                                                 columns_to_index=['variables'])

            variable_df = parameters_file.loc[variable]
            all_network_df.append(variable_df)

        # Concatenate dataframe
        all_network_df = data_management.concatenate_dataframes(list_of_dataframes=all_network_df, axis=1)
        # Replace columns name with network name
        all_network_df.columns = network_name
        # Transpose it
        all_network_df = all_network_df.T
        # Round dataframe
        all_network_df = all_network_df.round(decimals=4)
        # Save it
        all_network_df.to_csv(os.path.join(summary_results_directory, model + '_' + variable + '.csv'))

# Whole brain model
for model in whole_brain_models:
    all_variable_df = []
    for variable in variable_to_read:
        # parameters file
        parameters_file = pd.read_csv(os.path.join(results_directory,
                                                   model + '_parameters.csv'))
        # Fetch interesting value
        parameters_file = data_management.shift_index_column(panda_dataframe=parameters_file,
                                                             columns_to_index=['variables'])

        variable_df = parameters_file.loc[variable]
        # Round dataframe
        variable_df = variable_df.round(decimals=4)
        all_variable_df.append(variable_df)
    # Concatenate dataframe
    all_variable_df = data_management.concatenate_dataframes(list_of_dataframes=all_variable_df,
                                                             axis=1)
    all_variable_df = all_variable_df.T
    # Save it
    all_variable_df.to_csv(os.path.join(summary_results_directory, model + '.csv'))


# Model joint correction
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
    # flatten nested lists of np.ndarray to np.ndarray
    return np.concatenate(list(_flatten(values)))


def unflatten(flat_values, prototype):
    # unflatten np.ndarray to nested lists with structure of prototype
    result, offset = _unflatten(flat_values, prototype, 0)
    assert(offset == len(flat_values))
    return result


root_analysis_directory = '/media/db242421/db242421_data/ConPagnon_data/patients_behavior_ACM_test/' \
                          'pc1_language_zscores/regression_analysis'
kinds = ['correlation', 'partial correlation', 'tangent']
models = ['intra', 'ipsi_intra', 'contra_intra', 'intra_homotopic']
networks = network_name
correction_methods = ['fdr_bh', 'bonferroni']
alpha = 0.05

import numpy as np
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import itertools
# If networks is None, the models are whole brain models,
# so we jus loop over whole brain model at the root of
# the analysis directory
for kind in kinds:
    if networks is None:
        # loop over the models
        all_models_p_values = []
        for model in models:
            # read current model dataframe
            model_df = pd.read_csv(os.path.join(root_analysis_directory, kind,
                                                model + '_parameters.csv'))
            # Fetch the raw p_values
            all_models_p_values.append(model_df['p_value'])

        # Convert list to array of shape (n_models, n_tests)
        # each row (beta_0, beta_1, ...., beta_N) for the N variables in
        # the model.
        all_models_p_values_array = np.array(all_models_p_values)
        # Flatten the array
        all_models_p_values_array_flat = all_models_p_values_array.flatten()

        # Feed the flat p values array to chosen correction method
        for correction in correction_methods:
            _, corrected_p_values, _, _ = multipletests(pvals=all_models_p_values_array_flat,
                                                        alpha=alpha,
                                                        method=correction)

            # rebuild the array
            corrected_p_values_array = corrected_p_values.reshape(len(models), model_df.shape[0])

            # Append in the corresponding CSV model the corrected p values
            for model in models:
                # read current model dataframe
                model_df = pd.read_csv(os.path.join(root_analysis_directory, kind,
                                                    model + '_parameters.csv'))
                # Append the corrected p value column
                model_df[correction + "corrected_pvalues"] = corrected_p_values_array[models.index(model), ...]
                # Overwrite the dataframe
                model_df.to_csv(os.path.join(root_analysis_directory, kind,
                                             model + '_parameters_test.csv'),
                                index=False)

    else:
        network_models_dictionary = dict.fromkeys(models)
        all_models_p_values = []
        for model in models:
            network_model_p_values = []
            for network in network_name:
                model_network_df_path = Path(os.path.join(root_analysis_directory, kind, network,
                                             model + '_parameters.csv'))
                if model_network_df_path.exists():

                    model_network_df = pd.read_csv(os.path.join(root_analysis_directory, kind,
                                                                network, model + '_parameters.csv'))
                    network_model_p_values.append(model_network_df['p_value'])

                else:
                    pass
            # Convert to array
            network_model_p_values_array = np.array(network_model_p_values)
            # Fill the dictionary of the model
            network_models_dictionary[model] = {'p values array': network_model_p_values_array,
                                                'p values flat array': network_model_p_values_array.flatten(),
                                                'number of networks': network_model_p_values_array.shape[0],
                                                'number of variables': network_model_p_values_array.shape[1]}

            # Append the network array of the model
            all_models_p_values.append(network_model_p_values_array)

        # convert it to array
        all_models_p_values_array = np.array(all_models_p_values)
        list_of_p_values = []

        for i in range(len(models)):
            list_of_p_values.append(all_models_p_values_array[i])
        # Flatten the nested list of p values
        all_models_p_values_list_flat = flatten(values=list_of_p_values)

        for correction in correction_methods:
            _, corrected_p_values, _, _ = multipletests(pvals=all_models_p_values_list_flat,
                                                        alpha=alpha,
                                                        method=correction)
            # Rebuild the nested list of p values
            corrected_p_values_unflatten = unflatten(corrected_p_values, list_of_p_values)

            # Write the corrected p values in corresponding CSV
            for model in models:
                for network in network_name:
                    model_network_df_path = Path(os.path.join(root_analysis_directory, kind, network,
                                                              model + '_parameters.csv'))
                    if model_network_df_path.exists():
                        model_network_df = pd.read_csv(os.path.join(root_analysis_directory, kind,
                                                                    network, model + '_parameters.csv'))
                        # Append the corrected p value column
                        model_network_df[correction + "corrected_pvalues"] = \
                            corrected_p_values_unflatten[models.index(model)][network_name.index(network), ...]

                        # Overwrite the dataframe
                        model_network_df.to_csv(os.path.join(root_analysis_directory, kind, network,
                                                     model + '_parameters_test.csv'),
                                        index=False)

                    else:
                        pass





a = list_of_p_values
b = flatten(a)


c = unflatten(b, a)
