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
                'Salience', 'Secondary_Visual', 'Sensorimotor', 'Visuospatial']

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
root_analysis_directory = '/media/db242421/db242421_data/ConPagnon_data/patients_behavior_ACM_test/' \
                          'pc1_language_zscores/regression_analysis'
kinds = ['correlation', 'partial correlation', 'tangent']
models = ['intra', 'ipsi_intra', 'contra_intra']
networks = network_name
correction_methods = ['fdr_bh', 'bonferroni']
alpha = 0.05

import numpy as np
from statsmodels.stats.multitest import multipletests

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
        # Stack over all network and models
        all_network_models_p_values = []
        for model in models:
            for network in network_name:
                model_network_df = pd.read_csv(os.path.join(root_analysis_directory, kind,
                                                            network, model + '_parameters.csv'))
                all_network_models_p_values.append(model_network_df['p_value'])

        # Convert list of p values to array
        all_network_models_p_values_array = np.array(all_network_models_p_values)

