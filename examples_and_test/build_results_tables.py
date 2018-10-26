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
from connectivity_statistics import regression_analysis_model
from pathlib import Path

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
importlib.reload(regression_analysis_model)

"""
Build results table from NAIS statistical analysis
Author: Dhaif BEKHA (dhaif.bekha@cea.fr) 

"""

# Results directory
root_analysis_directory = '/media/db242421/db242421_data/ConPagnon_data/patients_behavior_ACM/' \
                          'pc1_language_zscores/regression_analysis'
results_directory = '/media/db242421/db242421_data/ConPagnon_data/patients_behavior_ACM/' \
                    'pc1_language_zscores/regression_analysis/tangent'
summary_results_directory = '/media/db242421/db242421_data/ConPagnon_data/patients_behavior_ACM/pc1_language_zscores/' \
                            'regression_analysis/tangent/summary'
# models to read
network_models = ['intra_homotopic']
whole_brain_models = ['mean_connectivity', 'mean_homotopic', 'mean_contralesional',
                      'mean_ipsilesional']

# Variable to read
variable_to_read = ['pc1_language_zscores', 'Sexe[T.M]', 'lesion_normalized']

network_name = ['DMN', 'Executive', 'Language', 'MTL', 'Primary_Visual',
                'Salience', 'Secondary_Visual', 'Sensorimotor', 'Visuospatial',
                'Basal_Ganglia', 'Precuneus', 'Auditory']

kinds = ['correlation', 'partial correlation', 'tangent']
correction_methods = ['fdr_bh', 'bonferroni']
alpha = 0.05

# joint models correction
regression_analysis_model.joint_models_correction(root_analysis_directory=root_analysis_directory,
                                                  kinds=kinds,
                                                  models=['mean_connectivity', 'mean_homotopic',
                                                          'mean_ipsilesional', 'mean_contralesional'],
                                                  correction_methods=correction_methods,
                                                  networks=None)

regression_analysis_model.joint_models_correction(root_analysis_directory=root_analysis_directory,
                                                  kinds=kinds,
                                                  models=['intra', 'intra_homotopic', 'contra_intra',
                                                          'ipsi_intra'],
                                                  correction_methods=correction_methods,
                                                  networks=['DMN', 'Executive', 'Language', 'MTL',
                                                            'Primary_Visual', 'Salience', 'Secondary_Visual',
                                                            'Sensorimotor', 'Visuospatial', 'Basal_Ganglia',
                                                            'Precuneus', 'Auditory'])

# Network level model
for variable in variable_to_read:
    for model in network_models:
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
