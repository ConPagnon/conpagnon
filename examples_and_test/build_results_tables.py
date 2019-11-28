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
root_analysis_directory = '/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/AVCnn_Dhaif/' \
                          'ConPagnon_data/patients_behavior_ACM/pc1_language_zscores_all_151119/regression_analysis'
results_directory = os.path.join(root_analysis_directory, 'tangent')
summary_results_directory = os.path.join(results_directory, 'summary')
# models to read
network_models = ['intra', 'intra_homotopic', 'contra_intra', 'ipsi_intra']
whole_brain_models = ['mean_connectivity', 'mean_homotopic', 'mean_contralesional',
                      'mean_ipsilesional']

# Variable to read
variable_to_read = ['mean_connectivity', 'lesion_normalized']

network_name = ['DMN', 'Sensorimotor', 'Language',
               'Executive', 'Visuospatial', 'Primary_Visual',
           'Secondary_Visual', 'MTL', 'Salience']
#network_name = ['Auditory', 'Precuneus', 'Basal_Ganglia']

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

for network_model in network_models:
    regression_analysis_model.joint_models_correction(root_analysis_directory=root_analysis_directory,
                                                      kinds=kinds,
                                                      models=[network_model],
                                                      correction_methods=correction_methods,
                                                      networks=['DMN', 'Executive', 'Language', 'MTL',
                                                                'Primary_Visual', 'Salience', 'Secondary_Visual',
                                                                'Sensorimotor', 'Visuospatial', 'Basal_Ganglia',
                                                                'Precuneus', 'Auditory'])

# Network level model
network_models = ['contra_intra']

network_dict = dict.fromkeys(network_name)

for network in network_name:
    variable_to_read = ["intra_" + network + "_connectivity", "lesion_normalized"]
    network_dict[network] = dict.fromkeys(network_models)
    for model in network_models:
        network_dict[network][model] = dict.fromkeys(variable_to_read)
        for variable in variable_to_read:
            # parameters file
            parameters_file = pd.read_csv(os.path.join(results_directory, network,
                                                       model + '_parameters.csv'))
            # quality fit parameters file
            quality_fit_file = pd.read_csv(os.path.join(results_directory, network, model + '_qualitity_fit.csv'))
            # Fetch interesting value
            parameters_file = data_management.shift_index_column(panda_dataframe=parameters_file,
                                                                 columns_to_index=['variables'])

            # We are just interested in t, p value (corrected)
            variable_df = parameters_file.loc[variable][['t', 'fdr_bhcorrected_pvalues']]
            # Fetch the adj R2
            r2 = quality_fit_file['adj_r_squared']
            # concatenate df
            variable_df = pd.concat(objs=[variable_df, r2])
            # Round dataframe
            variable_df = pd.DataFrame(variable_df.round(decimals=4), columns=[model])
            variable_df = variable_df.rename({0: "Adj R2", "t": "T", "fdr_bhcorrected_pvalues": "p FDR"})
            # transpose the dataframe for convenience
            variable_df = variable_df.T
            network_dict[network][model][variable] = variable_df

# Read the network dict for each model, for each variable
networks_variable_results =[]
for network in network_name:
    variable_to_read = ["intra_" + network + "_connectivity"]
    #variable_to_read = ["lesion_normalized"]
    for model in network_models:
        for variable in variable_to_read:
            networks_variable_results.append(network_dict[network][model][variable])
# Concatenate and save the dataframe
networks_variable_results_df = pd.concat(networks_variable_results)
networks_variable_results_df.to_excel(os.path.join(summary_results_directory,
                                                   "networks_model.xlsx"))

# Whole brain model
model_df = {"mean_connectivity": ["mean_connectivity", "lesion_normalized"],
            "mean_homotopic": ["mean_homotopic", "lesion_normalized"],
            "mean_ipsilesional": ["mean_ipsi", "lesion_normalized"],
            "mean_contralesional": ["mean_contra", "lesion_normalized"]}
for model in whole_brain_models:
    all_variable_df = []
    for variable in model_df[model]:
        # parameters file
        parameters_file = pd.read_csv(os.path.join(results_directory,
                                                   model + '_parameters.csv'))
        # quality fit parameters file
        quality_fit_file = pd.read_csv(os.path.join(results_directory, model + '_qualitity_fit.csv'))
        # Fetch interesting value
        parameters_file = data_management.shift_index_column(panda_dataframe=parameters_file,
                                                             columns_to_index=['variables'])

        # We are just interested in t, p value (corrected)
        variable_df = parameters_file.loc[variable][['t', 'fdr_bhcorrected_pvalues']]
        # Fetch the adj R2
        r2 = quality_fit_file['adj_r_squared']
        # concatenate df
        variable_df = pd.concat(objs=[variable_df, r2])
        # Round dataframe
        variable_df = pd.DataFrame(variable_df.round(decimals=4), columns=[model])
        variable_df = variable_df.rename({0: "Adj R2", "t": "T", "fdr_bhcorrected_pvalues": "p FDR"})
        # transpose the dataframe for convenience
        variable_df = variable_df.T
        # save the dataframe with the name of the model
        variable_df.to_excel(os.path.join(summary_results_directory, model + "_" + variable +
                                          "_summary.xlsx"), index=True)

