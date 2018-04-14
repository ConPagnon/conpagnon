"""
Modules to perform connectivity analysis at te network
level

Author: Dhaif BEKHA.

"""
from data_handling import data_management
from utils import folders_and_files_management
import os
from connectivity_statistics import parametric_tests
from pylearn_mulm import mulm
from patsy import dmatrix
import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests


def regression_analysis_network_level(groups, kinds, networks_list, root_analysis_directory,
                                      network_model, variables_in_model, behavioral_dataframe,
                                      correction_method, alpha):
    """Regress a linear model at the network level
    """
    # The design matrix is the same for all model
    design_matrix = dmatrix('+'.join(variables_in_model), behavioral_dataframe, return_type='dataframe')
    for model in network_model:
        for kind in kinds:
            all_network_connectivity = []
            for network in networks_list:
                data_directory = os.path.join(root_analysis_directory,
                                              kind)
                # Concatenate all the intra-network dataframe
                network_model_dataframe = data_management.concatenate_dataframes([data_management.read_csv(
                    csv_file=os.path.join(data_directory, network, group + '_' + model + '_' +
                                          network + '_' + 'connectivity.csv')) for group in groups])
                # Shift index to be the subjects identifiers
                network_model_dataframe = data_management.shift_index_column(
                    panda_dataframe=network_model_dataframe,
                    columns_to_index=['subjects'])
                # Add variables in the model to complete the overall DataFrame
                network_model_dataframe = data_management.merge_by_index(
                    dataframe1=network_model_dataframe,
                    dataframe2=behavioral_dataframe[variables_in_model])
                # Build response variable vector, build matrix design
                network_response, network_design = parametric_tests.design_matrix_builder(
                    dataframe=network_model_dataframe,
                    formula=network_model_dataframe.columns[0] + '~' + '+'.join(variables_in_model)
                )
                # Fit the model
                network_model_fit = parametric_tests.ols_regression(y=network_response,
                                                                    X=network_design)

                # Creation of a directory for the current analysis for the current network
                regression_output_directory = folders_and_files_management.create_directory(
                    directory=os.path.join(root_analysis_directory,
                                           'regression_analysis', kind, network))

                # Write output regression results in csv files
                data_management.write_ols_results(ols_fit=network_model_fit, design_matrix=network_design,
                                                  response_variable=network_response,
                                                  output_dir=regression_output_directory,
                                                  model_name=model,
                                                  design_matrix_index_name='subjects')

                all_network_connectivity.append(network_response)

            # Multiple comparison correction

            # merge by index the dataframe from all the network for the current model
            all_network_response = data_management.merge_list_dataframes(all_network_connectivity)
            # Re-index the response variable dataframe to match the index of design matrix
            all_networks_connectivity = all_network_response.reindex(design_matrix.index)
            # Fit a linear model and correcting for maximum statistic
            mulm_fit = mulm.MUOLS(Y=np.array(all_networks_connectivity), X=np.array(design_matrix)).fit()
            # t-test for each variable in the model
            contrasts = np.identity(np.array(design_matrix).shape[1])
            raw_t, raw_p, df = mulm_fit.t_test(contrasts=contrasts, two_tailed=True, pval=True)
            if correction_method == 'maxT':

                _, p_values_maximum_T, _, null_distribution_max_T = mulm_fit.t_test_maxT(contrasts=contrasts,
                                                                                         two_tailed=True,
                                                                                         nperms=10000)
                corrected_p_values = p_values_maximum_T
                # save the null distribution
                folders_and_files_management.save_object(object_to_save=null_distribution_max_T,
                                                         saving_directory=os.path.join(root_analysis_directory,
                                                                                       'regression_analysis', kind),
                                                         filename=model+'_network_maximum_statistic_null.pkl')
            elif correction_method == 'FDR':
                raw_p_shape = raw_p.shape
                fdr_corrected_p_values = multipletests(pvals=raw_p.flatten(),
                                                       method='fdr_bh', alpha=alpha)[1].reshape(raw_p_shape)
                corrected_p_values = fdr_corrected_p_values
            else:
                # return raw p
                corrected_p_values = raw_p


            # Append in each model for all networkCSV file, the corrected p-values for the chosen
            # correction method
            for network in networks_list:
                model_network_csv_file = os.path.join(root_analysis_directory, 'regression_analysis', kind,
                                                      network, model + '_parameters.csv')

                # Read the csv file
                model_network_parameters = data_management.read_csv(model_network_csv_file)
                # Add a last column for adjusted p-values
                model_network_parameters[correction_method + 'corrected_pvalues'] = \
                    corrected_p_values[:, networks_list.index(network)]
                # Write it back to csf format
                data_management.dataframe_to_csv(dataframe=model_network_parameters,
                                                 path=model_network_csv_file)


def regression_analysis_whole_brain(groups, kinds, root_analysis_directory,
                                    whole_brain_model, variables_in_model, behavioral_dataframe,
                                    correction_method, alpha):
    """Regression analysis on composite connectivity score over the whole brain.

    """

    # The design matrix is the same for all model
    design_matrix = dmatrix('Groupe + Sexe', behavioral_dataframe, return_type='dataframe')
    # For each model: read the csv for each group, concatenate resultings dataframe, and append
    # (merging by index) all variable of interest in the model.
    for kind in kinds:
        # directory where the data are
        data_directory = os.path.join(root_analysis_directory,
                                      kind)
        all_model_response = []
        for model in whole_brain_model:
            # List of the corresponding dataframes
            model_dataframe = data_management.concatenate_dataframes([data_management.read_csv(
                csv_file=os.path.join(data_directory, group + '_' + kind + '_' + model + '.csv'))
                for group in groups])
            # Shift index to be the subjects identifiers
            model_dataframe = data_management.shift_index_column(panda_dataframe=model_dataframe,
                                                                 columns_to_index=['subjects'])
            # Add variables in the model to complete the overall DataFrame
            model_dataframe = data_management.merge_by_index(dataframe1=model_dataframe,
                                                             dataframe2=behavioral_dataframe[variables_in_model])
            # Build the model formula: the variable to explain is the first column of the
            # dataframe, and we add to the left all variable in the model
            model_formulation = model_dataframe.columns[0] + '~' + '+'.join(variables_in_model)
            # Build response, and design matrix from the model model formulation
            model_response, model_design = parametric_tests.design_matrix_builder(dataframe=model_dataframe,
                                                                                  formula=model_formulation,
                                                                                  return_type='dataframe')
            # regression with a simple OLS model
            model_fit = parametric_tests.ols_regression(y=model_response, X=model_design)

            # Creation of a directory for the current analysis
            regression_output_directory = folders_and_files_management.create_directory(
                directory=os.path.join(root_analysis_directory, 'regression_analysis', kind))

            # Write output regression results in csv files
            data_management.write_ols_results(ols_fit=model_fit, design_matrix=model_design,
                                              response_variable=model_response,
                                              output_dir=regression_output_directory,
                                              model_name=model,
                                              design_matrix_index_name='subjects')
            # Appending current model response
            all_model_response.append(model_response)

        # merge by index the dataframe
        df_tmp = data_management.merge_list_dataframes(all_model_response)
        # Re-index the response variable dataframe to match the index of design matrix
        model_whole_brain_connectivity = df_tmp.reindex(design_matrix.index)

        # Fit a linear model and correcting for maximum statistic
        mulm_fit = mulm.MUOLS(Y=np.array(model_whole_brain_connectivity),
                              X=np.array(design_matrix)).fit()
        contrasts = np.identity(np.array(design_matrix).shape[1])
        raw_t, raw_p, df = mulm_fit.t_test(contrasts=contrasts, two_tailed=True, pval=True)
        if correction_method == 'maxT':

            _, p_values_maximum_T, _, null_distribution_max_T = mulm_fit.t_test_maxT(contrasts=contrasts,
                                                                                     two_tailed=True,
                                                                                     nperms=10000)
            corrected_p_values = p_values_maximum_T
            # save the null distribution
            folders_and_files_management.save_object(object_to_save=null_distribution_max_T,
                                                     saving_directory=os.path.join(root_analysis_directory,
                                                                                   'regression_analysis', kind),
                                                     filename='whole_brain_connectivity_maximum_statistic_null.pkl')
        elif correction_method == 'FDR':
            raw_p_shape = raw_p.shape
            fdr_corrected_p_values = multipletests(pvals=raw_p.flatten(),
                                                   method='fdr_bh', alpha=alpha)[1].reshape(raw_p_shape)
            corrected_p_values = fdr_corrected_p_values
        else:
            corrected_p_values = raw_p

        # Append in each model CSV file, the corrected p-values for maximum statistic
        for model in whole_brain_model:
            model_csv_file = os.path.join(root_analysis_directory, 'regression_analysis', kind,
                                          model + '_parameters.csv')
            # Read the csv file
            model_parameters = data_management.read_csv(model_csv_file)
            # Add a last column for adjusted p-values
            model_parameters[correction_method + 'corrected_pvalues'] = \
                corrected_p_values[:, whole_brain_model.index(model)]
            # Write it back to csf format
            data_management.dataframe_to_csv(dataframe=model_parameters, path=model_csv_file)
