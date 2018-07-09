"""
Modules to perform connectivity analysis at te network
level

Author: Dhaif BEKHA.

"""
from data_handling import data_management, dictionary_operations
from utils import folders_and_files_management
import os
from connectivity_statistics import parametric_tests
from pylearn_mulm import mulm
from patsy import dmatrix
import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from matplotlib.backends.backend_pdf import PdfPages
from plotting import display
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def regression_analysis_network_level(groups, kinds, networks_list, root_analysis_directory,
                                      network_model, variables_in_model, behavioral_dataframe,
                                      correction_method=['FDR'], alpha=0.05,
                                      two_tailed=True, n_permutations=10000):
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

                # Take only the index subjects present in the analysis, because design matrix is for the whole
                # cohort !!!
                design_matrix = design_matrix.loc[network_design.index]

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
            for corr_method in correction_method:
                if corr_method == 'maxT':

                    _, p_values_maximum_T, _, null_distribution_max_T = \
                        mulm_fit.t_test_maxT(contrasts=contrasts, two_tailed=two_tailed,
                                             nperms=n_permutations)
                    corrected_p_values = p_values_maximum_T
                    # save the null distribution
                    folders_and_files_management.save_object(object_to_save=null_distribution_max_T,
                                                             saving_directory=os.path.join(root_analysis_directory,
                                                                                           'regression_analysis', kind),
                                                             filename=model+'_network_maximum_statistic_null.pkl')
                elif corr_method == 'FDR':
                    raw_p_shape = raw_p.shape
                    fdr_corrected_p_values = multipletests(pvals=raw_p.flatten(),
                                                           method='fdr_bh', alpha=alpha)[1].reshape(raw_p_shape)
                    corrected_p_values = fdr_corrected_p_values
                else:
                    # return raw p
                    corrected_p_values = raw_p

                # Append in each model for all networkCSV file,
                # the corrected p-values for the chosen
                # correction method
                for network in networks_list:
                    model_network_csv_file = os.path.join(root_analysis_directory,
                                                          'regression_analysis', kind,
                                                          network, model + '_parameters.csv')

                    # Read the csv file
                    model_network_parameters = data_management.read_csv(model_network_csv_file)
                    # Add a last column for adjusted p-values
                    model_network_parameters[corr_method + 'corrected_pvalues'] = \
                        corrected_p_values[:, networks_list.index(network)]
                    # Write it back to csf format
                    data_management.dataframe_to_csv(dataframe=model_network_parameters,
                                                     path=model_network_csv_file)


def regression_analysis_whole_brain(groups, kinds, root_analysis_directory,
                                    whole_brain_model, variables_in_model,
                                    behavioral_dataframe,
                                    correction_method=['FDR'], alpha=0.05,
                                    n_permutations=10000,
                                    two_tailed=True):
    """Regression analysis on composite connectivity score over the whole brain.

    """

    # The design matrix is the same for all model
    design_matrix = dmatrix('+'.join(variables_in_model), behavioral_dataframe, return_type='dataframe')
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
            model_response, model_design = \
                parametric_tests.design_matrix_builder(dataframe=model_dataframe,
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

            # Take only the index subjects present in the analysis, because design matrix is for the whole
            # cohort !!!
            design_matrix = design_matrix.loc[model_design.index]

        # merge by index the dataframe
        df_tmp = data_management.merge_list_dataframes(all_model_response)
        # Re-index the response variable dataframe to match the index of design matrix
        model_whole_brain_connectivity = df_tmp.reindex(design_matrix.index)

        # Fit a linear model and correcting for maximum statistic
        mulm_fit = mulm.MUOLS(Y=np.array(model_whole_brain_connectivity),
                              X=np.array(design_matrix)).fit()
        contrasts = np.identity(np.array(design_matrix).shape[1])
        raw_t, raw_p, df = mulm_fit.t_test(contrasts=contrasts, two_tailed=True, pval=True)
        for corr_method in correction_method:

            if corr_method == 'maxT':

                _, p_values_maximum_T, _, null_distribution_max_T = \
                    mulm_fit.t_test_maxT(contrasts=contrasts, two_tailed=two_tailed,
                                         nperms=n_permutations)
                corrected_p_values = p_values_maximum_T
                # save the null distribution
                folders_and_files_management.save_object(
                    object_to_save=null_distribution_max_T,
                    saving_directory=os.path.join(root_analysis_directory,
                                                  'regression_analysis', kind),
                                                  filename='whole_brain_connectivity_maximum_statistic_null.pkl')
            elif corr_method == 'FDR':
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
                model_parameters[corr_method + 'corrected_pvalues'] = \
                    corrected_p_values[:, whole_brain_model.index(model)]
                # Write it back to csf format
                data_management.dataframe_to_csv(dataframe=model_parameters, path=model_csv_file)
                
                
def regression_analysis_internetwork_level(internetwork_subjects_connectivity_dictionary, groups_in_model, behavioral_data_path,
                                           sheet_name, subjects_to_drop, model_formula, kinds_to_model,  root_analysis_directory,
                                           inter_network_model, network_labels_list, network_labels_colors,
                                           pvals_correction_method=['maxT', 'FDR'], vectorize=True,
                                           discard_diagonal=False, nperms_maxT = 10000, contrasts = 'Id',
                                           compute_pvalues = 'True', pvalues_tail = 'True', NA_action='drop',
                                           alpha=0.05
                                           ):
    
    # Merge the dictionary into one, to give it as data argument to
    # linear regression function
    group_dictionary_list = [internetwork_subjects_connectivity_dictionary[group] for group in groups_in_model]
    connectivity_data = dictionary_operations.merge_dictionary(group_dictionary_list,
                                                               'all_subjects')['all_subjects']
    
    # We iterate for each kind
    for kind in kinds_to_model:
        # Create and save the raw data dictionary 
        raw_connectivity_data_output = data_management.create_directory(
                directory=os.path.join(root_analysis_directory, kind, inter_network_model))
        
        folders_and_files_management.save_object(object_to_save=connectivity_data,
                                                 saving_directory=raw_connectivity_data_output, 
                                                 filename=inter_network_model + 'data.pkl'
                                                 )
        # Create the output directory for the current kind
        inter_network_analysis_output = data_management.create_directory(
                directory=os.path.join(root_analysis_directory, 'regression_analysis',
                                       kind, 
                                       inter_network_model))
        
        regression_results, X_df, y, y_prediction, regression_subjects_list = parametric_tests.linear_regression(
            connectivity_data=connectivity_data,
            data=behavioral_data_path,
            sheetname=sheet_name,
            subjects_to_drop=subjects_to_drop,
            pvals_correction_method=['FDR', 'maxT'],
            save_regression_directory=inter_network_analysis_output,
            kind=kind,
            NA_action=NA_action,
            formula=model_formula,
            vectorize=vectorize,
            discard_diagonal=discard_diagonal,
            nperms_maxT=nperms_maxT,
            contrasts=contrasts,
            compute_pvalues=compute_pvalues,
            pvalues_tail=pvalues_tail)
        # Save the design matrix 
        data_management.dataframe_to_csv(dataframe=X_df, 
                                         path=os.path.join(inter_network_analysis_output, 'design_matrix.csv'),
                                         index=True)
        
        # Create the figure for each correction method: matrix plot
        # of significant p-values, and t-values
        
        # For each variable plot and save a matrix, with T statistic masked
        # for significant values, and significant p values for each variables
        corr_method_output_pdf = os.path.join(inter_network_analysis_output,
                                              kind + '_' + inter_network_model + '.pdf')
        
        with PdfPages(corr_method_output_pdf) as pdf:
            
            for corr_method in pvals_correction_method:
                # Fetch the results of the model 
                corr_method_results = regression_results[corr_method]['results']
                # iterate over all the keys of dictionary results, i.e
                # all variable
                for variable in corr_method_results.keys():  
                    # Significant t statistic
                    display.plot_matrix(matrix=corr_method_results[variable]['significant tvalues'],
                                        labels_colors=network_labels_colors, mpart='lower', k=-1, 
                                        colormap='RdBu_r',
                                        colorbar_params={'shrink': .5}, center=0,
                                        vmin=None, vmax=None, labels_size=8, 
                                        horizontal_labels=network_labels_list,
                                        vertical_labels=network_labels_list,
                                        linewidths=.5, linecolor='black', 
                                        title=corr_method + '_' + variable + '_effect',
                                        figure_size=(12, 9))
                    pdf.savefig()
                    # Significant p values
                    display.plot_matrix(matrix=corr_method_results[variable]['corrected pvalues'],
                        labels_colors=network_labels_colors, mpart='lower', k=-1, 
                        colormap='hot',
                        colorbar_params={'shrink': .5}, center=0,
                        vmin=0, vmax=alpha, labels_size=8, 
                        horizontal_labels=network_labels_list,
                        vertical_labels=network_labels_list,
                        linewidths=.5, linecolor='black', 
                        title=corr_method + '_' + variable + '_p values',
                        figure_size=(12, 9))
                    pdf.savefig()


def write_raw_data(output_csv_directory, kinds, groups, models, variables_in_model,
                   behavioral_dataframe):
    """Write the response variable along with the co-variables in a unique
    dataframe inside a csv files.

    Parameters
    ----------
    output_csv_directory: str
        The directory path containing the data for each connectivity
        metric. It should contain one folder for each metric in you're analysis.
    kinds: list
        The list of connectivity metric needed in the statistical analysis.
    groups: list
        The list of subjects group names needed in the statistical analysis.
    models: list
        The list of models needed in the analysis. The model name should be
        contain in the group csv data.
    variables_in_model: list
        The list of co-variables to add to the response variable.
    behavioral_dataframe: pandas.DataFrame
        A pandas dataframe containing the co-variables to add. Note
        that the dataframe must have a columns called 'subjects' with the
        identifier of each subjects. The response variable dataframe and
        the behavioral dataframe will be merge considering the subject
        columns as index.

    """
    # Read and concatenate list of the data for each group
    for model in models:
        for kind in kinds:
            # Read and concatenate the data for each group to build the response variable
            response_variable_data = data_management.concatenate_dataframes([data_management.read_csv(
                csv_file=os.path.join(output_csv_directory, kind, group + '_' + kind + '_' + model +
                                      '.csv')) for group in groups])
            # Make the 'subjects' column the index of the response variable dataframe
            response_variable_data = data_management.shift_index_column(
                panda_dataframe=response_variable_data,
                columns_to_index=['subjects'])
            # Merge the response variable dataframe and the behavioral
            # dataframe to build the overall dataframe
            model_dataframe = data_management.merge_by_index(
                dataframe1=response_variable_data,
                dataframe2=behavioral_dataframe[variables_in_model])

            model_dataframe.to_csv(os.path.join(output_csv_directory, kind, model + '_raw_data.csv'))


def write_raw_data_network(output_csv_directory, kinds, groups, models, variables_in_model,
                           behavioral_dataframe, network_list):
    """Write the response variable along with the co-variables in a unique
    dataframe inside a csv files, at the network level.

    Parameters
    ----------
    output_csv_directory: str
        The directory path containing the data for each connectivity
        metric. It should contain one folder for each metric in you're analysis.
    kinds: list
        The list of connectivity metric needed in the statistical analysis.
    groups: list
        The list of subjects group names needed in the statistical analysis.
    models: list
        The list of models needed in the analysis. The model name should be
        contain in the group csv data.
    variables_in_model: list
        The list of co-variables to add to the response variable.
    behavioral_dataframe: pandas.DataFrame
        A pandas dataframe containing the co-variables to add. Note
        that the dataframe must have a columns called 'subjects' with the
        identifier of each subjects. The response variable dataframe and
        the behavioral dataframe will be merge considering the subject
        columns as index.
    network_list: list
        The list of network names. Each network  in each connectivity metric
        should have a directory containing the data for each group.
    """
    for network in network_list:
        # Read and concatenate list of the data for each group
        for model in models:
            for kind in kinds:
                # Read and concatenate the data for each group to build the response variable
                response_variable_data = data_management.concatenate_dataframes([data_management.read_csv(
                    csv_file=os.path.join(output_csv_directory, kind, network, group + '_' + model + '_' + network +
                                          '_connectivity.csv')) for group in groups])
                # Make the 'subjects' column the index of the response variable dataframe
                response_variable_data = data_management.shift_index_column(
                    panda_dataframe=response_variable_data,
                    columns_to_index=['subjects'])
                # Merge the response variable dataframe and the behavioral
                # dataframe to build the overall dataframe
                model_dataframe = data_management.merge_by_index(
                    dataframe1=response_variable_data,
                    dataframe2=behavioral_dataframe[variables_in_model])

                model_dataframe.to_csv(os.path.join(output_csv_directory, kind, network, model + '_raw_data.csv'))


def anova_with_post_hoc_whole_brain(groups, kinds, root_analysis_directory,
                                    whole_brain_model, variables_in_model,
                                    behavioral_dataframe,
                                    correction_method=['FDR'], alpha=0.05):
    """Perform a ANOVA analysis, i.e comparing the mean of the continuous response
    variable with respect to one categorical independent variable (with different
    level). ANOVA follow with post hoc analysis (t-test between all possible groups pairs).

    :param groups:
    :param kinds:
    :param root_analysis_directory:
    :param whole_brain_model:
    :param variables_in_model:
    :param behavioral_dataframe:
    :param correction_method:
    :param alpha:
    :return:
    """

    # The design matrix is the same for all model
    design_matrix = dmatrix('+'.join(variables_in_model), behavioral_dataframe, return_type='dataframe')
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
            model_response, model_design = \
                parametric_tests.design_matrix_builder(dataframe=model_dataframe,
                                                       formula=model_formulation,
                                                       return_type='dataframe')
            # regression with a simple OLS model
            model_fit = parametric_tests.ols_regression(y=model_response, X=model_design)

            # Creation of a directory for the current analysis, ANOVA results
            # will be stored in a sub-directory called ANOVA
            regression_output_directory = folders_and_files_management.create_directory(
                directory=os.path.join(root_analysis_directory, 'regression_analysis/ANOVA', kind))
            # Perform one-way ANOVA with statsmodel
            model_anova = sm.stats.anova_lm(model_fit, typ=2)
            # Compute t-test between all pairs of groups in the
            # analysis
            tukey_analysis = pairwise_tukeyhsd(model_dataframe[model],
                                               model_dataframe[variables_in_model])
