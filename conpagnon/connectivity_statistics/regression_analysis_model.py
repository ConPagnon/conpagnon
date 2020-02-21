"""
Modules to perform connectivity analysis at the network
level

Author: Dhaif BEKHA.

"""
from conpagnon.data_handling import dictionary_operations, data_management
from conpagnon.utils import folders_and_files_management
import os
from conpagnon.connectivity_statistics import parametric_tests
from conpagnon.pylearn_mulm import mulm
from patsy import dmatrix
import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from matplotlib.backends.backend_pdf import PdfPages
from conpagnon.plotting import display
import statsmodels.api as sm
import pandas as pd
import statsmodels.stats.multicomp as multi
from scipy import stats
from pathlib import Path


def regression_analysis_network_level(groups, kinds, networks_list, root_analysis_directory,
                                      network_model, variables_in_model, behavioral_dataframe,
                                      correction_method=['fdr_bh'], alpha=0.05,
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
                elif corr_method in ['fdr_bh', 'bonferroni']:
                    raw_p_shape = raw_p.shape
                    fdr_corrected_p_values = multipletests(pvals=raw_p.flatten(),
                                                           method=corr_method, alpha=alpha)[1].reshape(raw_p_shape)
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


def regression_analysis_network_level_v2(groups, kinds, networks_list, root_analysis_directory,
                                         network_model, variables_in_model, score_of_interest,
                                         behavioral_dataframe,
                                         correction_method=['fdr_bh'],
                                         alpha=0.05):

    """Perform linear regression between a behavioral score and a functional connectivity "score".
    The connectivity score is usually a simple mean between the connectivity coefficient
    of a single network. ConPagnon has some function to compute those kind of score from
    a atlas, where at least a few network are identified from the user.

    Parameters
    ----------
    groups: list
        The list of string of the name of the
        group involved in the analysis. Usually,
        it's simply the entries of the subjects
        connectivity dictionary.
    kinds: list
        Repeat the statistical analysis for the
        all the connectivity metrics present
        in the list.
    networks_list: list
        Repeat the statistical analysis for
        all the network in the list. The network
        name should match the keys of the network
        dictionary containing the connectivity score.
    root_analysis_directory: str
        The full path to the text file containing
        all the text data to read and feed to the
        linear model.
    network_model: list
        A list containing the name of the model,
        matching the prefix of the corresponding
        text data to read.
    variables_in_model: list
        A list of the variables to put
        in the linear model. The variables
        name should match the columns name
        present in the text data.
    score_of_interest: str
        The behavioral score to analyze, it's
        simply the Y variable in the classical
        linear model: Y = X*Beta
    behavioral_dataframe: pandas.DataFrame
        The dataframe containing all the behavioral
        variables for each subjects.
    correction_method: list, optional
        A list containing all the desired correction method.
        Be careful, we do not stack all the models before
        the correction. By default, it's the Bonferonni
        method.
    alpha: float, optional
        The type I error rate, set to 0.05 by default.

    See Also
    --------
    compute_connectivity_matrices.intra_network_functional_connectivity :
        This function compute a intra-network connectivity score, by computing
        the mean of all connectivity coefficient inside a single network.
    """

    # The design matrix is the same for all model
    design_matrix = dmatrix('+'.join(variables_in_model), behavioral_dataframe, return_type='dataframe')
    for model in network_model:
        for kind in kinds:
            all_network_connectivity = []
            all_network_p_values = []
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

                connectivity_score_name = network_model_dataframe.columns[0]
                # fetch the score of interest in the behavioral dataframe
                score_dataframe = behavioral_dataframe[[score_of_interest]]
                # fetch the variables in the model, except connectivity score
                model_variables = behavioral_dataframe[variables_in_model]
                # Add variables in the model to complete the overall DataFrame
                network_model_dataframe = data_management.merge_list_dataframes(
                    [network_model_dataframe, score_dataframe,
                     model_variables])
                # Build the model formula: the variable to explain is the first column of the
                # dataframe, and we add to the left all variable in the model
                model_formulation = score_of_interest + '~' + '+'.join(variables_in_model + [connectivity_score_name])
                # Build response variable vector, build matrix design
                network_response, network_design = parametric_tests.design_matrix_builder(
                    dataframe=network_model_dataframe,
                    formula=model_formulation)

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
                all_network_p_values.append(network_model_fit.pvalues)

                # Take only the index subjects present in the analysis, because design matrix is for the whole
                # cohort !!!
                design_matrix = design_matrix.loc[network_design.index]

            # Multiple comparison correction
            # merge by index the dataframe from all the network for the current model
            all_network_response = data_management.merge_list_dataframes(all_network_connectivity)
            # Re-index the response variable dataframe to match the index of design matrix
            all_networks_connectivity = all_network_response.reindex(design_matrix.index)
            # t-test for each variable in the model
            contrasts = np.identity(np.array(design_matrix).shape[1])
            raw_p = np.array(all_network_p_values).T
            for corr_method in correction_method:
                if corr_method in ['fdr_bh', 'bonferroni']:
                    raw_p_shape = raw_p.shape
                    fdr_corrected_p_values = multipletests(pvals=raw_p.flatten(),
                                                           method=corr_method, alpha=alpha)[1].reshape(raw_p_shape)
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
            # List of the corresponding dataframe
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
            elif corr_method in ['fdr_bh', 'bonferroni']:
                raw_p_shape = raw_p.shape
                fdr_corrected_p_values = multipletests(pvals=raw_p.flatten(),
                                                       method=corr_method, alpha=alpha)[1].reshape(raw_p_shape)
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


def regression_analysis_whole_brain_v2(groups, kinds, root_analysis_directory,
                                       whole_brain_model, variables_in_model, score_of_interest,
                                       behavioral_dataframe,
                                       correction_method=['fdr_bh'],
                                       alpha=0.05):
    """Compute a linear model with a continuous score test as the response variable.

    :param groups:
    :param kinds:
    :param root_analysis_directory:
    :param whole_brain_model:
    :param variables_in_model:
    :param score_of_interest:
    :param behavioral_dataframe:
    :param correction_method:
    :param alpha:
    :return:
    """
    for kind in kinds:
        # directory where the data are
        data_directory = os.path.join(root_analysis_directory,
                                      kind)
        all_model_response = []
        all_model_p_values = []
        for model in whole_brain_model:
            # List of the corresponding dataframe
            model_dataframe = data_management.concatenate_dataframes([data_management.read_csv(
                csv_file=os.path.join(data_directory, group + '_' + kind + '_' + model + '.csv'))
                for group in groups])

            # Shift index to be the subjects identifiers
            model_dataframe = data_management.shift_index_column(panda_dataframe=model_dataframe,
                                                                 columns_to_index=['subjects'])
            # Fetch the name of the columns containing the
            # connectivity score. There is only one element
            # since we shifted the index to subject list
            # previously.
            connectivity_score_name = model_dataframe.columns[0]
            # fetch the score of interest in the behavioral dataframe
            score_dataframe = behavioral_dataframe[[score_of_interest]]
            # fecth the variables in the model, except connectivity score
            model_variables = behavioral_dataframe[variables_in_model]
            # Add variables in the model to complete the overall DataFrame
            model_dataframe = data_management.merge_list_dataframes([model_dataframe, score_dataframe, model_variables])
            # Build the model formula: the variable to explain is the first column of the
            # dataframe, and we add to the left all variable in the model
            model_formulation = score_of_interest + '~' + '+'.join(variables_in_model + [connectivity_score_name])
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
            design_matrix = model_design.loc[model_design.index]

            # append p values
            all_model_p_values.append(np.array(model_fit.pvalues))

        raw_p = np.array(all_model_p_values).T
        for corr_method in correction_method:
            if corr_method in ['fdr_bh', 'bonferroni']:
                raw_p_shape = raw_p.shape
                fdr_corrected_p_values = multipletests(pvals=raw_p.flatten(),
                                                       method=corr_method, alpha=alpha)[1].reshape(raw_p_shape)
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


def regression_analysis_internetwork_level(internetwork_subjects_connectivity_dictionary, groups_in_model,
                                           behavioral_data_path,
                                           sheet_name, subjects_to_drop, model_formula, kinds_to_model,
                                           root_analysis_directory,
                                           inter_network_model, network_labels_list, network_labels_colors,
                                           pvals_correction_method=['fdr_bh'], vectorize=True,
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
            pvals_correction_method=pvals_correction_method,
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
                                        colormap='hot', colorbar_params={'shrink': .5}, center=0,
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


def one_way_anova(models, groups, behavioral_dataframe, kinds,
                  correction_method, root_analysis_directory,
                  variables_in_model,
                  alpha=.05):
    """Perform one way ANOVA analysis followed by a post hoc analysis with t test,
    and multiple comparison correction.

    Parameters
    ----------
    models: list
        List of models. The list model is defined as the string in the filename
        containing the raw data for each groups.
    groups: list
        The list of group in the study.
    behavioral_dataframe: pandas.DataFrame
        The dataframe containing the variable to study in the ANOVA analysis.
        The dataframe must contain a column named "subjects" containing the
        identifier for each subjects.
    kinds: list
        The list of the different connectivity metrics you want to
        perform the analysis.
    correction_method: list
        The list of multiple comparison correction method, as available
        in the statsmodels library.
    root_analysis_directory: string
        The full path to the directory containing the raw data.
    variables_in_model: list
        A list containing the categorical variable to study.
    alpha: float, optional
        The type I error threshold. The default is 0.05.

    Notes
    -----
    If models in a list containing more than one elements,
    the p values are jointly corrected for all models.

    """
    # For each model: read the csv for each group, concatenate resulting dataframe, and append
    # (merging by index) all variable of interest in the model.
    for kind in kinds:
        # directory where the data are
        data_directory = os.path.join(root_analysis_directory,
                                      kind)
        # Store the p value of all the tested model for multiple
        # comparison correction
        all_model_p_values_f_test = []
        # Store the p values resulting in all possible contrast
        # t test for multiple comparison correction
        all_model_post_hoc_p_values = []

        # Creation of a directory for the current analysis, ANOVA results
        # will be stored in a sub-directory called ANOVA
        regression_output_directory = folders_and_files_management.create_directory(
            directory=os.path.join(root_analysis_directory, 'regression_analysis/ANOVA', kind))
        for model in models:
            # List of the corresponding dataframe
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
            # regression with a simple OLS model
            model_fit = parametric_tests.ols_regression_formula(formula=model_formulation,
                                                                data=model_dataframe)

            # Perform one-way ANOVA with statsmodel
            model_anova = sm.stats.anova_lm(model_fit, typ=2)
            # Compute t-test between all pairs of groups in the
            # analysis with tukey HSD analysis
            model_tukey_analysis = multi.MultiComparison(
                model_dataframe[model_dataframe.columns[0]],
                model_dataframe[variables_in_model[0]])
            model_tukey_analysis_results = model_tukey_analysis.tukeyhsd()
            # tukey t-score are t score scaled by sqrt(2)
            model_t_scores = \
                model_tukey_analysis_results.meandiffs / model_tukey_analysis_results.std_pairs / np.sqrt(2)
            # Compute two-sided uncorrected p values
            model_raw_p_values = stats.t.sf(np.abs(model_t_scores), 50)*2

            # Stack the p values for the F test of the current model
            all_model_p_values_f_test.append(model_anova['PR(>F)'].loc[variables_in_model[0]])
            # Stack the p values for all contrast for the current model
            all_model_post_hoc_p_values.append(model_raw_p_values)

            # In the report the contrast is group 2 - group 1, build a dataframe to store
            # the results
            post_hoc_dataframe = pd.DataFrame(data=model_tukey_analysis_results._results_table.data[1:],
                                              columns=model_tukey_analysis_results._results_table.data[0])
            # Add t scores for each contrast and raw p values
            post_hoc_dataframe = post_hoc_dataframe.assign(t=model_t_scores, p_values=model_raw_p_values)
            # Save: the results containing the F value and p value for the model
            model_anova.to_csv(os.path.join(regression_output_directory,
                                            model + '_f_test_parameters.csv'),
                               index=False)
            # Save the parameters results of the post hoc analysis
            data_management.dataframe_to_csv(dataframe=post_hoc_dataframe,
                                             path=os.path.join(regression_output_directory,
                                                               model + '_post_hoc_parameters.csv'),
                                             index=False)

        # Correct the p values for all model, and post-hoc t test

        # The user can choose different correction method
        p_values_correction_post_hoc_all_method = np.zeros((len(correction_method), len(models),
                                                            np.array(all_model_post_hoc_p_values).shape[1]))
        p_values_correction_f_test_all_method = np.zeros((len(correction_method), len(models)))
        for correction in correction_method:

            # Correction with the chosen method for the p value of all between group t test
            all_model_p_values_post_hoc_array = np.array(all_model_post_hoc_p_values)
            post_hoc_t_test_corrected_p_values = multipletests(
                pvals=all_model_p_values_post_hoc_array.flatten(),
                method=correction,
                alpha=alpha)[1].reshape(all_model_p_values_post_hoc_array.shape)

            p_values_correction_post_hoc_all_method[correction_method.index(correction), ...] = \
                post_hoc_t_test_corrected_p_values

            # Correction with the chosen method for the p value of the whole model
            all_model_p_values_f_test_array = np.array(all_model_p_values_f_test)
            f_test_corrected_p_values = multipletests(
                pvals=all_model_p_values_f_test_array.flatten(),
                method=correction,
                alpha=alpha)[1].reshape(all_model_p_values_f_test_array.shape)

            p_values_correction_f_test_all_method[correction_method.index(correction), ...] =\
                f_test_corrected_p_values

        # Finally we can now append the corrected p value for all the chosen method in
        # for all model in the csv file
        for model in models:
            # Read the parameters file of the model for the f test first
            f_test_model_parameters = data_management.read_csv(os.path.join(
                regression_output_directory, model + '_f_test_parameters.csv'))
            # Add the corrected p values for all the correction method
            post_hoc_model_parameters = data_management.read_csv(os.path.join(
                regression_output_directory, model + '_post_hoc_parameters.csv'))

            # Append the corrected p values
            for correction in correction_method:
                corrected_p_value_column_name = correction + '_p_value'
                post_hoc_model_parameters[corrected_p_value_column_name] = \
                    p_values_correction_post_hoc_all_method[correction_method.index(correction),
                                                            models.index(model), ...]

                f_test_model_parameters[corrected_p_value_column_name] = \
                    p_values_correction_f_test_all_method[correction_method.index(correction),
                                                          models.index(model)]

                post_hoc_model_parameters.to_csv(os.path.join(regression_output_directory,
                                                              model + '_post_hoc_parameters.csv'),
                                                 index=False)

                f_test_model_parameters.to_csv(os.path.join(regression_output_directory,
                                                            model + '_f_test_parameters.csv'),
                                               index=False)


def one_way_anova_network(root_analysis_directory, kinds, groups,
                          networks_list, models, behavioral_dataframe,
                          variables_in_model, correction_method, alpha=0.05):
    """Perform a one way ANOVA at the network level.

    Parameters
    ----------
    models: list
        List of models. The list model is defined as the string in the filename
        containing the raw data for each groups.
    groups: list
        The list of group in the study.
    networks_list: list
        The list of network to include in the analysis
    behavioral_dataframe: pandas.DataFrame
        The dataframe containing the variable to study in the ANOVA analysis.
        The dataframe must contain a column named "subjects" containing the
        identifier for each subjects.
    kinds: list
        The list of the different connectivity metrics you want to
        perform the analysis.
    correction_method: list
        The list of multiple comparison correction method, as available
        in the statsmodels library.
    root_analysis_directory: string
        The full path to the directory containing the raw data.
    variables_in_model: list
        A list containing the categorical variable to study.
    alpha: float, optional
        The type I error threshold. The default is 0.05.

    Notes
    -----
    If models  and network_list is a list
    containing more than one elements, the p values
    are jointly corrected for the number of models * number of networks.

    """

    for kind in kinds:
        # List to stack the p values from the F test of the model
        all_network_f_test_p_values = []
        # List to stack the p values from all possible contrast from
        # the post hoc t test
        all_network_post_hoc_test_p_values = []
        for network in networks_list:
            for model in models:
                # Creation of a directory for the current analysis, ANOVA results
                # will be stored for each network
                regression_output_directory = folders_and_files_management.create_directory(
                    directory=os.path.join(root_analysis_directory, 'regression_analysis/ANOVA', kind,
                                           network))

                network_data_directory = os.path.join(root_analysis_directory, kind, network)
                # Concatenate all the intra-network dataframe
                network_model_dataframe = data_management.concatenate_dataframes([data_management.read_csv(
                    csv_file=os.path.join(network_data_directory, group + '_' + model + '_' +
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
                # Build the model formula: the variable to explain is the first column of the
                # dataframe, and we add to the left all variable in the model
                model_formulation = network_model_dataframe.columns[0] + '~' + '+'.join(variables_in_model)
                # regression with a simple OLS model
                network_model_fit = parametric_tests.ols_regression_formula(formula=model_formulation,
                                                                            data=network_model_dataframe)

                # Perform one-way ANOVA with statsmodel
                network_model_anova = sm.stats.anova_lm(network_model_fit, typ=2)
                # Compute t-test between all pairs of groups in the
                # analysis with tukey HSD analysis for the current network
                network_model_tukey_analysis = multi.MultiComparison(
                    network_model_dataframe[network_model_dataframe.columns[0]],
                    network_model_dataframe[variables_in_model[0]])
                network_model_tukey_analysis_results = network_model_tukey_analysis.tukeyhsd()
                # tukey t-score are t score scaled by sqrt(2)
                network_model_t_scores = \
                    network_model_tukey_analysis_results.meandiffs / network_model_tukey_analysis_results.std_pairs / np.sqrt(2)
                # Compute two-sided uncorrected p values
                network_model_raw_p_values = stats.t.sf(np.abs(network_model_t_scores), 50) * 2

                # Stack the p values for the F test of the current model
                all_network_f_test_p_values.append(network_model_anova['PR(>F)'].loc[variables_in_model[0]])
                # Stack the p values for all contrast for the current model
                all_network_post_hoc_test_p_values.append(network_model_raw_p_values)

                # In the report the contrast is group 2 - group 1, build a dataframe to store
                # the results
                post_hoc_dataframe = pd.DataFrame(data=network_model_tukey_analysis_results._results_table.data[1:],
                                                  columns=network_model_tukey_analysis_results._results_table.data[0])
                # Add t scores for each contrast and raw p values
                post_hoc_dataframe = post_hoc_dataframe.assign(t=network_model_t_scores,
                                                               p_values=network_model_raw_p_values)
                # Save: the results containing the F value and p value for the model
                network_model_anova.to_csv(os.path.join(regression_output_directory,
                                                        model + '_f_test_parameters.csv'),
                                           index=False)
                # Save the parameters results of the post hoc analysis
                data_management.dataframe_to_csv(dataframe=post_hoc_dataframe,
                                                 path=os.path.join(regression_output_directory,
                                                                   model + '_post_hoc_parameters.csv'),
                                                 index=False)

        # Convert list of post hoc t test p value,
        # and f test p value in array
        # I squeeze it to get rid of the first (1, ...) dimension
        all_models_post_hoc_t_test_array = np.squeeze(np.array(all_network_post_hoc_test_p_values))
        all_models_f_test_p_values_array = np.array(all_network_f_test_p_values)

        # The user can choose different correction method
        p_values_correction_post_hoc_all_method = np.zeros((len(correction_method), len(models)*len(networks_list),
                                                            all_models_post_hoc_t_test_array.shape[1]))
        p_values_correction_f_test_all_method = np.zeros((len(correction_method), len(models)*len(networks_list)))

        # Loop over the correction method wanted by the user
        for correction in correction_method:

            # Correction with the chosen method for the p value of all between group t test
            post_hoc_t_test_corrected_p_values = multipletests(
                pvals=all_models_post_hoc_t_test_array.flatten(),
                method=correction,
                alpha=alpha)[1].reshape(all_models_post_hoc_t_test_array.shape)

            p_values_correction_post_hoc_all_method[correction_method.index(correction), ...] = \
                post_hoc_t_test_corrected_p_values

            # Correction with the chosen method for the p value of the whole model
            f_test_corrected_p_values = multipletests(
                pvals=all_models_f_test_p_values_array.flatten(),
                method=correction,
                alpha=alpha)[1].reshape(all_models_f_test_p_values_array.shape)

            p_values_correction_f_test_all_method[correction_method.index(correction), ...] =\
                f_test_corrected_p_values

        # Reshape the array of corrected p values for easy
        # writing in parameters files
        reshape_p_values_correction_post_hoc_all_method = p_values_correction_post_hoc_all_method.flatten().reshape(
            (len(correction_method), len(networks_list), len(models), all_models_post_hoc_t_test_array.shape[1]))

        reshape_p_values_correction_f_test_all_method = p_values_correction_f_test_all_method.flatten().reshape(
            (len(correction_method), len(networks_list), len(models), 1)
        )

        for model in models:
            for network in networks_list:

                regression_output_directory = folders_and_files_management.create_directory(
                    directory=os.path.join(root_analysis_directory, 'regression_analysis/ANOVA', kind,
                                           network))
                # Read the parameters file of the model for the f test first
                f_test_model_parameters = data_management.read_csv(os.path.join(
                    regression_output_directory, model + '_f_test_parameters.csv'))
                # Add the corrected p values for all the correction method
                post_hoc_model_parameters = data_management.read_csv(os.path.join(
                    regression_output_directory, model + '_post_hoc_parameters.csv'))

                # Append the corrected p values
                for correction in correction_method:
                    corrected_p_value_column_name = correction + '_p_value'
                    post_hoc_model_parameters[corrected_p_value_column_name] = \
                        reshape_p_values_correction_post_hoc_all_method[correction_method.index(correction),
                                                                        networks_list.index(network),
                                                                        models.index(model), ...]

                    f_test_model_parameters[corrected_p_value_column_name] = \
                        reshape_p_values_correction_f_test_all_method[correction_method.index(correction),
                                                                      networks_list.index(network),
                                                                      models.index(model), 0]

                    post_hoc_model_parameters.to_csv(os.path.join(regression_output_directory,
                                                                  model + '_post_hoc_parameters.csv'),
                                                     index=False)

                    f_test_model_parameters.to_csv(os.path.join(regression_output_directory,
                                                                model + '_f_test_parameters.csv'),
                                                   index=False)


def joint_models_correction(root_analysis_directory, kinds, models,
                            correction_methods,
                            networks=None, alpha=0.05):
    """Performs a joint models correction for the whole brains models,
    or the networks models.


    """
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
                                                 model + '_parameters.csv'),
                                    index=False)

        else:
            all_models_p_values = []
            for model in models:
                network_model_p_values = []
                for network in networks:
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

                # Append the network array of the model
                all_models_p_values.append(network_model_p_values_array)

            # convert it to array
            all_models_p_values_array = np.array(all_models_p_values)
            list_of_p_values = []

            for i in range(len(models)):
                list_of_p_values.append(all_models_p_values_array[i])
            # Flatten the nested list of p values
            all_models_p_values_list_flat = data_management.flatten(values=list_of_p_values)

            for correction in correction_methods:
                _, corrected_p_values, _, _ = multipletests(pvals=all_models_p_values_list_flat,
                                                            alpha=alpha,
                                                            method=correction)
                # Rebuild the nested list of p values
                corrected_p_values_unflatten = data_management.unflatten(corrected_p_values, list_of_p_values)

                # Write the corrected p values in corresponding CSV
                for model in models:
                    for network in networks:
                        model_network_df_path = Path(os.path.join(root_analysis_directory, kind, network,
                                                                  model + '_parameters.csv'))
                        if model_network_df_path.exists():
                            model_network_df = pd.read_csv(os.path.join(root_analysis_directory, kind,
                                                                        network, model + '_parameters.csv'))
                            # Append the corrected p value column
                            model_network_df[correction + "corrected_pvalues"] = \
                                corrected_p_values_unflatten[models.index(model)][networks.index(network), ...]

                            # Overwrite the dataframe
                            model_network_df.to_csv(os.path.join(root_analysis_directory, kind, network,
                                                                 model + '_parameters.csv'),
                                                    index=False)

                        else:
                            pass
