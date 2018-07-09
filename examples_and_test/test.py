import importlib
from data_handling import atlas, data_architecture, dictionary_operations
from utils import folders_and_files_management
from computing import compute_connectivity_matrices as ccm
from machine_learning import classification
from connectivity_statistics import parametric_tests
from plotting import display
from data_handling import data_management
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
from scipy.stats import ttest_ind
from scipy import stats
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


# Post-hoc analysis after a ANOVA
output_csv_directory = '/media/db242421/db242421_data/ConPagnon_data/language_study'
regression_analysis_directory = os.path.join(output_csv_directory, 'regression_analysis')
groups = ['impaired_language', 'non_impaired_language', 'controls']
kinds = ['tangent', 'partial correlation', 'correlation']
models = ['mean_connectivity', 'mean_homotopic', 'mean_ipsilesional', 'mean_contralesional']
variables_in_model = ['langage_clinique']

# Read behavioral dataframe CSV file
behavioral_dataframe = data_management.read_csv('/media/db242421/db242421_data/ConPagnon_data/language_study/'
                                                'behavioral_data.csv')
behavioral_dataframe = behavioral_dataframe.rename(columns={behavioral_dataframe.columns[0]: 'subjects'})
# Shift index to be the subjects identifiers
behavioral_dataframe = data_management.shift_index_column(
    panda_dataframe=behavioral_dataframe,
    columns_to_index=['subjects'])


def write_raw_data(output_csv_directory, kinds, groups, models, variables_in_model,
                   behavioral_dataframe):
    """Write the response variable along with the co-variables in a unique
    dataframe inside a csv files.
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
    dataframe inside a csv files, at the network level
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

write_raw_data(output_csv_directory=output_csv_directory,
               kinds=kinds,
               groups=groups,
               models=models,
               variables_in_model=variables_in_model,
               behavioral_dataframe=behavioral_dataframe)

write_raw_data_network(output_csv_directory=output_csv_directory,
                       kinds=kinds,
                       groups=groups,
                       models=['intra_homotopic'],
                       behavioral_dataframe=behavioral_dataframe,
                       network_list=['DMN'],
                       variables_in_model=['langage_clinique'])


model_dataframe_ = data_management.read_csv('')


# Fit a classical linear model with a categorical variable
form = 'mean_homotopic~langage_clinique'
connectivity_model = smf.ols(formula=form, data=model_dataframe).fit()
# Perform anova on language status
connectivity_model_anova = sm.stats.anova_lm(connectivity_model, typ=2)
print(connectivity_model_anova)

# Pairwise t-test
from statsmodels.stats.libqsturng import psturng
tukey_analysis = pairwise_tukeyhsd(model_dataframe_['mean_homotopic'],
                                   model_dataframe_['langage_clinique'])

# Compute t scores  from all contrast coefficient estimation
t_scores = tukey_analysis.meandiffs / tukey_analysis.std_pairs / np.sqrt(2)
# Compute p values from t scores
raw_pvalues = stats.t.sf(np.abs(t_scores), 50) * 2
from statsmodels.stats.multitest import multipletests
import statsmodels.stats.multicomp as multi
# Correct the p values for all test.
pvalues_corrected = multipletests(raw_pvalues, method='fdr_bh', alpha=0.05)[1]




from patsy import dmatrix
import statsmodels.stats.multicomp as multi
from statsmodels.stats.multitest import multipletests
root_analysis_directory = '/media/db242421/db242421_data/ConPagnon_data/language_study_anova'
whole_brain_model = ['mean_homotopic']
behavioral_dataframe = data_management.read_csv(os.path.join(root_analysis_directory, 'behavioral_data.csv'))
behavioral_dataframe = behavioral_dataframe.rename(columns={behavioral_dataframe.columns[0]: 'subjects'})
# Shift index to be the subjects identifiers
behavioral_dataframe = data_management.shift_index_column(
    panda_dataframe=behavioral_dataframe,
    columns_to_index=['subjects'])

# The design matrix is the same for all model
design_matrix = dmatrix('+'.join(variables_in_model), behavioral_dataframe, return_type='dataframe')
# For each model: read the csv for each group, concatenate resultings dataframe, and append
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
        model_fit = parametric_tests.ols_regression_formula(formula=model_formulation,
                                                            data=model_dataframe)

        # Creation of a directory for the current analysis, ANOVA results
        # will be stored in a sub-directory called ANOVA
        regression_output_directory = folders_and_files_management.create_directory(
            directory=os.path.join(root_analysis_directory, 'regression_analysis/ANOVA', kind))
        # Perform one-way ANOVA with statsmodel
        model_anova = sm.stats.anova_lm(model_fit, typ=2)
        # Compute t-test between all pairs of groups in the
        # analysis with tukey HSD analysis
        model_tukey_analysis = multi.MultiComparison(
            model_dataframe[model],
            model_dataframe[variables_in_model[0]])
        model_tukey_analysis_results = model_tukey_analysis.tukeyhsd()
        # tukey t-score are t score scaled by sqrt(2)
        model_t_scores = model_tukey_analysis_results .meandiffs / model_tukey_analysis_results.std_pairs / np.sqrt(2)
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
        model_anova.to_csv(os.path.join(root_analysis_directory, 'regression_analysis/ANOVA', kind,
                                        model + '_f_test_parameters.csv'))
        # Save the parameters results of the post hoc analysis
        data_management.dataframe_to_csv(dataframe=post_hoc_dataframe,
                                         path=os.path.join(root_analysis_directory, 'regression_analysis/ANOVA',
                                                           kind, model + '_post_hoc_parameters.csv'))

    # Correct the p values for all model, and post-hoc t test
    all_model_p_values_post_hoc_array = np.array(all_model_post_hoc_p_values)
    post_hoc_t_test_corrected_p_values = multipletests(
        pvals=all_model_p_values_post_hoc_array.flatten(),
        method='fdr_bh')[1].reshape(all_model_p_values_post_hoc_array.shape)
    
    all_model_p_values_f_test_array = np.array(all_model_p_values_f_test)
    f_test_corrected_p_values = multipletests(
        pvals=all_model_p_values_f_test_array,
        method='fdr_bh')[1].reshape(all_model_p_values_f_test_array.shape)