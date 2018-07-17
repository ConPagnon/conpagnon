import importlib
from data_handling import atlas, data_architecture, dictionary_operations
from utils import folders_and_files_management
from computing import compute_connectivity_matrices as ccm
from machine_learning import classification
from connectivity_statistics import parametric_tests
from plotting import display
from data_handling import data_management
import os
from connectivity_statistics.regression_analysis_model import one_way_anova, one_way_anova_network
import statsmodels.api as sm
import pandas as pd
import statsmodels.stats.multicomp as multi
from scipy import stats
import numpy as np
from statsmodels.stats.multitest import multipletests

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


groups = ['impaired_language', 'non_impaired_language', 'controls']
kinds = ['tangent', 'partial correlation', 'correlation']
models = ['intra', 'intra_homotopic']
variables_in_model = ['langage_clinique']
root_analysis_directory = '/media/db242421/db242421_data/ConPagnon_data/language_study_anova'
# Put a name for the subjects columns
behavioral_dataframe = data_management.read_csv(os.path.join(root_analysis_directory, 'behavioral_data.csv'))
behavioral_dataframe = behavioral_dataframe.rename(columns={behavioral_dataframe.columns[0]: 'subjects'})
# Shift index to be the subjects identifiers
behavioral_dataframe = data_management.shift_index_column(
    panda_dataframe=behavioral_dataframe,
    columns_to_index=['subjects'])
# Correction method
correction_method = ['fdr_bh', 'bonferroni']
# network list for model at the network level
networks_list = ['DMN']
# Type one error threshold
alpha = 0.05

# One way anova: whole brain level
one_way_anova(models=models,
              groups=groups,
              kinds=kinds,
              correction_method=correction_method,
              root_analysis_directory=root_analysis_directory,
              variables_in_model=variables_in_model,
              behavioral_dataframe=behavioral_dataframe)

# One way anova: network level
one_way_anova_network(root_analysis_directory=root_analysis_directory,
                      groups=groups,
                      kinds=kinds,
                      networks_list=networks_list,
                      models=models,
                      behavioral_dataframe=behavioral_dataframe,
                      variables_in_model=variables_in_model,
                      correction_method=correction_method)