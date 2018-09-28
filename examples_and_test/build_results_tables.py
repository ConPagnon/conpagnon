import importlib
from data_handling import atlas, data_architecture, dictionary_operations
from utils import folders_and_files_management
from computing import compute_connectivity_matrices as ccm
from sklearn import covariance
from machine_learning import classification
from connectivity_statistics import parametric_tests
from plotting import display
from matplotlib.backends import backend_pdf
from data_handling import data_management
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from connectivity_statistics import regression_analysis_model
from plotting.display import t_and_p_values_barplot
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
results_directory = "/media/db242421/db242421_data/ConPagnon_data/patients_behavior_ACM/" \
                    "pc1_language_zscores/regression_analysis/tangent"

# models to read
models = ['intra', 'ipsi_intra', 'contra_intra']

# Variable to read
variable_to_read = ['pc1_language_zscores', 'Sexe[T.M]', 'lesion_normalized']

# Variable name in the final filename
final_variable_name = ['PC1_language', 'Sex_[Male]', 'Lesion_size']
columns_names = ['Network', 'Beta', 'Beta_std', 't-score', '[0.025 ', ' 0.0975]',
                 'p', 'p_fdr', 'p_bonferroni']

network_name = ['DMN', 'Executive', 'Language', 'MTL', 'Primary_Visual',
                'Salience', 'Secondary_Visual', 'Sensorimotor', 'Visuospatial']


for model in models:
    for network in network_name:
        # parameters file
        parameters_file = pd.read_csv(os.path.join(results_directory, network,
                                                   model + '_parameters.csv'))
        # Fetch interesting value
        parameters_file = data_management.shift_index_column(panda_dataframe=parameters_file,
                                                             columns_to_index=['variables'])

        for variable in variable_to_read:
            parameters_file.loc[variable]
