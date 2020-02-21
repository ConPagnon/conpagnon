"""
 Created by db242421 at 07/12/18

 """

import importlib
from conpagnon.data_handling import data_architecture, dictionary_operations, atlas, data_management
from conpagnon.utils import folders_and_files_management
from conpagnon.computing import compute_connectivity_matrices as ccm
from conpagnon.machine_learning import classification
from conpagnon.connectivity_statistics import parametric_tests
from conpagnon.plotting import display
from sklearn import covariance
import pandas as pd
import numpy as np

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

import time


# Identify important brain connection in each network
# Atlas set up
atlas_folder = '/neurospin/grip/protocols/MRI/Gradient_hippocampe_AB_2017/rsfmri-for-neurospin/sub01_eb110475/MTL_ROI'
atlas_name = 'sub01_eb110475_atlas.nii'
labels_text_file = '/neurospin/grip/protocols/MRI/Gradient_hippocampe_AB_2017/atlas/Atlas_custom_AB/STM_atlas_db.csv'
# Atlas path
# Read labels regions files
atlas_nodes, labels_regions, labels_colors, n_nodes = atlas.fetch_atlas(
    atlas_folder=atlas_folder,
    atlas_name=atlas_name,
    network_regions_number='auto',
    colors_labels='auto',
    labels=labels_text_file,
    normalize_colors=False)


# File extension for individual atlases images, and corresponding
# text labels files.
individual_atlas_file_extension = '*.nii'
individual_atlas_labels_extension = '*.csv'

# Full path, including extension, to the text file containing
# all the subject identifiers.
subjects_ID_data_path = \
    '/neurospin/grip/protocols/MRI/Gradient_hippocampe_AB_2017/ressources_txt/sujetsmemomix.txt'

# Full to the following directories: individual atlases images,
# individual text labels files, and individual confounds directories.
individual_atlases_directory = \
    '/neurospin/grip/protocols/MRI/Gradient_hippocampe_AB_2017/atlas/atlas_indiv_images'
individual_atlases_labels_directory = \
    '/neurospin/grip/protocols/MRI/Gradient_hippocampe_AB_2017/atlas/atlas_indiv_labels'
individual_confounds_directory = \
    '/neurospin/grip/protocols/MRI/Gradient_hippocampe_AB_2017/regressors'

root_fmri_data_directory = '/neurospin/grip/protocols/MRI/Gradient_hippocampe_AB_2017/fmri_images'

groupes = ['controls']

# Check all functional files existence
folders_and_files_management.check_directories_existence(
    root_directory=root_fmri_data_directory,
    directories_list=groupes)

# Metrics list of interest, connectivity matrices will be
# computed according to this list
kinds = ['tangent', 'partial correlation', 'correlation']

# Fetch data in the case of individual atlases for each subjects.
organised_data_with_individual_atlas = data_architecture.fetch_data_with_individual_atlases(
    subjects_id_data_path=subjects_ID_data_path,
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groupes,
    individual_atlases_directory=individual_atlases_directory,
    individual_atlases_labels_directory=individual_atlases_labels_directory,
    individual_atlas_labels_extension=individual_atlas_labels_extension,
    individual_atlas_file_extension=individual_atlas_file_extension,
    individual_counfounds_directory=individual_confounds_directory)

organised_data_wo_individual_atlas = data_architecture.fetch_data(
    subjects_id_data_path=subjects_ID_data_path,
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groupes,
    individual_counfounds_directory=None
)

# Nilearn cache directory
nilearn_cache_directory = '/neurospin/grip/protocols/MRI/Gradient_hippocampe_AB_2017/nilearn_cache'

# Repetition time between time points in the fmri acquisition, usually 2.4s or 2.5s
repetition_time = 2.4


# Computing time series for each subjects according to individual atlases for each subjects
start = time.time()
times_series_individual_atlases = ccm.time_series_extraction_with_individual_atlases(
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groupes,
    subjects_id_data_path=subjects_ID_data_path,
    group_data=organised_data_with_individual_atlas,
    repetition_time=repetition_time,
    nilearn_cache_directory=nilearn_cache_directory)
end = time.time()
total_extraction_ts = (end - start)/60.

times_series_wo_individual_atlases = ccm.time_series_extraction(
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groupes,
    subjects_id_data_path=subjects_ID_data_path,
    reference_atlas='/neurospin/grip/protocols/MRI/'
                    'Gradient_hippocampe_AB_2017/atlas/Atlas_custom_AB/4D_AAL_STM.nii',
    group_data=organised_data_wo_individual_atlas,
    repetition_time=repetition_time,
    nilearn_cache_directory=nilearn_cache_directory
)

# Compute connectivity matrices for multiple metrics
# Covariance estimator
covariance_estimator = covariance.LedoitWolf()

# Choose the time series dictionary
time_series_dict = times_series_wo_individual_atlases

# Computing for each metric, and each subjects in each groups,
# the connectivity matrices using the computed time series.
subjects_connectivity_matrices = ccm.individual_connectivity_matrices(
    time_series_dictionary=time_series_dict,
    kinds=kinds, covariance_estimator=covariance_estimator,
    vectorize=False, z_fisher_transform=False)

# Computing for each metric, and each subjects in each groups,
# the fisher transform connectivity matrices using the computed time series.
Z_subjects_connectivity_matrices = ccm.individual_connectivity_matrices(
    time_series_dictionary=time_series_dict,
    kinds=kinds, covariance_estimator=covariance_estimator,
    vectorize=False, z_fisher_transform=True,
    discarding_diagonal=False)

# Computing for each metric, and each groups the mean connectivity matrices,
# from the z fisher transform matrices.
Z_mean_groups_connectivity_matrices = ccm.group_mean_connectivity(
    subjects_connectivity_matrices=Z_subjects_connectivity_matrices,
    kinds=kinds)

subjects_list = list(subjects_connectivity_matrices['controls'].keys())
# Statistical analysis

connectivity_matrices_loaded = folders_and_files_management.load_object(
    full_path_to_object=''
)

# read the behavioral data
behavioral_data = pd.read_csv(
    filepath_or_buffer='/neurospin/grip/protocols/MRI/Gradient_hippocampe_AB_2017/'
                       'ressources_txt/PCA-notesbrutes.csv',
    delimiter=';')
# Shift the index column to the subject nip column
behavioral_data = data_management.shift_index_column(
    panda_dataframe=behavioral_data,
    columns_to_index=['subject'])

# Reorder the behavioral dataframe to match the subjects list
behavioral_data.reindex(subjects_list)

# Stack the connectivity matrices
from nilearn.connectome import sym_matrix_to_vec
kind = 'correlation'

all_connectivity_matrices = sym_matrix_to_vec(
    np.array([connectivity_matrices_loaded['controls'][s][kind]
              for s in subjects_list]), discard_diagonal=True)

from conpagnon.machine_learning import predictors_selection_correlation


score = np.zeros((len(subjects_list), 1))
score[:, 0] = behavioral_data['Age']

R_mat, P_mat = predictors_selection_correlation(
    training_connectivity_matrices=all_connectivity_matrices,
    training_set_behavioral_scores=score)
from statsmodels.stats.multitest import multipletests
P_mat_corrected = multipletests(pvals=P_mat, alpha=0.05,
                                method='fdr_bh')[1]





import statsmodels.api as sm
X = sm.add_constant(behavioral_data['Age'])

R_mat_v2 = np.zeros(all_connectivity_matrices.shape[1])
T_mat = np.zeros(all_connectivity_matrices.shape[1])

for i in range(all_connectivity_matrices.shape[1]):
    model = sm.OLS(all_connectivity_matrices[:, i], X).fit()
    T_mat[i] = model.tvalues[1]

from scipy.stats import t
R_mat_v2 = np.sign(T_mat) * np.sqrt(T_mat**2 / (model.df_resid + T_mat**2))
P_mat_v2 = 2*t.sf(np.abs(T_mat), model.df_resid)