"""
 Created by db242421 at 13/12/18

 """
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
import numpy as np
import matplotlib.pyplot as plt
from nilearn.connectome import sym_matrix_to_vec
from sklearn.model_selection import LeaveOneOut
from nilearn.connectome import ConnectivityMeasure
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

# Atlas set up
atlas_folder = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference'
atlas_name = 'atlas4D_2.nii'
labels_text_file = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference/atlas4D_2_labels.csv'
colors = ['navy', 'sienna', 'orange', 'orchid', 'indianred', 'olive',
          'goldenrod', 'turquoise', 'darkslategray', 'limegreen', 'black',
          'lightpink']
# Number of regions in each user defined networks
networks = [2, 10, 2, 6, 10, 2, 8, 6, 8, 8, 6, 4]
# Atlas path
# Read labels regions files
atlas_nodes, labels_regions, labels_colors, n_nodes = atlas.fetch_atlas(
    atlas_folder=atlas_folder,
    atlas_name=atlas_name,
    network_regions_number=networks,
    colors_labels=colors,
    labels=labels_text_file,
    normalize_colors=True)

# Groups name to include in the study
groups = ['patients_r', 'TDC']
# The root fmri data directory containing all the fmri files directories
root_fmri_data_directory = \
    '/media/db242421/db242421_data/ConPagnon_data/fmri_images'
# Check all functional files existence
folders_and_files_management.check_directories_existence(
    root_directory=root_fmri_data_directory,
    directories_list=groups)

# Full path, including extension, to the text file containing
# all the subject identifiers.
subjects_ID_data_path = \
    '/media/db242421/db242421_data/ConPagnon_data/text_data/acm_patients_and_controls_all.txt'

organised_data = data_architecture.fetch_data(
    subjects_id_data_path=subjects_ID_data_path,
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groups,
    individual_counfounds_directory=None
)

# kinds
kinds = ['tangent']

# Nilearn cache directory
nilearn_cache_directory = '/media/db242421/db242421_data/ConPagnon/nilearn_cache'

# Repetition time between time points in the fmri acquisition, usually 2.4s or 2.5s
repetition_time = 2.5

times_series = ccm.time_series_extraction(
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groups,
    subjects_id_data_path=subjects_ID_data_path,
    reference_atlas=os.path.join(atlas_folder, atlas_name),
    group_data=organised_data,
    repetition_time=repetition_time,
    nilearn_cache_directory=nilearn_cache_directory
)
#  Subjects list
controls = list(times_series['TDC'].keys())
patients = list(times_series['patients_r'].keys())

# Stack the times series for each group
controls_time_series = np.array([times_series['TDC'][s]['time_series'] for s in controls])
patients_time_series = np.array([times_series['patients_r'][s]['time_series'] for s in patients])

loo = LeaveOneOut()
# generate null distribution with the controls group
# by leave one out
one_patients_time_series = patients_time_series[0, ...]
for train, test in loo.split(controls):
    # Compute mean matrices, and connectitivity matrices
    # in the tangent on the subset of controls without
    # the leftout subject.
    connectivity_measure = ConnectivityMeasure(kind='tangent',
                                               vectorize=True,
                                               discard_diagonal=True)
    controls_subset_matrices = \
        connectivity_measure.fit_transform(X=controls_time_series[train, ...])
    controls_subset_mean = connectivity_measure.mean_
    # Project at the previously computed mean all the controls group
    # including the leftout controls
    all_controls_matrices = \
        connectivity_measure.transform(X=controls_time_series[np.hstack((train, test)), ...])

    # One sample t test, betweend the tangent matrix of the left out control, and the
    # subset tangent matrices of the rest of the control


connectivity_measure_2 = ConnectivityMeasure(kind='tangent', vectorize=True, discard_diagonal=True)
all_controls_at_once = connectivity_measure_2.fit_transform(X=controls_time_series[np.hstack((train, test)), ...])
all_controls_at_once_mean = connectivity_measure_2.mean_
