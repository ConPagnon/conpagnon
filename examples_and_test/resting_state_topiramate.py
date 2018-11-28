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
from nilearn.connectome import sym_matrix_to_vec
from math import ceil
from connectivity_statistics import regression_analysis_model
from plotting.display import t_and_p_values_barplot
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, LeaveOneOut, GridSearchCV
from sklearn.svm import LinearSVC
from nilearn.plotting import plot_prob_atlas
from machine_learning.features_indentification import features_weights_max_t_correction, \
    features_weights_parametric_correction
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

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:08:51 

@author: Dhaif BEKHA(dhaif.bekha@cea.fr)

Resting state: Topiramate

"""

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
groups = ['controls_topiramate', 'wotopi', 'topi']
# The root fmri data directory containing all the fmri files directories
root_fmri_data_directory = \
    '/media/db242421/db242421_data/ConPagnon_data/fmri_images'
# Check all functional files existence
folders_and_files_management.check_directories_existence(
    root_directory=root_fmri_data_directory,
    directories_list=groups)

# output csv directory
output_csv_directory_path = '/media/db242421/db242421_data/ConPagnon_data/topi_wotopi'
output_csv_directory = data_management.create_directory(
    directory=output_csv_directory_path,
    erase_previous=True)

# Figure directory which can be useful for illustrating
output_figure_directory_path = os.path.join(output_csv_directory, 'figures')
output_figure_directory = data_management.create_directory(
    directory=output_figure_directory_path,
    erase_previous=True)

# Full path, including extension, to the text file containing
# all the subject identifiers.
subjects_ID_data_path = \
    '/media/db242421/db242421_data/ConPagnon_data/text_data/topi_wotopi_patients_id.txt'

# Clean the behavioral data by keeping only the subjects in the study
subjects_list = open(subjects_ID_data_path).read().split()

# Metrics list of interest, connectivity matrices will be
# computed according to this list
kinds = ['tangent', 'partial correlation', 'correlation']

organised_data = data_architecture.fetch_data(
    root_fmri_data_directory=root_fmri_data_directory,
    subjects_id_data_path=subjects_ID_data_path,
    groupes=groups,
    individual_counfounds_directory=None)

# Nilearn cache directory
nilearn_cache_directory = '/media/db242421/db242421_data/ConPagnon/nilearn_cache'

# Repetition time between time points in the fmri acquisition, usually 2.4s or 2.5s
repetition_time = 2.4

# Computing time series for each subjects according to individual atlases for each subjects
start = time.time()
times_series = ccm.time_series_extraction(
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groups,
    subjects_id_data_path=subjects_ID_data_path,
    reference_atlas=os.path.join(atlas_folder, atlas_name),
    group_data=organised_data,
    repetition_time=repetition_time,
    nilearn_cache_directory=nilearn_cache_directory)
end = time.time()
total_extraction_ts = (end - start)/60.

# Take one subject and plot the atlas
with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                       'atlas_verification_registration.pdf')) as pdf:

    a_t1_file = '/neurospin/grip/protocols/MRI/topiramate_SRetGD_2011/patients/wo_topi/' \
                'sub02_ld110465/mmx12/anatstar/wmanat_ld110465-2861_20111117_02.nii'
    a_mean_epi = '/neurospin/grip/protocols/MRI/topiramate_SRetGD_2011/patients/wo_topi/' \
                 'sub02_ld110465/mmx12/fMRIstar/wmeanrars110465-2862_20111117_10.nii'
    plt.figure()
    plot_prob_atlas(maps_img=os.path.join(atlas_folder, atlas_name),
                    title='AVCnn atlas on MNI template')
    pdf.savefig()
    plt.show()
    plt.figure()
    plot_prob_atlas(maps_img=os.path.join(atlas_folder, atlas_name),
                    title='AVCnn atlas on a T1 anatomical (sub02_ld110465)',
                    bg_img=a_t1_file)
    pdf.savefig()
    plt.show()
    plt.figure()
    plot_prob_atlas(maps_img=os.path.join(atlas_folder, atlas_name),
                    title='AVCnn atlas on a mean EPI (sub02_ld110465)',
                    bg_img=a_mean_epi)
    pdf.savefig()
    plt.show()

# Compute connectivity matrices for multiple metrics
# Covariance estimator
covariance_estimator = covariance.LedoitWolf()

# Choose the time series dictionary
time_series_dict = times_series

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

folders_and_files_management.save_object(
    object_to_save=subjects_connectivity_matrices,
    saving_directory=output_csv_directory,
    filename='topiramate_connectivity_matrices.pkl'
)

# Plot matrices
for group in groups:
    subjects = list(subjects_connectivity_matrices[group].keys())
    with backend_pdf.PdfPages(os.path.join(output_figure_directory, 'tangent_' + group + '.pdf')) as pdf:
        for s in subjects:
            plt.figure()
            display.plot_matrix(matrix=subjects_connectivity_matrices[group][s]['tangent'],
                                labels_colors=labels_colors,
                                mpart='all',
                                horizontal_labels=labels_regions,
                                vertical_labels=labels_regions,
                                title='{} connectivity'.format(s))
            pdf.savefig()
            plt.show()

# Classification
subjects_connectivity_matrices = folders_and_files_management.load_object(
    full_path_to_object=os.path.join('/media/db242421/db242421_data/ConPagnon_data/topi_wotopi/'
                                     'topiramate_connectivity_matrices.pkl')
)
kind = 'tangent'
groups_to_classify = ['controls_topiramate', 'wotopi']
features_labels = np.hstack((1*np.ones(len(subjects_connectivity_matrices[groups_to_classify[0]].keys())),
                             2*np.ones(len(subjects_connectivity_matrices[groups_to_classify[1]].keys()))))
features = np.array([sym_matrix_to_vec(subjects_connectivity_matrices[group][s][kind], discard_diagonal=True)
                     for group in groups_to_classify
                     for s in list(subjects_connectivity_matrices[group].keys())])

if __name__ == '__main__':
    # Find the best C parameters in linearSVC
    sss = StratifiedShuffleSplit(n_splits=10000)
    search_C = GridSearchCV(estimator=LinearSVC(),
                            param_grid={'C': [10, 100, 1000]},
                            cv=sss,
                            n_jobs=10,
                            verbose=1)
    search_C.fit(X=features, y=features_labels)

    print("Classification between {} and {} with a L2 linear SVM achieved the best score of {} % accuracy "
          "with C={}".format(groups_to_classify[0], groups_to_classify[1],
                             search_C.best_score_ * 100, search_C.best_params_['C']))

    C = search_C.best_params_['C']
    cv_scores = cross_val_score(estimator=LinearSVC(C=C), X=features,
                                y=features_labels, cv=sss,
                                scoring='accuracy', n_jobs=16,
                                verbose=1)
    print('mean accuracy {} % +- {} %'.format(cv_scores.mean() * 100, cv_scores.std() * 100))

# not enough subjects in the patients groups to apply the algorithm
# discriminative brain connections....
n_permutations = 10000
labels_permuted = np.array([np.random.permutation(features_labels)
                            for n in range(n_permutations)])

null_weight_distribution = np.zeros((n_permutations, features.shape[1]))

true_svc = LinearSVC(C=0.001)
true_svc.fit(X=features, y=features_labels)
true_weights_distribution = true_svc.coef_[0, ...]
for n in range(n_permutations):
    print("Compute classification for permutation # {} out {}".format(n, n_permutations))
    svc = LinearSVC(C=0.001)
    svc.fit(X=features, y=labels_permuted[n, ...])
    permuted_weights = svc.coef_[0, ...]
    null_weight_distribution[n, ...] = permuted_weights

p_values_corrected = features_weights_parametric_correction(
    null_distribution_features_weights=null_weight_distribution,
    normalized_mean_weight=true_weights_distribution)


# Extract homotopic connectivity coefficients on connectivity matrices
# Homotopic roi couple position in the connectivity matrices.
homotopic_roi_indices = np.array([
    (1, 0), (2, 3), (4, 5), (6, 7), (8, 11), (9, 10), (13, 12), (14, 15), (16, 17), (18, 19), (20, 25),
    (21, 26), (22, 29), (23, 28), (24, 27), (30, 31), (32, 33), (35, 34), (36, 37), (38, 39), (44, 40),
    (41, 45), (42, 43), (46, 49), (47, 48), (50, 53), (53, 54), (54, 57), (55, 56), (58, 61), (59, 60),
    (62, 63), (64, 65), (66, 67), (68, 69), (70, 71)])

# Atlas excel information file
atlas_excel_file = '/media/db242421/db242421_data/atlas_AVCnn/atlas_version2.xlsx'
sheetname = 'complete_atlas'

atlas_information = pd.read_excel(atlas_excel_file, sheetname='complete_atlas')
# Indices of left and homotopic right regions indices
left_regions_indices = homotopic_roi_indices[:, 0]
right_regions_indices = homotopic_roi_indices[:, 1]
# number of regions in the left and right side, should be the same because
# it's homotopic regions only we are interested in
n_left_regions = len(left_regions_indices)
n_right_regions = len(right_regions_indices)
try:
    n_left_regions == n_right_regions
except ValueError:
    print('Number of regions in the left and the right side are not equal !')

# Extract from the Z fisher subject connectivity dictionary the connectivity coefficients of interest
homotopic_connectivity = ccm.subjects_mean_connectivity_(
    subjects_individual_matrices_dictionnary=subjects_connectivity_matrices,
    connectivity_coefficient_position=homotopic_roi_indices, kinds=[kind],
    groupes=groups)

# Compute mean and standard deviation assuming gaussian behavior
kinds =[kind]
homotopic_distribution_parameters = dict.fromkeys(groups)
for groupe in groups:
    homotopic_distribution_parameters[groupe] = dict.fromkeys(kinds)
    for kind in kinds:
        # Stack the mean homotopic connectivity of each subject for the current group
        subjects_mean_homotopic_connectivity = np.array(
            [homotopic_connectivity[groupe][subject][kind]['mean connectivity']
             for subject in homotopic_connectivity[groupe].keys()])
        # Estimate the mean and std assuming a Gaussian behavior
        subjects_mean_homotopic_connectivity_, mean_homotopic_estimation, std_homotopic_estimation = \
            parametric_tests.functional_connectivity_distribution_estimation(subjects_mean_homotopic_connectivity)
        # Fill a dictionnary saving the results for each groups and kind
        homotopic_distribution_parameters[groupe][kind] = {
            'subjects mean homotopic connectivity': subjects_mean_homotopic_connectivity_,
            'homotopic distribution mean': mean_homotopic_estimation,
            'homotopic distribution standard deviation': std_homotopic_estimation}

output_figure_directory = '/media/db242421/db242421_data/ConPagnon_data/topi_wotopi/figures'
fitted_dist_color1, fitted_dist_color2, fitted_dist_color3 = 'blue', 'red', 'green'
raw_data_color1, raw_data_color2, raw_data_color3 = 'blue', 'red', 'green'
hist_color = ['blue', 'red', 'green']
fit_color = ['blue', 'red', 'green']
with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                       'mean_homotopic_connectivity_distribution.pdf')) as pdf:
    for kind in kinds:
        plt.figure()
        for groupe in groups:
            group_connectivity = homotopic_distribution_parameters[groupe][kind][
                'subjects mean homotopic connectivity']
            group_mean = homotopic_distribution_parameters[groupe][kind][
                'homotopic distribution mean']
            group_std = homotopic_distribution_parameters[groupe][kind][
                'homotopic distribution standard deviation']
            display.display_gaussian_connectivity_fit(
                vectorized_connectivity=group_connectivity,
                estimate_mean=group_mean,
                estimate_std=group_std,
                raw_data_colors=hist_color[groups.index(groupe)],
                fitted_distribution_color=fit_color[groups.index(groupe)],
                title='Whole brain mean homotopic connectivity distribution  for {}'.format(kind),
                xtitle='Functional connectivity coefficient', ytitle='Density (a.u)',
                legend_fitted='{} gaussian fitted distribution'.format(groupe),
                legend_data=groupe, display_fit='yes', ms=3.5)
            plt.axvline(x=group_mean, color=fit_color[groups.index(groupe)],
                        linewidth=4)
        pdf.savefig()
        plt.show()

