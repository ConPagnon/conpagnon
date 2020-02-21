import importlib
from conpagnon.data_handling import data_architecture, dictionary_operations, atlas, data_management
from conpagnon.utils import folders_and_files_management
from conpagnon.computing import compute_connectivity_matrices as ccm
from sklearn import covariance
from conpagnon.machine_learning import classification
from conpagnon.connectivity_statistics import parametric_tests
from conpagnon.plotting import display
from matplotlib.backends import backend_pdf
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilearn.connectome import sym_matrix_to_vec
from sklearn.model_selection import cross_val_score
from nilearn.plotting import plot_prob_atlas

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
groups = ['wotopi_all', 'controls_topiramate']
# The root fmri data directory containing all the fmri files directories
root_fmri_data_directory = \
    '/media/db242421/db242421_data/ConPagnon_data/fmri_images'
# Check all functional files existence
folders_and_files_management.check_directories_existence(
    root_directory=root_fmri_data_directory,
    directories_list=groups)

# output csv directory
output_csv_directory_path = '/media/db242421/db242421_data/ConPagnon_data/wotopi_all_controls'
output_csv_directory = data_management.create_directory(
    directory=output_csv_directory_path,
    erase_previous=True)

# Figure directory which can be useful for illustrating
output_figure_directory_path = os.path.join(output_csv_directory, 'figures')
output_figure_directory = data_management.create_directory(
    directory=output_figure_directory_path,
    erase_previous=False)

# Full path, including extension, to the text file containing
# all the subject identifiers.
subjects_ID_data_path = \
    '/media/db242421/db242421_data/ConPagnon_data/text_data/wotopi_all_controls_all.txt'

# Clean the behavioral data by keeping only the subjects in the study
subjects_list = open(subjects_ID_data_path).read().split()

# Metrics list of interest, connectivity matrices will be
# computed according to this list
kinds = ['tangent', 'correlation']

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
subjects_connectivity_matrices = Z_subjects_connectivity_matrices
# Plot matrices
for kind in kinds:

    for group in groups:
        subjects = list(subjects_connectivity_matrices[group].keys())
        with backend_pdf.PdfPages(os.path.join(output_figure_directory, kind + '_' + group + '.pdf')) as pdf:
            for s in subjects:
                plt.figure()
                display.plot_matrix(matrix=subjects_connectivity_matrices[group][s][kind],
                                    labels_colors=labels_colors,
                                    mpart='all',
                                    horizontal_labels=labels_regions,
                                    vertical_labels=labels_regions,
                                    title='{} connectivity'.format(s))
                pdf.savefig()
                #plt.show()

# Extract homotopic connectivity coefficients on connectivity matrices
# Homotopic roi couple position in the connectivity matrices.
homotopic_roi_indices = np.array([
    (1, 0), (2, 3), (4, 5), (6, 7), (8, 11), (9, 10), (13, 12), (14, 15), (16, 17), (18, 19), (20, 25),
    (21, 26), (22, 29), (23, 28), (24, 27), (30, 31), (32, 33), (35, 34), (36, 37), (38, 39), (44, 40),
    (41, 45), (42, 43), (46, 49), (47, 48), (50, 53), (53, 54), (54, 57), (55, 56), (58, 61), (59, 60),
    (62, 63), (64, 65), (66, 67), (68, 69), (70, 71)])
# Indices of left and homotopic right regions indices
left_regions_indices = homotopic_roi_indices[:, 0]
right_regions_indices = homotopic_roi_indices[:, 1]
# Atlas excel information file
atlas_excel_file = '/media/db242421/db242421_data/atlas_AVCnn/atlas_version2.xlsx'
sheetname = 'complete_atlas'

atlas_information = pd.read_excel(atlas_excel_file, sheetname='complete_atlas')

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
    connectivity_coefficient_position=homotopic_roi_indices, kinds=kinds,
    groupes=groups)

# Compute mean and standard deviation assuming gaussian behavior
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
                title='Mean homotopic connectivity distribution  for {}'.format(kind),
                xtitle='Functional connectivity coefficient', ytitle='Proportion of subjects',
                legend_fitted='{} distribution'.format(groupe),
                legend_data=groupe, display_fit='yes', ms=3.5)
            plt.axvline(x=group_mean, color=fit_color[groups.index(groupe)],
                        linewidth=4)
        pdf.savefig()
        plt.show()

# Connectivity intra-network
groupes = groups

# Connectivity intra-network
intra_network_connectivity_dict, network_dict, network_labels_list, network_label_colors = \
    ccm.intra_network_functional_connectivity(
        subjects_individual_matrices_dictionnary=Z_subjects_connectivity_matrices,
        groupes=groupes, kinds=kinds,
        atlas_file=atlas_excel_file,
        sheetname=sheetname,
        roi_indices_column_name='atlas4D index',
        network_column_name='network',
        color_of_network_column='Color')

# Estimation of mean intra network connectivity mean and std
intra_network_distribution_parameters = dict.fromkeys(network_labels_list)
for network in network_labels_list:
    intra_network_distribution_parameters[network] = dict.fromkeys(groupes)
    for groupe in groupes:
        intra_network_distribution_parameters[network][groupe] = dict.fromkeys(kinds)
        for kind in kinds:
            # Stack the mean homotopic connectivity of each subject for the current group
            subjects_mean_intra_network_connectivity = np.array(
                [intra_network_connectivity_dict[groupe][subject][kind][network]['network connectivity strength']
                 for subject in intra_network_connectivity_dict[groupe].keys()])
            # Estimate the mean and std assuming a Gaussian behavior
            subjects_mean_intra_network_connectivity_, mean_intra_estimation, std_intra_estimation = \
                parametric_tests.functional_connectivity_distribution_estimation(
                    subjects_mean_intra_network_connectivity)
            # Fill a dictionary saving the results for each groups and kind
            intra_network_distribution_parameters[network][groupe][kind] = {
                'subjects mean intra connectivity': subjects_mean_intra_network_connectivity_,
                'intra distribution mean': mean_intra_estimation,
                'intra distribution standard deviation': std_intra_estimation}

for kind in kinds:
    with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                           kind + '_intra_network_connectivity_distribution.pdf')) as pdf:

        for network in network_labels_list:
            plt.figure(constrained_layout=True)
            for groupe in groupes:
                group_connectivity = intra_network_distribution_parameters[network][groupe][kind][
                    'subjects mean intra connectivity']
                group_mean = intra_network_distribution_parameters[network][groupe][kind][
                    'intra distribution mean']
                group_std = intra_network_distribution_parameters[network][groupe][kind][
                    'intra distribution standard deviation']
                display.display_gaussian_connectivity_fit(
                    vectorized_connectivity=group_connectivity,
                    estimate_mean=group_mean,
                    estimate_std=group_std,
                    raw_data_colors=hist_color[groupes.index(groupe)],
                    fitted_distribution_color=fit_color[groupes.index(groupe)],
                    title='',
                    xtitle='Functional connectivity', ytitle='Proportion of subjects',
                    legend_fitted='{} distribution'.format(groupe),
                    legend_data=groupe, display_fit='yes', ms=3.5, line_width=2.5)
                plt.axvline(x=group_mean, color=fit_color[groupes.index(groupe)],
                            linewidth=2)
                plt.title('Connectivity distribution for the {} network'.format(network))

            pdf.savefig()
            plt.show()


# We can compute the homotopic connectivity for each network, i.e a intra-network homotopic connectivity
homotopic_intra_network_connectivity_d = dict.fromkeys(network_labels_list)
for network in network_labels_list:
    # Pick the 4D index of the roi in the network
    network_roi_ind = network_dict[network]['dataframe']['atlas4D index']
    # Extract connectivity coefficient couple corresponding to homotopic regions in the network
    network_homotopic_couple_ind = np.array([couple for couple in homotopic_roi_indices if (couple[0] or couple[1])
                                             in network_roi_ind])
    # Compute homotopic connectivity dictionary for the current network
    network_homotopic_d = ccm.subjects_mean_connectivity_(
        subjects_individual_matrices_dictionnary=Z_subjects_connectivity_matrices,
        connectivity_coefficient_position=network_homotopic_couple_ind,
        kinds=kinds, groupes=groupes)
    homotopic_intra_network_connectivity_d[network] = network_homotopic_d

# Create a homotopic intra-network dictionary with the same structure as the overall intra network dictionary
homotopic_intranetwork_d = dict.fromkeys(groupes)
for groupe in groupes:
    homotopic_intranetwork_d[groupe] = dict.fromkeys(Z_subjects_connectivity_matrices[groupe].keys())
    for subject in Z_subjects_connectivity_matrices[groupe].keys():
        homotopic_intranetwork_d[groupe][subject] = dict.fromkeys(kinds)
        for kind in kinds:
            homotopic_intranetwork_d[groupe][subject][kind] = \
                dict.fromkeys(list(homotopic_intra_network_connectivity_d.keys()))
            for network in list(homotopic_intra_network_connectivity_d.keys()):
                homotopic_intranetwork_d[groupe][subject][kind][network] =\
                    {'network connectivity strength':
                     homotopic_intra_network_connectivity_d[network][groupe][subject][kind]['mean connectivity']}

# estimate mean and std for each network, for the display of the intra-network homotopic connectivity
network_homotopic_distribution_parameters = dict.fromkeys(network_labels_list)
for network in network_labels_list:
    network_homotopic_distribution_parameters[network] = dict.fromkeys(groupes)
    for groupe in groupes:
        network_homotopic_distribution_parameters[network][groupe] = dict.fromkeys(kinds)
        for kind in kinds:
            # Stack the mean homotopic connectivity of each subject for the current group
            subjects_mean_homotopic_connectivity = np.array(
                [homotopic_intra_network_connectivity_d[network][groupe][subject][kind]['mean connectivity']
                 for subject in homotopic_intra_network_connectivity_d[network][groupe].keys()])
            # Estimate the mean and std assuming a Gaussian behavior
            subjects_mean_homotopic_connectivity_, mean_homotopic_estimation, std_homotopic_estimation = \
                parametric_tests.functional_connectivity_distribution_estimation(
                    subjects_mean_homotopic_connectivity)
            # Fill a dictionary saving the results for each groups and kind
            network_homotopic_distribution_parameters[network][groupe][kind] = {
                'subjects mean homotopic connectivity': subjects_mean_homotopic_connectivity_,
                'homotopic distribution mean': mean_homotopic_estimation,
                'homotopic distribution standard deviation': std_homotopic_estimation}

# Display for each network the mean homotopic distribution for all group
# Gaussian fit of homotopic connectivity
for kind in kinds:
    with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                           kind + '_intra_network_homotopic_connectivity_distribution.pdf')) as pdf:

        for network in network_labels_list:
            plt.figure(constrained_layout=True)
            for groupe in groupes:
                group_connectivity = network_homotopic_distribution_parameters[network][groupe][kind][
                    'subjects mean homotopic connectivity']
                group_mean = network_homotopic_distribution_parameters[network][groupe][kind][
                    'homotopic distribution mean']
                group_std = network_homotopic_distribution_parameters[network][groupe][kind][
                    'homotopic distribution standard deviation']
                display.display_gaussian_connectivity_fit(
                    vectorized_connectivity=group_connectivity,
                    estimate_mean=group_mean,
                    estimate_std=group_std,
                    raw_data_colors=hist_color[groupes.index(groupe)],
                    fitted_distribution_color=fit_color[groupes.index(groupe)],
                    title='',
                    xtitle='Functional connectivity', ytitle='Proportion of subjects',
                    legend_fitted='{} distribution'.format(groupe),
                    legend_data=groupe, display_fit='yes', ms=3.5, line_width=2.5)
                plt.axvline(x=group_mean, color=fit_color[groupes.index(groupe)],
                            linewidth=2)
                plt.title('Mean homotopic connectivity for the {} network'.format(network))

            pdf.savefig()
            plt.show()


# Fetch left and right connectivity
left_connectivity_matrices = ccm.extract_sub_connectivity_matrices(
    subjects_connectivity_matrices=subjects_connectivity_matrices,
    kinds=kinds, regions_index=left_regions_indices,
    vectorize=False, discard_diagonal=False)
right_connectivity_matrices = ccm.extract_sub_connectivity_matrices(
    subjects_connectivity_matrices=subjects_connectivity_matrices,
    kinds=kinds, regions_index=right_regions_indices,
    vectorize=False, discard_diagonal=False)


left_connectivity = ccm.mean_of_flatten_connectivity_matrices(
    subjects_individual_matrices_dictionary=left_connectivity_matrices,
    groupes=groups,
    kinds=kinds
)

right_connectivity = ccm.mean_of_flatten_connectivity_matrices(
    subjects_individual_matrices_dictionary=right_connectivity_matrices,
    groupes=groups,
    kinds=kinds
)
# Estimate mean and standard deviation of mean ipsilesional connectivity
left_connectivity_parameters = dict.fromkeys(groupes)
for groupe in groupes:
    left_connectivity_parameters[groupe] = dict.fromkeys(kinds)
    for kind in kinds:
        # Stack the mean homotopic connectivity of each subject for the current group
        subjects_mean_left_connectivity = np.array(
            [left_connectivity[groupe][subject][kind]['mean connectivity']
             for subject in left_connectivity[groupe].keys()])
        # Estimate the mean and std assuming a Gaussian behavior
        subjects_mean_left_connectivity_, mean_estimation, std_estimation = \
            parametric_tests.functional_connectivity_distribution_estimation(subjects_mean_left_connectivity)
        # Fill a dictionary saving the results for each groups and kind
        left_connectivity_parameters[groupe][kind] = {
            'subjects mean left connectivity': subjects_mean_left_connectivity_,
            'left distribution mean': mean_estimation,
            'left distribution standard deviation': std_estimation}

# Plot the distribution of mean left connectivity assuming gaussian
# behavior
with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                       'left_connectivity_distribution.pdf')) as pdf:
    for kind in kinds:
        plt.figure()
        for groupe in groupes:
            group_connectivity = left_connectivity_parameters[groupe][kind][
                'subjects mean left connectivity']
            group_mean = left_connectivity_parameters[groupe][kind][
                'left distribution mean']
            group_std = left_connectivity_parameters[groupe][kind][
                'left distribution standard deviation']
            display.display_gaussian_connectivity_fit(
                vectorized_connectivity=group_connectivity,
                estimate_mean=group_mean,
                estimate_std=group_std,
                raw_data_colors=hist_color[groupes.index(groupe)],
                fitted_distribution_color=fit_color[groupes.index(groupe)],
                title='Left mean connectivity distribution  for {}'.format(kind),
                xtitle='Functional connectivity coefficient', ytitle='Proportion of subjects',
                legend_fitted='{} distribution'.format(groupe),
                legend_data=groupe, display_fit='yes', ms=3.5, line_width=2.5)
            plt.axvline(x=group_mean, color=fit_color[groupes.index(groupe)],
                        linewidth=2)
        pdf.savefig()
        plt.show()

# Estimate mean and standard deviation of mean contralesional connectivity
right_connectivity_parameters = dict.fromkeys(groupes)
for groupe in groupes:
    right_connectivity_parameters[groupe] = dict.fromkeys(kinds)
    for kind in kinds:
        # Stack the mean homotopic connectivity of each subject for the current group
        subjects_mean_contra_connectivity = np.array(
            [right_connectivity[groupe][subject][kind]['mean connectivity']
             for subject in right_connectivity[groupe].keys()])
        # Estimate the mean and std assuming a Gaussian behavior
        subjects_mean_contra_connectivity_, mean_estimation, std_estimation = \
            parametric_tests.functional_connectivity_distribution_estimation(subjects_mean_contra_connectivity)
        # Fill a dictionary saving the results for each groups and kind
        right_connectivity_parameters[groupe][kind] = {
            'subjects mean right connectivity': subjects_mean_contra_connectivity_,
            'right distribution mean': mean_estimation,
            'right distribution standard deviation': std_estimation}

# Plot the distribution of mean ipsilesional connectivity assuming gaussian
# behavior
with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                       'right_mean_connectivity_distribution.pdf')) as pdf:
    for kind in kinds:
        plt.figure()
        for groupe in groupes:
            group_connectivity = right_connectivity_parameters[groupe][kind][
                'subjects mean right connectivity']
            group_mean = right_connectivity_parameters[groupe][kind][
                'right distribution mean']
            group_std = right_connectivity_parameters[groupe][kind][
                'right distribution standard deviation']
            display.display_gaussian_connectivity_fit(
                vectorized_connectivity=group_connectivity,
                estimate_mean=group_mean,
                estimate_std=group_std,
                raw_data_colors=hist_color[groupes.index(groupe)],
                fitted_distribution_color=fit_color[groupes.index(groupe)],
                title='Right mean connectivity distribution for {}'.format(kind),
                xtitle='Functional connectivity coefficient', ytitle='Proportion of subjects',
                legend_fitted='{} distribution'.format(groupe),
                legend_data=groupe, display_fit='yes', ms=3.5, line_width=2)
            plt.axvline(x=group_mean, color=fit_color[groupes.index(groupe)],
                        linewidth=2)
        pdf.savefig()
        plt.show()

# Compute the intra-network connectivity for the left hemisphere in
# groups
left_intra_network_connectivity_dict, \
    left_network_dict, left_network_labels_list, left_network_label_colors = \
    ccm.intra_network_functional_connectivity(
        subjects_individual_matrices_dictionnary=left_connectivity_matrices,
        groupes=groupes, kinds=kinds,
        atlas_file=atlas_excel_file,
        sheetname='Hemisphere_regions',
        roi_indices_column_name='index',
        network_column_name='network',
        color_of_network_column='Color')

# Estimate mean and standard deviation of left intra network parameters
left_intra_network_distribution_parameters = dict.fromkeys(network_labels_list)
for network in network_labels_list:
    if network not in ['Basal_Ganglia', 'Precuneus', 'Auditory']:
        left_intra_network_distribution_parameters[network] = dict.fromkeys(groupes)
        for groupe in groupes:
            left_intra_network_distribution_parameters[network][groupe] = dict.fromkeys(kinds)
            for kind in kinds:
                # Stack the mean homotopic connectivity of each subject for the current group
                subjects_mean_left_intra_network_connectivity = np.array(
                    [left_intra_network_connectivity_dict[groupe][subject][kind][network][
                         'network connectivity strength']
                     for subject in left_intra_network_connectivity_dict[groupe].keys()])
                # Estimate the mean and std assuming a Gaussian behavior
                subjects_mean_left_intra_network_connectivity_, mean_intra_estimation, std_intra_estimation = \
                    parametric_tests.functional_connectivity_distribution_estimation(
                        subjects_mean_left_intra_network_connectivity)
                # Fill a dictionary saving the results for each groups and kind
                left_intra_network_distribution_parameters[network][groupe][kind] = {
                    'subjects mean left intra connectivity': subjects_mean_left_intra_network_connectivity_,
                    'left intra distribution mean': mean_intra_estimation,
                    'left intra distribution standard deviation': std_intra_estimation}

for kind in kinds:
    with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                           kind + '_left_intra_network_connectivity_distribution.pdf')) as pdf:

        for network in network_labels_list:
            if network not in ['Basal_Ganglia', 'Precuneus', 'Auditory']:
                plt.figure(constrained_layout=True)
                for groupe in groupes:
                    group_connectivity = left_intra_network_distribution_parameters[network][groupe][kind][
                        'subjects mean left intra connectivity']
                    group_mean = left_intra_network_distribution_parameters[network][groupe][kind][
                        'left intra distribution mean']
                    group_std =  left_intra_network_distribution_parameters[network][groupe][kind][
                        'left intra distribution standard deviation']
                    display.display_gaussian_connectivity_fit(
                        vectorized_connectivity=group_connectivity,
                        estimate_mean=group_mean,
                        estimate_std=group_std,
                        raw_data_colors=hist_color[groupes.index(groupe)],
                        fitted_distribution_color=fit_color[groupes.index(groupe)],
                        title='',
                        xtitle='Functional connectivity', ytitle='Proportions of subjects',
                        legend_fitted='{} distribution'.format(groupe),
                        legend_data=groupe, display_fit='yes', ms=3.5, line_width=2.5)
                    plt.axvline(x=group_mean, color=fit_color[groupes.index(groupe)],
                                linewidth=2)
                    plt.title('Mean left connectivity distribution for the {} network'.format(network))

                pdf.savefig()
                plt.show()


# Compute the intra-network connectivity for the right hemisphere in
# groups
right_intra_network_connectivity_dict, \
    right_network_dict, right_network_labels_list, right_network_label_colors = \
    ccm.intra_network_functional_connectivity(
        subjects_individual_matrices_dictionnary=right_connectivity_matrices,
        groupes=groupes, kinds=kinds,
        atlas_file=atlas_excel_file,
        sheetname='Hemisphere_regions',
        roi_indices_column_name='index',
        network_column_name='network',
        color_of_network_column='Color')

# Estimate mean and standard deviation of right intra network parameters
right_intra_network_distribution_parameters = dict.fromkeys(network_labels_list)
for network in network_labels_list:
    if network not in ['Basal_Ganglia', 'Precuneus', 'Auditory']:
        right_intra_network_distribution_parameters[network] = dict.fromkeys(groupes)
        for groupe in groupes:
            right_intra_network_distribution_parameters[network][groupe] = dict.fromkeys(kinds)
            for kind in kinds:
                # Stack the mean homotopic connectivity of each subject for the current group
                subjects_mean_right_intra_network_connectivity = np.array(
                    [right_intra_network_connectivity_dict[groupe][subject][kind][network][
                         'network connectivity strength']
                     for subject in right_intra_network_connectivity_dict[groupe].keys()])
                # Estimate the mean and std assuming a Gaussian behavior
                subjects_mean_right_intra_network_connectivity_, mean_intra_estimation, std_intra_estimation = \
                    parametric_tests.functional_connectivity_distribution_estimation(
                        subjects_mean_right_intra_network_connectivity)
                # Fill a dictionary saving the results for each groups and kind
                right_intra_network_distribution_parameters[network][groupe][kind] = {
                    'subjects mean right intra connectivity': subjects_mean_right_intra_network_connectivity_,
                    'right intra distribution mean': mean_intra_estimation,
                    'right intra distribution standard deviation': std_intra_estimation}

for kind in kinds:
    with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                           kind + '_right_intra_network_connectivity_distribution.pdf')) as pdf:

        for network in network_labels_list:
            if network not in ['Basal_Ganglia', 'Precuneus', 'Auditory']:
                plt.figure(constrained_layout=True)
                for groupe in groupes:
                    group_connectivity = right_intra_network_distribution_parameters[network][groupe][kind][
                        'subjects mean right intra connectivity']
                    group_mean = right_intra_network_distribution_parameters[network][groupe][kind][
                        'right intra distribution mean']
                    group_std =  right_intra_network_distribution_parameters[network][groupe][kind][
                        'right intra distribution standard deviation']
                    display.display_gaussian_connectivity_fit(
                        vectorized_connectivity=group_connectivity,
                        estimate_mean=group_mean,
                        estimate_std=group_std,
                        raw_data_colors=hist_color[groupes.index(groupe)],
                        fitted_distribution_color=fit_color[groupes.index(groupe)],
                        title='',
                        xtitle='Functional connectivity', ytitle='Proportions of subjects',
                        legend_fitted='{} distribution'.format(groupe),
                        legend_data=groupe, display_fit='yes', ms=3.5, line_width=2.5)
                    plt.axvline(x=group_mean, color=fit_color[groupes.index(groupe)],
                                linewidth=2)
                    plt.title('Mean right connectivity distribution for the {} network'.format(network))

                pdf.savefig()
                plt.show()

parametric_tests.two_sample_t_test_(connectivity_dictionnary_=right_connectivity_parameters,
                                    groupes=groups,
                                    kinds=kinds,
                                    field='subjects mean right connectivity',
                                    contrast=[1.0, -1.0],
                                    paired=False)
# T-test
intra_network_t_test = parametric_tests.intra_network_two_samples_t_test(
    intra_network_connectivity_dictionary=homotopic_intranetwork_d,
    groupes=['wotopi_all', 'controls_topiramate'],
    kinds=kinds,
    contrast=[1.0, -1.0],
    network_labels_list=network_labels_list,
    paired=False)

intra_network_homotopic_t_test = parametric_tests.intra_network_two_samples_t_test(
    intra_network_connectivity_dictionary=right_intra_network_connectivity_dict,
    groupes=['wotopi_all', 'controls_topiramate'],
    kinds=kinds,
    contrast=[1.0, -1.0],
    network_labels_list=right_network_labels_list,
    paired=False)

# Classification
subjects_connectivity_matrices = folders_and_files_management.load_object(
    full_path_to_object='/media/db242421/db242421_data/ConPagnon_data/topi_wotopi_paired/topiramate_connectivity_matrices.pkl'
)
groups = list(subjects_connectivity_matrices.keys())
features_labels = np.hstack((1*np.ones(len(subjects_connectivity_matrices[groups[0]].keys())),
                             2*np.ones(len(subjects_connectivity_matrices[groups[1]].keys()))))
features = sym_matrix_to_vec(symmetric=np.array([subjects_connectivity_matrices[group][s]['tangent']
                                                for group in groups
                                                for s in list(subjects_connectivity_matrices[group].keys())]),
                             discard_diagonal=True)

from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit

scores = cross_val_score(estimator=LinearSVC(), X=features, y=features_labels,
                         cv=StratifiedShuffleSplit(n_splits=10000), n_jobs=10)
scores = np.array(scores)
mean_accuracy = scores.mean()
std_accuracy = scores.std()
print('Mean accuracy of {} % +- {} %'.format(mean_accuracy*100, std_accuracy*100))