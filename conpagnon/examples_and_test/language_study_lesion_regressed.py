"""
 Created by db242421 at 13/12/18

 """

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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

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
Created on Mon Sep 18 16:37:22 2017

@author: Dhaif BEKHA(dhaif.bekha@cea.fr)

Analysis of AVCnn cohort : patients with typical and atypical language
and controls. 
Same Atlas for all, but lesion driven artifact was removed in patients after 
pre-processing.

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
groupes = ['PALr', 'PNLr', 'TDC']
# The root fmri data directory containing all the fmri files directories
root_fmri_data_directory = \
    '/media/db242421/db242421_data/ConPagnon_data/fmri_images'
# Check all functional files existence
folders_and_files_management.check_directories_existence(
    root_directory=root_fmri_data_directory,
    directories_list=groupes)

# File extension for individual atlases images, and corresponding
# text labels files.
individual_atlas_file_extension = '*.nii'
individual_atlas_labels_extension = '*.csv'

# Full path, including extension, to the text file containing
# all the subject identifiers.
subjects_ID_data_path = \
    '/media/db242421/db242421_data/ConPagnon_data/text_data/acm_patients_and_controls_all.txt'

organised_data = data_architecture.fetch_data(
    subjects_id_data_path=subjects_ID_data_path,
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groupes,
    individual_counfounds_directory=None
)

# kinds
kinds = ['correlation', 'tangent']

# Nilearn cache directory
nilearn_cache_directory = '/media/db242421/db242421_data/ConPagnon/nilearn_cache'

# Repetition time between time points in the fmri acquisition, usually 2.4s or 2.5s
repetition_time = 2.5

# output csv directory
output_csv_directory_path = \
    '/media/db242421/db242421_data/ConPagnon_data/language_study_ANOVA_ACM_controls_lesion_regressed'
output_csv_directory = data_management.create_directory(directory=output_csv_directory_path, erase_previous=True)

# Figure directory which can be useful for illustrating
output_figure_directory_path = os.path.join(output_csv_directory, 'figures')
output_figure_directory = data_management.create_directory(directory=output_figure_directory_path,
                                                           erase_previous=True)
# Cohort behavioral data
cohort_excel_file_path = '/media/db242421/db242421_data/ConPagnon_data/regression_data/regression_data.xlsx'
behavioral_data = data_management.read_excel_file(excel_file_path=cohort_excel_file_path,
                                                  sheetname='cohort_functional_data')

# Clean the behavioral data by keeping only the subjects in the study
subjects_list = open(subjects_ID_data_path).read().split()
behavioral_data = behavioral_data.loc[subjects_list]
# save a CSV file format for the behavioral data
behavioral_data.to_csv(os.path.join(output_csv_directory, 'behavioral_data.csv'))

times_series = ccm.time_series_extraction(
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groupes,
    subjects_id_data_path=subjects_ID_data_path,
    reference_atlas=os.path.join(atlas_folder, atlas_name),
    group_data=organised_data,
    repetition_time=repetition_time,
    nilearn_cache_directory=nilearn_cache_directory
)

# Compute connectivity matrices for multiple metrics
# Covariance estimator
covariance_estimator = covariance.LedoitWolf()

# Computing for each metric, and each subjects in each groups,
# the connectivity matrices using the computed time series.
subjects_connectivity_matrices = ccm.individual_connectivity_matrices(
    time_series_dictionary=times_series,
    kinds=kinds, covariance_estimator=covariance_estimator,
    vectorize=False, z_fisher_transform=False)

# Computing for each metric, and each subjects in each groups,
# the fisher transform connectivity matrices using the computed time series.
Z_subjects_connectivity_matrices = ccm.individual_connectivity_matrices(
    time_series_dictionary=times_series,
    kinds=kinds, covariance_estimator=covariance_estimator,
    vectorize=False, z_fisher_transform=True,
    discarding_diagonal=False)

# Computing for each metric, and each groups the mean connectivity matrices,
# from the z fisher transform matrices.
Z_mean_groups_connectivity_matrices = ccm.group_mean_connectivity(
    subjects_connectivity_matrices=Z_subjects_connectivity_matrices,
    kinds=kinds)


# Fitting a gaussian distribution mean connectivity matrices in each groups, after a Z fisher transform.
parameters_estimation = parametric_tests.mean_functional_connectivity_distribution_estimation(
                        mean_groups_connectivity_matrices=Z_mean_groups_connectivity_matrices)

# Plot of mean connectivity distribution in each group, with the gaussian fit in overlay
fitted_dist_color1, fitted_dist_color2, fitted_dist_color3 = 'blue', 'red', 'green'
raw_data_color1, raw_data_color2, raw_data_color3 = 'blue', 'red', 'green'
hist_color = ['blue', 'red', 'green']
fit_color = ['blue', 'red', 'green']
# Display results
with backend_pdf.PdfPages(os.path.join(output_figure_directory, 'overall_connectivity_distribution.pdf')) as pdf:
    for kind in kinds:
        plt.figure()
        for groupe in groupes:
            display.display_gaussian_connectivity_fit(
                vectorized_connectivity=parameters_estimation[groupe][kind]['vectorized connectivity'],
                estimate_mean=parameters_estimation[groupe][kind]['mean'],
                estimate_std=parameters_estimation[groupe][kind]['std'],
                raw_data_colors=hist_color[groupes.index(groupe)],
                fitted_distribution_color=fit_color[groupes.index(groupe)],
                title='Overall functional connectivity distribution  for {}'.format(kind),
                xtitle='Functional connectivity coefficient', ytitle='proportion of edges',
                legend_fitted='{} distribution'.format(groupe),
                legend_data=groupe, display_fit='no', ms=0.5, line_width=1)
        pdf.savefig()
        plt.show()
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
    subjects_individual_matrices_dictionnary=Z_subjects_connectivity_matrices,
    connectivity_coefficient_position=homotopic_roi_indices, kinds=kinds,
    groupes=groupes)

# Compute mean and standard deviation assuming gaussian behavior
homotopic_distribution_parameters = dict.fromkeys(groupes)
for groupe in groupes:
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


with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                       'mean_homotopic_connectivity_distribution.pdf')) as pdf:
    for kind in kinds:
        plt.figure(constrained_layout=True)
        for groupe in groupes:
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
                raw_data_colors=hist_color[groupes.index(groupe)],
                fitted_distribution_color=fit_color[groupes.index(groupe)],
                title='Whole brain mean homotopic connectivity distribution for {}'.format(kind),
                xtitle='Functional connectivity coefficient', ytitle='Proportion of subjects',
                legend_fitted='{}  distribution'.format(groupe),
                legend_data=groupe, display_fit='yes', ms=3.5, line_width=2.5)
            plt.axvline(x=group_mean, color=fit_color[groupes.index(groupe)],
                        linewidth=2)
        pdf.savefig()
        plt.show()
        plt.close()

# Compute whole brain mean connectivity for each subjects i.e  the mean of the flatten subjects
# connectivity matrices
whole_brain_mean_connectivity = ccm.mean_of_flatten_connectivity_matrices(
    subjects_individual_matrices_dictionary=Z_subjects_connectivity_matrices, groupes=groupes, kinds=kinds)

# Estimate mean and standard deviation of whole brain mean connectivity for each group and kinds
whole_brain_mean_connectivity_parameters = dict.fromkeys(groupes)
for groupe in groupes:
    whole_brain_mean_connectivity_parameters[groupe] = dict.fromkeys(kinds)
    for kind in kinds:
        # Stack the mean homotopic connectivity of each subject for the current group
        subjects_mean_whole_brain_connectivity = np.array(
            [whole_brain_mean_connectivity[groupe][subject][kind]['mean connectivity']
             for subject in whole_brain_mean_connectivity[groupe].keys()])
        # Estimate the mean and std assuming a Gaussian behavior
        subjects_mean_whole_brain_connectivity_, mean_estimation, std_estimation = \
            parametric_tests.functional_connectivity_distribution_estimation(
                subjects_mean_whole_brain_connectivity)
        # Fill a dictionnary saving the results for each groups and kind
        whole_brain_mean_connectivity_parameters[groupe][kind] = {
            'subjects mean whole brain connectivity': subjects_mean_whole_brain_connectivity_,
            'whole brain distribution mean': mean_estimation,
            'whole brain distribution standard deviation': std_estimation}

# Display results: whole brain mean connectivity
with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                       'whole_brain_mean_connectivity_distribution.pdf')) as pdf:
    for kind in kinds:
        plt.figure()
        for groupe in groupes:
            group_connectivity = whole_brain_mean_connectivity_parameters[groupe][kind][
                'subjects mean whole brain connectivity']
            group_mean = whole_brain_mean_connectivity_parameters[groupe][kind][
                'whole brain distribution mean']
            group_std = whole_brain_mean_connectivity_parameters[groupe][kind][
                'whole brain distribution standard deviation']
            display.display_gaussian_connectivity_fit(
                vectorized_connectivity=group_connectivity,
                estimate_mean=group_mean,
                estimate_std=group_std,
                raw_data_colors=hist_color[groupes.index(groupe)],
                fitted_distribution_color=fit_color[groupes.index(groupe)],
                title='Whole brain mean connectivity distribution for {}'.format(kind),
                xtitle='Functional connectivity coefficient', ytitle='Proportion of subjects',
                legend_fitted='{} distribution'.format(groupe),
                legend_data=groupe, display_fit='yes', ms=3.5, line_width=2.5)
            plt.axvline(x=group_mean, color=fit_color[groupes.index(groupe)],
                        linewidth=2)
        pdf.savefig()
        plt.show()

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
                homotopic_intranetwork_d[groupe][subject][kind][network] = \
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

# Save the whole brain intra-network connectivity
data_management.csv_from_intra_network_dictionary(subjects_dictionary=intra_network_connectivity_dict,
                                                  groupes=groupes, kinds=kinds,
                                                  network_labels_list=network_labels_list,
                                                  field_to_write='network connectivity strength',
                                                  output_directory=output_csv_directory,
                                                  csv_prefix='intra')

# Save the whole brain intra-network homotopic connectivity
data_management.csv_from_intra_network_dictionary(subjects_dictionary=homotopic_intranetwork_d,
                                                  groupes=groupes, kinds=kinds,
                                                  network_labels_list=network_labels_list,
                                                  field_to_write='network connectivity strength',
                                                  output_directory=output_csv_directory,
                                                  csv_prefix='intra_homotopic')
# Save the whole brain mean connectivity
for group in groupes:
    for kind in kinds:
        data_management.csv_from_dictionary(subjects_dictionary=whole_brain_mean_connectivity,
                                            groupes=groupes,
                                            kinds=kinds,
                                            field_to_write='mean connectivity',
                                            header=['subjects', 'mean_connectivity'],
                                            csv_filename='mean_connectivity.csv',
                                            output_directory=os.path.join(output_csv_directory, kind),
                                            delimiter=',')

# Save the whole brain homotopic connectivity
for group in groupes:
    for kind in kinds:
        data_management.csv_from_dictionary(subjects_dictionary=homotopic_connectivity,
                                            groupes=[group],
                                            kinds=[kind],
                                            field_to_write='mean connectivity',
                                            header=['subjects', 'mean_homotopic'],
                                            csv_filename='mean_homotopic.csv',
                                            output_directory=os.path.join(output_csv_directory, kind),
                                            delimiter=',')

# Overall and Within network, ipsilesional and contralesional connectivity differences
# Compute the connectivity matrices dictionary with factor as keys.
group_by_factor_subjects_connectivity, population_df_by_factor, factor_keys = \
    dictionary_operations.groupby_factor_connectivity_matrices(population_data_file=cohort_excel_file_path,
                                                               sheetname='cohort_functional_data',
                                                               subjects_connectivity_matrices_dictionnary=Z_subjects_connectivity_matrices,
                                                               groupes=['PALr', 'PNLr'],
                                                               factors=['Lesion', 'langage_clinique'],
                                                               drop_subjects_list=None)

# Create ipsilesional and contralesional dictionary
ipsi_dict = {}
contra_dict = {}

# Fetch left and right connectivity for each group attribute keys
left_connectivity = ccm.extract_sub_connectivity_matrices(
    subjects_connectivity_matrices=group_by_factor_subjects_connectivity,
    kinds=kinds, regions_index=left_regions_indices)
right_connectivity = ccm.extract_sub_connectivity_matrices(
    subjects_connectivity_matrices=group_by_factor_subjects_connectivity,
    kinds=kinds, regions_index=right_regions_indices)

# Fill ipsi-lesional and contra-lesional dictionary
for attribute in factor_keys:
    subject_for_this_attribute = list(population_df_by_factor[attribute])
    ipsi_dict[attribute] = dict.fromkeys(subject_for_this_attribute)
    contra_dict[attribute] = dict.fromkeys(subject_for_this_attribute)
    for s in subject_for_this_attribute:
        if attribute[0] == 'D':
            # Fill the ipsi-lesional dictionary for a right lesion with the
            # right connectivity matrices for the current attribute
            ipsi_dict[attribute][s] = right_connectivity[attribute][s]
            # Fill the contra-lesional dictionary for a right lesion with the
            # left connectivity matrices for the current attribute
            contra_dict[attribute][s] = left_connectivity[attribute][s]
        elif attribute[0] == 'G':
            # Fill the ipsi-lesional dictionary for a left lesion with the
            # left connectivity matrices for the current attribute
            ipsi_dict[attribute][s] = left_connectivity[attribute][s]
            # Fill the contra-lesional dictionary for a left lesion with the
            # right connectivity matrices for the current attribute
            contra_dict[attribute][s] = right_connectivity[attribute][s]

# Construct a corresponding ipsilesional and contralesional dictionary for controls
n_left_tot = len(ipsi_dict[('G', 'A')]) + len(ipsi_dict[('G', 'N')])
n_right_tot = len(ipsi_dict[('D', 'A')]) + len(ipsi_dict[('D', 'N')])

# Compute the percentage of right lesion and left lesion
n_total_patients = n_left_tot + n_right_tot
percent_left_lesioned = n_left_tot / n_total_patients
percent_right_lesioned = n_right_tot / n_total_patients

# total number of controls
n_controls = len(Z_subjects_connectivity_matrices['TDC'].keys())
# Number of left and right impaired hemisphere to pick randomly in the controls group
n_left_hemisphere = ceil(percent_left_lesioned * n_controls)
n_right_hemisphere = n_controls - n_left_hemisphere

# Random pick of n_left_tot in proportion to the number of left lesion in controls group
n_left_ipsilesional_controls, n_left_ipsi_controls_ids = \
    dictionary_operations.random_draw_of_connectivity_matrices(
        subjects_connectivity_dictionary=Z_subjects_connectivity_matrices,
        groupe='TDC', n_matrices=int(n_left_hemisphere),
        extract_kwargs={'kinds': kinds, 'regions_index': left_regions_indices,
                        'discard_diagonal': False,
                        'vectorize': False})
# Now, pick the right hemisphere in the left-out pool of subjects
leftout_controls_id = \
    list(set(Z_subjects_connectivity_matrices['TDC'].keys()) - set(n_left_ipsi_controls_ids))

n_right_ipsilesional_controls, n_right_ipsi_controls_ids = \
    dictionary_operations.random_draw_of_connectivity_matrices(
        subjects_connectivity_dictionary=Z_subjects_connectivity_matrices,
        groupe='TDC', n_matrices=int(n_right_hemisphere),
        extract_kwargs={'kinds': kinds, 'regions_index': right_regions_indices,
                        'discard_diagonal': False,
                        'vectorize': False},
        subjects_id_list=leftout_controls_id)

# Merge the right and left ipsilesional to have the 'ipsilesional' controls dictionary
ipsilesional_controls_dictionary = dictionary_operations.merge_dictionary(
    new_key='TDC',
    dict_list=[n_left_ipsilesional_controls['TDC'],
               n_right_ipsilesional_controls['TDC']])

# In the same, we have to generate, a "contralesional" dictionary for the
# controls group

# contralesional dictionary for left lesions, we fetch all RIGHT side roi this time
n_left_contralesional_controls, n_left_contra_controls_ids = \
    dictionary_operations.random_draw_of_connectivity_matrices(
        subjects_connectivity_dictionary=Z_subjects_connectivity_matrices,
        groupe='TDC', n_matrices=int(n_left_hemisphere),
        extract_kwargs={'kinds': kinds, 'regions_index': right_regions_indices,
                        'discard_diagonal': False,
                        'vectorize': False})

# In the left-out pool of subjects, we fetch the LEFT side rois, for the
# contralesional side of right lesion.
leftout_controls_id = \
    list(set(Z_subjects_connectivity_matrices['TDC'].keys()) - set(n_left_contra_controls_ids))

n_right_contralesional_controls, n_right_contra_controls_ids = \
    dictionary_operations.random_draw_of_connectivity_matrices(
        subjects_connectivity_dictionary=Z_subjects_connectivity_matrices,
        groupe='TDC', n_matrices=int(n_right_hemisphere),
        extract_kwargs={'kinds': kinds, 'regions_index': left_regions_indices,
                        'discard_diagonal': False,
                        'vectorize': False}, subjects_id_list=leftout_controls_id)

# Merge the right and left ipsilesional to have the 'ipsilesional' controls dictionary
contralesional_controls_dictionary = dictionary_operations.merge_dictionary(
    new_key='TDC',
    dict_list=[n_left_contralesional_controls['TDC'],
               n_right_contralesional_controls['TDC']])

# Finally, we have to merge ipsilesional/contralesional dictionaries of the different group
# Merge the dictionary to build the overall contra and ipsi-lesional
# subjects connectivity matrices dictionary

# First, the two group of patients
ipsilesional_patients_connectivity_matrices = {
    'PALr': {**ipsi_dict[('G', 'A')], **ipsi_dict[('D', 'A')]},
    'PNLr': {**ipsi_dict[('G', 'N')], **ipsi_dict[('D', 'N')]}
}

contralesional_patients_connectivity_matrices = {
    'PNLr': {**contra_dict[('G', 'N')], **contra_dict[('D', 'N')]},
    'PALr': {**contra_dict[('G', 'A')], **contra_dict[('D', 'A')]}
}

# Merged overall patients and controls dictionaries
ipsilesional_subjects_connectivity_matrices = dictionary_operations.merge_dictionary(
    dict_list=[ipsilesional_controls_dictionary,
               ipsilesional_patients_connectivity_matrices]
)
contralesional_subjects_connectivity_matrices = dictionary_operations.merge_dictionary(
    dict_list=[contralesional_controls_dictionary,
               contralesional_patients_connectivity_matrices]
)

# Estimate mean and std of ipsilesional connectivity assuming gaussian behavior
# Compute of mean ipsilesional distribution
ipsilesional_mean_connectivity = ccm.mean_of_flatten_connectivity_matrices(
    subjects_individual_matrices_dictionary=ipsilesional_subjects_connectivity_matrices,
    groupes=groupes,
    kinds=kinds)
contralesional_mean_connectivity = ccm.mean_of_flatten_connectivity_matrices(
    subjects_individual_matrices_dictionary=contralesional_subjects_connectivity_matrices,
    groupes=groupes,
    kinds=kinds)

# Estimate mean and standard deviation of mean ipsilesional connectivity
ipsilesional_mean_connectivity_parameters = dict.fromkeys(groupes)
for groupe in groupes:
    ipsilesional_mean_connectivity_parameters[groupe] = dict.fromkeys(kinds)
    for kind in kinds:
        # Stack the mean homotopic connectivity of each subject for the current group
        subjects_mean_ipsi_connectivity = np.array(
            [ipsilesional_mean_connectivity[groupe][subject][kind]['mean connectivity']
             for subject in ipsilesional_mean_connectivity[groupe].keys()])
        # Estimate the mean and std assuming a Gaussian behavior
        subjects_mean_ipsi_connectivity_, mean_estimation, std_estimation = \
            parametric_tests.functional_connectivity_distribution_estimation(subjects_mean_ipsi_connectivity)
        # Fill a dictionary saving the results for each groups and kind
        ipsilesional_mean_connectivity_parameters[groupe][kind] = {
            'subjects mean ipsilesional connectivity': subjects_mean_ipsi_connectivity_,
            'ipsilesional distribution mean': mean_estimation,
            'ipsilesional distribution standard deviation': std_estimation}

# Plot the distribution of mean ipsilesional connectivity assuming gaussian
# behavior
with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                       'ipsilesional_mean_connectivity_distribution.pdf')) as pdf:
    for kind in kinds:
        plt.figure()
        for groupe in groupes:
            group_connectivity = ipsilesional_mean_connectivity_parameters[groupe][kind][
                'subjects mean ipsilesional connectivity']
            group_mean = ipsilesional_mean_connectivity_parameters[groupe][kind][
                'ipsilesional distribution mean']
            group_std = ipsilesional_mean_connectivity_parameters[groupe][kind][
                'ipsilesional distribution standard deviation']
            display.display_gaussian_connectivity_fit(
                vectorized_connectivity=group_connectivity,
                estimate_mean=group_mean,
                estimate_std=group_std,
                raw_data_colors=hist_color[groupes.index(groupe)],
                fitted_distribution_color=fit_color[groupes.index(groupe)],
                title='Ipsilesional mean connectivity distribution  for {}'.format(kind),
                xtitle='Functional connectivity coefficient', ytitle='Proportion of subjects',
                legend_fitted='{} distribution'.format(groupe),
                legend_data=groupe, display_fit='yes', ms=3.5, line_width=2.5)
            plt.axvline(x=group_mean, color=fit_color[groupes.index(groupe)],
                        linewidth=2)
        pdf.savefig()
        plt.show()

# Estimate mean and standard deviation of mean contralesional connectivity
contralesional_mean_connectivity_parameters = dict.fromkeys(groupes)
for groupe in groupes:
    contralesional_mean_connectivity_parameters[groupe] = dict.fromkeys(kinds)
    for kind in kinds:
        # Stack the mean homotopic connectivity of each subject for the current group
        subjects_mean_contra_connectivity = np.array(
            [contralesional_mean_connectivity[groupe][subject][kind]['mean connectivity']
             for subject in contralesional_mean_connectivity[groupe].keys()])
        # Estimate the mean and std assuming a Gaussian behavior
        subjects_mean_contra_connectivity_, mean_estimation, std_estimation = \
            parametric_tests.functional_connectivity_distribution_estimation(subjects_mean_contra_connectivity)
        # Fill a dictionary saving the results for each groups and kind
        contralesional_mean_connectivity_parameters[groupe][kind] = {
            'subjects mean contralesional connectivity': subjects_mean_contra_connectivity_,
            'contralesional distribution mean': mean_estimation,
            'contralesional distribution standard deviation': std_estimation}

# Plot the distribution of mean ipsilesional connectivity assuming gaussian
# behavior
with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                       'contralesional_mean_connectivity_distribution.pdf')) as pdf:
    for kind in kinds:
        plt.figure()
        for groupe in groupes:
            group_connectivity = contralesional_mean_connectivity_parameters[groupe][kind][
                'subjects mean contralesional connectivity']
            group_mean = contralesional_mean_connectivity_parameters[groupe][kind][
                'contralesional distribution mean']
            group_std = contralesional_mean_connectivity_parameters[groupe][kind][
                'contralesional distribution standard deviation']
            display.display_gaussian_connectivity_fit(
                vectorized_connectivity=group_connectivity,
                estimate_mean=group_mean,
                estimate_std=group_std,
                raw_data_colors=hist_color[groupes.index(groupe)],
                fitted_distribution_color=fit_color[groupes.index(groupe)],
                title='Contralesional mean connectivity distribution for {}'.format(kind),
                xtitle='Functional connectivity coefficient', ytitle='Proportion of subjects',
                legend_fitted='{} distribution'.format(groupe),
                legend_data=groupe, display_fit='yes', ms=3.5, line_width=2)
            plt.axvline(x=group_mean, color=fit_color[groupes.index(groupe)],
                        linewidth=2)
        pdf.savefig()
        plt.show()

# Compute the intra-network connectivity for the ipsilesional hemisphere in
# groups
ipsilesional_intra_network_connectivity_dict, \
ipsi_network_dict, ipsi_network_labels_list, ipsi_network_label_colors = \
    ccm.intra_network_functional_connectivity(
        subjects_individual_matrices_dictionnary=ipsilesional_subjects_connectivity_matrices,
        groupes=groupes, kinds=kinds,
        atlas_file=atlas_excel_file,
        sheetname='Hemisphere_regions',
        roi_indices_column_name='index',
        network_column_name='network',
        color_of_network_column='Color')

# Estimate mean and standard deviation of ipsilesional intra network parameters
ipsi_intra_network_distribution_parameters = dict.fromkeys(network_labels_list)
for network in network_labels_list:
    if network not in ['Basal_Ganglia', 'Precuneus', 'Auditory']:
        ipsi_intra_network_distribution_parameters[network] = dict.fromkeys(groupes)
        for groupe in groupes:
            ipsi_intra_network_distribution_parameters[network][groupe] = dict.fromkeys(kinds)
            for kind in kinds:
                # Stack the mean homotopic connectivity of each subject for the current group
                subjects_mean_ipsi_intra_network_connectivity = np.array(
                    [ipsilesional_intra_network_connectivity_dict[groupe][subject][kind][network][
                         'network connectivity strength']
                     for subject in ipsilesional_intra_network_connectivity_dict[groupe].keys()])
                # Estimate the mean and std assuming a Gaussian behavior
                subjects_mean_ipsi_intra_network_connectivity_, mean_intra_estimation, std_intra_estimation = \
                    parametric_tests.functional_connectivity_distribution_estimation(
                        subjects_mean_ipsi_intra_network_connectivity)
                # Fill a dictionary saving the results for each groups and kind
                ipsi_intra_network_distribution_parameters[network][groupe][kind] = {
                    'subjects mean ipsi intra connectivity': subjects_mean_ipsi_intra_network_connectivity_,
                    'ipsi intra distribution mean': mean_intra_estimation,
                    'ipsi intra distribution standard deviation': std_intra_estimation}

for kind in kinds:
    with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                           kind + '_ipsi_intra_network_connectivity_distribution.pdf')) as pdf:

        for network in network_labels_list:
            if network not in ['Basal_Ganglia', 'Precuneus', 'Auditory']:
                plt.figure(constrained_layout=True)
                for groupe in groupes:
                    group_connectivity = ipsi_intra_network_distribution_parameters[network][groupe][kind][
                        'subjects mean ipsi intra connectivity']
                    group_mean = ipsi_intra_network_distribution_parameters[network][groupe][kind][
                        'ipsi intra distribution mean']
                    group_std = ipsi_intra_network_distribution_parameters[network][groupe][kind][
                        'ipsi intra distribution standard deviation']
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
                    plt.title('Mean ipsilesional connectivity distribution for the {} network'.format(network))

                pdf.savefig()
                plt.show()

# Compute the intra-network connectivity for the contralesional hemisphere in
# groups
contralesional_intra_network_connectivity_dict, \
contra_network_dict, contra_network_labels_list, contra_network_label_colors = \
    ccm.intra_network_functional_connectivity(
        subjects_individual_matrices_dictionnary=contralesional_subjects_connectivity_matrices,
        groupes=groupes, kinds=kinds,
        atlas_file=atlas_excel_file,
        sheetname='Hemisphere_regions',
        roi_indices_column_name='index',
        network_column_name='network',
        color_of_network_column='Color')

# Estimate mean and standard deviation of contralesional intra network parameters
contra_intra_network_distribution_parameters = dict.fromkeys(network_labels_list)
for network in network_labels_list:
    if network not in ['Basal_Ganglia', 'Precuneus', 'Auditory']:
        contra_intra_network_distribution_parameters[network] = dict.fromkeys(groupes)
        for groupe in groupes:
            contra_intra_network_distribution_parameters[network][groupe] = dict.fromkeys(kinds)
            for kind in kinds:
                # Stack the mean homotopic connectivity of each subject for the current group
                subjects_mean_contra_intra_network_connectivity = np.array(
                    [contralesional_intra_network_connectivity_dict[groupe][subject][kind][network][
                         'network connectivity strength']
                     for subject in contralesional_intra_network_connectivity_dict[groupe].keys()])
                # Estimate the mean and std assuming a Gaussian behavior
                subjects_mean_contra_intra_network_connectivity_, mean_intra_estimation, std_intra_estimation = \
                    parametric_tests.functional_connectivity_distribution_estimation(
                        subjects_mean_contra_intra_network_connectivity)
                # Fill a dictionary saving the results for each groups and kind
                contra_intra_network_distribution_parameters[network][groupe][kind] = {
                    'subjects mean contra intra connectivity': subjects_mean_contra_intra_network_connectivity_,
                    'contra intra distribution mean': mean_intra_estimation,
                    'contra intra distribution standard deviation': std_intra_estimation}

for kind in kinds:
    with backend_pdf.PdfPages(os.path.join(output_figure_directory,
                                           kind + '_contra_intra_network_connectivity_distribution.pdf')) as pdf:

        for network in network_labels_list:
            if network not in ['Basal_Ganglia', 'Precuneus', 'Auditory']:
                plt.figure(constrained_layout=True)
                for groupe in groupes:
                    group_connectivity = contra_intra_network_distribution_parameters[network][groupe][kind][
                        'subjects mean contra intra connectivity']
                    group_mean = contra_intra_network_distribution_parameters[network][groupe][kind][
                        'contra intra distribution mean']
                    group_std = contra_intra_network_distribution_parameters[network][groupe][kind][
                        'contra intra distribution standard deviation']
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
                    plt.title('Mean contralesional connectivity distribution for the {} network'.format(network))

                pdf.savefig()
                plt.show()

# Save the ipsilesional intra-network connectivity for each groups and network
data_management.csv_from_intra_network_dictionary(subjects_dictionary=ipsilesional_intra_network_connectivity_dict,
                                                  groupes=groupes,
                                                  kinds=kinds,
                                                  network_labels_list=ipsi_network_labels_list,
                                                  field_to_write='network connectivity strength',
                                                  output_directory=output_csv_directory,
                                                  csv_prefix='ipsi_intra',
                                                  delimiter=',')

# Save the contra-lesional intra-network connectivity for each group and networks
data_management.csv_from_intra_network_dictionary(subjects_dictionary=contralesional_intra_network_connectivity_dict,
                                                  groupes=groupes,
                                                  kinds=kinds,
                                                  network_labels_list=contra_network_labels_list,
                                                  field_to_write='network connectivity strength',
                                                  output_directory=output_csv_directory,
                                                  csv_prefix='contra_intra',
                                                  delimiter=',')

# Save the overall mean ipsilesional connectivity for each group and kind
for group in groupes:
    for kind in kinds:
        data_management.csv_from_dictionary(subjects_dictionary=ipsilesional_mean_connectivity,
                                            groupes=[group],
                                            kinds=[kind],
                                            field_to_write='mean connectivity',
                                            header=['subjects', 'mean_ipsi'],
                                            csv_filename='mean_ipsilesional.csv',
                                            output_directory=os.path.join(output_csv_directory, kind),
                                            delimiter=',')

        # Save the overall mean contralesional connectivity for each group and kind
        data_management.csv_from_dictionary(subjects_dictionary=contralesional_mean_connectivity,
                                            groupes=[group],
                                            kinds=[kind],
                                            field_to_write='mean connectivity',
                                            header=['subjects', 'mean_contra'],
                                            csv_filename='mean_contralesional.csv',
                                            output_directory=os.path.join(output_csv_directory, kind),
                                            delimiter=',')
