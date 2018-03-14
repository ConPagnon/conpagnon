import importlib
from data_handling import atlas, data_architecture, dictionary_operations
from utils import folders_and_files_management
from computing import compute_connectivity_matrices as ccm
from sklearn import covariance
from machine_learning import classification
from connectivity_statistics import parametric_tests
from plotting import display
from matplotlib.backends import backend_pdf
import os
import errno
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests
from random import sample
from math import ceil
# Reload all module
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

@author: db242421 (dhaif.bekha@cea.fr)

This files is useful to construct you're analysis.
"""


# Atlas set up
atlas_folder = 'C:\\Users\\Dhaïf\\Documents\\atlas_AVCnn'
atlas_name ='atlas4D_2.nii'
monAtlas = atlas.Atlas(path=atlas_folder,
                       name=atlas_name)
# Atlas path
atlas_path = monAtlas.fetch_atlas()
# Read labels regions files
labels_text_file = 'C:\\Users\\Dhaïf\\Documents\\atlas_AVCnn\\atlas4D_2_labels.csv'
labels_regions = monAtlas.GetLabels(labels_text_file)
# User defined colors for labels ROIs regions
colors = ['navy', 'sienna', 'orange', 'orchid', 'indianred', 'olive', 
          'goldenrod', 'turquoise', 'darkslategray', 'limegreen', 'black',
          'lightpink']
# Number of regions in each user defined networks
networks = [2, 10, 2, 6, 10, 2, 8, 6, 8, 8, 6, 4]
# Transformation of string colors list to an RGB color array, 
# all colors ranging between 0 and 1.
labels_colors = (1./255)*monAtlas.UserLabelsColors(networks=networks,
                                                   colors=colors)
# Fetch nodes coordinates
atlas_nodes = monAtlas.GetCenterOfMass()
# Fetch number of nodes in the parcellation
n_nodes = monAtlas.GetRegionNumbers()

# Groups name to include in the study
groupes = ['L_Clin_Atyp_pat', 'L_Clin_Typ_pat', 'controls']
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
    '/media/db242421/db242421_data/ConPagnon_data/text_data/subjects_ID.txt'

# Full to the following directories: individual atlases images, 
# individual text labels files, and individual confounds directories.
individual_atlases_directory = \
    '/media/db242421/db242421_data/ConPagnon_data/atlas/individual_atlases_V2'
individual_atlases_labels_directory = \
    '/media/db242421/db242421_data/ConPagnon_data/atlas/individual_atlases_labels_V2'
individual_confounds_directory = \
    '/media/db242421/db242421_data/ConPagnon_data/regressors'

# Set the type I error (alpha level), usually .05.
alpha = .05
# Set the contrast vector, use for the direction of the mean effect
contrast = [1.0, -1.0]

# Metrics list of interest, connectivity matrices will be
# computed according to this list
kinds = ['tangent', 'partial correlation', 'correlation']

# Creation of the directory which will contains all figure saved by the user
directory = '/media/db242421/db242421_data/ConPagnon_reports/analysis_01march2018_TEST/'+groupes[0]+'_'+groupes[1]
try:
    os.makedirs(directory)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

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

# Nilearn cache directory
nilearn_cache_directory = '/media/db242421/db242421_data/ConPagnon/nilearn_cache'

# Repetition time between time points in the fmri acquisition, usually 2.4s or 2.5s
repetition_time = 2.5


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

# Compute connectivity matrices for multiple metrics
# Covariance estimator
covariance_estimator = covariance.LedoitWolf()

# Choose the time series dictionnary
time_series_dict = times_series_individual_atlases

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
    vectorize=False, z_fisher_transform=True)

# Computing for each metric, and each groups the mean connectivity matrices.
mean_groups_connectivity_matrices = ccm.group_mean_connectivity(
    subjects_connectivity_matrices=subjects_connectivity_matrices,
    kinds=kinds)

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
# with backend_pdf.PdfPages(os.path.join(directory, 'overall_connectivity_distribution.pdf')) as pdf:
for kind in kinds:
        #plt.figure()
        for groupe in groupes:
            display.display_gaussian_connectivity_fit(
                vectorized_connectivity=parameters_estimation[groupe][kind]['vectorized connectivity'],
                estimate_mean=parameters_estimation[groupe][kind]['mean'],
                estimate_std=parameters_estimation[groupe][kind]['std'],
                raw_data_colors=hist_color[groupes.index(groupe)],
                fitted_distribution_color=fit_color[groupes.index(groupe)],
                title='Overall functional connectivity distribution  for {}'.format(kind),
                xtitle='Functional connectivity coefficient', ytitle='proportion of edges',
                legend_fitted='{} gaussian fitted distribution'.format(groupe),
                legend_data=groupe, display_fit='yes', ms=0.5)
        plt.show()
#        pdf.savefig()
# Extract homotopic connectivity coefficients on connectivity matrices
# Homotopic roi couple position in the connectivity matrices.
homotopic_roi_indices = np.array([
    (1, 0), (2, 3), (4, 5), (6, 7), (8, 11), (9, 10), (13, 12), (14, 15), (16, 17), (18, 19), (20, 25),
    (21, 26), (22, 29), (23, 28), (24, 27), (30, 31), (32, 33), (35, 34), (36, 37), (38, 39), (44, 40),
    (41, 45), (42, 43), (46, 49), (47, 48), (50, 53), (53, 54), (54, 57), (55, 56), (58, 61), (59, 60),
    (62, 63), (64, 65), (66, 67), (68, 69), (70, 71)])
# TODO: Construct a dictionnary with homotopic, ipsilesional, left hemisphere and
# TODO: right hemisphere rois within each defined network ?
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

# Extract from the Z fisher subject connectivity dictionnary the connectivity coefficients of interest
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

# Two sample t-test on mean homotopic connectivity between the two group
homotopic_t_test_ = parametric_tests.two_sample_t_test_(
    connectivity_dictionnary_=homotopic_distribution_parameters,
    groupes=groupes, kinds=kinds, field='subjects mean homotopic connectivity',
    contrast=contrast)

# Display barplot results on homotopic connectivity group mean
ind = np.arange(1)
n_labels = len(groupes)
groupes_color = ['blue', 'red', 'green']
width = 0.1
for kind in kinds:
    plt.figure()
    for i in range(n_labels):
        plt.bar(ind + i*width, homotopic_distribution_parameters[groupes[i]][kind]['homotopic distribution mean'],
                width, color=groupes_color[i], alpha=0.5, label=groupes[i])
        plt.title(kind + ' Homotopic functional connectivity')
        plt.legend(groupes)
    plt.show()

# Mean matrix plot, grouping ROI by side for more clarity of reading

# Left roi first, and right roi in second
new_roi_order = np.concatenate((left_regions_indices, right_regions_indices), axis=0)
new_labels_regions = [labels_regions[i] for i in new_roi_order]
new_labels_colors = labels_colors[new_roi_order, :]

# Group by roi matrix
for kind in kinds:
    for groupe in groupes:
        display.plot_matrix(matrix=Z_mean_groups_connectivity_matrices[groupe][kind][:, new_roi_order][new_roi_order],
                            mpart='all', labels_colors=new_labels_colors, horizontal_labels=new_labels_regions,
                            vertical_labels=new_labels_regions, title='mean ' + kind + ' ' + groupe, linewidths=0)
        plt.show()

for kind in kinds:
    difference = Z_mean_groups_connectivity_matrices[groupes[0]][kind][:, new_roi_order][new_roi_order] - \
                 Z_mean_groups_connectivity_matrices[groupes[1]][kind][:, new_roi_order][new_roi_order]

    display.plot_matrix(matrix=difference,
                        mpart='all', labels_colors=new_labels_colors, horizontal_labels=new_labels_regions,
                        vertical_labels=new_labels_regions,
                        title=groupes[0] + '-' + groupes[1] + ' mean ' + kind + ' difference', linewidths=0)
    plt.show()

# Connectivity intra-network
intra_network_connectivity_dict, network_dict, network_labels_list, network_label_colors = \
    ccm.intra_network_functional_connectivity(subjects_individual_matrices_dictionnary=Z_subjects_connectivity_matrices,
                                              groupes=groupes, kinds=kinds,
                                              atlas_file=atlas_excel_file,
                                              sheetname=sheetname, roi_indices_column_name='atlas4D index',
                                              network_column_name='network', color_of_network_column='Color')

# We can compute the homotopic connectivity for each network, i.e a intra-network homotopic connectivity
homotopic_intra_network_connectivity_d = dict.fromkeys(network_labels_list)
for network in network_labels_list:
    # Pick the 4D index of the roi in the network
    network_roi_ind = network_dict[network]['dataframe']['atlas4D index']
    # Extract connectivity coefficient couple corresponding to homotopic regions in the network
    network_homotopic_couple_ind = np.array([couple for couple in homotopic_roi_indices if (couple[0] or couple[1])
                                             in network_roi_ind])
    # Compute homotopic connectivity dictionnary for the current network
    network_homotopic_d = ccm.subjects_mean_connectivity_(
        subjects_individual_matrices_dictionnary=Z_subjects_connectivity_matrices,
        connectivity_coefficient_position=network_homotopic_couple_ind,
        kinds=kinds, groupes=groupes)
    homotopic_intra_network_connectivity_d[network] = network_homotopic_d

# Create a homotopic intra-network dictionnary with the same structure as the overall intra network dictionnary
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

# Test differences in the homotopic connectivity within each network between groups
homotopic_intra_network_strength_t_test = parametric_tests.intra_network_two_samples_t_test(
    intra_network_connectivity_dictionary=homotopic_intranetwork_d,
    groupes=groupes, kinds=kinds, contrast=[1.0, -1.0],
    network_labels_list=network_labels_list, assume_equal_var=True, alpha=alpha)

# Display the barplot for t statistic and p values for homotopic intra-network differences test
for kind in kinds:
    t_statistic = np.array([homotopic_intra_network_strength_t_test[kind][network]['t statistic'] for network
                            in network_labels_list])
    corrected_pvalues = np.array([homotopic_intra_network_strength_t_test[kind][network]['uncorrected p values'] for
                                  network in network_labels_list])
    plt.figure()
    display.t_and_p_values_barplot(t_values=t_statistic, p_values=corrected_pvalues, alpha_level=alpha,
                                   xlabel_color=network_label_colors, bar_labels=network_labels_list,
                                   t_xlabel='Network name', t_ylabel='t statistic values',
                                   p_xlabel='Network name', p_ylabel='FDR corrected p values',
                                   t_title='T statistic for homotopic intra-network connectivity \n'
                                           'comparison for {} between {} and {}'.format(kind, groupes[0], groupes[1]),
                                   p_title='P values for homotopic intra-network connectivity \n '
                                           'comparison for {} between {} and {}'.format(kind, groupes[0], groupes[1]))
    plt.show()

# T-test for the intra-network connectivity strength across subject between the group
intra_network_strength_t_test = parametric_tests.intra_network_two_samples_t_test(
    intra_network_connectivity_dictionary=intra_network_connectivity_dict,
    groupes=groupes, kinds=kinds, contrast=[1.0, -1.0],
    network_labels_list=network_labels_list, assume_equal_var=True, alpha=alpha)

# A barplot representation of the t-test for each network
for kind in kinds:
    t_statistic = np.array([intra_network_strength_t_test[kind][network]['t statistic'] for network
                            in network_labels_list])
    corrected_pvalues = np.array([intra_network_strength_t_test[kind][network]['uncorrected p values'] for network
                                  in network_labels_list])
    plt.figure()
    display.t_and_p_values_barplot(t_values=t_statistic, p_values=corrected_pvalues, alpha_level=alpha,
                                   xlabel_color=network_label_colors, bar_labels=network_labels_list,
                                   t_xlabel='Network name', t_ylabel='t statistic values',
                                   p_xlabel='Network name', p_ylabel='FDR corrected p values',
                                   t_title='T statistic for intra-network comparison for {} \n between {} and {}'.format
                                            (kind, groupes[0], groupes[1]),
                                   p_title='P values for intra-network comparison for {} \n between {} and {}'.format
                                            (kind, groupes[0], groupes[1]))
    plt.show()

# Overall and Within network, ipsilesional and contralesional connectivity differences
population_text_data = '/media/db242421/db242421_data/ConPagnon_data/regression_data/regression_data.xlsx'
# Drop subjects if wanted
subjects_to_drop = ['sub40_np130304']
# List of factor to group by
population_attribute = ['Lesion', 'langage_clinique']

# Compute the connectivity matrices dictionnary with factor as keys.
group_by_factor_subjects_connectivity, population_df_by_factor, factor_keys, =\
    dictionary_operations.groupby_factor_connectivity_matrices(
        population_data_file=population_text_data,
        sheetname='patients_data', subjects_connectivity_matrices_dictionnary=Z_subjects_connectivity_matrices,
        groupes=groupes, factors=['Lesion', 'langage_clinique'], drop_subjects_list=['sub40_np130304'])

# Create ipsilesional and contralesional dictionnary
ipsi_dict = {}
contra_dict = {}

# Fetch left and right connectivity for each group attribute keys
left_connectivity = ccm.extract_sub_connectivity_matrices(
    subjects_connectivity_matrices=group_by_factor_subjects_connectivity,
    kinds=kinds, regions_index=left_regions_indices)
right_connectivity = ccm.extract_sub_connectivity_matrices(
    subjects_connectivity_matrices=group_by_factor_subjects_connectivity,
    kinds=kinds, regions_index=right_regions_indices)

# Fill ipsi-lesional and contra-lesional dictionnary
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


# Construct a corresponding ipsilesional and contralesional dictionnary for controls
n_left_tot = len(ipsi_dict[('G', 'A')]) + len(ipsi_dict[('G', 'N')])
n_right_tot = len(ipsi_dict[('D', 'A')]) + len(ipsi_dict[('D', 'N')])

# Compute the percentage of right lesion and left lesion
n_total_patients = n_left_tot + n_right_tot
percent_left_lesioned = n_left_tot/n_total_patients
percent_right_lesioned = n_right_tot/n_total_patients

# total number of controls
n_controls = len(Z_subjects_connectivity_matrices['controls'].keys())
# Number of left and right impaired hemisphere to pick randomly in the controls group
n_left_hemisphere = ceil(percent_left_lesioned*n_controls)
n_right_hemisphere = n_controls - n_left_hemisphere

# Random pick of n_left_tot in proportion to the number of left lesion in controls group
n_left_ipsilesional_controls, n_left_ipsi_controls_ids = \
    dictionary_operations.random_draw_of_connectivity_matrices(
        subjects_connectivity_dictionary=Z_subjects_connectivity_matrices,
        groupe='controls', n_matrices=int(n_left_hemisphere),
        extract_kwargs={'kinds': kinds, 'regions_index': left_regions_indices,
                        'discard_diagonal': False,
                        'vectorize': False})
# Now, pick the right hemisphere in the left-out pool of subjects
leftout_controls_id = \
    list(set(Z_subjects_connectivity_matrices['controls'].keys())-set(n_left_ipsi_controls_ids))

n_right_ipsilesional_controls, n_right_ipsi_controls_ids = \
    dictionary_operations.random_draw_of_connectivity_matrices(
        subjects_connectivity_dictionary=Z_subjects_connectivity_matrices,
        groupe='controls', n_matrices=int(n_right_hemisphere),
        extract_kwargs={'kinds': kinds, 'regions_index': right_regions_indices,
                        'discard_diagonal': False,
                        'vectorize': False}, subjects_id_list=leftout_controls_id)

# Merge the right and left ipsilesional to have the 'ipsilesional' controls dictionnary
ipsilesional_controls_dictionary = dictionary_operations.merge_dictionary(
    new_key='controls',
    dict_list=[n_right_ipsilesional_controls['controls'],
               n_left_ipsilesional_controls['controls']])

# In the same, we have to generate, a "contralesional" dictionary for the
# controls group

# contralesional dictionary for left lesions, we fetch all RIGHT side roi this time
n_left_contralesional_controls, n_left_contra_controls_ids = \
    dictionary_operations.random_draw_of_connectivity_matrices(
        subjects_connectivity_dictionary=Z_subjects_connectivity_matrices,
        groupe='controls', n_matrices=int(n_left_hemisphere),
        extract_kwargs={'kinds': kinds, 'regions_index': right_regions_indices,
                        'discard_diagonal': False,
                        'vectorize': False})

# In the left-out pool of subjects, we fetch the LEFT side rois, for the
# contralesional side of right lesion.
leftout_controls_id = \
    list(set(Z_subjects_connectivity_matrices['controls'].keys())-set(n_left_contra_controls_ids))

n_right_contralesional_controls, n_right_contra_controls_ids = \
    dictionary_operations.random_draw_of_connectivity_matrices(
        subjects_connectivity_dictionary=Z_subjects_connectivity_matrices,
        groupe='controls', n_matrices=int(n_right_hemisphere),
        extract_kwargs={'kinds': kinds, 'regions_index': left_regions_indices,
                        'discard_diagonal': False,
                        'vectorize': False}, subjects_id_list=leftout_controls_id)

# Merge the right and left ipsilesional to have the 'ipsilesional' controls dictionnary
contralesional_controls_dictionary = dictionary_operations.merge_dictionary(
    new_key='controls',
    dict_list=[n_right_contralesional_controls['controls'],
               n_left_contralesional_controls['controls']])

# Finally, we have to merge ipsilesional/contralesional dictionaries of the different group
# Merge the dictionnary to build the overall contra and ipsi-lesional
# subjects connectivity matrices dictionnary

# First, the two group of patients
ipsilesional_patients_connectivity_matrices = {
    'L_Clin_Atyp_pat': {**ipsi_dict[('D', 'A')], **ipsi_dict[('G', 'A')]},
    'L_Clin_Typ_pat': {**ipsi_dict[('D', 'N')], **ipsi_dict[('G', 'N')]}}

contralesional_patients_connectivity_matrices = {
     'L_Clin_Atyp_pat': {**contra_dict[('D', 'A')], **contra_dict[('G', 'A')]},
     'L_Clin_Typ_pat': {**contra_dict[('D', 'N')], **contra_dict[('G', 'N')]}}

# Merged overall patients and controls dictionaries
ipsilesional_subjects_connectivity_matrices = dictionary_operations.merge_dictionary(
    dict_list=[ipsilesional_controls_dictionary,
               ipsilesional_patients_connectivity_matrices]
)
contralesional_subjects_connectivity_matrices = dictionary_operations.merge_dictionary(
    dict_list=[contralesional_controls_dictionary,
               contralesional_patients_connectivity_matrices]
)
groupes = groupes[1:]

# Compute the intra-network connectivity for the ipsilesional hemisphere in
# groups
ipsilesional_intra_network_connectivity_dict, \
    ipsi_network_dict, ipsi_network_labels_list, ipsi_network_label_colors = \
    ccm.intra_network_functional_connectivity(
        subjects_individual_matrices_dictionnary=ipsilesional_subjects_connectivity_matrices,
        groupes=groupes, kinds=kinds,
        atlas_file=atlas_excel_file,
        sheetname='Hemisphere_regions', roi_indices_column_name='index',
        network_column_name='network', color_of_network_column='Color')

# T-test for the intra-network connectivity strength across subject between the group
ipsilesional_intra_network_strength_t_test = parametric_tests.intra_network_two_samples_t_test(
    intra_network_connectivity_dictionary=ipsilesional_intra_network_connectivity_dict,
    groupes=groupes, kinds=kinds, contrast=[1.0, -1.0],
    network_labels_list=ipsi_network_labels_list, assume_equal_var=True, alpha=alpha)

# Barplot of the results for intra-network connectivity on the ipsilesional side
for kind in kinds:
    t_statistic = np.array([ipsilesional_intra_network_strength_t_test[kind][network]['t statistic'] for network
                            in ipsi_network_labels_list])
    corrected_pvalues = np.array([ipsilesional_intra_network_strength_t_test[kind][network]['uncorrected p values']
                                  for network in ipsi_network_labels_list])
    plt.figure()
    display.t_and_p_values_barplot(t_values=t_statistic, p_values=corrected_pvalues, alpha_level=alpha,
                                   xlabel_color=ipsi_network_label_colors, bar_labels=ipsi_network_labels_list,
                                   t_xlabel='Network name', t_ylabel='t statistic values',
                                   p_xlabel='Network name', p_ylabel='FDR corrected p values',
                                   t_title='T statistic for ipsilesional '
                                           'intra-network comparison for {} \n between {} and {}'.format
                                            (kind, groupes[0], groupes[1]),
                                   p_title='P values for ipsilesional '
                                           'intra-network comparison for {} \n between {} and {}'.format
                                            (kind, groupes[0], groupes[1]))
    plt.show()

# Compute the intra-network connectivity for the contralesional hemisphere in
# groups
contralesional_intra_network_connectivity_dict, \
    contra_network_dict, contra_network_labels_list, contra_network_label_colors = \
    ccm.intra_network_functional_connectivity(
        subjects_individual_matrices_dictionnary=contralesional_subjects_connectivity_matrices,
        groupes=groupes, kinds=kinds,
        atlas_file=atlas_excel_file,
        sheetname='Hemisphere_regions', roi_indices_column_name='index',
        network_column_name='network', color_of_network_column='Color')

# T-test for the intra-network connectivity strength across subject between the group
contralesional_intra_network_strength_t_test = parametric_tests.intra_network_two_samples_t_test(
    intra_network_connectivity_dictionary=contralesional_intra_network_connectivity_dict,
    groupes=groupes, kinds=kinds, contrast=[1.0, -1.0],
    network_labels_list=contra_network_labels_list, assume_equal_var=True, alpha=alpha)

for kind in kinds:
    t_statistic = np.array([contralesional_intra_network_strength_t_test[kind][network]['t statistic'] for network
                            in contra_network_labels_list])
    corrected_pvalues = np.array([contralesional_intra_network_strength_t_test[kind][network]['uncorrected p values']
                                  for network in contra_network_labels_list])
    plt.figure()
    display.t_and_p_values_barplot(t_values=t_statistic, p_values=corrected_pvalues, alpha_level=alpha,
                                   xlabel_color=ipsi_network_label_colors, bar_labels=contra_network_labels_list,
                                   t_xlabel='Network name', t_ylabel='t statistic values',
                                   p_xlabel='Network name', p_ylabel='FDR corrected p values',
                                   t_title='T statistic for contralesional '
                                           'intra-network comparison for {} \n between {} and {}'.format
                                            (kind, groupes[0], groupes[1]),
                                   p_title='P values for contralesional '
                                           'intra-network comparison for {} \n between {} and {}'.format
                                            (kind, groupes[0], groupes[1]))
    plt.show()

# Compute of mean ipsilesional distribution
ipsilesional_mean_connectivity = ccm.mean_of_flatten_connectivity_matrices(
    subjects_individual_matrices_dictionnary=ipsilesional_subjects_connectivity_matrices,
    groupes=groupes, kinds=kinds)
contralesional_mean_connectivity = ccm.mean_of_flatten_connectivity_matrices(
    subjects_individual_matrices_dictionnary=contralesional_subjects_connectivity_matrices,
    groupes=groupes, kinds=kinds)

import itertools
# Perform t-test for all possible pairs without replacement
pairs_of_groups = list(itertools.combinations(groupes, 2))
all_pairs_ipsilesional_t_test_ = dict.fromkeys([g for g in pairs_of_groups])
all_pairs_contralesional_t_test_ = dict.fromkeys([g for g in pairs_of_groups])
for group_pair in pairs_of_groups:
    all_pairs_ipsilesional_t_test_[group_pair] = parametric_tests.two_sample_t_test_(
        connectivity_dictionnary_=ipsilesional_distribution_parameters, groupes=list(group_pair), kinds=kinds,
        field='subjects mean ipsilesional connectivity', contrast=contrast)
    all_pairs_contralesional_t_test_[group_pair] = parametric_tests.two_sample_t_test_(
        connectivity_dictionnary_=contralesional_distribution_parameters, groupes=list(group_pair), kinds=kinds,
        field='subjects mean contralesional connectivity', contrast=contrast
    )

# Some plot of inter network result

# Inter-network connectivity :
subjects_inter_network_connectivity_matrices = ccm.inter_network_subjects_connectivity_matrices(
    subjects_individual_matrices_dictionnary=Z_subjects_connectivity_matrices, groupes=groupes, kinds=kinds,
    atlas_file=atlas_excel_file, sheetname=sheetname, network_column_name='network',
    roi_indices_column_name='atlas4D index')
from nilearn.plotting import plot_matrix
# Display mean connectivity inter-network matrices
for groupe in groupes:
    for kind in kinds:
        matrix_stack = np.array([subjects_inter_network_connectivity_matrices[groupe][s][kind] for s in
                                 subjects_inter_network_connectivity_matrices[groupe].keys()])
        mean_interwork_matrix = matrix_stack.mean(axis=0)
        if kind == 'tangent':
            np.fill_diagonal(mean_interwork_matrix, 0)
        #plot_matrix(mean_interwork_matrix, title=groupe + kind + ' mean inter-work matrix ', labels=network_labels_list)
        display.plot_matrix(matrix=mean_interwork_matrix, mpart='all', horizontal_labels=network_labels_list,
                            vertical_labels=network_labels_list, labels_colors=network_label_colors, title=groupe + ' mean ' + kind+
                                                                                                           ' inter-network connectivity matrix', vmin=mean_interwork_matrix.min(), vmax=mean_interwork_matrix.max(), k=0)


# Perform a two-sample t-test for the inter-network strength between the two network
# Two sample t-test on inter network connectivity matrices between the two group under study
inter_network_t_test_result = parametric_tests.inter_network_two_sample_t_test(
    subjects_inter_network_connectivity_matrices=subjects_inter_network_connectivity_matrices,
    groupes=['L_Clin_Typ_pat', 'controls'], kinds=kinds, contrast=contrast, assuming_equal_var=True, network_label_list=network_labels_list, alpha=alpha)

for kind in kinds:
    # Display significant t values
    display.plot_matrix(matrix=inter_network_t_test_result[kind]['significant t values'], labels_colors=network_label_colors, k=0,
                        horizontal_labels=network_labels_list, vertical_labels=network_labels_list,
                        title='Inter network T statistic for {}'.format(kind), linecolor='black', labels_size=12)
    display.plot_matrix(matrix=inter_network_t_test_result[kind]['corrected p values'], labels_colors=network_label_colors, k=0,
                        horizontal_labels=network_labels_list, vertical_labels=network_labels_list, colormap='hot', vmin=0, vmax=alpha,
                        title='Corrected p values for {} at {} threshold for inter network connectivity'.format(kind, alpha),
                        linecolor='black', labels_size=12)

# Compute the equivalent of degree centrality
# Integrity of the node: for the node i, absolute sum of all connection from that node i.
# Initialisation of the dictionary containing the node integrity
node_integrity_dictionary = dict.fromkeys(groupes)
for groupe in groupes:
    subjects_list = Z_subjects_connectivity_matrices[groupe].keys()
    node_integrity_dictionary[groupe] = dict.fromkeys(subjects_list)
    for subject in subjects_list:
        node_integrity_dictionary[groupe][subject] = dict.fromkeys(kinds)
        for kind in kinds:
            node_integrity_dictionary[groupe][subject][kind] = dict.fromkeys(range(n_nodes))
            for node in range(n_nodes):
                # Fetch the list for subject connectivity coefficient to sum for each node
                subject_connectivity_to_sum = [Z_subjects_connectivity_matrices[groupe][subject][kind][node, j]
                                               for j in range(n_nodes)]
                # Fetch the corresponding boolean mask for the corresponding node
                subject_masked_array_to_sum = [Z_subjects_connectivity_matrices[groupe][subject]['masked_array'][node, j]
                                               for j in range(n_nodes)]
                # Construct the masked array structure
                subject_masked_array = np.ma.array(data=subject_connectivity_to_sum, mask=subject_masked_array_to_sum)
                # Sum the absolute value of coefficient, via the masked array structure accounting for discarded rois, and nan
                # value due to possible Z-fisher transform of connectivity coefficient.
                node_integrity = np.nansum(subject_masked_array)
                # Fill the dictionnary saving the node integrity for each subject
                node_integrity_dictionary[groupe][subject][kind][node] = {'node integrity': node_integrity,
                                                                          'node masked array': subject_masked_array,
                                                                          'node index': node}

# Two sample t-test on node integrity for each node
p_value_correction_method = 'fdr_bh'
node_integrity_t_test = dict.fromkeys(kinds)
for kind in kinds:
    all_nodes_p_values = []
    node_integrity_t_test[kind] = dict.fromkeys(range(n_nodes))
    for node in range(n_nodes):
        # Fetch the node integrity for the first group
        x = [node_integrity_dictionary[groupes[0]][subject][kind][node]['node integrity']
             for subject in node_integrity_dictionary[groupes[0]].keys()]
        # Fetch the node integrity for the second group
        y = [node_integrity_dictionary[groupes[1]][subject][kind][node]['node integrity']
             for subject in node_integrity_dictionary[groupes[1]].keys()]
        if contrast == [1.0, -1.0]:
            t_statistic, uncorrected_p_values = ttest_ind(x, y, nan_policy='omit', equal_var=True)
        elif contrast == [-1.0, 1.0]:
            t_statistic, uncorrected_p_values = ttest_ind(y, x, nan_policy='omit', equal_var=True)
        else:
            raise ValueError('Unrecognized contrast vector !')
        # Append the uncorrected p-value for the current nodes
        all_nodes_p_values.append(uncorrected_p_values)
        # Fill the dictionnary for saving the raw results for the current node
        node_integrity_t_test[kind][node] = {'t statistic': t_statistic, 'uncorrected p values': uncorrected_p_values, 'contrast': contrast,
                                             groupes[0] + 'node integrity': x, groupes[1] + 'node integrity': y}

    # Correction of p-values, for each kinds correct for the number of test i.e the number of nodes
    reject_boolean_mask, corrected_pvalues, _, _ = multipletests(pvals=np.array(all_nodes_p_values), method=p_value_correction_method,
                                                                 alpha=alpha)

    # Fill the dictionnary, appending a new key containing the corrected p values
    for node in range(n_nodes):
        node_integrity_t_test[kind][node]['corrected pvalues'] = corrected_pvalues[node]

# Representation of significant difference in node integrity
size_x_title = 10
size_labels_nodes = 8
for kind in kinds:
    plt.figure()
    # Fetch the corrected p values integrity for each node
    corrected_pvalues = np.array([node_integrity_t_test[kind][node]['uncorrected p values'] for node in range(n_nodes)])
    # Plot a seaborn barplot, with each color node bar defined in the the labels colors array
    color_bar = [labels_colors[i] if (corrected_pvalues[i] < alpha) else 'grey' for i in range(n_nodes)]
    ax = sns.barplot(x=labels_regions, y=corrected_pvalues, palette=color_bar)
    plt.xlabel('Nodes labels', size=size_x_title)
    plt.ylabel('FDR corrected p values in log scale', size=size_x_title)
    # Set font size for the x axis, and rotate the labels of 90° for visibility
    for xtick, color in zip(ax.get_xticklabels(), labels_colors):
        xtick.set_color(color)
        xtick.set_fontsize(size_labels_nodes)
        xtick.set_rotation(90)
    # Plot a asymptote for significativity threshold
    plt.hlines(y=alpha, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], label=str(alpha) + ' threshold', colors='red')
    plt.title('Node integrity for {}'.format(kind))
    plt.legend()
    # Plot the T statistic distribution
    plt.figure()
    t_statistic = np.array([node_integrity_t_test[kind][node]['t statistic'] for node in range(n_nodes)])
    ax = sns.barplot(x=labels_regions, y=t_statistic, palette=color_bar)
    plt.xlabel('Nodes labels', size=size_x_title)
    plt.ylabel('T statistic values', size=size_x_title)
    for xtick, color in zip(ax.get_xticklabels(), labels_colors):
        xtick.set_color(color)
        xtick.set_fontsize(size_labels_nodes)
        xtick.set_rotation(90)
    plt.title('T statistic values distribution for {}'.format(kind))

# Computing the arithmetic mean of the pooled groups. See doctring function for details.
tangent_pooled_group_mean = ccm.pooled_groups_tangent_mean(time_series_dictionary=time_series_dict,
                                                           covariance_estimator=covariance_estimator)

# Computing connectivity matrices for pooled groups directly.
pooled_groups_connectivity_matrices, _ = ccm.pooled_groups_connectivity(time_series_dictionary=time_series_dict,
                                                                        covariance_estimator=covariance_estimator,
                                                                        kinds=kinds, vectorize=True)

# Group classification
time_series_dictionary = time_series_dict
group_stacked_time_series = []
labels = []
for groupe in time_series_dictionary.keys():
    subject_list = time_series_dictionary[groupe].keys()
    for subject in subject_list:
        group_stacked_time_series.append(time_series_dictionary[groupe][subject]['time_series'])
        labels.append(groupes.index(groupe))

# Build labels for each class we want to discriminate.
labels = labels
pooled_groups_connectivity_matrices = pooled_groups_connectivity_matrices
n_splits = 1000
test_size = 0.2
train_size = 0.8

# Compute mean scores for classification
mean_scores, mean_scores_dict = classification.two_groups_classification(
    pooled_groups_connectivity_matrices=pooled_groups_connectivity_matrices,
    labels=labels, n_splits=n_splits, test_size=test_size,
    train_size=train_size, kinds=kinds)

# bar plot of classification results
plt.figure()
sns.barplot(x=list(mean_scores_dict.keys()), y=list(mean_scores_dict.values()))
plt.xlabel('kind')
plt.ylabel('Mean scores of classification')
plt.title('Mean scores of classification using different kind of connectivity')

# Perform linear regression analysis of connectivity coefficients
# The kind to regress
kind = 'tangent'
# Design matrix parameters
formula = 'EHI'
NA_action = 'drop'
# Subject or list of subjects to drop
subjects_to_drop = ['sub40_np130304']
# Correction method
pvals_correction_method = 'maxT'
# Number of permutations for maxT correction
nperms_maxT = 10000
# Type I error rate
alpha = .05

# Load the connectivity data
connectivity_data = folders_and_files_management.load_object('regression_data.pkl')
# Excel file containing the dependent variables
data_xlsx = '/media/db242421/db242421_data/ConPagnon_data/regression_data/regression_data.xlsx'
# Regression results directory
regression_results_directory = '/media/db242421/db242421_data/ConPagnon_reports/regression_results_test'

regression_results, X_df, y = parametric_tests.linear_regression(connectivity_data=connectivity_data, data=data_xlsx, formula=formula,
                                                                 NA_action=NA_action, subjects_to_drop=subjects_to_drop, kind=kind,
                                                                 pvals_correction_method=pvals_correction_method, nperms_maxT=nperms_maxT,
                                                                 save_regression_directory=regression_results_directory)
# Plot p values matrix for significant results
with backend_pdf.PdfPages(os.path.join(regression_results_directory, 'significant_results.pdf')) as pdf:
    for i in range(len(X_df.columns)):
        display.plot_matrix(matrix=regression_results[X_df.columns[i]]['corrected pvalues'], labels_colors=labels_colors,
                            colormap='hot', vmin=0, vmax=alpha, horizontal_labels=labels_regions, vertical_labels=labels_regions,
                            title='Significant connection for {} at {} level, {} corrected'.format(X_df.columns[i], alpha,
                                                                                                   pvals_correction_method),
                            linecolor='black')
        pdf.savefig()


display.plot_matrix(matrix=regression_results[X_df.columns[1]]['corrected pvalues'], labels_colors=labels_colors,
                    colormap='hot', vmin=0, vmax=alpha, horizontal_labels=labels_regions, vertical_labels=labels_regions,
                    title='Significant connection for {} at {} level, {} corrected'.format(X_df.columns[1], alpha,
                                                                                           pvals_correction_method))
