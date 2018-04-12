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


"""

# Atlas set up
atlas_folder = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference'
atlas_name = 'atlas4D_2.nii'
monAtlas = atlas.Atlas(path=atlas_folder,
                       name=atlas_name)
# Atlas path
atlas_path = monAtlas.fetch_atlas()
# Read labels regions files
labels_text_file = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference/atlas4D_2_labels.csv'
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
groupes = ['patients', 'controls']
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

# output csv directory
output_csv_directory_path = '/media/db242421/db242421_data/ConPagnon_data/text_output_11042018'
output_csv_directory = data_management.create_directory(directory=output_csv_directory_path, erase_previous=True)

# Cohort behavioral data
cohort_excel_file_path = '/media/db242421/db242421_data/ConPagnon_data/regression_data/regression_data.xlsx'
behavioral_data = data_management.read_excel_file(excel_file_path=cohort_excel_file_path,
                                                  sheetname='cohort_functional_data')

# save a CSV file format for the behavioral data
behavioral_data.to_csv(os.path.join(output_csv_directory, 'behavioral_data.csv'))

# Set the type I error (alpha level), usually .05.
alpha = .05
# Set the contrast vector, use for the direction of the mean effect
contrast = [1.0, -1.0]

# Metrics list of interest, connectivity matrices will be
# computed according to this list
kinds = ['tangent', 'partial correlation', 'correlation']

# Creation of the directory which will contains all figure saved by the user
directory = '/media/db242421/db242421_data/ConPagnon_reports/analysis_01march2018_TEST/' + groupes[0] + '_' + groupes[1]
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
# with backend_pdf.PdfPages(os.path.join(directory, 'overall_connectivity_distribution.pdf')) as pdf:
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
                legend_fitted='{} gaussian fitted distribution'.format(groupe),
                legend_data=groupe, display_fit='no', ms=0.5)
        #plt.show()
#        pdf.savefig()
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

# Extract from the Z fisher subject connectivity dictionnary the connectivity coefficients of interest
homotopic_connectivity = ccm.subjects_mean_connectivity_(
    subjects_individual_matrices_dictionnary=Z_subjects_connectivity_matrices,
    connectivity_coefficient_position=homotopic_roi_indices, kinds=kinds,
    groupes=groupes)

# Left roi first, and right roi in second
new_roi_order = np.concatenate((left_regions_indices, right_regions_indices), axis=0)
new_labels_regions = [labels_regions[i] for i in new_roi_order]
new_labels_colors = labels_colors[new_roi_order, :]

# Group by roi matrix
for kind in kinds:
    for groupe in groupes:
        display.plot_matrix(
            matrix=Z_mean_groups_connectivity_matrices[groupe][kind][:, new_roi_order][new_roi_order],
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
    ccm.intra_network_functional_connectivity(
        subjects_individual_matrices_dictionnary=Z_subjects_connectivity_matrices,
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
population_text_data = '/media/db242421/db242421_data/ConPagnon_data/regression_data/regression_data.xlsx'
# Drop subjects if wanted
subjects_to_drop = ['sub40_np130304']
# List of factor to group by
population_attribute = ['Lesion']

# Compute the connectivity matrices dictionnary with factor as keys.
group_by_factor_subjects_connectivity, population_df_by_factor, factor_keys, =\
    dictionary_operations.groupby_factor_connectivity_matrices(
        population_data_file=population_text_data,
        sheetname='cohort_functional_data',
        subjects_connectivity_matrices_dictionnary=Z_subjects_connectivity_matrices,
        groupes=['patients'], factors=population_attribute, drop_subjects_list=['sub40_np130304'])

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
n_left_tot = len(ipsi_dict['G'])
n_right_tot = len(ipsi_dict['D'])

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
    'patients': {**ipsi_dict['D'], **ipsi_dict['G']},
    }

contralesional_patients_connectivity_matrices = {
     'patients': {**contra_dict['D'], **contra_dict['G']},
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

# Compute of mean ipsilesional distribution
ipsilesional_mean_connectivity = ccm.mean_of_flatten_connectivity_matrices(
    subjects_individual_matrices_dictionnary=ipsilesional_subjects_connectivity_matrices,
    groupes=groupes, kinds=kinds)
contralesional_mean_connectivity = ccm.mean_of_flatten_connectivity_matrices(
    subjects_individual_matrices_dictionnary=contralesional_subjects_connectivity_matrices,
    groupes=groupes, kinds=kinds)

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


# Compute the inter-network connectivity for whole brain:
subjects_inter_network_connectivity_matrices = ccm.inter_network_subjects_connectivity_matrices(
    subjects_individual_matrices_dictionnary=Z_subjects_connectivity_matrices, groupes=groupes, kinds=kinds,
    atlas_file=atlas_excel_file, sheetname=sheetname, network_column_name='network',
    roi_indices_column_name='atlas4D index')


# Compute ipsilesional inter-network connectivity
subjects_inter_network_ipsilesional_connectivity_matrices = ccm.inter_network_subjects_connectivity_matrices(
    subjects_individual_matrices_dictionnary=ipsilesional_subjects_connectivity_matrices, groupes=groupes, kinds=kinds,
    atlas_file=atlas_excel_file, sheetname='Hemisphere_regions', network_column_name='network',
    roi_indices_column_name='index')

# Compute contralesional inter-network connectivity
subjects_inter_network_contralesional_connectivity_matrices = ccm.inter_network_subjects_connectivity_matrices(
    subjects_individual_matrices_dictionnary=contralesional_subjects_connectivity_matrices, groupes=groupes,
    kinds=kinds,
    atlas_file=atlas_excel_file, sheetname='Hemisphere_regions', network_column_name='network',
    roi_indices_column_name='index')

# Now save the different dictionary for the current analysis
# Save the times series dictionary
save_dict_path = os.path.join(output_csv_directory, 'dictionary')
dictionary_output_directory = data_management.create_directory(save_dict_path)
folders_and_files_management.save_object(object_to_save=times_series_individual_atlases,
                                         saving_directory=dictionary_output_directory,
                                         filename='times_series_individual_atlases.pkl')
# Save the raw subjects matrices dictionary
folders_and_files_management.save_object(object_to_save=subjects_connectivity_matrices,
                                         saving_directory=dictionary_output_directory,
                                         filename='raw_subjects_connectivity_matrices.pkl')
# Save the Z-fisher transform matrices dictionary
folders_and_files_management.save_object(object_to_save=Z_subjects_connectivity_matrices,
                                         saving_directory=dictionary_output_directory,
                                         filename='z_fisher_transform_subjects_connectivity_matrices.pkl')
# Save the Z-fisher mean groups connectivity matrices
folders_and_files_management.save_object(object_to_save=Z_mean_groups_connectivity_matrices,
                                         saving_directory=dictionary_output_directory,
                                         filename='z_fisher_groups_mean_connectivity_matrices.pkl')
# Save the whole brain homotopic connectivity dictionary
folders_and_files_management.save_object(object_to_save=homotopic_connectivity,
                                         saving_directory=dictionary_output_directory,
                                         filename='whole_brain_mean_homotopic_connectivity.pkl')
# Save the intra-network mean homotopic dictionary
folders_and_files_management.save_object(object_to_save=homotopic_intranetwork_d,
                                         saving_directory=dictionary_output_directory,
                                         filename='intra_network_mean_homotopic_connectivity.pkl')
# Save the whole brain intra_network connectivity dictionary
folders_and_files_management.save_object(object_to_save=intra_network_connectivity_dict,
                                         saving_directory=dictionary_output_directory,
                                         filename='whole_brain_intra_network_connectivity.pkl')
# Save the ipsilesional intra-network connectivity dictionary
folders_and_files_management.save_object(object_to_save=ipsilesional_intra_network_connectivity_dict,
                                         saving_directory=dictionary_output_directory,
                                         filename='ipsilesional_intra_network_connectivity.pkl')
# Save the contralesional intra-network connectivity dictionary
folders_and_files_management.save_object(object_to_save=contralesional_intra_network_connectivity_dict,
                                         saving_directory=dictionary_output_directory,
                                         filename='contralesional_intra_network_connectivity.pkl')
# Save the whole-brain inter-network connectivity dictionary
folders_and_files_management.save_object(object_to_save=subjects_inter_network_connectivity_matrices,
                                         saving_directory=dictionary_output_directory,
                                         filename='subjects_inter_network_connectivity_matrices.pkl')
# Save the ipsilesional inter-network connectivity dictionary
folders_and_files_management.save_object(object_to_save=subjects_inter_network_ipsilesional_connectivity_matrices,
                                         saving_directory=dictionary_output_directory,
                                         filename='subjects_inter_network_ipsi_connectivity_matrices.pkl')
# Save the contralesional inter-network connectivity dictionary
folders_and_files_management.save_object(object_to_save=subjects_inter_network_contralesional_connectivity_matrices,
                                         saving_directory=dictionary_output_directory,
                                         filename='subjects_inter_network_contra_connectivity_matrices.pkl')

# Stastical analysis
# Overall homotopic, ipsilesional and contralesional connectivity with respect to groups, controling for gender
# with a linear model
kinds_to_model  = ['correlation', 'partial correlation', 'tangent']
groups_in_models = ['patients', 'controls']

# data_directory = os.path.join('D:\\text_output_11042018', kind)
# Choose the correction method
correction_method = 'FDR'
# Fit three linear model for the three type of overall connections
models_to_build = ['mean_homotopic', 'mean_ipsilesional', 'mean_contralesional']

# variables in the model
variables_model = ['Groupe', 'Sexe']

# formulation of the model
model_formula = 'Groupe + Sexe'

# Load behavioral data
# cohort_excel_file_path = 'D:\\regression_data\\regression_data.xlsx'
behavioral_data = data_management.read_excel_file(excel_file_path=cohort_excel_file_path,
                                                  sheetname='cohort_functional_data')
# output_csv_directory = 'D:\\text_output_11042018'
# Clean the data: drop subjects if needed
drop_subjects_list = ['sub40_np130304']
if drop_subjects_list:
    behavioral_data_cleaned = behavioral_data.drop(drop_subjects_list)
else:
    behavioral_data_cleaned = behavioral_data

# For each model: read the csv for each group, concatenate resultings dataframe, and append
# (merging by index) all variable of interest in the model.
for kind in kinds_to_model:
    # directory where the data are
    data_directory = os.path.join('/media/db242421/db242421_data/ConPagnon_data/text_output_11042018',
                                  kind)
    all_model_response = []
    for model in models_to_build:
        # List of the corresponding dataframes
        model_dataframe = data_management.concatenate_dataframes([data_management.read_csv(
            csv_file=os.path.join(data_directory, group + '_' + kind + '_' + model + '.csv'))
                                                                  for group in groups_in_models])
        # Shift index to be the subjects identifiers
        model_dataframe = data_management.shift_index_column(panda_dataframe=model_dataframe,
                                                             columns_to_index='subjects')
        # Add variables in the model to complete the overall DataFrame
        model_dataframe = data_management.merge_by_index(dataframe1=model_dataframe,
                                                         dataframe2=behavioral_data[variables_model])
        # Build the model formula: the variable to explain is the first column of the
        # dataframe, and we add to the left all variable in the model
        model_formulation = model_dataframe.columns[0] + '~' + '+'.join(variables_model)
        # Build response, and design matrix from the model model formulation
        model_response, model_design = parametric_tests.design_matrix_builder(dataframe=model_dataframe,
                                                                              formula=model_formulation,
                                                                              return_type='dataframe')
        # regression with a simple OLS model
        model_fit = parametric_tests.ols_regression(y=model_response, X=model_design)

        # Creation of a directory for the current analysis
        regression_output_directory = folders_and_files_management.create_directory(
            directory=os.path.join(output_csv_directory, 'regression_analysis', kind))

        # Write output regression results in csv files
        data_management.write_ols_results(ols_fit=model_fit, design_matrix=model_design,
                                          response_variable=model_response,
                                          output_dir=regression_output_directory,
                                          model_name=model,
                                          design_matrix_index_name='subjects')
        # Appending current model response
        all_model_response.append(model_response)



    # p-values correction for the 3 models with permutation method in mulm package
    from pylearn_mulm import mulm
    from patsy import dmatrix

    # The design matrix is the same for all model
    design_matrix = dmatrix('Groupe + Sexe', behavioral_data_cleaned, return_type='dataframe')
    # Append the dataframe
    all_model_response.append(design_matrix)
    # merge by index the dataframe
    df_tmp = data_management.merge_by_index(dataframe1=all_model_response[0], dataframe2=all_model_response[1])
    df_tmp = data_management.merge_by_index(dataframe1=df_tmp, dataframe2=all_model_response[2])
    # Re-index the response variable dataframe to match the index of design matrix
    ipsi_homo_contra_connectivity = df_tmp.reindex(design_matrix.index)

    # Fit a linear model and correcting for maximum statistic
    mulm_fit = mulm.MUOLS(Y=np.array(ipsi_homo_contra_connectivity), X=np.array(design_matrix)).fit()
    contrasts = np.identity(np.array(design_matrix).shape[1])
    raw_t, raw_p, df = mulm_fit.t_test(contrasts=contrasts, two_tailed=True, pval=True)
    if correction_method == 'maxT':

        _, p_values_maximum_T, _, null_distribution_max_T = mulm_fit.t_test_maxT(contrasts=contrasts, two_tailed=True,
                                                                                 nperms=10000)
        corrected_p_values = p_values_maximum_T
    elif correction_method == 'FDR':
        raw_p_shape = raw_p.shape
        fdr_corrected_p_values = multipletests(pvals=raw_p.flatten(),
                                               method='fdr_bh', alpha=alpha)[1].reshape(raw_p_shape)
        corrected_p_values = fdr_corrected_p_values

    # Append in each model CSV file, the corrected p-values for maximum statistic
    for model in models_to_build:
        model_csv_file = os.path.join(output_csv_directory, 'regression_analysis', kind,
                                 model + '_parameters.csv')
        # Read the csv file
        model_parameters = data_management.read_csv(model_csv_file)
        # Add a last column for adjusted p-values
        model_parameters[correction_method + 'corrected_pvalues'] = corrected_p_values[:, models_to_build.index(model)]
        # Write it back to csf format
        data_management.dataframe_to_csv(dataframe=model_parameters, path=model_csv_file)
