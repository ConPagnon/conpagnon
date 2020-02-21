import importlib
from conpagnon.data_handling import data_architecture, dictionary_operations, atlas, data_management
from conpagnon.utils import folders_and_files_management
from conpagnon.computing import compute_connectivity_matrices as ccm
from conpagnon.machine_learning import classification
from conpagnon.connectivity_statistics import parametric_tests
from conpagnon.plotting import display
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from conpagnon.plotting.display import t_and_p_values_barplot
from scipy.stats import ttest_rel
from matplotlib.backends.backend_pdf import PdfPages
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


root_directory = "/media/db242421/db242421_data/ConPagnon_data/language_study_ANOVA_ACM_controls"
kinds = ['correlation', 'partial correlation', 'tangent']
network_name = ['DMN', 'Executive', 'Language', 'MTL', 'Primary_Visual',
                'Salience', 'Secondary_Visual', 'Sensorimotor', 'Visuospatial']

subjects_connectivity_matrices = folders_and_files_management.load_object(
    full_path_to_object=os.path.join(root_directory,  "dictionary/z_fisher_transform_subjects_connectivity_matrices.pkl")
)
subjects_connectivity_matrices = {'patients': {**subjects_connectivity_matrices['impaired_language'],
                                               **subjects_connectivity_matrices['non_impaired_language']},
                                  'controls': {**subjects_connectivity_matrices['controls']}}
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

groups = list(subjects_connectivity_matrices.keys())
asymmetry_results_dictionary = dict.fromkeys(groups)
# Fetch left and right connectivity for each group attribute keys
left_connectivity = ccm.extract_sub_connectivity_matrices(
    subjects_connectivity_matrices=subjects_connectivity_matrices,
    kinds=kinds, regions_index=left_regions_indices)
right_connectivity = ccm.extract_sub_connectivity_matrices(
    subjects_connectivity_matrices=subjects_connectivity_matrices,
    kinds=kinds, regions_index=right_regions_indices)

right_intra_network_connectivity_dict, right_network_dict, right_network_labels_list, \
right_network_label_colors = ccm.intra_network_functional_connectivity(
    subjects_individual_matrices_dictionnary=right_connectivity,
    groupes=groups, kinds=kinds,
    atlas_file=atlas_excel_file,
    sheetname='Hemisphere_regions',
    roi_indices_column_name='index',
    network_column_name='network',
    color_of_network_column='Color')
# Left hemisphere
left_intra_network_connectivity_dict, left_network_dict, left_network_labels_list, \
left_network_label_colors = ccm.intra_network_functional_connectivity(
    subjects_individual_matrices_dictionnary=left_connectivity,
    groupes=groups, kinds=kinds,
    atlas_file=atlas_excel_file,
    sheetname='Hemisphere_regions',
    roi_indices_column_name='index',
    network_column_name='network',
    color_of_network_column='Color')

ipsilesional_patients_dictionary = folders_and_files_management.load_object(
    full_path_to_object="/media/db242421/db242421_data/ConPagnon_data/language_study_ANOVA_ACM_controls/"
                        "dictionary/ipsilesional_intra_network_connectivity.pkl"
)
contralesional_patients_dictionary = folders_and_files_management.load_object(
    full_path_to_object="/media/db242421/db242421_data/ConPagnon_data/language_study_ANOVA_ACM_controls/"
                        "dictionary/contralesional_intra_network_connectivity.pkl"
)

ipsilesional_patients_dictionary = {'patients': {**ipsilesional_patients_dictionary['impaired_language'],
                                               **ipsilesional_patients_dictionary['non_impaired_language']},
                                  'controls': {**ipsilesional_patients_dictionary['controls']}}

contralesional_patients_dictionary = {'patients': {**contralesional_patients_dictionary['impaired_language'],
                                               **contralesional_patients_dictionary['non_impaired_language']},
                                  'controls': {**contralesional_patients_dictionary['controls']}}

for kind in kinds:
    # Paired t-test for intra-network connectivity between ipsilesional and contralesional
    # hemisphere in patients groups
    for group in groups:
        subjects_list = list(subjects_connectivity_matrices[group].keys())
        asymmetry_results_dictionary[group] = dict.fromkeys(network_name)
        for network in network_name:
            network_mean_ipsi_connectivity = np.array(
                [ipsilesional_patients_dictionary[group][s][kind][network]['network connectivity strength']
                 for s in subjects_list])
            network_mean_contra_connectivity = np.array(
                [contralesional_patients_dictionary[group][s][kind][network]['network connectivity strength']
                 for s in subjects_list])
            # paired t-test between ipsilesional and contralesional for the current network
            network_t, network_p = ttest_rel(a= network_mean_ipsi_connectivity,
                                             b=network_mean_contra_connectivity)
            # Fill dictionary
            asymmetry_results_dictionary[group][network] = {'t': network_t,
                                                            'p': network_p}

    with PdfPages("/media/db242421/db242421_data/ConPagnon_data/language_study_ANOVA_ACM_controls/asymetry/" + kind +
                  "_asymmetry_report.pdf") as pdf:

        for group in list(asymmetry_results_dictionary.keys()):
            t_values = [asymmetry_results_dictionary[group][network]['t'] for network in left_network_labels_list]
            p_values = [asymmetry_results_dictionary[group][network]['p'] for network in left_network_labels_list]
            labels_colors = left_network_label_colors
            plt.figure()
            t_and_p_values_barplot(t_values=t_values, p_values=p_values, alpha_level=0.05,
                                   xlabel_color=labels_colors, bar_labels=left_network_labels_list,
                                   t_xlabel='', t_ylabel='t value (paired)',
                                   p_xlabel='', p_ylabel='p value (uncorrected)',
                                   t_title='left/ipsi - right/contra hemisphere for {}'.format(group),
                                   p_title='left/ipsi - right/contra hemisphere for {}'.format(group))
            pdf.savefig()
            plt.show()