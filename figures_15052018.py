# Figure generation for AVCnn presentation
from nilearn.plotting import plot_epi, show, plot_prob_atlas, plot_roi
import os
from nilearn.image.image import mean_img
from matplotlib.backends.backend_pdf import PdfPages
from utils.folders_and_files_management import load_object
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotting.display import plot_matrix
from data_handling import atlas
# Illustration of an EPI image
epi_folders = '/media/db242421/db242421_data/ConPagnon_data/fmri_images/controls'
an_epi_img = os.path.join(epi_folders, 'art_mv_fmv_wm_vent_ext_beat_hv_RSc12_sub03_ct110201.nii.gz')
save_figures = '/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/figures_presentation_15052018'
mean_epi = mean_img(an_epi_img)


with PdfPages(os.path.join(save_figures, 'epi.pdf')) as pdf:

    plot_epi(epi_img=mean_epi, cmap='Greys', black_bg=False, draw_cross=False)
    pdf.savefig()
    show()

# Plot some of the time series
times_series = load_object('/media/db242421/db242421_data/ConPagnon_data/25042018_Patients_LangScore/dictionary/'
                           'times_series_individual_atlases.pkl')
one_times_series_subject = times_series['controls']['sub26_ep120255']['time_series']
# Plot five times series
all_time_series = np.array([times_series['controls'][s]['time_series'] for s in times_series['controls'].keys()])
colors = ['indianred', 'peru', 'mediumblue', 'olive', 'lightseagreen']
with PdfPages(os.path.join(save_figures, 'time_series.pdf')) as pdf:
    for i in range(4):
        plt.figure()
        plt.plot(np.arange(0, 180), all_time_series[0, :, i], np.random.rand(3,1), color=colors[i])
        pdf.savefig()
        plt.show()

# Plot some connectivity matrix
connectivity_matrices = load_object('/media/db242421/db242421_data/ConPagnon_data/patient_controls/'
                                    'dictionary/raw_subjects_connectivity_matrices.pkl')
all_matrices = np.array([connectivity_matrices['controls'][s]['correlation'] for s in
                         connectivity_matrices['controls'].keys()])
with PdfPages(os.path.join(save_figures, 'connectivity_matrices.pdf')) as pdf:
    for i in range(4):
        plt.figure()
        plot_matrix(matrix=all_matrices[0, ...], title='', mpart='all')
        pdf.savefig()
        plt.show()

# plot some functional atlases
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
labels_colors = (1./255) * monAtlas.user_labels_colors(networks=networks, colors=colors)

from matplotlib import colors
cmap = colors.ListedColormap(labels_colors)

functional_atlas_folder = '/media/db242421/db242421_data/ConPagnon_data/atlas/individual_atlases_V2'
subjects_atlas_list = ['sub40_np130304_atlas.nii', 'sub02_rf110332_atlas.nii', 'sub04_rc110343_atlas.nii',
                       'sub28_pp110331_atlas.nii']
with PdfPages(os.path.join(save_figures, 'atlas_indiv.pdf')) as pdf:
    for i in range(len(subjects_atlas_list)):
        atlas_path = os.path.join(functional_atlas_folder, subjects_atlas_list[i])
        plt.figure()
        plot_prob_atlas(maps_img=atlas_path, view_type='filled_contours', title=subjects_atlas_list[i],
                        draw_cross=False, cut_coords=(0, 0, 0), threshold=0., alpha=0.7,
                        cmap=cmap)
        pdf.savefig()
        plt.show()

# Classification barplot
from computing import compute_connectivity_matrices as ccm
from sklearn import covariance
from machine_learning import classification

time_series_dict = times_series
covariance_estimator = covariance.LedoitWolf()
kinds = ['correlation', 'partial correlation', 'tangent']
groupes = ['patients', 'controls']


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
n_splits = 10000
test_size = 0.3
train_size = 0.7

# Compute mean scores for classification
mean_scores, mean_scores_dict = classification.two_groups_classification(
    pooled_groups_connectivity_matrices=pooled_groups_connectivity_matrices,
    labels=labels, n_splits=n_splits, test_size=test_size,
    train_size=train_size, kinds=kinds)

# bar plot of classification results
with PdfPages(os.path.join(save_figures, 'classif.pdf')) as pdf:

    plt.figure()
    sns.barplot(x=list(mean_scores_dict.keys()), y=list(mean_scores_dict.values()),
                palette=['red', 'green', 'goldenrod'])
    plt.xlabel('metric')
    plt.ylabel('Mean scores of classification')
    plt.title('Mean scores of classification using different kind of connectivity')
    pdf.savefig()
    plt.show()

# Display of results: For each network, take the t-value for : intra-network, homotopic intra network, ipsilesional
# intra network, contralesional intra-network
network_to_plot = ['DMN', 'Executive',
                   'Language', 'MTL',
                   'Salience', 'Sensorimotor', 'Visuospatial',
                   'Primary_Visual', 'Secondary_Visual']

from data_handling import data_management
from plotting.display import t_and_p_values_barplot
from itertools import repeat

network_color_df = data_management.shift_index_column(atlas_information[['network', 'Color']], 'network')

network_color = [list(set(network_color_df.loc[n]['Color']))[0] for n in network_to_plot]

model_to_plot = ['mean_connectivity', 'mean_homotopic', 'mean_contralesional', 'mean_ipsilesional']
bar_label = ['G', 'H', 'CL', 'IPS']

results_directory = '/media/db242421/db242421_data/ConPagnon_data/25042018_Patients_LangScore/regression_analysis'
model_dictionary = dict.fromkeys(network_to_plot)

output_fig_folder = '/media/db242421/db242421_data/ConPagnon_data/figures_Patients_LangScore'

for network in network_to_plot:
    all_models_t = []
    all_models_p = []
    for model in model_to_plot:
        model_result = pd.read_csv(os.path.join(results_directory, 'tangent', network,
                                                model + '_parameters.csv'))
        all_models_t.append(model_result['t'].loc[1])
        all_models_p.append(model_result['FDRcorrected_pvalues'].loc[1])

    model_dictionary[network] = {'t_values': all_models_t, 'p_values': all_models_p}

# For the whole brain model
whole_brain_t = []
whole_brain_p = []
for model in model_to_plot:
    model_result = pd.read_csv(os.path.join(results_directory, 'tangent', model + '_parameters.csv'))
    whole_brain_t.append(model_result['t'].loc[2])
    whole_brain_p.append(model_result['maxTcorrected_pvalues'].loc[2])

with backend_pdf.PdfPages(os.path.join(output_fig_folder, 'patients_LangScore_3.pdf')) as pdf:

    for network in network_to_plot:
        # plt.figure()
        xlabel_color = [x for item in [network_color[network_to_plot.index(network)]] for
                        x in repeat(network_color[network_to_plot.index(network)], len(model_to_plot))]
        t_and_p_values_barplot(t_values=model_dictionary[network]['t_values'],
                               p_values=model_dictionary[network]['p_values'],
                               alpha_level=0.05,
                               xlabel_color=xlabel_color, bar_labels=bar_label,
                               t_xlabel=' ', t_ylabel='t statistic', p_xlabel=' ', p_ylabel='corrected p value',
                               t_title=network, p_title=network, xlabel_size=20
                               )
        pdf.savefig()

with backend_pdf.PdfPages(os.path.join(output_fig_folder, 'patients_LangScore_3.pdf')) as pdf:
    t_and_p_values_barplot(t_values=whole_brain_t,
                           p_values=whole_brain_p,
                           alpha_level=0.05,
                           xlabel_color=['black', 'black', 'black', 'black'], bar_labels=bar_label,
                           t_xlabel=' ', t_ylabel='t statistic', p_xlabel=' ', p_ylabel='corrected p value',
                           t_title='', p_title='', xlabel_size=20
                           )
    pdf.savefig()

# Illustration for may 15 presentation
from matplotlib import colors

cmap = colors.ListedColormap(labels_colors)
from nilearn.plotting import plot_prob_atlas, plot_connectome
from data_handling import data_management

reference_anatomical_image = '/media/db242421/db242421_data/ConPagnon_data/atlas/' \
                             'atlas_reference/wanat1_nc110193-2604_20110427_02.nii'
plot_prob_atlas(maps_img=atlas_path, view_type='filled_contours', title='AVCnn resting state atlas',
                draw_cross=False, cut_coords=(0, 0, 0), threshold=0., alpha=0.7,
                output_file='/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/'
                            'figures_presentation_15052018/atlas_avcnn_72rois_ver/roi_plot_on_MNI.pdf',
                cmap=cmap)
plt.show()
# TODO: Batch all this figure !!!!
# Plot each network on a glass brain
network_to_plot = ['Sensorimotor', 'Visuospatial', 'Salience'
                   ]
network_color_df = data_management.shift_index_column(atlas_information[['network', 'Color']],
                                                      'network')

network_nodes_df = data_management.shift_index_column(atlas_information[['network', 'x_', 'y_', 'z_']],
                                                      'network')

network_color = [list(set(network_color_df.loc[n]['Color']))[0] for n in network_to_plot]
n_regions_network = [len(list((network_color_df.loc[n]['Color']))) for n in network_to_plot]

with backend_pdf.PdfPages('/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/'
                          'figures_presentation_15052018/atlas_avcnn_72rois_ver/one_side_inter_network_sensorimotor_language.pdf') as pdf:
    for network in network_to_plot:
        plt.figure()
        plot_connectome(adjacency_matrix=np.zeros((n_regions_network[network_to_plot.index(network)],
                                                   n_regions_network[network_to_plot.index(network)])),
                        node_coords=np.array(network_nodes_df.loc[network]),
                        node_color=network_color[network_to_plot.index(network)],
                        title=network + ' network', display_mode='lyrz')
        pdf.savefig()

        plt.show()

# One side plot
network_node_ = network_nodes_df.loc[['Sensorimotor', 'Visuospatial', 'Salience']]
chosen_node_ = network_node_.iloc[[0, 2, 4, 6, 7, 10, 11, 14, 16, 18]]
chosen_node_adjacency_matrix = np.zeros((chosen_node_.shape[0], chosen_node_.shape[0]))
plot_connectome(adjacency_matrix=chosen_node_adjacency_matrix, node_coords=np.array(chosen_node_),
                node_color=network_color, title='Sensorimotor, Visuospatial and Salience',
                output_file='/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/figures_presentation_15052018/'
                            'atlas_avcnn_72rois_ver/One_side_sensorimotor_visuospatial_salience.pdf')
plt.show()

# Plot the full set of nodes on the glass brain
plot_connectome(adjacency_matrix=np.zeros((n_nodes, n_nodes)),
                node_coords=atlas_nodes,
                node_color=labels_colors,
                title='AVCnn atlas', display_mode='lyrz',
                output_file='/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/'
                            'figures_presentation_15052018/atlas_avcnn_72rois_ver/empty_glass_brain.pdf')