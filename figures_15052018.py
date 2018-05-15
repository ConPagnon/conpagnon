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
labels_colors = (1./255)*monAtlas.UserLabelsColors(networks=networks,
                                                   colors=colors)

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