from conpagnon.computing import compute_connectivity_matrices as ccm
from conpagnon.utils.folders_and_files_management import load_object, save_object
from sklearn.covariance import LedoitWolf
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
from conpagnon.plotting.display import plot_matrix
import os
from conpagnon.data_handling import atlas

saving_data_dir = '/media/db242421/db242421_data/ConPagnon_data/' \
                  'language_study_ANOVA_ACM_controls_new_figures/' \
                  'discriminative_connection_identification/Lesion_flip_TDC'

# Load the time series dictionary
times_series = load_object(os.path.join(saving_data_dir,
                                        'times_series_individual_atlases_Lesion_flip_MCA_TDC.pkl'))

groups = list(times_series.keys())
# For some reason, you may want to discard some subjects (optional), comment with #
# if you don't
# subjects_to_drop = ['sub13_vl110480', 'sub14_rs120006', 'sub43_mc130373', 'sub40_np130304',
#                    'sub44_av130474', 'sub02_rf110332', 'sub03_mc120272', 'sub18_mg110111',
#                    'sub01_rm110247', ]

# for subject in subjects_to_drop:
#    times_series['patients'].pop(subject, None)

# Choice of one or several connectivity metrics
metrics = ['correlation', 'partial correlation', 'tangent']

# Compute connectivity matrices for each subjects, and store it into a dictionary
connectivity_matrices = ccm.individual_connectivity_matrices(
    time_series_dictionary=times_series,
    kinds=metrics,
    covariance_estimator=LedoitWolf(),
    z_fisher_transform=False)

# Save the connectivity matrices dictionary
save_object(object_to_save=connectivity_matrices,
            saving_directory=saving_data_dir,
            filename='connectivity_matrices_' + groups[0] + '_' + groups[1] + '.pkl')

# Compute the mean connectivity matrices of each groups in
# the dictionary
mean_connectivity_matrices = ccm.group_mean_connectivity(
    subjects_connectivity_matrices=connectivity_matrices,
    kinds=metrics)

# Save the mean connectivity matrices dictionary
save_object(object_to_save=mean_connectivity_matrices,
            saving_directory=saving_data_dir,
            filename='mean_connectivity_matrices_' + groups[0] + '_' + groups[1] + '.pkl')


# Illustration: time series plot
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

with PdfPages('/media/db242421/db242421_data/Presentation/Royaumont/mean_matrices.pdf') as pdf:
    for group in mean_connectivity_matrices.keys():
        for metric in metrics:
            plt.figure()
            m = mean_connectivity_matrices[group][metric]
            np.fill_diagonal(m, 0)
            plot_matrix(matrix=m,
                        linewidths=0,
                        linecolor='black',
                        title='mean {} matrix ({})'.format(group, metric),
                        horizontal_labels=labels_regions,
                        vertical_labels=labels_regions,
                        labels_colors=labels_colors,
                        labels_size=8, mpart='all',
                        vmax=+np.abs(np.max(m)),
                        vmin=-np.abs(np.max(m)))
            pdf.savefig()
            plt.show()
            plt.close('all')
