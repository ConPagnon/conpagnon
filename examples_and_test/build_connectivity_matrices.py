from computing import compute_connectivity_matrices as ccm
from utils.folders_and_files_management import load_object, save_object
from sklearn.covariance import LedoitWolf
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_handling.data_management import read_csv
from plotting.display import plot_matrix
from nilearn import plotting
# Load the time series dictionary
times_series = load_object('/media/db242421/db242421_data/AICHA_test/times_series_aicha_atlas.pkl')

# For some reason, you may want to discard some subjects (optional), comment with #
# if you don't
subjects_to_drop = ['sub13_vl110480', 'sub14_rs120006', 'sub43_mc130373', 'sub40_np130304',
                    'sub44_av130474', 'sub02_rf110332', 'sub03_mc120272', 'sub18_mg110111',
                    'sub01_rm110247', ]

for subject in subjects_to_drop:
    times_series['patients'].pop(subject, None)

# Choice of one or several connectivity metrics
metrics = ['correlation', 'partial correlation', 'tangent']

# Compute connectivity matrices for each subjects, and store it into a dictionary
connectivity_matrices = ccm.individual_connectivity_matrices(
    time_series_dictionary=times_series,
    kinds=metrics,
    covariance_estimator=LedoitWolf())

# Save the connectivity matrices dictionary
save_object(object_to_save=connectivity_matrices,
            saving_directory='/media/db242421/db242421_data/AICHA_test',
            filename='aicha_connectivity_matrices.pkl')

# Compute the mean connectivity matrices of each groups in
# the dictionary
mean_connectivity_matrices = ccm.group_mean_connectivity(
    subjects_connectivity_matrices=connectivity_matrices,
    kinds=metrics)

# Save the mean connectivity matrices dictionary
save_object(object_to_save=mean_connectivity_matrices,
            saving_directory='/media/db242421/db242421_data/AICHA_test',
            filename='aicha_mean_connectivity_matrices.pkl')


atlas_labels = read_csv(csv_file='/media/db242421/db242421_data/AICHA_test/aicha_labels.csv')['labels']
with PdfPages('/media/db242421/db242421_data/AICHA_test/mean_matrices.pdf') as pdf:
    for group in mean_connectivity_matrices.keys():
        for metric in metrics:
            plt.figure()
            m = mean_connectivity_matrices[group][metric]
            np.fill_diagonal(m, 0)
            plot_matrix(matrix=m,
                        linewidths=0,
                        linecolor='black',
                        title='mean {} matrix ({})'.format(group, metric),
                        horizontal_labels=atlas_labels,
                        vertical_labels=atlas_labels,
                        labels_size=2.5, mpart='all',
                        vmax=+np.abs(np.max(m)),
                        vmin=-np.abs(np.max(m)))
            pdf.savefig()
            plt.show()
            plt.close('all')


with PdfPages('/media/db242421/db242421_data/AICHA_test/mean_difference_matrices.pdf') as pdf:
    for metric in metrics:
        mean_difference_connectivity = \
            mean_connectivity_matrices['controls'][metric] - mean_connectivity_matrices['patients'][metric]
        plt.figure()
        plot_matrix(matrix=mean_difference_connectivity, linewidths=0,
                    linecolor='black', title='mean {} matrix ({})'.format('patients - controls', metric),
                    horizontal_labels=atlas_labels, vertical_labels=atlas_labels,
                    labels_size=2.5, mpart='all')

        pdf.savefig()
        plt.show()
        plt.close('all')

import numpy as np
plt.figure()
M = mean_connectivity_matrices['patients']['tangent']
plotting.plot_matrix(mat=M,
                     vmin=-np.abs(np.max(M)),
                     vmax=+np.abs(np.max(M)))
plt.show()

plt.figure()
plot_matrix(matrix=M, linewidths=0,
            linecolor='black',
            horizontal_labels=atlas_labels, vertical_labels=atlas_labels,
            labels_size=2.5, mpart='all', vmax=+np.abs(np.max(M)),
            vmin=-np.abs(np.max(M)))
plt.show()