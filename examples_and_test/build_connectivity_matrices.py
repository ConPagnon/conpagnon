from computing import compute_connectivity_matrices as ccm
from utils.folders_and_files_management import load_object, save_object
from sklearn.covariance import LedoitWolf

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
