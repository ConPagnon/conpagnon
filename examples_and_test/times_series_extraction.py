from data_handling import data_architecture
from utils import folders_and_files_management
from computing import compute_connectivity_matrices as ccm
import time
from utils.folders_and_files_management import save_object
from data_handling import atlas
import matplotlib.pyplot as plt

"""
Example code: time series extraction with individual atlases

"""

# Groups name to include in the study
groups = [ 'patients', 'controls']
# The root fmri data directory containing all the fmri files directories
root_fmri_data_directory = \
    '/media/db242421/db242421_data/ConPagnon_data/fmri_images'
# Check all functional files existence
folders_and_files_management.check_directories_existence(
    root_directory=root_fmri_data_directory,
    directories_list=groups)

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

# Fetch data in the case of individual atlases for each subjects.
organised_data_with_individual_atlas = data_architecture.fetch_data_with_individual_atlases(
    subjects_id_data_path=subjects_ID_data_path,
    root_fmri_data_directory=root_fmri_data_directory,
    groupes=groups,
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
    groupes=groups,
    subjects_id_data_path=subjects_ID_data_path,
    group_data=organised_data_with_individual_atlas,
    repetition_time=repetition_time,
    nilearn_cache_directory=nilearn_cache_directory,
    resampling_target='data')
end = time.time()
total_extraction_ts = (end - start)/60.

# Save the subjects time series dictionary
save_object(object_to_save=times_series_individual_atlases,
            saving_directory='/media/db242421/db242421_data/Presentation/Royaumont',
            filename='times_series_individual_atlases_patients_controls.pkl')


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

# Take one subject image for illustration
# purpose
one_subjects_time_series = times_series_individual_atlases['controls']['sub27_ea130507']['time_series']
for region in range(n_nodes):
    plt.figure()
    plt.plot(range(0, one_subjects_time_series.shape[0]),
             one_subjects_time_series[:,region], '-', color=tuple(labels_colors[region]))
    plt.xlabel('Temps')
    plt.ylabel('Intensité normée')
    plt.title(labels_regions[region])
    plt.savefig('/media/db242421/db242421_data/Presentation/Royaumont/' + labels_regions[region] + '.png')
    plt.show()
