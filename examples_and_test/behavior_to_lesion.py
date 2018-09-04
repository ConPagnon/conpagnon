# Code to link structural damage to behavioral scores
# Author: Dhaif BEKHA.
import pandas as pd
import numpy as np
from nilearn.image import load_img
import os
import glob
import nibabel as nb
from utils.folders_and_files_management import save_object
from sklearn.preprocessing import StandardScaler
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.decomposition import PCA

# Set path, read subjects list
lesions_root_directory = "/media/db242421/db242421_data/AVCnn_2016_DARTEL/AVCnn_data/patients"
subjects_list_txt = "/media/db242421/db242421_data/ConPagnon_data/text_data/acm_patients.txt"
subjects_list = pd.read_csv(subjects_list_txt, header=None)[0]
saving_directory = "/media/db242421/db242421_data/ConPagnon_reports/resultsPCA"

all_images = []
all_lesion_flatten = []

# Load shape and affine of one normalized lesion map
someone_lesion_map = load_img(os.path.join(lesions_root_directory,
                                           "sub44_av130474/lesion/wav130474_lesion_totale_s16.nii"))
target_affine = someone_lesion_map.affine
target_shape = someone_lesion_map.shape

# Read and load lesion image for each subject
for subject in subjects_list:
    # path to the subject image
    subject_lesion_path = os.path.join(lesions_root_directory, subject, 'lesion')
    # Get lesion filename
    lesion_file = glob.glob(os.path.join(subject_lesion_path, 'w*.nii'))[0]
    # Load the normalized lesion image
    subject_lesion = load_img(img=lesion_file)
    # get lesion images affine
    lesion_affine = subject_lesion.affine
    # load image into an array
    subject_lesion_array = subject_lesion.get_data()
    # Append lesion array to all images list
    all_images.append(subject_lesion_array)
    # flatten the array for a PCA later on
    all_lesion_flatten.append(np.array(subject_lesion_array.flatten(),
                                       dtype=np.int8))

lesions_maps = np.array(all_lesion_flatten, dtype=np.int8)
all_lesion_array = np.array(all_images, dtype=np.double)

# Compute lesion overlap image
lesion_overlap_array = np.sum(all_lesion_array, axis=0)

# Save lesions maps overlay
lesion_overlap_nifti = nb.Nifti1Image(dataobj=lesion_overlap_array,
                                      affine=target_affine)
nb.save(img=lesion_overlap_nifti,
        filename="/media/db242421/db242421_data/ConPagnon_reports/overlap_acm.nii")

# Save flatten lesion map
save_object(object_to_save=lesions_maps, 
            saving_directory=saving_directory,
            filename='lesion_maps_acm.pkl')
# Save flatten lesion array in numpy format
np.save(os.path.join(saving_directory, "lesion_maps.npy"),
        arr=lesions_maps)
# Perform principal components analysis on
# lesion map
sc = StandardScaler()
scaled_lesion_maps = sc.fit_transform(lesions_maps)

pca = PCA(n_components=0.9)
scaled_lesion_maps = pca.fit_transform(scaled_lesion_maps)

explained_variance = pca.explained_variance_ratio_
cumulative_sum_of_variance = explained_variance.cumsum()

# Construct a dataframe with the PCA result
columns_names = ['PC{}'.format(i) for i in range(1, len(cumulative_sum_of_variance)+1)]
pca_results_df = pd.DataFrame(data=scaled_lesion_maps,
                              columns=columns_names)
pca_results_df.head()
# rename index with subject identifier
pca_results_df = pca_results_df.rename(index=subjects_list)
pca_results_df.to_csv(index_label="subjects",
                      path_or_buf=os.path.join(saving_directory, "pca_lesion_map.csv"))

# It interesting to see the reconstruction based on the principal components in
# the image space
inverse_lesion_maps = pca.inverse_transform(scaled_lesion_maps)
# Reshape to image format
inverse_lesions_maps_array = np.zeros((inverse_lesion_maps.shape[0],
                                       target_shape[0], target_shape[1],
                                       target_shape[2]))
for s in range(inverse_lesion_maps.shape[0]):
    inverse_lesions_maps_array[s, ...] = np.reshape(inverse_lesion_maps[s, ...],
                                                    newshape=target_shape)

# Compute the reconstructed overlap image
reconstructed_overlap_lesions = np.sum(inverse_lesions_maps_array, axis=0)
reconstructed_overlap_lesions_img = nb.Nifti1Image(dataobj=reconstructed_overlap_lesions,
                                                   affine=target_affine)
nb.save(img=reconstructed_overlap_lesions_img,
        filename=os.path.join(saving_directory, "reconstructed_overlap_acm.nii"))

# Add clinical variable
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive'
         ]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    '/media/db242421/db242421_data/ConPagnon_data/text_data/avccn-b1a97159e635.json',
    scope)
client = gspread.authorize(creds)

# Access AVCnn spreadsheets
acm_data_spreadsheet = client.open('Resting State AVCnn: cohort data').sheet1
