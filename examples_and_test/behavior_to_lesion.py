# Code to link structural damage to behavioral scores
# @uthor: Dhaif BEKHA.

import pandas as pd
import numpy as np
from nilearn.image import load_img
import os
import glob
import nibabel as nb
# Lesion root directory
lesions_root_directory = "/media/db242421/db242421_data/AVCnn_2016_DARTEL/AVCnn_data/patients"
subjects_list_txt = "/media/db242421/db242421_data/ConPagnon_data/text_data/acm_patients.txt"
subjects_list = pd.read_csv(subjects_list_txt, header=None)[0]


all_images = []
all_lesion_flatten = []
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
    all_lesion_flatten.append(np.array(subject_lesion_array.flatten(), dtype=np.int8))

flatten_lesions = np.array(all_lesion_flatten, dtype=np.int8)
all_lesion_array = np.array(all_images, dtype=np.int8)

# Compute lesion overlap image
lesion_overlap_array = np.sum(all_lesion_array, axis=0)

# Save image
lesion_overlap_nifti = nb.Nifti1Image(dataobj=lesion_overlap_array, affine=lesion_affine)
nb.save(img=lesion_overlap_nifti, filename="/media/db242421/db242421_data/ConPagnon_reports/overlap_acm.nii")

# Save flatten array
flatten_lesions.tofile("/media/db242421/db242421_data/ConPagnon_data/lesion_map.txt")
np.savetxt('/media/db242421/db242421_data/ConPagnon_data/lesion_map.txt',
           flatten_lesions, delimiter = ',')