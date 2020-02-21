"""
 Created by db242421 at 06/03/19

 """

"""
 Created by db242421 at 27/02/19

 """
import nibabel as nb
import os
from nilearn.image import resample_to_img, load_img


"""
AntsApplyTransform and tensor fitting for preprocessed diffusion data.
"""

# Directory containing the subjects
# image folder
root_data_directory = "/neurospin/grip/protocols/MRI/Ines_2018/images/patients"
# Subject text list
subject_txt = "/media/db242421/db242421_data/DTI_preproc_state_Ines/images/patients/patients_acm.txt"
subjects = open(subject_txt).read().split()

# Dti directory name
dti_directory = "diffusion"
# T1 directory name
T1_directory = "T1"
# Lesion folder
lesion_directory = "lesion"

subjects.pop(subjects.index("jc130100"))
subject = 'jc130100'
for subject in subjects:

    t1 = os.path.join(root_data_directory, subject, T1_directory, "nifti", "t1_" + subject + ".nii.gz")
    t1_affine = load_img(t1).affine
    lesion = os.path.join(root_data_directory, subject, lesion_directory, subject + "_lesion_totale_s16.nii.gz")
    lesion_resampled = resample_to_img(lesion, t1, "nearest")
    nb.save(lesion_resampled, filename=os.path.join(root_data_directory, subject, lesion_directory,
                                                    "r_" + subject + "_lesion_totale_s16.nii.gz"))
