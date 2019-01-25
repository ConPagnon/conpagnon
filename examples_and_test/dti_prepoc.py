"""
 Created by db242421 on 25/01/19
"""
import pandas as pd
import glob
from subprocess import Popen, PIPE
import numpy as np
from nilearn.image import load_img
import os

"""
This script performs the pre-processing of DTI 
data. It follows several step:
    1) Conversion of DICOM dti images to the
    NifTi format.
    
    2) Correction of motion and eddy current
    with fsl_eddy for the DTI image.
    
    3) Skull Stripping of the T1 image.
    
    4) Bias correction of the T1 image.
    
    5) Contrast inversion of the T1 image.
    
    6) Normalisation of the b0 image on 
    inverted skull stripped T1 image using 
    Symmetric Normalisation of the ANTs tools.
    
    7) Warp the weighted diffusion image applying
    the transformation previously computed.

End of the pre-processing steps.
"""

# Directory containing the subjects
# image folder
root_data_directory = "/media/db242421/db242421_data/DTI_preproc_state_Ines"
# Subject text list
subject_txt = "/media/db242421/db242421_data/DTI_preproc_state_Ines/text/subjects.txt"
subjects = list(pd.read_csv(subject_txt))

# Dti directory name
dti_directory = "diffusion"
# T1 directory name
T1_directory = "T1"

# Loop over all the subjects
for subject in subjects:
    # TODO: Use dcm2niix instead of dcm2nii
    # Convert Dicom to nifti image
    subject_dicom_dti = os.path.join(root_data_directory, subject, dti_directory, "dicom")
    dti_output = os.path.join(root_data_directory, subject, dti_directory, "nifti")

    dcm2nii_dti = "dcm2nii -o {} -t {} -i {} {}".format(dti_output, "y", "test", subject_dicom_dti)

    dc2mii_dti_process = Popen(dcm2nii_dti.split(), stdout=PIPE)
    dc2mii_dti_output, dc2nii_dti_error = dc2mii_dti_process.communicate()

