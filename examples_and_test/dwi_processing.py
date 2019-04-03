"""
 Created by Dhaif BEKHA, (dhaif.bekha@cea.fr).

 """
import os
import pandas as pd
from subprocess import PIPE, Popen

"""
This script perform pre-processing 
and orientation density function estimation 
based on spherical deconvolution as implemented
in MRTrix3.

"""

# Set paths
root_directory = '/media/db242421/db242421_data/antoine_test_batch'
subjects_list_txt = '/media/db242421/db242421_data/antoine_test_batch/subjects.txt'
subjects = list(pd.read_csv(subjects_list_txt, header=None)[0])


# dwi_preproc parameters
phase_encoding_direction = "PA"
readout_time = 0.0759809

# general parameters
# number of threads
nthreads = 8

for subject in subjects:
    print("Processing subject {}".format(subject))
    # raw dwi image
    subject_dwi = os.path.join(root_directory, subject, "dwi_b1500_PA.mif")
    # denoise raw dwi image
    denoise_dwi = ["dwidenoise",
                   subject_dwi,
                   os.path.join(root_directory, subject, "dwi_b1500_PA_denoise.mif",
                                ),
                   "-noise",
                   os.path.join(root_directory, subject, "noise.mif"),
                   "-force",
                   "-nthreads", str(nthreads),
                   ]
    denoise_dwi_command = Popen(denoise_dwi, stdout=PIPE)
    denoise_dwi_output, denoise_dwi_error = denoise_dwi_command.communicate()

    # remove Gibbs artefacts
    unring_dwi = ["mrdegibbs",
                  os.path.join(root_directory, subject, "dwi_b1500_PA_denoise.mif",
                               ),
                  os.path.join(root_directory, subject, "dwi_b1500_PA_denoise_unring.mif",
                                ),
                  "-axes",
                  "0,1",
                  "-force",
                  "-nthreads", nthreads
                  ]
    unring_dwi_command = Popen(unring_dwi, stdout=PIPE)
    unring_dwi_output, unring_dwi_error = unring_dwi_command.communicate()

    # Full path to the denoised dwi image
    subject_denoised_dwi = os.path.join(root_directory, subject, "dwi_b1500_PA_denoise_unring.mif")

    # output preprocessed dwi image
    subject_preproc_dwi = os.path.join(root_directory, subject, "dwi_preprocessed.mif")
    # non weighted diffusion images (b=0)
    b0s = os.path.join(root_directory, subject, "b0s.mif")
    # Preprocess raw diffusion data with dwi_preproc
    dwi_preproc = ["dwipreproc",
                   subject_dwi,
                   subject_preproc_dwi,
                   "-pe_dir", phase_encoding_direction,
                   "-rpe_pair",
                   "-se_epi",
                   b0s,
                   "-readout_time", str(readout_time),
                   "-eddy_options", "--repol ",
                   "-tempdir", os.path.join(root_directory, subject),
                   "-force",
                   "-nocleanup"
                   ]
    dwi_preproc_command = Popen(dwi_preproc, stdout=PIPE)
    dwi_preproc_output, dwi_preproc_error = dwi_preproc_command.communicate()

