"""
 Created by Dhaif BEKHA, (dhaif.bekha@cea.fr).

 """
import os
import pandas as pd
from subprocess import Popen, PIPE
import glob
import warnings
from nilearn.image import resample_to_img, load_img
import numpy as np
import nibabel as nb
from pytictoc import TicToc
"""

Goal
----
After the creation of the population specific 
FA template with tbss_create_controls_template.py, 
this script aims to register the controls and patients
FA map to the FA template. For patients, we used the lesion
mask to improve the accuracy of the registration.
All registration computation is done with the help of 
ANTs.

Data directory architecture
---------------------------
The data should be organised in a certain manner for this
code to work properly:

- root_data_directory
|____ subject_1_directory
     |___ FA_map_directory 
          |___ *_FA.nii.gz
          |___ *_V1.nii.gz
          |___ *_V2.nii.gz
          |___ *_V3.nii.gz
          |___ *_L1.nii.gz
          |___ *_L2.nii.gz
          |___ *_L3.nii.gz
     |___ subject_mask_file 
|____ subject_2_directory
     |___ FA_map_directory
         .....
         
The root directory contains all the subject folder, named
with the subject identifiers in you're subject list txt file.
In each folder, you should have a directory containing the output
of DTIFIT from FSL. Each file should at least ended with the computed
metric (FA, MD, V1, ...). If needed, one should have a mask 
for the registration, in the subject folder.
     
Step 0
------
This is the only step: we call AntsRegistration 
to register the native FA subject image to the template
computed in the previous step. If needed, one can use 
a mask for registration, make sure to modify the boolean
use_mask to True. Each registration registration results 
are written in a directory called ants_output in each subject
directory. 
It's a classic ANTs call: A rigid transformation, followed
by affine transformation, ending with the nonlinear SyN transform
developed in the package. This step can be very long. All other 
parameter are default's.
Note that for a mask, ANTs as multiple options: by default,
you must have defined in the FIXED space (target image). Or you're
can provide a mask for the moving AND fixed image. For now, in this
code only the first option is supported. That's why in the call
of AntsRegistration we register the template on the FA map, and not
the inverse, because the mask you have most likely be defined in the
native space. When Applying transform, we simply take the inverse path.
Ants compute automatically the warped image in the template space. We move
the warped image at the root the of the subject folder.

"""
# As usual, some directory path

# root directory containing groups subject directory
fa_maps_root_directory = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/fa_map_registrations'
# group directory name
group_directory = 'controls'
# FA map directory name
fa_map_directory = 'dtifit'
# text file containing subject identifier
subjects_list_txt = '/neurospin/grip/protocols/MRI/Ines_2018/controls_list.txt'
subjects = list(pd.read_csv(subjects_list_txt,
                            header=None)[0])
# If you need to use a mask during registration, put True
# and False instead
use_mask = False
# the mask should be in the subject directory,
# alongside the folder containing the native FA map
mask_filter = 'negative_*.nii.gz'

# High resolution FA template path
FA_template = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/controls_dtitk/' \
              'final_template/mean_final_high_res_fa.nii.gz'

for subject in subjects:
    t = TicToc()
    t.tic()
    print("Registering FA for subject {} ...\n".format(subject))
    # Full path to the subject FA map
    subject_FA = os.path.join(fa_maps_root_directory, group_directory,
                              subject, fa_map_directory, 'dtifit_' + subject + '_FA.nii.gz')
    # Create a directory to store ANTs output
    subject_ants_output = os.path.join(fa_maps_root_directory, group_directory,
                                       subject, 'ants_output')
    if 'ants_output' not in os.listdir(os.path.join(fa_maps_root_directory, group_directory,
                                                    subject)):
        os.mkdir(subject_ants_output)
    # If the directory already delete all the file to avoid
    # potential conflict in filename
    else:
        for file in os.listdir(subject_ants_output):
            os.unlink(os.path.join(subject_ants_output, file))
    if use_mask is False:
        # Compute registration between the subject FA map and the template
        ants_registration = [
            "antsRegistration",
            "--dimensionality", "3",
            "--float", "0",
            "--output", "[{},{},{}]".format(os.path.join(subject_ants_output, subject + '_FA_'),
                                            os.path.join(subject_ants_output, subject + '_FA_Warped.nii.gz'),
                                            os.path.join(subject_ants_output, subject + '_FA_InverseWarped.nii.gz')),
            "--interpolation", "Linear",
            "--winsorize-image-intensities", "[0.005,0.995]",
            "--use-histogram-matching", "0",
            "--initial-moving-transform", "[{},{},1]".format(FA_template, subject_FA),
            "--transform", "Rigid[0.1]",
            "--metric", "MI[{},{},1,32,Regular,0.25]".format(FA_template, subject_FA),
            "--convergence", "[1000x500x250x100,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
            "--transform", "Affine[0.1]",
            "--metric", "MI[{},{},1,32,Regular,0.25]".format(FA_template, subject_FA),
            "--convergence", "[1000x500x250x100,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
            "--transform", "SyN[0.1,3,0]",
            "--metric", "CC[{},{},1,4]".format(FA_template, subject_FA),
            "--convergence", "[100x70x50x20,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
            "--verbose", "1"
        ]

        # Perform EPI to T1 registration
        ants_registration_process = Popen(ants_registration, stdout=PIPE)
        ants_registration_output, ants_registration_error = ants_registration_process.communicate()

        with open(os.path.join(subject_ants_output, subject + '_FA_register_ants_output.txt'), "w") as text_file:
            text_file.write(ants_registration_output.decode('utf-8'))

        # Move the warped FA image to the root subject directory
        warped_FA = os.path.join(subject_ants_output, subject + '_FA_Warped.nii.gz')
        move_warped_FA = ["cp", warped_FA, os.path.join(fa_maps_root_directory, group_directory, subject,
                                                        subject + "_warped_FA.nii.gz")
                          ]

        move_warped_FA_process = Popen(move_warped_FA, stdout=PIPE)
        move_warped_FA_process_output, move_warped_FA_process_error = move_warped_FA_process.communicate()
    else:
        # Fetch the subject mask
        list_of_mask = glob.glob(os.path.join(fa_maps_root_directory, group_directory,
                                              subject, mask_filter))

        if list_of_mask != 1:
            warnings.warn(message='Warning !! Multiple potential mask image found, \n '
                                  'taking the first one {}'.format(list_of_mask[0].split(os.sep))[-1])
            subject_mask = list_of_mask[0]
        else:
            subject_mask = list_of_mask[0]
        # Check that the mask is in the same space
        mask_affine = load_img(img=subject_mask).affine
        native_FA_affine = load_img(subject_FA).affine
        # Test if affine are equal
        affine_are_equal = np.allclose(mask_affine, native_FA_affine)
        # If not, resample the mask to the subject FA map in the
        # native space
        if affine_are_equal is False:
            resampled_mask = resample_to_img(source_img=subject_mask, target_img=subject_FA,
                                             interpolation='nearest')
            # save the resampled mask
            nb.save(img=resampled_mask,
                    filename=os.path.join(fa_maps_root_directory, group_directory, subject,
                                          'resampled_negative_lesion_' + subject + '.nii.gz'))
            subject_mask = os.path.join(fa_maps_root_directory, group_directory, subject,
                                        'resampled_negative_lesion_' + subject + '.nii.gz')

        # Compute registration between the subject FA map and the template: note
        # that this time, the template is the moving image, and the FA image
        # is the fixed image. The result we interested in, is the FA image
        # in the template space, this time stored in the file ending with
        # inverseWarped.nii.gz. The subject mask, is given by the -x
        # input argument.
        ants_registration = [
            "antsRegistration",
            "--dimensionality", "3",
            "--float", "0",
            "--output", "[{},{},{}]".format(os.path.join(subject_ants_output, subject + '_Template_to_FA_'),
                                            os.path.join(subject_ants_output, subject + '_Template_Warped.nii.gz'),
                                            os.path.join(subject_ants_output, subject + '_Template_InverseWarped.nii.gz')),
            "--interpolation", "Linear",
            "--winsorize-image-intensities", "[0.005,0.995]",
            "--use-histogram-matching", "0",
            "--initial-moving-transform", "[{},{},1]".format(subject_FA, FA_template),
            "--transform", "Rigid[0.1]",
            "--metric", "MI[{},{},1,32,Regular,0.25]".format(subject_FA, FA_template),
            "--convergence", "[1000x500x250x100,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
            "--transform", "Affine[0.1]",
            "--metric", "MI[{},{},1,32,Regular,0.25]".format(subject_FA, FA_template),
            "--convergence", "[1000x500x250x100,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
            "--transform", "SyN[0.1,3,0]",
            "--metric", "CC[{},{},1,4]".format(subject_FA, FA_template),
            "--convergence", "[100x70x50x20,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
            "--verbose", "1",
            "--collapse-output-transforms", "1",
            "-x", subject_mask
        ]

        # Perform EPI to T1 registration
        ants_registration_process = Popen(ants_registration, stdout=PIPE)
        ants_registration_output, ants_registration_error = ants_registration_process.communicate()

        with open(os.path.join(subject_ants_output, subject + '_FA_register_ants_output.txt'), "w") as text_file:
            text_file.write(ants_registration_output.decode('utf-8'))

        # Move the warped FA image to the root subject directory. Note
        # that we move the Inverse warped image, which is the FA image
        # in the template space.
        warped_FA = os.path.join(subject_ants_output, subject + '_Template_InverseWarped.nii.gz')
        move_warped_FA = ["cp", warped_FA, os.path.join(fa_maps_root_directory, group_directory, subject,
                                                        subject + "_warped_FA.nii.gz")
                          ]
        move_warped_FA_process = Popen(move_warped_FA, stdout=PIPE)
        move_warped_FA_process_output, move_warped_FA_process_error = move_warped_FA_process.communicate()
        t.toc()
"""
End of the scripts, all FA map are now registered to the
mean FA template. The next step will be to properly prepare 
the data for the statistical analysis with the FSL software.
"""
