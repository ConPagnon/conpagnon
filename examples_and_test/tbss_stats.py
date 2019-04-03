"""
 Created by Dhaif BEKHA, (dhaif.bekha@cea.fr).

 """
import os
import pandas as pd
from subprocess import Popen, PIPE
from nilearn.image import load_img, resample_to_img
import nibabel as nb
"""
This script performs the statistical 
analysis for the TBSS pipeline. 

For the lesioned data, voxels inside 
the lesion mask of each subject are useless.
Inside randomise function from FSL, there is 
a option to mask those voxels during the analyses.

Step 0
------
Project lesion mask onto the FA template space, and
prepare those mask to be feed to the randomise function
from FSL.

"""

# lesions root directory
lesions_root_directory = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/' \
                          'fa_map_registrations/patients'
# patients list in a text file
subjects_list_txt = '/neurospin/grip/protocols/MRI/Ines_2018/images/patients/patients_acm.txt'
subjects = list(pd.read_csv(subjects_list_txt, header=None)[0])

# Full path to the high resolution FA template
FA_template = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/stats/mean_FA.nii.gz'

# full path to the folder that will contain the
# normalized lesion mask for each patient
all_lesions_directory = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/all_lesions'

for subject in subjects:
    print('Projection lesion mask in template space for subject {}'.format(subject))
    # Collapsed the affine+Rigid transform with
    # the non linear field into one transform
    subject_ants_output = os.path.join(lesions_root_directory, subject, 'ants_output')
    # affine transformation
    affine_transformation = os.path.join(subject_ants_output, subject + '_Template_to_FA_0GenericAffine.mat')
    # non-linear transformation
    non_linear_warp = os.path.join(subject_ants_output, subject + '_Template_to_FA_1InverseWarp.nii.gz')
    # Apply the collapsed transformation to the subject lesion image
    subject_lesion_mask = os.path.join(lesions_root_directory, subject,
                                       subject + '_lesion_totale_s16.nii.gz')
    # resample the lesion mask to the native FA
    FA_template_affine = load_img(FA_template).affine
    resampled_lesion_mask = resample_to_img(source_img=subject_lesion_mask,
                                            target_img=os.path.join(lesions_root_directory, subject, 'dtifit',
                                                                    'dtifit_' + subject + '_FA.nii.gz'),
                                            interpolation='nearest')
    nb.save(resampled_lesion_mask, os.path.join(lesions_root_directory, subject,
                                                'resampled_' + subject + '_lesion_mask.nii.gz'))
    # Apply the displacement field to the diffusion data
    apply_transform_to_lesion = 'antsApplyTransforms -d 3 -r {} -t [{},1] -t {} -n NearestNeighbor -i {}' \
                                '-o {}'.format(FA_template, affine_transformation, non_linear_warp,
                                               os.path.join(lesions_root_directory, subject,
                                                            'resampled_' + subject + '_lesion_mask.nii.gz'),
                                               os.path.join(all_lesions_directory,
                                                            subject + '_warped_lesion.nii.gz')
                                               )
    apply_transform_to_lesion_ = ["antsApplyTransforms",
                                  "-d", "3",
                                  "-r", "{}".format(FA_template),
                                  "-t", "[{},1]".format(affine_transformation),
                                  "-t", "{}".format(non_linear_warp),
                                  "-n", "NearestNeighbor",
                                  "-i", "{}".format(os.path.join(lesions_root_directory, subject,
                                                                 'resampled_' + subject + '_lesion_mask.nii.gz')),
                                  "-o", "{}".format(os.path.join(all_lesions_directory,
                                                                 subject + '_warped_lesion.nii.gz'))
                                  ]

    apply_transform_to_lesion_process = Popen(apply_transform_to_lesion_, stdout=PIPE)
    apply_transform_to_lesion_output, apply_transform_to_lesion_error = \
        apply_transform_to_lesion_process.communicate()
