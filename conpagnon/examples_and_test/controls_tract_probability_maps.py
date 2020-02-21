"""
 Created by Dhaif BEKHA, (dhaif.bekha@cea.fr).

 """
import pandas as pd
import os
from subprocess import Popen, PIPE
import nibabel as nb
import numpy as np

"""
This script compute tract probability
maps for the NAIS contols group. The target
space is an FA template build from the same 
controls cohort. We will use the already
computed transform to this space for registering
our bundle mask. 
If you need to move the bundle mask to another space,
you will need to first compute the transformation from 
you're image to the template space, and then apply the
corresponding transformation. You will need to adapt the
code.
"""

# Root directory of the architecture of you're database
root_directory = "/neurospin/grip/protocols/MRI/Ines_2018/images/controls_v2"

# Path to the subjects list in text file, with one column per
# subject identifiers.
subjects_list_txt = "/neurospin/grip/protocols/MRI/Ines_2018/images/controls_v2_nip"
subjects = list(pd.read_csv(subjects_list_txt, header=None)[0])

# Bundle mask list to register
bundle_mask = ["AF_left.nii.gz", "AF_right.nii.gz"]

# Bundle masks directory architecture
bundles_mask_directories = "TractSeg_outputs/tractseg_output/bundle_segmentations_mask"

# Tract analysis directory name
tract_analysis_directory = "tract_analysis"

# Target template
template = "/neurospin/grip/protocols/MRI/Ines_2018/images/tbss_lvl2/stats/mean_FA.nii.gz"


for subject in subjects:
    print('Projection bundle mask in template space for subject {}'.format(subject))
    for bundle in bundle_mask:
        # Collapsed the affine+Rigid transform with
        # the non linear field into one transform
        subject_ants_output = os.path.join(root_directory, subject, 'tbss_lvl1', 'ants_output')
        # affine transformation
        affine_transformation = os.path.join(subject_ants_output, subject + '_FA_0GenericAffine.mat')
        # non-linear transformation
        non_linear_warp = os.path.join(subject_ants_output, subject + '_FA_1Warp.nii.gz')
        # full path to the bundle mask
        subject_bundle_mask = os.path.join(root_directory, subject, bundles_mask_directories, bundle)
        # Apply the displacement field to the bundle mask
        apply_transform_to_bundle = ["antsApplyTransforms",
                                      "-d", "3",
                                      "-r", "{}".format(template),
                                      "-t", "{}".format(non_linear_warp),
                                      "-t", "{}".format(affine_transformation),
                                      "-n", "NearestNeighbor",
                                      "-i", "{}".format(subject_bundle_mask),
                                      "-o", "{}".format(os.path.join(root_directory, subject, tract_analysis_directory,
                                                                     bundle[:-7],
                                                                     subject + '_warped_' + bundle[:-7] + '.nii.gz'))
                                      ]

        apply_transform_to_bundle_process = Popen(apply_transform_to_bundle, stdout=PIPE)
        apply_transform_to_bundle_output, apply_transform_to_lesion_error = \
            apply_transform_to_bundle_process.communicate()
        print('Done projecting bundle mask in template space for {}'.format(subject))

# Take the mean of all warped bundle mask
for bundle in bundle_mask:
    print("Compute probability map for bundle {}".format(bundle))
    warped_bundle_mask = [os.path.join(os.path.join(root_directory, subject, tract_analysis_directory,
                                                    bundle[:-7],
                                                    subject + '_warped_' + bundle[:-7] + '.nii.gz'))
                          for subject in subjects]
    warped_bundle_images = map(nb.load, warped_bundle_mask)
    mean_bundle_image = np.array([warped_bundle.get_data() for warped_bundle in warped_bundle_images]).mean(axis=0)
    affine = nb.load(template).affine
    mean_bundle_nifti = nb.Nifti1Image(mean_bundle_image, affine=affine)
    nb.save(mean_bundle_nifti, filename=os.path.join(root_directory, "probability_map_" + bundle))