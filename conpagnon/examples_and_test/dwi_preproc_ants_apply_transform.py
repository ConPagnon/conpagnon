"""
 Created by db242421 at 27/02/19

 """
from subprocess import Popen, PIPE
import os
import shutil


"""
AntsApplyTransform and tensor fitting for preprocessed diffusion data.
"""

# Directory containing the subjects
# image folder
root_data_directory = "/neurospin/grip/protocols/MRI/Ines_2018/images/controls_v2"
# Subject text list
subject_txt = "/neurospin/grip/protocols/MRI/Ines_2018/subjects.txt"
subjects = open(subject_txt).read().split()

# Dti directory name
dti_directory = "diffusion"
# T1 directory name
T1_directory = "T1"

eddy_n_threads = 12
ants_n_threads = 14

# Loop over all the subjects
for subject in subjects:
    dti_output = os.path.join(root_data_directory, subject, dti_directory, "nifti")
    motion_corrected_directory = os.path.join(root_data_directory, subject,
                                              'motion_corrected_data')
    t1_output = os.path.join(root_data_directory, subject, T1_directory, "nifti")
    resampled_t1_inverted_image = os.path.join(t1_output,
                                               'r_inverted_robex_t1_' + subject + '.nii.gz')

    # Apply the displacement field to the diffusion data
    apply_transform_to_dti = 'antsApplyTransforms -d 3 -e 3 -t {} -o {} -r {} -i {} ' \
                             '-n BSpline --float 1'.format(
                               os.path.join(dti_output,
                                            subject + '_b0_to_T1_' + 'CollapsedWarp.nii.gz'),
                               os.path.join(motion_corrected_directory,
                                            'distortion_and_eddy_corrected_dti_' + subject + '.nii.gz'),
                               resampled_t1_inverted_image,
                               os.path.join(motion_corrected_directory, 'eddy_corrected_dti_' + subject + '.nii.gz')
                                                                    )
    apply_transform_to_dti_process = Popen(apply_transform_to_dti.split(), stdout=PIPE)
    apply_transform_to_dti_output, apply_transform_to_dti_error = apply_transform_to_dti_process.communicate()

    # Before the end, apply dtifit to check FA map for example, and verify
    # pre-processing steps are well executed
    dtifit_directory = os.path.join(root_data_directory, subject, 'dtifit')
    if os.path.exists(dtifit_directory):
        shutil.rmtree(dtifit_directory)
    os.makedirs(dtifit_directory)

    new_eddy_directory = os.path.join(motion_corrected_directory, 'eddy_outputs')

    eddy_mask = os.path.join(new_eddy_directory, 'eddy_mask.nii')
    eddy_b_vals = os.path.join(new_eddy_directory, 'bvals')
    eddy_b_vecs = os.path.join(new_eddy_directory, 'bvecs')

    dtifit = 'dtifit -k {} -o {} -m {} -r {} -b {}'.format(
        os.path.join(motion_corrected_directory,
                     'distortion_and_eddy_corrected_dti_' + subject + '.nii.gz'),
        os.path.join(dtifit_directory, 'dtifit_' + subject),
        eddy_mask,
        eddy_b_vecs,
        eddy_b_vals
    )
    dtifit_process = Popen(dtifit.split(), stdout=PIPE)
    dtifit_process_output, dtifit_process_error = dtifit_process.communicate()
