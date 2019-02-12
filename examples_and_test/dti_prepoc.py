"""
 Created by db242421 on 25/01/19
"""
from subprocess import Popen, PIPE
import numpy as np
from nilearn.image import load_img, resample_to_img, resample_img
import os
import nibabel as nb
import re
import shutil
import json
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

It requires a few dependencies: dcm2niix, FSL and MRtrix3, ANTs, ROBEX.
"""

# Directory containing the subjects
# image folder
root_data_directory = "/media/db242421/db242421_data/DTI_preproc_state_Ines/images/controls"
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
    # Convert Dicom to nifti image
    subject_dicom_dti = os.path.join(root_data_directory, subject, dti_directory, "dicom")
    dti_output = os.path.join(root_data_directory, subject, dti_directory, "nifti")

    dcm2nii_dti = "dcm2niix -z {} -o {}   {}".format("y", dti_output, subject_dicom_dti)

    dc2mii_dti_process = Popen(dcm2nii_dti.split(), stdout=PIPE)
    dc2mii_dti_output, dc2nii_dti_error = dc2mii_dti_process.communicate()

    # Rename the output nifti dti image
    # List all the dti image in the directory, and text file
    dti_list_files = [f for f in os.listdir(dti_output) if re.match(r'[a-z]+.*\.nii*.gz', f)]
    b_vec_file = [f for f in os.listdir(dti_output) if re.match(r'[a-z]+.*\.bvec', f)]
    b_val_file = [f for f in os.listdir(dti_output) if re.match(r'[a-z]+.*\.bval', f)]
    info_json_file = [f for f in os.listdir(dti_output) if re.match(r'[a-z]+.*\.json', f)]

    # Rename the dti image
    os.rename(os.path.join(dti_output, dti_list_files[0]),
              os.path.join(dti_output, 'dti_' + subject + '.nii.gz'))
    # Rename the bvec text file
    os.rename(os.path.join(dti_output, b_vec_file[0]),
              os.path.join(dti_output, 'dti_' + subject + '.bvec'))
    # Rename the bval text file
    os.rename(os.path.join(dti_output, b_val_file[0]),
              os.path.join(dti_output, 'dti_' + subject + '.bval'))
    # Rename the json file
    os.rename(os.path.join(dti_output, info_json_file[0]),
              os.path.join(dti_output, 'dti_' + subject + '_info.json'))

    # Read and extract the unweighted diffusion image and
    # compute brain extraction
    dti_image = os.path.join(dti_output, 'dti_' + subject + '.nii.gz')
    dti_image_ = load_img(img=os.path.join(dti_output, 'dti_' + subject + '.nii.gz'))
    dti_image_data = dti_image_.get_data()
    dti_image_affine = dti_image_.affine
    # Apply motion correction using fsl_eddy function
    bvec = os.path.join(dti_output, 'dti_' + subject + '.bvec')
    bval = os.path.join(dti_output, 'dti_' + subject + '.bval')
    json_file = os.path.join(dti_output, 'dti_' + subject + '_info.json')

    # Padding with zeros to extend the bounding box
    target_affine = np.copy(dti_image_affine)
    target_affine[..., 3] = np.array(np.hstack((target_affine[0:2, 3], [-100, 1])))
    new_img = resample_img(img=dti_image, target_affine=target_affine,
                           interpolation='nearest',
                           target_shape=(128, 128, 70))
    nb.save(new_img,
            os.path.join(dti_output, 'dti_' + subject + '.nii.gz'))

    # Read the Json file, just extracting the phase encoding direction
    with open(json_file) as f:
        json_data = json.load(f)

    phase_encoding_direction = json_data['PhaseEncodingDirection']

    # Create output directory to store the motion corrected data
    motion_corrected_directory = os.path.join(root_data_directory, subject,
                                              'motion_corrected_data')
    if os.path.exists(motion_corrected_directory):
        shutil.rmtree(motion_corrected_directory)
    os.makedirs(motion_corrected_directory)

    # Call dwipreproc from MRtrix package to perform eddy current
    # correction

    dwi_preproc = [
        "dwipreproc", dti_image,
        os.path.join(motion_corrected_directory, 'eddy_corrected_' + 'dti_' + subject + '.nii.gz'),
        "-fslgrad", bvec, bval,
        "-pe_dir", "j-",
        "-rpe_none",
        "-nocleanup",
        "-json_import", json_file,
        "-eddy_options", "--repol --slm=linear",
        "-tempdir", motion_corrected_directory]

    dwi_preproc_process = Popen(dwi_preproc, stdout=PIPE)
    dwi_preproc_output, dwi_preproc_error = dwi_preproc_process.communicate()

    # Rename the directory containing all the output of eddy correction
    eddy_directory = [f for f in os.listdir(motion_corrected_directory)
                      if re.match(r"dwipreproc", f)][0]
    os.rename(os.path.join(motion_corrected_directory, eddy_directory),
              os.path.join(motion_corrected_directory, 'eddy_outputs'))
    new_eddy_directory = os.path.join(motion_corrected_directory, 'eddy_outputs')

    # Quality check of data using eddy quad from FSL
    eddy_output_basename = os.path.join(new_eddy_directory, 'dwi_post_eddy')
    eddy_indices = os.path.join(new_eddy_directory, 'eddy_indices.txt')
    eddy_parameters = os.path.join(new_eddy_directory, 'eddy_config.txt')
    eddy_mask = os.path.join(new_eddy_directory, 'eddy_mask.nii')
    eddy_b_vals = os.path.join(new_eddy_directory, 'bvals')
    eddy_b_vecs = os.path.join(new_eddy_directory, 'bvecs')

    eddy_qc = 'eddy_quad {} -idx {} -par {} -m {} -b {} -o {}'.format(
        eddy_output_basename, eddy_indices, eddy_parameters, eddy_mask,
        eddy_b_vals, os.path.join(motion_corrected_directory, 'quality_check'))
    eddy_qc_process = Popen(eddy_qc.split(), stdout=PIPE)
    eddy_qc_output, eddy_qc_error = eddy_qc_process.communicate()

    # Finally rename file in the quality check directory
    prefix = 'dti_' + subject + '_'
    for f in os.listdir(os.path.join(motion_corrected_directory, 'quality_check')):
        os.rename(os.path.join(motion_corrected_directory, 'quality_check', f),
                  os.path.join(motion_corrected_directory, 'quality_check', prefix + f))
    # End of eddy current and motion correction.

    # Conversion of dicom to nifti for T1
    subject_dicom_t1 = os.path.join(root_data_directory, subject, T1_directory, "dicom")
    t1_output = os.path.join(root_data_directory, subject, T1_directory, "nifti")

    dcm2nii_t1 = "dcm2niix -z {} -o {}   {}".format("y", t1_output, subject_dicom_t1)

    dc2mii_t1_process = Popen(dcm2nii_t1.split(), stdout=PIPE)
    dc2mii_t1_output, dc2nii_t1_error = dc2mii_t1_process.communicate()

    # Rename nifti files
    t1_list_files = [f for f in os.listdir(t1_output) if re.match(r'[a-z]+.*\.nii*.gz', f)]
    t1_json_files = [f for f in os.listdir(t1_output) if re.match(r'[a-z]+.*\.json', f)]

    # Rename the json file
    os.rename(os.path.join(t1_output, t1_json_files[0]),
              os.path.join(t1_output, 't1_' + subject + '_info.json'))
    # Rename the t1 file
    os.rename(os.path.join(t1_output, t1_list_files[0]),
              os.path.join(t1_output, 't1_' + subject + '.nii.gz'))

    # Perform T1 intensity inversion, so that the contrast
    # ressemble to the EPI images, improve normalization process.
    t1_image = os.path.join(t1_output, 't1_' + subject + '.nii.gz')

    # Skull Stripping of the t1 image
    robex = os.path.join(root_data_directory, 'scripts/dti_preproc/ROBEX/runROBEX.sh')

    t1_skullstriping = '{} {} {}'.format(robex, t1_image,
                                         os.path.join(t1_output, 'robex_t1_' + subject + '.nii.gz'))
    t1_skullstriping_process = Popen(t1_skullstriping.split(), stdout=PIPE)
    t1_skullstriping_output, t1_skullstriping_error = t1_skullstriping_process.communicate()

    # Skull Stripping of the b=0 image from motion corrected data
    dti_motion_corrected_image = load_img(
        img=os.path.join(motion_corrected_directory,
                         'eddy_corrected_' + 'dti_' + subject + '.nii.gz'))
    dti_motion_corrected_data = dti_motion_corrected_image.get_data()
    dti_motion_corrected_affine = dti_motion_corrected_image.affine
    dti_bo_motion_corrected = dti_motion_corrected_data[..., 0]

    nb.save(img=nb.Nifti1Image(dti_bo_motion_corrected,
                               affine=dti_motion_corrected_affine),
            filename=os.path.join(motion_corrected_directory,
                                  'dti_bo_motion_corrected_' + subject + '.nii.gz'))

    b0_skullstripping = 'bet2 {} {} -f 0.3'.format(
        os.path.join(motion_corrected_directory, 'dti_bo_motion_corrected_' + subject + '.nii.gz'),
        os.path.join(dti_output, 'bet_dti_b0_' + subject + '.nii.gz'))

    b0_skullstripping_process = Popen(b0_skullstripping.split(), stdout=PIPE)
    b0_skullstripping_output, b0_skullstripping_error = b0_skullstripping_process.communicate()

    # Bias correction for the skull-stripped t1 image
    bias_correction = 'fast -B --nopve {}'.format(
        os.path.join(t1_output, 'robex_t1_' + subject + '.nii.gz'))
    bias_correction_process = Popen(bias_correction.split(), stdout=PIPE)
    bias_correction_output, bias_correction_error = bias_correction_process.communicate()

    # T1 contrast inversion
    robex_t1_image = load_img(img=os.path.join(t1_output, 'robex_t1_' + subject + '_restore.nii.gz'))
    robex_t1_image_data = robex_t1_image.get_data()
    robex_t1_affine = robex_t1_image.affine
    non_background_robex_t1 = robex_t1_image_data != 0

    bet_b0_dti_image = load_img(img=os.path.join(dti_output, 'bet_dti_b0_' + subject + '.nii.gz'))
    bet_b0_dti_data = bet_b0_dti_image.get_data()
    bet_bo_dti_affine = bet_b0_dti_image.affine

    robex_t1_max = np.max(robex_t1_image_data)
    robex_t1_min = np.min(robex_t1_image_data)

    b0_max = np.max(bet_b0_dti_data)
    b0_min = np.min(bet_b0_dti_data)

    robex_t1_inverted = \
        ((robex_t1_max - robex_t1_min)/(b0_max - b0_min))*(-1.0*(np.multiply(
            robex_t1_image_data, non_background_robex_t1)) + np.multiply(robex_t1_max, non_background_robex_t1))

    nb.save(img=nb.Nifti1Image(robex_t1_inverted, affine=robex_t1_affine),
            filename=os.path.join(t1_output, 'inverted_robex_t1_' + subject + '.nii.gz'))

    # Resample the bias corrected and brain extracted contrast inverted T1 to the EPI
    resampled_t1_inverted = \
        resample_to_img(source_img=os.path.join(t1_output, 'inverted_robex_t1_' + subject + '.nii.gz'),
                        target_img=os.path.join(dti_output, 'bet_dti_b0_' + subject + '.nii.gz'))
    nb.save(resampled_t1_inverted, filename=os.path.join(t1_output,
                                                         'r_inverted_robex_t1_' + subject + '.nii.gz'))

    # Register the inverted skull-stripped T1 image with the skull-stripped b0 image with
    # Symmetric Normalization (SyN) from ANTs.
    robex_t1_inverted_image = os.path.join(t1_output, 'inverted_robex_t1_' + subject + '.nii.gz')
    resampled_t1_inverted_image = os.path.join(t1_output,
                                               'r_inverted_robex_t1_' + subject + '.nii.gz')
    bet_b0_dti = os.path.join(dti_output, 'bet_dti_b0_' + subject + '.nii.gz')

    # ants_registration = 'antsRegistrationSyN.sh -d 3 -f {} -m {} -o {} -t s -n {}'.format(
    #    robex_t1_inverted_image, bet_b0_dti, os.path.join(dti_output, subject + '_b0_to_T1_'),
    #    ants_n_threads)

    ants_registration = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "0",
        "--output", "[{},{},{}]".format(os.path.join(dti_output, subject + '_b0_to_T1_'),
                                        os.path.join(dti_output, subject + '_b0_to_T1_Warped.nii.gz'),
                                        os.path.join(dti_output, subject + '_b0_to_T1_InverseWarped.nii.gz')),
        "--interpolation", "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "0",
        "--initial-moving-transform", "[{},{},1]".format(robex_t1_inverted_image, bet_b0_dti),
        "--transform", "Rigid[0.1]",
        "--metric", "MI[{},{},1,32,Regular,0.25]".format(robex_t1_inverted_image, bet_b0_dti),
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--restrict-deformation", "0.1x1x0.1",
        "--transform", "SyN[0.1,3,0]",
        "--metric", "CC[{},{},1,4]".format(robex_t1_inverted_image, bet_b0_dti),
        "--convergence", "[100x70x50x20,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--verbose", "1"
    ]

    # Perform EPI to T1 registration
    ants_registration_process = Popen(ants_registration, stdout=PIPE)
    ants_registration_output, ants_registration_error = ants_registration_process.communicate()

    # Save the output for the EPI to T1 registration
    with open(os.path.join(dti_output, subject + '_dti_bo_to_T1_ants_output.txt'), "w") as text_file:
        text_file.write(ants_registration_output.decode('utf-8'))

    # Collapse the deformation field and the affine transform into one single
    # displacement field
    collapse_transformation = 'antsApplyTransforms -d 3 -o [{}, 1] -t {} -t {} -r {}'.format(
        os.path.join(dti_output, subject + '_b0_to_T1_' + 'CollapsedWarp.nii.gz'),
        os.path.join(dti_output, subject + '_b0_to_T1_' + '1Warp.nii.gz'),
        os.path.join(dti_output, subject + '_b0_to_T1' + '_0GenericAffine.mat'),
        robex_t1_inverted_image
    )

    collapse_transformation_process = Popen(collapse_transformation.split(), stdout=PIPE)
    collapse_transformation_output, collapse_transformation_error = collapse_transformation_process.communicate()

    # Apply the displacement field to the diffusion data
    apply_transform_to_dti = 'antsApplyTransforms -d 3 -e 3 -t {} -o {} -r {} -i {} ' \
                             '-n NearestNeighbor --float 1'.format(
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
    # End of the pre-processing pipeline
