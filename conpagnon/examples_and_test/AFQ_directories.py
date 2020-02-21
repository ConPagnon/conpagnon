import os
from subprocess import Popen, PIPE

"""
Create the archicture of directories suitable for
DTIinit, before launching AFQ. Organise the tree directory
as follow: 
      <subdir>
    _____|___________
         |           |
     t1.nii.gz     <ROIs>
     dwi.nii.gz
      dwi.bval
      dwi.bvec

"""

# Path prefix to the external drive containing the data
prefix_path = "/media/dhaif/Samsung_T5"

# directory path of the groups of images
root_directory = os.path.join(prefix_path,
                              "Work/Neurospin/AVCnn/Ines_2018/images/controls_v2")

# subjects list
subjects = open(os.path.join(prefix_path, "Work/Neurospin/AVCnn/Ines_2018/images/controls_v2_nip")).read().split()

# results directory
afq_controls = os.path.join(prefix_path, "Work/Neurospin/AVCnn/Ines_2018/AFQ_controls")

# Loop over the subjects
for subject in subjects:
    print("Processing subject: {}".format(subject))
    # Create a subject directory
    subject_dir = os.path.join(afq_controls, subject)
    if subject not in os.listdir(afq_controls):
        os.mkdir(subject_dir)
    else:
        print("Warning ! Subject directory already here for subject: {}".format(subject))
        pass

    # Create ROI directory inside
    if "ROIs" not in os.listdir(subject_dir):
        os.mkdir(os.path.join(subject_dir, "ROIs"))
    else:
        print("Warning ! ROIs subject directory already here for subject: {}".format(subject))
        pass

    # Full path to the T1 image of the subject
    t1_image = os.path.join(root_directory, subject, "T1/nifti", "t1_" + subject + ".nii.gz")
    # Full path to the motion and eddy current corrected,
    # distortion corrected diffusion image
    dwi_image = os.path.join(root_directory, subject, "motion_corrected_data",
                             "distortion_and_eddy_corrected_dti_" + subject + ".nii.gz")
    # Full path to bvec and bvals file
    bvecs = os.path.join(root_directory, subject, "diffusion/nifti", "dti_" + subject + ".bvec")
    bvals = os.path.join(root_directory, subject, "diffusion/nifti", "dti_" + subject + ".bval")

    # Copy and rename all the files into the results directory
    # according DTIinit filename rules:

    # Copy T1
    cp_t1 = ["cp",
             t1_image,
             os.path.join(afq_controls, subject_dir, "t1.nii.gz")]
    cp_t1_command = Popen(cp_t1, stdout=PIPE)
    cp_t1_output, cp_t1_error = cp_t1_command.communicate()
    # Copy diffusion image
    cp_dwi = ["cp",
              dwi_image,
              os.path.join(afq_controls, subject_dir, "dwi.nii.gz")]
    cp_dwi_command = Popen(cp_dwi, stdout=PIPE)
    cp_dwi_output, cp_dwi_error = cp_dwi_command.communicate()
    # Copy bvec file
    cp_bvecs = ["cp",
                bvecs,
                os.path.join(afq_controls, subject_dir, "dwi.bvec")]
    cp_bvecs_command = Popen(cp_bvecs, stdout=PIPE)
    cp_bvecs_output, cp_bvecs_error = cp_bvecs_command.communicate()
    # Copy bval file
    cp_bvals = ["cp",
                bvals,
                os.path.join(afq_controls, subject_dir, "dwi.bval")]
    cp_bvals_command = Popen(cp_bvals, stdout=PIPE)
    cp_bvals_output, cp_bvals_error = cp_bvals_command.communicate()

    print("Done for subject: {}".format(subject))