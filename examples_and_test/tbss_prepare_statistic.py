"""
 Created by Dhaif BEKHA, (dhaif.bekha@cea.fr).

 """
import os
import pandas as pd
from subprocess import Popen, PIPE, call

"""
This code prepare the data for the voxelswise 
analysis according to the TBSS pipeline.

"""

# root directory
root_directory = '/media/db242421/db242421_data/DTI_TBSS_M2Ines'
# subject order list
all_subjects_txt = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/all_subjects.txt'
subjects = list(pd.read_csv(all_subjects_txt, header=None)[0])

# Warped FA images directory
all_FA_directory = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/all_FA'

# Full path to the skeleton of the FA template
FA_skeleton = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/controls_dtitk/' \
              'final_template/mean_final_high_res_fa_skeleton.nii.gz'

# Full path the high resolution FA template
FA_template = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/controls_dtitk/' \
              'final_template/mean_final_high_res_fa.nii.gz'

# Create a directory called stats
list_of_directory_in_root = os.listdir(root_directory)
if 'stats' not in list_of_directory_in_root:
    os.mkdir(os.path.join(root_directory, 'stats'))
else:
    # Clear the existing directory to avoid confusion
    # with old file
    os.rmdir(os.path.join(root_directory, 'stats'))
    # Create the desired directory
    os.mkdir(os.path.join(root_directory, 'stats'))
stats_directory = os.path.join(root_directory, 'stats')
# Concatenate all FA images
# List that contain the full path to
# subjects FA in the order precised in
# all_subjects_txt
FA_in_good_order = [os.path.join(all_FA_directory, subject + '_warped_FA.nii.gz') for
                    subject in subjects]
# Write for convenience the command in a text file:
with open(os.path.join(all_FA_directory, 'merge_FA_command.txt'), 'w') as merge_FA_txt:
    merge_FA_txt.writelines("fslmerge -t {}/all_FA.nii.gz ".format(stats_directory))
    for warped_FA in range(len(FA_in_good_order)):
        merge_FA_txt.writelines('{} '.format(FA_in_good_order[warped_FA]))
# Read and execute the command
with open(os.path.join(all_FA_directory, 'merge_FA_command.txt'), 'r') as merge_FA_txt:
    merge_FA_command = merge_FA_txt.read()
merge_FA = Popen(merge_FA_command.split(), stdout=PIPE)
merge_FA_output, merge_FA_error = merge_FA.communicate()

# Copy the FA skeleton and rename it mean_FA_skeleton
move_skeleton = ["cp", FA_skeleton,
                 os.path.join(stats_directory, 'mean_FA_skeleton.nii.gz')]
move_skeleton_command = Popen(move_skeleton, stdout=PIPE)
move_skeleton_output, move_skeleton_error = move_skeleton_command.communicate()

# Copy the FA template and rename it mean_FA
move_template = ["cp", FA_template,
                 os.path.join(stats_directory, 'mean_FA.nii.gz')]
move_template_command = Popen(move_template, stdout=PIPE)
move_template_output, move_template_error = move_template_command.communicate()

# Create a binary volume for the 4D file all_FA.nii.gz
binary_mask_volume = ["fslmaths", os.path.join(stats_directory, 'all_FA.nii.gz'),
                      "-max", "0",
                      "-Tmin",
                      "-bin",
                      os.path.join(stats_directory, 'mean_FA_mask.nii.gz'),
                      "-odt", "char"]
binary_mask_volume_command = Popen(binary_mask_volume, stdout=PIPE)
binary_mask_volume_output, binary_mask_volume_error = binary_mask_volume_command.communicate()

# Choose a threshold for the mean FA skeleton
# A typical value is 0.2
# To help you, we can plot the mean FA skeleton on all_FA
plot_FA_skeleton = ["fsleyes",
                    os.path.join(stats_directory, 'all_FA.nii.gz'),
                    "-dr", "0 0.8",
                    os.path.join(stats_directory, 'mean_FA_skeleton.nii.gz'),
                    "-dr", "0.2 0.8",
                    "-cm", "Green"]
plot_FA_skeleton_command = Popen(plot_FA_skeleton, stdout=PIPE)
plot_FA_skeleton_output, plot_FA_skeleton_error = plot_FA_skeleton_command.communicate()

# Threshold the FA skeleton, and project the FA data onto
# the mean FA skeleton: here, the threshold is 0.2
tbss_4_prestats = ['cd {} && tbss_4_prestats 0.2'.format(root_directory)]
tbss_4_prestats_command = Popen(tbss_4_prestats, shell=True, stdout=PIPE)
tbss_4_prestats_output, tbss_4_prestats_error = tbss_4_prestats_command.communicate()

"""
End of data preparation. The next and last step
are the statistical analysis.
"""