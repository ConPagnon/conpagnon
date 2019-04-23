"""
 Created by Dhaif BEKHA, (dhaif.bekha@cea.fr).

 """
"""
This script display in simple 
manner the results of a TBSS
analysis. Basically it thicken 
the skeleton of the p values image and plot this 
skeleton with the significant area circled 
in the color of you're choice on top of the 
FA skeleton.
You will need a p-value corrected image,
the mean_FA.nii.gz image, 
the mean_FA_skeleton.nii.gz image, and
the threshold you used 

"""
from subprocess import Popen, PIPE
import os

# Set path for plotting
results_directory = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/Groupe_patients_controls"
# The corrected p value
corrected_p_values_image = os.path.join(results_directory,
                                        "tbss_language_profile_tfce_corrp_tstat2.nii.gz")

# the thicken corrected p value image
# The FA template
mean_FA_image = os.path.join(results_directory, "mean_FA.nii.gz")
mean_FA_image_range = "0 0.6"
# The FA template skeleton
# The threshold you used in the analysis
skeleton_threshold = str(0.3)
mean_FA_skeleton_image = os.path.join(results_directory, "mean_FA_skeleton.nii.gz")
mean_FA_skeleton_color = "Green"
mean_FA_skeleton_range = "{} 0.7".format(skeleton_threshold)

# Filled corrected p values
tbss_filled_image_color_map = "Red-Yellow"
tbss_fill_output_image = os.path.join(results_directory, "tbss_fill.nii.gz")

tbss_fill = ["tbss_fill",
             corrected_p_values_image,
             "0.99",
             mean_FA_image,
             tbss_fill_output_image
             ]
tbss_fill_command = Popen(tbss_fill, stdout=PIPE)
tbss_fill_output, tbss_fill_error = tbss_fill_command.communicate()

display_results = ["fsleyes",
                   mean_FA_image,
                   "-dr", mean_FA_image_range,
                   mean_FA_skeleton_image,
                   "-cm", mean_FA_skeleton_color,
                   "-dr", mean_FA_skeleton_range,
                   tbss_fill_output_image,
                   "-cm", tbss_filled_image_color_map
                   ]
display_results_command = Popen(display_results, stdout=PIPE)
display_results_output, display_results_error = display_results_command.communicate()
