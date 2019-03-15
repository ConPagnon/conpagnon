"""
 Created by Dhaif BEKHA, (dhaif.bekha@cea.fr).

 """
import pandas as pd
from subprocess import Popen, PIPE
import os
import numpy as np
"""

Goal
----
Create a Fractional Anisotropy (FA) template image based 
on the NAIS controls children group. After that, we compute
the white matter skeleton based on the FA template. The FA template and
the corresponding white matter skeleton can be directly use with
the TBSS methodology.

This code mostly use python wrapper's code around FSL and DTI-TK,
some of A. Grigis (antoine.grigis@cea.fr), and some of mine. Please visit :
https://github.com/neurospin/pyconnectome/tree/master/pyconnectome.


All the steps for the creation of the template are done with the help
of DTI-TK and can be found at:
http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.HomePage

All the DTI-TK template preparation steps for the TBSS pipeline can be found at:
http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.TBSS

All the steps of the TBSS pipeline can be found at:
https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS/UserGuide
----

Step 1
------
Convert the DTIFIT FSL output to the DTI-TK tensors
file format.

Step 2
------
Compute a tensors template from a subset group of 
subjects. This step increase in time with the number
of subject.

Step 3
------
Warp all subject in the tensor template space 
previously computed

Step 4
------
Generate a final refine template by taking
the mean of all normalized tensors image 
previously computed. 

Step 5
------
On the final template, compute the 1mm3
FA template map.

Step 6
------
On the previously computed FA map, compute
the skeleton of major tracks.

"""
# read subjects list
subjects = list(pd.read_csv('/neurospin/grip/protocols/MRI/Ines_2018/controls_list.txt',
                            header=None)[0])
# main TBSS analysis directory
tbss_directory = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/controls_dtitk/dtitk_native_tensors'

# PyConnectome script directory: Change according to you're installation !!
pyconnectome_script = '/media/db242421/db242421_data/pyconnectome/pyconnectome/scripts'

# Number of controls used for the template creation
subset_of_controls = 12

# Template output directory
template_outdir = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/controls_dtitk/' \
                  'controls_dti_template'

# Normalized tensors output directory
normalized_tensors_outdir = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/controls_dtitk/normalized_tensors'

# Final template output directory for the TBSS analysis
final_high_res_template_outdir = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/controls_dtitk/final_template'

"""
Convert DTIFIT fsl output to DTI-TK tensors file format.
"""
for subject in subjects:
    print('Converting FSL tensor to DTI-TK tensor file format for {}'.format(subject))

    # Build the tensor file format requested for dti-tk
    fsl_to_dtitk = ["python",
                    "{}".format(os.path.join(pyconnectome_script,
                                             'pyconnectome_dtitk_import_tensors')),
                    "-b", "{}".format(os.path.join(tbss_directory,
                                                   subject, "dtifit", "dtifit_" + subject)),
                    "-s", "{}".format(subject),
                    "-o", "{}".format(os.path.join(tbss_directory, subject))]
    fsl_to_dtitk_process = Popen(fsl_to_dtitk, stdout=PIPE)
    fsl_to_dtitk_output, fsl_to_dtitk_error = fsl_to_dtitk_process.communicate()


"""
Randomly choose control subject to create a bootstrapped sample.
"""
# Extract a subset of controls for the creation of the template
subset_of_controls_nip = list(np.random.choice(a=subjects, size=subset_of_controls))
# build full path to the dti-tk tensors file of the subset
subset_controls_tensors_path = [os.path.join(tbss_directory, s_subset,
                                             s_subset, 'dtifit_' + s_subset + '_dtitk.nii.gz')
                                for s_subset in subset_of_controls_nip]

"""
Build the DTI template based on the bootstrapped sample.
"""
# For formatting purposes, I have to write down the command in a text
with open(os.path.join(template_outdir, "cmd_create_template.txt"), 'w') as command_txt:
    command_txt.writelines('python {}/pyconnectome_dtitk_create_templates '.format(pyconnectome_script))
    command_txt.writelines('-t ')
    for tensor_path in range(len(subset_controls_tensors_path)):
        command_txt.writelines('{} '.format(subset_controls_tensors_path[tensor_path]))
    command_txt.writelines('-o {} '.format(template_outdir))
    command_txt.writelines('-V 2')

with open(os.path.join(template_outdir, "cmd_create_template.txt"), 'r') as command_txt:
    dtitk_create_template = command_txt.read()

dtitk_create_template_process = Popen(dtitk_create_template.split(), stdout=PIPE)
dtitk_create_template_output, dtitk_create_template_error = \
    dtitk_create_template_process.communicate()

# Write command output
with open(os.path.join(template_outdir, "output_create_template.txt"), 'w') as create_template_out:
    create_template_out.write(dtitk_create_template_output.decode('utf-8'))

"""
Warp all controls tensors in the template space, with final resolution 
of 1mm3 
"""
# Loop over all controls subject
for subject in subjects:
    dtitk_register = ["python", os.path.join(pyconnectome_script,
                                             "pyconnectome_dtitk_register"),
                      "-t", os.path.join(tbss_directory, subject, subject,
                                         "dtifit_" + subject + "_dtitk.nii.gz"),
                      "-s", subject,
                      "-e", os.path.join(template_outdir,
                                         "mean_diffeomorphic_initial6.nii.gz"),
                      "-o", normalized_tensors_outdir,
                      "-b", os.path.join(template_outdir, "mask.nii.gz"),
                      "-V 2"]
    dtitk_register_process = Popen(dtitk_register, stdout=PIPE)
    dtitk_register_output, dtitk_register_error = dtitk_register_process.communicate()

"""
Generate the population specific DTI template with
isotropic 1mm3 spacing
"""
# Write down in a text file the path to the normalized
# tensors for all controls
with open(os.path.join(final_high_res_template_outdir,
                       "subjects_normalized.txt"), "w") as tensors_path_file:
    for subject in subjects:
        tensors_path_file.write(os.path.join(normalized_tensors_outdir, subject,
                                             "dtifit_" + subject + "_dtitk_diffeo.nii.gz"))
        tensors_path_file.write("\n")
# Full path to final tensor template to be written
mean_final_high_res = os.path.join(final_high_res_template_outdir,
                                   "mean_final_high_res.nii.gz")
# Compute the final template with the TVMean command
compute_final_template = ["TVMean",
                          "-in", os.path.join(final_high_res_template_outdir,
                                              "subjects_normalized.txt"),
                          "-out", mean_final_high_res,
                          "-type", "ORIGINAL",
                          "-interp", "LEI"
                          ]
print("Computing the final tensor template with isotropic resolution...")
compute_final_template_process = Popen(compute_final_template, stdout=PIPE)
compute_final_template_output, compute_final_template_error = \
    compute_final_template_process.communicate()

# Write command output
with open(os.path.join(final_high_res_template_outdir,
                       "TVMean_output.txt"), "w") as TVMean_output:
    TVMean_output.write(compute_final_template_output.decode('utf-8'))

"""
Compute the high resolution FA template map from
the previously computed tensor template
"""
generate_FA_template = ["TVtool",
                        "-in", mean_final_high_res,
                        "-fa"]
generate_FA_template_process = Popen(generate_FA_template, stdout=PIPE)
generate_FA_template_output, generate_FA_template_error = \
    generate_FA_template_process.communicate()
# Write command output
with open(os.path.join(final_high_res_template_outdir,
                       "TVtool_FA_output.txt"), "w") as TVtool_output:
    TVtool_output.write(compute_final_template_output.decode('utf-8'))


"""
Optional: Compute the RGB image from
the final tensor template. Very convenient
with ITK-SNAP viewer for example.

"""
generate_RGB_template = ["TVtool",
                         "-in", mean_final_high_res,
                         "-rgb"]
generate_RGB_template_process = Popen(generate_RGB_template, stdout=PIPE)
generate_RGB_template_output, generate_RGB_template_error = \
    generate_RGB_template_process.communicate()
# Write command output
with open(os.path.join(final_high_res_template_outdir,
                       "TVtool_RGB_output.txt"), "w") as TVtool_RGB_output:
    TVtool_RGB_output.write(generate_RGB_template_output.decode('utf-8'))

"""
Compute the skeleton from
the FA map template

"""
skeletonize_FA = ["tbss_skeleton",
                  "-i", os.path.join(final_high_res_template_outdir,
                                     "mean_final_high_res_fa.nii.gz"),
                  "-o", os.path.join(final_high_res_template_outdir,
                                     "mean_final_high_res_fa_skeleton.nii.gz")
                  ]
skeletonize_FA_process = Popen(skeletonize_FA, stdout=PIPE)
skeletonize_FA_output, skeletonize_FA_error = \
    skeletonize_FA_process.communicate()
# Write command output
with open(os.path.join(final_high_res_template_outdir,
                       "tbss_skeleton_RGB_output.txt"), "w") as skeleton_output:
    skeleton_output.write(skeletonize_FA_output.decode('utf-8'))

"""
End of the creation of the FA and corresponding skeleton template.
The output of this script can now replace the standard templates provide by 
the FSL software for the TBSS analysis.
"""
