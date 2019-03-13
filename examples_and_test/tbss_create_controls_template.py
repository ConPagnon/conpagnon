"""
 Created by db242421, (dhaif.bekha@cea.fr).

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

This code mostly use python wrapper's code around FSL and DTI-TK
from A. Grigis (antoine.grigis@cea.fr). Please visit :
https://github.com/neurospin/pyconnectome/tree/master/pyconnectome

All the steps for the creation of the template are done with the help
of DTI-TK and can be found at:
http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.HomePage

All the DTI-TK template preparation steps for the TBSS pipeline can be found at:
http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.TBSS

All the steps of the TBSS pipeline can be found at:
https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS/UserGuide
----

Step 0: Build DTI-TK files format tensors 
        from the DTIFIT function output from FSL.

Step 1: Create a group specific DTI template with bootstrapping.

Step 3: Project all controls in the template space with a final voxels
        size of 1mm3
        
TBSS template preparation
-----
Step 4: Generate the population specific template with the isotropic 
        1 mm3 spacing from the normalized DTI data.

Step 5: Generate the FA map of the high resolution 
        population specific DTI template

Step 6: Generate the white matter skeleton from 
        the high resolution FA map of the DTI template.

"""
# read subjects list
subjects = list(pd.read_csv('/neurospin/grip/protocols/MRI/Ines_2018/controls_list.txt',
                            header=None)[0])
# main TBSS analysis directory
tbss_directory = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/controls_dtitk/dtitk_preprocessing'

# PyConnectome script directory
pyconnectome_script = '/media/db242421/db242421_data/pyconnectome/pyconnectome/scripts'

# Number of controls used for the template creation
subset_of_controls = 12

# Template output directory
template_outdir = '/media/db242421/db242421_data/DTI_TBSS_M2Ines/controls_dtitk/' \
                  'controls_dti_template'

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
dtitk_create_template_output, dtitk_create_template_error = dtitk_create_template_process.communicate()

# Write command output
with open(os.path.join(template_outdir, "output_create_template.txt"), 'r') as create_template_out:
    create_template_out.write(dtitk_create_template_output.decode('utf-8'))

"""
Warp all controls tensors in the template space
"""
















