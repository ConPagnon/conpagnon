"""
 Created by Dhaif BEKHA, (dhaif.bekha@cea.fr).

 """
import pandas as pd
from nilearn.image import load_img
import os
import numpy as np
import nibabel as nb
import patsy
from subprocess import Popen, PIPE
import glob
import shutil

"""
This script compute a statistical
analysis, using the randomise
function in FSL.
For lesioned patients, we take into
account lesioned voxels by masking out
the corresponding voxels with the help
of the "setup_masks" function.

"""

# the prepare_stats directory
prepare_stats_directory = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/prepare_stats"
# directory containing all the mask
all_subject_mask_directory = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/all_subjects_mask"
# Your're statistical analysis directory
# Advise: on directory per study.
stats_results_directory = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/LD_controls2"
os.mkdir(stats_results_directory)
# subjects text file : note that the NIP should in the same order of the FA image
# in the all_FA.nii.gz file !
subjects_list_txt = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/LD_controls.txt"
subjects = list(pd.read_csv(subjects_list_txt, header=None)[0])

# Excel file path to the cohort clinical information.
clinical_data_excel = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/prepare_stats/design_matrix_example.xlsx"
clinical_data = pd.read_excel(clinical_data_excel)

# Te equation of you're model
model_equation = "Groups"
variables_in_model = model_equation.split(sep=' + ')

# Important path
mean_FA_skeleton_mask_path = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/stats/mean_FA_skeleton_mask.nii.gz"
all_FA_skeletonised_path = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/stats/all_FA_skeletonised.nii.gz"
mean_FA_path = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/stats/mean_FA.nii.gz"
mean_FA_mask_path = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/stats/mean_FA_mask.nii.gz"
all_FA_path = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/stats/all_FA.nii.gz"

# Full path to the contrast file. User should create you're own contrast file.
contrast_file = os.path.join(prepare_stats_directory, "contrast_matrix.txt")

"""
First Step: create a design matrix in a format suitable
for the randomise function.
The user must create a excel following some rules:
    - The first column must contain the subject's NIP
    in the SAME ORDER as the all_FA.nii.gz image. this
    column must be called 'NIP'.
    - The other column contain all the other clinical
    variable. One column per variable.
    - The first line, contain the column name.
If one clinical variable is missing, depending on you're 
the corresponding subject is automatically removed 
from the design matrix, and 
as a consequence also from the all_FA, and all_FA_skeletonised
images. 
"""
# Sanity check : is there any missing value ?
is_subject_missing = clinical_data.isnull().any().any()
if is_subject_missing:
    # Look for the NIP of missing subjects, and
    # corresponding variables
    damaged_variable_df = clinical_data.isnull().any()
    damaged_variable_list = list(damaged_variable_df[damaged_variable_df == True].index)
    missing_data_dictionary = dict.fromkeys(damaged_variable_list)
    for damaged_variable in damaged_variable_list:
        # index of subjects which had a missing value for the
        # current damaged variable
        damaged_subject = list(clinical_data[clinical_data[damaged_variable].isnull()].index)
        missing_data_dictionary[damaged_variable] = damaged_subject
    # Depending on variable on the equation find all the subjects to be removed
    # from analysis
    missing_subject = [missing_data_dictionary[variable] for variable in damaged_variable_list if
                       variable in variables_in_model]
    with open(os.path.join(prepare_stats_directory, "missing_subject_log.txt"), "w") as missing_subject_log:
        missing_subject_log.writelines("Variables {} contain missing data. \n".format(
            ", ".join(list(missing_data_dictionary.keys()))))
        missing_subject_log.writelines("Subjects {} will be removed.".format(
            ", ".join(list(clinical_data.loc[np.array(missing_subject).flatten()]['NIP']))))
    # Finally, clean the dataframe, and remove the corresponding subject also
    # in the all_FA, and all_FA_skeletonised image.
    all_FA = load_img(all_FA_path)
    all_FA_skeletonised = load_img(all_FA_skeletonised_path)
    # Get data
    all_FA_data = all_FA.get_data()
    all_FA_skeletonised_data = all_FA_skeletonised.get_data()
    # delete the subject index corresponding to damaged subject
    index_to_remove = np.array(missing_subject).flatten()
    new_all_FA_data = np.delete(all_FA_data, index_to_remove, axis=3)
    new_all_FA_skeletonised_data = np.delete(all_FA_skeletonised_data, index_to_remove, axis=3)
    # Save the new image in the prepare stats directory
    nb.save(nb.Nifti1Image(new_all_FA_data, affine=all_FA.affine),
            filename=os.path.join(prepare_stats_directory, 'new_all_FA.nii.gz'))
    nb.save(nb.Nifti1Image(new_all_FA_skeletonised_data, affine=all_FA_skeletonised.affine),
            filename=os.path.join(prepare_stats_directory, 'new_all_FA_skeletonised.nii.gz'))

    # Finally, we delete the same index in the excel file
    new_clinical_data = clinical_data.drop(index=index_to_remove)
    # We save it
    new_clinical_data.to_excel(os.path.join(prepare_stats_directory,
                                            'new_design_matrix_example.xlsx'),
                               index=False)

    # We now, a proper excel file containing no missing data,
    # and we removed the corresponding subject in the all_FA,
    # and all_FA_skeletonised images.
    all_FA_skeletonised_path = os.path.join(prepare_stats_directory,
                                            "new_all_FA_skeletonised.nii.gz")
    clinical_data_excel = os.path.join(prepare_stats_directory, "new_design_matrix_example.xlsx")
    clinical_data = pd.read_excel(clinical_data_excel)
else:
    # If no subject is missing in the data, we have nothing
    # to do.
    all_FA_skeletonised_path = all_FA_skeletonised_path
    clinical_data = clinical_data

"""
Now that we have clean data, we can now build our
design matrix, in a format suitable for the randomise
data. Each categorical in the clinical dataset is coded
with a binary variable. We first save the design matrix
as tabulated separated values file (tsv), and we call
the function Text2Vest of the FSL software.

In this format, each line correspond to a subject. The 
order is naturally the same as the all_FA_skeletonised 
image. Each column correspond to a variable in the analysis.
The library patsy we use, try to avoid redundancy when coding
for a categorical variable. You should always check the results
to see what are the coding scheme.

The user should also create manually a contrast file in a 
tsv file format. The tsv contrast file will also be converted
to a suitable contrast file format for randomise. Each
line, correspond to one contrast.

At the end of this section, we have a clean design matrix
in the .mat format, and a contrast file in the .con  format.
"""
design_matrix = patsy.dmatrix('C({}) - 1'.format("+".join(variables_in_model)), data=clinical_data,
                              return_type="dataframe")
design_matrix.to_csv(os.path.join(prepare_stats_directory, "design_matrix.txt"), sep="\t",
                     index=False, header=False)
# Save the matrix with header to keep track of columns
design_matrix.to_csv(os.path.join(prepare_stats_directory, "design_matrix_with_header.txt"), sep=",",
                     index=False, header=True)
text2vest = ["Text2Vest",
             os.path.join(prepare_stats_directory, "design_matrix.txt"),
             os.path.join(prepare_stats_directory, "design.mat")]
text2vest_command = Popen(text2vest, stdout=PIPE)
text2vest_output, text2vest_error = text2vest_command.communicate()


text2vest_contrast = ["Text2Vest",
                      contrast_file,
                      os.path.join(prepare_stats_directory, "contrast.con")]
text2vest_contrast_command = Popen(text2vest_contrast, stdout=PIPE)
text2vest_contrast_output, text2vest_contrast_error = text2vest_contrast_command.communicate()

"""
The final step before calling randomise
is to take care of possible missing data.
Here, missing data = lesioned tissue. 
We call setup_mask to this purpose.


"""
# We create a list of the mask in the RIGHT ORDER:
# same order as the subjects list text file.
setup_mask_list = [glob.glob(os.path.join(all_subject_mask_directory, subject + "*.nii.gz"))[0]
                   for subject in list(clinical_data['NIP'])]
# Full path to the design matrix in the .mat format
design_matrix_mat = os.path.join(prepare_stats_directory, "design.mat")
# Full path to the contrast file in the .con format
contrast_matrix_con = os.path.join(prepare_stats_directory, "contrast.con")
# For formatting purpose we write the command in a text file
with open(os.path.join(prepare_stats_directory, "setup_mask_command"), "w") as setup_mask_txt:
    setup_mask_txt.writelines("setup_masks {} {} {} ".format(design_matrix_mat, contrast_matrix_con,
                                                             os.path.join(prepare_stats_directory, "new_design")))
    # write all mask path
    for mask in setup_mask_list:
        setup_mask_txt.writelines("{} ".format(mask))

# We read the command
with open(os.path.join(prepare_stats_directory, "setup_mask_command"), "r") as setup_mask_txt:
    setup_mask_command = setup_mask_txt.read()

setup_mask = Popen(setup_mask_command.split(), stdout=PIPE)
setup_mask_output, setup_mask_error = setup_mask.communicate()
with open(os.path.join(stats_results_directory, "setup_mask_log"), "w") as setup_mask_log:
    setup_mask_log.writelines(setup_mask_output.decode("utf-8"))

"""
We now have all the file properly formatted 
to feed the randomise function. We first move
all the needed file in the analysis directory:
the all_FA_skeletonised image, the modified design and 
contrast matrix.
Randomise output for each contrast, the raw statistic 
and the corresponding p-value corrected image. 
See the next script to simply display the TBSS results
with FSLEYES.

"""
# Move the modified design matrix, contrast, mask file
# in the analysis directory
new_design_matrix = os.path.join(prepare_stats_directory,
                                 "new_design.mat")
new_contrast = os.path.join(prepare_stats_directory,
                            "new_design.con")
mask_image = os.path.join(prepare_stats_directory,
                          "new_design.nii.gz")

shutil.copyfile(new_design_matrix,
                os.path.join(stats_results_directory, "design.mat"))
shutil.copyfile(new_contrast,
                os.path.join(stats_results_directory, "design.con"))
shutil.copyfile(mask_image,
                os.path.join(stats_results_directory, "design_mask.nii.gz"))
shutil.copyfile(all_FA_skeletonised_path,
                os.path.join(stats_results_directory, "all_FA_skeletonised.nii.gz"))
shutil.copyfile(mean_FA_skeleton_mask_path,
                os.path.join(stats_results_directory, "mean_FA_skeleton_mask.nii.gz"))

# For illustration purpose we move the mean_FA and mean_FA_skeleton images
shutil.copyfile(mean_FA_skeleton_mask_path,
                os.path.join(stats_results_directory, "mean_FA_skeleton.nii.gz"))
shutil.copyfile(mean_FA_path,
                os.path.join(stats_results_directory, "mean_FA.nii.gz"))
# Choose the output basename of the study
tbss_output_basename = "LD_controls"
# Set the mask option in randomise (check the setup_mask command
# log file to see an example.
vxl = str(-3)
vxf = os.path.join(stats_results_directory, "design_mask.nii.gz")
# Set the number of permutations
n_permutations = str(10000)

# Call randomise
randomise = ["randomise",
             "-i",  os.path.join(stats_results_directory, "all_FA_skeletonised.nii.gz"),
             "-o", os.path.join(stats_results_directory, tbss_output_basename),
             "-d", os.path.join(stats_results_directory, "design.mat"),
             "-t", os.path.join(stats_results_directory, "design.con"),
             "-m", os.path.join(stats_results_directory, "mean_FA_skeleton_mask.nii.gz"),
             "--vxl={}".format(vxl),
             "--vxf={}".format(vxf),
             "-n", n_permutations,
             "--T2"]
randomise_call = Popen(randomise, stdout=PIPE)
randomise_output, randomise_error = randomise_call.communicate()
with open(os.path.join(stats_results_directory, "randomise_log"), "w") as randomise_log:
    randomise_log.writelines(randomise_output.decode("utf-8"))

