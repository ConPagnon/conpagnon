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


"""
This script compute a statistical
analysis, using the randomise
function in FSL.
For lesioned patients, we take into
account lesioned voxels by masking out
the corresponding voxels with the help
of the "setup_masks" function.

"""

# the stats directory
stats_directory = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/stats"
# the prepare_stats directory
prepare_stats_directory = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/prepare_stats"
# directory containing all the mask
all_subject_mask_directory = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/all_subjects_mask"
# Your're statistical analysis directory
stats_results_directory = ""
# subjects text file : note that the NIP should in the same order of the FA image
# in the all_FA.nii.gz file !
subjects_list_txt = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/all_subjects.txt"
subjects = list(pd.read_csv(subjects_list_txt, header=None)[0])

# Excel file path to the cohort clinical information.
clinical_data_excel = "/media/db242421/db242421_data/DTI_TBSS_M2Ines/prepare_stats/design_matrix_example.xlsx"
clinical_data = pd.read_excel(clinical_data_excel)

# Te equation of you're model
model_equation = "Groups + language"
variables_in_model = model_equation.split(sep=' + ')

# Important argument for the randomise function
mean_FA_skeleton_mask_path = os.path.join(stats_directory,
                                          "mean_FA_skeleton.nii.gz")
all_FA_skeletonised_path = os.path.join(stats_directory, "all_FA_skeletonised.nii.gz")

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
    all_FA = load_img(os.path.join(stats_directory, 'all_FA.nii.gz'))
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
"""
design_matrix = patsy.dmatrix('0 + {}'.format("+".join(variables_in_model)), data=clinical_data,
                              return_type="dataframe")
design_matrix.to_csv(os.path.join(prepare_stats_directory, "design_matrix.txt"), sep="\t",
                     index=False, header=False)
text2vest = ["Text2Vest",
             os.path.join(prepare_stats_directory, "design_matrix.txt"),
             os.path.join(prepare_stats_directory, "design.mat")]
text2vest_command = Popen(text2vest, stdout=PIPE)
text2vest_output, text2vest_error = text2vest_command.communicate()
