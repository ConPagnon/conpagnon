# Code to link structural damage to behavioral scores
# Author: Dhaif BEKHA.
import pandas as pd
import numpy as np
from nilearn.image import load_img
import os
import glob
import nibabel as nb
from utils.folders_and_files_management import save_object
import decimal
import matplotlib.pyplot as plt
from nilearn.plotting import plot_glass_brain
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.decomposition import TruncatedSVD


# Set path, read subjects list
lesions_root_directory = "/media/db242421/db242421_data/AVCnn_2016_DARTEL/AVCnn_data/patients"
controls_root_directory = "/media/db242421/db242421_data/AVCnn_2016_DARTEL/AVCnn_data/control"
controls_list_txt = "/media/db242421/db242421_data/ConPagnon_data/text_data/controls.txt"
subjects_list_txt = "/media/db242421/db242421_data/ConPagnon_data/text_data/acm_patients.txt"
subjects_list = pd.read_csv(subjects_list_txt, header=None)[0]
controls_list = pd.read_csv(controls_list_txt, header=None)[0]
saving_directory = "/media/db242421/db242421_data/ConPagnon_reports/lesion_behavior_mapping"

all_images = []
all_lesion_flatten = []
all_controls_images = []

# Load shape and affine of one normalized lesion map
someone_lesion_map = load_img(os.path.join(lesions_root_directory,
                                           "sub44_av130474/lesion/wav130474_lesion_totale_s16.nii"))
target_affine = someone_lesion_map.affine
target_shape = someone_lesion_map.shape

# Just to check, load a control image
# and get affine and shape
some_control_image = load_img(os.path.join(
    controls_root_directory,
    "sub01_nc110193/RS1/anatstar/wnobias_anat1_nc110193-2604_20110427_02.nii"))

target_controls_affine = some_control_image.affine
target_controls_shape = some_control_image.shape

# Build a mean controls anatomical
for control in controls_list:
    # path to the control image
    subject_anatomical_path = os.path.join(controls_root_directory, control,
                                           'RS1/anatstar')
    # Get anatomical image filename
    anatomical_file = glob.glob(os.path.join(subject_anatomical_path, 'wnobias*.nii'))[0]
    # Load the control anatomical  image
    control_anatomical_image = load_img(img=anatomical_file)
    # load image into an array
    control_anatomical_array = control_anatomical_image.get_data()
    # Append the control image
    all_controls_images.append(control_anatomical_array)

# compute the arithmetic mean of controls anatomical
# image
all_controls_array = np.array(all_controls_images)
mean_controls_array = np.mean(all_controls_array, axis=0)

# save mean controls image
mean_controls_nifti = nb.Nifti1Image(dataobj=mean_controls_array,
                                     affine=target_affine)
nb.save(img=mean_controls_nifti,
        filename=os.path.join(saving_directory, "mean_controls.nii"))

# Read and load lesion image for each subject,
for subject in subjects_list:
    # path to the subject image
    subject_lesion_path = os.path.join(lesions_root_directory, subject, 'lesion')
    # Get lesion filename
    lesion_file = glob.glob(os.path.join(subject_lesion_path, 'w*.nii'))[0]
    # Load the normalized lesion image
    subject_lesion = load_img(img=lesion_file)
    # get lesion images affine
    lesion_affine = subject_lesion.affine
    # load image into an array
    subject_lesion_array = subject_lesion.get_data()
    # Append lesion array to all images list
    all_images.append(subject_lesion_array)
    # flatten the lesion array for a PCA later on
    all_lesion_flatten.append(np.array(subject_lesion_array.flatten(),
                                       dtype=np.int8))

lesions_maps = np.array(all_lesion_flatten, dtype=np.int8)
all_lesion_array = np.array(all_images, dtype=np.double)

# Compute lesion overlap image
lesion_overlap_array = np.sum(all_lesion_array, axis=0)

# Save lesions maps overlay
lesion_overlap_nifti = nb.Nifti1Image(dataobj=lesion_overlap_array,
                                      affine=target_affine)
nb.save(img=lesion_overlap_nifti,
        filename=os.path.join(saving_directory, "overlap_acm.nii"))

# Save flatten lesion map
save_object(object_to_save=lesions_maps,
            saving_directory=saving_directory,
            filename='lesion_maps_acm.pkl')
# Save flatten lesion array in numpy format
np.save(os.path.join(saving_directory, "lesion_maps.npy"),
        arr=lesions_maps)

# Reduce the number of voxels, with a mask
# unused voxel is background information
lesions_maps_sum = np.sum(lesions_maps, axis=0)
mask = np.where(lesions_maps_sum >= 1)[0]
lesions_maps_masked = lesions_maps[..., mask]
# Perform a truncated svd
svd_lesions_maps = TruncatedSVD(n_components=14)

svd_lesions_maps_reduced = svd_lesions_maps.fit_transform(lesions_maps_masked)
svd_lesions_maps_reconstructed = svd_lesions_maps.inverse_transform(svd_lesions_maps_reduced)

# Compute loading defined as eigenvectors scaled by
# the squared root of singular values
svd_lesion_maps_loadings = svd_lesions_maps.components_.T * \
                           np.sqrt(svd_lesions_maps.explained_variance_)
svd_lesion_maps_loadings = svd_lesion_maps_loadings.T

print("Percentage of variance explained with {} components "
      "retained: {}%".format(svd_lesions_maps.n_components,
                             round(svd_lesions_maps.explained_variance_ratio_.sum()*100, 2)))

# Percentage of variance explained by each retained components
D = decimal.Decimal
percentage_of_variance_each_components = np.array([round(np.float(
    D(str(svd_lesions_maps.explained_variance_ratio_[i])).quantize(D('0.001'), rounding=decimal.ROUND_UP)), 3)
                                          for i in range(svd_lesions_maps.n_components)])*100
# reconstruct each principal components as an image
# projecting loadings back to brain space
loading_nii_img = {}
for c in range(svd_lesions_maps.n_components):
    # initialize an flatten empty image
    pc_loading_on_brain = np.zeros(lesions_maps.shape[1])
    # Fill the voxels in the mask by the voxels loading
    # from PCA
    pc_loading_on_brain[mask] = svd_lesion_maps_loadings[c, ...]
    # Reshape the array of pc loading
    pc_loading_on_brain_reshaped = pc_loading_on_brain.reshape(target_shape)
    # Convert in a nifti image
    pc_loading_on_brain_reshaped_nii = nb.Nifti1Image(dataobj=pc_loading_on_brain_reshaped,
                                                      affine=target_affine)
    loading_nii_img["PC"+str(c+1)] = pc_loading_on_brain_reshaped_nii
    # Save in nifti format
    nb.save(img=pc_loading_on_brain_reshaped_nii,
            filename=os.path.join(saving_directory, 'PC_' + str(c) + '.nii'))

# Construct a dataframe with the PCA result
columns_names = ['PC{}'.format(i) for i in range(1, svd_lesions_maps.n_components+1)]
pca_results_df = pd.DataFrame(data=svd_lesions_maps_reduced,
                              columns=columns_names)
pca_results_df.head()
# rename index with subject identifier
pca_results_df = pca_results_df.rename(index=subjects_list)
pca_results_df.to_csv(index_label="subjects",
                      path_or_buf=os.path.join(saving_directory, "pca_lesion_map.csv"))

components_names = list(loading_nii_img.keys())
for component in components_names:
    plt.figure()
    plot_glass_brain(
        stat_map_img=loading_nii_img[component], plot_abs=True,
        cmap="RdBu_r", colorbar=True,
        title="{} loadings: {} % of variance explained".format(
            component,
            str(percentage_of_variance_each_components[components_names.index(component)])[0:3]),
        output_file=os.path.join(saving_directory, component + '.pdf'))
    plt.show()

# plot on glass brain the lesion mapping
plot_glass_brain(stat_map_img=lesion_overlap_nifti, plot_abs=False,
                 cmap="rainbow", colorbar=True,
                 title="Lesion overlap of NAIS MCA stroke (N=25)",
                 output_file=os.path.join(saving_directory, "lesion_mapping.pdf"))

# Barplot of explained variance per components
sns.barplot(x=columns_names, y=svd_lesions_maps.explained_variance_ratio_*100)
plt.title("Explained variance in % for each components (total: {} %)".
          format(round(svd_lesions_maps.explained_variance_ratio_.sum()*100, 2)))
plt.savefig(fname=os.path.join(saving_directory, "explained_variance_ratio.pdf"))
plt.show()

# Linear model with lesion principal components
# as predictor, and language score from PCA of
# NEEL battery test

# Load regression data file
acm_spreadsheet = pd.read_excel(
    io="/media/db242421/db242421_data/ConPagnon_data/regression_data/Resting State AVCnn_ cohort data.xlsx")

# subset to patients only
acm_patients_data = acm_spreadsheet.loc[acm_spreadsheet["Groupe"] == "P"][['pc1_language',
                                                                           'pc2_language',
                                                                           'pc3_language']]

# Merge by index with the dataframe containing
# the score from lesion mapping PCA
from data_handling.data_management import merge_by_index
acm_lesion_and_language_data = merge_by_index(dataframe1=acm_patients_data,
                                              dataframe2=pca_results_df)

# Set variable in the model
predictors = columns_names
target_variables = ["pc1_language", "pc2_language", "pc3_language"]

# Compute a linear model for each
# target variable with all the predictor
for target_variable in target_variables[0:1]:
    # formula
    formula = target_variable + "~" + "+".join(predictors)
    # Call OLS object
    language_lesion_model = smf.ols(formula=formula,
                                    data=acm_lesion_and_language_data).fit(method='qr')

    # write model results
    model_summary = language_lesion_model.summary2()
