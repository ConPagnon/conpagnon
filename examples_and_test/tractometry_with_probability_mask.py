import os
import nibabel as nb
import numpy as np
import collections
import pandas as pd


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Directory containing the subjects image folder
root_directory = '/neurospin/grip/protocols/MRI/Ines_2018/images/controls_v2'

# Subject text list
subject_txt = '/neurospin/grip/protocols/MRI/Ines_2018/images/controls_v2_nip'
subjects = open(subject_txt).read().split()
# Hemisphere side
sides = ["left", "right"]
# Diffusion parameters
params = ["FA", "ADC", "AD", "RD"]
# Tract name
tracts = ["AF", "MLF", "SLF_III"]


# Dictionary that will stock the diffusion parameters
# for each tract, for each side, for each subjects.
params_values_result = dict.fromkeys(subjects)
for subject in subjects:
    print("Extract mean values diffusion parameter for subject {}".format(subject))
    params_values_result[subject] = dict.fromkeys(tracts)
    for tract in tracts:
        params_values_result[subject][tract] = dict.fromkeys(sides)
        for side in sides:
            params_values_result[subject][tract][side] = dict.fromkeys(params)

            params_values_result[subject][tract][side] = dict.fromkeys(params)
            # full path of the corresponding bundle probability map
            tract_probability_map = os.path.join(root_directory, subject,
                                                 "TractSeg_outputs/tractseg_output/bundle_probabilities_old",
                                                 tract + "_" + side + ".nii.gz")
            tract_probability_image = nb.load(filename=tract_probability_map)
            tract_probability_data = tract_probability_image.get_data()
            tract_probability_affine = tract_probability_image.affine

            # Load bundle segmentation mask
            bundle_mask = os.path.join(root_directory, subject, "TractSeg_outputs", "tractseg_output",
                                                 "bundle_segmentations_mask", tract + "_" + side + ".nii.gz")
            bundle_mask_data = nb.load(bundle_mask).get_data()
            # Clean the probability map by multiplying the bundle binary mask with
            # the probability map to only keep voxels in the mask
            clean_probability_map = np.multiply(bundle_mask_data, tract_probability_data)
            # Get the non-zero voxel
            i, j, k = np.where(clean_probability_map != 0)
            # Get the probability values
            probability = clean_probability_map[i, j, k]
            for param in params:
                # print('Compute {} on {} for side {}'.format(param, tract, side))
                # Load the parametric map for the current diffusion parameter
                param_image_data = nb.load(os.path.join(
                    root_directory, subject,
                    "Microstructure", subject + "_" + param + ".nii.gz")).get_data()
                # Get the voxels within the probability mask
                param_voxels = param_image_data[i, j, k]
                # Compute the sum of the voxels values weighted by their
                # probability
                weighted_sum = np.sum(np.multiply(probability, param_voxels))
                # Finally compute the mean values, by dividing
                # the previous weighted sum by the number of voxels
                mean_param = weighted_sum / len(param_voxels)

                params_values_result[subject][tract][side][param] = mean_param
# Stack on top of each other, individual dataframe containing the
# previously computed parameters.
all_subjects_dataframe = []
all_values = []
for subject in subjects:
    subject_flat_dictionary = flatten(params_values_result[subject])
    columns_name = list(subject_flat_dictionary.keys())
    values = list(subject_flat_dictionary.values())
    values_array = np.zeros(shape=(1, len(values)))
    values_array[0, :] = values
    all_values.append(values_array)
    all_subjects_dataframe.append(pd.DataFrame(index=[subject], columns=columns_name,
                                               data=values_array))

# Concatenate a final dataframe with all of the variables
final_dataframe = pd.concat(all_subjects_dataframe)
# Save in the CSV format the final dataframe
final_dataframe.to_csv(os.path.join(root_directory, "all_diffusion_parameters_probability_mask.csv"))
