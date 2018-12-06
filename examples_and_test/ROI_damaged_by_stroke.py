"""
 Created by db242421 at 30/11/18

 """
import importlib
from data_handling import atlas, data_architecture, dictionary_operations
from utils import folders_and_files_management
from computing import compute_connectivity_matrices as ccm
from sklearn import covariance
from machine_learning import classification
from connectivity_statistics import parametric_tests
from plotting import display
from matplotlib.backends import backend_pdf
from data_handling import data_management
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from connectivity_statistics import regression_analysis_model
from plotting.display import t_and_p_values_barplot
from nilearn.image import load_img, resample_to_img
# Reload all module
importlib.reload(data_management)
importlib.reload(atlas)
importlib.reload(display)
importlib.reload(parametric_tests)
importlib.reload(ccm)
importlib.reload(folders_and_files_management)
importlib.reload(classification)
importlib.reload(data_architecture)
importlib.reload(dictionary_operations)

# Atlas setting

# Atlas set up
atlas_folder = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference'
atlas_name = 'atlas4D_2.nii'
labels_text_file = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference/atlas4D_2_labels.csv'
colors = ['navy', 'sienna', 'orange', 'orchid', 'indianred', 'olive',
          'goldenrod', 'turquoise', 'darkslategray', 'limegreen', 'black',
          'lightpink']
# Number of regions in each user defined networks
networks = [2, 10, 2, 6, 10, 2, 8, 6, 8, 8, 6, 4]
# Atlas path
# Read labels regions files
atlas_nodes, labels_regions, labels_colors, n_nodes = atlas.fetch_atlas(
    atlas_folder=atlas_folder,
    atlas_name=atlas_name,
    network_regions_number=networks,
    colors_labels=colors,
    labels=labels_text_file,
    normalize_colors=True)
atlas_img = load_img(os.path.join(atlas_folder, atlas_name))

# Lesions path setting
root_directory = "/media/db242421/db242421_data/AVCnn_2016_DARTEL/AVCnn_data/patients"
patients_txt = "/media/db242421/db242421_data/ConPagnon_data/text_data/acm_patients.txt"
subjects = list(pd.read_csv(patients_txt, header=None)[0])
lesion_folder = "lesion"
save_in = "/media/db242421/db242421_data/ConPagnon_data/ROI_analysis"

# Store damaged ROIs for each subjects
damaged_rois = dict.fromkeys(subjects)
# Store the count for each roi
count_damaged_rois = dict.fromkeys(labels_regions)
import glob

for subject in subjects:
    damaged_rois[subject] = dict.fromkeys(labels_regions)
    lesion_image_path = glob.glob(os.path.join(root_directory,
                                               subject, lesion_folder, 'w*.nii'))[0]
    lesion_image = load_img(lesion_image_path)
    # Resample lesion image to the atlas ROI resolution
    lesion_image_resampled = resample_to_img(source_img=lesion_image,
                                             target_img=atlas_img,
                                             interpolation='nearest')
    lesion_image_data = lesion_image_resampled.get_data()
    for roi in range(n_nodes):
        # For each ROI, compute the intersection and count the number of voxel
        intersection_roi_lesion = np.sum(np.multiply(lesion_image_data, atlas_img.get_data()[..., roi]))
        if intersection_roi_lesion != 0:
            damaged_rois[subject][labels_regions[roi]] = {
                'status': 'damaged',
                'common voxels': intersection_roi_lesion,
                '% lesioned': intersection_roi_lesion/np.sum(atlas_img.get_data()[..., roi])*100,
                'volume lesioned ROI': np.sum(atlas_img.get_data()[..., roi]) - intersection_roi_lesion
                                                          }
        else:
            damaged_rois[subject][labels_regions[roi]] = {'status': 'intact'}

# Count for each ROI the number of time
# the ROI is damaged across subject
for roi in range(n_nodes):
    count_damage = 0
    for subject in subjects:
        if 'common voxels' in damaged_rois[subject][labels_regions[roi]].keys():
            print(labels_regions[roi])
            count_damage += 1

            count_damaged_rois[labels_regions[roi]] = {'# of hit': count_damage}


import seaborn as sns
# Plot histogram
all_counts = [count_damaged_rois[labels_regions[i]]['# of hit'] for i in range(n_nodes) if
              '# of hit' in list(count_damaged_rois[labels_regions[i]].keys())]
# Put in grey the bar above the type I error threshold
bar_color = labels_colors

with backend_pdf.PdfPages(os.path.join(save_in, "ROI_damaged.pdf")) as pdf:

    plt.figure(constrained_layout=True)
    ax = sns.barplot(x=all_counts, y=labels_regions, palette=bar_color,
                     orient="h")
    ax.set(xlim=(0, 15), ylabel="",
           xlabel="Number of time a ROI take a hit")
    # Set font size for the x axis, and rotate the labels of 90° for visibility
    for ytick, color in zip(ax.get_yticklabels(), bar_color):
        ytick.set_color(color)
        ytick.set_fontsize(4)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    pdf.savefig()
    plt.show()

folders_and_files_management.save_object(
    object_to_save=damaged_rois,
    saving_directory=save_in,
    filename='damaged_rois.pkl'
)
folders_and_files_management.save_object(
    object_to_save=count_damaged_rois,
    saving_directory=save_in,
    filename='count_damaged_rois.pkl'
)