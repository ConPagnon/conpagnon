"""
 Created by db242421 at 04/12/18

 """

import importlib
from conpagnon.data_handling import data_architecture, dictionary_operations, atlas, data_management
from conpagnon.utils import folders_and_files_management
from conpagnon.computing import compute_connectivity_matrices as ccm
from conpagnon.machine_learning import classification
from conpagnon.connectivity_statistics import parametric_tests
from conpagnon.plotting import display
import os
import pandas as pd
import numpy as np
from nilearn.image import load_img
import nibabel as nb
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

# Load the atlas information excel file
atlas_excel_file = '/media/db242421/db242421_data/atlas_AVCnn/atlas_version2.xlsx'
sheetname = 'complete_atlas'

atlas_information = pd.read_excel(atlas_excel_file, sheetname='complete_atlas')

atlas_img = load_img(img=os.path.join(atlas_folder, atlas_name))
atlas_affine = atlas_img.affine
atlas_data = atlas_img.get_data()

# Compute network information
network_dict = atlas.fetch_atlas_functional_network(atlas_excel_file=atlas_excel_file,
                                                    sheetname=sheetname,
                                                    network_column_name='network')
networks = list(network_dict.keys())

# Save in ...
save_in = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference/illustration_atlas'

# Compute a 3D image for each network
all_networks_data = []
for network in networks:
    # Network ROIs indices
    network_indices = np.array(network_dict[network]['dataframe']['atlas4D index'])
    # extract the corresponding ROI from the 4D atlas
    network_data = atlas_data[..., network_indices]
    network_data[network_data > 0.5] = 1

    # Compute the sum in the 4th dimension to
    # have a 3D image
    network_3d = np.sum(network_data, axis=3, dtype=np.int8)
    # Save a nifti image
    network_3d_img = nb.Nifti1Image(dataobj=network_3d,
                                    affine=atlas_affine)
    all_networks_data.append(network_3d)
    # Create a directory for each network
    network_directory = data_management.create_directory(
        directory=os.path.join(save_in, network),
        erase_previous=False)
    nb.save(img=network_3d_img,
            filename=os.path.join(network_directory, network + '.nii'))


all_3d_network = np.array(all_networks_data)
for n in range(len(networks)):
    i_non_zero, j_non_zero, k_non_zero = np.where(all_3d_network[n, :, :, :] != 0)
    # replace the non zero value by the network index in the list
    # of network name
    all_3d_network[n, i_non_zero, j_non_zero, k_non_zero] = n + 1


# Sum along the 4th dimension
atlas_3d = np.sum(all_3d_network, 0,
                  dtype=np.int8)

# Save all ROIs in one image
all_3d_network_img = nb.Nifti1Image(dataobj=atlas_3d,
                                    affine=atlas_affine)
nb.save(img=all_3d_network_img,
        filename=os.path.join(save_in, 'atlas3D.nii'))

