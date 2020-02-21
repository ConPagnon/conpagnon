import numpy as np
import os
from nilearn import image
from nilearn import plotting
import pandas as pd
import webcolors
import nibabel as nb

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:07:18 2017

ComPagnon version 2.0

@author: db242421
"""

"""

Atlas class for easy manipulation of atlas.
"""


class Atlas:
    """A class Atlas for computing useful information when dealing with
    atlases.
    
    Methods
    -------
    
    fetch_atlas
        Return the complete path to the atlas.
        
    loadAtlas
        Load the atlas images and return a 4D numpy array.
        
    GetRegionNumbers
        Return the numbers of regions in the Atlas.
        
    GetLabels
        Return the list of the labels of the atlas.
        
    GetCenterOfMass
        Return tha array of coordinates of center of mass to
        each atlas regions.

    UserLabelsColors
        Generate an array of users defined colors to the 
        labels for display purpose.
    
    """
    
    def __init__(self, path, name):
        """Class constructor : an atlas file is defined by it's name and
        absolute path.
        
        Parameters
        ----------
        
        path : str
            The absolute path to the atlas file.
            
        name : str
            The name of the atlas file including it's extension.
        
        """
        self.path = path
        self.name = name
        
    def fetch_atlas(self):
        """Fetch the complete path to the atlas file.
                
        """
        return os.path.join(self.path, self.name)
        
    def load_atlas(self):
        """Load the atlas image and return the corresponding
        4D numpy array.
        
        """
        
        atlas_obj = image.load_img(os.path.join(self.path,self.name))
        atlas_data = atlas_obj.get_data()
        return atlas_data
    
    def get_region_numbers(self):
        
        """Return the number of regions of the atlas
        
        """
        atlas_data = self.load_atlas()
        region_numbers = atlas_data.shape[3]
        return region_numbers
        
    def GetLabels(self, labelsFile, colname='labels'):
        """Read the labels text file of the atlas
        
        Parameters
        ----------
        
        labelsFile : str
            The full path to the label file of the atlas. Supported
            extension are : .csv, .txt, .xlsx or .xls. By default,
            the header of the labels file is the column labels name 
            entitled 'labels'.
            
        colname : str, optional
            The columns name containing the labels. Default is labels.
            If no header, leave None.
            
        Returns
        -------
        
        output : list
            The list of the labels.
        
        """
        
        if labelsFile.lower().endswith(('.csv', '.txt')):
            labels = pd.read_csv(labelsFile)
        elif labelsFile.lower().endswith(('.xlsx', '.xls')):
            labels = pd.read_excel(labelsFile)
            
        if colname:
            labels = labels[colname]
            
        return labels
        
    def get_center_of_mass(self, asanarray=False):
        """Compute centers of mass of the different atlas regions.
        
        Parameters
        ----------
        asanarray : bool, optional
            If True, then the array are return if numpy.array of 
            shape (number of regions, 3).
            
        Returns
        -------
        output : list or numpy.array
            The coordinates of the centers of mass for each regions 
            of the atlas.

        """
        
        regions_center = [plotting.find_xyz_cut_coords(roi) for roi in image.iter_img(
            imgs=os.path.join(self.path, self.name))]
    
        if asanarray:
            regions_center = np.array(regions_center)
        
        return regions_center

    def user_labels_colors(self, networks, colors):
        """Generates user defined labels colors for each label of the atlas.
        
        Parameters 
        ----------
        
        networks : list
            The list containing the numbers of regions for each networks.
            
        colors : list
            The list of colors for each networks.
            
        Returns
        -------
        
        output : numpy.array, shape(number of regions, 3)
            The array containing the numerical values for the RGB space
            for the given colors entered in colors. For the normalized RGB space, 
            you just have divided the values of this array by 255.
            
        References
        ----------
        
        Please find all possible colors at
        [1] https://matplotlib.org/examples/color/named_colors.html
        
        """
        # Number of network detected
        n_networks = len(networks)
        # Number of colors, must match the number of network
        n_colors = len(colors)

        name_colors_to_rgb = [webcolors.name_to_rgb(colors[i]) for i in range(n_colors)]
        # convert string color name to rgb triplet
        rgb_labels_colors = np.array(name_colors_to_rgb)
        # Fill an array containing the colors for each regions, according to
        # network groups
        labels_colors = np.zeros([1,3])
        for i in range(n_networks):
            # temporary array for current networks labels
            tmp_labels_colors = np.zeros([networks[i], 3])
            # Fill the temporary array with the labels corresponding to the current network
            tmp_labels_colors[:] = rgb_labels_colors[i]
            # We append the current network labels colors array to the general array
            labels_colors = np.append(labels_colors, tmp_labels_colors,axis=0)
        # Delete the first line
        labels_colors = np.delete(labels_colors, 0, axis=0)
        return labels_colors


def fetch_atlas_functional_network(atlas_excel_file, sheetname, network_column_name):
    """Return a dictionary containing information for all functional networks.
       Information for all network is simply fetch in the excel file.

    Parameters
    ----------
    atlas_excel_file: str
        The full path to the excel file containing all the information on your atlas.
    sheetname: str
        The active sheet name in the atlas excel file.
    network_column_name: str
        The name of columns containing the label for all the functional networks.

    Returns
    -------
    output: dict
        A dictionnary with the networks name as keys, and the sub-dataframe and the number
        of roi for each networks as values.
    """
    df = pd.read_excel(atlas_excel_file, sheetname=sheetname)

    # Find the different network
    # List of networks
    networks_name_list = list(set(df[network_column_name]))

    # Create a dictionary with networks name as keys, and all the corresponding information columns
    # as values.
    networks_dictionary = dict.fromkeys(networks_name_list)
    # Group by functional network according to the labels in network_column_name
    df_group_by_networks = df.groupby(network_column_name)
    # Save the sub-dataframe corresponding to each network in the networks_dictionary.
    for network in networks_name_list:
        network_df = df_group_by_networks.get_group(name=network)
        networks_dictionary[network] = {'dataframe': network_df, 'number of rois': len(network_df.index)}

    return networks_dictionary


def generate_3d_img_network(reference_4datlas, atlas_information_xlsx_file, network_column_name,
                            sheetname, atlas4d_index_keys, atlas3d_label_key,
                            save_network_img_directory):
    """This function generate a 3D NifTi file for each defined functional network in a 4D atlas.

    """
    # Fetch network information for all networks
    networks_dictionnary = fetch_atlas_functional_network(
        atlas_excel_file=atlas_information_xlsx_file,
        sheetname=sheetname, network_column_name=network_column_name)

    # Load the atlas to get the shape and affine of images
    atlas4d = image.load_img(reference_4datlas)
    atlas4d_array = atlas4d.get_data()
    atlas_affine = atlas4d.affine
    atlas_shape = atlas4d.shape[:3]

    for network in networks_dictionnary.keys():
        # Get the number of rois
        n_roi_network = networks_dictionnary[network]['number of rois']
        # Initialize a 4D array containing the roi of the current network in 4D dim
        network4d = np.zeros(atlas_shape + (n_roi_network,))
        # Fetch the number of current network rois in the 4D atlas file
        network_roi_number = networks_dictionnary[network]['dataframe'][atlas4d_index_keys]
        # Find the corresponding 3D label of the network, because network_roi_number is a dataframe
        # we use the iloc method to find the corresponding value for the network_roi_number INDEX in the
        # right order.
        network_roi_index = network_roi_number.index
        network_3d_labels = list(networks_dictionnary[network]['dataframe'][atlas3d_label_key].loc[network_roi_index])
        # Fill the network 4D array with the corresponding 3D labels of each roi
        for r in range(n_roi_network):
            # Fill the 4d network array with the corresponding rois of the current network
            network4d[:, :, :, r] = atlas4d_array[:, :, :, list(network_roi_number)[r]]
            # Put the right 3D label for each roi in the current network
            x, y, z = np.where(network4d[:, :, :, r] != 0)
            network4d[x, y, z, r] = network_3d_labels[r]
            # Sum along the 4d dim to get one 3D image for the current network
            network_3d_array = np.sum(network4d, axis=3)
            # Save the network 3D image in NiFti format
            nb.save(img=nb.Nifti1Image(dataobj=network_3d_array, affine=atlas_affine),
                    filename=os.path.join(save_network_img_directory, '3d_labels_' + network + '.nii'))


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def fetch_atlas(atlas_folder, atlas_name,
                colors_labels='auto',
                network_regions_number='auto',
                labels='auto',
                normalize_colors = False):
    """Return important information from an atlas file.

    Parameters
    ----------
        atlas_folder: str
            The full path to the directory containing the atlas
        atlas_name: str
            The filename of the atlas file.
        colors_labels: str, or list. Optional
            If set to 'auto', the labels of the
            ROI name will get a random colors. Else,
            if a list of colors is provided, ROIs belonging
            to a network will get the desired colors. The colors
            should be in the same order of the network in the atlas
            file. The length of the list should match the number of
            network in the atlas.
        network_regions_number: list, Optional.
            If set to 'auto', random colors will be chosen.
            If a list of the number of regions in each network
            is provided, the corresponding color, in the color list
            will be applied to the corresponding number in the list.
        labels: list, optional
            The list of the ROI labels. If not provided,
            the ROI name is simply it's position in the atlas
            file.
        normalize_colors: bool, optional
            If True, all triplets in the RGB space are
            divided by the maximum 255.

    Returns
    -------
    output 1: numpy.array
        The coordinates of the center of mass of
        each ROI in the atlas. An array of shape (n_rois, 3).
    output 2: list
        The name of each ROIs in the atlas. A list
        of length (n_rois, ).
    output 3: numpy.array
        The array containing the colors in the
        RGB space of each ROIs. An Array of shape
        (n_rois, 3).
    output_4: int
        The number of ROIs in the atlas.

    """

    atlas_ = Atlas(path=atlas_folder,
                   name=atlas_name)

    # Fetch nodes coordinates
    atlas_nodes = atlas_.get_center_of_mass()
    # Fetch number of nodes in the atlas
    n_nodes = atlas_.get_region_numbers()
    # Atlas path
    if labels == 'auto':
        labels_regions = np.arange(n_nodes)
    else:
        labels_regions = atlas_.GetLabels(labels)
    # if the user give a list containing the number of regions in each networks,
    # but not the colors
    if (colors_labels == 'auto') & (type(network_regions_number) == list):
        n_network = len(network_regions_number)
        # Generate n_network random colors
        random_colors = np.random.randint(0, 256, size=(n_network, 3))
        # Convert list of triplet to name with webcolors
        regions_colors_name = [get_colour_name(random_colors[i])[1] for i in range(n_network)]
        # Generate the colors for each labels according to the network they belong
        labels_colors = atlas_.user_labels_colors(networks=network_regions_number,
                                                  colors=regions_colors_name)
        if normalize_colors:
            labels_colors = (1/255)*labels_colors
    # if the user doesn't give any colors labels nor networks region number
    # we create random color for each regions labels
    elif (colors_labels == 'auto') & (network_regions_number == 'auto'):
        # Generate n_network random colors
        random_colors = np.random.randint(0, 256, size=(n_nodes, 3))
        # Convert list of triplet to name with webcolors
        regions_colors_name = [get_colour_name(random_colors[i])[1] for i in range(n_nodes)]
        # Generate the colors for each labels according to the network they belong
        labels_colors = regions_colors_name
        if normalize_colors:
            labels_colors = (1/255)*np.array(labels_colors)
    else:
        labels_colors = atlas_.user_labels_colors(networks=network_regions_number,
                                                  colors=colors_labels)
        if normalize_colors:
            labels_colors = (1/255)*labels_colors

    return atlas_nodes, labels_regions, labels_colors, n_nodes


