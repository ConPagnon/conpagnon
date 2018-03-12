import numpy as np
import os
from nilearn import image
from nilearn import plotting
import pandas as pd
import webcolors
import warnings
import nibabel as nb

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:07:18 2017

ComPagnon version 2.0

@author: db242421
"""

"""

Création de la classe de base Atlas
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
        
    RandomNodesLabelsColors
        Generate an array of randoms colors to the labels for
        display purposes.
        
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
        
    def loadAtlas(self): 
        """Load the atlas image and return the corresponding
        4D numpy array.
        
        """
        
        atlas_obj = image.load_img(os.path.join(self.path,self.name))
        atlas_data = atlas_obj.get_data()
        return atlas_data
    
    def GetRegionNumbers(self):
        
        """Return the number of regions of the atlas
        
        """
        atlas_data = self.loadAtlas()
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
        
    def GetCenterOfMass(self, asanarray = False):
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
        
        regions_center = [plotting.find_xyz_cut_coords(roi) for roi in image.iter_img(imgs= 
                          os.path.join(self.path, self.name))]
    
        if asanarray:
            regions_center = np.array(regions_center)
        
        return regions_center
    
    
    def RandomNodesLabelsColors(self):
        """Generates randoms labels colors for each label of the atlas.
        
        """
        # On recupere le nombres de labels, soit le nombre de régions
        n_labels = self.GetRegionNumbers()
        
        # On genere un tableaux, de triplet reprensentant une couleur dans l'espace RGB
        nodes_colors = np.random.rand(n_labels, 3)
        
        random_nodes_colors = dict([(i,nodes_colors[i]) for i in range(n_labels)])
        
        return random_nodes_colors
        
    def UserLabelsColors(self, networks, colors):
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
        # nombre de reseaux entré
        n_networks = len(networks)
        # Nombre de couleurs entré
        n_colors = len(colors)

        
        #Si le nombre de couleurs diffère de celui du nombre de reseaux on 
        #choisit les couleurs de facon aléatoire et on affiche un warning
        if n_networks != n_colors:
            warnings.warn("Le nombre de couleur est differente du nombre de reseau,les couleurs seront choisies au hasard")
            random_colors = self.RandomNodesLabelsColors()
            labels_colors = np.array(random_colors.values())
        elif n_networks == n_colors:
            #On convertis les noms en couleurs RGB
            name_colors_to_rgb = [ webcolors.name_to_rgb(colors[i]) for i in range(n_colors) ]
            #On convertis la liste de tuple en tableaux de tableaux
            rgb_labels_colors = np.array(name_colors_to_rgb)
            #Je crée un labels factice de zeros pour pouvoir enmpiler 
            #les couleurs des different reseaux
            labels_colors = np.zeros([1,3])
            for i in range(n_networks):
                #Je crée un tableau de labels temporaire pour le reseaux i
                tmp_labels_colors = np.zeros([networks[i], 3])
                #On remplis tous les labels appartenant au reseaux i par la bonne 
                #couleur
                tmp_labels_colors[:] = rgb_labels_colors[i]
                #On concatene le tableaux de labels du reseau i avec le tableau 
                #general regroupant toutes les couleurs
                labels_colors = np.append(labels_colors, tmp_labels_colors,axis=0)
            #On efface la premiere ligne qui était un labels factice
            labels_colors = np.delete(labels_colors, 0, axis = 0)
        return labels_colors


def fetch_atlas_functional_network(atlas_excel_file, sheetname, network_column_name):
    """Return a dictionnary containing information for all functional networks.
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
    # number of networks
    n_networks = len(networks_name_list)

    # Create a dictionnary with networks name as keys, and all the corresponding information columns
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


