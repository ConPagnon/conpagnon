"""
Manipulating a brain atlas
==========================
**What you'll learn**: Manipulating and visualizing an atlas file.
"""

#%%
# **Author**: `Dhaif BEKHA <dhaif@dhaifbekha.com>`_

#############################################################################
# Retrieve the example atlas
# --------------------------
#
# For this example we will work on an functional brain atlas. Please download the
# `atlas <https://www.dropbox.com/s/wwmg0a4g3cjnfvv/atlas.nii?dl=1>`_ file
# and the corresponding regions `label <https://www.dropbox.com/s/3wuzwn14l7nksvy/atlas_labels.csv?dl=1>`_ file.
# This brain atlas cover the whole cortex with **72** symmetric regions of interest. We divided those regions
# of interest into **twelve known functional brain network.**


#%%
# .. note::
#    This example atlas is just an example among the numerous existing
#    brain atlas in the literature. In you're day to day analysis, if
#    don't have a tailored brain atlas, you can easily fetch one from
#    the `Nilearn datasets <https://nilearn.github.io/modules/reference.html#module-nilearn.datasets>`_
#    module.

##############################################################################
# Import all the module we'll need
# --------------------------------
from pathlib import Path
import os
from conpagnon.data_handling import atlas
from nilearn import plotting

##############################################################################
# Load the atlas, set labels colors
# ---------------------------------
# We assume that the atlas file, and labels file are in your home directory.

# Fetch the path of the home directory
home_directory = str(Path.home())
# Filename of the atlas file.
atlas_file_name = 'atlas.nii'
# Full path to atlas labels file
atlas_label_file = os.path.join(home_directory, 'atlas_labels.csv')
# Set the colors of the twelves network in the atlas
colors = ['navy', 'sienna', 'orange', 'orchid', 'indianred', 'olive',
          'goldenrod', 'turquoise', 'darkslategray', 'limegreen', 'black',
          'lightpink']
# Number of regions in each of the network
networks = [2, 10, 2, 6, 10, 2, 8, 6, 8, 8, 6, 4]

#%%
# .. important::
#    We enumerate the number of regions in each defined `networks` in the networks variable
#    in the order that they are presented in the atlas file. So the first network have
#    2 regions of interest, the second network have 10 regions of interest, and so on.
#    The list of colors follow naturally the same order. So the labels of the first two
#    regions of interest will be navy, and so on.

# We can call fetch_atlas to retrieve useful information about the atlas
atlas_nodes, labels_regions, labels_colors, n_nodes = atlas.fetch_atlas(
    atlas_folder=home_directory,
    atlas_name=atlas_file_name,
    network_regions_number=networks,
    colors_labels=colors,
    labels=atlas_label_file,
    normalize_colors=True)


##############################################################################
# :py:func:`conpagnon.data_handling.atlas.fetch_atlas` is very useful to retrieve
# useful information about the atlas such as the number of regions, the coordinates
# of each regions center of mass, or the colors in the RGB for each regions.

#############################################################################
# We can print the coordinates of each center of mass :

# Coordinates of the center of mass for each region in the atlas
print(atlas_nodes)

#############################################################################
# We can print the colors of each label region :

# Colors attributed to each region in the atlas
print(labels_colors)

#%%
# .. note::
#    To see all the labels colors available, please
#    visit `this page <https://matplotlib.org/3.1.0/gallery/color/named_colors.html>`_


#############################################################################
# We can print the number of region :

# Number of regions in this example atlas
print(n_nodes)

##############################################################################
# Basic atlas plotting
# --------------------
# We use `Nilearn plotting <https://nilearn.github.io/modules/reference.html#module-nilearn.plotting>`_
# capabilities to plot and visualize our atlas in a convenient way
plotting.plot_prob_atlas(maps_img=os.path.join(home_directory, atlas_file_name),
                         title="Example atlas of 72 brain regions")

##############################################################################
# Atlas with no supplementary information
# ---------------------------------------
# Sometimes, you won't have at disposal the segmentation of
# the atlas in networks, or you won't have a particular set
# of colors for the labels. In that case, you can simply
# import the atlas, and generate random colors:

# We can call fetch_atlas to retrieve useful information about the atlas
atlas_nodes, labels_regions, labels_colors, n_nodes = atlas.fetch_atlas(
    atlas_folder=home_directory,
    atlas_name=atlas_file_name,
    normalize_colors=False)

#############################################################################
# Now the label of a region is just the number of the region in the atlas,
# since we did not provide a label file:

# The labels of each region in now just a number
print(labels_regions)