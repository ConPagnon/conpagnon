"""
Two samples t-test
==================
**What you'll learn**: Compute a two-samples t-test between two groups of
connectivity matrices, and plot the results on a glass brain.
"""

#%%
# **Author**: `Dhaif BEKHA <dhaif@dhaifbekha.com>`_

#############################################################################
# Retrieve the example dataset
# ----------------------------
#
# In this example, we will work directly on a pre-computed dictionary,
# that contain two set of connectivity matrices, from two different groups.
# The first group, called *controls* is a set of connectivity matrices from healthy
# seven years old children, and the second group called *patients*, is a set of
# connectivity matrices from seven years old children who have suffered a stroke.
# You can download the dictionary use in this example
# `here <https://www.dropbox.com/s/kwdrx4liauo10kr/raw_subjects_connectivity_matrices.pkl?dl=1>`_.


##############################################################################
# Module import
# -------------

from conpagnon.utils.folders_and_files_management import load_object, save_object
from conpagnon.connectivity_statistics.parametric_tests import two_samples_t_test
from conpagnon.plotting.display import plot_ttest_results, plot_matrix
from conpagnon.data_handling import atlas
from pathlib import Path
import os
import matplotlib.pyplot as plt
##############################################################################
# Load data, and set Path
# -----------------------
#
# We first load the dictionary containing the connectivity matrices for each
# group of subjects. We will work as usual in your home directory. We will also
# explore what's in this dictionary, such as the different group, the number of subject ...

# Fetch the path of the home directory
home_directory = str(Path.home())

# Load the dictionary containing the connectivity matrices
subjects_connectivity_matrices = load_object(
    full_path_to_object=os.path.join(home_directory, 'raw_subjects_connectivity_matrices.pkl'))

# Fetch the group name
groups = list(subjects_connectivity_matrices.keys())
print(groups)

# Number of subjects in the control, and
# patients group
print('There is {} subjects in the {} group, and {} in the {} group'.format(
    len(subjects_connectivity_matrices[groups[0]]), groups[0],
    len(subjects_connectivity_matrices[groups[1]]), groups[1]))

# Print the list of connectivity metric available,
# taking the first subject, in the first group for
# example:
print('List of computed connectivity metric: {}'.format(subjects_connectivity_matrices['controls']))

#%%
# .. note::
#   As you can see, the dictionary is a very convenient way to store data. You can
#   as many field as you want, and you can fetch very easily any data from a particular
#   subject.

##############################################################################
# Compute a simple t-test
# ----------------------
#
# We will compute a two samples t-test between the control group and the
# patients group. We will compute this test for the three connectivity
# metric we have at disposal in the dictionary. The results, will be store
# in a dictionary for convenience.

# Call the t-test function:
t_test_dictionary = two_samples_t_test(subjects_connectivity_matrices_dictionnary=subjects_connectivity_matrices,
                                       groupes=groups,
                                       kinds=['correlation', 'partial correlation', 'tangent'],
                                       contrast=[1, -1],
                                       preprocessing_method='fisher',
                                       alpha=.05,
                                       multicomp_method='fdr_bh')


#%%
# As you can see in the code above, we compute a t-test for three
# connectivity metric: **correlation**, **partial correlation** and
# **tangent**. The contrast we use between patients and controls is
# the vector [1, -1], that means the controls are the reference.
# We specify **fisher** as *preprocessing_method*, that mean for
# correlation and the partial correlation matrices, a z-fisher transform
# is applied before the t-test.

#%%
# .. note::
#   We applied a correction to deal with the classical
#   problem of multiple comparison. The correction by
#   default is FDR. Please, read the docstring of
#   the :py:func:`conpagnon.connectivity_statistics.parametric_tests.two_samples_t_test`
#   function for detailed explanation of the arguments.


# Explore the t_test_dictionary

# The first set of keys, is the list of
# connectivity metric we computed the t-test
# for:
print(list(t_test_dictionary.keys()))

# And in each connectivity key, we find different
# matrices storing the result of the t-test, for the
# correlation key for example:
print(list(t_test_dictionary['correlation'].keys()))

##############################################################################
# Plot the results on a glass brain
# ---------------------------------
#
# For a better understanding of the results, we can plot the results,
# directly on a glass brain. In ConPagnon, you can do it easily with
# the dedicated function **plot_ttest_results**. For plotting purposes
# only we will use in this section, the atlas we already manipulate in the
# first section. You can download the `atlas <https://www.dropbox.com/s/wwmg0a4g3cjnfvv/atlas.nii?dl=1>`_,
# and the corresponding `labels <https://www.dropbox.com/s/3wuzwn14l7nksvy/atlas_labels.csv?dl=1>`_
# for each regions.

#%%
# .. warning::
#   All those files, as a reminder, should be in your home
#   directory.

# First, we will load the atlas, and fetching
# in particular, the nodes coordinates of each regions
# because we will need those coordinates for the glass brain
# plotting

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
# We can call fetch_atlas to retrieve useful information about the atlas
atlas_nodes, labels_regions, labels_colors, n_nodes = atlas.fetch_atlas(
    atlas_folder=home_directory,
    atlas_name=atlas_file_name,
    network_regions_number=networks,
    colors_labels=colors,
    labels=atlas_label_file,
    normalize_colors=True)

# Now we can plot the t-test results
# on a glass brain
plot_ttest_results(t_test_dictionnary=t_test_dictionary,
                   groupes=groups,
                   contrast=[1, -1],
                   node_coords=atlas_nodes,
                   node_color=labels_colors,
                   output_pdf=os.path.join(home_directory, 't_test_results.pdf'))
plt.show()

#%%
# We plotted the t-test results for each
# connectivity metrics. In each glass brain,
# we only plot the edges between rois
# associated with a corrected p-values under
# the user type I error rate. For those edges,
# we plot the difference in the mean connectivity
# between the two group, according the desired
# contrast. We also generate in your home directory
# a simple Pdf report with the three glass brain.

#%%
# .. important::
#   As you can see, the results are quite similar
#   between *partial correlation* and the *tangent*
#   connectivity metric, but very different from the
#   *correlation* metric. Indeed, you have to choose
#   very carefully the metric, depending on various
#   parameter: the size of your sample, the effect size
#   of the parameter you study, the problem you wan to resolve.....

##############################################################################
# Plot the results on a matrix
# ----------------------------
#
# The glass brain is very good to have a quick visual
# view of the results projected on a brain, but we can
# also display the same results with a 2D matrix: a t-value
# matrix, with the corresponding p-value matrix. In that way,
# you will identified clearly which brain regions are involved
# in the computed contrast. We will compute those matrices for
# the tangent metric only, but it naturally apply for the other
# two metric.

# Metric we want to plot
metric = 'tangent'
# First we fetch the threshold t-values
# edges matrix
significant_edges_matrix = t_test_dictionary[metric]['significant edges']
# We also fetch the corrected p-values matrix
corrected_p_values_matrix = t_test_dictionary[metric]['corrected pvalues']

# We can plot the t-values matrix
plot_matrix(matrix=significant_edges_matrix, labels_colors=labels_colors,
            horizontal_labels=labels_regions, vertical_labels=labels_regions,
            linecolor='black', linewidths=.1,
            title='Thresholded t-values matrix for the {} metric'.format(metric))
plt.show()

# We can now plot the p-values matrix
plot_matrix(matrix=corrected_p_values_matrix, labels_colors=labels_colors,
            horizontal_labels=labels_regions, vertical_labels=labels_regions,
            linecolor='black', linewidths=.1, colormap='hot', vmax=0.05,
            title='Thresholded t-values matrix for the {} metric'.format(metric))
plt.show()

# Finally you can save the t test dictionary for further
# use if you want
save_object(object_to_save=t_test_dictionary,
            saving_directory=home_directory,
            filename='t_test_dictionary_example.pkl')

#%%
# .. note::
#   In those matrix plot, we only plot the lower triangle
#   of the matrix, indeed we only did half the test because
#   connectivity matrices are symetric. Note also the liberty we
#   have in the :py:func:`plot_matrix` function, in term of colormap,
#   max and min values....
