"""
Statistics in the tangent space
===============================
**What you'll learn**: Compute individual
statistics in the the **tangent** space,
and pinpoint functional connectivity
differences in an individual from a
reference group.
"""

#%%
# **Author**: `Dhaif BEKHA <dhaif@dhaifbekha.com>`_

#############################################################################
# Retrieve the example dataset
# ----------------------------
#
# In this section of the tutorials, we will need a set of brain signals
# from two distinct group of subjects. You will work on pre-computed
# set of times series extracted from 72 brain regions atlas. The first
# group, called *controls* is a set of brain signals from healthy
# seven years old children, and the second group called *patients*, is a set of
# brain signals from seven years old children who have suffered a stroke. You can
# download the times series dictionary storing those signals
# `here <https://www.dropbox.com/s/1r71emzacxt93rv/times_series.pkl?dl=1>`_.
# Finally, we will plot some results on a glass brain, and we will need
# the nodes coordinates of the atlas regions in which the signals were extracted.
# You can download the `atlas <https://www.dropbox.com/s/wwmg0a4g3cjnfvv/atlas.nii?dl=1>`_,
# and the corresponding `labels <https://www.dropbox.com/s/3wuzwn14l7nksvy/atlas_labels.csv?dl=1>`_.
# As usual, we will suppose that all needed files are in your **home directory**.

#############################################################################
# Modules import
# --------------
from pathlib import Path
import os
from conpagnon.utils.folders_and_files_management import load_object
from conpagnon.computing.compute_connectivity_matrices import tangent_space_projection
from conpagnon.data_handling import atlas
import numpy as np
from matplotlib.backends import backend_pdf
import matplotlib.pyplot as plt
from nilearn.connectome import vec_to_sym_matrix
from nilearn.plotting import plot_connectome
import networkx as nx

#############################################################################
# Load the functional atlas
# -------------------------
#
# We load the atlas that we already used in multiple section of
# the tutorials section. Please, refer to the very first section
# were we manipulated the exact same atlas.


# Fetch the path of the home directory
home_directory = str(Path.home())

# Atlas set up
atlas_folder = home_directory
atlas_name = 'atlas.nii'
labels_text_file = os.path.join(home_directory, 'atlas_labels.csv')
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


#############################################################################
# A brief theoretical background
# ------------------------------
#
# In this section we will you the theoretical basis and mathematical
# development to help you understand the advantages of using the
# **tangent space** metric in functional connectivity. We encourage
# you to dive in more depth in the subject by reading the references
# we select
# `here <https://www.dropbox.com/s/2io3k55r5n4o6rd/reference_1.pdf?dl=1>`_
# and `here <https://www.dropbox.com/s/xa6cqm6p8ry9enm/reference_2.pdf?dl=1>`_ !

##########################################################################
# Problematic: Detecting functional connectivity difference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The statistical analyses of functional connectivity based
# on connectivity matrices aim to find **differences**
# between subject in the same group, or in two different
# group for example. So the first difficulty is to find
# a sensible enough parametrization of those functional
# connectivity matrices. It is common to use the correlation
# as the base metric, and compare correlation coefficients
# across subjects. This simple procedure can be modelled
# by the following linear model, for a subject :math:`s`,
# the correlation matrix :math:`\Sigma^{s}` is:

#%%
# .. math::
#
#    \Sigma^{s} = \Sigma_{0} + d\Sigma^{s}

#%%
# where :math:`\Sigma_{0}` is the mean covariance matrix
# of the whole group, and :math:`d\Sigma^{s}` encode the
# subject specific contribution in functional connectivity.
# Based on correlation metric, :math:`d\Sigma^{s}` can be
# very difficult to model, drown in the natural dependence
# that inherently exist between the functional connectivity
# coefficients in a correlation matrix. In the following
# sub-section, we detailed one solution based on the
# **projection** of those correlation matrices in
# another space.

##########################################################################
# The projection in the tangent space
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The main problem to mitigate is the natural inter-dependence
# of functional connectivity coefficient that arise with
# correlation-like metric. Indeed, mathematically, those
# matrices belong to the **symmetric positive definite**
# space, that is, for any vector :math:`v`,
# and :math:`\Sigma` a correlation matrix, then we have
# that  :math:`v^{T}\Sigma v > 0`. Considering two
# correlation matrices :math:`\Sigma_{1}` and :math:`\Sigma_{2}`,
# we can alleviate this problem by **projecting onto the tangent
# space** :math:`\Sigma_{2}` **at the point** :math:`\Sigma_{1}`.
# Back to our functional connectivity, let's consider
# :math:`\Sigma^{s}` the subject correlation matrix of
# subject :math:`s`, :math:`\Sigma_{0}` the mean
# correlation matrix of the whole group, and finally
# :math:`d\Sigma^{s}`, the subject-specific contribution
# we have to compute:

#%%
# .. math::
#
#    \Sigma^{s} = \Sigma_{0}^{1/2} exp(d\Sigma^{s}) \Sigma_{0}^{1/2}

#%%
# The :math:`d\Sigma^{s}` matrix, is the **tangent space matrix**
# of the subject :math:`s`. It is reasonable to consider that
# :math:`d\Sigma^{s}` is very small, :math:`||d\Sigma^{s}||_{2} << 1`,
# so that the above equation simplify in:

#%%
# .. math::
#
#    \Sigma^{s} = \Sigma_{0}^{1/2} (I_{n} + d\Sigma^{s}) \Sigma_{0}^{1/2}

#%%
# Using the above equation, we can finally compute :math:`d\Sigma^{s}`:

#%%
# .. math::
#
#    d\Sigma^{s} = \Sigma_{0}^{-1/2} \Sigma^{s} \Sigma_{0}^{-1/2} - I_{n}

#%%
# Giving :math:`\Sigma_{0}`, the projection matrix point, we can now
# compute for every subject, at that point, the tangent space functional
# connectivity matrix.

#%%
# .. tip::
#    The computation of the tangent space matrices
#    is implemented in the **Nilearn** library, that
#    you may already use, in the previous tutorial in
#    ConPagnon. Please, see the official documentation
#    of the `Nilearn package <https://nilearn.github.io/index.html>`_
#    for more information. In this implementation, you **must**
#    give the **whole stack** of all times series from all groups
#    to compute the tangent space matrices. In ConPagnon, we implement
#    a more flexible version, where you can give **a reference point**,
#    and the **points to project**, in the function
#    :py:func:`conpagnon.computing.compute_connectivity_matrices.tangent_space_projection`
#    function

#%%
# In the next section of this tutorial, we will
# use the tangent space metric to compute a
# test statistic, that pinpoint **for each patient**
# in the patients group, the functional connectivity
# differences from the mean correlation matrices
# of the control group.

#############################################################################
# Subject specific statistic in the tangent space
# -----------------------------------------------
#
# In this section, we will adapt the algorithm describe
# by Gael Varoquaux et al in
# this `reference <https://www.dropbox.com/s/2io3k55r5n4o6rd/reference_1.pdf?dl=1>`_.
# With the same notation as above, we will first compute :math:`\Sigma_{0}`, which
# which is the mean correlation matrix of the controls group. Then, for each
# patient :math:`s`, we will project the correlation matrix :math:`\Sigma^{s}` at
# :math:`\Sigma_{0}` to compute the tangent space connectivity matrix, :math:`d\Sigma^{s}`. We
# also compute the tangent matrices for the controls group.
# Finally, for each patient, we will compute a **one sample t-test** between :math:`d\Sigma^{s}`
# and :math:`{ d\Sigma_{0}^{1} ... d\Sigma_{0}^{N} }`,
# the tangent matrices of the controls group. To assess the
# significance of each p-value, for each subject, we generate
# the null distribution for each coefficient by **bootstrapping of the reference group.**.
# The user have to set the size of the bootstrapped sample  :math:`m`, the number of time
# we repeat the bootstrapping process, and the level of significance  :math:`\alpha`.


# Load the times series dictionary in your
# home directory
times_series_dictionary = load_object(
    full_path_to_object=os.path.join(home_directory, 'times_series.pkl'))

# Retrieve the groups in the study: it's
# simply the keys of the dictionary
groups = list(times_series_dictionary.keys())

# Name of the projected group
projected_group_name = "patients"

# Name of the reference group
reference_group_name = "controls"

# Subjects list in each group:
reference_group = list(times_series_dictionary[reference_group_name].keys())
subjects_to_project = list(times_series_dictionary[projected_group_name].keys())

# Stack the times series for each group
reference_group_time_series = np.array([times_series_dictionary[reference_group_name][s]['time_series']
                                        for s in reference_group])
group_to_project_time_series = np.array([times_series_dictionary[projected_group_name][s]['time_series']
                                         for s in subjects_to_project])

# Number of bootstrap
m = 10000
# Size of the bootstrapped sample
size_subset_reference_group = 15
# Level of Type-I error rate
alpha = 0.01

# Compute the tangent space projection, followed
# by the one sample t-test and estimation of the
# null distribution.
tangent_space_projection_dict = tangent_space_projection(
    reference_group=reference_group_time_series,
    group_to_project=group_to_project_time_series,
    bootstrap_number=m,
    bootstrap_size=size_subset_reference_group,
    output_directory=home_directory,
    verif_null=False,
    statistic='t',
    correction_method="fdr_bh",
    alpha=alpha)

#%%
# .. note::
#    The null distribution :math:`H_{i,j}` for
#    functional connectivity coefficient :math:`i,j`
#    is estimated by leave-one-out inside the bootstrapped
#    sample. We compute the mean matrix of all subject in
#    the reference group, and project all subject in the
#    sample, onto the tangent space at that mean matrix.
#    Finally, we perform a one sample t-test between the
#    leftout subject and the mean group in the tangent space.

#%%
# We stored all the useful results (the null distribution, the statistic
# for each subjects....) in the variable ``tangent_space_projection_dict``.
# We will first retrieve the corrected p-value and corresponding t-statistic
# for each subject, and then plot it on a glass brain. We will use the atlas
# that you previously downloaded in the first section of this tutorial.


# Corrected p values for each projected subject
p_values_corrected = tangent_space_projection_dict['p_values_corrected']
# Tangent Connectivity matrices for each projected subject
group_to_project_tangent_matrices = tangent_space_projection_dict['group_to_project_tangent_matrices']
# Reference group mean correlation matrices
reference_group_tangent_mean = tangent_space_projection_dict['reference_group_tangent_mean']
# output statistic for each projected subject
group_to_project_stats = tangent_space_projection_dict['group_to_project_stats']

# Initialize a empty adjacency matrix for plotting purpose,
# if for a subject no significant nodes are to be found.
empty_adjacency_matrix = np.zeros(shape=(n_nodes, n_nodes))


# We will plot only the first five patients glass brain
with backend_pdf.PdfPages(os.path.join(home_directory, 'tangent_space_statistic_report')) as pdf:
    for subject in range(len(subjects_to_project))[0:5]:
        # compute node degree for each subject
        # based on the surviving connection
        group_to_project_significant_edges = vec_to_sym_matrix(p_values_corrected[subject, ...] < alpha,
                                                               diagonal=np.zeros(n_nodes))
        patient_adjacency_matrices = nx.from_numpy_array(group_to_project_significant_edges)
        degrees = np.array([val for (node, val) in patient_adjacency_matrices.degree()])*40
        # plot corrected connection
        if np.unique(group_to_project_significant_edges).size == 1:
            plot_connectome(
                adjacency_matrix=empty_adjacency_matrix,
                node_coords=atlas_nodes,
                node_color=labels_colors,
                title='{}'.format(
                    subjects_to_project[subject]),
                colorbar=False,
                node_kwargs={'edgecolor': 'black', 'alpha': 1})
            plt.tight_layout()
            pdf.savefig()
            plt.show()
        else:
            plot_connectome(
                adjacency_matrix=vec_to_sym_matrix(np.multiply(p_values_corrected[subject, ...] < alpha,
                                                               group_to_project_tangent_matrices[subject, ...]),
                                                   diagonal=np.zeros(n_nodes)),
                node_coords=atlas_nodes,
                node_color=labels_colors,
                title='{}'.format(
                    subjects_to_project[subject]),
                colorbar=True,
                node_size=degrees,
                node_kwargs={'edgecolor': 'black', 'alpha': 1},
                edge_threshold=None)
            plt.tight_layout()
            pdf.savefig()
            plt.show()

    plot_connectome(adjacency_matrix=reference_group_tangent_mean,
                    node_coords=atlas_nodes,
                    node_color=labels_colors, edge_threshold='80%',
                    title='{} tangent mean'.format(reference_group_name),
                    colorbar=True)
    plt.tight_layout()
    pdf.savefig()
    plt.show()

#%%
# We only plot a couple of glass brain in the above plot.
# As you can see, the way functional connectivity is
# impacted **differ patients to patients**. You see
# that the number of time a certain node is impacted
# is also different. The projection in the tangent space
# allow us then, to explore **at the individual level**
# the difference in functional connectivity in a subject
# regarding to a **mean reference** group.

#%%
# .. note::
#    In the above glass brain, the size of the nodes are
#    proportional to the number of times they make significant
#    link with other nodes. The edge between two nodes, are
#    red if the connectivity coefficient are positive and blue
#    this coefficient is negative.


#%%
# Depending on what you are studying, with our implementation
# of the projection of the tangent space, you can easily choose
# another reference group. For example, we can imagine that inside
# the only patients population, there is a homogeneous sub-group
# of patients regarding one or multiple parameters that can serve
# as a reference to study the remaining patients subjects.
