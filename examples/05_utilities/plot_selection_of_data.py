"""
Manipulation of the data dictionary
===================================
**What you'll learn**: Learn to quickly manipulate the subjects
connectivity dictionary, selecting sub-sets of connectivity
matrices regarding behavioral group or variables.
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
# `here <https://www.dropbox.com/s/60ehxt3fohnea2j/raw_subjects_connectivity_matrices.pkl?dl=1>`_.
# You will also need, the data table containing a set of continuous or categorical behavioral
# variable regarding all the subjects in the dictionary. You can download the table
# `here <https://www.dropbox.com/s/4fhexvm4ci9d6nz/data_table.xlsx?dl=1>`_. When
# downloaded, all the files must be stored in your **home directory**.

#############################################################################
# Modules import
# --------------

from conpagnon.data_handling import dictionary_operations, atlas, data_management
from conpagnon.utils.folders_and_files_management import load_object
import pandas as pd
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt


#############################################################################
# Load the data
# -------------
#
# We will first load the subjects connectivity dictionary, storing
# for each groups and subject, the connectivity matrices for
# different connectivity metric. We will also the corresponding
# data table.

# Fetch the path of the home directory
home_directory = str(Path.home())

# Load the dictionary containing the connectivity matrices
subjects_connectivity_matrices = load_object(
    full_path_to_object=os.path.join(home_directory, 'raw_subjects_connectivity_matrices.pkl'))

# load the data table
data_table = pd.read_excel(os.path.join(home_directory, 'data_table.xlsx'))
print(data_table.to_markdown())

# For convenience, we shift the index
# of the dataframe to the subjects
# identifiers column
data_table = data_table.set_index(['subjects'])

#%%
# This data table have a very common
# structure with a mix of categorical
# and continous variable. Let's barplot
# the score for the group of female and male
# in the patients population:

sns.barplot(x='Sex', y='score', data=data_table)
plt.show()

#############################################################################
# Selecting a subset of data
# --------------------------
#
# It's common to extract and compute the connectivity
# matrices on your whole cohort of data, and entering
# them in one or multiple statistical analysis. In
# practice, you may want only selecting a sub-set
# of your connectivity matrices. For example,
# you might want to select inside the
# **patients group**, the left lesioned subject
# and male only. For convenience, and avoiding
# a fastidious manual extraction inside the
# subjects connectivity matrices dictionary,
# we create a special function dedicated
# to this task. The main inputs are
# the connectivity dictionary of your
# population and the corresponding table.

# Select the male, and left lesioned
# patients.
# Select a subset of patients
# Compute the connectivity matrices dictionary with factor as keys.
group_by_factor_subjects_connectivity, population_df_by_factor, factor_keys, = \
    dictionary_operations.groupby_factor_connectivity_matrices(
        population_data_file=os.path.join(home_directory, 'data_table.xlsx'),
        sheetname='behavioral_data',
        subjects_connectivity_matrices_dictionnary=subjects_connectivity_matrices,
        groupes=['patients'], factors=['Lesion', 'Sex'])

#%%
# The :py:func:`groupby_factor_connectivity_matrices`
# output 3 objects: ``group_by_factor_subjects_connectivity`` is a dictionary
# with all possible combination of the ``factors`` list you've entered. Here,
# we entered *Lesion*, and *Sex*, two categorical variable with 2 levels
# each. So number of keys of the `groupby_factor_connectivity_matrices`
# dictionary should be 2x2, **4**: the right lesioned **AND** female patients,
# the right lesioned **AND** male patients, the left lesioned **AND** male patients,
# the left lesioned **AND** female patients. Let's print out the keys list
# to verify it:

print(list(group_by_factor_subjects_connectivity.keys()))

#%%
# The second output is another  dictionary, with the previous
# list as key, and the list of subjects for each sub-group.
# For example, let's print out the list of subjects in the
# male right lesioned group:

print(population_df_by_factor[('D', 'M')])

#%%
# The last output is simply the keys list
# of the new group:

print(factor_keys)

#%%
# Now, we can create a new dictionary of
# patients that contains only the sub-group
# we wanted: the left lesioned and male
# patients. It's easy, because you
# just computed it:

left_lesioned_male_matrices = dict()
left_lesioned_male_matrices['patients'] = group_by_factor_subjects_connectivity[('G', 'M')]