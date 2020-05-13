.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_01_Organizing_data_plot_organizing_data_example.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_01_Organizing_data_plot_organizing_data_example.py:


Basic data organization
=======================
**What you'll learn**: Organize you're dataset for ConPagnon processing, and learn how to retrieve it,
and store it efficiently.

**Author**: `Dhaif BEKHA <dhaif@dhaifbekha.com>`_

Retrieve the example dataset
----------------------------

For this example, we will work on a very simple dataset consisting in five
pre-processed resting state images. We will show you after how to organize
the data with confound to be regressed out if you have it.
Please, download the example `dataset <https://www.dropbox.com/sh/07r0i5kyfyweesx/AAA798Z2pWYO9FPd8gtny_E2a?dl=1>`_.

.. important::
    Once you've downloaded the example dataset, make sure to unzip it,
    and move it to you're home directory. If you are not in you're home
    directory, make sure to change all the path leading to data !
    For the rest of the tutorial, we will assume that you've moved
    the folder 'data' in you're home directory.

Data folder structure
-----------------------

In the data folder, you will find a folder called **group_1** containing
the resting state images, in the .nii.gz format. You will find  a second folder
called **text_data** containing a single .txt file, with only one column
representing the subject identifier or ids, for each subject.

Import all the module we will need
----------------------------------


.. code-block:: default

    from conpagnon.data_handling.data_architecture import fetch_data
    from conpagnon.utils.folders_and_files_management import check_directories_existence
    from pathlib import Path
    import os
    import pandas as pd








Setting paths to the data
-------------------------


.. code-block:: default


    # The first steps is to set the paths to all the relevant data paths.

    # Fetch the path of the home directory
    home_directory = str(Path.home())

    # The root fmri data directory containing all the fmri files directories
    # This is the 'data' folder in you're **home directory**
    root_fmri_data_directory = os.path.join(home_directory, 'data')

    # Groups to include in the study: this is
    # simply the name of the folder
    group = ['group_1']

    # We simply check that the root data directory,
    # and the group directory exist, if not exception
    # is raised we're good to go !
    check_directories_existence(root_directory=root_fmri_data_directory,
                                directories_list=group)








.. important::
   The **name** of the folder containing
   you're resting state images, is considered
   in ConPagnon to be the name of the group
   in the study. Often those groups are called
   'patients', 'controls', etc. In this example, it's
   simply 'group_1".


.. code-block:: default


    # Full path to the text file containing the subjects identifiers
    subjects_text_list = os.path.join(root_fmri_data_directory, 'text_data/subjects_list.txt')
    # We read this files with pandas
    subjects = list(pd.read_csv(subjects_text_list, header=None)[0])
    # Naturally, when we print we get:
    print(subjects)

    # Now, we can fetch the data, i.e the functional image
    # present in the group_1 directory.
    data_dictionary = fetch_data(subjects_id_data_path=subjects_text_list,
                                 root_fmri_data_directory=root_fmri_data_directory,
                                 groupes=group,
                                 individual_confounds_directory=None)

    # Let's print the result:
    print(data_dictionary)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ['subject_1', 'subject_2', 'subject_3', 'subject_4', 'subject_5']
    {'group_1': {'subject_2': {'functional_file': ['/home/dhaif/data/group_1/subject_2.nii.gz'], 'counfounds_file': []}, 'subject_1': {'functional_file': ['/home/dhaif/data/group_1/subject_1.nii.gz'], 'counfounds_file': []}, 'subject_5': {'functional_file': ['/home/dhaif/data/group_1/subject_5.nii.gz'], 'counfounds_file': []}, 'subject_4': {'functional_file': ['/home/dhaif/data/group_1/subject_4.nii.gz'], 'counfounds_file': []}, 'subject_3': {'functional_file': ['/home/dhaif/data/group_1/subject_3.nii.gz'], 'counfounds_file': []}}}




Once you fetch you're data with the :py:func:`conpagnon.data_handling.data_architecture.fetch_data` function, you
get a Python dictionary. The first level of keys, is simply the groups names, here
it's just 'group_1', since we only have one group. The value of this key, is another
dictionary, with the subject identifier name as keys, and as value, a dictionary with
two field: 'functional_file', indicating the full path to the subject image, and 'confound_file',
indicating the full path to the text file containing possible confound to regress. It's empty here,
because the images are already preprocessed.

With confounds files
--------------------

We previously assumed that you have already completely
preprocessed you're data. In some case, you may have not completed
this step. The steps are exactly the same, the only difference is that
you have to create a directory, at the root of the data folder.  In this
directory, you should put the confounds file in the .csv format. The filename
rules, is the same as the functional images: the filename of all confound files
should contain the subject identifier at least.
Once you've done that, you should put the full path of the confound directory
in the argument called **individual_confounds_directory** in the
:py:func:`conpagnon.data_handling.data_architecture.fetch_data` function.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.132 seconds)


.. _sphx_glr_download_examples_01_Organizing_data_plot_organizing_data_example.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_organizing_data_example.py <plot_organizing_data_example.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_organizing_data_example.ipynb <plot_organizing_data_example.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
