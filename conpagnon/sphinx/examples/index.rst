:orphan:



.. _sphx_glr_examples:

ConPagnon usage examples
========================

.. warning::

    If you want to run the example if the Jupyter
    Notebook format (.ipynb file), make sure that
    you have Jupyter Notebook installed in you're
    Python environment.


.. contents:: **Contents**
    :local:
    :depth: 1

.. _tutorial_examples:

Tutorial examples
------------------

The following examples will teach you on how
to use ConPagnon. Do not hesitate to copy and paste
those examples, to understand the core logic of ConPagnon.

| **Author**: `Dhaif BEKHA <dhaif@dhaifbekha.com>`_

.. raw:: html

    <div class="sphx-glr-clear"></div>



.. _sphx_glr_examples_01_Organizing_data:

Organization of data for ConPagnon
----------------------------------

For easier data processing along you're analysis with ConPagnon, your
data need to be organized in a specific manner. In this section, we will
go through simple examples to get you started.



.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Basic data organization">

.. only:: html

 .. figure:: /examples/01_Organizing_data/images/thumb/sphx_glr_plot_organizing_data_thumb.png

     :ref:`sphx_glr_examples_01_Organizing_data_plot_organizing_data.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/01_Organizing_data/plot_organizing_data

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Manipulating a brain atlas">

.. only:: html

 .. figure:: /examples/01_Organizing_data/images/thumb/sphx_glr_plot_import_atlas_example_thumb.png

     :ref:`sphx_glr_examples_01_Organizing_data_plot_import_atlas_example.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/01_Organizing_data/plot_import_atlas_example
.. raw:: html

    <div class="sphx-glr-clear"></div>



.. _sphx_glr_examples_02_Functional_Connectivity:

Functional Connectivity: Computing
----------------------------------

In this section of the tutorials, we assume that your data are nicely
organised according to the first section of the tutorials, and you've
fetched the data with the :py:func:`conpagnon.data_handling.data_architecture.fetch_data`
function. Next steps involve the extraction of clean brain signals and the computing of the
connectivity matrices.

.. important:: ConPagnon extract the brain signals in each region from
                a brain atlas only. If you don't have one, you can fetch a
                atlas among the numerous examples in the `Nilearn dataset
                module <https://nilearn.github.io/modules/reference.html#module-nilearn.datasets>`_ !


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Extracting brain signals">

.. only:: html

 .. figure:: /examples/02_Functional_Connectivity/images/thumb/sphx_glr_plot_times_series_extraction_thumb.png

     :ref:`sphx_glr_examples_02_Functional_Connectivity_plot_times_series_extraction.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/02_Functional_Connectivity/plot_times_series_extraction

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build the connectivity matrices">

.. only:: html

 .. figure:: /examples/02_Functional_Connectivity/images/thumb/sphx_glr_plot_connectivity_matrices_thumb.png

     :ref:`sphx_glr_examples_02_Functional_Connectivity_plot_connectivity_matrices.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/02_Functional_Connectivity/plot_connectivity_matrices
.. raw:: html

    <div class="sphx-glr-clear"></div>



.. _sphx_glr_examples_03_Basic_Statistical_Analyses:

Basic Statistical Analyses
--------------------------

In this section, we will cover simple examples of
statistical analyses you might encounter in classic
resting state analysis. Those tests are simple and quick
to run, and it might help you to have a better understanding
of your data, before running into more advanced statistical analysis
in the next section.


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two groups classification">

.. only:: html

 .. figure:: /examples/03_Basic_Statistical_Analyses/images/thumb/sphx_glr_plot_two_groups_classification_thumb.png

     :ref:`sphx_glr_examples_03_Basic_Statistical_Analyses_plot_two_groups_classification.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/03_Basic_Statistical_Analyses/plot_two_groups_classification

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Regression analysis">

.. only:: html

 .. figure:: /examples/03_Basic_Statistical_Analyses/images/thumb/sphx_glr_plot_linear_regression_thumb.png

     :ref:`sphx_glr_examples_03_Basic_Statistical_Analyses_plot_linear_regression.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/03_Basic_Statistical_Analyses/plot_linear_regression

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two samples t-test">

.. only:: html

 .. figure:: /examples/03_Basic_Statistical_Analyses/images/thumb/sphx_glr_plot_t_test_analysis_thumb.png

     :ref:`sphx_glr_examples_03_Basic_Statistical_Analyses_plot_t_test_analysis.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/03_Basic_Statistical_Analyses/plot_t_test_analysis
.. raw:: html

    <div class="sphx-glr-clear"></div>



.. _sphx_glr_examples_04_Advanced_Statistical_Analyses:

Advanced Statistical Analyses
-----------------------------

In this section, we will cover more advanced type
of analyses on resting state functional data. Those
advanced method often came from different machine learning
tactics. Machine learning on resting state data is an area
in continuous expansion, and may allow you to explore your data
in a new way. We also cover some analysis developed to exploit
specifically the ``tangent`` space connectivity metric.


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Statistics in the tangent space">

.. only:: html

 .. figure:: /examples/04_Advanced_Statistical_Analyses/images/thumb/sphx_glr_plot_individual_statistics_in_the_tangent_space_thumb.png

     :ref:`sphx_glr_examples_04_Advanced_Statistical_Analyses_plot_individual_statistics_in_the_tangent_space.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/04_Advanced_Statistical_Analyses/plot_individual_statistics_in_the_tangent_space

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Discriminative connection identification">

.. only:: html

 .. figure:: /examples/04_Advanced_Statistical_Analyses/images/thumb/sphx_glr_plot_discriminative_connection_identification_thumb.png

     :ref:`sphx_glr_examples_04_Advanced_Statistical_Analyses_plot_discriminative_connection_identification.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/04_Advanced_Statistical_Analyses/plot_discriminative_connection_identification

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Connectome Predictive Modelling">

.. only:: html

 .. figure:: /examples/04_Advanced_Statistical_Analyses/images/thumb/sphx_glr_plot_connectome_predictive_modelling_thumb.png

     :ref:`sphx_glr_examples_04_Advanced_Statistical_Analyses_plot_connectome_predictive_modelling.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/04_Advanced_Statistical_Analyses/plot_connectome_predictive_modelling
.. raw:: html

    <div class="sphx-glr-clear"></div>



.. _sphx_glr_examples_05_utilities:

Utilities
---------

In this section, we present some tips and tricks
that will make your life easier while manipulating
data, as well as some other useful function for
specific cases.


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Manipulation of the data dictionary">

.. only:: html

 .. figure:: /examples/05_utilities/images/thumb/sphx_glr_plot_selection_of_data_thumb.png

     :ref:`sphx_glr_examples_05_utilities_plot_selection_of_data.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/05_utilities/plot_selection_of_data
.. raw:: html

    <div class="sphx-glr-clear"></div>



.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-gallery


  .. container:: sphx-glr-download

    :download:`Download all examples in Python source code: examples_python.zip <//media/dhaif/Samsung_T5/Work/Programs/ConPagnon/conpagnon/sphinx/./examples/examples_python.zip>`



  .. container:: sphx-glr-download

    :download:`Download all examples in Jupyter notebooks: examples_jupyter.zip <//media/dhaif/Samsung_T5/Work/Programs/ConPagnon/conpagnon/sphinx/./examples/examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
