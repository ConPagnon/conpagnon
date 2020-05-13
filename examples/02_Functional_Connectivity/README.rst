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