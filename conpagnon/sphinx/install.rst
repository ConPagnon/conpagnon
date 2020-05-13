

.. _install:

Installing ConPagnon
====================

Before the following installation steps, we recommend that you install the
`Anaconda <https://docs.continuum.io/anaconda/>`_ distribution. Anaconda
comes with it's own package manager, and numerous pre-installed distribution.
It's easy to manipulate, and we suggest Anaconda as you're default Python environment.

Python support
--------------

ConPagnon supports all Python version since the 3.5 Python release. Prior version may work,
but we do not recommend nor supports their utilisation here. Python 2.X is not supported
at all, and we do not recommend, if you can, to use that version at all in you're project.

Dependencies
------------

ConPagnon need other Python packages to work properly. To install them all at once,
you can run the following command in a command prompt:


.. code-block:: bash

    pip install numpy joblib matplotlib seaborn nibabel pandas patsy scipy statsmodels scipy webcolors PyPDF2
                webcolors scikit-learn nilearn psutil networkx tabulate xlrd

Installation from PyPi
----------------------

To install the latest version of ConPagnon, open a command prompt and type the following command:

.. code-block:: bash

    pip install conpagnon

If you can, you can download the sources
at the official `PyPi repository of ConPagnon <https://pypi.org/project/conpagnon/>`_

Installation from GitHub
------------------------

You can also download ConPagnon directly from the master branch of the repository on Github. Once you have Git
installed you can open a command prompt and type the following command:

.. code-block:: bash

    git clone https://github.com/ConPagnon/conpagnon.git

If you want to keep you're version ConPagnon up to date as much as possible, you have to pull the changes from
the master branch by entering in a command prompt the following command:

.. code-block:: bash

    git pull

Optional Dependencies
---------------------

* `jupyter <https://jupyter.org/>`__ is needed to run the examples from the .ipynb files.
* `IPython <https://ipython.org>`__ >= 5.0 is required to build the docs locally or to use the notebooks.
