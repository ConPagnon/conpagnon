"""
Easy Resting State Analysis in Python
------------------------------------
Documentation is available online:
https://conpagnon.github.io/conpagnon/

Contents
--------
ConPagnon is python library designed to facilitate the analysis of Resting State data.
ConPagnon is mainly build around Nilearn and Scikit-learn. We provide not only useful
wrapper around commonly used statistical method, but new algorithm build around
state of the art metric, and machine learning methods.

Author
------
Dhaif BEKHA (dhaif@dhaifbekha.com)

Contributors
------------
Dhaif BEKHA (dhaif@dhaifbekha.com)

Submodules list
---------------
computing               --- Times series extractions, and connectivity
                            matrices computing for different metrics
connectivity_statistics --- Statistics toolbox containing a wide range
                            of statistical method adapted to functional
                            connectivity analysis.
data_handling           --- Set of tools for manipulating connectivity
                            matrices dictionary, and text table.
machine_learning        --- Set of tools containing machine learning
                            algorithm adapted to functional connectivity
                            analysis
plotting                --- Set of plotting function for connectivity
                            matrices, and glass-brain figures.
pylearn_mulm            --- The pylearn_mulm library, created
                            by Edouard Duchesnay.
utils                   --- Utilities for saving and loading object,
                            pre-processing, files and folder management.
"""


import sys
import warnings

from conpagnon.version import _check_module_dependencies, __version__


def _py35_deprecation_warning():
    py35_warning = ('Python 3.5 support is deprecated and will be removed in '
                    'a future release. Consider switching to Python 3.6 or 3.7'
                    )
    warnings.filterwarnings('once', message=py35_warning)
    warnings.warn(message=py35_warning,
                  category=FutureWarning,
                  stacklevel=3,
                  )


def _python_deprecation_warnings():
    if sys.version_info.major == 3 and sys.version_info.minor == 5:
        _py35_deprecation_warning()


_check_module_dependencies()
_python_deprecation_warnings()

# list all submodules available in ConPagnon
__all__ = ['computing', 'connectivity_statistics', 'data_handling',
           'machine_learning', 'pylearn_mulm',
           'sphinx', 'plotting', 'utils', '__version__']
