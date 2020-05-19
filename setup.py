from setuptools import find_packages, setup
import os
import sys

"""
Setup file for ConPagnon,
checks require dependencies
version, fetch version from
version.py

Author: Dhaif BEKHA (dhaif@dhaifbekha.com)
"""


def load_version():
    """Executes conpagnon/version.py in a globals dictionary and return it.
    """
    # load all vars into globals, otherwise
    #   the later function call using global vars doesn't work.
    globals_dict = {}
    with open(os.path.join('conpagnon', 'version.py')) as fp:
        exec(fp.read(), globals_dict)

    return globals_dict


def is_installing():
    # Allow command-lines such as "python setup.py build install"
    install_commands = {'install', 'develop'}
    return install_commands.intersection(set(sys.argv))


def list_required_packages():
    required_packages = []
    required_packages_orig = ['%s>=%s' % (mod, meta['min_version'])
                              for mod, meta
                              in _VERSION_GLOBALS['REQUIRED_MODULE_METADATA']
                              ]
    for package in required_packages_orig:
        if package.startswith('sklearn'):
            package = package.replace('sklearn', 'scikit-learn')
        required_packages.append(package)
    required_packages.append('sklearn')
    return required_packages


_VERSION_GLOBALS = load_version()
DISTNAME = 'conpagnon'
DESCRIPTION = 'Easy Resting State Analysis with Python'
with open('README.rst') as fp:
    LONG_DESCRIPTION = fp.read()
MAINTAINER = 'Dhaif BEKHA'
MAINTAINER_EMAIL = 'dhaif@dhaifbekha.com'
URL = 'https://conpagnon.github.io/conpagnon/'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/ConPagnon/conpagnon/archive/v2.0.10.tar.gz'
VERSION = _VERSION_GLOBALS['__version__']


if __name__ == "__main__":
    if is_installing():
        module_check_fn = _VERSION_GLOBALS['_check_module_dependencies']
        module_check_fn(is_conpagnon_installing=True)

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Programming Language :: Python :: 3.8',
          ],
          packages=find_packages(),
          install_requires=['numpy>==1.16.4',
                            'joblib',
                            'matplotlib',
                            'nibabel',
                            'pandas>==1.0.0',
                            'scipy>==1.4.1',
                            'statsmodels',
                            'seaborn',
                            'webcolors',
                            'PyPDF2',
                            'scikit-learn',
                            'nilearn',
                            'psutil',
                            'networkx',
                            'patsy',
                            'xlrd',
                            'tabulate'],
          python_requires='>=3.6',
          )
