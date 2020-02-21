from setuptools import find_packages, setup


DISTNAME = 'conpagnon'
DESCRIPTION = 'Easy Resting State Analysis with Python'
with open('README.rst') as fp:
    LONG_DESCRIPTION = fp.read()
MAINTAINER = 'Dhaif BEKHA'
MAINTAINER_EMAIL = 'dhaif@dhaifbekha.com'
URL = 'https://conpagnon.github.io/conpagnon/'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/ConPagnon/conpagnon/archive/v2.0.3.tar.gz'
VERSION = '2.0.3'


if __name__ == "__main__":
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
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
          ],
          packages=find_packages(),
          install_requires=['numpy',
                            'joblib',
                            'matplotlib',
                            'nibabel',
                            'pandas',
                            'scipy',
                            'statsmodels',
                            'seaborn',
                            'webcolors',
                            'PyPDF2',
                            'scikit-learn',
                            'nilearn',
                            'psutil', 'networkx', 'patsy'],
          python_requires='>=3.5',
          )
