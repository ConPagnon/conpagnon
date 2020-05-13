# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config
# Created by Dhaif BEKHA (dhaif@dhaifbekha.com)

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
import sphinx_material

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc',
              # numpydoc or sphinx.ext.napoleon, but not both
              'sphinx.ext.napoleon',
             # 'numpydoc',
              'sphinx.ext.doctest',
              'sphinx.ext.extlinks',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.autosectionlabel',
              # One of mathjax or imgmath
              'nbsphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              # 'sphinx.ext.autosummary',
              'sphinx.ext.inheritance_diagram',
              'matplotlib.sphinxext.plot_directive',
              'IPython.sphinxext.ipython_console_highlighting',
              'IPython.sphinxext.ipython_directive',
              'sphinx_gallery.gen_gallery'
              ]

try:
    import sphinxcontrib.spelling  # noqa: F401
except ImportError as err:  # noqa: F841
    pass
else:
    extensions.append('sphinxcontrib.spelling')

# nbsphinx options
nbsphinx_allow_errors = True
# sphinxcontrib-spelling options
spelling_word_list_filename = ['spelling_wordlist.txt', 'names_wordlist.txt']
spelling_ignore_pypi_package_names = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'ConPagnon 2.0'
version = u'2.0'
copyright = u'2016 - 2020. Created by Dhaif BEKHA.'
autosummary_generate = True
autoclass_content = 'class'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '**.ipynb_checkpoints', '*/autosummary/*.rst',
                    'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# Options for HTML output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
extensions.append('sphinx_material')
html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()
html_theme = 'sphinx_material'
html_title = project
html_short_title = project

# material theme options (see theme.conf for more information)
base_url = 'https://conpagnon.github.io/conpagnon/'
html_theme_options = {
    'base_url': base_url,
    'repo_url': 'https://github.com/ConPagnon/conpagnon',
    'repo_name': 'ConPagnon',
    'globaltoc_depth': 3,
    'globaltoc_collapse': True,
    'globaltoc_includehidden': True,
    'color_primary': 'red',
    'color_accent': 'blue',
    'nav_title': 'ConPagnon {0}'.format(version),
    'master_doc': False,
    'nav_links': [],
    'heroes': {'index': 'Easy resting state analysis in Python ',
               'examples/index': 'A series of examples to get you started with ConPagnon'},
    "version_dropdown": True,
    "version_json": "_static/versions.json",
}

language = 'en'
html_last_updated_fmt = ''

# ConPagnon Logo
html_logo = 'images/logo/logo_conpagnon_small_header.png'

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

# If false, no module index is generated.
html_domain_indices = True

# Create xrefs
numpydoc_use_autodoc_signature = True
numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False

# Configuration of Sphinx Gallery paths: those
# paths are relative to the sphinx folder.
sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': './examples',  # path to where to save gallery generated output
}
