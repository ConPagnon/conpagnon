"""
ConPagnon version, dependencies checking.
"""
# Author: Dhaif BEKHA (dhaif@dhaifbekha.com)
# Generic release markers:
# X.Y
# X.Y.Z # For bugfix releases
#
# Admissible pre-release markers:
# X.YaN # Alpha release
# X.YbN # Beta release
# X.YrcN # Release Candidate
# X.Y # Final release

__version__ = '2.0.9'

_CONPAGNON_INSTALL_MSG = 'See %s for installation information.' % (
    'https://conpagnon.github.io/conpagnon/install.html')

REQUIRED_MODULE_METADATA = (
    ('numpy', {
        'min_version': '1.11',
        'required_at_installation': True,
        'install_info': _CONPAGNON_INSTALL_MSG}),
    ('scipy', {
        'min_version': '0.19',
        'required_at_installation': True,
        'install_info': _CONPAGNON_INSTALL_MSG}),
    ('sklearn', {
        'min_version': '0.19',
        'required_at_installation': True,
        'install_info': _CONPAGNON_INSTALL_MSG}),
    ('joblib', {
        'min_version': '0.12',
        'required_at_installation': True,
        'install_info': _CONPAGNON_INSTALL_MSG}),
    ('nibabel', {
        'min_version': '2.0.2',
        'required_at_installation': False}),
    ('pandas', {
        'min_version': '0.18.0',
        'required_at_installation': True,
        'install_info': _CONPAGNON_INSTALL_MSG}),
    ('seaborn', {
        'min_version': '0.10.1',
        'required_at_installation': True,
        'install_info': _CONPAGNON_INSTALL_MSG}),
    ('statsmodels', {
        'min_version': '0.11.1',
        'required_at_installation': True,
        'install_info': _CONPAGNON_INSTALL_MSG}),
    ('xlrd', {
        'min_version': '1.2.0',
        'required_at_installation': True,
        'install_info': _CONPAGNON_INSTALL_MSG}),
)

OPTIONAL_MATPLOTLIB_MIN_VERSION = '2.0'


def _import_module_with_version_check(
        module_name,
        minimum_version,
        install_info=None):
    """Check that module is installed with a recent enough version
    """
    from distutils.version import LooseVersion

    try:
        module = __import__(module_name)
    except ImportError as exc:
        user_friendly_info = 'Module "{0}" could not be found. {1}'.format(
            module_name,
            install_info or 'Please install it properly to use ConPagnon.')
        exc.args += (user_friendly_info,)
        # Necessary for Python 3 because the repr/str of ImportError
        # objects was changed in Python 3
        if hasattr(exc, 'msg'):
            exc.msg += '. ' + user_friendly_info
        raise

    # Avoid choking on modules with no __version__ attribute
    module_version = getattr(module, '__version__', '0.0.0')

    version_too_old = (not LooseVersion(module_version) >=
                       LooseVersion(minimum_version))

    if version_too_old:
        message = (
            'A {module_name} version of at least {minimum_version} '
            'is required to use ConPagnon. {module_version} was found. '
            'Please upgrade {module_name}').format(
                module_name=module_name,
                minimum_version=minimum_version,
                module_version=module_version)

        raise ImportError(message)

    return module


def _check_module_dependencies(is_conpagnon_installing=False):
    """Throw an exception if nilearn dependencies are not installed.
    Parameters
    ----------
    is_conpagnon_installing: boolean
        if True, only error on missing packages that cannot be auto-installed.
        if False, error on any missing package.
    Throws
    -------
    ImportError
    """

    for (module_name, module_metadata) in REQUIRED_MODULE_METADATA:
        if not (is_conpagnon_installing and
                not module_metadata['required_at_installation']):
            # Skip check only when installing and it's a module that
            # will be auto-installed.
            _import_module_with_version_check(
                module_name=module_name,
                minimum_version=module_metadata['min_version'],
                install_info=module_metadata.get('install_info'))