# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.utils.compat.optional_deps import HAS_MATPLOTLIB

# The wcsaxes package imports without matplotlib, so that its _layout module
# can be used on its own, but these tests need matplotlib. They are left out
# of collection here, rather than skipped when this package is imported, so
# that importing the package, as pkgutil.walk_packages does, cannot raise
# pytest's Skipped.
if not HAS_MATPLOTLIB:
    collect_ignore_glob = ["test_*.py"]
