# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This sub-package makes use of image testing with the pytest-mpl package:
#
# https://pypi.org/project/pytest-mpl
#
# For more information on writing image tests, see the 'Image tests with
# pytest-mpl' section of the developer docs.

import pytest

# The wcsaxes package itself imports without matplotlib, so that its _layout
# module can be used on its own, but these tests need matplotlib.
pytest.importorskip("matplotlib")
