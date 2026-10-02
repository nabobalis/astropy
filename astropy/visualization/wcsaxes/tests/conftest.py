# Licensed under a 3-clause BSD style license - see LICENSE.rst

from pathlib import Path

from astropy.utils.compat.optional_deps import HAS_MATPLOTLIB

# Keep the layout tests runnable without matplotlib. Ignore rendering tests
# during collection rather than raising pytest's Skipped when importing wcsaxes.
if not HAS_MATPLOTLIB:
    collect_ignore = [
        path.name
        for path in Path(__file__).parent.glob("test_*.py")
        if path.name != "test_layout.py"
    ]
