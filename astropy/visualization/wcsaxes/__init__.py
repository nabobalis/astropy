# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import config as _config
from astropy.utils.compat.optional_deps import HAS_MATPLOTLIB as _HAS_MATPLOTLIB

# The public API needs matplotlib. The private _layout module, which computes
# the tick geometry, needs only numpy and astropy, so the package itself has
# to be importable without matplotlib.
if _HAS_MATPLOTLIB:
    from .coordinate_helpers import CoordinateHelper
    from .coordinates_map import CoordinatesMap
    from .core import *
    from .helpers import *
    from .patches import *
    from .wcsapi import custom_ucd_coord_meta_mapping
else:

    def __getattr__(name):
        # Say what is missing, rather than "cannot import name".
        if name in {
            "CoordinateHelper",
            "CoordinatesMap",
            "Quadrangle",
            "SphericalCircle",
            "WCSAxes",
            "WCSAxesSubplot",
            "add_beam",
            "add_scalebar",
            "custom_ucd_coord_meta_mapping",
        }:
            raise ModuleNotFoundError(
                f"{__name__}.{name} requires matplotlib", name="matplotlib"
            )
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astropy.visualization.wcsaxes`.
    """

    coordinate_range_samples = _config.ConfigItem(
        50,
        "The number of samples along each image axis when determining "
        "the range of coordinates in a plot.",
    )

    frame_boundary_samples = _config.ConfigItem(
        1000,
        "How many points to sample along the axes when determining tick locations.",
    )

    grid_samples = _config.ConfigItem(
        1000, "How many points to sample along grid lines."
    )

    contour_grid_samples = _config.ConfigItem(
        200, "The grid size to use when drawing a grid using contours"
    )


conf = Conf()
