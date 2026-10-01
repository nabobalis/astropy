# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import config as _config
from astropy.utils.compat.optional_deps import HAS_MATPLOTLIB as _HAS_MATPLOTLIB

# The private _layout and _model modules, which compute the tick, grid and
# label geometry and read the coordinate metadata of a WCS, need only numpy
# and astropy, so the package itself has to be importable without matplotlib.
# Of the public API, only custom_ucd_coord_meta_mapping is.
from ._model import custom_ucd_coord_meta_mapping

if _HAS_MATPLOTLIB:
    from .coordinate_helpers import CoordinateHelper
    from .coordinates_map import CoordinatesMap
    from .core import *
    from .helpers import *
    from .patches import *
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
