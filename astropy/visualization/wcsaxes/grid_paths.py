# Licensed under a 3-clause BSD style license - see LICENSE.rst


from matplotlib.lines import Path

# The tolerances moved to _layout with the code that uses them, and are
# imported here for code that imports them from this module. The grid lines
# use the values in _layout, so changing them here has no effect.
from ._layout import (  # noqa: F401
    DISCONT_FACTOR,
    ROUND_TRIP_RTOL,
    gridline_path_codes,
    lon_lat_path_codes,
)


def get_lon_lat_path(lon_lat, pixel, lon_lat_check):
    """
    Draw a curve, taking into account discontinuities.

    Parameters
    ----------
    lon_lat : ndarray
        The longitude and latitude values along the curve, given as a (n,2)
        array.
    pixel : ndarray
        The pixel coordinates corresponding to ``lon_lat``.
    lon_lat_check : ndarray
        The world coordinates derived from converting from ``pixel``, which is
        used to ensure round-tripping.
    """
    return Path(pixel, codes=lon_lat_path_codes(lon_lat, pixel, lon_lat_check))


def get_gridline_path(world, pixel):
    """
    Draw a grid line.

    Parameters
    ----------
    world : ndarray
        The longitude and latitude values along the curve, given as a (n,2)
        array.
    pixel : ndarray
        The pixel coordinates corresponding to ``lon_lat``.
    """
    return Path(pixel, codes=gridline_path_codes(world, pixel))
