# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
What a WCS says about the coordinates WCSAxes shows, without matplotlib.

This private module reads the coordinate metadata that WCSAxes derives from
an APE 14 WCS (the type, wrap, unit, format unit, name and label of each
world coordinate, and which of them are visible after slicing), and converts
pixel and world coordinates with such a WCS the way the WCSAxes transforms
do. It imports only numpy and astropy, so a toolkit other than matplotlib
can lay out the same coordinates with `~astropy.visualization.wcsaxes._layout`.
"""

from contextlib import contextmanager

import numpy as np

from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS

__all__ = [
    "apply_slices",
    "coord_meta_from_wcs",
    "custom_ucd_coord_meta_mapping",
    "wcs_pixel_to_world",
    "wcs_world_to_pixel",
]

UCD_COORD_META_MAPPING = {
    "lon": {"coord_type": "longitude"},
    "lat": {"coord_type": "latitude"},
    "ra": {"coord_type": "longitude"},
    "dec": {"coord_type": "latitude"},
    "alt": {"coord_type": "longitude"},
    "az": {"coord_type": "latitude"},
    "long": {"coord_type": "longitude"},
}

CUSTOM_UCD_COORD_META_MAPPING = {
    "pos.helioprojective.lon": {
        "coord_wrap": 180.0 * u.deg,
        "format_unit": u.arcsec,
        "coord_type": "longitude",
    },
    "pos.helioprojective.lat": {"format_unit": u.arcsec, "coord_type": "latitude"},
    "pos.heliographic.stonyhurst.lon": {
        "coord_wrap": 180.0 * u.deg,
        "format_unit": u.deg,
        "coord_type": "longitude",
    },
    "pos.heliographic.stonyhurst.lat": {"format_unit": u.deg, "coord_type": "latitude"},
    "pos.heliographic.carrington.lon": {
        "coord_wrap": 360.0 * u.deg,
        "format_unit": u.deg,
        "coord_type": "longitude",
    },
    "pos.heliographic.carrington.lat": {"format_unit": u.deg, "coord_type": "latitude"},
}


@contextmanager
def custom_ucd_coord_meta_mapping(mapping, *, overwrite=False):
    """
    A context manager that makes it possible to temporarily add new UCD+ to WCS coordinate
    plot metadata mappings.

    Parameters
    ----------
    mapping : dict
        A dictionary mapping a UCD to coordinate plot metadata.
        Note that custom UCD names have their "custom:" prefix stripped.
    overwrite : bool
        If `True` overwrite existing entries with ``mapping``.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from astropy.visualization.wcsaxes.wcsapi import custom_ucd_coord_meta_mapping
    >>> from astropy.wcs.wcsapi.fitswcs import custom_ctype_to_ucd_mapping
    >>> wcs = WCS(naxis=1)
    >>> wcs.wcs.ctype = ["eggs"]
    >>> wcs.wcs.cunit = ["deg"]
    >>> custom_mapping = {"eggs": "custom:pos.eggs"}
    >>> with custom_ctype_to_ucd_mapping(custom_mapping):
    ...     custom_meta = {
    ...         "pos.eggs": {
    ...             "coord_wrap": 360.0 * u.deg,
    ...             "format_unit": u.arcsec,
    ...             "coord_type": "longitude",
    ...         }
    ...     }
    ...     with custom_ucd_coord_meta_mapping(custom_meta):
    ...        fig = plt.figure()
    ...        ax = fig.add_subplot(111, projection=wcs)
    ...        ax.coords
    <CoordinatesMap with 1 world coordinates:
    <BLANKLINE>
      index       aliases           type   unit  wrap format_unit visible
                                                 deg
      ----- -------------------- --------- ---- ----- ----------- -------
          0 custom:pos.eggs eggs longitude  deg 360.0      arcsec     yes
    <BLANKLINE>
    >
    """
    normalized = {}
    for k, v in mapping.items():
        k = k.removeprefix("custom:")
        if k in normalized:
            raise ValueError(f"UCD metadata mapping {k} specified more than once.")
        normalized[k] = v
    mapping = normalized

    if not overwrite:
        for k in mapping:
            if k in CUSTOM_UCD_COORD_META_MAPPING:
                raise ValueError(f"UCD metadata mapping {k} already exists.")

    added_keys = []
    overwritten = {}
    for k, v in mapping.items():
        if k in CUSTOM_UCD_COORD_META_MAPPING:
            overwritten[k] = CUSTOM_UCD_COORD_META_MAPPING[k]
        else:
            added_keys.append(k)
        CUSTOM_UCD_COORD_META_MAPPING[k] = v

    try:
        yield
    finally:
        for k in added_keys:
            del CUSTOM_UCD_COORD_META_MAPPING[k]
        CUSTOM_UCD_COORD_META_MAPPING.update(overwritten)


def coord_meta_from_wcs(wcs, slices=None):
    """
    Read the coordinate metadata that WCSAxes derives from a WCS.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS.
    slices : tuple, optional
        For a WCS with more than two pixel dimensions, one element per pixel
        dimension: ``'x'``, ``'y'``, an integer or a slice.

    Returns
    -------
    coord_meta : dict
        The ``name``, ``type``, ``wrap``, ``unit``, ``format_unit``,
        ``visible`` and ``default_axis_label`` of each world coordinate, as
        `~astropy.visualization.wcsaxes.CoordinatesMap` reads them.
    transform_wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS after slicing, to convert pixel and world coordinates with.
    invert_xy : bool
        Whether ``slices`` swaps the two pixel coordinates.
    """
    if slices is not None:
        slices = tuple(slices)

    if wcs.pixel_n_dim > 2:
        if slices is None:
            raise ValueError(
                "WCS has more than 2 pixel dimensions, so 'slices' should be set"
            )
        elif len(slices) != wcs.pixel_n_dim:
            raise ValueError(
                "'slices' should have as many elements as WCS "
                f"has pixel dimensions (should be {wcs.pixel_n_dim})"
            )

    is_fits_wcs = isinstance(wcs, WCS) or (
        isinstance(wcs, SlicedLowLevelWCS) and isinstance(wcs._wcs, WCS)
    )

    coord_meta = {}
    coord_meta["name"] = []
    coord_meta["type"] = []
    coord_meta["wrap"] = []
    coord_meta["unit"] = []
    coord_meta["visible"] = []
    coord_meta["format_unit"] = []

    for idx in range(wcs.world_n_dim):
        axis_type = wcs.world_axis_physical_types[idx]
        axis_unit = u.Unit(wcs.world_axis_units[idx])
        coord_wrap = None
        format_unit = axis_unit
        coord_type = "scalar"

        dim_meta = {
            "coord_type": coord_type,
            "coord_wrap": coord_wrap,
            "format_unit": format_unit,
            "axis_unit": axis_unit,
        }

        if axis_type is not None:
            axis_type_split = axis_type.split(".")
            axis_type_split[0] = axis_type_split[0].replace("custom:", "")

            for ucd, meta in CUSTOM_UCD_COORD_META_MAPPING.items():
                if ucd in axis_type:
                    dim_meta.update(meta)
                    break
            else:
                for ucd, meta in UCD_COORD_META_MAPPING.items():
                    if ucd == axis_type_split[-1]:
                        dim_meta.update(meta)
                        # We only do the following if the original unit was
                        # degrees. If the unit was e.g. arcsec, it seems
                        # reasonable to stick to the WCS unit.
                        if ucd == "ra" and axis_unit == u.deg:
                            dim_meta["format_unit"] = u.hourangle
                        break

        coord_meta["type"].append(dim_meta["coord_type"])
        coord_meta["wrap"].append(dim_meta["coord_wrap"])
        coord_meta["format_unit"].append(dim_meta["format_unit"])
        coord_meta["unit"].append(dim_meta["axis_unit"])

        # For FITS-WCS, for backward-compatibility, we need to make sure that we
        # provide aliases based on CTYPE for the name.
        if is_fits_wcs:
            name = []
            if isinstance(wcs, WCS):
                name.append(wcs.wcs.ctype[idx].lower())
                name.append(wcs.wcs.ctype[idx][:4].replace("-", "").lower())
            elif isinstance(wcs, SlicedLowLevelWCS):
                name.append(wcs._wcs.wcs.ctype[wcs._world_keep[idx]].lower())
                name.append(
                    wcs._wcs.wcs.ctype[wcs._world_keep[idx]][:4]
                    .replace("-", "")
                    .lower()
                )
            if name[0] == name[1]:
                name = name[0:1]
            if axis_type:
                if axis_type not in name:
                    name.insert(0, axis_type)
            if wcs.world_axis_names and wcs.world_axis_names[idx]:
                if wcs.world_axis_names[idx] not in name:
                    name.append(wcs.world_axis_names[idx])
            name = tuple(name) if len(name) > 1 else name[0]
        else:
            name = axis_type or ""
            if wcs.world_axis_names:
                name = (
                    (name, wcs.world_axis_names[idx])
                    if wcs.world_axis_names[idx]
                    else name
                )

        coord_meta["name"].append(name)

    # If the world axis has a name use it, else display the world axis physical type.
    fallback_labels = [
        name[0] if isinstance(name, (list, tuple)) else name
        for name in coord_meta["name"]
    ]
    coord_meta["default_axis_label"] = [
        wcs.world_axis_names[i] or fallback_label
        for i, fallback_label in enumerate(fallback_labels)
    ]

    transform_wcs, invert_xy, world_map = apply_slices(wcs, slices)

    for i in range(len(coord_meta["type"])):
        coord_meta["visible"].append(i in world_map)

    return coord_meta, transform_wcs, invert_xy


def apply_slices(wcs, slices):
    """
    Take the input WCS and slices and return a sliced WCS for the transform and
    a mapping of world axes in the sliced WCS to the input WCS.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS to slice.
    slices : tuple or `None`
        A tuple with one element for each pixel dimension of the WCS, where
        the elements are either ``'x'``, ``'y'``, an integer, or a slice.
    """
    if isinstance(wcs, SlicedLowLevelWCS):
        world_keep = list(wcs._world_keep)
    else:
        world_keep = list(range(wcs.world_n_dim))

    # world_map is the index of the world axis in the input WCS for a given
    # axis in the transform_wcs
    world_map = list(range(wcs.world_n_dim))
    transform_wcs = wcs
    invert_xy = False
    if slices is not None:
        wcs_slice = list(slices)
        wcs_slice[wcs_slice.index("x")] = slice(None)
        if "y" in slices:
            wcs_slice[wcs_slice.index("y")] = slice(None)
            invert_xy = slices.index("x") > slices.index("y")

        transform_wcs = SlicedLowLevelWCS(wcs, wcs_slice[::-1])
        world_map = tuple(world_keep.index(i) for i in transform_wcs._world_keep)

    return transform_wcs, invert_xy, world_map


def wcs_pixel_to_world(wcs, pixel, invert_xy=False):
    """
    Convert pixel to world coordinates with a WCS, as ``WCSPixel2WorldTransform`` does.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS, with at most two pixel dimensions.
    pixel : (N, M) array
        The pixel coordinates, one row per point.
    invert_xy : bool, optional
        Whether the two pixel coordinates are swapped.

    Returns
    -------
    ndarray
        The world coordinates, one row per point.
    """
    # Convert to a list of arrays
    pixel = list(pixel.T)

    if len(pixel) != wcs.pixel_n_dim:
        raise ValueError(
            f"Expected {wcs.pixel_n_dim} pixel coordinates, got {len(pixel)}"
        )

    if invert_xy:
        pixel = pixel[::-1]

    if len(pixel[0]) == 0:
        if wcs.world_n_dim == 1:
            world = np.array([])
        else:
            world = [np.array([])] * wcs.world_n_dim
    else:
        world = wcs.pixel_to_world_values(*pixel)

    if wcs.world_n_dim == 1:
        world = [world]

    return np.array(world).T


def wcs_world_to_pixel(wcs, world, invert_xy=False):
    """
    Convert world to pixel coordinates with a WCS, as ``WCSWorld2PixelTransform`` does.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS, with at most two pixel dimensions.
    world : (N, M) array
        The world coordinates, one row per point.
    invert_xy : bool, optional
        Whether the two pixel coordinates are swapped.

    Returns
    -------
    ndarray
        The pixel coordinates, one row per point.
    """
    # Convert to a list of arrays
    world = list(world.T)

    if len(world) != 2:
        raise ValueError(f"Expected 2 world coordinates, got {len(world)}")

    if wcs.world_n_dim == 1:
        world_non_wcs = world[1]
        world = world[0:1]

    if len(world[0]) == 0:
        if wcs.pixel_n_dim == 1:
            pixel = np.array([])
        else:
            pixel = [np.array([])] * wcs.pixel_n_dim
    else:
        pixel = wcs.world_to_pixel_values(*world)

    if invert_xy:
        pixel = pixel[::-1]

    if wcs.world_n_dim == 1:
        pixel = [pixel, world_non_wcs]

    return np.array(pixel).T
