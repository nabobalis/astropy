# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The model of a WCSAxes that does not depend on matplotlib.

This private module holds what WCSAxes knows about the coordinates it shows,
apart from how they are drawn: the metadata derived from an APE 14 WCS (the
type, wrap, unit, format unit, name and label of each world coordinate, and
which of them are visible after slicing), the settings that choose and
format the ticks of each coordinate, and the pixel and world conversions of
such a WCS as the WCSAxes transforms do them. It imports only numpy and
astropy, so a toolkit other than matplotlib can lay out the same coordinates
with `~astropy.visualization.wcsaxes._layout`.
"""

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from astropy import units as u
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from ._auto import auto_assign_coord_positions
from ._layout import (
    CoordSpec,
    anchor_tick_labels,
    axis_label_position,
    elliptical_spines,
    grid_lines,
    keep_tick_labels,
    label_store,
    place_ticks,
    rectangular_1d_spines,
    rectangular_spines,
    simplify_labels,
    sort_labels,
    spine_midpoint,
    spine_normal_angle,
    wrap_angle_at,
)
from .coordinate_range import find_coordinate_range
from .formatter_locator import AngleFormatterLocator, ScalarFormatterLocator, _rcparam

__all__ = [
    "FRAMES",
    "AxesModel",
    "CoordinateLayout",
    "CoordinateModel",
    "Layout",
    "apply_slices",
    "coord_meta_from_wcs",
    "custom_ucd_coord_meta_mapping",
    "default_positions",
    "wcs_pixel_to_world",
    "wcs_world_to_pixel",
]

# The frames of WCSAxes: spine names, the function that samples the spines
# from the axes limits, and the order in which spines are assigned to
# coordinates whose position is automatic.
FRAMES = {
    "rectangular": ("brtl", rectangular_spines, "bltr"),
    "rectangular1d": ("bt", rectangular_1d_spines, "bt"),
    "elliptical": ("chv", elliptical_spines, "chv"),
}

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


def default_positions(coord_meta, frame, spine_names):
    """
    Add the default tick, tick label and axis label positions to coord_meta.

    Parameters
    ----------
    coord_meta : dict
        The coordinate metadata, from `coord_meta_from_wcs`. The three
        ``default_*_position`` lists are added to it.
    frame : str
        A key of `FRAMES`, or any other string for a custom frame.
    spine_names : str
        The names of the spines of the frame.
    """
    world_map = [i for i, visible in enumerate(coord_meta["visible"]) if visible]
    n_world = len(coord_meta["type"])
    coord_meta["default_axislabel_position"] = [""] * n_world
    coord_meta["default_ticklabel_position"] = [""] * n_world
    coord_meta["default_ticks_position"] = [""] * n_world

    if frame in ("rectangular", "rectangular1d"):
        for index in world_map:
            coord_meta["default_axislabel_position"][index] = "#"
            coord_meta["default_ticklabel_position"][index] = "#"
            coord_meta["default_ticks_position"][index] = "#"

        # In the special and common case where the frame is rectangular and we
        # are dealing with a 2-d WCS (after slicing) for RectangularFrame or a
        # 1-d WCS for RectangularFrame1D, we show all ticks on all axes.
        if (frame == "rectangular" and len(world_map) == 2) or (
            frame == "rectangular1d" and len(world_map) == 1
        ):
            for index in world_map:
                coord_meta["default_ticks_position"][index] = spine_names

    elif frame == "elliptical":
        if "longitude" in coord_meta["type"]:
            lon_idx = coord_meta["type"].index("longitude")
            coord_meta["default_axislabel_position"][lon_idx] = "h"
            coord_meta["default_ticklabel_position"][lon_idx] = "h"
            coord_meta["default_ticks_position"][lon_idx] = "h"

        if "latitude" in coord_meta["type"]:
            lat_idx = coord_meta["type"].index("latitude")
            coord_meta["default_axislabel_position"][lat_idx] = "c"
            coord_meta["default_ticklabel_position"][lat_idx] = "c"
            coord_meta["default_ticks_position"][lat_idx] = "c"

    else:
        for index in range(len(coord_meta["type"])):
            if index in world_map:
                coord_meta["default_axislabel_position"][index] = spine_names
                coord_meta["default_ticklabel_position"][index] = spine_names
                coord_meta["default_ticks_position"][index] = spine_names

    return coord_meta


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


class CoordinateModel:
    """
    The settings of one world coordinate that choose and format its ticks.

    `~astropy.visualization.wcsaxes.CoordinateHelper` holds one and keeps
    these settings in it; a toolkit other than matplotlib uses one on its
    own. Style, positions and visibility stay with the toolkit.

    Parameters
    ----------
    coord_index : int or None
        The column of this coordinate in the world values, or None if the
        coordinate is not shown.
    coord_type : {'longitude', 'latitude', 'scalar'}
        The type of the coordinate.
    coord_unit : `~astropy.units.Unit`
        The unit of the world values.
    coord_wrap : `~astropy.units.Quantity`, optional
        The angle at which a longitude wraps, 360 degrees by default.
    format_unit : `~astropy.units.Unit`, optional
        The unit of the tick labels, ``coord_unit`` by default.
    default_label : str, optional
        The axis label to use if none is set.
    spine_names : str, optional
        The names of the spines of the frame, which positions are checked
        against.

    Attributes
    ----------
    ticks_visible, ticklabels_visible, axislabel_visible : bool
        Whether to show the ticks, the tick labels and the axis label.
    exclude_overlapping : bool
        Whether to drop a tick label that overlaps one drawn before.
    simplify : bool
        Whether to drop the part of a tick label that repeats the one before.
    tick_out : bool
        Whether the ticks point out of the frame, which moves the labels out
        by the tick size.
    grid : bool
        Whether `AxesModel.layout` samples the grid lines.
    axislabel : str or None
        The axis label, or None for the default one.
    auto_axislabel : bool
        Whether to use the default axis label when none is set.
    axislabel_minpad : float or dict
        The gap between the axis label and the spine or tick labels, in
        units of the font size, for all spines or per spine.
    axislabel_visibility_rule : {'always', 'ticks', 'labels'}
        When to show the axis label on a spine: always, only if the
        coordinate has ticks there, or only if it has tick labels anywhere.
    """

    def __init__(
        self,
        coord_index=None,
        coord_type="scalar",
        coord_unit=None,
        coord_wrap=None,
        format_unit=None,
        default_label=None,
        spine_names="brtl",
    ):
        self.coord_index = coord_index
        self.coord_unit = coord_unit
        self.format_unit = format_unit
        self.default_label = default_label or ""
        self.spine_names = tuple(spine_names)
        self.custom_formatter = None
        self.minor_frequency = 5
        self.display_minor_ticks = False
        # The spacing of the major ticks of the last layout, for format_coord
        self.spacing = None
        self._placed = None
        self.set_coord_type(coord_type, coord_wrap)

        self.ticks_position = "all"
        self.ticklabel_position = "all"
        self.axislabel_position = "all"
        self.ticks_visible = True
        self.ticklabels_visible = True
        self.axislabel_visible = True
        self.exclude_overlapping = False
        self.simplify = True
        self.tick_out = True
        self.grid = False
        self.axislabel = None
        self.auto_axislabel = True
        self.axislabel_minpad = 1
        self.axislabel_visibility_rule = "labels"

    def _validate_positions(self, positions):
        # As BaseFrame._validate_positions
        if positions == "all":
            return positions

        valid_positions = []
        invalid_positions = []
        for position in positions:
            if position in self.spine_names or position == "#":
                valid_positions.append(position)
            else:
                invalid_positions.append(position)

        if invalid_positions:
            warnings.warn(
                f"Ignoring unrecognized position(s): {invalid_positions}, should "
                f"be one of {'/'.join(self.spine_names)}. In future this will "
                "raise an error.",
                AstropyDeprecationWarning,
            )

        return valid_positions

    def _positions(self, positions):
        if positions == "all":
            return list(self.spine_names)
        return list(positions)

    def set_ticks_position(self, position):
        """
        Set the spines on which ticks appear, with ``'#'`` for automatic.
        """
        self.ticks_position = self._validate_positions(position)

    def get_ticks_position(self):
        return self._positions(self.ticks_position)

    def set_ticklabel_position(self, position):
        """
        Set the spines on which tick labels appear, with ``'#'`` for automatic.
        """
        self.ticklabel_position = self._validate_positions(position)

    def get_ticklabel_position(self):
        return self._positions(self.ticklabel_position)

    def set_axislabel_position(self, position):
        """
        Set the spines on which the axis label appears, with ``'#'`` for automatic.
        """
        self.axislabel_position = self._validate_positions(position)

    def get_axislabel_position(self):
        return self._positions(self.axislabel_position)

    def get_ticks_visible(self):
        return self.ticks_visible

    def get_ticklabel_visible(self):
        return self.ticklabels_visible

    def tick_count(self, spine):
        """
        The number of major ticks placed on a spine by the last layout.
        """
        if self._placed is None:
            return 0
        return self._placed.major.axis.count(spine)

    def get_axislabel(self, unit_format="latex"):
        """
        The axis label: the one set, or the default one if that is allowed.
        """
        if self.auto_axislabel and self.axislabel is None:
            return self.default_axislabel(unit_format)
        return self.axislabel or ""

    def _minpad(self, axis):
        try:
            return self.axislabel_minpad[axis]
        except TypeError:
            return self.axislabel_minpad

    def set_coord_type(self, coord_type, coord_wrap=None):
        """
        Set the coordinate type for the axis.

        Parameters
        ----------
        coord_type : str
            One of 'longitude', 'latitude' or 'scalar'.
        coord_wrap : `~astropy.units.Quantity`, optional
            The value to wrap at for angular coordinates.
        """
        self.coord_type = coord_type

        if coord_wrap is not None and not isinstance(coord_wrap, u.Quantity):
            warnings.warn(
                "Passing 'coord_wrap' as a number is deprecated. Use a Quantity with units convertible to angular degrees instead.",
                AstropyDeprecationWarning,
            )
            coord_wrap = coord_wrap * u.deg

        if coord_type == "longitude" and coord_wrap is None:
            self.coord_wrap = 360 * u.deg
        elif coord_type != "longitude" and coord_wrap is not None:
            raise NotImplementedError(
                "coord_wrap is not yet supported for non-longitude coordinates"
            )
        else:
            self.coord_wrap = coord_wrap

        # Initialize tick formatter/locator
        if coord_type == "scalar":
            self.coord_scale_to_deg = None
            self.formatter_locator = ScalarFormatterLocator(unit=self.coord_unit)
        elif coord_type in ["longitude", "latitude"]:
            if self.coord_unit is u.deg:
                self.coord_scale_to_deg = None
            else:
                self.coord_scale_to_deg = self.coord_unit.to(u.deg)
            self.formatter_locator = AngleFormatterLocator(
                unit=self.coord_unit, format_unit=self.format_unit
            )
        else:
            raise ValueError(
                "coord_type should be one of 'scalar', 'longitude', or 'latitude'"
            )

    def set_major_formatter(self, formatter, show_decimal_unit=True):
        """
        Set the format string, or the callable, for the major tick labels.

        See `~astropy.visualization.wcsaxes.CoordinateHelper.set_major_formatter`.
        """
        if callable(formatter):
            self.custom_formatter = formatter
        elif isinstance(formatter, str):
            self.formatter_locator.format = formatter
            self.custom_formatter = None
        else:
            raise TypeError("formatter should be a string")

        self.formatter_locator.show_decimal_unit = show_decimal_unit

    def set_separator(self, separator):
        """
        Set the separator to use for the angle major tick labels.

        Parameters
        ----------
        separator : str or tuple or None
            The separator between numbers in sexagesimal representation. Can be
            either a string or a tuple (or `None` for default).
        """
        if not (self.formatter_locator.__class__ == AngleFormatterLocator):
            raise TypeError("Separator can only be specified for angle coordinates")
        if isinstance(separator, (str, tuple)) or separator is None:
            self.formatter_locator.sep = separator
        else:
            raise TypeError("separator should be a string, a tuple, or None")

    def set_format_unit(self, unit, decimal=None, show_decimal_unit=True):
        """
        Set the unit for the major tick labels.

        Parameters
        ----------
        unit : class:`~astropy.units.Unit`
            The unit to which the tick labels should be converted to.
        decimal : bool, optional
            Whether to use decimal formatting. By default this is `False`
            for degrees or hours (which therefore use sexagesimal formatting)
            and `True` for all other units.
        show_decimal_unit : bool, optional
            Whether to include units when in decimal mode.
        """
        self.formatter_locator.format_unit = u.Unit(unit)
        self.formatter_locator.decimal = decimal
        self.formatter_locator.show_decimal_unit = show_decimal_unit

    def get_format_unit(self):
        """
        Get the unit for the major tick labels.
        """
        return self.formatter_locator.format_unit

    def set_ticks(self, values=None, spacing=None, number=None):
        """
        Set where the major ticks go, by at most one of the three arguments.

        Parameters
        ----------
        values : iterable, optional
            The coordinate values at which to show the ticks.
        spacing : float, optional
            The spacing between ticks.
        number : float, optional
            The approximate number of ticks shown.
        """
        if sum([values is None, spacing is None, number is None]) < 2:
            raise ValueError(
                "At most one of values, spacing, or number should be specified"
            )

        if values is not None:
            self.formatter_locator.values = values
        elif spacing is not None:
            self.formatter_locator.spacing = spacing
        elif number is not None:
            self.formatter_locator.number = number

    @property
    def locator(self):
        return self.formatter_locator.locator

    @property
    def formatter(self):
        return self.custom_formatter or self.formatter_locator.formatter

    def spec(self):
        """
        What `~astropy.visualization.wcsaxes._layout.place_ticks` needs.
        """
        return CoordSpec(
            coord_index=self.coord_index,
            coord_type=self.coord_type,
            coord_unit=self.coord_unit,
            coord_wrap=self.coord_wrap,
            coord_scale_to_deg=self.coord_scale_to_deg,
            locator=self.locator,
            formatter=self.formatter,
            minor_locator=(
                self.formatter_locator.minor_locator
                if self.display_minor_ticks
                else None
            ),
            minor_frequency=self.minor_frequency,
        )

    def format_coord(self, value, format="auto"):
        """
        Format a value of this coordinate as the tick labels are formatted.

        Returns an empty string until ticks have been laid out, since the
        format depends on their spacing.

        Parameters
        ----------
        value : float
            The value to format, in ``coord_unit``.
        format : {'auto', 'ascii', 'latex'}, optional
            The format to use - by default the formatting will be adjusted
            depending on whether Matplotlib is using LaTeX or MathTex. To
            get plain ASCII strings, use format='ascii'.
        """
        if self.spacing is None:
            return ""

        fl = self.formatter_locator
        if isinstance(fl, AngleFormatterLocator):
            # Convert to degrees if needed
            if self.coord_scale_to_deg is not None:
                value *= self.coord_scale_to_deg

            if self.coord_type == "longitude":
                value = wrap_angle_at(value, self.coord_wrap.to_value(u.deg))
            value = value * u.degree
            value = value.to_value(fl._unit)

        string = self.formatter(
            values=[value] * fl._unit, spacing=self.spacing, format=format
        )

        return string[0]

    def default_axislabel(self, unit_format="latex"):
        """
        The axis label to use if none is set: the default label, with the
        unit for a scalar coordinate.

        Parameters
        ----------
        unit_format : str, optional
            The `~astropy.units.Unit` format for the unit.
        """
        unit = self.get_format_unit() or self.coord_unit

        if not unit or unit is u.one or self.coord_type in ("longitude", "latitude"):
            return f"{self.default_label}"
        else:
            return f"{self.default_label} [{unit:{unit_format}}]"


class AxesModel:
    """
    The coordinates that a WCSAxes shows, without matplotlib.

    Parameters
    ----------
    coords : list of `CoordinateModel`
        One per world coordinate, shown or not.
    aliases : dict
        Maps each lower-case name of a coordinate to its index in ``coords``.
    pixel_to_world : callable
        Converts an (N, 2) array of data pixels to an (N, n_world) array of
        world values.
    world_to_pixel : callable or None
        Converts an (N, n_world) array of world values to data pixels, or
        None if there is no inverse, in which case no grid lines are sampled.
    frame : str
        A key of `FRAMES`, or ``'custom'`` for a frame that `layout` does
        not know how to sample.
    """

    def __init__(
        self, coords, aliases, pixel_to_world, world_to_pixel=None, frame="rectangular"
    ):
        self.coords = coords
        self.aliases = aliases
        self.pixel_to_world = pixel_to_world
        self.world_to_pixel = world_to_pixel
        self.frame = frame

    @classmethod
    def from_coord_meta(
        cls,
        coord_meta,
        pixel_to_world,
        world_to_pixel=None,
        frame="rectangular",
        spine_names=None,
    ):
        """
        Build the coordinates described by a ``coord_meta`` dict.

        The dict has the keys that
        `~astropy.visualization.wcsaxes.CoordinatesMap` documents, as
        `coord_meta_from_wcs` returns them. Coordinates that are not
        ``visible`` get a ``coord_index`` of None. The ``default_*_position``
        lists, if present, set the positions.
        """
        if spine_names is None:
            spine_names = FRAMES[frame][0]
        coords = []
        aliases = {}

        visible_count = 0

        for index in range(len(coord_meta["type"])):
            # Extract coordinate metadata
            coord_type = coord_meta["type"][index]
            coord_wrap = coord_meta["wrap"][index]
            coord_unit = coord_meta["unit"][index]
            name = coord_meta["name"][index]

            visible = True
            if "visible" in coord_meta:
                visible = coord_meta["visible"][index]

            format_unit = None
            if "format_unit" in coord_meta:
                format_unit = coord_meta["format_unit"][index]

            default_label = name[0] if isinstance(name, (tuple, list)) else name
            if "default_axis_label" in coord_meta:
                default_label = coord_meta["default_axis_label"][index]

            coord_index = None
            if visible:
                visible_count += 1
                coord_index = visible_count - 1

            coord = CoordinateModel(
                coord_index=coord_index,
                coord_type=coord_type,
                coord_wrap=coord_wrap,
                coord_unit=coord_unit,
                format_unit=format_unit,
                default_label=default_label,
                spine_names=spine_names,
            )
            if "default_ticks_position" in coord_meta:
                coord.set_ticks_position(coord_meta["default_ticks_position"][index])
            if "default_ticklabel_position" in coord_meta:
                coord.set_ticklabel_position(
                    coord_meta["default_ticklabel_position"][index]
                )
            if "default_axislabel_position" in coord_meta:
                coord.set_axislabel_position(
                    coord_meta["default_axislabel_position"][index]
                )
            coords.append(coord)

            # Set up aliases for coordinates
            if isinstance(name, tuple):
                for nm in name:
                    nm = nm.lower()
                    # Do not replace an alias already in the map if we have
                    # more than one alias for this axis.
                    if nm not in aliases:
                        aliases[nm] = index
            else:
                aliases[name.lower()] = index

        return cls(coords, aliases, pixel_to_world, world_to_pixel, frame)

    @classmethod
    def from_wcs(cls, wcs, slices=None, frame=None):
        """
        Build the coordinates of a WCS, as WCSAxes shows them.

        Parameters
        ----------
        wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
            The WCS.
        slices : tuple, optional
            For a WCS with more than two pixel dimensions, one element per
            pixel dimension: ``'x'``, ``'y'``, an integer or a slice.
        frame : str, optional
            A key of `FRAMES`. By default, ``'rectangular1d'`` if the sliced
            WCS has one pixel dimension and ``'rectangular'`` otherwise, as
            WCSAxes chooses.
        """
        coord_meta, transform_wcs, invert_xy = coord_meta_from_wcs(wcs, slices)
        if frame is None:
            frame = "rectangular1d" if transform_wcs.pixel_n_dim == 1 else "rectangular"
        default_positions(coord_meta, frame, FRAMES[frame][0])
        return cls.from_coord_meta(
            coord_meta,
            partial(wcs_pixel_to_world, transform_wcs, invert_xy=invert_xy),
            partial(wcs_world_to_pixel, transform_wcs, invert_xy=invert_xy),
            frame=frame,
        )

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.coords[self.aliases[item.lower()]]
        else:
            return self.coords[item]

    def __contains__(self, item):
        if isinstance(item, str):
            return item.lower() in self.aliases
        else:
            return 0 <= item < len(self.coords)

    def __iter__(self):
        yield from self.coords

    def __len__(self):
        return len(self.coords)

    def layout(
        self,
        xlim,
        ylim,
        to_display,
        from_display,
        *,
        measure=None,
        tick_size=0.0,
        pad=0.0,
        font_size=0.0,
        unit_format="latex",
    ):
        """
        Lay out the ticks, tick labels, grid lines and axis labels of a view.

        This is what WCSAxes works out on each draw: it samples the spines of
        the frame, finds the range of each coordinate, places the ticks,
        assigns the spines of coordinates whose position is automatic, sorts
        and simplifies the tick labels, and, if the text can be measured,
        anchors the labels, drops those that overlap, and places the axis
        labels beyond them.

        Parameters
        ----------
        xlim, ylim : tuple
            The ``(min, max)`` limits of the view in data pixels.
        to_display, from_display : callable
            Convert (N, 2) arrays between data pixels and display pixels.
        measure : callable, optional
            Called as ``measure(text, x, y)``, it returns the width and
            height of a label in display pixels, as
            `~astropy.visualization.wcsaxes._layout.anchor_tick_labels`
            describes. Without it, labels are not anchored and axis labels
            are not placed.
        tick_size, pad : float, optional
            The length of a tick and the gap between a tick and its label, in
            display pixels.
        font_size : float, optional
            The font size of the axis labels in display pixels.
        unit_format : str, optional
            The `~astropy.units.Unit` format for the unit in a default axis
            label.

        Returns
        -------
        `Layout`
            The spines, the coordinate ranges, and one `CoordinateLayout` per
            coordinate that is shown.
        """
        from . import conf

        spine_names, sample_spines, spine_order = FRAMES[self.frame]
        frame_1d = self.frame == "rectangular1d"
        rectangular = self.frame == "rectangular"

        spines = sample_spines(
            xlim, ylim, conf.frame_boundary_samples, self.pixel_to_world, to_display
        )
        shown = [coord for coord in self.coords if coord.coord_index is not None]
        extent = [*xlim] if frame_1d else [*xlim, *ylim]
        ranges = find_coordinate_range(
            self.pixel_to_world,
            extent,
            [coord.coord_type for coord in shown],
            [coord.coord_unit for coord in shown],
            [coord.coord_wrap for coord in shown],
        )

        for coord in shown:
            coord._placed = place_ticks(
                coord.spec(),
                ranges[coord.coord_index],
                spines,
                self.pixel_to_world,
                to_display,
                from_display,
                frame_1d=frame_1d,
            )
            coord.spacing = coord._placed.spacing

        auto_assign_coord_positions([self.coords], spine_order)

        # As TickLabels.simplify_labels
        numerical_chars = "0123456789.+"
        if _rcparam("axes.unicode_minus") and not _rcparam("text.usetex"):
            numerical_chars += "\N{MINUS SIGN}"
        else:
            numerical_chars += "-"

        layouts = []
        existing = []  # the boxes of the tick labels kept so far
        label_spines = set()  # spines with tick labels, for the axis labels
        for coord in shown:
            labels = label_store(coord._placed)
            sort_labels(labels)
            if coord.simplify:
                simplify_labels(labels, numerical_chars)

            label_spines.update(coord.get_ticklabel_position())
            visible_axes = []
            if coord.ticklabels_visible:
                visible_axes = [a for a in coord.get_ticklabel_position() if a != "#"]

            anchors = kept = None
            if measure is not None and visible_axes:
                tick_pad = pad + (tick_size if coord.tick_out else 0.0)
                anchors = anchor_tick_labels(
                    labels, visible_axes, to_display, tick_pad, measure
                )

                def extent_of(axis, i, labels=labels, anchors=anchors):
                    text = labels.text[axis][i]
                    if text == "":
                        return None
                    (x, y), (w, h) = anchors[axis][i], measure(text, *anchors[axis][i])
                    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

                kept = list(
                    keep_tick_labels(
                        labels,
                        visible_axes,
                        extent_of,
                        coord.exclude_overlapping,
                        existing if coord.exclude_overlapping else [],
                    )
                )
                existing += [box for _, _, box in kept]

            grid = []
            if coord.grid and self.world_to_pixel is not None:
                grid = grid_lines(
                    coord.spec(),
                    ranges,
                    conf.grid_samples,
                    self.pixel_to_world,
                    self.world_to_pixel,
                )

            layouts.append(
                CoordinateLayout(
                    coord=coord,
                    ticks=coord._placed,
                    grid=grid,
                    labels=labels,
                    anchors=anchors,
                    kept=kept,
                    axis_label_text=coord.get_axislabel(unit_format),
                )
            )

        if measure is not None:
            # As AxisLabels.draw: the label goes beyond the box around every
            # tick label, on spines where tick labels are shown.
            union = None
            if existing:
                boxes = np.array(existing, dtype=float)
                union = (*boxes[:, :2].min(axis=0), *boxes[:, 2:].max(axis=0))
            for layout in layouts:
                coord = layout.coord
                if not coord.axislabel_visible:
                    continue
                for axis in coord.get_axislabel_position():
                    if axis == "#":
                        continue
                    rule = coord.axislabel_visibility_rule
                    if rule == "ticks":
                        if not (
                            coord.ticks_visible
                            and axis in coord.get_ticks_position()
                            and coord.tick_count(axis)
                        ):
                            continue
                    elif rule == "labels":
                        if not layout.kept:
                            continue
                    padding = font_size * coord._minpad(axis)
                    pixel = to_display(spines[axis].data)
                    x, y, normal = spine_midpoint(pixel, spine_normal_angle(pixel))
                    box = union if rectangular and axis in label_spines else None
                    layout.axis_labels[axis] = axis_label_position(
                        axis, x, y, normal, padding, font_size, rectangular, box
                    )

        return Layout(spines=spines, ranges=ranges, coords=layouts)


@dataclass
class CoordinateLayout:
    """
    The layout of one coordinate, from `AxesModel.layout`.

    Positions are in data pixels for ``ticks`` and ``grid``, and in display
    pixels for ``anchors``, ``kept`` and ``axis_labels``.

    Attributes
    ----------
    coord : `CoordinateModel`
        The coordinate.
    ticks : `~astropy.visualization.wcsaxes._layout.PlacedTicks`
        The major and minor ticks and the label texts, from ``place_ticks``.
    grid : list
        The grid lines as ``(vertices, codes)`` pairs, from ``grid_lines``,
        if the coordinate's ``grid`` is set.
    labels : `types.SimpleNamespace`
        The tick labels sorted along each spine and simplified, as the
        module docstring of `~astropy.visualization.wcsaxes._layout` says.
    anchors : dict or None
        Maps a spine to the centre of each label on it, if text was measured.
    kept : list or None
        The ``(spine, index, box)`` of each label to draw, if text was
        measured.
    axis_label_text : str
        The text of the axis label.
    axis_labels : dict
        Maps each spine on which the axis label is shown to its
        ``(x, y, rotation)``.
    """

    coord: CoordinateModel
    ticks: object
    grid: list
    labels: object
    anchors: dict | None
    kept: list | None
    axis_label_text: str
    axis_labels: dict = field(default_factory=dict)


@dataclass
class Layout:
    """
    The layout of one view, from `AxesModel.layout`.

    Attributes
    ----------
    spines : dict
        Maps each spine name to its `~astropy.visualization.wcsaxes._layout.SpineArrays`.
    ranges : list
        The ``(vmin, vmax)`` range of each coordinate that is shown.
    coords : list of `CoordinateLayout`
        One per coordinate that is shown.
    """

    spines: dict
    ranges: list
    coords: list
