# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tick, grid and label geometry for WCSAxes that does not depend on matplotlib.

This private module works out where the ticks of a coordinate cross the
spines of a frame, in which direction they point, where its grid lines run,
and where its tick labels and axis labels go. It imports only numpy and
astropy. The world coordinate transform, the display transform, the
locator, the formatter and the text measurement all come in as callables,
so a toolkit other than matplotlib can draw the same ticks, grid lines and
labels as WCSAxes.

Conventions:

* Data pixels are the image pixels of the axes. Spines, tick positions and
  grid lines are in data pixels.
* Display pixels have their origin at the lower left, y pointing up and one
  unit per device pixel, as in matplotlib. Every angle is in display space,
  in degrees, counter-clockwise from +x. Tick label positions and boxes are
  in display pixels.
* Tick labels are held by an object whose attributes ``world``, ``data``,
  ``angle``, ``tick_angle``, ``text`` and ``disp`` are dicts that map a spine
  name to a list with one entry per label, as in ``TickLabels``. ``angle`` is
  the spine normal at the tick, which is the direction of the label padding.
  `label_store` makes one from the output of `place_ticks`.
* ``to_display`` converts data pixels to display pixels. `place_ticks` gives
  it an (N, 2) array and `anchor_tick_labels` one ``(x, y)`` tuple, so a
  function passed to both has to accept both, as one that starts with
  ``numpy.asarray`` does.
* A box is ``(x0, y0, x1, y1)`` or ``[[x0, y0], [x1, y1]]``. A matplotlib
  ``Bbox`` also works.
"""

from collections import defaultdict
from collections.abc import Callable
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from astropy import units as u
from astropy.coordinates import angular_separation

__all__ = [
    "DISCONT_FACTOR",
    "LINETO",
    "MOVETO",
    "ROUND_TRIP_RTOL",
    "CoordSpec",
    "PlacedTicks",
    "SpineArrays",
    "TickTable",
    "anchor_tick_labels",
    "axis_label_position",
    "count_overlaps",
    "find_start_of_last_number",
    "grid_lines",
    "gridline_path_codes",
    "keep_tick_labels",
    "label_store",
    "lon_lat_path_codes",
    "place_ticks",
    "resample_spine",
    "simplify_labels",
    "sort_labels",
    "sort_using",
    "spine_midpoint",
    "spine_normal_angle",
    "wrap_angle_at",
]

# The codes of the vertices of a grid line, equal to those of
# matplotlib.path.Path. A line breaks at each MOVETO.
MOVETO = 1
LINETO = 2

# Tolerance for WCS round-tripping, relative to the scale size
ROUND_TRIP_RTOL = 1.0

# Tolerance for discontinuities relative to the median
DISCONT_FACTOR = 10.0


def wrap_angle_at(values, coord_wrap):
    # On ARM processors, np.mod emits warnings if there are NaN values in the
    # array, although this doesn't seem to happen on other processors.
    with np.errstate(invalid="ignore"):
        return np.mod(values - coord_wrap, 360.0) - (360.0 - coord_wrap)


def resample_spine(data, n_samples):
    """
    Resample a spine outline at evenly spaced points.

    Parameters
    ----------
    data : ndarray
        The outline, an (N, 2) array in data pixels.
    n_samples : int
        The number of points to return.

    Returns
    -------
    ndarray
        An (n_samples, 2) array, or ``data`` itself if it is empty.
    """
    if data.size == 0:
        return data
    p = np.linspace(0.0, 1.0, data.shape[0])
    p_new = np.linspace(0.0, 1.0, n_samples)
    return np.array([np.interp(p_new, p, d) for d in data.T]).transpose()


def spine_normal_angle(pixel):
    """
    Find the inward normal of each segment of a spine.

    Parameters
    ----------
    pixel : ndarray
        The spine, an (N, 2) array in display pixels.

    Returns
    -------
    ndarray
        The (N - 1) normal angles in degrees.
    """
    # Find angle normal to border and inwards, in display coordinate
    dx = pixel[1:, 0] - pixel[:-1, 0]
    dy = pixel[1:, 1] - pixel[:-1, 1]
    return np.degrees(np.arctan2(dx, -dy))


def spine_midpoint(pixel, normal_angle):
    """
    Find the point halfway along a spine, and the outward normal there.

    Parameters
    ----------
    pixel : ndarray
        The spine, an (N, 2) array in display pixels.
    normal_angle : ndarray
        The (N - 1) inward normals of the spine, from `spine_normal_angle`.

    Returns
    -------
    tuple
        The display position ``x`` and ``y`` of the midpoint, and the
        outward normal of the segment it lies on, in degrees.
    """
    x_disp, y_disp = pixel[:, 0], pixel[:, 1]
    # Get distance along the path
    d = np.hstack(
        [0.0, np.cumsum(np.sqrt(np.diff(x_disp) ** 2 + np.diff(y_disp) ** 2))]
    )
    xcen = np.interp(d[-1] / 2.0, d, x_disp)
    ycen = np.interp(d[-1] / 2.0, d, y_disp)

    # Find segment along which the mid-point lies
    imin = np.searchsorted(d, d[-1] / 2.0) - 1

    # Find normal of the axis label facing outwards on that segment
    normal_angle = normal_angle[imin] + 180.0
    return xcen, ycen, normal_angle


class SpineArrays(NamedTuple):
    """
    A sampled spine, as `place_ticks` reads it.

    `~astropy.visualization.wcsaxes.frame.Spine` has the same attributes.

    Parameters
    ----------
    data : ndarray
        The (N, 2) sample points in data pixels.
    world : ndarray
        The (N, n_world) world values at ``data``.
    normal_angle : ndarray
        The (N - 1) inward normals, from `spine_normal_angle`.
    """

    data: np.ndarray
    world: np.ndarray
    normal_angle: np.ndarray


class CoordSpec(NamedTuple):
    """
    What `place_ticks` needs to know about one coordinate.

    The fields match the attributes of
    `~astropy.visualization.wcsaxes.CoordinateHelper`.

    Parameters
    ----------
    coord_index : int
        The column of this coordinate in the world values.
    coord_type : str
        One of ``'longitude'``, ``'latitude'`` or ``'scalar'``.
    coord_unit : `~astropy.units.UnitBase`
        The unit of the world values.
    coord_wrap : `~astropy.units.Quantity` or None
        The angle at which a longitude wraps.
    coord_scale_to_deg : float or None
        The factor from ``coord_unit`` to degrees for an angle that is not in
        degrees, otherwise None.
    locator : callable
        Called as ``locator(vmin, vmax)``, it returns the tick values and
        their spacing, both as `~astropy.units.Quantity`.
    formatter : callable
        Called as ``formatter(values, spacing=spacing)``, once with the values
        of all major ticks, it returns their labels.
    minor_locator : callable or None
        Called as ``minor_locator(spacing, minor_frequency, vmin, vmax)``, it
        returns the minor tick values. None means no minor ticks.
    minor_frequency : int
        The number of minor ticks per major tick.
    """

    coord_index: int
    coord_type: str
    coord_unit: u.UnitBase
    coord_wrap: u.Quantity | None
    coord_scale_to_deg: float | None
    locator: Callable
    formatter: Callable
    minor_locator: Callable | None
    minor_frequency: int


class TickTable(NamedTuple):
    """
    Ticks in placement order: by spine, then by value, then by crossing.

    Parameters
    ----------
    axis : list of str
        The name of the spine each tick is on.
    world : ndarray
        The world value, in the coordinate's unit, wrapped for longitudes.
    pixel : ndarray
        The (n, 2) positions in data pixels.
    angle : ndarray
        The direction of each tick in display space, in degrees. Tick labels
        hold it as ``tick_angle``.
    normal : ndarray
        The inward normal of the spine at each tick, in degrees. Tick labels
        hold it as ``angle``.
    disp : ndarray
        The axis displacement, which is the position along the sampled spine
        as the segment index plus the fraction along that segment.
    """

    axis: list
    world: np.ndarray
    pixel: np.ndarray
    angle: np.ndarray
    normal: np.ndarray
    disp: np.ndarray


class PlacedTicks(NamedTuple):
    """
    The result of `place_ticks`.

    Parameters
    ----------
    spacing : `~astropy.units.Quantity`
        The spacing of the major ticks, as given by the locator.
    major : `TickTable`
        The major ticks.
    minor : `TickTable`
        The minor ticks, empty if there is no minor locator.
    text : list of str
        The formatter output, one label per major tick.
    label_world : list of `~astropy.units.Quantity`
        The values given to the formatter.
    """

    spacing: u.Quantity
    major: TickTable
    minor: TickTable
    text: list
    label_world: list


def place_ticks(
    spec,
    coord_range,
    spines,
    pixel_to_world,
    to_display,
    from_display,
    *,
    origin="lower",
    frame_1d=False,
):
    """
    Find where the ticks of one coordinate cross the spines of a frame.

    Parameters
    ----------
    spec : `CoordSpec`
        The coordinate.
    coord_range : tuple
        The ``(vmin, vmax)`` range of the coordinate, given to the locators.
    spines : dict
        Maps each spine name to a sampled spine, which has the attributes of
        `SpineArrays`.
    pixel_to_world : callable
        Converts an (N, 2) array of data pixels to an (N, n_world) array of
        world values.
    to_display : callable
        Converts an (N, 2) array of data pixels to display pixels.
    from_display : callable
        Converts an (N, 2) array of display pixels to data pixels.
    origin : {'lower', 'upper'}, optional
        Whether the y axis of the data pixels points up or down on screen.
    frame_1d : bool, optional
        Whether the frame is a 1-D frame, whose ticks point straight up on
        the bottom spine and straight down on the top spine.

    Returns
    -------
    `PlacedTicks`
        The major and minor ticks, and the major tick labels.
    """
    # TODO: this method should be optimized for speed

    # Here we determine the location and rotation of all the ticks. For
    # each axis, we can check the intersections for the specific
    # coordinate and once we have the tick positions, we can use the WCS
    # to determine the rotations.

    # First find the ticks we want to show
    tick_world_coordinates, spacing = spec.locator(*coord_range)

    if spec.minor_locator is not None:
        minor_ticks_w_coordinates = spec.minor_locator(
            spacing,
            spec.minor_frequency,
            *coord_range,
        )

    major, minor, label_world = [], [], []

    for axis, spine in spines.items():
        if spine.data.size == 0:
            continue

        if not frame_1d:
            # Determine tick rotation in display coordinates and compare to
            # the normal angle in display coordinates.

            pixel0 = spine.data
            world0 = spine.world[:, spec.coord_index]
            if np.isnan(world0).all():
                continue
            axes0 = to_display(pixel0)

            # Define a helper function to minimize code repetition
            def shifted_pixel_to_world(index, shift):
                pixel = axes0.copy()
                pixel[:, index] += shift
                pixel = from_display(pixel)
                with np.errstate(invalid="ignore"):
                    return pixel_to_world(pixel)[:, spec.coord_index]

            # Advance 2 pixels to the right in figure coordinates
            world1 = shifted_pixel_to_world(0, 2)
            dx = world1 - world0

            # Where advancing to the right results in NaN, advance to the left instead
            invalid1 = np.isnan(world1)
            if invalid1.any():
                world1 = shifted_pixel_to_world(0, -2)
                dx[invalid1] = (world0 - world1)[invalid1]

            # Advance 2 pixels up in figure coordinates
            amount = 2.0 if origin == "lower" else -2.0
            world2 = shifted_pixel_to_world(1, amount)
            dy = world2 - world0

            # Where advancing up results in NaN, advance down instead
            invalid2 = np.isnan(world2)
            if invalid2.any():
                world2 = shifted_pixel_to_world(1, -amount)
                dy[invalid2] = (world0 - world2)[invalid2]

            # Rotate by 90 degrees
            dx, dy = -dy, dx

            if spec.coord_type == "longitude":
                if spec.coord_scale_to_deg is not None:
                    dx *= spec.coord_scale_to_deg
                    dy *= spec.coord_scale_to_deg

                # Here we wrap at 180 not self.coord_wrap since we want to
                # always ensure abs(dx) < 180 and abs(dy) < 180
                dx = wrap_angle_at(dx, 180.0)
                dy = wrap_angle_at(dy, 180.0)

            tick_angle = np.degrees(np.arctan2(dy, dx))

            normal_angle_full = np.hstack([spine.normal_angle, spine.normal_angle[-1]])
            with np.errstate(invalid="ignore"):
                reset = ((normal_angle_full - tick_angle) % 360 > 90.0) & (
                    (tick_angle - normal_angle_full) % 360 > 90.0
                )
            tick_angle[reset] -= 180.0

        else:
            rotation = 90 if axis == "b" else -90
            tick_angle = np.zeros((spine.data.shape[0],)) + rotation

        # We find for each interval the starting and ending coordinate,
        # ensuring that we take wrapping into account correctly for
        # longitudes.
        w1 = spine.world[:-1, spec.coord_index]
        w2 = spine.world[1:, spec.coord_index]

        if spec.coord_type == "longitude":
            if spec.coord_scale_to_deg is not None:
                w1 = w1 * spec.coord_scale_to_deg
                w2 = w2 * spec.coord_scale_to_deg

            w1 = wrap_angle_at(w1, spec.coord_wrap.to_value(u.deg))
            w2 = wrap_angle_at(w2, spec.coord_wrap.to_value(u.deg))
            with np.errstate(invalid="ignore"):
                w1[w2 - w1 > 180.0] += 360
                w2[w1 - w2 > 180.0] += 360

            if spec.coord_scale_to_deg is not None:
                w1 = w1 / spec.coord_scale_to_deg
                w2 = w2 / spec.coord_scale_to_deg

        # For longitudes, we need to check ticks as well as ticks + 360,
        # since the above can produce pairs such as 359 to 361 or 0.5 to
        # 1.5, both of which would match a tick at 0.75. Otherwise we just
        # check the ticks determined above.
        _ticks_on_spine(
            spec,
            tick_world_coordinates,
            spine,
            axis,
            w1,
            w2,
            tick_angle,
            major,
            label_world,
        )

        if spec.minor_locator is not None:
            _ticks_on_spine(
                spec,
                minor_ticks_w_coordinates,
                spine,
                axis,
                w1,
                w2,
                tick_angle,
                minor,
            )

    # format tick labels
    text = spec.formatter(u.Quantity(label_world), spacing=spacing)

    return PlacedTicks(
        spacing, _tick_table(major), _tick_table(minor), text, label_world
    )


def _ticks_on_spine(
    spec,
    tick_world_coordinates,
    spine,
    axis,
    w1,
    w2,
    tick_angle,
    rows,
    label_world=None,
):
    if spec.coord_type == "longitude":
        tick_world_coordinates_values = tick_world_coordinates.to_value(u.deg)
        tick_world_coordinates_values = np.hstack(
            [tick_world_coordinates_values, tick_world_coordinates_values + 360]
        )
        tick_world_coordinates_values *= u.deg.to(spec.coord_unit)
    else:
        tick_world_coordinates_values = tick_world_coordinates.to_value(spec.coord_unit)

    for t in tick_world_coordinates_values:
        # Find steps where a tick is present. We have to check
        # separately for the case where the tick falls exactly on the
        # frame points, otherwise we'll get two matches, one for w1 and
        # one for w2.
        with np.errstate(invalid="ignore"):
            intersections = np.hstack(
                [
                    np.nonzero((t - w1) == 0)[0],
                    np.nonzero(((t - w1) * (t - w2)) < 0)[0],
                ]
            )

        # But we also need to check for intersection with the last w2
        if t - w2[-1] == 0:
            intersections = np.append(intersections, len(w2) - 1)

        # Loop over ticks, and find exact pixel coordinates by linear
        # interpolation
        for imin in intersections:
            imax = imin + 1

            if np.allclose(w1[imin], w2[imin], rtol=1.0e-13, atol=1.0e-13):
                continue  # tick is exactly aligned with frame
            else:
                frac = (t - w1[imin]) / (w2[imin] - w1[imin])
                x_data_i = spine.data[imin, 0] + frac * (
                    spine.data[imax, 0] - spine.data[imin, 0]
                )
                y_data_i = spine.data[imin, 1] + frac * (
                    spine.data[imax, 1] - spine.data[imin, 1]
                )
                delta_angle = tick_angle[imax] - tick_angle[imin]
                if delta_angle > 180.0:
                    delta_angle -= 360.0
                elif delta_angle < -180.0:
                    delta_angle += 360.0
                angle_i = tick_angle[imin] + frac * delta_angle

            if spec.coord_type == "longitude":
                if spec.coord_scale_to_deg is not None:
                    world = t * spec.coord_scale_to_deg
                else:
                    world = t

                world = wrap_angle_at(world, spec.coord_wrap.to_value(u.deg))

                if spec.coord_scale_to_deg is not None:
                    world /= spec.coord_scale_to_deg

            else:
                world = t

            rows.append(
                (
                    axis,
                    world,
                    (x_data_i, y_data_i),
                    angle_i,
                    spine.normal_angle[imin],
                    imin + frac,
                )
            )

            if label_world is not None:
                label_world.append(
                    (world * spec.coord_unit).to(tick_world_coordinates.unit)
                )


def _tick_table(rows):
    axis, world, pixel, angle, normal, disp = zip(*rows) if rows else ([],) * 6
    return TickTable(
        list(axis),
        np.array(world, dtype=float),
        np.array(pixel, dtype=float).reshape(-1, 2),
        np.array(angle, dtype=float),
        np.array(normal, dtype=float),
        np.array(disp, dtype=float),
    )


def grid_lines(spec, coord_ranges, n_samples, pixel_to_world, world_to_pixel):
    """
    Sample the grid lines of one coordinate.

    Parameters
    ----------
    spec : `CoordSpec`
        The coordinate. Only its index, type, unit and locator are used.
    coord_ranges : list
        The ``(vmin, vmax)`` range of each of the two coordinates. The range
        of this coordinate is given to the locator, and each grid line spans
        the range of the other one.
    n_samples : int
        The number of points along each grid line.
    pixel_to_world : callable
        Converts an (N, 2) array of data pixels to an (N, 2) array of world
        values.
    world_to_pixel : callable
        Converts an (N, 2) array of world values to data pixels. It is called
        once, and not at all if the locator finds no ticks.

    Returns
    -------
    list of tuple
        One ``(pixel, codes)`` pair per tick value, where ``pixel`` is an
        (n_samples, 2) array of data pixels and ``codes`` holds the `MOVETO`
        or `LINETO` code of each vertex. Empty if the locator finds no ticks.
    """
    # For 3-d WCS with a correlated third axis, the *proper* way of
    # drawing a grid should be to find the world coordinates of all pixels
    # and drawing contours. What we are doing here assumes that we can
    # define the grid lines with just two of the coordinates (and
    # therefore assumes that the other coordinates are fixed and set to
    # the value in the slice). Here we basically assume that if the WCS
    # had a third axis, it has been abstracted away in the transformation.

    tick_world_coordinates, spacing = spec.locator(*coord_ranges[spec.coord_index])
    tick_world_coordinates_values = tick_world_coordinates.to_value(spec.coord_unit)

    n_coord = len(tick_world_coordinates_values)
    if n_coord == 0:
        return []

    xy_world = np.zeros((n_samples * n_coord, 2))

    lines = []

    for iw, w in enumerate(tick_world_coordinates_values):
        subset = slice(iw * n_samples, (iw + 1) * n_samples)
        if spec.coord_index == 0:
            xy_world[subset, 0] = np.repeat(w, n_samples)
            xy_world[subset, 1] = np.linspace(
                coord_ranges[1][0], coord_ranges[1][1], n_samples
            )
        else:
            xy_world[subset, 0] = np.linspace(
                coord_ranges[0][0], coord_ranges[0][1], n_samples
            )
            xy_world[subset, 1] = np.repeat(w, n_samples)

    # We now convert all the world coordinates to pixel coordinates in a
    # single go rather than doing this in the gridline to path conversion
    # to fully benefit from vectorized coordinate transformations.

    # Transform line to pixel coordinates
    pixel = world_to_pixel(xy_world)

    # Create round-tripped values for checking
    xy_world_round = pixel_to_world(pixel)

    for iw in range(n_coord):
        subset = slice(iw * n_samples, (iw + 1) * n_samples)
        lines.append(
            _gridline(spec, xy_world[subset], pixel[subset], xy_world_round[subset])
        )

    return lines


def _gridline(spec, xy_world, pixel, xy_world_round):
    if spec.coord_type == "scalar":
        return pixel, gridline_path_codes(xy_world, pixel)
    else:
        return pixel, lon_lat_path_codes(xy_world, pixel, xy_world_round)


def lon_lat_path_codes(lon_lat, pixel, lon_lat_check):
    """
    Find the path codes of a curve, taking into account discontinuities.

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

    Returns
    -------
    ndarray
        The uint8 code of each vertex: `MOVETO` where the curve starts or
        resumes, `LINETO` elsewhere.
    """
    # In some spherical projections, some parts of the curve are 'behind' or
    # 'in front of' the plane of the image, so we find those by reversing the
    # transformation and finding points where the result is not consistent.

    sep = angular_separation(
        np.radians(lon_lat[:, 0]),
        np.radians(lon_lat[:, 1]),
        np.radians(lon_lat_check[:, 0]),
        np.radians(lon_lat_check[:, 1]),
    )

    # Define the relevant scale size using the separation between the first two points
    scale_size = angular_separation(
        *np.radians(lon_lat[0, :]), *np.radians(lon_lat[1, :])
    )

    with np.errstate(invalid="ignore"):
        sep[sep > np.pi] -= 2.0 * np.pi

        mask = sep > ROUND_TRIP_RTOL * scale_size

    # Mask values with invalid pixel positions
    mask = mask | np.isnan(pixel[:, 0]) | np.isnan(pixel[:, 1])

    # We can now start to set up the codes for the Path.
    codes = np.zeros(lon_lat.shape[0], dtype=np.uint8)
    codes[:] = LINETO
    codes[0] = MOVETO
    codes[mask] = MOVETO

    # Also need to move to point *after* a hidden value
    codes[1:][mask[:-1]] = MOVETO

    # We now go through and search for discontinuities in the curve that would
    # be due to the curve going outside the field of view, invalid WCS values,
    # or due to discontinuities in the projection.

    # We start off by pre-computing the step in pixel coordinates from one
    # point to the next. The idea is to look for large jumps that might indicate
    # discontinuities.
    step = np.sqrt(
        (pixel[1:, 0] - pixel[:-1, 0]) ** 2 + (pixel[1:, 1] - pixel[:-1, 1]) ** 2
    )

    # We search for discontinuities by looking for places where the step
    # is larger by more than a given factor compared to the median
    # discontinuous = step > DISCONT_FACTOR * np.median(step)
    discontinuous = step[1:] > DISCONT_FACTOR * step[:-1]

    # Skip over discontinuities
    codes[2:][discontinuous] = MOVETO

    # The above missed the first step, so check that too
    if len(step) >= 2 and step[0] > DISCONT_FACTOR * step[1]:
        codes[1] = MOVETO

    return codes


def gridline_path_codes(world, pixel):
    """
    Find the path codes of a grid line.

    Parameters
    ----------
    world : ndarray
        The longitude and latitude values along the curve, given as a (n,2)
        array.
    pixel : ndarray
        The pixel coordinates corresponding to ``lon_lat``.

    Returns
    -------
    ndarray
        The uint8 code of each vertex: `MOVETO` where the line starts or
        resumes after an invalid pixel, `LINETO` elsewhere.
    """
    # Mask values with invalid pixel positions
    mask = np.isnan(pixel[:, 0]) | np.isnan(pixel[:, 1])

    # We can now start to set up the codes for the Path.
    codes = np.zeros(world.shape[0], dtype=np.uint8)
    codes[:] = LINETO
    codes[0] = MOVETO
    codes[mask] = MOVETO

    # Also need to move to point *after* a hidden value
    codes[1:][mask[:-1]] = MOVETO

    # We now go through and search for discontinuities in the curve that would
    # be due to the curve going outside the field of view, invalid WCS values,
    # or due to discontinuities in the projection.

    return codes


def label_store(placed):
    """
    Hold the major tick labels of a coordinate as ``TickLabels`` does.

    The labels are in the order of the ticks, and ready for `sort_labels`.

    Parameters
    ----------
    placed : `PlacedTicks`
        The ticks of the coordinate, from `place_ticks`.

    Returns
    -------
    `types.SimpleNamespace`
        The tick labels, as described in the module docstring. ``angle`` is
        ``TickTable.normal`` and ``tick_angle`` is ``TickTable.angle``.
    """
    labels = SimpleNamespace(
        world=defaultdict(list),
        data=defaultdict(list),
        angle=defaultdict(list),
        tick_angle=defaultdict(list),
        text=defaultdict(list),
        disp=defaultdict(list),
    )
    m = placed.major
    for axis, world, (x, y), angle, normal, disp, text in zip(
        m.axis, m.world, m.pixel, m.angle, m.normal, m.disp, placed.text
    ):
        labels.world[axis].append(world)
        labels.data[axis].append((x, y))
        labels.angle[axis].append(normal)
        labels.tick_angle[axis].append(angle)
        labels.text[axis].append(text)
        labels.disp[axis].append(disp)
    return labels


def sort_using(X, Y):
    return [x for (y, x) in sorted(zip(Y, X))]


def find_start_of_last_number(label, numerical_chars):
    """
    Given a label, find the index of the start of the last numerical value
    in the label.

    Parameters
    ----------
    label : str
        The label to search.
    numerical_chars : str
        The characters that a number is made of, including the minus sign
        used in the labels.
    """
    in_number = False
    for j in range(len(label) - 1, -1, -1):
        if in_number:
            if label[j] not in numerical_chars:
                return j + 1
        elif label[j] in numerical_chars:
            in_number = True


def sort_labels(labels):
    """
    Sort tick labels by axis displacement, in place.

    This allows us to figure out which parts of labels to not repeat.

    Parameters
    ----------
    labels : object
        The tick labels, as described in the module docstring.
    """
    for axis in labels.world:
        labels.world[axis] = sort_using(labels.world[axis], labels.disp[axis])
        labels.data[axis] = sort_using(labels.data[axis], labels.disp[axis])
        labels.angle[axis] = sort_using(labels.angle[axis], labels.disp[axis])
        labels.tick_angle[axis] = sort_using(labels.tick_angle[axis], labels.disp[axis])
        labels.text[axis] = sort_using(labels.text[axis], labels.disp[axis])
        labels.disp[axis] = sort_using(labels.disp[axis], labels.disp[axis])


def simplify_labels(labels, numerical_chars):
    """
    Figure out which parts of labels can be dropped to avoid repetition.

    The labels must already be sorted with `sort_labels`. Their text is
    shortened in place.

    Parameters
    ----------
    labels : object
        The tick labels, as described in the module docstring.
    numerical_chars : str
        The characters that a number is made of, as given to
        `find_start_of_last_number`.
    """
    for axis in labels.world:
        t1 = labels.text[axis][0]
        for i in range(1, len(labels.world[axis])):
            t2 = labels.text[axis][i]

            if t1 == t2:
                # In this case, we still need to preserve the last segment
                # of the label. We search backwards from the end, and
                # search for a number, and we then search for the first
                # non-number (and non-decimal place) character we can find.
                start = find_start_of_last_number(t2, numerical_chars)
            else:
                start = 0
                for j in range(min(len(t1), len(t2))):
                    if t1[j] != t2[j]:
                        start = find_start_of_last_number(t2[: j + 1], numerical_chars)
                        break
                else:
                    # One of the strings is a prefix of the other (up to
                    # the length of the shorter one) without any
                    # differing character, so the entire overlapping
                    # part can be considered shared and only the extra
                    # trailing part of t2 (if any) needs to be shown.
                    if len(t2) > len(t1):
                        start = find_start_of_last_number(
                            t2[: len(t1) + 1], numerical_chars
                        )

            if start != 0:
                starts_dollar = t2.startswith("$")
                labels.text[axis][i] = t2[start:]
                if starts_dollar:
                    labels.text[axis][i] = "$" + labels.text[axis][i]

            # Remove any empty LaTeX inline math mode string
            if labels.text[axis][i] == "$$":
                labels.text[axis][i] = ""

            t1 = t2


def anchor_tick_labels(labels, visible_axes, to_display, pad, measure):
    """
    Find where to centre each tick label.

    A label is moved away from its tick in the direction of the spine
    normal, until its box clears the tick by ``pad``. It is also moved along
    the spine towards the direction of the tick, by at most 60 degrees from
    the normal.

    Parameters
    ----------
    labels : object
        The tick labels, as described in the module docstring.
    visible_axes : list of str
        The spines on which labels are shown.
    to_display : callable
        Converts the position ``(x, y)`` of a tick in data pixels to display
        pixels. It is given one tuple at a time.
    pad : float
        The gap between a tick and its label, plus the length of the tick if
        it points out, in display pixels.
    measure : callable
        Called as ``measure(text, x, y)`` with the display position of the
        tick, it returns the width and height of the label in display pixels.
        To place labels as matplotlib does, these are what
        ``Text.get_window_extent`` gives for one line of text: the advance
        width, and the font's ascender plus descender, unless the ink is
        taller.

    Returns
    -------
    dict
        Maps each spine in ``visible_axes`` to a dict from the index of a label
        to the display position ``(x, y)`` of its centre. Empty labels are
        left out.
    """
    xy = {axis: {} for axis in visible_axes}

    for axis in visible_axes:
        for i in range(len(labels.world[axis])):
            # In the event that the label is empty (which is not expected
            # but could happen in unforeseen corner cases), we should just
            # skip to the next label.
            if labels.text[axis][i] == "":
                continue

            x, y = to_display(labels.data[axis][i])

            # Set initial position and find bounding box
            width, height = measure(labels.text[axis][i], x, y)

            # The pad direction (typically perpendicular to the spine)
            pad_angle = np.radians(labels.angle[axis][i])
            px = np.cos(pad_angle)
            py = np.sin(pad_angle)

            # If the tick angle is NaN, use the pad angle for the tick angle
            tick_angle = (
                np.radians(labels.tick_angle[axis][i])
                if not np.isnan(labels.tick_angle[axis][i])
                else pad_angle
            )
            tx = np.cos(tick_angle)
            ty = np.sin(tick_angle)

            # Set anchor point for label where the pad direction intersects bounding box
            with np.errstate(divide="ignore"):
                if np.abs(py / px) < np.abs(height / width):
                    ax = width / 2 * np.sign(px)
                    ay = width / 2 * py / np.abs(px)
                else:
                    ax = height / 2 * px / np.abs(py)
                    ay = height / 2 * np.sign(py)

            # Extract the component of the tick direction perpendicular to the pad direction
            scale = tx * px + ty * py
            vx = tx - px * scale
            vy = ty - py * scale

            # We scale the above vector for adding to the pad direction to get a combined
            # displacement.  When the tick direction is close to the pad direction, the scaling
            # is such that the displacement direction is exactly the tick direction.  When the
            # tick direction is close to perpendicular to the pad direction, we cap the scaling,
            # which means that the effective tick direction is never more than 60 degrees
            # perpendicular to the pad direction.  This prevents pushing the tick label too far
            # away from the tick.
            scale = np.max([scale, 0.5])  # 0.5 == cos(60 deg)
            vx /= scale
            vy /= scale

            # Pad the anchor point in the combined displacement direction
            dx = ax + (vx + px) * pad
            dy = ay + (vy + py) * pad

            xy[axis][i] = (x - dx, y - dy)

    return xy


def count_overlaps(box, boxes):
    """
    Count the boxes that overlap a box.

    This is ``Bbox.count_overlaps`` from matplotlib, with the same swaps and
    comparisons: boxes that only touch do not overlap, a box whose corners
    are reversed is swapped first, and a comparison with NaN is false.

    Parameters
    ----------
    box : array-like
        The box.
    boxes : list of array-like
        The boxes to compare with.

    Returns
    -------
    int
        The number of ``boxes`` that overlap ``box``.
    """
    ax0, ay0, ax1, ay1 = np.asarray(box, dtype=float).reshape(4)
    if ax1 < ax0:
        ax0, ax1 = ax1, ax0
    if ay1 < ay0:
        ay0, ay1 = ay1, ay0
    bx0, by0, bx1, by1 = (
        np.array([np.asarray(b, dtype=float).reshape(4) for b in boxes])
        .reshape(-1, 4)
        .T
    )
    swap = bx1 < bx0
    bx0, bx1 = np.where(swap, bx1, bx0), np.where(swap, bx0, bx1)
    swap = by1 < by0
    by0, by1 = np.where(swap, by1, by0), np.where(swap, by0, by1)
    return int(
        np.count_nonzero(~((bx1 <= ax0) | (by1 <= ay0) | (bx0 >= ax1) | (by0 >= ay1)))
    )


def keep_tick_labels(labels, visible_axes, extent, exclude_overlapping, existing):
    """
    Choose the tick labels to draw, in drawing order.

    This is a generator, so that the caller can draw each label as soon as
    it is yielded, before the next label is measured.

    Parameters
    ----------
    labels : object
        The tick labels, as described in the module docstring.
    visible_axes : list of str
        The spines on which labels are shown.
    extent : callable
        Called as ``extent(axis, i)``, it returns the box of label ``i`` on
        spine ``axis``, in display pixels, or None if the label is empty.
    exclude_overlapping : bool
        Whether to drop a label that overlaps one already kept, or one of
        ``existing``.
    existing : list
        The boxes of the tick labels of the coordinates drawn before.

    Yields
    ------
    tuple
        ``(axis, i, box)`` for each label to draw, with ``box`` as returned
        by ``extent``.
    """
    kept = []

    for axis in visible_axes:
        if axis == "#":
            continue

        for i in range(len(labels.world[axis])):
            # With matplotlib, this also sets the label text, position and
            # alignment
            bb = extent(axis, i)
            if bb is None:
                continue

            # TODO: the problem here is that we might get rid of a label
            # that has a key starting bit such as -0:30 where the -0
            # might be dropped from all other labels.
            if not exclude_overlapping or count_overlaps(bb, kept + existing) == 0:
                kept.append(bb)
                yield axis, i, bb


def axis_label_position(
    axis, x, y, normal_angle, padding, text_size, rectangular, union
):
    """
    Place the axis label of a coordinate on a spine.

    Parameters
    ----------
    axis : str
        The name of the spine. On a rectangular frame, ``'l'``, ``'r'``,
        ``'b'`` and ``'t'`` move the label out from the spine; any other name
        leaves it on the spine.
    x, y : float
        The midpoint of the spine in display pixels, from `spine_midpoint`.
    normal_angle : float
        The outward normal at the midpoint in degrees, from `spine_midpoint`.
    padding : float
        The gap between the label and the spine, or the tick labels, in
        display pixels.
    text_size : float
        The font size of the label in display pixels.
    rectangular : bool
        Whether the frame is rectangular. If not, the label is moved out
        along the normal by the padding plus one and a half times the font
        size.
    union : array-like or None
        On a rectangular frame, the box around the tick labels, in display
        pixels, that the label is placed beyond. If None, the label is placed
        from the spine.

    Returns
    -------
    tuple
        The display position ``x`` and ``y`` of the centre of the label, and
        its rotation in degrees.
    """
    label_angle = (normal_angle - 90.0) % 360.0
    if 135 < label_angle < 225:
        label_angle += 180

    # Find label position by looking at the bounding box of ticks'
    # labels and the image. It sets the default padding at 1 times the
    # axis label font size which can also be changed by setting
    # the minpad parameter.

    if rectangular:
        if union is not None:
            x0, y0, x1, y1 = np.asarray(union, dtype=float).reshape(4)

        if axis == "l":
            if union is not None:
                x = x0
            x = x - padding

        elif axis == "r":
            if union is not None:
                x = x1
            x = x + padding

        elif axis == "b":
            if union is not None:
                y = y0
            y = y - padding

        elif axis == "t":
            if union is not None:
                y = y1
            y = y + padding

    else:  # arbitrary axis
        x = x + np.cos(np.radians(normal_angle)) * (padding + text_size * 1.5)
        y = y + np.sin(np.radians(normal_angle)) * (padding + text_size * 1.5)

    return x, y, label_angle
