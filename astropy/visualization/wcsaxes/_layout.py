# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tick and grid geometry for WCSAxes that does not depend on matplotlib.

This private module works out where the ticks of a coordinate cross the
spines of a frame, in which direction they point, and where its grid lines
run. It imports only numpy and astropy. The world coordinate transform, the
display transform, the locator and the formatter all come in as callables,
so a toolkit other than matplotlib can draw the same ticks and grid lines as
WCSAxes.

Conventions:

* Data pixels are the image pixels of the axes. Spines, tick positions and
  grid lines are in data pixels.
* Display pixels have their origin at the lower left, y pointing up and one
  unit per device pixel, as in matplotlib. Every angle is in display space,
  in degrees, counter-clockwise from +x.
"""

from collections.abc import Callable
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
    "grid_lines",
    "gridline_path_codes",
    "lon_lat_path_codes",
    "place_ticks",
    "resample_spine",
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
        Converts an (N, 2) array of world values to data pixels.

    Returns
    -------
    list of tuple or None
        One ``(pixel, codes)`` pair per tick value, where ``pixel`` is an
        (n_samples, 2) array of data pixels and ``codes`` holds the `MOVETO`
        or `LINETO` code of each vertex. None if the locator finds no ticks.
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
        return None

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
