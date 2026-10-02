# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Matplotlib-free tick, grid, and label geometry shared by WCSAxes.

Transforms, locators, formatters, and text measurement come from the caller.
Data pixels are image coordinates; display pixels have their origin at the
lower left with y pointing up. All angles are counter-clockwise from +x in
display space. Tick and grid positions are in data pixels; labels and boxes
are in display pixels. Boxes are (x0, y0, x1, y1) or [[x0, y0], [x1, y1]].

Label stores have world, data, angle, tick_angle, text, and disp dictionaries
mapping spine names to lists. ``angle`` is the inward spine normal and
``tick_angle`` is the tick direction. ``label_store`` builds these dictionaries
from tick records and formatted labels; WCSAxes already has them in TickLabels.
"""

from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from astropy import units as u
from astropy.coordinates import angular_separation

# Vertex codes match matplotlib.path.Path; a line breaks at each MOVETO.
MOVETO = 1
LINETO = 2

# WCS round-trip tolerance relative to the scale size, and discontinuity factor.
ROUND_TRIP_RTOL = 1.0
DISCONT_FACTOR = 10.0


def wrap_angle_at(values, coord_wrap):
    # On ARM processors, np.mod emits warnings if there are NaN values in the
    # array, although this doesn't seem to happen on other processors.
    with np.errstate(invalid="ignore"):
        return np.mod(values - coord_wrap, 360.0) - (360.0 - coord_wrap)


def resample_spine(data, n_samples):
    """Resample an (N, 2) spine at evenly spaced points; leave empty data alone."""
    if data.size == 0:
        return data
    p = np.linspace(0.0, 1.0, data.shape[0])
    p_new = np.linspace(0.0, 1.0, n_samples)
    return np.array([np.interp(p_new, p, d) for d in data.T]).transpose()


def spine_normal_angle(pixel):
    """Return the inward display-space normal angle of each spine segment."""
    # Find angle normal to border and inwards, in display coordinate
    dx = pixel[1:, 0] - pixel[:-1, 0]
    dy = pixel[1:, 1] - pixel[:-1, 1]
    return np.degrees(np.arctan2(dx, -dy))


def spine_midpoint(pixel, normal_angle):
    """Return the display-space midpoint and outward normal of a spine."""
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


def rectangular_spines(xlim, ylim, n_samples, pixel_to_world, to_display):
    """Sample rectangular spines in the same order and direction as WCSAxes.

    Each spine has data (N, 2), world (N, n_world), and normal_angle (N - 1)
    arrays. Transforms accept arrays of points; limits are (min, max) pairs.
    """
    (x0, x1), (y0, y1) = xlim, ylim
    outlines = {
        "b": [[x0, y0], [x1, y0]],
        "r": [[x1, y0], [x1, y1]],
        "t": [[x1, y1], [x0, y1]],
        "l": [[x0, y1], [x0, y0]],
    }
    spines = {}
    for axis, outline in outlines.items():
        data = resample_spine(np.array(outline, dtype=float), n_samples)
        with np.errstate(invalid="ignore"):
            world = pixel_to_world(data)
        spines[axis] = SimpleNamespace(
            data=data, world=world, normal_angle=spine_normal_angle(to_display(data))
        )
    return spines


def place_ticks(
    coord,
    coord_range,
    spines,
    pixel_to_world,
    to_display,
    from_display,
    *,
    minor_locator=None,
    minor_frequency=5,
    origin="lower",
    frame_1d=False,
):
    """Return major/minor tick records, their spacing, and the formatter values.

    The coordinate provides coord_index, coord_type, coord_unit, coord_wrap,
    _coord_scale_to_deg, and locator, as CoordinateHelper does. Spines have
    data, world, and normal_angle arrays. All transforms accept (N, 2) arrays.
    Minor ticks are computed only when a minor_locator is supplied.

    Records contain the keyword arguments for TickLabels.add: axis, world,
    data, angle (spine normal), tick_angle, and axis_displacement. They are
    ordered by spine, then tick value, then crossing. The formatter values
    are quantities in the locator's unit, one per major tick.
    """
    # TODO: this method should be optimized for speed

    # Here we determine the location and rotation of all the ticks. For
    # each axis, we can check the intersections for the specific
    # coordinate and once we have the tick positions, we can use the WCS
    # to determine the rotations.

    # First find the ticks we want to show
    tick_world_coordinates, spacing = coord.locator(*coord_range)

    if minor_locator is not None:
        minor_ticks_w_coordinates = minor_locator(
            spacing,
            minor_frequency,
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
            world0 = spine.world[:, coord.coord_index]
            if np.isnan(world0).all():
                continue
            axes0 = to_display(pixel0)

            # Define a helper function to minimize code repetition
            def shifted_pixel_to_world(index, shift):
                pixel = axes0.copy()
                pixel[:, index] += shift
                pixel = from_display(pixel)
                with np.errstate(invalid="ignore"):
                    return pixel_to_world(pixel)[:, coord.coord_index]

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

            if coord.coord_type == "longitude":
                if coord._coord_scale_to_deg is not None:
                    dx *= coord._coord_scale_to_deg
                    dy *= coord._coord_scale_to_deg

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
        w1 = spine.world[:-1, coord.coord_index]
        w2 = spine.world[1:, coord.coord_index]

        if coord.coord_type == "longitude":
            if coord._coord_scale_to_deg is not None:
                w1 = w1 * coord._coord_scale_to_deg
                w2 = w2 * coord._coord_scale_to_deg

            w1 = wrap_angle_at(w1, coord.coord_wrap.to_value(u.deg))
            w2 = wrap_angle_at(w2, coord.coord_wrap.to_value(u.deg))
            with np.errstate(invalid="ignore"):
                w1[w2 - w1 > 180.0] += 360
                w2[w1 - w2 > 180.0] += 360

            if coord._coord_scale_to_deg is not None:
                w1 = w1 / coord._coord_scale_to_deg
                w2 = w2 / coord._coord_scale_to_deg

        # For longitudes, we need to check ticks as well as ticks + 360,
        # since the above can produce pairs such as 359 to 361 or 0.5 to
        # 1.5, both of which would match a tick at 0.75. Otherwise we just
        # check the ticks determined above.
        _ticks_on_spine(
            coord,
            tick_world_coordinates,
            spine,
            axis,
            w1,
            w2,
            tick_angle,
            major,
            label_world,
        )

        if minor_locator is not None:
            _ticks_on_spine(
                coord,
                minor_ticks_w_coordinates,
                spine,
                axis,
                w1,
                w2,
                tick_angle,
                minor,
            )

    return major, minor, spacing, label_world


def _ticks_on_spine(
    coord,
    tick_world_coordinates,
    spine,
    axis,
    w1,
    w2,
    tick_angle,
    rows,
    label_world=None,
):
    if coord.coord_type == "longitude":
        tick_world_coordinates_values = tick_world_coordinates.to_value(u.deg)
        tick_world_coordinates_values = np.hstack(
            [tick_world_coordinates_values, tick_world_coordinates_values + 360]
        )
        tick_world_coordinates_values *= u.deg.to(coord.coord_unit)
    else:
        tick_world_coordinates_values = tick_world_coordinates.to_value(
            coord.coord_unit
        )

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

            if coord.coord_type == "longitude":
                if coord._coord_scale_to_deg is not None:
                    world = t * coord._coord_scale_to_deg
                else:
                    world = t

                world = wrap_angle_at(world, coord.coord_wrap.to_value(u.deg))

                if coord._coord_scale_to_deg is not None:
                    world /= coord._coord_scale_to_deg

            else:
                world = t

            rows.append(
                dict(
                    axis=axis,
                    data=(x_data_i, y_data_i),
                    world=world,
                    angle=spine.normal_angle[imin],
                    tick_angle=angle_i,
                    axis_displacement=imin + frac,
                )
            )

            if label_world is not None:
                label_world.append(
                    (world * coord.coord_unit).to(tick_world_coordinates.unit)
                )


def grid_lines(coord, coord_ranges, n_samples, pixel_to_world, world_to_pixel):
    """Return (pixel, codes) pairs for the grid lines of one coordinate.

    Only coord_index, coord_type, coord_unit, and locator are read from the
    coordinate. coord_ranges holds both world-coordinate ranges. Each line
    has n_samples vertices in data pixels. world_to_pixel is called once,
    and only if the locator returns ticks.
    """
    # For 3-d WCS with a correlated third axis, the *proper* way of
    # drawing a grid should be to find the world coordinates of all pixels
    # and drawing contours. What we are doing here assumes that we can
    # define the grid lines with just two of the coordinates (and
    # therefore assumes that the other coordinates are fixed and set to
    # the value in the slice). Here we basically assume that if the WCS
    # had a third axis, it has been abstracted away in the transformation.

    tick_world_coordinates, spacing = coord.locator(*coord_ranges[coord.coord_index])
    tick_world_coordinates_values = tick_world_coordinates.to_value(coord.coord_unit)

    n_coord = len(tick_world_coordinates_values)
    if n_coord == 0:
        return []

    xy_world = np.zeros((n_samples * n_coord, 2))

    lines = []

    for iw, w in enumerate(tick_world_coordinates_values):
        subset = slice(iw * n_samples, (iw + 1) * n_samples)
        if coord.coord_index == 0:
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
        if coord.coord_type == "scalar":
            codes = gridline_path_codes(xy_world[subset], pixel[subset])
        else:
            codes = lon_lat_path_codes(
                xy_world[subset], pixel[subset], xy_world_round[subset]
            )
        lines.append((pixel[subset], codes))

    return lines


def lon_lat_path_codes(lon_lat, pixel, lon_lat_check):
    """Return path codes for a spherical curve, breaking at discontinuities.

    lon_lat and lon_lat_check are (N, 2) world coordinates in degrees, before
    and after round-tripping through the transform; pixel is (N, 2) data
    coordinates. Hidden and invalid vertices break the curve.
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
    """Return path codes for a scalar grid line, breaking at invalid pixels."""
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


def label_store(ticks, text):
    """Build a label store from major tick records and their formatted text."""
    labels = SimpleNamespace(
        world=defaultdict(list),
        data=defaultdict(list),
        angle=defaultdict(list),
        tick_angle=defaultdict(list),
        text=defaultdict(list),
        disp=defaultdict(list),
    )
    for tick, label in zip(ticks, text):
        axis = tick["axis"]
        labels.world[axis].append(tick["world"])
        labels.data[axis].append(tick["data"])
        labels.angle[axis].append(tick["angle"])
        labels.tick_angle[axis].append(tick["tick_angle"])
        labels.text[axis].append(label)
        labels.disp[axis].append(tick["axis_displacement"])
    return labels


def sort_using(X, Y):
    return [x for (y, x) in sorted(zip(Y, X))]


def find_start_of_last_number(label, numerical_chars):
    """Find the start of the last number, using the supplied numerical characters."""
    in_number = False
    for j in range(len(label) - 1, -1, -1):
        if in_number:
            if label[j] not in numerical_chars:
                return j + 1
        elif label[j] in numerical_chars:
            in_number = True


def sort_labels(labels):
    """Sort label fields by displacement along each spine, in place."""
    for axis in labels.world:
        labels.world[axis] = sort_using(labels.world[axis], labels.disp[axis])
        labels.data[axis] = sort_using(labels.data[axis], labels.disp[axis])
        labels.angle[axis] = sort_using(labels.angle[axis], labels.disp[axis])
        labels.tick_angle[axis] = sort_using(labels.tick_angle[axis], labels.disp[axis])
        labels.text[axis] = sort_using(labels.text[axis], labels.disp[axis])
        labels.disp[axis] = sort_using(labels.disp[axis], labels.disp[axis])


def simplify_labels(labels, numerical_chars):
    """Shorten sorted labels in place by dropping repeated prefixes.

    numerical_chars includes the minus sign used by the formatter.
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
    """Return label centres as {spine: {index: (x, y)}} in display pixels.

    to_display accepts a data-coordinate tuple. measure(text, x, y) returns
    the width and height of a label at its tick. Empty labels are omitted.
    Padding follows the spine normal, with the tick direction capped at
    60 degrees from that normal to avoid excessive displacement.
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
    """Count overlapping boxes, matching Matplotlib for touching edges and NaNs.

    Use a box's native implementation when available. Otherwise boxes are
    (x0, y0, x1, y1) arrays, with reversed corners normalized first.
    """
    if hasattr(box, "count_overlaps") and all(
        hasattr(other, "get_points") for other in boxes
    ):
        return box.count_overlaps(boxes)

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
    """Yield (spine, index, box) for labels in drawing order.

    extent(spine, index) returns the label box, or None for an empty label.
    The caller can draw immediately after each yield, before the next label
    is measured. Overlap checks include kept labels and existing boxes.
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
    """Return the display-space (x, y, rotation) of an axis label.

    normal_angle points outward from the spine midpoint. Rectangular labels
    sit padding pixels beyond union (the tick-label box), or the spine if
    union is None. Other frames add padding + 1.5 * text_size along the normal.
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
