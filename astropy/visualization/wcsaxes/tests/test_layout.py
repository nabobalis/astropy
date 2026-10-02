# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the tick, grid and label geometry in ``wcsaxes._layout``.

WCSAxes draws with ``_layout``, so the figure tests check that the code still
does what it did before it moved. The tests here lay out two images from
plain numpy inputs, without a figure, and check results that can be worked
out by hand, plus branches that no figure test reaches. matplotlib is only
imported inside the tests that compare with it, so that
test_layout_without_matplotlib can import this module where matplotlib
cannot be imported.
"""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astropy import units as u
from astropy.utils.compat.optional_deps import HAS_MATPLOTLIB
from astropy.visualization.wcsaxes._layout import (
    LINETO,
    MOVETO,
    anchor_tick_labels,
    axis_label_position,
    count_overlaps,
    grid_lines,
    gridline_path_codes,
    keep_tick_labels,
    label_store,
    lon_lat_path_codes,
    place_ticks,
    rectangular_spines,
    resample_spine,
    simplify_labels,
    sort_labels,
    spine_midpoint,
    spine_normal_angle,
)
from astropy.visualization.wcsaxes.coordinate_range import find_coordinate_range
from astropy.wcs import WCS

requires_matplotlib = pytest.mark.skipif(
    not HAS_MATPLOTLIB, reason="Matplotlib comparison"
)


class PixelToWorld:
    def __init__(self, wcs):
        self.wcs = wcs

    def transform(self, pixel):
        return np.array(self.wcs.pixel_to_world_values(*pixel.T)).T

    def world_to_pixel(self, world):
        return np.array(self.wcs.world_to_pixel_values(*world.T)).T


def to_display(xy):
    return 2.0 * np.asarray(xy) + 10.0


def from_display(xy):
    return (xy - 10.0) / 2.0


def locator(spacing):
    def locate(vmin, vmax):
        n = np.arange(np.ceil(vmin / spacing), np.floor(vmax / spacing) + 1)
        return n * spacing * u.deg, spacing * u.deg

    return locate


def minor_locator(spacing, frequency, vmin, vmax):
    return locator(spacing.to_value(u.deg) / frequency)(vmin, vmax)[0]


def formatter(values, spacing):
    # Degrees and arcminutes, so that simplify_labels has a prefix to drop
    return [f"{int(v)}d{round(abs(v) % 1 * 60):02d}m" for v in values.to_value(u.deg)]


def measure(text, x, y):
    return 7.0 * len(text), 12.0


def layout(ctype, crval, cdelt, shape, spacing, rectangular):
    """
    Lay out an image as WCSAxes would, with nothing but _layout.

    The frame is the outline of the image, 100 samples per spine, and display
    pixels are 2 * data + 10. Text is 7 pixels wide per character and 12
    high. Tick labels are on all four spines, 8 pixels from their ticks, and
    labels that overlap are dropped. Before the first coordinate, a box covers
    display pixels 100 to 110 in x and -10 to 0 in y. Axis labels have a
    padding of 12 and a font size of 10, and go beyond the tick labels kept
    so far if the frame is taken as rectangular.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ctype
    wcs.wcs.crval = crval
    wcs.wcs.cdelt = cdelt
    wcs.wcs.crpix = [(shape[0] + 1) / 2, (shape[1] + 1) / 2]
    p2w = PixelToWorld(wcs)
    x0, x1, y0, y1 = -0.5, shape[0] - 0.5, -0.5, shape[1] - 0.5
    spines = rectangular_spines((x0, x1), (y0, y1), 100, p2w.transform, to_display)
    types = ["longitude", "latitude"]
    wraps = [360 * u.deg, None]
    ranges = find_coordinate_range(p2w, [x0, x1, y0, y1], types, [u.deg] * 2, wraps)
    result = []
    existing = [(100.0, -10.0, 110.0, 0.0)]
    for i in range(2):
        coord = SimpleNamespace(
            coord_index=i,
            coord_type=types[i],
            coord_unit=u.deg,
            coord_wrap=wraps[i],
            _coord_scale_to_deg=None,
            locator=locator(spacing[i]),
        )
        major, minor, tick_spacing, values = place_ticks(
            coord,
            ranges[i],
            spines,
            p2w.transform,
            to_display,
            from_display,
            minor_locator=minor_locator,
            minor_frequency=2,
        )
        lines = grid_lines(coord, ranges, 50, p2w.transform, p2w.world_to_pixel)

        labels = label_store(major, formatter(u.Quantity(values), tick_spacing))
        sort_labels(labels)
        simplify_labels(labels, "0123456789.+-")
        anchors = anchor_tick_labels(labels, list("brtl"), to_display, 8.0, measure)

        def extent(axis, i):
            if labels.text[axis][i] == "":
                return None
            x, y = anchors[axis][i]
            width, height = measure(labels.text[axis][i], x, y)
            return (x - width / 2, y - height / 2, x + width / 2, y + height / 2)

        kept = list(keep_tick_labels(labels, list("brtl"), extent, True, existing))
        existing += [box for _, _, box in kept]

        boxes = np.array(existing[1:])
        union = (*boxes[:, :2].min(axis=0), *boxes[:, 2:].max(axis=0))
        axis_label = {}
        for axis in labels.text:
            # As Spine._halfway_x_y_angle, on the two corners of the spine
            pixel = to_display(spines[axis].data[[0, -1]])
            x, y, normal = spine_midpoint(pixel, spine_normal_angle(pixel))
            axis_label[axis] = axis_label_position(
                axis, x, y, normal, 12.0, 10.0, rectangular, union
            )

        result.append(
            SimpleNamespace(
                major=major,
                minor=minor,
                lines=lines,
                labels=labels,
                anchors=anchors,
                kept=" ".join(f"{axis}{i}" for axis, i, _ in kept),
                axis_label=axis_label,
            )
        )
    return result


def moveto(lines):
    return [np.flatnonzero(codes == MOVETO).tolist() for _, codes in lines]


def test_tan_layout():
    # A 100 by 80 TAN image of 0.002 degree pixels, with ticks every 0.05
    # degrees, or 3 arcminutes. In display pixels the spines are at x = 9 and
    # 209, and y = 9 and 169.
    ra, dec = layout(
        ["RA---TAN", "DEC--TAN"],
        [266.4, -28.9],
        [-0.002, 0.002],
        (100, 80),
        [0.05, 0.05],
        True,
    )
    major = ra.major
    assert "".join(tick["axis"] for tick in major) == "bbbbbttttt"
    assert_allclose(
        [tick["world"] for tick in major], [266.3, 266.35, 266.4, 266.45, 266.5] * 2
    )
    # The reference value is at the centre of the image.
    assert_allclose([major[2]["data"], major[7]["data"]], [[49.5, -0.5], [49.5, 79.5]])
    # Each tick points into the image, to within the tilt of the RA lines.
    assert_allclose(
        [tick["tick_angle"] % 360 for tick in major], [90] * 5 + [270] * 5, atol=0.1
    )
    # Minor ticks every 0.025 degrees, and grid lines that do not break
    assert "".join(tick["axis"] for tick in ra.minor) == "b" * 9 + "t" * 9
    assert moveto(ra.lines) == [[0]] * 5
    # Labels are sorted along each spine, and repeated degrees are dropped.
    assert ra.labels.text == {
        "b": ["266d30m", "27m", "24m", "21m", "18m"],
        "t": ["266d18m", "21m", "24m", "27m", "30m"],
    }
    # Label centres are 8 pixels and half a label height away from the spine.
    centres = [xy for axis in "bt" for xy in ra.anchors[axis].values()]
    assert_allclose([y for _, y in centres], [9 - 8 - 6] * 5 + [169 + 8 + 6] * 5)
    # The box drawn before covers the middle label on the bottom spine.
    assert ra.kept == "b0 b1 b3 b4 t0 t1 t2 t3 t4"
    # Axis labels go 12 pixels beyond the tick labels, and the one on the
    # bottom spine is turned upright.
    assert_allclose(ra.axis_label["b"], (109, 9 - 8 - 12 - 12, 360))
    assert_allclose(ra.axis_label["t"], (109, 169 + 8 + 12 + 12, 0))

    assert "".join(tick["axis"] for tick in dec.major) == "rrrlll"
    assert dec.labels.text == {
        "r": ["-28d57m", "54m", "51m"],
        "l": ["-28d51m", "54m", "57m"],
    }
    # The widest label is 49 pixels wide.
    assert_allclose(dec.axis_label["r"], (209 + 8 + 49 + 12, 89, 270))
    assert_allclose(dec.axis_label["l"], (9 - 8 - 49 - 12, 89, 90))


def test_allsky_layout():
    # A 300 by 150 AIT image of the whole sky, at 1 degree per pixel, whose
    # corners are off the sky. Longitude 0 is at the centre, and longitudes
    # wrap at 360 degrees. In display pixels the spines are at x = 9 and 609,
    # and y = 9 and 309.
    lon, lat = layout(
        ["GLON-AIT", "GLAT-AIT"], [0, 0], [-1, 1], (300, 150), [60, 50], False
    )
    # 0 is found as 360, so it comes after 300 on each spine.
    assert "".join(tick["axis"] for tick in lon.major) == "bbbbbttttt"
    assert_allclose([tick["world"] for tick in lon.major], [60, 120, 240, 300, 0] * 2)
    # The labels are close together, so every other one is dropped.
    assert lon.kept == "b0 b2 b4 t0 t2 t4"
    # On a frame that is not rectangular, axis labels move out from the
    # middle of the spine by the padding plus 1.5 times the font size.
    assert_allclose(lon.axis_label["b"], (309, 9 - 27, 360))
    assert_allclose(lon.axis_label["t"], (309, 309 + 27, 0))
    assert_allclose(lat.axis_label["r"], (609 + 27, 159, 270))

    assert "".join(tick["axis"] for tick in lat.major) == "rl"
    assert_allclose([tick["world"] for tick in lat.major], [0, 0])
    # Each latitude line breaks where it crosses the longitude seam, between
    # samples 24 and 25 of 50.
    assert moveto(lat.lines) == [[0, 25]] * 3


def test_layout_without_matplotlib():
    # Check the import contract and collect all pure layout tests, not just
    # imports: conftest must leave these tests runnable without matplotlib.
    code = """
import sys

sys.modules["matplotlib"] = None

import pytest
from astropy import units as u
from astropy.visualization import wcsaxes
from astropy.visualization.wcsaxes import _layout, conf
from astropy.visualization.wcsaxes.formatter_locator import (
    AngleFormatterLocator, ScalarFormatterLocator,
)
from astropy.visualization.wcsaxes.tests import test_layout

for name in (
    "CoordinateHelper", "CoordinatesMap", "Quadrangle", "SphericalCircle",
    "WCSAxes", "WCSAxesSubplot", "add_beam", "add_scalebar",
    "custom_ucd_coord_meta_mapping",
):
    with pytest.raises(ModuleNotFoundError) as exc:
        getattr(wcsaxes, name)
    assert exc.value.name == "matplotlib"
    assert str(exc.value) == f"astropy.visualization.wcsaxes.{name} requires matplotlib"
with pytest.raises(AttributeError):
    wcsaxes.missing

for fl, limits, expected in (
    (AngleFormatterLocator(number=5), (-2, 2), ["−2°", "−1°", "0°", "1°", "2°"]),
    (ScalarFormatterLocator(number=4, unit=u.m), (-1, 1), ["−0.5", "0.0", "0.5"]),
):
    values, spacing = fl.locator(*limits)
    assert fl.formatter(values, spacing=spacing) == expected

raise SystemExit(pytest.main([
    test_layout.__file__, "-q",
    "-k", "not without_matplotlib",
]))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "passed" in proc.stdout, proc.stdout


@requires_matplotlib
def test_rectangular_spines():
    # The same spines as RectangularFrame.sample gives WCSAxes
    from matplotlib.figure import Figure

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = [-0.01, 0.01]
    ax = Figure().add_subplot(projection=wcs)
    ax.set_xlim(-0.5, 99.5)
    ax.set_ylim(-0.5, 49.5)
    frame = ax.coords.frame
    expected = frame.sample(20)
    spines = rectangular_spines(
        ax.get_xlim(),
        ax.get_ylim(),
        20,
        frame.transform.transform,
        ax.transData.transform,
    )
    assert list(spines) == list(expected) == ["b", "r", "t", "l"]
    for axis in spines:
        assert_array_equal(spines[axis].data, expected[axis].data)
        assert_array_equal(spines[axis].world, expected[axis].world)
        assert_array_equal(spines[axis].normal_angle, expected[axis].normal_angle)


def test_resample_spine():
    line = np.array([[0.0, 0.0], [4.0, 8.0]])
    assert_allclose(resample_spine(line, 5), [[0, 0], [1, 2], [2, 4], [3, 6], [4, 8]])
    empty = np.zeros((0, 2))
    assert resample_spine(empty, 5) is empty


@requires_matplotlib
def test_path_codes_match_matplotlib():
    from matplotlib.path import Path

    assert MOVETO == Path.MOVETO
    assert LINETO == Path.LINETO


def test_gridline_path_codes():
    # A line breaks at an invalid pixel and resumes at the next one.
    pixel = np.array([[0, 0], [1, 1], [np.nan, 2], [3, 3], [4, 4]], dtype=float)
    assert_array_equal(
        gridline_path_codes(np.zeros((5, 2)), pixel),
        [MOVETO, LINETO, MOVETO, MOVETO, LINETO],
    )


def test_lon_lat_path_codes_first_step():
    # A first step 100 times longer than the second breaks the line there.
    lon_lat = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pixel = np.array([[0.0, 0.0], [100.0, 0.0], [101.0, 0.0]])
    assert_array_equal(
        lon_lat_path_codes(lon_lat, pixel, lon_lat), [MOVETO, MOVETO, LINETO]
    )


@requires_matplotlib
def test_count_overlaps_matches_matplotlib():
    from matplotlib.transforms import Bbox

    # Corners on a small integer grid give many boxes that only touch, and
    # many whose corners are reversed. One case in three has a NaN corner.
    rng = np.random.default_rng(0)
    for trial in range(2000):
        corners = rng.integers(0, 6, size=(8, 2, 2)).astype(float)
        if trial % 3 == 0:
            corners[tuple(rng.integers(0, (8, 2, 2)))] = np.nan
        box, *boxes = [Bbox(c) for c in corners]
        expected = box.count_overlaps(boxes)
        assert count_overlaps(box, boxes) == expected
        assert count_overlaps(box, corners[1:].reshape(-1, 4)) == expected
        assert (
            count_overlaps(corners[0].ravel(), corners[1:].reshape(-1, 4)) == expected
        )
    assert count_overlaps(Bbox.unit(), []) == Bbox.unit().count_overlaps([]) == 0


def test_anchor_tick_labels():
    # 40 by 10 pixel labels at the same tick, with a pad of 5. For the first,
    # the spine normal and the tick are both 30 degrees from +x: the label
    # moves away from the tick along the normal, which leaves the label box
    # through its long side, until the box clears the tick by the pad. The
    # second has no tick direction, so it takes the normal and lands on the
    # first. The third has its tick along the spine: it also moves along the
    # spine, by twice the pad, which is as far as the cap at 60 degrees from
    # the normal allows. An empty label is left out.
    labels = SimpleNamespace(
        world={"a": [0] * 4},
        data={"a": [(100.0, 50.0)] * 4},
        angle={"a": [30.0, 30.0, 90.0, 30.0]},
        tick_angle={"a": [30.0, np.nan, 0.0, 30.0]},
        text={"a": ["label"] * 3 + [""]},
        disp={"a": [0, 1, 2, 3]},
    )
    xy = anchor_tick_labels(labels, ["a"], np.asarray, 5.0, lambda *_: (40.0, 10.0))
    assert list(xy["a"]) == [0, 1, 2]
    cos, sin = np.cos(np.radians(30)), np.sin(np.radians(30))
    assert_allclose(
        xy["a"][0],
        (100 - 5 * cos / sin - 5 * cos, 50 - 5 - 5 * sin),
        rtol=0,
        atol=1e-12,
    )
    assert xy["a"][1] == xy["a"][0]
    assert_allclose(xy["a"][2], (100 - 2 * 5, 50 - 5 - 5), rtol=0, atol=1e-12)


def test_simplify_labels():
    # A label that only adds to the one before it keeps what it adds, from
    # the number where it starts. A label left as empty mathtext is emptied.
    labels = SimpleNamespace(
        world={"a": [0, 1], "b": [0, 1]},
        text={"a": ["1h", "1h30m"], "b": ["$$x", "$$"]},
    )
    simplify_labels(labels, "0123456789.+-")
    assert labels.text == {"a": ["1h", "30m"], "b": ["$$x", ""]}
