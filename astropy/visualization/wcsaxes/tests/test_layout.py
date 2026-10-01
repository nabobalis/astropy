# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the tick and grid geometry in ``wcsaxes._layout``, which does not
need matplotlib.
"""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.path import Path
from numpy.testing import assert_allclose, assert_array_equal

from astropy.visualization.wcsaxes import conf
from astropy.visualization.wcsaxes._layout import (
    LINETO,
    MOVETO,
    CoordSpec,
    SpineArrays,
    grid_lines,
    gridline_path_codes,
    place_ticks,
    resample_spine,
    spine_normal_angle,
)
from astropy.visualization.wcsaxes.coordinate_range import find_coordinate_range
from astropy.wcs import WCS

# Runs in a subprocess in which matplotlib cannot be imported. It places the
# ticks and samples the grid lines of a TAN image, and of an all-sky AIT image
# whose corners are off the sky, from plain numpy inputs: 100 samples per
# spine, 50 per grid line, a display transform of 2 * data + 10, and a locator
# and a formatter that are plain functions.
NO_MATPLOTLIB = """
import json
import sys

sys.modules["matplotlib"] = None

import numpy as np

from astropy import units as u
from astropy.visualization.wcsaxes._layout import (
    MOVETO,
    CoordSpec,
    SpineArrays,
    grid_lines,
    place_ticks,
    resample_spine,
    spine_normal_angle,
)
from astropy.visualization.wcsaxes.coordinate_range import find_coordinate_range
from astropy.wcs import WCS


class PixelToWorld:
    def __init__(self, wcs):
        self.wcs = wcs

    def transform(self, pixel):
        return np.array(self.wcs.pixel_to_world_values(*pixel.T)).T

    def world_to_pixel(self, world):
        return np.array(self.wcs.world_to_pixel_values(*world.T)).T


def to_display(xy):
    return 2.0 * xy + 10.0


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
    return [f"{v:g}" for v in values.to_value(u.deg)]


def layout(ctype, crval, cdelt, shape, spacing):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ctype
    wcs.wcs.crval = crval
    wcs.wcs.cdelt = cdelt
    wcs.wcs.crpix = [(shape[0] + 1) / 2, (shape[1] + 1) / 2]
    p2w = PixelToWorld(wcs)
    x0, x1, y0, y1 = -0.5, shape[0] - 0.5, -0.5, shape[1] - 0.5
    outlines = {
        "b": [[x0, y0], [x1, y0]],
        "r": [[x1, y0], [x1, y1]],
        "t": [[x1, y1], [x0, y1]],
        "l": [[x0, y1], [x0, y0]],
    }
    spines = {}
    for name, outline in outlines.items():
        data = resample_spine(np.array(outline), 100)
        with np.errstate(invalid="ignore"):
            world = p2w.transform(data)
        spines[name] = SpineArrays(data, world, spine_normal_angle(to_display(data)))
    types = ["longitude", "latitude"]
    wraps = [360 * u.deg, None]
    ranges = find_coordinate_range(p2w, [x0, x1, y0, y1], types, [u.deg] * 2, wraps)
    result = []
    for i in range(2):
        spec = CoordSpec(
            coord_index=i,
            coord_type=types[i],
            coord_unit=u.deg,
            coord_wrap=wraps[i],
            coord_scale_to_deg=None,
            locator=locator(spacing[i]),
            formatter=formatter,
            minor_locator=minor_locator,
            minor_frequency=2,
        )
        placed = place_ticks(
            spec, ranges[i], spines, p2w.transform, to_display, from_display
        )
        lines = grid_lines(spec, ranges, 50, p2w.transform, p2w.world_to_pixel)
        result.append(
            {
                "axis": "".join(placed.major.axis),
                "world": placed.major.world.tolist(),
                "pixel": placed.major.pixel.tolist(),
                "angle": placed.major.angle.tolist(),
                "text": list(placed.text),
                "minor": "".join(placed.minor.axis),
                "grid_moveto": [
                    np.flatnonzero(codes == MOVETO).tolist() for _, codes in lines
                ],
                "grid_pixel": [
                    pixel[[0, 25, -1]].ravel().tolist() for pixel, _ in lines
                ],
            }
        )
    return result


tan = layout(["RA---TAN", "DEC--TAN"], [266.4, -28.9], [-0.002, 0.002], (100, 80), [0.05, 0.05])
ait = layout(["GLON-AIT", "GLAT-AIT"], [0, 0], [-1, 1], (300, 150), [60, 50])
print(json.dumps({"tan": tan, "ait": ait}))
"""

# The output of NO_MATPLOTLIB, rounded to 6 decimals. For each grid line it
# gives the indices of the MOVETO codes, and the first, middle and last vertex.
EXPECTED = {
    "tan": [
        {
            "axis": "bbbbbttttt",
            "world": [266.3, 266.35, 266.4, 266.45, 266.5] * 2,
            "pixel": [
                [93.239531, -0.5],
                [71.369749, -0.5],
                [49.5, -0.5],
                [27.630251, -0.5],
                [5.760469, -0.5],
                [93.30701, 79.5],
                [71.403488, 79.5],
                [49.5, 79.5],
                [27.596512, 79.5],
                [5.69299, 79.5],
            ],
            "angle": [
                -270.048327,
                -270.024164,
                -270.0,
                -269.975836,
                -269.951673,
                -90.048327,
                -90.024164,
                -90.0,
                -89.975836,
                -89.951673,
            ],
            "text": ["266.3", "266.35", "266.4", "266.45", "266.5"] * 2,
            "minor": "b" * 9 + "t" * 9,
            "grid_moveto": [[0]] * 5,
            "grid_pixel": [
                [93.232766, -8.520914, 93.274092, 40.473429, 93.313765, 87.508],
                [71.366372, -8.507039, 71.387035, 40.487274, 71.406872, 87.521816],
                [49.5, -8.502414, 49.5, 40.491889, 49.5, 87.526422],
                [27.633628, -8.507039, 27.612965, 40.487274, 27.593128, 87.521816],
                [5.767234, -8.520914, 5.725908, 40.473429, 5.686235, 87.508],
            ],
        },
        {
            "axis": "rrrlll",
            "world": [-28.95, -28.9, -28.85] * 2,
            "pixel": [
                [99.5, 14.475857],
                [99.5, 39.475913],
                [99.5, 64.475969],
                [-0.5, 14.475857],
                [-0.5, 39.475913],
                [-0.5, 64.475969],
            ],
            "angle": [
                179.94413,
                179.944245,
                179.94436,
                -359.945236,
                -359.945349,
                -359.945462,
            ],
            "text": ["-28.95", "-28.9", "-28.85"] * 2,
            "minor": "r" * 7 + "l" * 7,
            "grid_moveto": [[0]] * 3,
            "grid_pixel": [
                [109.517351, 14.465217, 48.275158, 14.499979, -10.517351, 14.465217],
                [109.546277, 39.465261, 48.274567, 39.499986, -10.546277, 39.465261],
                [109.575204, 64.465306, 48.273977, 64.499992, -10.575204, 64.465306],
            ],
        },
    ],
    "ait": [
        {
            # The longitudes wrap at 360 degrees in the middle of the image,
            # so 0 is found as 360 and listed last on each spine.
            "axis": "bbbbbttttt",
            "world": [60.0, 120.0, 240.0, 300.0, 0.0] * 2,
            "pixel": [
                [137.390567, -0.5],
                [120.082263, -0.5],
                [178.917737, -0.5],
                [161.609433, -0.5],
                [149.5, -0.5],
                [137.390567, 149.5],
                [120.082263, 149.5],
                [178.917737, 149.5],
                [161.609433, 149.5],
                [149.5, 149.5],
            ],
            "angle": [
                -213.521078,
                -196.373103,
                -344.061059,
                -327.152196,
                -270.022216,
                -153.090082,
                -165.554601,
                -14.057813,
                -26.320427,
                -89.970028,
            ],
            "text": ["60", "120", "240", "300", "0"] * 2,
            "minor": "b" * 11 + "t" * 11,
            "grid_moveto": [[0]] * 6,
            "grid_pixel": [
                [149.5, -6.528468, 149.5, 76.336656, 149.5, 155.528468],
                [149.5, -6.528468, 90.206451, 76.401429, 149.5, 155.528468],
                [149.5, -6.528468, 34.957507, 76.620697, 149.5, 155.528468],
                [149.5, -6.528468, -12.473675, 77.09709, 149.5, 155.528468],
                [149.5, -6.528468, 264.042493, 76.620697, 149.5, 155.528468],
                [149.5, -6.528468, 208.793549, 76.401429, 149.5, 155.528468],
            ],
        },
        {
            "axis": "rl",
            "world": [0.0, 0.0],
            "pixel": [[299.5, 74.5], [-0.5, 74.5]],
            "angle": [180.000019, -1.8e-05],
            "text": ["0", "0"],
            "minor": "bbrttl",
            # Each latitude line breaks where it crosses the longitude seam,
            # between samples 24 and 25. The spacing of 50 degrees keeps the
            # lines off the poles, where the round-trip check depends on
            # rounding.
            "grid_moveto": [[0, 25]] * 3,
            "grid_pixel": [
                [149.5, 26.071515, 252.55846, 13.058287, 149.5, 26.071515],
                [149.5, 74.5, 308.938696, 74.5, 149.5, 74.5],
                [149.5, 122.928485, 252.55846, 135.941713, 149.5, 122.928485],
            ],
        },
    ],
}


def test_layout_without_matplotlib():
    proc = subprocess.run(
        [sys.executable, "-c", NO_MATPLOTLIB],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    for case, coords in EXPECTED.items():
        for actual, expected in zip(result[case], coords, strict=True):
            assert actual["axis"] == expected["axis"]
            assert_allclose(actual["world"], expected["world"], rtol=1e-12)
            assert_allclose(actual["pixel"], expected["pixel"], rtol=0, atol=1e-6)
            assert_allclose(actual["angle"], expected["angle"], rtol=0, atol=1e-6)
            assert actual["text"] == expected["text"]
            assert actual["minor"] == expected["minor"]
            assert actual["grid_moveto"] == expected["grid_moveto"]
            assert_allclose(
                actual["grid_pixel"], expected["grid_pixel"], rtol=0, atol=1e-6
            )


def test_path_codes_match_matplotlib():
    assert MOVETO == Path.MOVETO
    assert LINETO == Path.LINETO


def test_gridline_path_codes():
    # A line breaks at an invalid pixel and resumes at the next one.
    pixel = np.array([[0, 0], [1, 1], [np.nan, 2], [3, 3], [4, 4]], dtype=float)
    assert_array_equal(
        gridline_path_codes(np.zeros((5, 2)), pixel),
        [MOVETO, LINETO, MOVETO, MOVETO, LINETO],
    )


def test_place_ticks_matches_wcsaxes():
    # Recompute the ticks of a drawn WCSAxes from plain numpy inputs: the
    # frame outline, the WCS and the affine part of transData. WCSAxes places
    # its ticks with place_ticks too, so this checks that plain inputs give
    # the same ticks, not how the ticks are placed. The test that runs without
    # matplotlib and the figure tests check that.
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["arcsec", "arcsec"]
    wcs.wcs.cdelt = [0.33, 0.33]
    wcs.wcs.crpix = [250, 250]
    wcs.wcs.crval = [-300, 200]
    roll = np.radians(20)
    wcs.wcs.pc = [[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]]

    fig = Figure(figsize=(6, 6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(projection=wcs)
    ax.set_xlim(-0.5, 499.5)
    ax.set_ylim(-0.5, 499.5)
    canvas.draw()
    # Drawing sorts and simplifies the tick labels in place, so place the
    # ticks again, as the next draw would, to compare with the raw labels.
    ax._update_tick_and_label_positions()

    class PixelToWorld:
        def transform(self, pixel):
            return np.array(wcs.pixel_to_world_values(*pixel.T)).T

    p2w = PixelToWorld()
    matrix = ax.transData.get_matrix()
    inverse = np.linalg.inv(matrix)

    def to_display(xy):
        return xy @ matrix[:2, :2].T + matrix[:2, 2]

    def from_display(xy):
        return xy @ inverse[:2, :2].T + inverse[:2, 2]

    x0, x1, y0, y1 = -0.5, 499.5, -0.5, 499.5
    outlines = {
        "b": [[x0, y0], [x1, y0]],
        "r": [[x1, y0], [x1, y1]],
        "t": [[x1, y1], [x0, y1]],
        "l": [[x0, y1], [x0, y0]],
    }
    spines = {}
    for name, outline in outlines.items():
        data = resample_spine(np.array(outline), conf.frame_boundary_samples)
        spines[name] = SpineArrays(
            data, p2w.transform(data), spine_normal_angle(to_display(data))
        )

    coords = [ax.coords[0], ax.coords[1]]
    ranges = find_coordinate_range(
        p2w,
        [x0, x1, y0, y1],
        [coord.coord_type for coord in coords],
        [coord.coord_unit for coord in coords],
        [coord.coord_wrap for coord in coords],
    )

    def flat(per_axis):
        return [value for axis in "brtl" for value in per_axis[axis]]

    for i, coord in enumerate(coords):
        spec = CoordSpec(
            coord_index=i,
            coord_type=coord.coord_type,
            coord_unit=coord.coord_unit,
            coord_wrap=coord.coord_wrap,
            coord_scale_to_deg=None,
            locator=coord.locator,
            formatter=coord.formatter,
            minor_locator=None,
            minor_frequency=5,
        )
        assert spec == coord._layout_spec()
        placed = place_ticks(
            spec, ranges[i], spines, p2w.transform, to_display, from_display
        )
        major = placed.major
        assert len(major.axis) > 0
        assert_array_equal(flat(coord._ticks.world), major.world)
        assert_array_equal(flat(coord._ticks.pixel), major.pixel)
        assert flat(coord._ticklabels.text) == list(placed.text)
        assert_allclose(flat(coord._ticks.angle), major.angle, rtol=0, atol=1e-9)
        assert_allclose(flat(coord._ticklabels.angle), major.normal, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    ("ctype", "cunit"),
    [(["HPLN-TAN", "HPLT-TAN"], ["arcsec", "arcsec"]), (["", ""], ["", ""])],
)
def test_grid_lines_match_wcsaxes(ctype, cunit):
    # Recompute the grid lines of a drawn WCSAxes from plain numpy inputs,
    # for longitude and latitude, and for scalar coordinates. As for the
    # ticks, WCSAxes uses grid_lines too, so this checks the inputs, not the
    # sampling.
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ctype
    wcs.wcs.cunit = cunit
    wcs.wcs.cdelt = [0.33, 0.33]
    wcs.wcs.crpix = [250, 250]
    wcs.wcs.crval = [-300, 200]
    roll = np.radians(20)
    wcs.wcs.pc = [[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]]

    fig = Figure(figsize=(6, 6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(projection=wcs)
    ax.set_xlim(-0.5, 499.5)
    ax.set_ylim(-0.5, 499.5)
    ax.coords.grid()
    canvas.draw()

    def pixel_to_world(pixel):
        return np.array(wcs.pixel_to_world_values(*pixel.T)).T

    def world_to_pixel(world):
        return np.array(wcs.world_to_pixel_values(*world.T)).T

    coords = [ax.coords[0], ax.coords[1]]
    ranges = find_coordinate_range(
        SimpleNamespace(transform=pixel_to_world),
        [-0.5, 499.5, -0.5, 499.5],
        [coord.coord_type for coord in coords],
        [coord.coord_unit for coord in coords],
        [coord.coord_wrap for coord in coords],
    )

    for i, coord in enumerate(coords):
        spec = CoordSpec(
            coord_index=i,
            coord_type=coord.coord_type,
            coord_unit=coord.coord_unit,
            coord_wrap=coord.coord_wrap,
            coord_scale_to_deg=None,
            locator=coord.locator,
            formatter=coord.formatter,
            minor_locator=None,
            minor_frequency=5,
        )
        assert spec == coord._layout_spec()
        lines = grid_lines(
            spec, ranges, conf.grid_samples, pixel_to_world, world_to_pixel
        )
        assert len(lines) == len(coord._grid_lines) > 0
        for (pixel, codes), path in zip(lines, coord._grid_lines, strict=True):
            assert_array_equal(pixel, path.vertices)
            assert_array_equal(codes, path.codes)
