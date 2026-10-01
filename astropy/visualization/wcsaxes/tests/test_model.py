# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the matplotlib-free WCS interpretation in ``wcsaxes._model``.
"""

import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from astropy import units as u
from astropy.visualization.wcsaxes._model import (
    coord_meta_from_wcs,
    wcs_pixel_to_world,
    wcs_world_to_pixel,
)
from astropy.wcs import WCS


def celestial():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [266.4, -28.9]
    wcs.wcs.cdelt = [-0.002, 0.002]
    wcs.wcs.crpix = [50.5, 40.5]
    # As WCSAxes.reset_wcs does, so that the units are set
    wcs.wcs.set()
    return wcs


def cube():
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "FREQ"]
    wcs.wcs.crval = [266.4, -28.9, 1e9]
    wcs.wcs.cdelt = [-0.002, 0.002, 1e6]
    wcs.wcs.crpix = [50.5, 40.5, 1]
    wcs.wcs.set()
    return wcs


@pytest.mark.parametrize(
    "wcs, slices",
    [
        (celestial(), None),
        (cube(), ("x", "y", 0)),
        (cube(), ("y", "x", 0)),
        (cube(), ("x", 0, 0)),
    ],
)
def test_coord_meta_matches_transform_coord_meta(wcs, slices):
    # The same metadata as the matplotlib function, which adds the positions
    from astropy.visualization.wcsaxes.frame import RectangularFrame, RectangularFrame1D
    from astropy.visualization.wcsaxes.wcsapi import transform_coord_meta_from_wcs

    frame_class = RectangularFrame1D if slices == ("x", 0, 0) else RectangularFrame
    coord_meta, transform_wcs, invert_xy = coord_meta_from_wcs(wcs, slices)
    transform, expected = transform_coord_meta_from_wcs(wcs, frame_class, slices)
    positions = {key for key in expected if key.endswith("_position")}
    assert {
        key: expected[key] for key in expected if key not in positions
    } == coord_meta
    assert positions == {
        "default_axislabel_position",
        "default_ticklabel_position",
        "default_ticks_position",
    }
    # Slicing builds a new SlicedLowLevelWCS on each call
    assert type(transform_wcs) is type(transform.wcs)
    assert (transform_wcs.pixel_n_dim, transform_wcs.world_n_dim) == (
        transform.wcs.pixel_n_dim,
        transform.wcs.world_n_dim,
    )
    assert invert_xy == transform.invert_xy == (slices == ("y", "x", 0))


def test_coord_meta_celestial():
    coord_meta, transform_wcs, invert_xy = coord_meta_from_wcs(celestial())
    assert coord_meta["type"] == ["longitude", "latitude"]
    # The 360 degree default for a longitude is applied by set_coord_type
    assert coord_meta["wrap"] == [None, None]
    assert coord_meta["unit"] == [u.deg, u.deg]
    assert coord_meta["format_unit"] == [u.hourangle, u.deg]
    assert coord_meta["visible"] == [True, True]
    assert coord_meta["default_axis_label"] == ["pos.eq.ra", "pos.eq.dec"]
    assert not invert_xy


def test_wcs_pixel_to_world():
    wcs = celestial()
    pixel = np.array([[0.0, 0.0], [49.5, 39.5], [99.0, 79.0]])
    world = wcs_pixel_to_world(wcs, pixel)
    assert_array_equal(world, np.array(wcs.pixel_to_world_values(*pixel.T)).T)
    assert np.allclose(world[1], [266.4, -28.9])
    # Swapped pixel coordinates
    assert_array_equal(wcs_pixel_to_world(wcs, pixel[:, ::-1], invert_xy=True), world)
    # And back
    assert np.allclose(wcs_world_to_pixel(wcs, world), pixel)
    assert np.allclose(wcs_world_to_pixel(wcs, world, invert_xy=True), pixel[:, ::-1])
    # Empty input keeps its shape
    assert wcs_pixel_to_world(wcs, np.empty((0, 2))).shape == (0, 2)
    assert wcs_world_to_pixel(wcs, np.empty((0, 2))).shape == (0, 2)
    with pytest.raises(ValueError, match="Expected 2 pixel coordinates, got 3"):
        wcs_pixel_to_world(wcs, np.zeros((1, 3)))


def test_model_without_matplotlib():
    # Metadata and transforms where matplotlib cannot be imported, and the
    # ticks of the celestial image laid out from them
    code = """
import sys

sys.modules["matplotlib"] = None

from functools import partial

import numpy as np

from astropy import units as u
from astropy.visualization.wcsaxes import _layout, custom_ucd_coord_meta_mapping
from astropy.visualization.wcsaxes._model import coord_meta_from_wcs, wcs_pixel_to_world
from astropy.visualization.wcsaxes.coordinate_range import find_coordinate_range
from astropy.visualization.wcsaxes.formatter_locator import AngleFormatterLocator
from astropy.visualization.wcsaxes.tests.test_model import celestial

wcs = celestial()
coord_meta, transform_wcs, invert_xy = coord_meta_from_wcs(wcs)
assert coord_meta["format_unit"] == [u.hourangle, u.deg], coord_meta
p2w = partial(wcs_pixel_to_world, transform_wcs, invert_xy=invert_xy)
to_display = lambda xy: 2.0 * np.asarray(xy)
spines = _layout.rectangular_spines((-0.5, 99.5), (-0.5, 79.5), 100, p2w, to_display)
ranges = find_coordinate_range(
    p2w, [-0.5, 99.5, -0.5, 79.5], coord_meta["type"], coord_meta["unit"], coord_meta["wrap"]
)
fl = AngleFormatterLocator(unit=u.deg, format_unit=u.hourangle, spacing=20 * u.arcsec)
spec = _layout.CoordSpec(
    coord_index=0, coord_type="longitude", coord_unit=u.deg, coord_wrap=360 * u.deg,
    coord_scale_to_deg=None, locator=fl.locator, formatter=fl.formatter,
    minor_locator=None, minor_frequency=5,
)
placed = _layout.place_ticks(spec, ranges[0], spines, p2w, to_display, lambda xy: xy / 2.0)
assert "".join(placed.major.axis).startswith("b"), placed.major.axis
assert placed.text[0].startswith("17$"), placed.text
"""
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
