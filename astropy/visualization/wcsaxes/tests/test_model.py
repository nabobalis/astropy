# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the matplotlib-free WCS interpretation in ``wcsaxes._model``.
"""

import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astropy import units as u
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.visualization.wcsaxes._layout import CoordSpec
from astropy.visualization.wcsaxes._model import (
    AxesModel,
    CoordinateModel,
    coord_meta_from_wcs,
    default_positions,
    wcs_pixel_to_world,
    wcs_world_to_pixel,
)
from astropy.visualization.wcsaxes.formatter_locator import (
    AngleFormatterLocator,
    ScalarFormatterLocator,
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

import numpy as np

from astropy import units as u
from astropy.visualization.wcsaxes import custom_ucd_coord_meta_mapping
from astropy.visualization.wcsaxes._model import AxesModel
from astropy.visualization.wcsaxes.tests.test_model import celestial

model = AxesModel.from_wcs(celestial())
assert model["ra"].get_format_unit() == u.hourangle
model["ra"].set_ticks(spacing=20 * u.arcsec)
model["dec"].grid = True
layout = model.layout(
    (-0.5, 99.5), (-0.5, 79.5),
    lambda xy: 2.0 * np.asarray(xy), lambda xy: np.asarray(xy) / 2.0,
    measure=lambda text, x, y: (7.0 * len(text), 12.0), tick_size=5, pad=5, font_size=14,
)
ra, dec = layout.coords
assert "".join(ra.ticks.major.axis).startswith("b"), ra.ticks.major.axis
assert ra.ticks.text[0].startswith("17$"), ra.ticks.text
assert ra.kept and dec.kept and dec.grid, (ra.kept, dec.kept, dec.grid)
assert set(ra.axis_labels) | set(dec.axis_labels) == {"b", "l"}, (ra.axis_labels, dec.axis_labels)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_coordinate_model_coord_type():
    lon = CoordinateModel(0, "longitude", u.deg)
    assert lon.coord_wrap == 360 * u.deg
    assert lon.coord_scale_to_deg is None
    assert isinstance(lon.formatter_locator, AngleFormatterLocator)
    assert lon.get_format_unit() == u.deg

    # An angle in another unit is scaled to degrees for wrapping
    lat = CoordinateModel(1, "latitude", u.arcsec, format_unit=u.arcsec)
    assert lat.coord_wrap is None
    assert lat.coord_scale_to_deg == 1 / 3600
    with pytest.raises(NotImplementedError, match="coord_wrap is not yet supported"):
        CoordinateModel(1, "latitude", u.deg, coord_wrap=180 * u.deg)

    with pytest.warns(AstropyDeprecationWarning, match="as a number is deprecated"):
        wrapped = CoordinateModel(0, "longitude", u.deg, coord_wrap=180)
    assert wrapped.coord_wrap == 180 * u.deg

    scalar = CoordinateModel(0, "scalar", u.m)
    assert isinstance(scalar.formatter_locator, ScalarFormatterLocator)
    assert scalar.default_axislabel() == " [$\\mathrm{m}$]"
    assert scalar.default_axislabel("unicode") == " [m]"
    assert lon.default_axislabel() == ""

    with pytest.raises(ValueError, match="coord_type should be one of"):
        CoordinateModel(0, "spam", u.deg)


def test_coordinate_model_ticks_and_format():
    lon = CoordinateModel(0, "longitude", u.deg, default_label="Longitude")
    with pytest.raises(ValueError, match="At most one of"):
        lon.set_ticks(spacing=1 * u.deg, number=5)
    lon.set_ticks(spacing=30 * u.arcmin)
    assert lon.formatter_locator.spacing == 30 * u.arcmin
    assert lon.spec() == CoordSpec(
        coord_index=0,
        coord_type="longitude",
        coord_unit=u.deg,
        coord_wrap=360 * u.deg,
        coord_scale_to_deg=None,
        locator=lon.locator,
        formatter=lon.formatter,
        minor_locator=None,
        minor_frequency=5,
    )
    lon.display_minor_ticks = True
    assert lon.spec().minor_locator == lon.formatter_locator.minor_locator

    # Nothing has been laid out yet, so no spacing and no readout
    assert lon.format_coord(10.25) == ""
    lon.spacing = 30 * u.arcmin
    expected = lon.formatter([10.25] * u.deg, spacing=lon.spacing, format="ascii")[0]
    assert lon.format_coord(10.25, format="ascii") == expected
    # A longitude wraps, and a custom formatter is used as it is
    assert lon.format_coord(370.25, format="ascii") == expected
    lon.set_major_formatter(
        lambda values, spacing=None, format=None: ["x"] * len(values)
    )
    assert lon.format_coord(10.25) == "x"
    with pytest.raises(TypeError, match="formatter should be a string"):
        lon.set_major_formatter(3)

    lat = CoordinateModel(1, "scalar", u.m)
    with pytest.raises(TypeError, match="only be specified for angle"):
        lat.set_separator(":")


def test_axes_model_from_wcs():
    model = AxesModel.from_wcs(celestial())
    assert len(model) == 2
    assert [c.coord_index for c in model] == [0, 1]
    assert model["ra"] is model[0] is model["pos.eq.ra"]
    assert model["DEC"] is model[1]
    assert "ra" in model and 1 in model and 2 not in model
    assert model[0].coord_type == "longitude"
    assert model[0].coord_wrap == 360 * u.deg
    assert model[0].get_format_unit() == u.hourangle
    assert model[1].default_label == "pos.eq.dec"
    pixel = np.array([[49.5, 39.5]])
    assert np.allclose(model.pixel_to_world(pixel), [[266.4, -28.9]])
    assert np.allclose(model.world_to_pixel(model.pixel_to_world(pixel)), pixel)

    # A sliced cube: the third coordinate is not shown, and x and y swap
    sliced = AxesModel.from_wcs(cube(), slices=("y", "x", 0))
    assert [c.coord_index for c in sliced] == [0, 1, None]
    assert sliced["freq"].coord_type == "scalar"
    assert np.allclose(
        sliced.pixel_to_world(np.array([[39.5, 49.5]])), [[266.4, -28.9]]
    )


def test_helper_delegates_to_model():
    # CoordinateHelper and CoordinatesMap keep their settings in the model
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(projection=celestial())
    ra, dec = ax.coords
    assert ra._model is ax.coords._model[0]
    assert ax.coords["ra"] is ra
    assert ra.coord_type == "longitude" and ra.coord_wrap == 360 * u.deg
    assert ra._layout_spec() == ra._model.spec()

    ra.set_ticks(number=4)
    assert ra._model.formatter_locator.number == 4
    ra.set_format_unit(u.deg)
    assert ra.get_format_unit() == ra._model.get_format_unit() == u.deg
    ra.set_minor_frequency(3)
    ra.display_minor_ticks(True)
    assert ra._model.minor_frequency == 3 and ra._model.display_minor_ticks
    assert ra._model.spec().minor_frequency == 3
    dec.set_coord_type("scalar")
    assert dec._model.coord_type == "scalar"
    assert isinstance(dec._formatter_locator, ScalarFormatterLocator)
    with pytest.warns(AstropyDeprecationWarning):
        dec.coord_type = "latitude"
    assert dec._model.coord_type == "latitude"

    # The cursor readout formats through the model once ticks are laid out
    assert ra.format_coord(266.4) == ""
    ra.get_ticks()
    assert ra.format_coord(266.4, format="ascii") == ra._model.format_coord(
        266.4, format="ascii"
    )
    assert (
        ra.format_coord(266.4, format="ascii")
        == ra.formatter([266.4] * u.deg, spacing=ra._model.spacing, format="ascii")[0]
    )


def test_default_positions():
    def meta(types, visible):
        return {"type": types, "visible": visible}

    both = default_positions(
        meta(["longitude", "latitude"], [True, True]), "rectangular", "brtl"
    )
    assert both["default_ticks_position"] == ["brtl", "brtl"]
    assert (
        both["default_ticklabel_position"]
        == both["default_axislabel_position"]
        == ["#", "#"]
    )
    # A sliced cube shows two of three coordinates, so ticks go on all spines
    cube = default_positions(
        meta(["longitude", "latitude", "scalar"], [True, True, False]),
        "rectangular",
        "brtl",
    )
    assert cube["default_ticks_position"] == ["brtl", "brtl", ""]
    assert cube["default_ticklabel_position"] == ["#", "#", ""]
    one = default_positions(meta(["scalar"], [True]), "rectangular1d", "bt")
    assert one["default_ticks_position"] == ["bt"]
    ell = default_positions(
        meta(["longitude", "latitude"], [True, True]), "elliptical", "chv"
    )
    assert (
        ell["default_ticks_position"] == ell["default_axislabel_position"] == ["h", "c"]
    )
    custom = default_positions(
        meta(["longitude", "latitude"], [True, True]), "custom", "abcdef"
    )
    assert custom["default_ticklabel_position"] == ["abcdef", "abcdef"]


def test_coordinate_model_positions():
    lon = CoordinateModel(0, "longitude", u.deg, spine_names="brtl")
    assert lon.get_ticks_position() == ["b", "r", "t", "l"]
    lon.set_ticklabel_position("b#")
    assert lon.get_ticklabel_position() == ["b", "#"]
    with pytest.warns(AstropyDeprecationWarning, match="Ignoring unrecognized"):
        lon.set_axislabel_position(["b", "x"])
    assert lon.get_axislabel_position() == ["b"]
    assert lon.tick_count("b") == 0
    assert lon.get_axislabel() == ""
    lon.axislabel = "RA"
    assert lon.get_axislabel() == "RA"
    lon.axislabel_minpad = {"b": 2}
    assert lon._minpad("b") == 2


def test_axes_model_from_wcs_positions():
    model = AxesModel.from_wcs(celestial())
    assert model.frame == "rectangular"
    assert model["ra"].get_ticks_position() == ["b", "r", "t", "l"]
    assert model["ra"].get_ticklabel_position() == ["#"]
    one_d = WCS(naxis=1)
    one_d.wcs.ctype = ["FREQ"]
    model = AxesModel.from_wcs(one_d)
    assert model.frame == "rectangular1d"
    assert model[0].get_ticks_position() == ["b", "t"]


def layout_of_axes(ax, model, **kwargs):
    """The model's layout of the view of a WCSAxes, in its display pixels."""
    return model.layout(
        ax.get_xlim(),
        ax.get_ylim(),
        ax.transData.transform,
        ax.transData.inverted().transform,
        **kwargs,
    )


@pytest.mark.parametrize("slices", [None, ("x", "y", 0), ("y", "x", 0), ("x", 0, 0)])
def test_layout_matches_wcsaxes(ignore_matplotlibrc, slices):
    # Without measuring text, layout() places the ticks WCSAxes places, and
    # assigns the same spines automatically
    from matplotlib.figure import Figure

    wcs = celestial() if slices is None else cube()
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(projection=wcs, slices=slices)
    ax.set_xlim(-0.5, 99.5)
    ax.set_ylim(-0.5, 79.5)
    ax.coords[0].display_minor_ticks(True)
    model = AxesModel.from_wcs(wcs, slices=slices)
    model[0].display_minor_ticks = True

    layout = layout_of_axes(ax, model)
    shown = [coord for coord in ax.coords if coord.coord_index is not None]
    assert [c.coord.coord_index for c in layout.coords] == [
        c.coord_index for c in shown
    ]
    for helper, lay in zip(shown, layout.coords):
        ticks = helper.get_ticks()
        for spine in ticks:
            on_spine = np.array(lay.ticks.major.axis) == spine
            assert_array_equal(lay.ticks.major.pixel[on_spine], ticks[spine])
        minor = helper.get_ticks(minor=True)
        assert sum(len(v) for v in minor.values()) == len(lay.ticks.minor.axis)
        assert helper.get_ticklabels() == {
            spine: [
                t for t, a in zip(lay.ticks.text, lay.ticks.major.axis) if a == spine
            ]
            for spine in ticks
        }
        # The spines assigned automatically agree
        assert helper.get_ticklabel_position() == lay.coord.get_ticklabel_position()
        assert helper.get_axislabel_position() == lay.coord.get_axislabel_position()
        assert lay.anchors is None and lay.kept is None and lay.axis_labels == {}
        assert lay.axis_label_text == helper.get_axislabel()
        assert lay.coord.format_coord(1.0, format="ascii") == helper.format_coord(
            1.0, format="ascii"
        )


def test_layout_with_text(ignore_matplotlibrc):
    # Text of 7 by 12 pixels: labels are anchored, overlaps dropped, and the
    # axis labels placed beyond the tick labels on the spines assigned
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(projection=celestial())
    ax.set_xlim(-0.5, 99.5)
    ax.set_ylim(-0.5, 79.5)
    model = AxesModel.from_wcs(celestial())
    model["ra"].axislabel = "Right ascension"
    model["ra"].grid = True
    model["dec"].exclude_overlapping = True

    def measure(text, x, y):
        return 7.0 * len(text), 12.0

    layout = layout_of_axes(
        ax, model, measure=measure, tick_size=5, pad=5, font_size=14
    )
    ra, dec = layout.coords
    assert ra.axis_label_text == "Right ascension"
    assert dec.axis_label_text == "pos.eq.dec"
    # One spine each, assigned automatically: RA runs along x here
    (ra_spine,) = [a for a in ra.coord.get_ticklabel_position() if a != "#"]
    (dec_spine,) = [a for a in dec.coord.get_ticklabel_position() if a != "#"]
    assert {ra_spine, dec_spine} == {"b", "l"}
    assert list(ra.anchors) == [ra_spine] and list(dec.anchors) == [dec_spine]
    # Every label kept has a box 12 pixels high centred on its anchor
    for lay, spine in [(ra, ra_spine), (dec, dec_spine)]:
        assert len(lay.kept) > 1
        for axis, i, (x0, y0, x1, y1) in lay.kept:
            assert axis == spine
            assert_allclose(
                (y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2), (12, *lay.anchors[axis][i])
            )
    # The RA labels are 10 pixels below the bottom spine, 5 of tick and 5 of pad
    bottom = ax.transData.transform([[0, -0.5]])[0, 1]
    assert_allclose([y for _, y in ra.anchors["b"].values()], bottom - 10 - 6)
    # Axis labels on the assigned spines only, pushed out past the labels
    assert list(ra.axis_labels) == [ra_spine] and list(dec.axis_labels) == [dec_spine]
    x, y, rotation = ra.axis_labels["b"]
    assert rotation % 360 == 0 and y < bottom - 10 - 12
    assert ra.grid and not dec.grid
    # With the 'ticks' rule and no ticks on a spine, no axis label there
    model["ra"].axislabel_visibility_rule = "ticks"
    model["ra"].set_ticks_position("t")
    model["ra"].set_axislabel_position("b")
    layout = layout_of_axes(
        ax, model, measure=measure, tick_size=5, pad=5, font_size=14
    )
    assert layout.coords[0].axis_labels == {}


def test_text_format(ignore_matplotlibrc):
    # With text_format 'unicode', the labels, the readout and the default
    # axis labels have no mathtext
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(projection=celestial())
    ax.set_xlim(-0.5, 99.5)
    ax.set_ylim(-0.5, 79.5)
    model = AxesModel.from_wcs(celestial())
    assert model["ra"].spec().formatter == model["ra"].formatter

    ra = layout_of_axes(ax, model).coords[0]
    assert all("$" in text for text in ra.ticks.text)

    model.text_format = "unicode"
    ra = layout_of_axes(ax, model).coords[0]
    assert ra.ticks.text and all("$" not in text for text in ra.ticks.text)
    assert "ʰ" in "".join(ra.ticks.text)
    assert "$" not in model["ra"].format_coord(266.4, format="unicode")

    # A custom formatter is used as it is
    model["ra"].set_major_formatter(lambda values, spacing=None: ["x"] * len(values))
    assert layout_of_axes(ax, model).coords[0].ticks.text == ["x"] * len(ra.ticks.text)

    one_d = WCS(naxis=1)
    one_d.wcs.ctype = ["FREQ"]
    one_d.wcs.cunit = ["Hz"]
    one_d.wcs.set()
    ax = fig.add_subplot(projection=one_d)
    ax.set_xlim(-0.5, 99.5)
    model = AxesModel.from_wcs(one_d)
    assert (
        layout_of_axes(ax, model).coords[0].axis_label_text
        == "em.freq [$\\mathrm{Hz}$]"
    )
    model.text_format = "unicode"
    assert layout_of_axes(ax, model).coords[0].axis_label_text == "em.freq [Hz]"
