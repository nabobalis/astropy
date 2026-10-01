"""
WCSAxes-style ticks, tick labels, grid lines and axis labels on a bqplot
image figure, laid out by astropy.visualization.wcsaxes._layout without a
matplotlib Figure, Axes or renderer.

    python wcs_bqplot.py [OUTDIR]     write OUTDIR/<case>.html for each case

In a notebook, ``figure("hpc")`` returns a live figure: drag to pan, scroll to
zoom. Every change of the image scales reruns the layout and updates the marks.
bqplot's own axes are not used; the frame, ticks and grid are Lines marks and
the text is Label marks, all in display pixels.
"""

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import bqplot as bq
import ipywidgets as widgets
import matplotlib  # only to find the DejaVu Sans file that matplotlib ships
import numpy as np
from astropy import units as u
from astropy.visualization.wcsaxes import _layout, conf
from astropy.visualization.wcsaxes.coordinate_range import find_coordinate_range
from astropy.visualization.wcsaxes.formatter_locator import AngleFormatterLocator
from astropy.wcs import WCS
from PIL import Image, ImageFont

# Matplotlib's defaults at 100 dpi, in display pixels, so that the numbers can
# be compared with WCSAxes (compare.py). Ticks point out, as xtick.direction.
DPI = 100
PT = DPI / 72
TICK_SIZE, TICK_PAD, FONT_SIZE = 3.5 * PT, 3.5 * PT, 10 * PT
MARGIN = {"top": 25, "bottom": 75, "left": 105, "right": 25}
# As TickLabels.simplify_labels with matplotlib's default rcParams
NUMERICAL_CHARS = "0123456789.+\N{MINUS SIGN}"
# The formatter writes hour angles in mathtext; the browser gets Unicode
MATHTEXT = {r"$\mathregular{^h}$": "ʰ", r"$\mathregular{^m}$": "ᵐ", r"$\mathregular{^s}$": "ˢ"}
# The kernel cannot ask the browser how wide a text is. Pillow measures it in
# DejaVu Sans, matplotlib's default font, which gives the same width as
# matplotlib for plain text; the height is the font size.
FONT = ImageFont.truetype(f"{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans.ttf", FONT_SIZE)


def plain(text):
    for mathtext, unicode in MATHTEXT.items():
        text = text.replace(mathtext, unicode)
    return text


def measure(text, x=None, y=None):
    return FONT.getlength(plain(text)), FONT_SIZE


class PixelToWorld:
    """The APE 14 calls, with the .transform/.inverted() that the core and
    find_coordinate_range use."""

    def __init__(self, wcs, inverse=False):
        self.wcs, self.inverse = wcs, inverse

    def transform(self, xy):
        f = self.wcs.world_to_pixel_values if self.inverse else self.wcs.pixel_to_world_values
        return np.array(f(*np.asarray(xy).T)).T

    def inverted(self):
        return PixelToWorld(self.wcs, not self.inverse)


class View:
    """Data pixels to and from display pixels (origin lower left, y up)."""

    def __init__(self, xlim, ylim, size):
        self.origin = np.array([xlim[0], ylim[0]], dtype=float)
        self.scale = np.array(size, dtype=float) / [xlim[1] - xlim[0], ylim[1] - ylim[0]]

    def to_display(self, xy):
        return (np.asarray(xy, dtype=float) - self.origin) * self.scale

    def from_display(self, xy):
        return np.asarray(xy, dtype=float) / self.scale + self.origin


def make_case(title, ctype, cunit, crval, cdelt, shape, roll, coords, brightness):
    # The same WCSs as the Qt demo
    wcs = WCS(naxis=2)
    wcs.wcs.ctype, wcs.wcs.cunit, wcs.wcs.crval, wcs.wcs.cdelt = ctype, cunit, crval, cdelt
    ny, nx = shape
    wcs.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    wcs.wcs.pc = [[c, -s], [s, c]]
    # Until wcslib's set() runs, world_axis_units says arcsec for the
    # helioprojective case, but pixel_to_world_values returns degrees
    wcs.wcs.set()
    # The frame is the image scaled to fit 720 x 480 display pixels
    zoom = min(720 / nx, 480 / ny)
    return SimpleNamespace(
        title=title, wcs=wcs, shape=shape, coords=coords, brightness=brightness,
        size=(round(nx * zoom), round(ny * zoom)),
    )


def coord(type, wrap, format_unit, label, spine):
    """What WCSAxes reads from the WCS for one coordinate, and where its labels go."""
    return SimpleNamespace(type=type, wrap=wrap, format_unit=format_unit, label=label, spine=spine)


def blob(lon, lat, lon0, lat0, sigma, ripple):
    """Something to look at, laid out in world coordinates: a blob centred on a
    grid crossing, which shows that image and grid agree, plus ripples."""
    r2 = (lon - lon0) ** 2 + (lat - lat0) ** 2
    ripples = np.cos(2 * np.pi * lon / ripple) * np.cos(2 * np.pi * lat / ripple)
    return np.exp(-r2 / 2 / sigma**2) + 0.15 * ripples


CASES = {
    # IRIS-like slit-jaw image: helioprojective, in arcsec, rolled by 20 degrees
    "hpc": make_case(
        "IRIS-like helioprojective, rolled by 20°",
        ["HPLN-TAN", "HPLT-TAN"], ["arcsec"] * 2, [-350, 250], [0.333, 0.333], (512, 512), 20,
        [coord("longitude", 180 * u.deg, u.arcsec, "Solar X", "b"),
         coord("latitude", None, u.arcsec, "Solar Y", "l")],
        lambda x, y: blob(3600 * x, 3600 * y, -300, 200, 25, 20),
    ),
    # All-sky plate carree: longitude wraps from 0 to 360 at the centre
    "car": make_case(
        "All-sky galactic plate carrée, longitude wraps",
        ["GLON-CAR", "GLAT-CAR"], ["deg"] * 2, [0, 0], [-1, 1], (180, 360), 0,
        [coord("longitude", 360 * u.deg, u.deg, "Galactic longitude", "b"),
         coord("latitude", None, u.deg, "Galactic latitude", "l")],
        lambda lon, lat: blob(lon, lat, 90, 30, 6, 60)
        + np.exp(-np.abs(lat) / 8) * (0.6 + 0.4 * np.cos(np.radians(lon))),
    ),
    # Plain TAN: right ascension is labelled in hours, which needs mathtext
    "tan": make_case(
        "Plain TAN, right ascension in hours",
        ["RA---TAN", "DEC--TAN"], ["deg"] * 2, [266.4, -28.9], [-0.0005, 0.0005], (300, 400), 0,
        [coord("longitude", 360 * u.deg, u.hourangle, "Right ascension", "b"),
         coord("latitude", None, u.deg, "Declination", "l")],
        # 17h45m24s, -28d56m
        lambda ra, dec: blob(ra, dec, 266.35, -28 - 56 / 60, 0.015, 0.02),
    ),
}


def layout(case, xlim, ylim):
    """Run the core for one view. Everything it returns is in display pixels,
    except the ticks and grid lines, which are in data pixels."""
    view = View(xlim, ylim, case.size)
    p2w = PixelToWorld(case.wcs)
    (x0, x1), (y0, y1) = xlim, ylim
    # The spines of RectangularFrame
    outlines = {
        "b": [[x0, y0], [x1, y0]], "r": [[x1, y0], [x1, y1]],
        "t": [[x1, y1], [x0, y1]], "l": [[x0, y1], [x0, y0]],
    }
    spines = {}
    for name, outline in outlines.items():
        data = _layout.resample_spine(np.array(outline, dtype=float), conf.frame_boundary_samples)
        normal = _layout.spine_normal_angle(view.to_display(data))
        spines[name] = _layout.SpineArrays(data, p2w.transform(data), normal)

    units = [u.Unit(unit) for unit in case.wcs.world_axis_units]
    types, wraps = [c.type for c in case.coords], [c.wrap for c in case.coords]
    ranges = find_coordinate_range(p2w, [x0, x1, y0, y1], types, units, wraps)

    coords, boxes = [], []
    for index, (c, unit) in enumerate(zip(case.coords, units)):
        fl = AngleFormatterLocator(unit=unit, format_unit=c.format_unit)
        spec = _layout.CoordSpec(
            coord_index=index,
            coord_type=c.type,
            coord_unit=unit,
            coord_wrap=c.wrap,
            coord_scale_to_deg=None if unit is u.deg else unit.to(u.deg),
            locator=fl.locator,
            formatter=fl.formatter,
            minor_locator=None,
            minor_frequency=5,
        )
        placed = _layout.place_ticks(
            spec, ranges[index], spines, p2w.transform, view.to_display, view.from_display
        )

        # Tick labels, stored as TickLabels.add stores them
        labels = _layout.label_store(placed)
        _layout.sort_labels(labels)
        _layout.simplify_labels(labels, NUMERICAL_CHARS)
        xy = _layout.anchor_tick_labels(
            labels, [c.spine], view.to_display, TICK_PAD + TICK_SIZE, measure
        )

        def extent(axis, i, labels=labels, xy=xy):
            if labels.text[axis][i] == "":
                return None
            (x, y), (w, h) = xy[axis][i], measure(labels.text[axis][i])
            return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

        kept = list(_layout.keep_tick_labels(labels, [c.spine], extent, False, boxes))
        boxes += [box for _, _, box in kept]

        grid = _layout.grid_lines(
            spec, ranges, conf.grid_samples, p2w.transform, p2w.inverted().transform
        )
        coords.append(SimpleNamespace(
            placed=placed, labels=labels, xy=xy, kept=kept, grid=grid, axis_label=None
        ))

    # Axis labels go past the union of all the tick labels, and a coordinate
    # without tick labels gets none (WCSAxes' 'labels' visibility rule).
    if boxes:
        union = np.array(boxes)
        union = (*union[:, :2].min(axis=0), *union[:, 2:].max(axis=0))
    for c, out in zip(case.coords, coords):
        if out.kept:
            outline = view.to_display(outlines[c.spine])
            x, y, normal = _layout.spine_midpoint(outline, _layout.spine_normal_angle(outline))
            out.axis_label = _layout.axis_label_position(
                c.spine, x, y, normal, FONT_SIZE, FONT_SIZE, True, union
            )
    return SimpleNamespace(view=view, coords=coords)


def polyline(pieces):
    """One (N, 2) array for a Lines mark, with NaN rows between the pieces."""
    nan = np.full((1, 2), np.nan)
    return np.concatenate([np.vstack([p, nan]) for p in pieces] or [np.empty((0, 2))])


def png(case):
    ny, nx = case.shape
    world = case.wcs.pixel_to_world_values(*np.mgrid[:ny, :nx][::-1])
    image = np.nan_to_num(case.brightness(*world))
    image = 255 * (image - image.min()) / np.ptp(image)
    buffer = io.BytesIO()
    # PNG rows run top down; image row 0 is at the bottom
    Image.fromarray(image[::-1].astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def figure(name, pan_zoom=True):
    case = CASES[name]
    (ny, nx), (w, h) = case.shape, case.size
    sx = bq.LinearScale(min=-0.5, max=nx - 0.5)
    sy = bq.LinearScale(min=-0.5, max=ny - 0.5)
    display = {"x": bq.LinearScale(min=0.0, max=float(w)), "y": bq.LinearScale(min=0.0, max=float(h))}
    image = bq.Image(
        image=widgets.Image(value=png(case), format="png"),
        x=[-0.5, nx - 0.5], y=[-0.5, ny - 0.5], scales={"x": sx, "y": sy},
    )
    # Grid lines are clipped to the plot area, which is the frame
    grid = bq.Lines(scales=display, colors=["white"], opacities=[0.6], stroke_width=1)
    ticks = bq.Lines(scales=display, colors=["black"], stroke_width=1, apply_clip=False)
    frame = bq.Lines(
        x=[0, w, w, 0, 0], y=[0, 0, h, h, 0], scales=display, colors=["black"], apply_clip=False
    )
    text = dict(
        scales=display, colors=["black"], default_size=FONT_SIZE, font_weight="normal",
        align="middle", apply_clip=False,
    )
    tick_labels = [bq.Label(**text) for _ in case.coords]
    axis_labels = [bq.Label(**text, text=[c.label]) for c in case.coords]
    fig = bq.Figure(
        marks=[image, grid, ticks, frame, *tick_labels, *axis_labels],
        fig_margin=MARGIN, padding_x=0, padding_y=0,
        layout=widgets.Layout(
            width=f"{w + MARGIN['left'] + MARGIN['right']}px",
            height=f"{h + MARGIN['top'] + MARGIN['bottom']}px",
        ),
    )
    if pan_zoom:
        fig.interaction = bq.interacts.PanZoom(scales={"x": [sx], "y": [sy]})

    def update(change=None):
        xlim, ylim = (sx.min, sx.max), (sy.min, sy.max)
        if getattr(fig, "wcs_view", None) == (xlim, ylim):
            return  # the other end of the same scale change
        fig.wcs_view, fig.wcs_layout = (xlim, ylim), layout(case, xlim, ylim)
        view, coords = fig.wcs_layout.view, fig.wcs_layout.coords

        lines = [line for c in coords for line in c.grid]
        # A line breaks at each MOVETO; the NaN row in front of it does that here
        lines = [np.insert(p, np.flatnonzero(codes == _layout.MOVETO), np.nan, axis=0) for p, codes in lines]
        points = view.to_display(polyline(lines))
        with grid.hold_sync():
            grid.x, grid.y = points.T

        # Every coordinate has ticks on all four spines
        start = view.to_display(np.vstack([c.placed.major.pixel for c in coords]))
        angle = np.radians(np.hstack([c.placed.major.angle for c in coords]) + 180)
        end = start + TICK_SIZE * np.column_stack([np.cos(angle), np.sin(angle)])
        points = polyline(np.stack([start, end], axis=1))
        with ticks.hold_sync():
            ticks.x, ticks.y = points.T

        for c, tick_label, axis_label in zip(coords, tick_labels, axis_labels):
            anchors = [c.xy[axis][i] for axis, i, _ in c.kept] or np.empty((0, 2))
            with tick_label.hold_sync():
                tick_label.x, tick_label.y = np.transpose(anchors)
                tick_label.text = [plain(c.labels.text[axis][i]) for axis, i, _ in c.kept]
            x, y, rotation = c.axis_label or (np.nan, np.nan, 0.0)
            with axis_label.hold_sync():
                # bqplot rotates clockwise, in a y-down space
                axis_label.x, axis_label.y, axis_label.rotate_angle = [x], [y], -rotation

    sx.observe(update, ["min", "max"])
    sy.observe(update, ["min", "max"])
    update()
    return fig


def main(outdir=Path(__file__).parent / "html"):
    from ipywidgets.embed import dependency_state, embed_minimal_html

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    for name, case in CASES.items():
        # A static page has no kernel to rerun the layout, so no pan and zoom
        fig = figure(name, pan_zoom=False)
        embed_minimal_html(outdir / f"{name}.html", views=[fig], title=case.title,
                           state=dependency_state(fig))
        drawn = [len(c.kept) for c in fig.wcs_layout.coords]
        print(f"{name}: {drawn} tick labels drawn, wrote {outdir / name}.html")

    # The core ran without a matplotlib Figure, Axes or renderer
    used = [m for m in sys.modules if m in ("matplotlib.pyplot", "matplotlib.figure")
            or m.startswith("matplotlib.backends.backend_")]
    assert not used, used


if __name__ == "__main__":
    main(*sys.argv[1:])
