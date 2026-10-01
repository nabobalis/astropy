"""
WCSAxes-style ticks, tick labels, grid lines and axis labels on a bqplot
image figure, laid out by astropy.visualization.wcsaxes._model without a
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
from astropy.visualization.wcsaxes._layout import MOVETO
from astropy.visualization.wcsaxes._model import AxesModel
from astropy.wcs import WCS
from PIL import Image, ImageFont

# Matplotlib's defaults at 100 dpi, in display pixels, so that the numbers can
# be compared with WCSAxes (compare.py). Ticks point out, as xtick.direction.
DPI = 100
PT = DPI / 72
TICK_SIZE, TICK_PAD, FONT_SIZE = 3.5 * PT, 3.5 * PT, 10 * PT
MARGIN = {"top": 25, "bottom": 75, "left": 105, "right": 25}
# The kernel cannot ask the browser how wide a text is. Pillow measures it in
# DejaVu Sans, matplotlib's default font, which gives the same width as
# matplotlib for plain text; the height is the font size.
FONT = ImageFont.truetype(f"{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans.ttf", FONT_SIZE)


def measure(text, x=None, y=None):
    return FONT.getlength(text), FONT_SIZE


class View:
    """Data pixels to and from display pixels (origin lower left, y up)."""

    def __init__(self, xlim, ylim, size):
        self.origin = np.array([xlim[0], ylim[0]], dtype=float)
        self.scale = np.array(size, dtype=float) / [xlim[1] - xlim[0], ylim[1] - ylim[0]]

    def to_display(self, xy):
        return (np.asarray(xy, dtype=float) - self.origin) * self.scale

    def from_display(self, xy):
        return np.asarray(xy, dtype=float) / self.scale + self.origin


def make_case(title, ctype, cunit, crval, cdelt, shape, roll, labels, spines, brightness):
    # The same WCSs as the Qt demo
    wcs = WCS(naxis=2)
    wcs.wcs.ctype, wcs.wcs.cunit, wcs.wcs.crval, wcs.wcs.cdelt = ctype, cunit, crval, cdelt
    ny, nx = shape
    wcs.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    wcs.wcs.pc = [[c, -s], [s, c]]
    # As WCSAxes.reset_wcs does: until wcslib's set() runs, world_axis_units
    # says arcsec for the helioprojective case, but pixel_to_world_values
    # returns degrees
    wcs.wcs.set()
    # The model reads the type, wrap, unit and format unit of each coordinate
    # from the WCS, as WCSAxes does. The browser draws plain SVG text.
    model = AxesModel.from_wcs(wcs)
    model.text_format = "unicode"
    for coord, label, spine in zip(model, labels, spines):
        coord.axislabel = label
        coord.set_ticklabel_position(spine)
        coord.set_axislabel_position(spine)
        coord.grid = True
    # The frame is the image scaled to fit 720 x 480 display pixels
    zoom = min(720 / nx, 480 / ny)
    return SimpleNamespace(
        title=title, wcs=wcs, model=model, shape=shape, brightness=brightness,
        size=(round(nx * zoom), round(ny * zoom)),
    )


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
        ["Solar X", "Solar Y"], "bl",
        lambda x, y: blob(3600 * x, 3600 * y, -300, 200, 25, 20),
    ),
    # All-sky plate carree: longitude wraps from 0 to 360 at the centre
    "car": make_case(
        "All-sky galactic plate carrée, longitude wraps",
        ["GLON-CAR", "GLAT-CAR"], ["deg"] * 2, [0, 0], [-1, 1], (180, 360), 0,
        ["Galactic longitude", "Galactic latitude"], "bl",
        lambda lon, lat: blob(lon, lat, 90, 30, 6, 60)
        + np.exp(-np.abs(lat) / 8) * (0.6 + 0.4 * np.cos(np.radians(lon))),
    ),
    # Plain TAN: right ascension is labelled in hours
    "tan": make_case(
        "Plain TAN, right ascension in hours",
        ["RA---TAN", "DEC--TAN"], ["deg"] * 2, [266.4, -28.9], [-0.0005, 0.0005], (300, 400), 0,
        ["Right ascension", "Declination"], "bl",
        # 17h45m24s, -28d56m
        lambda ra, dec: blob(ra, dec, 266.35, -28 - 56 / 60, 0.015, 0.02),
    ),
}


def layout(case, xlim, ylim):
    """Run the model for one view. Everything it returns is in display pixels,
    except the ticks and grid lines, which are in data pixels."""
    view = View(xlim, ylim, case.size)
    result = case.model.layout(
        xlim, ylim, view.to_display, view.from_display,
        measure=measure, tick_size=TICK_SIZE, pad=TICK_PAD, font_size=FONT_SIZE,
    )
    return SimpleNamespace(view=view, coords=result.coords)


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
    tick_labels = [bq.Label(**text) for _ in case.model]
    axis_labels = [bq.Label(**text) for _ in case.model]
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
        lines = [np.insert(p, np.flatnonzero(codes == MOVETO), np.nan, axis=0) for p, codes in lines]
        points = view.to_display(polyline(lines))
        with grid.hold_sync():
            grid.x, grid.y = points.T

        # Every coordinate has ticks on all four spines
        start = view.to_display(np.vstack([c.ticks.major.pixel for c in coords]))
        angle = np.radians(np.hstack([c.ticks.major.angle for c in coords]) + 180)
        end = start + TICK_SIZE * np.column_stack([np.cos(angle), np.sin(angle)])
        points = polyline(np.stack([start, end], axis=1))
        with ticks.hold_sync():
            ticks.x, ticks.y = points.T

        for c, tick_label, axis_label in zip(coords, tick_labels, axis_labels):
            anchors = [c.anchors[axis][i] for axis, i, _ in c.kept] or np.empty((0, 2))
            with tick_label.hold_sync():
                tick_label.x, tick_label.y = np.transpose(anchors)
                tick_label.text = [c.labels.text[axis][i] for axis, i, _ in c.kept]
            # One spine per coordinate here, so one Label mark each
            x, y, rotation = next(iter(c.axis_labels.values()), (np.nan, np.nan, 0.0))
            with axis_label.hold_sync():
                axis_label.text = [c.axis_label_text]
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

    # The model ran without a matplotlib Figure, Axes or renderer
    used = [m for m in sys.modules if m in ("matplotlib.pyplot", "matplotlib.figure")
            or m.startswith("matplotlib.backends.backend_")]
    assert not used, used


if __name__ == "__main__":
    main(*sys.argv[1:])
