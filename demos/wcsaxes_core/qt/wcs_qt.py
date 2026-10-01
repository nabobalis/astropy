"""
WCSAxes-style ticks, tick labels, grid lines and axis labels in a plain Qt
widget, laid out by astropy.visualization.wcsaxes._layout without a matplotlib
Figure, Axes or renderer.

    python wcs_qt.py [hpc|car|tan]            drag to pan, scroll to zoom
    python wcs_qt.py tan --png tan.png        render offscreen instead

Everything is recomputed in paintEvent, so panning, zooming and resizing just
repaint.
"""

import sys
from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QApplication, QWidget

from astropy import units as u
from astropy.visualization.wcsaxes import _layout, conf
from astropy.visualization.wcsaxes.coordinate_range import find_coordinate_range
from astropy.visualization.wcsaxes.formatter_locator import AngleFormatterLocator
from astropy.wcs import WCS

# Matplotlib's defaults at 100 dpi, in display pixels
DPI = 100
PT = DPI / 72
TICK_SIZE, TICK_PAD, FONT_SIZE, LINE_WIDTH = 3.5 * PT, 3.5 * PT, 10 * PT, 0.8 * PT
# Space around the frame for the labels: left, bottom, right, top
MARGINS = np.array([105, 75, 25, 25])
# As TickLabels.simplify_labels with matplotlib's default rcParams
NUMERICAL_CHARS = "0123456789.+\N{MINUS SIGN}"
# The formatter writes hour angles in mathtext; Qt gets Unicode instead
MATHTEXT = {
    r"$\mathregular{^h}$": "ʰ",
    r"$\mathregular{^m}$": "ᵐ",
    r"$\mathregular{^s}$": "ˢ",
}


def plain(text):
    for mathtext, unicode in MATHTEXT.items():
        text = text.replace(mathtext, unicode)
    return text


def make_case(ctype, unit, crval, cdelt, shape, coords, roll=0):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ctype
    wcs.wcs.cunit = [unit] * 2
    wcs.wcs.crval = crval
    wcs.wcs.cdelt = cdelt
    ny, nx = shape
    wcs.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    wcs.wcs.pc = [[c, -s], [s, c]]

    def pixel_to_world(xy):
        return np.array(wcs.pixel_to_world_values(*xy.T)).T

    def world_to_pixel(world):
        return np.array(wcs.world_to_pixel_values(*world.T)).T

    # Something to look at: a blob with ripples, row 0 at the bottom
    y, x = np.mgrid[:ny, :nx]
    r2 = ((x - nx / 2) ** 2 + (y - ny / 2) ** 2) / max(nx, ny) ** 2
    image = np.exp(-4 * r2) * (0.75 + 0.25 * np.sin(x / 7) * np.sin(y / 11))
    image = (40 + 180 * image).astype(np.uint8)
    return SimpleNamespace(
        wcs=wcs,
        # find_coordinate_range wants an object with .transform
        p2w=SimpleNamespace(transform=pixel_to_world),
        world_to_pixel=world_to_pixel,
        image=image,
        qimage=QImage(
            image[::-1].tobytes(), nx, ny, nx, QImage.Format_Grayscale8
        ).copy(),
        coords=coords,
    )


def coord(type, wrap, format_unit, label, spine):
    """What WCSAxes reads from the WCS for one coordinate, and where its labels go."""
    return SimpleNamespace(
        type=type, wrap=wrap, format_unit=format_unit, label=label, spine=spine
    )


CASES = {
    # IRIS-like slit-jaw image: helioprojective, in arcsec, rolled by 20 degrees
    "hpc": lambda: make_case(
        ctype=["HPLN-TAN", "HPLT-TAN"],
        unit="arcsec",
        crval=[-350, 250],
        cdelt=[0.333, 0.333],
        shape=(512, 512),
        roll=20,
        coords=[
            coord("longitude", 180 * u.deg, u.arcsec, "Solar X", "b"),
            coord("latitude", None, u.arcsec, "Solar Y", "l"),
        ],
    ),
    # All-sky plate carree: longitude wraps from 360 to 0 at the centre
    "car": lambda: make_case(
        ctype=["GLON-CAR", "GLAT-CAR"],
        unit="deg",
        crval=[0, 0],
        cdelt=[-1, 1],
        shape=(180, 360),
        coords=[
            coord("longitude", 360 * u.deg, u.deg, "Galactic longitude", "b"),
            coord("latitude", None, u.deg, "Galactic latitude", "l"),
        ],
    ),
    # Plain TAN: right ascension is labelled in hours, which needs mathtext
    "tan": lambda: make_case(
        ctype=["RA---TAN", "DEC--TAN"],
        unit="deg",
        crval=[266.4, -28.9],
        cdelt=[-0.0005, 0.0005],
        shape=(300, 400),
        coords=[
            coord("longitude", 360 * u.deg, u.hourangle, "Right ascension", "b"),
            coord("latitude", None, u.deg, "Declination", "l"),
        ],
    ),
}


def display_transform(box, xlim, ylim):
    """Data pixels to and from display pixels (origin bottom left, y up)."""
    origin = np.array([xlim[0], ylim[0]])
    scale = (box[2:] - box[:2]) / [xlim[1] - xlim[0], ylim[1] - ylim[0]]

    def to_display(xy):
        return (np.asarray(xy) - origin) * scale + box[:2]

    def from_display(xy):
        return (np.asarray(xy) - box[:2]) / scale + origin

    return to_display, from_display


def layout(case, box, xlim, ylim, measure):
    """Everything WCSAxes would draw, in display pixels, from the _layout core."""
    to_display, from_display = display_transform(box, xlim, ylim)
    (x0, x1), (y0, y1) = xlim, ylim
    outlines = {
        "b": [[x0, y0], [x1, y0]],
        "r": [[x1, y0], [x1, y1]],
        "t": [[x1, y1], [x0, y1]],
        "l": [[x0, y1], [x0, y0]],
    }
    spines = {}
    for name, outline in outlines.items():
        data = _layout.resample_spine(np.array(outline), conf.frame_boundary_samples)
        with np.errstate(invalid="ignore"):
            world = case.p2w.transform(data)
        normal = _layout.spine_normal_angle(to_display(data))
        spines[name] = _layout.SpineArrays(data, world, normal)

    ranges = find_coordinate_range(
        case.p2w,
        [x0, x1, y0, y1],
        [c.type for c in case.coords],
        [u.deg] * 2,
        [c.wrap for c in case.coords],
    )

    result, existing = [], []  # existing: tick label boxes kept so far
    for index, c in enumerate(case.coords):
        # WCSLIB gives world values in degrees, even for an arcsec header
        fl = AngleFormatterLocator(unit=u.deg, format_unit=c.format_unit)
        spec = _layout.CoordSpec(
            coord_index=index,
            coord_type=c.type,
            coord_unit=u.deg,
            coord_wrap=c.wrap,
            coord_scale_to_deg=None,
            locator=fl.locator,
            formatter=fl.formatter,
            minor_locator=None,
            minor_frequency=5,
        )
        placed = _layout.place_ticks(
            spec, ranges[index], spines, case.p2w.transform, to_display, from_display
        )
        grid = _layout.grid_lines(
            spec, ranges, conf.grid_samples, case.p2w.transform, case.world_to_pixel
        )

        # Tick labels, stored as TickLabels stores them
        labels = _layout.tick_labels(placed)
        _layout.sort_labels(labels)
        _layout.simplify_labels(labels, NUMERICAL_CHARS)
        anchors = _layout.anchor_tick_labels(
            labels, [c.spine], to_display, TICK_SIZE + TICK_PAD, measure
        )

        def extent(axis, i):
            if labels.text[axis][i] == "":
                return None
            (x, y), (w, h) = anchors[axis][i], measure(labels.text[axis][i])
            return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

        kept = list(_layout.keep_tick_labels(labels, [c.spine], extent, True, existing))
        existing += [box for *_, box in kept]
        result.append(
            SimpleNamespace(
                placed=placed,
                grid=grid,
                labels=labels,
                anchors=anchors,
                kept=kept,
                axis_label=None,
            )
        )

    # Axis labels go beyond the box around all the tick labels
    union = None
    if existing:
        boxes = np.array(existing)
        union = (*boxes[:, :2].min(axis=0), *boxes[:, 2:].max(axis=0))
    for c, r in zip(case.coords, result):
        if r.kept:  # WCSAxes' default rule: no tick labels, no axis label
            pixel = to_display(np.array(outlines[c.spine]))
            x, y, normal = _layout.spine_midpoint(
                pixel, _layout.spine_normal_angle(pixel)
            )
            r.axis_label = _layout.axis_label_position(
                c.spine, x, y, normal, FONT_SIZE, FONT_SIZE, True, union
            )
    return result


def draw(painter, case, box, xlim, ylim):
    """Paint the image, grid, ticks and labels; return what layout() gave."""
    device = painter.device()
    height = device.height()
    to_display, _ = display_transform(box, xlim, ylim)

    def qt(xy):  # display pixels (y up) to Qt (y down)
        return QPointF(xy[0], height - xy[1])

    font = QFont()  # the application font
    font.setPointSizeF(FONT_SIZE * 72 / device.logicalDpiY())
    metrics = QFontMetricsF(font, device)

    def measure(text, *xy):
        # As matplotlib sizes a line of text: the advance width, and one em
        # (the font's typographic ascender plus descender) unless ink is taller
        text = plain(text)
        ink = metrics.tightBoundingRect(text).height()
        return metrics.horizontalAdvance(text), max(ink, FONT_SIZE)

    result = layout(case, box, xlim, ylim, measure)

    painter.setRenderHint(QPainter.Antialiasing)
    painter.setFont(font)
    frame = QRectF(qt(box[[0, 3]]), qt(box[[2, 1]]))
    painter.setClipRect(frame)
    ny, nx = case.image.shape
    painter.drawImage(
        QRectF(qt(to_display((-0.5, ny - 0.5))), qt(to_display((nx - 0.5, -0.5)))),
        case.qimage,
    )

    painter.setPen(QPen(QColor(255, 255, 255, 153), LINE_WIDTH))
    for r in result:
        for pixel, codes in r.grid or []:
            starts = np.flatnonzero(codes == _layout.MOVETO)[1:]
            for part in np.split(to_display(pixel), starts):
                # A NaN vertex is a MOVETO of its own, so parts are clean
                if len(part) > 1 and np.isfinite(part).all():
                    painter.drawPolyline(QPolygonF([qt(p) for p in part]))
    painter.setClipping(False)

    painter.setPen(QPen(Qt.black, LINE_WIDTH))
    for r in result:
        m = r.placed.major
        out = np.radians(m.angle + 180)  # ticks point out
        start = to_display(m.pixel)
        end = start + TICK_SIZE * np.column_stack([np.cos(out), np.sin(out)])
        for a, b in zip(start, end):
            painter.drawLine(qt(a), qt(b))

    def text_at(text, xy, rotation=0):
        painter.save()
        painter.translate(qt(xy))
        painter.rotate(-rotation)  # counter-clockwise, in y-up display space
        painter.drawText(QRectF(-500, -50, 1000, 100), Qt.AlignCenter, plain(text))
        painter.restore()

    for c, r in zip(case.coords, result):
        for axis, i, _ in r.kept:
            text_at(r.labels.text[axis][i], r.anchors[axis][i])
        if r.axis_label is not None:
            x, y, rotation = r.axis_label
            text_at(c.label, (x, y), rotation)

    painter.drawRect(frame)
    return result


class WCSView(QWidget):
    """An image with WCS ticks and labels: drag to pan, scroll to zoom."""

    def __init__(self, case):
        super().__init__()
        self.case = case
        ny, nx = case.image.shape
        self.scale = 640 / max(nx, ny)  # display pixels per image pixel
        self.center = np.array([(nx - 1) / 2, (ny - 1) / 2])
        self.resize(
            *(np.array([nx, ny]) * self.scale + MARGINS[:2] + MARGINS[2:]).astype(int)
        )

    def view(self):
        """The frame in display pixels, and the data limits it shows."""
        box = np.array([0, 0, self.width(), self.height()]) + MARGINS * [1, 1, -1, -1]
        half = (box[2:] - box[:2]) / 2 / self.scale
        lo, hi = self.center - half, self.center + half
        return box.astype(float), (lo[0], hi[0]), (lo[1], hi[1])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.white)
        draw(painter, self.case, *self.view())

    def mousePressEvent(self, event):
        self.last = event.pos()

    def mouseMoveEvent(self, event):
        move, self.last = event.pos() - self.last, event.pos()
        self.center -= np.array([move.x(), -move.y()]) / self.scale
        self.update()

    def wheelEvent(self, event):
        # Zoom about the cursor: the data point under it stays put
        zoom = 1.25 ** (event.angleDelta().y() / 120)
        box, _, _ = self.view()
        cursor = np.array([event.position().x(), self.height() - event.position().y()])
        self.center += (cursor - (box[:2] + box[2:]) / 2) / self.scale * (1 - 1 / zoom)
        self.scale *= zoom
        self.update()

    def render_image(self):
        image = QImage(self.width(), self.height(), QImage.Format_ARGB32)
        image.fill(Qt.white)
        painter = QPainter(image)
        result = draw(painter, self.case, *self.view())
        painter.end()
        return image, result


if __name__ == "__main__":
    app = QApplication(sys.argv)
    view = WCSView(CASES[sys.argv[1] if len(sys.argv) > 1 else "hpc"]())
    if "--png" in sys.argv:
        view.render_image()[0].save(sys.argv[sys.argv.index("--png") + 1])
    else:
        view.show()
        app.exec_()
    # The formatter imports matplotlib, but nothing here draws with it
    drawing = ("matplotlib.pyplot", "matplotlib.figure", "matplotlib.backends.backend_")
    assert not [name for name in sys.modules if name.startswith(drawing)]
