"""
WCSAxes-style ticks, tick labels, grid lines and axis labels in a plain Qt
widget, laid out by astropy.visualization.wcsaxes._model without a matplotlib
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

from astropy.visualization.wcsaxes._layout import MOVETO
from astropy.visualization.wcsaxes._model import AxesModel
from astropy.wcs import WCS

# Matplotlib's defaults at 100 dpi, in display pixels
DPI = 100
PT = DPI / 72
TICK_SIZE, TICK_PAD, FONT_SIZE, LINE_WIDTH = 3.5 * PT, 3.5 * PT, 10 * PT, 0.8 * PT
# Space around the frame for the labels: left, bottom, right, top
MARGINS = np.array([105, 75, 25, 25])


def make_case(ctype, unit, crval, cdelt, shape, labels, spines, roll=0):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ctype
    wcs.wcs.cunit = [unit] * 2
    wcs.wcs.crval = crval
    wcs.wcs.cdelt = cdelt
    ny, nx = shape
    wcs.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    wcs.wcs.pc = [[c, -s], [s, c]]
    # As WCSAxes.reset_wcs does: until wcslib's set() runs, an arcsec header
    # reports arcsec as its unit although the WCS returns degrees
    wcs.wcs.set()

    # The model reads the type, wrap, unit and format unit of each coordinate
    # from the WCS, as WCSAxes does. The labels are drawn as plain text.
    model = AxesModel.from_wcs(wcs)
    model.text_format = "unicode"
    for coord, label, spine in zip(model, labels, spines):
        coord.axislabel = label
        coord.set_ticklabel_position(spine)
        coord.set_axislabel_position(spine)
        coord.exclude_overlapping = True
        coord.grid = True

    # Something to look at: a blob with ripples, row 0 at the bottom
    y, x = np.mgrid[:ny, :nx]
    r2 = ((x - nx / 2) ** 2 + (y - ny / 2) ** 2) / max(nx, ny) ** 2
    image = np.exp(-4 * r2) * (0.75 + 0.25 * np.sin(x / 7) * np.sin(y / 11))
    image = (40 + 180 * image).astype(np.uint8)
    return SimpleNamespace(
        wcs=wcs,
        model=model,
        image=image,
        qimage=QImage(
            image[::-1].tobytes(), nx, ny, nx, QImage.Format_Grayscale8
        ).copy(),
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
        labels=["Solar X", "Solar Y"],
        spines="bl",
    ),
    # All-sky plate carree: longitude wraps from 360 to 0 at the centre
    "car": lambda: make_case(
        ctype=["GLON-CAR", "GLAT-CAR"],
        unit="deg",
        crval=[0, 0],
        cdelt=[-1, 1],
        shape=(180, 360),
        labels=["Galactic longitude", "Galactic latitude"],
        spines="bl",
    ),
    # Plain TAN: right ascension is labelled in hours
    "tan": lambda: make_case(
        ctype=["RA---TAN", "DEC--TAN"],
        unit="deg",
        crval=[266.4, -28.9],
        cdelt=[-0.0005, 0.0005],
        shape=(300, 400),
        labels=["Right ascension", "Declination"],
        spines="bl",
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
    """Everything WCSAxes would draw, in display pixels, from the model."""
    to_display, from_display = display_transform(box, xlim, ylim)
    return case.model.layout(
        xlim,
        ylim,
        to_display,
        from_display,
        measure=measure,
        tick_size=TICK_SIZE,
        pad=TICK_PAD,
        font_size=FONT_SIZE,
    )


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
    for c in result.coords:
        for pixel, codes in c.grid:
            starts = np.flatnonzero(codes == MOVETO)[1:]
            for part in np.split(to_display(pixel), starts):
                # A NaN vertex is a MOVETO of its own, so parts are clean
                if len(part) > 1 and np.isfinite(part).all():
                    painter.drawPolyline(QPolygonF([qt(p) for p in part]))
    painter.setClipping(False)

    painter.setPen(QPen(Qt.black, LINE_WIDTH))
    for c in result.coords:
        m = c.ticks.major
        out = np.radians(m.angle + 180)  # ticks point out
        start = to_display(m.pixel)
        end = start + TICK_SIZE * np.column_stack([np.cos(out), np.sin(out)])
        for a, b in zip(start, end):
            painter.drawLine(qt(a), qt(b))

    def text_at(text, xy, rotation=0):
        painter.save()
        painter.translate(qt(xy))
        painter.rotate(-rotation)  # counter-clockwise, in y-up display space
        painter.drawText(QRectF(-500, -50, 1000, 100), Qt.AlignCenter, text)
        painter.restore()

    for c in result.coords:
        for axis, i, _ in c.kept:
            text_at(c.labels.text[axis][i], c.anchors[axis][i])
        for x, y, rotation in c.axis_labels.values():
            text_at(c.axis_label_text, (x, y), rotation)

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
    # With matplotlib installed, importing the wcsaxes package imports its
    # matplotlib API, but nothing here draws with it
    drawing = ("matplotlib.pyplot", "matplotlib.figure", "matplotlib.backends.backend_")
    assert not [name for name in sys.modules if name.startswith(drawing)]
