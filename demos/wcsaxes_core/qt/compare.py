"""
Draw each case with the Qt demo and with matplotlib's WCSAxes, side by side,
and check that they agree.

    python compare.py        writes output/<case>_<view>.png and a table

Both get the same canvas size, frame box, data limits and dpi, and both use
DejaVu Sans. The "moved" view is reached through the widget's own mouse and
wheel handlers, then a resize.
"""

import os
import sys

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from PyQt5.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PyQt5.QtGui import QFont, QFontDatabase, QImage, QMouseEvent, QPainter, QWheelEvent
from PyQt5.QtWidgets import QApplication

from astropy.visualization.wcsaxes import WCSAxes
from astropy.visualization.wcsaxes.axislabels import AxisLabels
from astropy.visualization.wcsaxes.ticklabels import TickLabels

import wcs_qt

# Ticks come from the same code on the same inputs; only the rounding of the
# two display transforms differs. Grid lines must be identical.
TOL_GEOMETRY = 1e-6  # display pixels, and degrees for tick angles
# Label positions depend on text size, which Qt and Agg measure differently.
# Matplotlib renders hour-angle labels as mathtext, with raised superscripts,
# where Qt draws Unicode superscript letters.
TOL_TEXT = 1.0  # display pixels
TOL_MATHTEXT = 4.0

# matplotlib draws hour angles as mathtext; the demos draw Unicode letters
MATHTEXT = {
    r"$\mathregular{^h}$": "\N{MODIFIER LETTER SMALL H}",
    r"$\mathregular{^m}$": "\N{MODIFIER LETTER SMALL M}",
    r"$\mathregular{^s}$": "\N{MODIFIER LETTER SMALL S}",
}


def plain(text):
    for mathtext, unicode in MATHTEXT.items():
        text = text.replace(mathtext, unicode)
    return text


def matplotlib_render(view):
    case, (box, xlim, ylim) = view.case, view.view()
    w, h = view.width(), view.height()
    fig = Figure(figsize=(w / wcs_qt.DPI, h / wcs_qt.DPI), dpi=wcs_qt.DPI)
    canvas = FigureCanvasAgg(fig)
    rect = [box[0] / w, box[1] / h, (box[2] - box[0]) / w, (box[3] - box[1]) / h]
    ax = fig.add_axes(WCSAxes(fig, rect, wcs=case.wcs))
    ax.imshow(case.image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_aspect("auto")  # the view already has square pixels; keep its box
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for coord, c in zip(ax.coords, case.model):
        # Both read the coordinate metadata from the WCS the same way
        assert coord.coord_type == c.coord_type and coord.coord_wrap == c.coord_wrap
        assert coord.get_format_unit() == c.get_format_unit()
        coord.set_ticklabel_position(c.get_ticklabel_position())
        coord.set_axislabel_position(c.get_axislabel_position())
        coord.set_axislabel(c.axislabel)
        coord.set_ticklabel(exclude_overlapping=True)
        coord.grid(color="white", alpha=0.6)

    drawn = []  # (text, x, y, rotation) of each label matplotlib draws

    def spy(text, renderer):
        if isinstance(text, (TickLabels, AxisLabels)):
            drawn.append((text.get_text(), *text.get_position(), text.get_rotation()))
        text_draw(text, renderer)

    text_draw, Text.draw = Text.draw, spy
    try:
        canvas.draw()
    finally:
        Text.draw = text_draw
    return ax, drawn, np.asarray(canvas.buffer_rgba())


def differences(view, result, ax, drawn):
    """Largest differences between the Qt layout and what matplotlib drew."""
    to_display, _ = wcs_qt.display_transform(*view.view())
    tick_px = tick_deg = 0.0
    n_ticks = 0
    for coord, r in zip(ax.coords, result.coords):
        m = r.ticks.major
        for axis in "brtl":
            on = np.array(m.axis) == axis
            assert np.array_equal(m.world[on], coord._ticks.world[axis])
            mpl = ax.transData.transform(np.reshape(coord._ticks.pixel[axis], (-1, 2)))
            tick_px = max(tick_px, np.abs(to_display(m.pixel[on]) - mpl).max(initial=0))
            # -180 and 180 are the same tick direction
            angle = np.abs((m.angle[on] - coord._ticks.angle[axis] + 180) % 360 - 180)
            tick_deg = max(tick_deg, angle.max(initial=0))
            n_ticks += on.sum()
        mpl_text = {a: [plain(t) for t in ts] for a, ts in coord._ticklabels.text.items()}
        assert dict(r.labels.text) == mpl_text, "label strings"
        assert len(r.grid) == len(coord._grid_lines)
        for (pixel, codes), path in zip(r.grid, coord._grid_lines):
            assert np.array_equal(codes, path.codes)
            assert np.array_equal(pixel, path.vertices, equal_nan=True)

    # Tick labels of every coordinate, then axis labels, as WCSAxes draws them
    mine = [
        (r.labels.text[axis][i], *r.anchors[axis][i], 0.0)
        for r in result.coords
        for axis, i, _ in r.kept
    ]
    mine += [
        (r.axis_label_text, *position)
        for r in result.coords
        for position in r.axis_labels.values()
    ]
    assert [t[0] for t in mine] == [plain(t[0]) for t in drawn], "drawn labels"
    assert all(a[3] % 360 == b[3] % 360 for a, b in zip(mine, drawn)), "rotations"
    text_px = max(abs(a[k] - b[k]) for a, b in zip(mine, drawn) for k in (1, 2))
    return n_ticks, tick_px, tick_deg, len(mine), text_px


def pan_zoom_resize(view):
    def mouse(kind, x, y):
        return QMouseEvent(
            kind, QPointF(x, y), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier
        )

    view.mousePressEvent(mouse(QEvent.MouseButtonPress, 300, 300))
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 360, 260))
    at = QPointF(view.width() / 2, view.height() / 2)
    view.wheelEvent(
        QWheelEvent(
            at,
            at,
            QPoint(),
            QPoint(0, 240),
            Qt.NoButton,
            Qt.NoModifier,
            Qt.NoScrollPhase,
            False,
        )
    )
    view.resize(view.width() + 100, view.height() - 60)


def side_by_side(qt_image, rgba, path):
    h, w = rgba.shape[:2]
    mpl = QImage(rgba.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888)
    out = QImage(2 * w, h + 30, QImage.Format_ARGB32)
    out.fill(Qt.white)
    painter = QPainter(out)
    painter.drawText(
        QRectF(0, 0, w, 30), Qt.AlignCenter, "Qt, laid out by wcsaxes._model"
    )
    painter.drawText(QRectF(w, 0, w, 30), Qt.AlignCenter, "matplotlib WCSAxes")
    painter.drawImage(0, 30, qt_image)
    painter.drawImage(w, 30, mpl)
    painter.drawLine(w, 0, w, h + 30)
    painter.end()
    out.save(path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    QFontDatabase.addApplicationFont(font)
    app.setFont(QFont("DejaVu Sans"))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out, exist_ok=True)

    print("case view    ticks  tick px  tick deg  labels  label px  limit")
    ok = True
    for name, make in wcs_qt.CASES.items():
        view = wcs_qt.WCSView(make())
        for state in ("initial", "moved"):
            if state == "moved":
                pan_zoom_resize(view)
            image, result = view.render_image()
            ax, drawn, rgba = matplotlib_render(view)
            side_by_side(image, rgba, os.path.join(out, f"{name}_{state}.png"))
            n, tick_px, tick_deg, n_labels, text_px = differences(
                view, result, ax, drawn
            )
            mathtext = any("$" in text for text, *_ in drawn)
            limit = TOL_MATHTEXT if mathtext else TOL_TEXT
            ok &= max(tick_px, tick_deg) < TOL_GEOMETRY and text_px < limit
            print(
                f"{name:4} {state:7} {n:6} {tick_px:8.1e} {tick_deg:9.1e} "
                f"{n_labels:7} {text_px:9.2f} {limit:6.1f}"
            )

    print("Tick world values, label strings (hours in Unicode), the labels")
    print(f"drawn, and grid lines are identical. Tick limit: {TOL_GEOMETRY:g}.")
    sys.exit(0 if ok else 1)
