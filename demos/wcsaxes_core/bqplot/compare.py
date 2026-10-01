"""
Compare the bqplot demo with matplotlib's WCSAxes, number by number.

For each case there are two views: the whole image, and a zoomed and panned
view, which is set through the figure's bqplot scales as PanZoom sets them.
For each view, a WCSAxes is drawn on Agg with its axes box over the same
display pixels and the same settings, and is compared with what the demo
computed and drew.

    python compare.py

Exact: tick world values, tick label texts (after sorting and simplifying,
on every spine), grid line vertices and codes, the number of ticks and of
drawn labels. Tick positions and angles agree to rounding, because the demo's
display transform is not matplotlib's transData. Label anchors, boxes and
axis labels depend on text measurement, which the demo approximates.
"""

import sys

import numpy as np
from astropy.visualization.wcsaxes import WCSAxes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import wcs_bqplot as demo

# Zoomed and panned views in data pixels, with the frame's aspect ratio. The
# CAR view runs off the sky on the left, past the edge of the image, and
# crosses longitude 0, where the labels wrap from 0 to 360.
ZOOMED = {
    "hpc": ((150, 350), (250, 450)),
    "car": ((-10.5, 229.5), (20, 140)),
    "tan": ((50, 210), (150, 270)),
}


def wcsaxes(case, xlim, ylim):
    w, h = case.size
    fig = Figure(figsize=(w / demo.DPI, h / demo.DPI), dpi=demo.DPI)
    FigureCanvasAgg(fig)
    ax = fig.add_axes(WCSAxes(fig, [0, 0, 1, 1], wcs=case.wcs))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.coords.grid()
    for coord, c in zip(ax.coords, case.coords):
        coord.set_ticklabel_position(c.spine)
        coord.set_axislabel_position(c.spine)
        coord.set_axislabel(c.label)
        # Keep the drawn tick label boxes, which AxisLabels then overwrites
        # with their union
        def spy(renderer, coord=coord, draw=coord._ticklabels.draw):
            draw(renderer)
            coord.drawn_boxes = [bb.extents for bb in coord._ticklabels._all_bboxes]

        coord._ticklabels.draw = spy
    fig.canvas.draw()
    return ax


def flat(d):
    return [v for axis in d for v in d[axis]]


def max_diff(a, b, period=None):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    d = a - b if period is None else (a - b + period / 2) % period - period / 2
    return float(np.max(np.abs(d), initial=0.0))


def compare(case, xlim, ylim, result):
    ax = wcsaxes(case, xlim, ylim)
    rows = []
    for coord, out in zip(ax.coords, result.coords):
        ticks, labels, major = coord._ticks, coord._ticklabels, out.placed.major
        spine = coord._ticklabels.get_visible_axes()[0]
        if out.axis_label:
            axis_label = coord._axislabels
            axis_label = max(
                max_diff(axis_label.get_position(), out.axis_label[:2]),
                max_diff(axis_label.get_rotation(), out.axis_label[2], period=360),
            )
        else:
            # No tick labels, so neither draws an axis label
            axis_label = "none" if not coord.drawn_boxes else "MISSING"
        rows.append({
            "ticks": f"{len(major.world)}/{len(flat(ticks.world))}",
            "world": np.array_equal(flat(ticks.world), major.world),
            "pixel": max_diff(flat(ticks.pixel), major.pixel),
            "angle": max_diff(flat(ticks.angle), major.angle, period=360),
            "texts": dict(labels.text) == dict(out.labels.text),
            "grid": len(coord._grid_lines) == len(out.grid) and all(
                np.array_equal(path.vertices, pixel, equal_nan=True)
                and np.array_equal(path.codes, codes)
                for path, (pixel, codes) in zip(coord._grid_lines, out.grid)
            ),
            "drawn": f"{len(out.kept)}/{len(coord.drawn_boxes)}",
            "anchor": max_diff(
                [labels.xy[spine].get(i, (np.inf,) * 2) for i in out.xy[spine]],
                list(out.xy[spine].values()),
            ),
            "box": max_diff([box for _, _, box in out.kept], coord.drawn_boxes),
            "axis label": axis_label,
        })
    return rows


def main():
    ok = True
    print("bqplot demo vs WCSAxes; ticks and drawn are demo/WCSAxes, differences in display px or deg")
    print(f"{'case':4} {'view':6} {'coord':>5} {'ticks':>6} {'world':>6} {'pixel':>8} {'angle':>8} "
          f"{'texts':>6} {'grid':>6} {'drawn':>6} {'anchor':>7} {'box':>6} {'axis label':>10}")
    for name, case in demo.CASES.items():
        fig = demo.figure(name)
        scales = fig.marks[0].scales["x"], fig.marks[0].scales["y"]
        views = {"full": (fig.wcs_view, fig.wcs_layout)}
        # Set both ends of each scale at once, as PanZoom does; the figure
        # reruns the layout from its scale observers.
        for scale, (lo, hi) in zip(scales, ZOOMED[name]):
            with scale.hold_trait_notifications():
                scale.min, scale.max = lo, hi
        assert fig.wcs_view == ZOOMED[name]
        views["zoomed"] = (fig.wcs_view, fig.wcs_layout)

        for view, ((xlim, ylim), result) in views.items():
            for index, r in enumerate(compare(case, xlim, ylim, result)):
                exact = r["world"] and r["texts"] and r["grid"] and r["pixel"] < 1e-9 and r["angle"] < 1e-6
                exact &= len(set(r["ticks"].split("/"))) == 1 and len(set(r["drawn"].split("/"))) == 1
                exact &= r["axis label"] != "MISSING"
                ok &= exact
                label = r["axis label"]
                label = label if isinstance(label, str) else f"{label:.2f}"
                print(f"{name:4} {view:6} {index:>5} {r['ticks']:>6} {r['world']!s:>6} {r['pixel']:8.1e} "
                      f"{r['angle']:8.1e} {r['texts']!s:>6} {r['grid']!s:>6} {r['drawn']:>6} "
                      f"{r['anchor']:7.2f} {r['box']:6.2f} {label:>10}")
    print("exact parts agree" if ok else "MISMATCH in an exact part")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
