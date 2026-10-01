"""
Check that the branch moved the WCSAxes geometry into _layout unchanged.

    python check_move.py ASTROPY_CLONE [BASE [HEAD]]

BASE defaults to c55a2b2067, the commit the branch starts from, and HEAD to
wcsaxes-layout-core. For each piece of code that the branch moved into
astropy/visualization/wcsaxes/_layout.py, the script takes the old code at
BASE, applies the renames listed below (``self.coord_index`` to
``spec.coord_index`` and so on), and compares it with the new function at HEAD.
Docstrings, comments and line wrapping are ignored: both sides go through
``ast.unparse``. Comments are compared on their own.

Most moved lines only rename an attribute of ``self``, so git's own move
detection (``git diff --color-moved``) marks only part of the move. What this
prints for each piece is what differs beyond the renames: lines that stayed
in the matplotlib class, and the few lines that changed.
"""

import ast
import difflib
import io
import re
import subprocess
import sys
import textwrap
import tokenize

W = "astropy/visualization/wcsaxes/"
LAYOUT = W + "_layout.py"

# The attributes of CoordinateHelper that became fields of its CoordSpec, and
# the attributes of TickLabels that the label functions read from ``labels``
SPEC = [
    ("self._coord_scale_to_deg", "spec.coord_scale_to_deg"),
    ("self.coord_index", "spec.coord_index"),
    ("self.coord_type", "spec.coord_type"),
    ("self.coord_unit", "spec.coord_unit"),
    ("self.coord_wrap", "spec.coord_wrap"),
    ("self.locator", "spec.locator"),
]
LABELS = [
    (f"self.{key}", f"labels.{key}")
    for key in ("world", "data", "angle", "tick_angle", "text", "disp")
]

# (old file, old function or line ranges at BASE, new function, renames)
PIECES = [
    ("coordinate_helpers.py", "wrap_angle_at", "wrap_angle_at", []),
    (
        "frame.py",
        "Spine._update_normal",
        "spine_normal_angle",
        [("self.normal_angle = ", "return ")],
    ),
    (
        "frame.py",
        [(240, 247)],  # the resampling in BaseFrame.sample
        "resample_spine",
        [("spines[axis].data = ", "return ")],
    ),
    (
        "coordinate_helpers.py",
        "CoordinateHelper._update_ticks",
        "place_ticks",
        [
            ("*coord_range[self.coord_index]", "*coord_range"),
            (
                "tick_world_coordinates, self._fl_spacing",
                "tick_world_coordinates, spacing",
            ),
            ("self._fl_spacing", "spacing"),
            ("self._ticks.get_display_minor_ticks()", "spec.minor_locator is not None"),
            ("self._formatter_locator.minor_locator", "spec.minor_locator"),
            ("self.get_minor_frequency()", "spec.minor_frequency"),
            ("frame.items()", "spines.items()"),
            ("not isinstance(self.frame, RectangularFrame1D)", "not frame_1d"),
            ("invertedTransLimits.transform", "from_display"),
            ("transData.transform", "to_display"),
            ("self.transform.transform", "pixel_to_world"),
            ("self.frame.origin", "origin"),
            ("self._compute_ticks(", "_ticks_on_spine(spec, "),
            *SPEC,
        ],
    ),
    (
        "coordinate_helpers.py",
        "CoordinateHelper._compute_ticks",
        "_ticks_on_spine",
        SPEC,
    ),
    (
        "coordinate_helpers.py",
        "CoordinateHelper._update_grid_lines",
        "grid_lines",
        [
            ("self.transform.inverted().transform", "world_to_pixel"),
            ("self.transform.transform", "pixel_to_world"),
            ("self._grid_lines", "lines"),
            ("self._get_gridline(", "_gridline(spec, "),
            ("coord_range", "coord_ranges"),
            *SPEC,
        ],
    ),
    (
        "coordinate_helpers.py",
        "CoordinateHelper._get_gridline",
        "_gridline",
        [
            ("get_gridline_path(", "pixel, gridline_path_codes("),
            ("get_lon_lat_path(", "pixel, lon_lat_path_codes("),
            *SPEC,
        ],
    ),
    (
        "grid_paths.py",
        "get_lon_lat_path",
        "lon_lat_path_codes",
        [("Path.MOVETO", "MOVETO"), ("Path.LINETO", "LINETO")],
    ),
    (
        "grid_paths.py",
        "get_gridline_path",
        "gridline_path_codes",
        [("Path.MOVETO", "MOVETO"), ("Path.LINETO", "LINETO")],
    ),
    ("ticklabels.py", "sort_using", "sort_using", []),
    (
        "ticklabels.py",
        "_find_start_of_last_number",
        "find_start_of_last_number",
        [],
    ),
    ("ticklabels.py", "TickLabels.sort", "sort_labels", LABELS),
    (
        "ticklabels.py",
        "TickLabels.simplify_labels",
        "simplify_labels",
        [
            (
                r"/_find_start_of_last_number\((.+)\)$",
                r"find_start_of_last_number(\1, numerical_chars)",
            ),
            *LABELS,
        ],
    ),
    (
        "ticklabels.py",
        "TickLabels._set_xy_alignments",
        "anchor_tick_labels",
        [
            ("self._frame.parent_axes.transData.transform", "to_display"),
            ("self.xy", "xy"),
            *LABELS,
        ],
    ),
    (
        "ticklabels.py",
        "TickLabels.draw",
        "keep_tick_labels",
        [
            ("self.get_visible_axes()", "visible_axes"),
            ("self._get_bb(axis, i, renderer)", "extent(axis, i)"),
            ("self._exclude_overlapping", "exclude_overlapping"),
            (
                "bb.count_overlaps(self._all_bboxes + self._existing_bboxes)",
                "count_overlaps(bb, kept + existing)",
            ),
            *LABELS,
        ],
    ),
    (
        "frame.py",
        "Spine._halfway_x_y_angle",
        "spine_midpoint",
        [("self.normal_angle", "normal_angle")],
    ),
    (
        "axislabels.py",
        # In AxisLabels.draw: the rotation, and the placement without the
        # union of the tick-label boxes, which stays in AxisLabels
        [(98, 106), (108, 108), (123, 145)],
        "axis_label_position",
        [
            ("isinstance(self._frame, RectangularFrame)", "rectangular"),
            ("if visible:", "if union is not None:"),
            ("coord_ticklabels_bbox[axis][0].xmin", "x0"),
            ("coord_ticklabels_bbox[axis][0].x1", "x1"),
            ("coord_ticklabels_bbox[axis][0].ymin", "y0"),
            ("coord_ticklabels_bbox[axis][0].y1", "y1"),
        ],
    ),
]


def git_show(clone, rev, path):
    return subprocess.run(
        ["git", "-C", clone, "show", f"{rev}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def find(tree, qualname):
    node = tree
    for name in qualname.split("."):
        node = next(
            n
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
        )
    return node


def function_source(source, qualname):
    """The body of a function, without its docstring, and where it is."""
    node = find(ast.parse(source), qualname)
    body = node.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    lines = source.splitlines()
    # Take in the comments above the first statement
    start = body[0].lineno - 1
    while lines[start - 1].strip().startswith("#") or not lines[start - 1].strip():
        start -= 1
    lines = lines[start : node.end_lineno]
    return textwrap.dedent("\n".join(lines)), (node.lineno, node.end_lineno)


def range_source(source, ranges):
    lines = source.splitlines()
    picked = [line for start, end in ranges for line in lines[start - 1 : end]]
    return textwrap.dedent("\n".join(picked)), (ranges[0][0], ranges[-1][1])


def rename(code, renames):
    for old, new in renames:
        if old.startswith("/"):  # a regular expression, per line
            code = re.sub(old[1:], new, code, flags=re.MULTILINE)
        else:
            pattern = re.escape(old)
            if old[0].isalnum() or old[0] == "_":
                pattern = r"(?<![\w.])" + pattern
            if old[-1].isalnum() or old[-1] == "_":
                pattern += r"\b"
            code = re.sub(pattern, new.replace("\\", r"\\"), code)
    return code


def normalized(code):
    return ast.unparse(ast.parse(code)).splitlines()


def comments(code):
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    return [
        t.string.lstrip("# ").rstrip() for t in tokens if t.type == tokenize.COMMENT
    ]


def main(clone, base="c55a2b2067", head="wcsaxes-layout-core"):
    layout = git_show(clone, head, LAYOUT)
    same = old_only = new_only = 0
    for path, old_where, new_name, renames in PIECES:
        old_source = git_show(clone, base, W + path)
        if isinstance(old_where, str):
            old_code, (start, end) = function_source(old_source, old_where)
            label = f"{path} {old_where}"
        else:
            old_code, (start, end) = range_source(old_source, old_where)
            label = f"{path} lines " + ", ".join(f"{a}-{b}" for a, b in old_where)
        new_code, (new_start, new_end) = function_source(layout, new_name)

        old = normalized(rename(old_code, renames))
        new = normalized(new_code)
        matched = sum(
            block.size
            for block in difflib.SequenceMatcher(None, old, new).get_matching_blocks()
        )
        same += matched
        old_only += len(old) - matched
        new_only += len(new) - matched
        code_diff = list(
            difflib.unified_diff(old, new, "old, renamed", "new", n=1, lineterm="")
        )
        comment_diff = list(
            difflib.unified_diff(
                comments(old_code), comments(new_code), "old", "new", n=0, lineterm=""
            )
        )
        print(
            f"== {label} ({start}-{end}) -> _layout.{new_name} ({new_start}-{new_end})"
        )
        if renames:
            print(
                "   renames: "
                + "; ".join(f"{o.lstrip('/')} -> {n}" for o, n in renames)
            )
        if not code_diff and not comment_diff:
            print("   identical after the renames")
        for line in code_diff[2:]:
            print("   code    " + line)
        for line in comment_diff[2:]:
            if not line.startswith("@@"):
                print("   comment " + line)
        print()
    print(
        f"Counting lines as ast.unparse writes them, {same} are the same after "
        f"the renames. {old_only} old lines did not move or were replaced, and "
        f"{new_only} new lines replace them."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:])
