# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
from collections import defaultdict

from matplotlib import rcParams
from matplotlib.artist import allow_rasterization
from matplotlib.text import Text

from astropy.utils.decorators import deprecated_renamed_argument
from astropy.utils.exceptions import AstropyDeprecationWarning

from . import _layout

# sort_using moved to _layout. It is imported here for code that imports it
# from this module.
from ._layout import sort_using  # noqa: F401


class TickLabels(Text):
    def __init__(self, frame, *args, **kwargs):
        self.clear()
        self._frame = frame
        super().__init__(*args, **kwargs)
        self.set_clip_on(True)
        self.set_visible_axes("all")
        self.set_pad(rcParams["xtick.major.pad"])
        self._exclude_overlapping = False
        self._simplify = True

        # Mapping from axis > list[bounding boxes]
        self._axis_bboxes = defaultdict(list)

        # Stale if either xy positions haven't been calculated, or if
        # something changes that requires recomputing the positions
        self._stale = True

        # Check rcParams
        if "color" not in kwargs:
            self.set_color(rcParams["xtick.color"])

        if "size" not in kwargs:
            self.set_size(rcParams["xtick.labelsize"])

    def clear(self):
        self.world = defaultdict(list)
        self.data = defaultdict(list)
        self.angle = defaultdict(list)
        self.tick_angle = defaultdict(list)
        self.text = defaultdict(list)
        self.disp = defaultdict(list)

    def add(
        self,
        axis=None,
        world=None,
        pixel=None,
        angle=None,
        text=None,
        axis_displacement=None,
        data=None,
        tick_angle=None,
    ):
        """
        Add a label.

        Parameters
        ----------
        axis : str
            Axis to add label to.
        world : Quantity
            Coordinate value along this axis.
        pixel : [float, float]
            Pixel coordinates of the label. Deprecated and no longer used.
        angle : float
            Angle of the label.
        text : str
            Label text.
        axis_displacement : float
            Displacement from axis.
        data : [float, float]
            Data coordinates of the label.
        tick_angle : float
            Angle of the corresponding tick. Defaults to be equal to ``angle``.
        """
        required_args = ["axis", "world", "angle", "text", "axis_displacement", "data"]
        if pixel is not None:
            warnings.warn(
                "Setting the pixel coordinates of a label does nothing and is"
                " deprecated, as these can only be accurately calculated when"
                " Matplotlib is drawing a figure. To prevent this warning pass the"
                f" following arguments as keyword arguments: {required_args}",
                AstropyDeprecationWarning,
            )
        if (
            axis is None
            or world is None
            or angle is None
            or text is None
            or axis_displacement is None
            or data is None
        ):
            raise TypeError(
                f"All of the following arguments must be provided: {required_args}"
            )

        self.world[axis].append(world)
        self.data[axis].append(data)
        self.angle[axis].append(angle)
        self.tick_angle[axis].append(tick_angle if tick_angle is not None else angle)
        self.text[axis].append(text)
        self.disp[axis].append(axis_displacement)

        self._stale = True

    def sort(self):
        """
        Sort by axis displacement, which allows us to figure out which parts
        of labels to not repeat.
        """
        _layout.sort_labels(self)
        self._stale = True

    def simplify_labels(self):
        """
        Figure out which parts of labels can be dropped to avoid repetition.
        """
        self.sort()

        numerical_chars = "0123456789.+"
        if rcParams["axes.unicode_minus"] and not rcParams["text.usetex"]:
            numerical_chars += "\N{MINUS SIGN}"
        else:
            numerical_chars += "-"

        _layout.simplify_labels(self, numerical_chars)

        self._stale = True

    def set_pad(self, value):
        self._pad = value
        self._stale = True

    def get_pad(self):
        return self._pad

    def set_visible_axes(self, visible_axes):
        self._visible_axes = self._frame._validate_positions(visible_axes)
        self._stale = True

    def get_visible_axes(self):
        if self._visible_axes == "all":
            return list(self._frame.keys())
        else:
            return [x for x in self._visible_axes if x in self._frame or x == "#"]

    def set_exclude_overlapping(self, exclude_overlapping):
        self._exclude_overlapping = exclude_overlapping

    def set_simplify(self, simplify):
        self._simplify = simplify

    def _set_xy_alignments(self, renderer):
        """
        Compute and set the x, y positions and the horizontal/vertical alignment of
        each label.

        Parameters
        ----------
        renderer : `~matplotlib.backend_bases.RendererBase`
            The renderer to use to compute text sizes.
        """
        if not self._stale:
            return

        if self._simplify:
            self.simplify_labels()

        window_extent = super().get_window_extent

        def measure(text, x, y):
            # Set initial position and find bounding box
            self.set_text(text)
            self.set_position((x, y))
            bb = window_extent(renderer)
            return bb.width, bb.height

        visible_axes = self.get_visible_axes()

        # CoordinateHelper sets the tick size before it draws the labels. As
        # before, it is only read when there is a label to place.
        pad = None
        if any(text != "" for axis in visible_axes for text in self.text.get(axis, [])):
            pad = renderer.points_to_pixels(self.get_pad() + self._tick_out_size)

        self.xy = _layout.anchor_tick_labels(
            self,
            visible_axes,
            self._frame.parent_axes.transData.transform,
            pad,
            measure,
        )
        self.ha = {axis: dict.fromkeys(xy, "center") for axis, xy in self.xy.items()}
        self.va = {axis: dict.fromkeys(xy, "center") for axis, xy in self.xy.items()}

        self._stale = False

    def _get_bb(self, axis, i, renderer):
        """
        Get the bounding box of an individual label. n.b. _set_xy_alignment()
        must be called before this method.

        Parameters
        ----------
        axis : str
            The axis the label is on.
        i : int
            The index of the label along the axis.
        renderer : `~matplotlib.backend_bases.RendererBase`
            The renderer to use to compute text sizes.
        """
        if self.text[axis][i] == "":
            return

        self.set_text(self.text[axis][i])
        self.set_position(self.xy[axis][i])
        self.set_ha(self.ha[axis][i])
        self.set_va(self.va[axis][i])
        return super().get_window_extent(renderer)

    @property
    def _all_bboxes(self):
        # List of all tick label bounding boxes
        ret = []
        for axis in self._axis_bboxes:
            ret += self._axis_bboxes[axis]
        return ret

    def _set_existing_bboxes(self, bboxes):
        self._existing_bboxes = bboxes

    @allow_rasterization
    @deprecated_renamed_argument(old_name="bboxes", new_name=None, since="6.0")
    @deprecated_renamed_argument(old_name="ticklabels_bbox", new_name=None, since="6.0")
    @deprecated_renamed_argument(old_name="tick_out_size", new_name=None, since="6.0")
    def draw(self, renderer, bboxes=None, ticklabels_bbox=None, tick_out_size=None):
        # Reset bounding boxes
        self._axis_bboxes = defaultdict(list)

        if not self.get_visible():
            return

        self._set_xy_alignments(renderer)

        # As with the tick size, the boxes of the other coordinates are only
        # read when they are needed.
        for axis, _, bb in _layout.keep_tick_labels(
            self,
            self.get_visible_axes(),
            lambda axis, i: self._get_bb(axis, i, renderer),
            self._exclude_overlapping,
            self._existing_bboxes if self._exclude_overlapping else [],
        ):
            super().draw(renderer)
            self._axis_bboxes[axis].append(bb)
