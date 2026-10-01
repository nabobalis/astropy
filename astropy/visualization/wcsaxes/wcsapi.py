# Functions/classes for WCSAxes related to APE14 WCSes
from __future__ import annotations

from astropy.coordinates import ICRS, BaseCoordinateFrame, SkyCoord
from astropy.wcs import WCS

from ._model import (
    CUSTOM_UCD_COORD_META_MAPPING,  # noqa: F401 (moved to _model, re-exported)
    UCD_COORD_META_MAPPING,  # noqa: F401 (moved to _model, re-exported)
    apply_slices,  # noqa: F401 (moved to _model, re-exported)
    coord_meta_from_wcs,
    custom_ucd_coord_meta_mapping,
    default_positions,
    wcs_pixel_to_world,
    wcs_world_to_pixel,
)
from .frame import EllipticalFrame, RectangularFrame, RectangularFrame1D
from .transforms import CurvedTransform

__all__ = [
    "WCSPixel2WorldTransform",
    "WCSWorld2PixelTransform",
    "custom_ucd_coord_meta_mapping",
    "transform_coord_meta_from_wcs",
]

IDENTITY = WCS(naxis=2)
IDENTITY.wcs.ctype = ["X", "Y"]
IDENTITY.wcs.crval = [0.0, 0.0]
IDENTITY.wcs.crpix = [1.0, 1.0]
IDENTITY.wcs.cdelt = [1.0, 1.0]


def transform_coord_meta_from_wcs(wcs, frame_class, slices=None):
    """
    The transform and coordinate metadata of a WCSAxes for a WCS.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS.
    frame_class : type
        The `~astropy.visualization.wcsaxes.frame.BaseFrame` subclass of the
        axes, which sets the default tick, tick label and axis label positions.
    slices : tuple, optional
        For a WCS with more than two pixel dimensions, one element per pixel
        dimension: ``'x'``, ``'y'``, an integer or a slice.

    Returns
    -------
    transform : `WCSPixel2WorldTransform`
        The pixel-to-world transform of the sliced WCS.
    coord_meta : dict
        The coordinate metadata, from
        `~astropy.visualization.wcsaxes._model.coord_meta_from_wcs`, plus the
        default positions.
    """
    coord_meta, transform_wcs, invert_xy = coord_meta_from_wcs(wcs, slices)
    transform = WCSPixel2WorldTransform(transform_wcs, invert_xy=invert_xy)
    _add_default_positions(coord_meta, frame_class)
    return transform, coord_meta


def _frame_kind(frame_class):
    """
    The name of a frame class for the model, or 'custom'.
    """
    kinds = {
        RectangularFrame: "rectangular",
        RectangularFrame1D: "rectangular1d",
        EllipticalFrame: "elliptical",
    }
    return kinds.get(frame_class, "custom")


def _add_default_positions(coord_meta, frame_class):
    """
    Add the default tick, tick label and axis label positions to coord_meta.
    """
    default_positions(coord_meta, _frame_kind(frame_class), frame_class.spine_names)


def wcsapi_to_celestial_frame(wcs):
    for cls, _, kwargs, *_ in wcs.world_axis_object_classes.values():
        if issubclass(cls, SkyCoord):
            return kwargs.get("frame", ICRS())
        elif issubclass(cls, BaseCoordinateFrame):
            return cls(**kwargs)


class WCSWorld2PixelTransform(CurvedTransform):
    """
    WCS transformation from world to pixel coordinates.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS defining the transformation, which should have at most two
        pixel dimensions.
    invert_xy : bool, optional
        Whether to swap the two pixel coordinates.
    """

    has_inverse = True
    frame_in = None
    units_in = None

    def __init__(self, wcs, invert_xy=False):
        super().__init__()

        if wcs.pixel_n_dim > 2:
            raise ValueError("Only pixel_n_dim =< 2 is supported")

        self.wcs = wcs
        self.invert_xy = invert_xy

        self.frame_in = wcsapi_to_celestial_frame(wcs)
        self.units_in = wcs.world_axis_units

    def __hash__(self):
        return hash((type(self), self.wcs, self.invert_xy))

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.wcs is other.wcs
            and self.invert_xy == other.invert_xy
        )

    @property
    def input_dims(self):
        return self.wcs.world_n_dim

    def transform(self, world):
        return wcs_world_to_pixel(self.wcs, world, invert_xy=self.invert_xy)

    transform_non_affine = transform

    def inverted(self):
        """
        Return the inverse of the transform.
        """
        return WCSPixel2WorldTransform(self.wcs, invert_xy=self.invert_xy)


class WCSPixel2WorldTransform(CurvedTransform):
    """
    WCS transformation from pixel to world coordinates.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS defining the transformation, which should have at most two
        pixel dimensions.
    invert_xy : bool, optional
        Whether to swap the two pixel coordinates.
    """

    has_inverse = True

    def __init__(self, wcs, invert_xy=False):
        super().__init__()

        if wcs.pixel_n_dim > 2:
            raise ValueError("Only pixel_n_dim =< 2 is supported")

        self.wcs = wcs
        self.invert_xy = invert_xy

        self.frame_out = wcsapi_to_celestial_frame(wcs)
        self.units_out = wcs.world_axis_units

    def __hash__(self):
        return hash((type(self), self.wcs, self.invert_xy))

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.wcs is other.wcs
            and self.invert_xy == other.invert_xy
        )

    @property
    def output_dims(self):
        return self.wcs.world_n_dim

    def transform(self, pixel):
        return wcs_pixel_to_world(self.wcs, pixel, invert_xy=self.invert_xy)

    transform_non_affine = transform

    def inverted(self):
        """
        Return the inverse of the transform.
        """
        return WCSWorld2PixelTransform(self.wcs, invert_xy=self.invert_xy)
