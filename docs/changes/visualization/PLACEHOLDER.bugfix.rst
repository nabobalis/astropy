Fixed a ``ValueError`` when drawing the ticks or grid of a ``WCSAxes``
coordinate that is NaN over the whole view, for example a WCS that returns NaN
outside its domain.
