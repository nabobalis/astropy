Added ``CoordinateHelper.get_ticks()`` and ``CoordinateHelper.get_ticklabels()``,
which return where the ticks of a coordinate cross each spine of the frame,
and their labels, computed from the current axes limits without drawing the
figure. The WCSAxes locators and formatters can now be used without
matplotlib.
