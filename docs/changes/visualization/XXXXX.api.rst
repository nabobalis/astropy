``astropy.visualization.wcsaxes`` can now be imported when matplotlib is not
installed. Its classes and functions still need matplotlib, and accessing one
without it raises ``ModuleNotFoundError``.
