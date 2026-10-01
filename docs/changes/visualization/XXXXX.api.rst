``astropy.visualization.wcsaxes`` can now be imported when matplotlib is not
installed. Its classes and functions still need matplotlib, apart from
``custom_ucd_coord_meta_mapping``, and accessing one without it raises
``ModuleNotFoundError``.
