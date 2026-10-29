Reading the values of a parsed FITS header is faster. When the value of a
card is read, ``strip_header_whitespace`` is now read from the configuration
only if the value is a string, and ``enable_record_valued_keyword_cards`` only
if the card image contains ``': '``.
