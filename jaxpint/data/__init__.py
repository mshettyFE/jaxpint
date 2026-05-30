"""Packaged data for JaxPINT.

Currently holds the clock subsystem's committed schema
(``clock/clock_metadata.json``).  The bulk clock time-series (``*.clk``/``*.dat``,
``index.txt``, ``SNAPSHOT.json``) is *not* committed -- it is downloaded into
``clock/`` (or ``$JAXPINT_CLOCK_DIR``) at runtime and is gitignored.
"""
