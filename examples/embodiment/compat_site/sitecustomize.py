"""Compatibility hooks loaded before DreamDojo/OpenPI imports."""

import datetime as _datetime


if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc

try:
    import coverage

    coverage.process_startup()
except ImportError:
    pass
