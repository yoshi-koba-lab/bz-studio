"""Single source of truth for the application version.

Same convention as exp-note: one constant in the code, surfaced by the app and
mirrored in the packaging metadata. Semantic versioning — MAJOR.MINOR.PATCH:
  MAJOR  incompatible change to the on-disk output or the reading of .ktf
  MINOR  a substantial new capability
  PATCH  ordinary updates: fixes and small improvements (the usual bump)
"""

__version__ = "2.1.1"
APP_NAME = "BZ Studio"
# Keep the established QSettings namespace so upgrading does not hide saved
# Conditions, recent folders, or Series Builder labels.
SETTINGS_APP_NAME = "BZ Plate Studio"
