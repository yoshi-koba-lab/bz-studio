"""Single source of truth for the application version.

Same convention as exp-note: one constant in the code, surfaced by the app and
mirrored in the packaging metadata. Semantic versioning — MAJOR.MINOR.PATCH:
  MAJOR  incompatible change to the on-disk output or the reading of .ktf
  MINOR  a substantial new capability
  PATCH  ordinary updates: fixes and small improvements (the usual bump)
"""

__version__ = "1.4.1"
APP_NAME = "BZ Plate Studio"
