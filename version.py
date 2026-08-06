"""Single source of truth for the application version.

Same convention as exp-note: one constant in the code, surfaced by the app and
mirrored in the packaging metadata. Semantic versioning — MAJOR.MINOR.PATCH:
  MAJOR  incompatible change to the on-disk output or the reading of .ktf
  MINOR  new capability, backwards compatible
  PATCH  fixes only
"""

__version__ = "1.3.1"
APP_NAME = "KTF Viewer"
