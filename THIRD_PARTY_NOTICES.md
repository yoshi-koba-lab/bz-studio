# BZ Studio — Third-Party Notices

This document applies to the standalone BZ Studio application. BZ Studio's
original code remains subject to the repository `LICENSE`; each component below
remains subject to its own license. Nothing in the BZ Studio license limits the
rights granted for these components.

## Qt for Python / Qt

BZ Studio 2.0.0 uses the following separately dynamically linked components:

- Qt for Python **PySide6-Essentials 6.10.3** and **Shiboken6 6.10.3**
- Qt **6.10.3** Core, Gui, Widgets, Network, DBus and Svg libraries, plus the
  platform, style, icon and image-format plugins selected by the release build
  (including Cocoa/offscreen/minimal, SVG, TIFF, WebP, JPEG, GIF and related
  formats where available on the target platform), and Qt translation catalogs

Copyright © The Qt Company Ltd. and other contributors. BZ Studio distributes
these components under the **GNU Lesser General Public License version 3.0
only (LGPL-3.0-only)** option offered upstream. They are not covered by BZ
Studio's proprietary license.

The required license texts accompany every build:

- `licenses/LGPL-3.0.txt`
- `licenses/GPL-3.0.txt` (incorporated by reference by LGPLv3)
- `licenses/GFDL-1.3.txt` (license for the redistributed Qt documentation)
- `licenses/QT-6.10.3-THIRD-PARTY-NOTICES.html`
- `licenses/PYSIDE6-6.10.3-THIRD-PARTY-NOTICES.html`

The two HTML files above are offline snapshots of the official Qt 6.10 and Qt
for Python 6.10 third-party attribution/license indexes. The `licenses`
directory also contains all 53 official attribution pages linked by the index
for the bundled Qt Core, DBus, Gui, Image Formats, Network and Svg modules. No
license or attribution text was edited. The complete set is included inside the
application so these notices remain available without network access.
These official offline Qt documentation pages retain their copyright and
license notices and are redistributed under the GNU Free Documentation License
version 1.3, with their documentation content unmodified. The complete GFDL
1.3 text is included as `licenses/GFDL-1.3.txt`.

Corresponding source for the exact release is available from the upstream
release archives:

- PySide6/Shiboken6 6.10.3:
  <https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.10.3-src/pyside-setup-everywhere-src-6.10.3.tar.xz>
- Qt Base 6.10.3:
  <https://download.qt.io/official_releases/qt/6.10/6.10.3/submodules/qtbase-everywhere-src-6.10.3.tar.xz>
- Qt SVG 6.10.3:
  <https://download.qt.io/official_releases/qt/6.10/6.10.3/submodules/qtsvg-everywhere-src-6.10.3.tar.xz>
- Qt Image Formats 6.10.3:
  <https://download.qt.io/official_releases/qt/6.10/6.10.3/submodules/qtimageformats-everywhere-src-6.10.3.tar.xz>
- Qt Translations 6.10.3:
  <https://download.qt.io/official_releases/qt/6.10/6.10.3/submodules/qttranslations-everywhere-src-6.10.3.tar.xz>
- Qt third-party acknowledgements and license texts:
  <https://doc.qt.io/qt-6.10/licenses-used-in-qt.html>
- Qt for Python third-party acknowledgements:
  <https://doc.qt.io/qtforpython-6.10/licenses.html>

The five exact source archives above, including their upstream license and
third-party attribution files, are attached to every tagged BZ Studio release.
Their SHA-256 digests are recorded in `qt-source-assets.sha256` and verified by
the release workflow before publication.

### Replacing the LGPL libraries

The frozen application keeps PySide6 and Qt as separate shared libraries. You
may replace them with ABI-compatible modified builds; recombine or relink the
application; run the resulting Combined Work; and exercise all other rights
granted by LGPLv3. You may reverse engineer BZ Studio to the extent needed to
debug such LGPL modifications. BZ Studio imposes no technical restriction on
that replacement.

- Windows: replace the relevant PySide6/Qt DLLs inside the unpacked
  `BZ Studio` directory while preserving their relative paths.
- macOS: replace the relevant PySide6/Qt dynamic libraries or frameworks inside
  `BZ Studio.app`. Replacing a library invalidates the ad-hoc signature; the
  modified local copy can be signed again with
  `codesign --force --deep --sign - "/path/to/BZ Studio.app"`.

Keep an original copy before replacement. Modified third-party libraries are
unsupported by the BZ Studio authors, but that does not reduce LGPL rights.

## Other runtime components

The standalone build also uses or contains:

| Component | License |
| --- | --- |
| Python | Python Software Foundation License 2.0 |
| NumPy | BSD 3-Clause (plus notices for bundled numerical libraries) |
| SciPy | BSD 3-Clause (plus notices for bundled numerical libraries) |
| Pillow | HPND / Historical Permission Notice and Disclaimer |
| tifffile | BSD 3-Clause |
| imagecodecs and bundled codec libraries | BSD 3-Clause and the component-specific licenses shipped by imagecodecs |

The release bundle includes each installed Python distribution's metadata and
license material. The imagecodecs `licenses` directory is also copied in full
because the application explicitly bundles its lazily loaded codec modules.
Python's complete license is included as `licenses/PYTHON-PSF-2.0.txt`.

## Build tool

PyInstaller is used to create the standalone application. Its bootloader is
licensed under GPL-2.0-or-later with the upstream exception that permits its use
to build and distribute non-free programs. PyInstaller itself is a build-time
dependency and is not an application framework.

This notice is informational and is not a substitute for the complete license
texts included in the distribution.
