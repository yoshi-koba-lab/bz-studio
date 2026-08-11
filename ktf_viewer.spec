# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — builds a standalone app on macOS (.app) and Windows (folder).
# Build:  pyinstaller ktf_viewer.spec
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

APP_NAME = "BZ Studio"
_ICON = "app_icon.icns" if sys.platform == "darwin" else "app_icon.ico"
ICON = _ICON if os.path.exists(_ICON) else None  # optional; add later for a custom icon

# Preserve the upstream license files shipped by runtime Python distributions.
# PySide6/Qt's LGPL and GPL texts are kept as project files because the wheels
# do not include them in dist-info.
RUNTIME_METADATA = []
for distribution in (
        "PySide6_Essentials", "shiboken6", "numpy", "Pillow", "tifffile",
        "imagecodecs", "scipy"):
    RUNTIME_METADATA += copy_metadata(distribution)

NOTICE_DATA = [
    ("LICENSE", "."),
    ("THIRD_PARTY_NOTICES.md", "."),
    ("qt-source-assets.sha256", "."),
    # Keep the official Qt/PySide index pages and every linked attribution page
    # together so the relative links continue to work without network access.
    ("licenses", "licenses"),
]
CODEC_LICENSE_DATA = collect_data_files("imagecodecs", includes=["licenses/**"])

# PySide6-Essentials wheels contain a few optional plugins outside the modules
# used by this application.  In particular Qt Virtual Keyboard is GPL-only and
# the qpdf image plugin has an unavailable QtPdf dependency.  PyInstaller's
# QtGui hook discovers plugins by category, so module excludes alone are not
# sufficient: filter every collected binary/data entry and verify the finished
# bundle independently in CI.
FORBIDDEN_QT_ARTIFACTS = (
    "virtualkeyboard", "qpdf", "qtpdf", "qt6pdf", "qtgraphs", "qt6graphs",
    "qthttpserver", "qt6httpserver", "qtnetworkauth", "qt6networkauth",
    "qtquick3d", "qt6quick3d", "qtgrpc", "qt6grpc", "qtlottie", "qt6lottie",
    "qtmqtt", "qt6mqtt", "qtcoap", "qt6coap", "quicktimeline",
    "waylandcompositor", "qmlcompiler", "qtwebengine", "qt6webengine",
)


def _filter_forbidden_qt_artifacts(entries):
    kept = []
    for entry in entries:
        candidate = " ".join(str(value) for value in entry[:2]).replace("\\", "/").lower()
        if any(token in candidate for token in FORBIDDEN_QT_ARTIFACTS):
            print(f"BZ Studio: excluded unused/restricted Qt artifact: {entry[0]}")
            continue
        kept.append(entry)
    return kept

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=NOTICE_DATA + RUNTIME_METADATA + CODEC_LICENSE_DATA,
    # imagecodecs resolves the compiled codec modules lazily with importlib.
    # PyInstaller cannot discover those imports from the module graph, yet the
    # ordinary-image workflow needs them for LZW input and compressed OME-TIFF
    # output.  Bundle the codec modules explicitly and verify them in the
    # packaged-app smoke test.
    hiddenimports=collect_submodules("imagecodecs"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter", "matplotlib", "PyQt5", "PyQt6",
        "PySide6.QtPdf", "PySide6.QtGraphs", "PySide6.QtHttpServer",
        "PySide6.QtNetworkAuth", "PySide6.QtQuick3D",
        "PySide6.QtVirtualKeyboard", "PySide6.QtWebEngineCore",
    ],
    noarchive=False,
)
a.binaries = _filter_forbidden_qt_artifacts(a.binaries)
a.datas = _filter_forbidden_qt_artifacts(a.datas)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    # The app opens datasets through its own dialogs and does not consume
    # Finder-open command-line events.  PyInstaller's argv emulation starts an
    # Apple Events/XPC bridge, which is unnecessary and can abort in restricted
    # execution environments before package smoke mode reaches main().
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=ICON,
        bundle_identifier="io.github.ktf-viewer",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": "2.0.1",
            "CFBundleVersion": "2.0.1",
        },
    )
