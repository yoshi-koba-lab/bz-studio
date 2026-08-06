# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — builds a standalone app on macOS (.app) and Windows (folder).
# Build:  pyinstaller ktf_viewer.spec
import os
import sys

APP_NAME = "BZ Plate Studio"
_ICON = "app_icon.icns" if sys.platform == "darwin" else "app_icon.ico"
ICON = _ICON if os.path.exists(_ICON) else None  # optional; add later for a custom icon

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PyQt5", "PySide6"],
    noarchive=False,
)
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
    console=False,             # windowed GUI app (no terminal)
    disable_windowed_traceback=False,
    argv_emulation=(sys.platform == "darwin"),
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
            "CFBundleShortVersionString": "1.4.1",
        },
    )
