#!/usr/bin/env python3
"""Fail a release build if unwanted Qt modules or license files are missing."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


FORBIDDEN = (
    "pyqt",
    "virtualkeyboard",
    "qpdf",
    "qtpdf",
    "qt6pdf",
    "qtgraphs",
    "qt6graphs",
    "qthttpserver",
    "qt6httpserver",
    "qtnetworkauth",
    "qt6networkauth",
    "qtquick3d",
    "qt6quick3d",
    "qtgrpc",
    "qt6grpc",
    "qtlottie",
    "qt6lottie",
    "qtmqtt",
    "qt6mqtt",
    "qtcoap",
    "qt6coap",
    "quicktimeline",
    "waylandcompositor",
    "qmlcompiler",
    "qtwebengine",
    "qt6webengine",
)

# Dependency names as they appear in Mach-O, PE, or ELF import tables.  Scan
# binary payloads as a second line of defence: a forbidden framework/DLL could
# otherwise be referenced under an innocuous renamed filename.
FORBIDDEN_BINARY_REFERENCES = (
    b"qt6virtualkeyboard", b"qtvirtualkeyboard.framework",
    b"libqt6virtualkeyboard", b"qt6pdf", b"qtpdf.framework", b"libqt6pdf",
    b"qt6graphs", b"qtgraphs.framework", b"libqt6graphs",
    b"qt6httpserver", b"qthttpserver.framework", b"libqt6httpserver",
    b"qt6networkauth", b"qtnetworkauth.framework", b"libqt6networkauth",
    b"qt6quick3d", b"qtquick3d.framework", b"libqt6quick3d",
    b"qt6grpc", b"qtgrpc.framework", b"libqt6grpc",
    b"qt6webengine", b"qtwebengine.framework", b"libqt6webengine",
)

REQUIRED_NOTICE_FILES = {
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "qt-source-assets.sha256",
    "GPL-3.0.txt",
    "LGPL-3.0.txt",
    "GFDL-1.3.txt",
    "PYTHON-PSF-2.0.txt",
    "QT-6.10.3-THIRD-PARTY-NOTICES.html",
    "PYSIDE6-6.10.3-THIRD-PARTY-NOTICES.html",
}

# The Qt index page links to individual attribution/license statements.  Each
# module that is actually frozen into BZ Studio must have its complete linked
# statements available offline; an index page by itself is not sufficient.
REQUIRED_QT_ATTRIBUTION_SECTIONS = (
    ("qt-core", "qtcore-attribution-"),
    ("qt-d-bus", "qtdbus-attribution-"),
    ("qt-gui", "qtgui-attribution-"),
    ("qt-image-formats", "qtimageformats-attribution-"),
    ("qt-network", "qtnetwork-attribution-"),
    ("qt-svg", "qtsvg-attribution-"),
)


def _linked_attribution_files(index_text: str) -> set[str]:
    """Return every attribution page linked by the bundled-module sections."""
    headings = list(re.finditer(r'<h2\s+id=["\\\']([^"\\\']+)["\\\']', index_text))
    found: set[str] = set()
    missing_sections = []
    for section, expected_prefix in REQUIRED_QT_ATTRIBUTION_SECTIONS:
        heading_index = next(
            (index for index, match in enumerate(headings)
             if match.group(1) == section),
            None,
        )
        if heading_index is None:
            missing_sections.append(section)
            continue
        start = headings[heading_index].start()
        end = (
            headings[heading_index + 1].start()
            if heading_index + 1 < len(headings)
            else len(index_text)
        )
        links = {
            match.group(1).rsplit("/", 1)[-1]
            for match in re.finditer(
                r'href=["\\\']([^"\\\']*-attribution-[^"\\\']+\.html)["\\\']',
                index_text[start:end],
            )
            if match.group(1).rsplit("/", 1)[-1].startswith(expected_prefix)
        }
        if not links:
            missing_sections.append(section)
        found.update(links)
    if missing_sections:
        raise RuntimeError(
            "Qt notice index has no bundled-module attribution section for: "
            + ", ".join(missing_sections))
    return found


def verify(root: Path) -> dict:
    root = root.resolve()
    if not root.exists():
        raise RuntimeError(f"bundle does not exist: {root}")

    files = [path for path in root.rglob("*") if path.is_file()]
    relative = [path.relative_to(root).as_posix() for path in files]
    forbidden = sorted(
        name for name in relative
        if any(token in name.casefold() for token in FORBIDDEN)
    )
    if forbidden:
        sample = "\n  ".join(forbidden[:30])
        raise RuntimeError(f"forbidden Qt/PyQt artifacts in bundle:\n  {sample}")

    binary_files = [
        path for path, name in zip(files, relative)
        if path.suffix.casefold() in {".dll", ".dylib", ".so", ".pyd", ".exe"}
        or ".framework/" in name.casefold()
        or "/contents/macos/" in f"/{name.casefold()}"
    ]
    forbidden_references = []
    for path in binary_files:
        payload = path.read_bytes().lower()
        matches = [
            token.decode("ascii") for token in FORBIDDEN_BINARY_REFERENCES
            if token in payload
        ]
        if matches:
            forbidden_references.append(
                (path.relative_to(root).as_posix(), sorted(set(matches))))
    if forbidden_references:
        sample = "\n  ".join(
            f"{name}: {', '.join(tokens)}"
            for name, tokens in forbidden_references[:30])
        raise RuntimeError(f"forbidden Qt binary dependencies in bundle:\n  {sample}")

    basenames = {path.name for path in files}
    missing = sorted(REQUIRED_NOTICE_FILES - basenames)
    if missing:
        raise RuntimeError(f"required license/notice files missing: {', '.join(missing)}")

    qt_index = next(
        path for path in files
        if path.name == "QT-6.10.3-THIRD-PARTY-NOTICES.html")
    required_attributions = _linked_attribution_files(
        qt_index.read_text(encoding="utf-8"))
    missing_attributions = sorted(required_attributions - basenames)
    if missing_attributions:
        raise RuntimeError(
            "required offline Qt attribution pages missing: "
            + ", ".join(missing_attributions))

    if not any("pyside6" in name.casefold() for name in relative):
        raise RuntimeError("PySide6 runtime was not found in the frozen bundle")
    if not any("qtcore" in name.casefold() for name in relative):
        raise RuntimeError("QtCore runtime was not found in the frozen bundle")

    return {
        "ok": True,
        "bundle": str(root),
        "file_count": len(files),
        "required_notices": sorted(REQUIRED_NOTICE_FILES),
        "qt_attribution_pages": len(required_attributions),
        "forbidden_artifacts": 0,
        "binary_dependencies_scanned": len(binary_files),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} BUNDLE", file=sys.stderr)
        return 2
    try:
        report = verify(Path(argv[1]))
    except Exception as exc:
        print(f"bundle verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
