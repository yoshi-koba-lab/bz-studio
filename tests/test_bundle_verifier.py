from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.verify_frozen_bundle import (
    REQUIRED_NOTICE_FILES,
    REQUIRED_QT_ATTRIBUTION_SECTIONS,
    verify,
)


class FrozenBundleVerifierTests(unittest.TestCase):
    def _valid_bundle(self, root: Path) -> Path:
        bundle = root / "BZ Studio"
        for name in REQUIRED_NOTICE_FILES:
            path = bundle / "Resources" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
        index_sections = []
        for section, prefix in REQUIRED_QT_ATTRIBUTION_SECTIONS:
            filename = f"{prefix}sample.html"
            index_sections.append(
                f'<h2 id="{section}"></h2><a href="{filename}"></a>')
            path = bundle / "Resources" / "licenses" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(prefix, encoding="utf-8")
        index = (
            bundle / "Resources"
            / "QT-6.10.3-THIRD-PARTY-NOTICES.html")
        index.write_text("".join(index_sections), encoding="utf-8")
        runtime = bundle / "Frameworks" / "PySide6" / "QtCore.dll"
        runtime.parent.mkdir(parents=True, exist_ok=True)
        runtime.write_bytes(b"runtime")
        return bundle

    def test_accepts_minimal_pyside_bundle_with_notices(self):
        with tempfile.TemporaryDirectory(prefix="bz-bundle-gate-") as temp:
            report = verify(self._valid_bundle(Path(temp)))
        self.assertTrue(report["ok"])
        self.assertEqual(report["forbidden_artifacts"], 0)

    def test_rejects_gpl_only_virtual_keyboard_plugin(self):
        with tempfile.TemporaryDirectory(prefix="bz-bundle-gate-") as temp:
            bundle = self._valid_bundle(Path(temp))
            plugin = bundle / "plugins" / "libqtvirtualkeyboardplugin.dylib"
            plugin.parent.mkdir(parents=True, exist_ok=True)
            plugin.write_bytes(b"plugin")
            with self.assertRaisesRegex(RuntimeError, "forbidden Qt/PyQt"):
                verify(bundle)

    def test_rejects_missing_license_text(self):
        with tempfile.TemporaryDirectory(prefix="bz-bundle-gate-") as temp:
            bundle = self._valid_bundle(Path(temp))
            (bundle / "Resources" / "LGPL-3.0.txt").unlink()
            with self.assertRaisesRegex(RuntimeError, "required license"):
                verify(bundle)

    def test_rejects_missing_module_attribution_pages(self):
        with tempfile.TemporaryDirectory(prefix="bz-bundle-gate-") as temp:
            bundle = self._valid_bundle(Path(temp))
            page = (
                bundle / "Resources" / "licenses"
                / "qtsvg-attribution-sample.html")
            page.unlink()
            with self.assertRaisesRegex(RuntimeError, "offline Qt attribution"):
                verify(bundle)

    def test_rejects_forbidden_binary_dependency_under_safe_filename(self):
        with tempfile.TemporaryDirectory(prefix="bz-bundle-gate-") as temp:
            bundle = self._valid_bundle(Path(temp))
            runtime = bundle / "Frameworks" / "safe-name.dylib"
            runtime.write_bytes(b"load command: @rpath/Qt6VirtualKeyboard.dll")
            with self.assertRaisesRegex(RuntimeError, "forbidden Qt binary dependencies"):
                verify(bundle)


if __name__ == "__main__":
    unittest.main()
