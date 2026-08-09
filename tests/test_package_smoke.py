from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import main


class PackageSmokeTests(unittest.TestCase):
    def test_codecs_and_atomic_presentations_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix="bz-package-smoke-") as temp:
            output = Path(temp)
            self.assertEqual(main._run_package_smoke(output), 0)
            report = json.loads((output / "package-smoke.json").read_text("utf-8"))
            self.assertTrue(report["ok"])
            self.assertEqual(report["qt_binding"], "PySide6")
            self.assertTrue(report["qt_binding_version"])
            self.assertEqual(report["shape"], [2, 64, 80])
            self.assertTrue((output / report["lzw"]).is_file())
            self.assertTrue((output / report["ome"]).is_file())
            self.assertTrue((output / report["png"]).is_file())
            pdf = output / report["pdf"]
            self.assertTrue(pdf.is_file())
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))


if __name__ == "__main__":
    unittest.main()
