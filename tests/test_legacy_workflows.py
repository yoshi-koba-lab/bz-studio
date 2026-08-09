from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# These must be set before importing Qt or the application.  In particular,
# BZPS_TEST prevents a regression test from ever sharing the user's settings.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["BZPS_TEST"] = "1"

import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from pypdf import PdfReader

import ktf_reader
import main
import stitcher
from make_synthetic import make_ktf


# QSettings is forced to an INI file in this temporary directory.  Merely using
# a different application name is not enough protection on every Qt platform.
_SETTINGS_HOME = tempfile.TemporaryDirectory(prefix="bz-legacy-settings-")


def _make_ktf_capture(folder: Path, pattern: str = "checker") -> np.ndarray:
    """One well with brightfield and fluorescence, small enough for fast PDF QA."""
    folder.mkdir(parents=True, exist_ok=True)
    truth = make_ktf(
        folder / "Capture_A01_CH4.ktf", 320, 240, planes=1,
        channel="Channel4", channel_comment="Normal BF", pattern=pattern,
    )
    make_ktf(
        folder / "Capture_A01_CH1-2.ktf", 320, 240, planes=3,
        channel="Channel1", channel_comment="Alexa 488", pattern="gradient",
    )
    return truth


def _make_raw_capture(folder: Path, *, nx: int = 4, ny: int = 3,
                      seed: int = 20260810) -> np.ndarray:
    """Write a deterministic overlapping raw-tile plate and return its truth."""
    tile_h, tile_w = 128, 160
    step_y, step_x = 80, 100
    rng = np.random.default_rng(seed)
    scene = rng.integers(
        0, 256,
        size=((ny - 1) * step_y + tile_h, (nx - 1) * step_x + tile_w),
        dtype=np.uint8,
    )
    # Broad deterministic features make both axes unambiguous on every platform.
    for y in range(0, scene.shape[0], 31):
        scene[y:y + 4] = np.clip(
            scene[y:y + 4].astype(np.int16) + 70, 0, 255,
        ).astype(np.uint8)
    ome = (
        '<?xml version="1.0"?><OME><Image><Pixels PhysicalSizeX="0.8">'
        '<Channel Name="Brightfield"/></Pixels></Image></OME>'
    )
    for x in range(nx):
        for y in range(ny):
            position = folder / "A01" / f"X{x:03d}Y{y:03d}"
            position.mkdir(parents=True, exist_ok=True)
            tile = scene[
                y * step_y:y * step_y + tile_h,
                x * step_x:x * step_x + tile_w,
            ]
            path = position / f"Capture_A01_X{x:03d}Y{y:03d}_CH4.bz.ome.tif"
            Image.fromarray(tile).save(path, format="TIFF", description=ome)
    return scene


class LegacyWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setOrganizationName("BZStudioLegacyTests")
        cls.app.setApplicationName("BZ Studio legacy workflow tests")
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(
            QSettings.Format.IniFormat,
            QSettings.Scope.UserScope,
            _SETTINGS_HOME.name,
        )

    def setUp(self):
        settings = QSettings()
        settings.clear()
        settings.setValue("check_updates", "0")
        settings.sync()

    def _wait_until(self, predicate, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            QTest.qWait(5)
        self.app.processEvents()
        return bool(predicate())

    def _close_window(self, window: main.MainWindow):
        # A failed assertion must not leave a QThread alive while its temporary
        # fixture is being removed.  This only waits/cancels test-owned workers.
        debounce = getattr(getattr(window, "canvas", None), "_debounce", None)
        if debounce is not None:
            debounce.stop()
        workers = list(getattr(window, "_workers", ()))
        for name in (
            "_detail_worker", "_scan_worker", "_stitch_worker",
            "_mosaic_metadata_worker", "_mosaic_build_worker",
            "_mosaic_export_worker", "_mosaic_detail_worker", "_update_checker",
        ):
            worker = getattr(window, name, None)
            if worker is not None:
                workers.append(worker)
        for worker in workers:
            try:
                if worker.isRunning():
                    cancel = getattr(worker, "cancel", None)
                    if cancel is not None:
                        cancel()
                    worker.wait(10_000)
            except RuntimeError:
                pass
        with patch.object(
                main.QMessageBox, "question",
                return_value=main.QMessageBox.StandardButton.Yes), \
                patch.object(main.QMessageBox, "information"):
            window.close()
        window.deleteLater()
        self.app.processEvents()

    def test_ktf_reader_full_and_region_reconstruction(self):
        with tempfile.TemporaryDirectory(prefix="bz-legacy-ktf-") as temp:
            capture = Path(temp) / "Capture"
            truth = _make_ktf_capture(capture)
            experiment = ktf_reader.scan_experiment_folder(capture)

            self.assertEqual(set(experiment["wells"]), {"A01"})
            self.assertEqual(set(experiment["wells"]["A01"]), {"CH4", "CH1-2"})
            self.assertEqual(experiment["errors"], [])
            info = experiment["wells"]["A01"]["CH4"]
            full = ktf_reader.reconstruct_image_full(info.path)
            region = ktf_reader.reconstruct_region(info.path, 37, 29, 211, 181)

            np.testing.assert_array_equal(full, truth)
            np.testing.assert_array_equal(region, truth[29:181, 37:211])
            self.assertEqual(full.shape, (240, 320))
            self.assertAlmostEqual(info.metadata.um_per_pixel, 2.0)

    def test_raw_discovery_and_stitch_match_truth(self):
        with tempfile.TemporaryDirectory(prefix="bz-legacy-raw-") as temp:
            capture = Path(temp) / "RawCapture"
            truth = _make_raw_capture(capture)
            wells = stitcher.discover_wells(capture)

            self.assertEqual(set(wells), {"A01"})
            well = wells["A01"]
            self.assertEqual(well.n_tiles, 12)
            self.assertEqual(well.tile_shape, (128, 160))
            self.assertEqual(well.channels, ["CH4"])
            self.assertAlmostEqual(stitcher.tile_pixel_um(well), 0.8)

            result = stitcher.stitch_well(
                well, flatfield=False, subpixel=False,
            )
            geometry = result["__geometry__"]
            self.assertEqual(
                geometry["step_source"], {"x": "measured", "y": "measured"},
            )
            np.testing.assert_allclose(geometry["step_x"], (0, 100), atol=1)
            np.testing.assert_allclose(geometry["step_y"], (80, 0), atol=1)
            mosaic = result["CH4"][1]
            self.assertEqual(mosaic.shape, truth.shape)
            self.assertLessEqual(
                int(np.abs(mosaic.astype(np.int16) - truth.astype(np.int16)).max()),
                1,
            )

    def test_gui_loads_ktf_and_raw_modes(self):
        with tempfile.TemporaryDirectory(prefix="bz-legacy-gui-") as temp:
            root = Path(temp)
            ktf_capture = root / "KtfCapture"
            raw_capture = root / "RawCapture"
            _make_ktf_capture(ktf_capture)
            _make_raw_capture(raw_capture, nx=1, ny=1)
            window = main.MainWindow()
            try:
                window._set_mode(main.StartModeDialog.KTF)
                self.assertTrue(window._load_experiment(ktf_capture))
                self.assertEqual(set(window._experiment["wells"]), {"A01"})
                self.assertTrue(window.btn_series_pdf.isEnabled())

                window._on_well_clicked("A01")
                self.assertTrue(self._wait_until(lambda: window._pending == 0))
                self.assertEqual(set(window._channel_images), {"CH4", "CH1-2"})
                self.assertEqual(
                    {image.shape for image in window._channel_images.values()},
                    {(240, 320)},
                )
                self.assertIsNotNone(window.canvas._pixmap)

                window._set_mode(main.StartModeDialog.RAW)
                self.assertTrue(window._load_raw_experiment(raw_capture))
                window._on_raw_well_selected("A01")
                self.assertEqual(window._current_raw_well, "A01")
                self.assertIn("A01", window.raw_summary.text())
                self.assertFalse(window.btn_series_pdf.isEnabled())
                self.assertFalse(window.raw_panel.isHidden())
                self.assertTrue(window.ktf_bottom.isHidden())
            finally:
                self._close_window(window)

    def test_stack_time_series_metadata_conditions_and_two_page_pdf(self):
        with tempfile.TemporaryDirectory(prefix="bz-legacy-series-") as temp:
            root = Path(temp)
            first = root / "Capture1"
            second = root / "Capture2"
            _make_ktf_capture(first, "checker")
            _make_ktf_capture(second, "gradient")
            output = root / "combined.pdf"
            window = main.MainWindow()
            try:
                window._set_mode(main.StartModeDialog.KTF)
                self.assertTrue(window._load_experiment(first))
                builder = window._plate_series_builder
                builder.reset(first)
                first_item = builder._path_items[builder._path_key(first)]
                second_item = builder._add_path_item(
                    second, checked=True, require_ktf=True,
                )
                self.assertIsNotNone(second_item)

                for item, name, time_point, stack in (
                    (first_item, "Plate repeat 1", "T0", "Z1"),
                    (second_item, "Plate repeat 2", "T1", "Z2"),
                ):
                    builder._set_editor_item(item)
                    builder.ed_name.setText(name)
                    builder.ed_time.setText(time_point)
                    builder.ed_stack.setText(stack)
                    builder._save_editor()

                window.conditions.item(0, 1).setText("control")
                window._save_conditions()
                self.assertTrue(window._activate_series_capture(second))
                window.conditions.item(0, 1).setText("treated")
                window._save_conditions()
                self.assertTrue(window._activate_series_capture(first))
                self.assertEqual(window.conditions.item(0, 1).text(), "control")
                self.assertTrue(window._activate_series_capture(second))
                self.assertEqual(window.conditions.item(0, 1).text(), "treated")
                self.assertEqual(
                    window._load_conditions_for_path(first)["A01"][0], "control",
                )
                self.assertEqual(
                    window._load_conditions_for_path(second)["A01"][0], "treated",
                )

                second_series_item = builder._series_items[builder._path_key(second)]
                builder.series_tree.setCurrentItem(second_series_item)
                builder._move_current(-1)
                selected = builder.selected
                self.assertEqual([path for path, _label in selected], [second, first])
                self.assertEqual([label for _path, label in selected], [
                    "Plate repeat 2 · Time point: T1 · Stack: Z2",
                    "Plate repeat 1 · Time point: T0 · Stack: Z1",
                ])

                builder.cmb_quality.setCurrentIndex(0)  # thumbnail grid: one page/capture
                rendered_series = []
                original_pages = window._iter_plate_series_pages

                def capture_pages(series, *args):
                    rendered_series.extend(series)
                    return original_pages(series, *args)

                with patch.object(main, "_ask_save_path", return_value=output), \
                        patch.object(
                            window, "_iter_plate_series_pages",
                            side_effect=capture_pages,
                        ):
                    builder._request_export()

                self.assertEqual(
                    [experiment["name"] for experiment, _conditions in rendered_series],
                    [label for _path, label in selected],
                )
                self.assertEqual(
                    [conditions[0]["A01"][0]
                     for _experiment, conditions in rendered_series],
                    ["treated", "control"],
                )

                payload = output.read_bytes()
                self.assertTrue(payload.startswith(b"%PDF-"))
                self.assertIn(b"%%EOF", payload[-4096:])
                self.assertGreater(len(payload), 1000)
                document = PdfReader(str(output))
                self.assertEqual(len(document.pages), 2)
            finally:
                self._close_window(window)


if __name__ == "__main__":
    unittest.main()
