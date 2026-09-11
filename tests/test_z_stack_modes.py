"""Z-stack handling of the plate stitcher: full-focus fusion, every-Nth-slice
stack output, the writers behind each format, and the stitch button pulse.

Every check is against exact truth: the synthetic stack is built from a known
sharp scene whose in-focus depth varies across the field, so the fused image
and every per-slice mosaic can be compared pixel-for-pixel.
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["BZPS_TEST"] = "1"

import numpy as np
import tifffile
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from pypdf import PdfReader

import main
import stitcher

_SETTINGS_HOME = tempfile.TemporaryDirectory(prefix="bz-zstack-settings-")

TILE_H, TILE_W = 96, 128
STEP_Y, STEP_X = 60, 80
NX, NY = 3, 3
N_Z = 5


def _sharp_scene(seed: int) -> np.ndarray:
    h = (NY - 1) * STEP_Y + TILE_H
    w = (NX - 1) * STEP_X + TILE_W
    rng = np.random.default_rng(seed)
    scene = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    for y in range(0, h, 31):          # broad features keep registration unambiguous
        scene[y:y + 4] = np.clip(scene[y:y + 4].astype(np.int16) + 70, 0, 255).astype(np.uint8)
    return scene


def _defocused_slices(sharp: np.ndarray, n_z: int) -> list:
    """Slices of a scene whose sharp depth is z=0 on the left, mid-Z in the middle
    and the last Z on the right; blur grows with distance from that depth."""
    h, w = sharp.shape
    focus = np.zeros((h, w), int)
    focus[:, w // 3: 2 * w // 3] = n_z // 2
    focus[:, 2 * w // 3:] = n_z - 1
    slices = []
    for z in range(n_z):
        dist = np.abs(focus - z)
        out = np.empty_like(sharp)
        for r in np.unique(dist):
            mask = dist == r
            blurred = sharp if r == 0 else np.clip(
                stitcher._box(sharp, int(r) * 2), 0, 255).astype(np.uint8)
            out[mask] = blurred[mask]
        slices.append(out)
    return slices


def _make_z_capture(folder: Path, n_z: int = N_Z, seed: int = 20260911) -> dict:
    """Write an A01 well with CH4 (brightfield) and CH1 (its inverse) as Z stacks.

    Returns {channel: [per-z truth scene]} so mosaics can be checked exactly.
    """
    sharp = _sharp_scene(seed)
    truth = {"CH4": _defocused_slices(sharp, n_z)}
    truth["CH1"] = [255 - s for s in truth["CH4"]]
    names = {"CH4": "Brightfield", "CH1": "Green"}
    for ch, scenes in truth.items():
        ome = ('<?xml version="1.0"?><OME><Image><Pixels PhysicalSizeX="0.8">'
               f'<Channel Name="{names[ch]}"/></Pixels></Image></OME>')
        for x in range(NX):
            for y in range(NY):
                position = folder / "A01" / f"X{x:03d}Y{y:03d}"
                position.mkdir(parents=True, exist_ok=True)
                for z, scene in enumerate(scenes):
                    tile = scene[y * STEP_Y:y * STEP_Y + TILE_H,
                                 x * STEP_X:x * STEP_X + TILE_W]
                    Image.fromarray(tile).save(
                        position / f"Capture_A01_X{x:03d}Y{y:03d}_Z{z:03d}_{ch}.bz.ome.tif",
                        format="TIFF", description=ome)
    return truth


def _mae(a, b) -> float:
    return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean())


class FocusFusionTests(unittest.TestCase):
    def test_focus_stack_recovers_the_sharp_scene(self):
        sharp = _sharp_scene(1)
        slices = _defocused_slices(sharp, N_Z)
        fused = stitcher.focus_stack(slices)
        self.assertEqual(fused.dtype, np.uint8)
        self.assertEqual(fused.shape, sharp.shape)
        err = _mae(fused, sharp)
        # Only the strips where the sharp depth changes can be mis-assigned.
        self.assertLess(err, 4.0)
        for s in slices:
            self.assertGreater(_mae(s, sharp), 5 * err)
        self.assertGreater(_mae(np.mean(slices, axis=0), sharp), 5 * err)
        self.assertGreater(_mae(np.max(slices, axis=0), sharp), 5 * err)

    def test_z_reduce_focus_mode_and_single_slice(self):
        sharp = _sharp_scene(2)
        slices = _defocused_slices(sharp, N_Z)
        np.testing.assert_array_equal(stitcher._z_reduce(slices, "focus"),
                                      stitcher.focus_stack(slices))
        self.assertIs(stitcher._z_reduce([sharp], "focus"), sharp)

    def test_fusion_handles_rgb_composites(self):
        sharp = _sharp_scene(3)
        rgb = [np.stack([s, s // 2, s // 3], axis=2) for s in _defocused_slices(sharp, 3)]
        fusion = stitcher.FocusFusion()
        for s in rgb:
            fusion.add(s)
        out = fusion.result()
        self.assertEqual(out.shape, rgb[0].shape)
        self.assertLess(_mae(out[:, :, 0], sharp), 4.0)


class StackStitchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bz-zstack-")
        self.capture = Path(self.temp.name) / "Capture"
        self.truth = _make_z_capture(self.capture)
        self.wells = stitcher.discover_wells(self.capture)

    def tearDown(self):
        self.temp.cleanup()

    def test_discovery_sees_every_slice(self):
        well = self.wells["A01"]
        self.assertEqual(well.z_values, list(range(N_Z)))
        self.assertEqual(well.channels, ["CH1", "CH4"])
        self.assertEqual(well.n_tiles, NX * NY)

    def test_stack_streams_every_slice_on_one_geometry(self):
        res = stitcher.stitch_well_stack(self.wells["A01"], z_step=1,
                                         flatfield=False, subpixel=False)
        geo = res["__geometry__"]
        self.assertEqual(geo["step_source"], {"x": "measured", "y": "measured"})
        np.testing.assert_allclose(geo["step_x"], (0, STEP_X), atol=1)
        np.testing.assert_allclose(geo["step_y"], (STEP_Y, 0), atol=1)
        self.assertEqual(res["z_values"], list(range(N_Z)))
        order = [p.channel for p in self.wells["A01"].planes]     # discovery order
        self.assertEqual([p.channel for p in res["planes"]], order)

        pages = list(res["pages"])
        self.assertEqual(len(pages), N_Z * 2)
        expected = [(z, ch) for z in range(N_Z) for ch in order]
        self.assertEqual([(z, p.channel) for p, z, _ in pages], expected)
        for plane, z, mosaic in pages:
            truth = self.truth[plane.channel][z]
            self.assertEqual(mosaic.shape, truth.shape)
            self.assertLessEqual(
                int(np.abs(mosaic.astype(np.int16) - truth.astype(np.int16)).max()), 1,
                f"{plane.channel} z={z} differs from truth")

    def test_every_other_slice(self):
        res = stitcher.stitch_well_stack(self.wells["A01"], z_step=2,
                                         flatfield=False, subpixel=False)
        self.assertEqual(res["z_values"], [0, 2, 4])
        self.assertEqual(res["__geometry__"]["z_step"], 2)
        self.assertEqual(len(list(res["pages"])), 3 * 2)

    def test_focus_mode_through_stitch_well(self):
        res = stitcher.stitch_well(self.wells["A01"], z_mode="focus",
                                   flatfield=False, subpixel=False)
        sharp = _sharp_scene(20260911)
        self.assertLess(_mae(res["CH4"][1], sharp), 4.0)
        self.assertLess(_mae(res["CH1"][1], 255 - sharp.astype(np.int16)), 4.0)


class StackWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setOrganizationName("BZStudioZStackTests")
        cls.app.setApplicationName("BZ Studio z-stack tests")
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                          _SETTINGS_HOME.name)

    def setUp(self):
        settings = QSettings()
        settings.clear()
        settings.setValue("check_updates", "0")
        settings.sync()
        self.temp = tempfile.TemporaryDirectory(prefix="bz-zstack-worker-")
        self.capture = Path(self.temp.name) / "Capture"
        self.truth = _make_z_capture(self.capture)
        self.wells = stitcher.discover_wells(self.capture)

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, fmt, z_step=1):
        out = Path(self.temp.name) / f"out_{fmt}"
        out.mkdir()
        worker = main.StitchWorker(self.wells, out, "stack", fmt, flatfield=False,
                                   subpixel=False, exp_name="Exp", base="run",
                                   z_step=z_step)
        worker.run()                     # synchronous: this is the QThread body
        written = sorted(p.name for p in out.iterdir() if not p.name.endswith(".part"))
        stack_z = {w: list(wt.z_values[::z_step]) for w, wt in self.wells.items()}
        planned = sorted({Path(p).name for p in main.StitchWorker.planned_outputs(
            out, "run", sorted(self.wells), fmt, stack_z)})
        self.assertEqual(written, planned, f"[{fmt}] planned names ≠ written names")
        self.assertEqual(worker.warnings, [])
        return out

    def test_ome_tiff_carries_the_z_axis(self):
        out = self._run("both")
        with tifffile.TiffFile(str(out / "run_A01.ome.tif")) as tf:
            self.assertTrue(tf.is_ome)
            series = tf.series[0]
            self.assertEqual(series.axes, "ZCYX")
            arr = series.asarray()
        h, w = self.truth["CH4"][0].shape
        self.assertEqual(arr.shape, (N_Z, 2, h, w))
        order = [p.channel for p in self.wells["A01"].planes]
        for z in range(N_Z):
            for c, ch in enumerate(order):
                self.assertLessEqual(int(np.abs(
                    arr[z, c].astype(np.int16) - self.truth[ch][z].astype(np.int16)).max()), 1)
        # the preview next to it is the all-in-focus composite, not a slice
        preview = np.asarray(Image.open(out / "run_A01.png"))
        self.assertEqual(preview.shape, (h, w, 3))

    def test_png_writes_one_file_per_slice(self):
        out = self._run("png", z_step=2)
        names = sorted(p.name for p in out.glob("*.png"))
        self.assertEqual(names, ["run_A01_Z000.png", "run_A01_Z002.png", "run_A01_Z004.png"])

    def test_split_tiff_has_a_page_per_slice(self):
        out = self._run("split")
        for label in ("Brightfield", "Green"):
            arr = tifffile.imread(str(out / f"run_A01_{label}.tif"))
            self.assertEqual(arr.shape[0], N_Z, label)
        bf = tifffile.imread(str(out / "run_A01_Brightfield.tif"))
        for z in range(N_Z):
            self.assertLessEqual(int(np.abs(
                bf[z].astype(np.int16) - self.truth["CH4"][z].astype(np.int16)).max()), 1)

    def test_pdf_has_a_page_per_slice(self):
        out = self._run("pdf_pages")
        self.assertEqual(len(PdfReader(str(out / "run_A01.pdf")).pages), N_Z)

    def test_contact_sheet_uses_one_fused_panel(self):
        out = self._run("pdf_sheet_pages")
        self.assertEqual(len(PdfReader(str(out / "run_plate_and_wells.pdf")).pages), 2)


class DialogAndButtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setOrganizationName("BZStudioZStackTests")
        cls.app.setApplicationName("BZ Studio z-stack tests")
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                          _SETTINGS_HOME.name)

    def setUp(self):
        settings = QSettings()
        settings.clear()
        settings.setValue("check_updates", "0")
        settings.sync()

    def test_dialog_maps_choices_to_modes(self):
        with tempfile.TemporaryDirectory(prefix="bz-zstack-dlg-") as temp:
            capture = Path(temp) / "Capture"
            _make_z_capture(capture)
            wells = stitcher.discover_wells(capture)
            dlg = main.StitchDialog(wells, None, current_well="A01")
            self.assertEqual(dlg.cmb_z.count(), 5)
            self.assertEqual((dlg.z_mode, dlg.z_step), ("max", 1))
            self.assertFalse(dlg.spin_step.isEnabled())
            dlg.cmb_z.setCurrentIndex(3)
            self.assertEqual(dlg.z_mode, "focus")
            self.assertFalse(dlg.z_note.isVisibleTo(dlg))
            dlg.cmb_z.setCurrentIndex(4)
            self.assertEqual(dlg.z_mode, "stack")
            self.assertTrue(dlg.spin_step.isEnabled())
            self.assertTrue(dlg.z_note.isVisibleTo(dlg))
            dlg.spin_step.setValue(2)
            self.assertEqual(dlg.z_step, 2)
            self.assertIn("3 / 5", dlg.lbl_step.text())
            dlg.cmb_z.setCurrentIndex(0)
            self.assertEqual(dlg.z_step, 1)      # the step only applies to stack output
            dlg.deleteLater()

    def test_sample_and_planned_names_follow_stack_mode(self):
        self.assertEqual(main.StitchWorker.sample_name("png", True), "{base}_A01_Z001.png")
        self.assertEqual(main.StitchWorker.sample_name("png", False), "{base}_A01.png")
        self.assertEqual(main.StitchWorker.sample_name("both", True), "{base}_A01.ome.tif")
        names = [p.name for p in main.StitchWorker.planned_outputs(
            Path("/x"), "b", ["A01"], "png", {"A01": [0, 2]})]
        self.assertEqual(names, ["b_A01_Z000.png", "b_A01_Z002.png", "b_stitch_qc.csv"])

    def test_stitch_button_pulses_only_while_actionable(self):
        window = main.MainWindow()
        try:
            btn = window.btn_stitch_raw
            self.assertEqual(btn.objectName(), "stitchRawButton")
            window._set_stitch_pulse(True)          # nothing loaded → must stay quiet
            self.assertFalse(window._pulse_timer.isActive())

            window._mode = main.StartModeDialog.RAW
            window._raw_experiment = {"name": "E", "path": Path("."), "wells": {"A01": None}}
            window._set_stitch_pulse(True)
            self.assertTrue(window._pulse_timer.isActive())
            seen = set()
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and len(seen) < 2:
                QTest.qWait(50)
                seen.add(btn.property("pulse"))
            self.assertEqual(seen, {"on", "off"})

            window._set_stitch_pulse(False)         # work started / mode left
            self.assertFalse(window._pulse_timer.isActive())
            self.assertEqual(btn.property("pulse"), "off")
        finally:
            window._pulse_timer.stop()
            window.close()
            window.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
