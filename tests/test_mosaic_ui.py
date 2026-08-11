from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["BZPS_TEST"] = "1"

import numpy as np
from shiboken6 import Shiboken
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QSettings, QThread
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QLabel, QPushButton, QScrollArea,
)

import main
import mosaic_engine
import render
from mosaic_ui import MosaicExportWorker, OutputQualityDialog, ScaleBarDialog


class _RunningWorker:
    """Small QThread-shaped double whose stop deliberately remains pending."""

    def __init__(self):
        self.running = True
        self.cancelled = False
        self.waited = []

    def isRunning(self):
        return self.running

    def cancel(self):
        self.cancelled = True

    def wait(self, milliseconds):
        self.waited.append(milliseconds)
        return False


class _QuickWorker(QThread):
    """A real parented QThread that exits immediately for ownership tests."""

    def run(self):
        return


class MosaicUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setOrganizationName("BZStudioTests")
        cls.app.setApplicationName("BZ Studio UI tests")
        QSettings().clear()
        QSettings().setValue("check_updates", "0")

    def _window(self):
        window = main.MainWindow()
        self.addCleanup(window.close)
        return window

    def test_start_dialog_has_three_cards_in_workflow_order(self):
        dialog = main.StartModeDialog()
        self.addCleanup(dialog.close)
        cards = []
        layout = dialog.layout()
        for index in range(layout.count()):
            widget = layout.itemAt(index).widget()
            if isinstance(widget, QGroupBox):
                button = widget.findChild(QPushButton)
                cards.append((widget.title(), button.text()))

        self.assertEqual(cards, [
            ("1  通常画像セットを読み込む", "通常画像セットを選ぶ…"),
            ("2  プレート画像セットを読み込む", "プレート画像セットを選ぶ…"),
            ("3  .ktfファイルを読み込む", ".ktf画像セットを選ぶ…"),
        ])
        card_names = [
            widget.objectName() for widget in dialog.findChildren(QGroupBox)
        ]
        button_names = [
            button.objectName() for button in dialog.findChildren(QPushButton)
            if button.objectName().endswith("WorkflowButton")
        ]
        self.assertEqual(card_names, [
            "mosaicWorkflowCard", "rawWorkflowCard", "ktfWorkflowCard",
        ])
        self.assertEqual(button_names, [
            "mosaicWorkflowButton", "rawWorkflowButton", "ktfWorkflowButton",
        ])
        stylesheet = dialog.styleSheet()
        for color in ("#2878c8", "#34865a", "#7651b5"):
            self.assertIn(color, stylesheet)

    def test_mosaic_action_buttons_are_not_clipped_at_minimum_size(self):
        window = self._window()
        window._set_mode(main.StartModeDialog.MOSAIC)
        window.resize(1200, 800)
        window.show()
        channels = tuple(
            SimpleNamespace(
                key=f"CH{index}", label=f"Channel {index}", color=None,
                dtype=np.dtype("uint16"), file_tag="CHF",
            )
            for index in range(1, 4)
        )
        dataset = SimpleNamespace(
            name="Example scan", grid_shape=(2, 2), tiles=tuple(range(4)),
            pixel_size_um_yx=(0.65, 0.65), channels=channels, warnings=(),
        )
        window.mosaic_workspace.set_dataset(dataset)
        self.app.processEvents()
        self.assertEqual(
            window.mosaic_workspace.title(),
            "通常画像セット — 広範囲Stitching")
        self.assertEqual(window.mosaic_workspace.build_button.text(), "Stitching実行")

        buttons = [
            window.mosaic_workspace.build_button,
            window.mosaic_workspace.scale_button,
            window.mosaic_workspace.quality_button,
            *window.mosaic_workspace.export_buttons,
        ]
        for button in buttons:
            with self.subTest(button=button.text()):
                self.assertTrue(button.isVisible())
                self.assertGreaterEqual(button.height(), button.fontMetrics().height() + 10)
                self.assertGreaterEqual(
                    button.width(), button.fontMetrics().horizontalAdvance(button.text()) + 12)
        self.assertEqual(
            [button.text() for button in window.mosaic_workspace.export_buttons],
            ["OME-TIFF…", "PNG…", "TIFF…", "PDF…"])
        self.assertEqual(window.mosaic_workspace.scale_button.text(), "スケールバー…")
        self.assertEqual(window.mosaic_workspace.quality_button.text(), "出力品質…")
        self.assertEqual(window.mosaic_workspace.presentation_max_side, 0)
        self.assertIn("Maximum", window.mosaic_workspace.scale_summary.text())
        for control in (
                window.mosaic_workspace.reference_label,
                window.mosaic_workspace.reference,
                window.mosaic_workspace.blend_label,
                window.mosaic_workspace.blend):
            with self.subTest(control=control.objectName() or type(control).__name__):
                self.assertTrue(control.isVisible())
                self.assertGreaterEqual(control.height(), 30)
                self.assertGreaterEqual(
                    control.height(), control.fontMetrics().height() + 10)

        window.mosaic_workspace.set_ready(True)
        self.assertEqual(
            window.mosaic_workspace.build_button.text(), "Stitching再実行")
        window.mosaic_workspace.geometry = object()
        window.mosaic_workspace._settings_changed()
        self.assertEqual(
            window.mosaic_workspace.build_button.text(),
            "設定を反映してStitching再実行")

    def test_scale_bar_preview_uses_arial(self):
        self.assertEqual(main.SCALE_BAR_FONT_FAMILY, "Arial")
        font = main.QFont(main.SCALE_BAR_FONT_FAMILY, 16)
        self.assertEqual(font.family(), "Arial")

    def test_zoom_is_an_explicit_numeric_control(self):
        canvas = main.ImageCanvas()
        self.addCleanup(canvas.close)
        canvas.resize(400, 300)
        canvas.show()
        canvas.set_overview(QPixmap(200, 100), 2, 400, 200, 0.8)
        self.app.processEvents()

        labels = [label.text() for label in canvas.zoom_control.findChildren(QLabel)]
        self.assertIn("倍率", labels)
        self.assertTrue(canvas.zoom_control.isVisible())
        canvas.zoom_edit.setText("150")
        canvas.zoom_edit.editingFinished.emit()
        self.assertAlmostEqual(canvas.screen_px_per_full_px, 1.5, places=6)
        self.assertEqual(canvas.zoom_edit.text(), "150%")

    def test_scale_bar_can_be_dragged_and_dialog_preserves_position(self):
        canvas = main.ImageCanvas()
        self.addCleanup(canvas.close)
        canvas.resize(500, 340)
        canvas.show()
        canvas.set_overview(QPixmap(300, 160), 1, 300, 160, 1.0)
        spec = mosaic_engine.ScaleBarSpec(length_um=50, show_label=False)
        canvas.set_scale_bar_spec(spec)
        moved = []
        canvas.scale_bar_position_changed.connect(
            lambda x, y: moved.append((x, y)))
        self.app.processEvents()
        canvas.repaint()
        self.app.processEvents()

        start = canvas._scale_bar_hit_rect.center().toPoint()
        target = start + QPoint(120, -45)
        QTest.mousePress(canvas, main.Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(canvas, target, delay=1)
        QTest.mouseRelease(canvas, main.Qt.MouseButton.LeftButton, pos=target)
        self.app.processEvents()
        self.assertTrue(moved)
        self.assertEqual(spec.position, "custom")
        self.assertIsNotNone(spec.anchor_x)
        self.assertIsNotNone(spec.anchor_y)

        dialog = ScaleBarDialog(spec)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.position.currentData(), "custom")
        retained = dialog.values()
        self.assertAlmostEqual(retained.anchor_x, spec.anchor_x)
        self.assertAlmostEqual(retained.anchor_y, spec.anchor_y)
        dialog.position.setCurrentIndex(dialog.position.findData("top-right"))
        corner = dialog.values()
        self.assertIsNone(corner.anchor_x)
        self.assertIsNone(corner.anchor_y)

    def test_output_quality_defaults_to_unlimited_maximum(self):
        dialog = OutputQualityDialog(0, 300)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.quality.currentText(), "Maximum（上限なし）")
        self.assertEqual(dialog.values(), (0, 300))
        dialog.quality.setCurrentIndex(dialog.quality.findData(8000))
        dialog.dpi.setValue(600)
        self.assertEqual(dialog.values(), (8000, 600))

    def test_about_discloses_pyside_and_lgpl(self):
        window = self._window()
        with patch.object(main.QMessageBox, "about") as about:
            window._about()
        html = about.call_args.args[2]
        self.assertIn("PySide6", html)
        self.assertIn("LGPLv3", html)
        self.assertIn("THIRD_PARTY_NOTICES.md", html)

    def test_supported_sizes_and_mode_visibility(self):
        window = self._window()
        window.show()
        for width, height in ((1200, 800), (1500, 950)):
            window.resize(width, height)
            self.app.processEvents()
            self.assertEqual((window.width(), window.height()), (width, height))

        expected = {
            main.StartModeDialog.MOSAIC: (False, True, False, False),
            main.StartModeDialog.RAW: (True, False, False, True),
            main.StartModeDialog.KTF: (False, False, True, True),
        }
        for mode, visibility in expected.items():
            window._set_mode(mode)
            self.app.processEvents()
            actual = (
                window.raw_panel.isVisible(),
                window.mosaic_workspace.isVisible(),
                window.ktf_bottom.isVisible(),
                window.well_group.isVisible(),
            )
            self.assertEqual(actual, visibility)

    def test_gamma_control_is_visible_without_horizontal_scroll_at_1200(self):
        window = self._window()
        window._set_mode(main.StartModeDialog.MOSAIC)
        window.resize(1200, 800)
        window.show()
        channels = tuple(
            SimpleNamespace(
                key=f"CH{index}", label=f"Channel {index}", color=None,
                dtype=np.dtype("uint16"),
            )
            for index in range(1, 4)
        )
        dataset = SimpleNamespace(
            name="Example scan", grid_shape=(2, 2), tiles=tuple(range(4)),
            pixel_size_um_yx=(0.65, 0.65), channels=channels, warnings=(),
        )
        window.mosaic_workspace.set_dataset(dataset)
        self.app.processEvents()

        row = window.mosaic_workspace.rows["CH1"]
        scroll = next(
            candidate for candidate in window.mosaic_workspace.findChildren(QScrollArea)
            if candidate.widget() is not None and candidate.widget().isAncestorOf(row)
        )
        gamma_right = row.gamma.mapTo(
            scroll.viewport(), QPoint(row.gamma.width() - 1, 0)).x()
        self.assertFalse(scroll.horizontalScrollBar().isVisible())
        self.assertGreaterEqual(gamma_right, 0)
        self.assertLess(gamma_right, scroll.viewport().width())

    def test_scientific_export_uses_all_channels_even_when_hidden(self):
        dataset = SimpleNamespace(channels=(
            SimpleNamespace(key="c1"), SimpleNamespace(key="c2"),
        ))
        views = [
            render.ChannelView("c1", (255, 0, 0), 0, 255, 1.0, True, False),
            render.ChannelView("c2", (0, 255, 0), 0, 255, 1.0, False, False),
        ]
        worker = MosaicExportWorker(
            dataset, object(), Path("/tmp/all-channels.ome.tif"), "ome", views,
            "feather", mosaic_engine.ScaleBarSpec(),
        )
        with patch("mosaic_ui.mosaic_engine.write_pyramidal_ome") as write_ome:
            worker.run()

        self.assertEqual(write_ome.call_args.args[3], ["c1", "c2"])
        self.assertEqual(write_ome.call_args.kwargs["blend_mode"], "nearest")
        self.assertFalse(write_ome.call_args.kwargs["radiometric_correction"])

    def test_presentation_export_honours_cancel_before_composite(self):
        dataset = SimpleNamespace(
            channels=(SimpleNamespace(key="c1"),),
            pixel_size_um_yx=(0.8, 0.8))
        views = [
            render.ChannelView("c1", (255, 0, 0), 0, 255, 1.0, True, False),
        ]
        worker = MosaicExportWorker(
            dataset, object(), Path("/tmp/cancelled.png"), "png", views,
            "feather", mosaic_engine.ScaleBarSpec())
        cancelled, completed = [], []
        worker.cancelled.connect(lambda: cancelled.append(True))
        worker.done.connect(completed.append)

        def finish_then_cancel(*_args, **_kwargs):
            worker.cancel()
            return {"c1": np.zeros((4, 4), dtype=np.uint8)}, 1

        with patch.object(
                mosaic_engine, "render_preview_channels",
                side_effect=finish_then_cancel), \
                patch.object(mosaic_engine, "atomic_save_presentation") as save:
            worker.run()

        self.assertEqual(cancelled, [True])
        self.assertEqual(completed, [])
        save.assert_not_called()

    def test_transmitted_light_starts_hidden_when_fluorescence_exists(self):
        window = self._window()
        channels = (
            SimpleNamespace(
                key="CH4", label="CH4", color=(255, 255, 255),
                file_tag="CH4", dtype=np.dtype("uint8")),
            SimpleNamespace(
                key="CHF-p0", label="DAPI", color=(0, 0, 255),
                file_tag="CHF", dtype=np.dtype("uint8")),
        )
        dataset = SimpleNamespace(
            name="Fluorescence scan", grid_shape=(2, 2), tiles=tuple(range(4)),
            pixel_size_um_yx=(0.8, 0.8), channels=channels, warnings=(),
        )
        window.mosaic_workspace.set_dataset(dataset)

        self.assertFalse(window.mosaic_workspace.rows["CH4"].enabled.isChecked())
        self.assertTrue(window.mosaic_workspace.rows["CHF-p0"].enabled.isChecked())
        self.assertEqual(window.mosaic_workspace.reference.currentData(), "CH4")

    def test_compound_ome_suffix_is_added_once(self):
        cases = (
            ("/tmp/result", Path("/tmp/result.ome.tif")),
            ("/tmp/result.ome.tif", Path("/tmp/result.ome.tif")),
            ("/tmp/result.ome.tiff", Path("/tmp/result.ome.tiff")),
        )
        for chosen, expected in cases:
            with self.subTest(chosen=chosen), \
                    patch.object(main.QFileDialog, "getSaveFileName",
                                 return_value=(chosen, "")), \
                    patch.object(main, "_confirm_overwrite", return_value=True):
                actual = main._ask_save_path(
                    None, "save", "/tmp/default.ome.tif",
                    "OME-TIFF (*.ome.tif *.ome.tiff)",
                )
            self.assertEqual(actual, expected)

    def test_close_is_ignored_until_mosaic_worker_stops(self):
        window = self._window()
        window._plate_series_builder = None
        worker = _RunningWorker()
        window._mosaic_build_worker = worker
        event = QCloseEvent()
        with patch.object(
                main.QMessageBox, "question",
                return_value=main.QMessageBox.StandardButton.Yes), \
                patch.object(main.QMessageBox, "information") as information:
            window.closeEvent(event)

        self.assertTrue(worker.cancelled)
        self.assertEqual(worker.waited, [15000])
        self.assertFalse(event.isAccepted())
        information.assert_called_once()
        worker.running = False
        window._mosaic_build_worker = None

    def test_finished_parented_mosaic_workers_are_disposed(self):
        window = self._window()
        for attribute in (
                "_mosaic_metadata_worker", "_mosaic_build_worker",
                "_mosaic_export_worker"):
            worker = _QuickWorker(window)
            setattr(window, attribute, worker)
            worker.finished.connect(window._on_mosaic_worker_finished)
            worker.start()
            self.assertTrue(worker.wait(2000))
            self.app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.assertIsNone(getattr(window, attribute))
            self.assertFalse(Shiboken.isValid(worker), attribute)

    def test_stale_detail_finish_does_not_clear_new_worker(self):
        window = self._window()
        old_worker = _QuickWorker(window)
        new_worker = _QuickWorker(window)
        window._mosaic_detail_worker = new_worker
        window._detail_pending = True
        old_worker.finished.connect(window._on_mosaic_detail_finished)

        old_worker.start()
        self.assertTrue(old_worker.wait(2000))
        self.app.processEvents()

        self.assertIs(window._mosaic_detail_worker, new_worker)
        self.assertTrue(window._detail_pending)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertFalse(Shiboken.isValid(old_worker))
        window._mosaic_detail_worker = None
        new_worker.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_ktf_detail_worker_emits_qimage_not_gui_pixmap(self):
        view = render.ChannelView(
            "c", (255, 255, 255), 0, 255, 1.0, True, False)
        worker = main.DetailWorker(
            {"c": (Path("/fake/c.ktf"), (4, 4))}, [view],
            (4, 4), (0, 0, 4, 4), 1, 7)
        payloads = []
        worker.ready.connect(lambda image, rect, gen: payloads.append(
            (image, rect, gen)))
        with patch.object(
                main.ktf_reader, "reconstruct_region",
                return_value=np.arange(16, dtype=np.uint8).reshape(4, 4)), \
                patch.object(
                    main.render, "composite",
                    return_value=np.zeros((4, 4, 3), dtype=np.uint8)):
            worker.run()

        self.assertEqual(len(payloads), 1)
        image, rect, generation = payloads[0]
        self.assertIsInstance(image, QImage)
        self.assertNotIsInstance(image, QPixmap)
        self.assertEqual(rect, (0, 0, 4, 4))
        self.assertEqual(generation, 7)


if __name__ == "__main__":
    unittest.main()
