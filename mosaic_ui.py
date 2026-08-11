"""Qt widgets and background jobs for the wide-area image-set workflow.

The registration and file writers stay in :mod:`mosaic_engine`; this module is
only the presentation/controller boundary.  Workers receive immutable
snapshots, so changing a control while an export is running cannot alter the
file half way through.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QSlider, QSpinBox,
    QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

import mosaic
import mosaic_engine
import render


DEFAULT_CHANNEL_COLORS = (
    (54, 112, 255),
    (16, 210, 92),
    (238, 60, 74),
    (245, 145, 36),
    (230, 230, 230),
    (177, 87, 255),
)


def recommended_presentation_dpi(max_side: int,
                                 native_long_side: Optional[int] = None) -> int:
    """Keep automatic exports near the print size of 8,000 px at 300 dpi."""
    limit = int(max_side)
    native = int(native_long_side) if native_long_side else 0
    if native > 0:
        if limit <= 0:
            output_long_side = native
        else:
            downsample = max(1, int(math.ceil(native / max(512, limit))))
            output_long_side = int(math.ceil(native / downsample))
    else:
        output_long_side = limit if limit > 0 else 16000
    raw_dpi = output_long_side * 300.0 / 8000.0
    rounded = int(math.floor(raw_dpi / 50.0 + 0.5) * 50)
    return max(150, min(1200, rounded))


class MosaicMetadataWorker(QThread):
    """Read and validate a GCI/OME image set without blocking the window."""

    progress = Signal(str, float)
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: Path, parent=None):
        super().__init__(parent)
        self.folder = Path(folder)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            def report(message, value):
                if self._cancel:
                    raise mosaic_engine.MosaicCancelled("cancelled")
                self.progress.emit(message, value)
            dataset = mosaic.load_mosaic_dataset(
                self.folder, progress=report)
            self.ready.emit(dataset)
        except mosaic_engine.MosaicCancelled:
            return
        except Exception as exc:
            self.failed.emit(str(exc))


class MosaicBuildWorker(QThread):
    """Register all neighbours and render a bounded interactive preview."""

    progress = Signal(str, float)
    ready = Signal(object, object, int)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, dataset, reference_channel: Optional[str], blend_mode: str,
                 preview_side: int = 4096, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.reference_channel = reference_channel or None
        self.blend_mode = blend_mode
        self.preview_side = int(preview_side)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            geometry = mosaic_engine.estimate_geometry(
                self.dataset, self.reference_channel,
                progress=lambda message, value: self.progress.emit(message, value * 0.72),
                cancel=lambda: self._cancel)
            images, downsample = mosaic_engine.render_preview_channels(
                self.dataset, geometry, max_side=self.preview_side,
                blend_mode=self.blend_mode,
                progress=lambda message, value: self.progress.emit(
                    message, 0.72 + value * 0.28),
                cancel=lambda: self._cancel)
            self.ready.emit(geometry, images, downsample)
        except mosaic_engine.MosaicCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class MosaicDetailWorker(QThread):
    """Render only the currently visible rectangle at the required resolution."""

    ready = Signal(object, object, int)

    def __init__(self, dataset, geometry, channel_views, blend_mode: str,
                 rect_full, downsample: int, generation: int, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.geometry = geometry
        self.channel_views = [render.ChannelView(**vars(view)) for view in channel_views]
        self.blend_mode = blend_mode
        self.rect_full = tuple(int(value) for value in rect_full)
        self.downsample = max(1, int(downsample))
        self.generation = generation

    def run(self):
        try:
            x0, y0, x1, y1 = self.rect_full
            dx0, dy0 = x0 // self.downsample, y0 // self.downsample
            dx1 = int(np.ceil(x1 / self.downsample))
            dy1 = int(np.ceil(y1 / self.downsample))
            images = {}
            for view in self.channel_views:
                if not view.enabled:
                    continue
                renderer = mosaic_engine.ChunkRenderer(
                    self.dataset, self.geometry, view.ch_id,
                    downsample=self.downsample, blend_mode=self.blend_mode,
                    cache_items=12)
                images[view.ch_id] = renderer.render_block(
                    dy0, dx0, max(1, dy1 - dy0), max(1, dx1 - dx0))
            if not images:
                return
            rgb = render.composite(self.channel_views, images)
            rect = (dx0 * self.downsample, dy0 * self.downsample,
                    rgb.shape[1] * self.downsample, rgb.shape[0] * self.downsample)
            self.ready.emit(rgb, rect, self.generation)
        except Exception:
            self.ready.emit(None, None, self.generation)


class MosaicExportWorker(QThread):
    """Write one scientific or presentation export off the GUI thread."""

    progress = Signal(str, float)
    done = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, dataset, geometry, path: Path, kind: str,
                 channel_views: list[render.ChannelView], blend_mode: str,
                 scale_bar: mosaic_engine.ScaleBarSpec, max_side: int = 8000,
                 dpi: int = 300, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.geometry = geometry
        self.path = Path(path)
        self.kind = kind
        self.channel_views = [render.ChannelView(**vars(view)) for view in channel_views]
        self.blend_mode = blend_mode
        self.scale_bar = mosaic_engine.ScaleBarSpec(**asdict(scale_bar))
        self.max_side = int(max_side)
        self.dpi = int(dpi)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            # Visibility is a presentation setting.  The scientific export is
            # explicitly labelled "all channels" in the UI, so hiding a channel
            # must never silently remove it from the OME-TIFF.
            if self.kind == "ome":
                selected = [channel.key for channel in self.dataset.channels]
            else:
                selected = [view.ch_id for view in self.channel_views if view.enabled]
            if not selected:
                raise mosaic_engine.MosaicExportError("書き出すチャンネルが選択されていません")
            if self.kind == "ome":
                mosaic_engine.write_pyramidal_ome(
                    self.dataset, self.geometry, self.path, selected,
                    # Keep the quantitative export independent from the
                    # presentation preview.  Nearest-neighbour placement and
                    # disabled radiometry preserve native detector values.
                    blend_mode="nearest", radiometric_correction=False,
                    progress=lambda message, value: self.progress.emit(message, value),
                    cancel=lambda: self._cancel,
                    qc_extra={"scale_bar": "not burned into quantitative pixels"})
            else:
                images, downsample = mosaic_engine.render_preview_channels(
                    self.dataset, self.geometry, selected, max_side=self.max_side,
                    blend_mode=self.blend_mode,
                    progress=lambda message, value: self.progress.emit(message, value * 0.84),
                    cancel=lambda: self._cancel)
                if self._cancel:
                    raise mosaic_engine.MosaicCancelled("cancelled")
                rgb = render.composite(self.channel_views, images)
                pixel_um = float(self.dataset.pixel_size_um_yx[1]) * downsample
                if self._cancel:
                    raise mosaic_engine.MosaicCancelled("cancelled")
                decorated = mosaic_engine.draw_scale_bar(
                    Image.fromarray(rgb), pixel_um, self.scale_bar)
                # Maximum can be substantially larger than the bounded options.
                # Release the per-channel mosaics and composite before Pillow
                # allocates its encoder buffers instead of holding every large
                # representation through the final save.
                del images, rgb
                self.progress.emit("画像ファイルを検証しながら保存中", 0.9)
                mosaic_engine.atomic_save_presentation(
                    decorated, self.path, self.kind, dpi=self.dpi,
                    cancel=lambda: self._cancel)
                self.progress.emit("保存が完了しました", 1.0)
            self.done.emit(str(self.path))
        except mosaic_engine.MosaicCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class MosaicChannelRow(QWidget):
    changed = Signal()

    def __init__(self, channel, color, enabled=True, parent=None):
        super().__init__(parent)
        self.channel = channel
        self.color = tuple(channel.color or color)
        dtype = np.dtype(channel.dtype)
        self.maximum = int(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 65535
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(5)

        self.enabled = QCheckBox(channel.label)
        self.enabled.setChecked(bool(enabled))
        self.enabled.setMinimumWidth(100)
        self.enabled.setToolTip(f"{channel.label}\n{channel.key}")
        layout.addWidget(self.enabled)
        self.color_button = QToolButton()
        self.color_button.setFixedSize(22, 22)
        self.color_button.clicked.connect(self._choose_color)
        layout.addWidget(self.color_button)
        self.solo = QToolButton()
        self.solo.setText("S")
        self.solo.setCheckable(True)
        self.solo.setFixedSize(22, 22)
        self.solo.setToolTip("このチャンネルだけをグレースケール表示")
        layout.addWidget(self.solo)

        self.low = self._slider(0, self.maximum, 0)
        self.high = self._slider(1, self.maximum, self.maximum)
        self.gamma = self._slider(10, 300, 100)
        for label, slider in (("min", self.low), ("max", self.high), ("γ", self.gamma)):
            layout.addWidget(QLabel(label))
            layout.addWidget(slider, 1)
        self._update_color()
        self.enabled.toggled.connect(self.changed.emit)
        self.solo.toggled.connect(self.changed.emit)

    def _slider(self, low, high, value):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.setMinimumWidth(42)
        slider.valueChanged.connect(self.changed.emit)
        return slider

    def _choose_color(self):
        value = QColorDialog.getColor(QColor(*self.color), self, "チャンネル色")
        if value.isValid():
            self.color = (value.red(), value.green(), value.blue())
            self._update_color()
            self.changed.emit()

    def _update_color(self):
        self.color_button.setStyleSheet(
            f"background:rgb({self.color[0]},{self.color[1]},{self.color[2]});"
            "border:1px solid #718096;border-radius:5px;")

    def set_auto_levels(self, image):
        values = np.asarray(image)
        finite = values[np.isfinite(values)]
        finite = finite[finite > 0]
        if not finite.size:
            return
        low = int(np.percentile(finite, 1.0))
        high = int(np.percentile(finite, 99.7))
        high = max(low + 1, min(self.maximum, high))
        for slider, value in ((self.low, low), (self.high, high)):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)

    def view(self):
        return render.ChannelView(
            self.channel.key, self.color, self.low.value(),
            max(self.low.value() + 1, self.high.value()),
            self.gamma.value() / 100.0, self.enabled.isChecked(),
            self.solo.isChecked())


class ScaleBarDialog(QDialog):
    """Scale-bar appearance editor."""

    def __init__(self, spec: mosaic_engine.ScaleBarSpec, parent=None):
        super().__init__(parent)
        self.setWindowTitle("スケールバー")
        self.setMinimumWidth(470)
        self._color = tuple(spec.color)
        self._original_anchor = (spec.anchor_x, spec.anchor_y)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.visible = QCheckBox("画像にスケールバーを入れる")
        self.visible.setChecked(spec.visible)
        form.addRow("表示", self.visible)
        length_host = QWidget()
        length_layout = QHBoxLayout(length_host)
        length_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_length = QCheckBox("自動")
        self.auto_length.setChecked(spec.length_um is None)
        self.length = QDoubleSpinBox()
        self.length.setRange(0.1, 1_000_000.0)
        self.length.setDecimals(1)
        self.length.setSuffix(" µm")
        self.length.setValue(float(spec.length_um or 1000.0))
        self.length.setEnabled(not self.auto_length.isChecked())
        self.auto_length.toggled.connect(lambda checked: self.length.setEnabled(not checked))
        length_layout.addWidget(self.auto_length)
        length_layout.addWidget(self.length, 1)
        form.addRow("バーの長さ（実寸）", length_host)

        self.position = QComboBox()
        self.position.addItem("左下", "bottom-left")
        self.position.addItem("右下", "bottom-right")
        self.position.addItem("左上", "top-left")
        self.position.addItem("右上", "top-right")
        if spec.anchor_x is not None and spec.anchor_y is not None:
            self.position.addItem("プレビューでドラッグした位置", "custom")
        idx = self.position.findData(
            "custom" if spec.anchor_x is not None and spec.anchor_y is not None
            else spec.position)
        self.position.setCurrentIndex(max(0, idx))
        form.addRow("位置", self.position)

        self.color = QPushButton()
        self.color.clicked.connect(self._choose_color)
        self._update_color()
        form.addRow("色", self.color)
        self.background = QComboBox()
        for label, value in (("黒い背景板", "dark"), ("白い背景板", "light"),
                             ("背景なし", "none")):
            self.background.addItem(label, value)
        idx = self.background.findData(spec.background)
        self.background.setCurrentIndex(max(0, idx))
        form.addRow("背景", self.background)

        self.thickness = QSpinBox(); self.thickness.setRange(1, 80)
        self.thickness.setValue(spec.thickness_px); self.thickness.setSuffix(" px")
        form.addRow("線の太さ", self.thickness)
        self.font_size = QSpinBox(); self.font_size.setRange(8, 240)
        self.font_size.setValue(spec.font_size_px); self.font_size.setSuffix(" px")
        form.addRow("文字サイズ", self.font_size)
        self.margin = QSpinBox(); self.margin.setRange(0, 500)
        self.margin.setValue(spec.margin_px); self.margin.setSuffix(" px")
        form.addRow("余白", self.margin)
        self.show_label = QCheckBox("長さのラベルを表示")
        self.show_label.setChecked(spec.show_label)
        form.addRow("ラベル", self.show_label)
        self.label = QLineEdit(spec.label)
        self.label.setPlaceholderText("空欄なら 1 mm / 500 µm などを自動表示")
        form.addRow("任意ラベル", self.label)

        layout.addLayout(form)

        note = QLabel(
            "長さは「自動」を外して数値指定できます。位置はプレビュー上の"
            "スケールバーを直接ドラッグして変更できます。\n"
            "スケールバーはPNG・TIFF・PDFに描画されます。"
            "OME-TIFFの画素には描き込みません。字体はArialです。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#64748b;font-size:11px;")
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_color(self):
        value = QColorDialog.getColor(QColor(*self._color), self, "スケールバー色")
        if value.isValid():
            self._color = (value.red(), value.green(), value.blue())
            self._update_color()

    def _update_color(self):
        self.color.setText(f"RGB {self._color[0]}, {self._color[1]}, {self._color[2]}")
        self.color.setStyleSheet(
            f"background:rgb({self._color[0]},{self._color[1]},{self._color[2]});"
            f"color:{'#111' if sum(self._color) > 420 else '#fff'};")

    def values(self):
        position = self.position.currentData()
        custom = position == "custom"
        spec = mosaic_engine.ScaleBarSpec(
            visible=self.visible.isChecked(),
            length_um=None if self.auto_length.isChecked() else self.length.value(),
            position="custom" if custom else position,
            anchor_x=self._original_anchor[0] if custom else None,
            anchor_y=self._original_anchor[1] if custom else None,
            color=self._color,
            thickness_px=self.thickness.value(), font_size_px=self.font_size.value(),
            margin_px=self.margin.value(), show_label=self.show_label.isChecked(),
            label=self.label.text().strip(), background=self.background.currentData())
        return spec


class OutputQualityDialog(QDialog):
    """Presentation resolution and file-DPI editor."""

    def __init__(self, max_side: int, dpi: int, parent=None, *,
                 dpi_auto: bool = True,
                 native_long_side: Optional[int] = None):
        super().__init__(parent)
        self.native_long_side = native_long_side
        self.setWindowTitle("出力品質")
        self.setMinimumWidth(430)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.quality = QComboBox()
        for label, value in (
                ("Maximum（上限なし）", 0),
                ("High（長辺 12,000 px）", 12000),
                ("Medium（長辺 8,000 px）", 8000),
                ("Compact（長辺 4,000 px）", 4000)):
            self.quality.addItem(label, value)
        idx = self.quality.findData(int(max_side))
        self.quality.setCurrentIndex(idx if idx >= 0 else 0)
        form.addRow("PNG / TIFF / PDF", self.quality)
        self.auto_dpi = QCheckBox("品質に合わせて自動設定")
        self.auto_dpi.setChecked(bool(dpi_auto))
        form.addRow("自動連動", self.auto_dpi)
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(dpi)
        self.dpi.setSuffix(" dpi")
        form.addRow("DPI（印刷サイズ）", self.dpi)
        self._manual_dpi = int(dpi)
        self.dpi.valueChanged.connect(self._remember_manual_dpi)
        self.quality.currentIndexChanged.connect(self._quality_changed)
        self.auto_dpi.toggled.connect(self._auto_toggled)
        if self.auto_dpi.isChecked():
            self._apply_recommended_dpi()
        self.dpi.setEnabled(not self.auto_dpi.isChecked())
        layout.addLayout(form)
        note = QLabel(
            "MaximumはStitching結果を縮小せず、全ピクセルで書き出します。"
            "大規模データでは処理時間とメモリ使用量が増えます。\n"
            "DPIは画素数を変えず、PNG/TIFFの印刷密度とPDFのページ寸法を"
            "指定します。自動は8,000 px＝300 dpiと同程度の印刷サイズを"
            "目安にします。\nOME-TIFFはこの設定に関係なく、常に全解像度です。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#64748b;font-size:11px;")
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _apply_recommended_dpi(self):
        self.dpi.setValue(recommended_presentation_dpi(
            int(self.quality.currentData()), self.native_long_side))

    def _quality_changed(self, *_args):
        if self.auto_dpi.isChecked():
            self._apply_recommended_dpi()

    def _auto_toggled(self, automatic: bool):
        if automatic:
            self._manual_dpi = self.dpi.value()
            self._apply_recommended_dpi()
        else:
            self.dpi.setValue(self._manual_dpi)
        self.dpi.setEnabled(not automatic)

    def _remember_manual_dpi(self, value: int):
        if not self.auto_dpi.isChecked():
            self._manual_dpi = int(value)

    def values(self):
        return (int(self.quality.currentData()), self._manual_dpi,
                self.auto_dpi.isChecked())


class MosaicWorkspace(QGroupBox):
    """Compact, task-oriented controls shown below the mosaic viewer."""

    build_requested = Signal(str, str)
    export_requested = Signal(str)
    views_changed = Signal()
    scale_bar_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__("通常画像セット — 広範囲Stitching", parent)
        self.dataset = None
        self.geometry = None
        self.rows: dict[str, MosaicChannelRow] = {}
        self.scale_bar = mosaic_engine.ScaleBarSpec()
        self.presentation_max_side = 0  # Maximum: no presentation downsampling
        self.presentation_dpi = 300  # last manual value; auto resolves separately
        self.presentation_dpi_auto = True

        outer = QHBoxLayout(self)
        outer.setContentsMargins(9, 12, 9, 7)
        outer.setSpacing(10)

        channels = QGroupBox("チャンネル表示")
        channel_layout = QVBoxLayout(channels)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        host = QWidget(); self.channel_layout = QVBoxLayout(host)
        self.channel_layout.setContentsMargins(2, 2, 2, 2)
        self.channel_layout.addStretch()
        scroll.setWidget(host); channel_layout.addWidget(scroll)
        # At the supported 1200 px minimum window width, a 5:3:3 split hid the
        # gamma control behind horizontal scrolling.  Give the primary display
        # controls enough room to keep min/max/gamma visible together.
        outer.addWidget(channels, 7)

        build = QGroupBox("位置合わせ")
        build_layout = QVBoxLayout(build)
        self.summary = QLabel("通常画像セットを選んでください")
        self.summary.setWordWrap(True)
        build_layout.addWidget(self.summary)
        form = QGridLayout()
        form.setHorizontalSpacing(7)
        form.setVerticalSpacing(7)
        form.setColumnStretch(1, 1)
        self.reference_label = QLabel("基準")
        self.reference_label.setMinimumHeight(30)
        self.reference_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        form.addWidget(self.reference_label, 0, 0)
        self.reference = QComboBox()
        self.reference.setMinimumHeight(30)
        self.reference.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        form.addWidget(self.reference, 0, 1)
        self.blend_label = QLabel("継ぎ目")
        self.blend_label.setMinimumHeight(30)
        self.blend_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        form.addWidget(self.blend_label, 1, 0)
        self.blend = QComboBox()
        self.blend.setMinimumHeight(30)
        self.blend.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.blend.addItem("滑らか（表示向け）", "feather")
        self.blend.addItem("最近傍（定量向け）", "nearest")
        form.addWidget(self.blend, 1, 1)
        self.reference.currentIndexChanged.connect(self._settings_changed)
        self.blend.currentIndexChanged.connect(self._settings_changed)
        build_layout.addLayout(form)
        self.build_button = QPushButton("Stitching実行")
        self.build_button.setObjectName("primaryExportButton")
        self.build_button.setMinimumHeight(34)
        self.build_button.setEnabled(False)
        self.build_button.clicked.connect(
            lambda: self.build_requested.emit(
                str(self.reference.currentData() or ""), str(self.blend.currentData())))
        build_layout.addWidget(self.build_button)
        self.qc = QLabel("")
        self.qc.setWordWrap(True)
        self.qc.setStyleSheet("color:#64748b;font-size:10px;")
        build_layout.addWidget(self.qc)
        outer.addWidget(build, 3)

        export = QGroupBox("書き出し")
        export_layout = QVBoxLayout(export)
        export_layout.setSpacing(5)
        settings_row = QHBoxLayout()
        settings_row.setContentsMargins(0, 0, 0, 0)
        settings_row.setSpacing(4)
        self.scale_button = QPushButton("スケールバー…")
        self.scale_button.setMinimumHeight(32)
        self.scale_button.clicked.connect(self.edit_scale_bar)
        settings_row.addWidget(self.scale_button, 1)
        self.quality_button = QPushButton("出力品質…")
        self.quality_button.setMinimumHeight(32)
        self.quality_button.clicked.connect(self.edit_output_quality)
        settings_row.addWidget(self.quality_button, 1)
        export_layout.addLayout(settings_row)
        self.scale_summary = QLabel()
        self.scale_summary.setWordWrap(True)
        self.scale_summary.setStyleSheet("color:#64748b;font-size:10px;")
        export_layout.addWidget(self.scale_summary)
        export_grid = QGridLayout()
        export_grid.setContentsMargins(0, 0, 0, 0)
        export_grid.setHorizontalSpacing(4)
        export_grid.setVerticalSpacing(5)
        self.export_buttons = []
        for label, kind, tip in (
            ("OME-TIFF…", "ome",
             "全チャンネル・全解像度。最近傍・輝度補正なしで元のDN値を保持"),
            ("PNG…", "png", "表示色とスケールバーを含む合成画像"),
            ("TIFF…", "tiff", "表示色とスケールバーを含む高精細画像"),
            ("PDF…", "pdf", "表示色とスケールバーを含む1ページPDF"),
        ):
            button = QPushButton(label)
            button.setMinimumHeight(30)
            button.setToolTip(tip)
            button.setEnabled(False)
            button.clicked.connect(lambda _=False, value=kind: self.export_requested.emit(value))
            if kind == "ome":
                export_grid.addWidget(button, 0, 0, 1, 3)
            else:
                export_grid.addWidget(button, 1, {"png": 0, "tiff": 1, "pdf": 2}[kind])
            self.export_buttons.append(button)
        for column in range(3):
            export_grid.setColumnStretch(column, 1)
        export_layout.addLayout(export_grid)
        outer.addWidget(export, 3)
        self.setMinimumHeight(260)
        self.setMaximumHeight(310)
        self._update_scale_summary()

    def clear(self):
        self.dataset = None
        self.geometry = None
        self.reference.clear()
        self.summary.setText("通常画像セットを選んでください")
        self.qc.clear()
        self._clear_rows()
        self.set_ready(False)

    def _clear_rows(self):
        for row in self.rows.values():
            row.setParent(None)
            row.deleteLater()
        self.rows.clear()

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.geometry = None
        self.reference.clear()
        self._clear_rows()
        rows, cols = dataset.grid_shape
        py, px = dataset.pixel_size_um_yx
        calibration = (f"{px:g} µm/px" if px is not None else "画素寸法なし")
        self.summary.setText(
            f"<b>{dataset.name}</b><br>{rows} × {cols} = {len(dataset.tiles)}視野 · "
            f"{len(dataset.channels)}チャンネル · {calibration}")
        has_fluorescence = any(
            str(getattr(channel, "file_tag", "")).upper() == "CHF"
            for channel in dataset.channels)
        for index, channel in enumerate(dataset.channels):
            self.reference.addItem(channel.label, channel.key)
            # CH4 is an excellent alignment reference, but overlaying its
            # transmitted-light illumination field on a fluorescence composite
            # obscures the fluorescent signal.  Keep it available and useable as
            # the reference while starting it hidden whenever CHF channels exist.
            visible = not (
                has_fluorescence
                and str(getattr(channel, "file_tag", "")).upper() == "CH4")
            row = MosaicChannelRow(
                channel, DEFAULT_CHANNEL_COLORS[index % len(DEFAULT_CHANNEL_COLORS)],
                enabled=visible)
            row.changed.connect(self.views_changed.emit)
            self.rows[channel.key] = row
            self.channel_layout.insertWidget(self.channel_layout.count() - 1, row)
        self.qc.setText(
            f"GCIの格子・Stage座標を基準に、隣接画像で位置を精密化します。"
            + (f" メタデータ警告 {len(dataset.warnings)}件。" if dataset.warnings else ""))
        self.build_button.setEnabled(True)
        self.set_ready(False)
        self._update_scale_summary()

    def set_geometry(self, geometry, images):
        self.geometry = geometry
        for key, row in self.rows.items():
            if key in images:
                row.set_auto_levels(images[key])
        ratio = (geometry.accepted_edges / geometry.expected_edges * 100.0
                 if geometry.expected_edges else 0.0)
        residual = (f" · 残差p95 {geometry.residual_p95:.2f}px"
                    if geometry.residual_p95 is not None else "")
        self.qc.setText(
            f"画像で確認できた継ぎ目 {geometry.accepted_edges}/{geometry.expected_edges} "
            f"({ratio:.0f}%){residual} · 出力 {geometry.output_shape[1]:,} × "
            f"{geometry.output_shape[0]:,} px"
            + ("<br>⚠ " + " ".join(geometry.warnings) if geometry.warnings else ""))
        self.set_ready(True)
        self._update_scale_summary()

    def _settings_changed(self):
        if self.geometry is None:
            return
        self.set_ready(False)
        self.build_button.setText("設定を反映してStitching再実行")
        self.qc.setText(
            "基準チャンネルまたは継ぎ目方式を変更しました。"
            "Stitchingを再実行してください。")

    def set_ready(self, ready: bool):
        for button in self.export_buttons:
            button.setEnabled(bool(ready))
        self.scale_button.setEnabled(bool(self.dataset))
        self.quality_button.setEnabled(bool(self.dataset))
        if ready:
            self.build_button.setText("Stitching再実行")

    def channel_views(self):
        return [self.rows[key].view() for key in self.rows]

    def edit_scale_bar(self):
        dialog = ScaleBarDialog(self.scale_bar, self)
        dialog.setStyleSheet("")
        if dialog.exec():
            self.scale_bar = dialog.values()
            self._update_scale_summary()
            self.scale_bar_changed.emit(self.scale_bar)

    def edit_output_quality(self):
        dialog = OutputQualityDialog(
            self.presentation_max_side, self.presentation_dpi,
            self, dpi_auto=self.presentation_dpi_auto,
            native_long_side=self._native_long_side())
        dialog.setStyleSheet("")
        if dialog.exec():
            (self.presentation_max_side, self.presentation_dpi,
             self.presentation_dpi_auto) = dialog.values()
            self._update_scale_summary()

    def _native_long_side(self) -> Optional[int]:
        if self.geometry is None:
            return None
        return int(max(self.geometry.output_shape))

    def effective_presentation_dpi(self) -> int:
        if self.presentation_dpi_auto:
            return recommended_presentation_dpi(
                self.presentation_max_side, self._native_long_side())
        return int(self.presentation_dpi)

    def set_scale_bar_position(self, anchor_x: float, anchor_y: float):
        """Store a preview drag as resolution-independent export coordinates."""
        self.scale_bar.anchor_x = max(0.0, min(1.0, float(anchor_x)))
        self.scale_bar.anchor_y = max(0.0, min(1.0, float(anchor_y)))
        self.scale_bar.position = "custom"
        self._update_scale_summary()
        self.scale_bar_changed.emit(self.scale_bar)

    def _update_scale_summary(self):
        length = "自動" if self.scale_bar.length_um is None else f"{self.scale_bar.length_um:g} µm"
        position = "・ドラッグ位置" if self.scale_bar.anchor_x is not None else ""
        shown = (f"スケールバー {length}{position}"
                 if self.scale_bar.visible else "スケールバーなし")
        quality = ("Maximum" if self.presentation_max_side <= 0
                   else f"長辺 {self.presentation_max_side:,}px")
        dpi_mode = "自動" if self.presentation_dpi_auto else "手動"
        self.scale_summary.setText(
            f"{shown}\n出力 {quality} · {self.effective_presentation_dpi()}dpi（{dpi_mode}）")
