#!/usr/bin/env python3
"""BZ Studio — wide-area mosaics, plate images, and .ktf viewing.

Features: well-plate browser, multi-channel pseudocolor compositing, full-resolution
detail-on-zoom, real-world scale bar, cursor readout (stage position + intensity),
per-channel color / gamma / window-level, and TIFF/PNG export.
"""

import sys
import os
import csv
import json
import math
from pathlib import Path

# Make the app self-contained when run from source: point Qt at PySide6's bundled
# platform plugins. Some Python environments (e.g. non-activated conda) don't set
# this, causing a "could not find the Qt platform plugin" crash on launch.
# Skip entirely when frozen (PyInstaller) — it configures Qt itself, and overriding
# the path there breaks plugin discovery.
if not getattr(sys, "frozen", False):
    try:
        from PySide6.QtCore import QLibraryInfo
        _plugins = os.path.join(
            QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath), "platforms")
        if os.path.isdir(_plugins):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _plugins)
    except Exception:
        pass

import io
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Contact sheets at original resolution legitimately exceed PIL's bomb threshold.
Image.MAX_IMAGE_PIXELS = None

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeWidget, QTreeWidgetItem, QLabel, QScrollArea,
    QSlider, QGroupBox, QCheckBox, QPushButton, QToolButton,
    QFileDialog, QGridLayout, QSizePolicy, QTextEdit,
    QProgressBar, QColorDialog, QToolBar, QInputDialog, QLineEdit,
    QTabWidget, QTableWidget, QTableWidgetItem, QAbstractItemView, QMessageBox, QMenu,
    QProxyStyle, QStyle, QDialogButtonBox, QDialog, QComboBox, QDockWidget,
    QHeaderView,
)
from PySide6.QtCore import (
    Qt, QSize, QSizeF, QMarginsF, Signal, QPoint, QRect, QRectF, QThread, QTimer,
    QPointF, QEvent, QSettings, QUrl, QSaveFile, QIODevice,
)
from PySide6.QtGui import (
    QImage, QPixmap, QIcon, QPainter, QColor, QAction, QWheelEvent,
    QMouseEvent, QPen, QFont, QKeySequence, QBrush, QDesktopServices,
    QPdfWriter, QPageSize, QPageLayout,
)

import ktf_reader
import render
import stitcher
import mosaic
import mosaic_engine
from mosaic_ui import (
    MosaicBuildWorker, MosaicDetailWorker, MosaicExportWorker,
    MosaicMetadataWorker, MosaicWorkspace,
)
from version import __version__, APP_NAME, SETTINGS_APP_NAME

# Default pseudocolor per channel id
CHANNEL_COLORS = {
    "CH1-1": (60, 120, 255),   # DAPI → blue
    "CH1-2": (0, 255, 0),      # GFP / Alexa488 → green
    "CH2": (255, 40, 40),      # mCherry / Alexa555 → red
    "CH1-4": (255, 140, 0),    # Alexa594 → orange
    "CH4": (255, 255, 255),    # Brightfield → white
    "CHF": (200, 200, 200),
}

MAX_DISPLAY_MEGAPIXELS = 16  # overview cap
SCALE_BAR_FONT_FAMILY = "Arial"


def _path_key(path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def auto_downsample(width: int, height: int) -> int:
    """Downsample factor for the overview, snapped up to a power of two.

    Powers of two divide the 512px tile size exactly, so the tiled overview
    stays pixel-aligned with no per-tile drift.
    """
    mpx = (width * height) / 1e6
    if mpx <= MAX_DISPLAY_MEGAPIXELS:
        return 1
    need = math.sqrt(mpx / MAX_DISPLAY_MEGAPIXELS)
    ds = 1
    while ds < need:
        ds *= 2
    return ds


def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
    else:
        raise ValueError(f"Unsupported shape: {arr.shape}")
    return QPixmap.fromImage(qimg.copy())


class ImageCanvas(QWidget):
    """Zoomable/pannable canvas with scale bar, cursor readout, and detail overlay."""

    view_changed = Signal()          # emitted after zoom/pan settles (debounced)
    cursor_moved = Signal(float, float)  # full-res image coords under cursor
    scale_bar_position_changed = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None                # overview composite pixmap
        self._overview_ds = 1              # full-res px per overview px
        self._full_w = 0
        self._full_h = 0
        self._um_per_px_full = 0.0
        self._custom_scale_bar = None
        self._empty_message = "画像セットを開いてください"

        self._detail_pixmap = None         # QPixmap covering a sub-rect at higher res
        self._detail_rect_full = None      # QRect in full-res image coords

        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._dragging = False
        self._dragging_scale_bar = False
        self._drag_start = QPointF()
        self._scale_drag_offset = QPointF()
        self._scale_bar_hit_rect = QRectF()
        self._scale_bar_drag_limits = None
        self._scale_bar_hover = False
        self._fitted_once = False   # first fit is automatic; later resizes keep the user's zoom
        self._zoom_edit_busy = False  # clearFocus() re-fires editingFinished; guard re-entry

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self.view_changed.emit)

        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # Clearly labelled editable zoom field, overlaid at the bottom-left corner.
        self.zoom_control = QWidget(self)
        self.zoom_control.setObjectName("zoomControl")
        self.zoom_control.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        zoom_layout = QHBoxLayout(self.zoom_control)
        zoom_layout.setContentsMargins(7, 3, 5, 3)
        zoom_layout.setSpacing(4)
        zoom_label = QLabel("倍率", self.zoom_control)
        zoom_label.setObjectName("zoomLabel")
        zoom_layout.addWidget(zoom_label)
        self.zoom_edit = QLineEdit(self.zoom_control)
        self.zoom_edit.setFixedWidth(66)
        self.zoom_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_edit.setAccessibleName("表示倍率（数値入力）")
        self.zoom_edit.setToolTip("倍率を数値で入力（例: 150 または 150%）し、Enterキーで適用")
        zoom_layout.addWidget(self.zoom_edit)
        self.zoom_control.setFixedSize(116, 32)
        self.zoom_control.setStyleSheet("""
            QWidget#zoomControl {
                background:rgba(255,255,255,235); color:#1c1e21;
                border:1px solid #8aa9cf; border-radius:6px;
            }
            QLabel#zoomLabel { border:none; color:#34465c; font-size:11px; }
            QLineEdit { background:#ffffff; color:#1c1e21;
                border:1px solid #4b8fda; border-radius:4px; padding:1px 3px;
                font-family:Menlo; font-size:11px; selection-background-color:#1674c5; }
        """)
        self.zoom_edit.editingFinished.connect(self._on_zoom_edit)
        self.zoom_control.hide()
        self._position_overlays()

    # --- image setup ---
    def set_overview(self, pixmap, overview_ds, full_w, full_h, um_per_px_full):
        self._pixmap = pixmap
        self._overview_ds = max(1, overview_ds)
        self._full_w = full_w
        self._full_h = full_h
        self._um_per_px_full = um_per_px_full
        self._detail_pixmap = None
        self._detail_rect_full = None
        self.fit_in_view()

    def update_overview_pixmap(self, pixmap):
        """Replace overview pixmap without resetting zoom/pan (e.g. after level change)."""
        self._pixmap = pixmap
        self.update()

    def update_geometry(self, overview_ds, full_w, full_h, um_per_px_full):
        """Keep scale/coordinate metadata in step with the image being shown.

        Called on every overview rebuild so a partially-loaded well never renders
        using the previous well's calibration.
        """
        self._overview_ds = max(1, overview_ds)
        self._full_w = full_w
        self._full_h = full_h
        self._um_per_px_full = um_per_px_full

    def clear_image(self):
        """Drop the displayed image and return to the empty state."""
        self._pixmap = None
        self._detail_pixmap = None
        self._detail_rect_full = None
        self._full_w = self._full_h = 0
        self._um_per_px_full = 0.0
        self.zoom_control.hide()
        self._scale_bar_hit_rect = QRectF()
        self._scale_bar_drag_limits = None
        self._scale_bar_hover = False
        self.update()

    def set_scale_bar_spec(self, spec=None):
        """Use an export-style scale bar overlay, or the ordinary auto bar."""
        self._custom_scale_bar = spec
        if spec is None or not getattr(spec, "visible", False):
            self._scale_bar_hit_rect = QRectF()
            self._scale_bar_drag_limits = None
            self._scale_bar_hover = False
        self.update()

    def set_empty_message(self, text: str):
        self._empty_message = str(text)
        if not self._pixmap:
            self.update()

    def set_detail(self, pixmap, rect_full):
        self._detail_pixmap = pixmap
        self._detail_rect_full = rect_full
        self.update()

    # --- coordinate transforms ---
    def screen_to_full(self, sx, sy):
        ox = (sx - self._pan.x()) / self._zoom
        oy = (sy - self._pan.y()) / self._zoom
        return ox * self._overview_ds, oy * self._overview_ds

    def full_to_screen(self, fx, fy):
        ox = fx / self._overview_ds
        oy = fy / self._overview_ds
        return ox * self._zoom + self._pan.x(), oy * self._zoom + self._pan.y()

    @property
    def screen_px_per_full_px(self):
        return self._zoom / self._overview_ds

    @property
    def um_per_screen_px(self):
        if self.screen_px_per_full_px <= 0:
            return 0.0
        return self._um_per_px_full / self.screen_px_per_full_px

    def visible_full_rect(self):
        x0, y0 = self.screen_to_full(0, 0)
        x1, y1 = self.screen_to_full(self.width(), self.height())
        x0 = max(0, min(x0, self._full_w))
        y0 = max(0, min(y0, self._full_h))
        x1 = max(0, min(x1, self._full_w))
        y1 = max(0, min(y1, self._full_h))
        return int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))

    def fit_in_view(self):
        if not self._pixmap:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw == 0 or ph == 0:
            return
        self._zoom = min(self.width() / pw, self.height() / ph) * 0.95
        self._pan = QPointF((self.width() - pw * self._zoom) / 2,
                            (self.height() - ph * self._zoom) / 2)
        self._fitted_once = True
        self._sync_zoom_field()
        self.update()
        self._debounce.start()

    def zoom_actual_pixels(self):
        """Set 1 screen pixel = 1 full-resolution image pixel, keeping the view centre."""
        if not self._pixmap:
            return
        cx, cy = self.width() / 2, self.height() / 2
        fx, fy = self.screen_to_full(cx, cy)
        self._zoom = float(self._overview_ds)  # screen_px_per_full_px == 1
        sx, sy = self.full_to_screen(fx, fy)
        self._pan += QPointF(cx - sx, cy - sy)
        self._sync_zoom_field()
        self.update()
        self._debounce.start()

    # --- painting ---
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(246, 247, 249))
        if not self._pixmap:
            p.setPen(QColor(120, 125, 132))
            p.setFont(QFont("Menlo", 13))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       self._empty_message +
                       "\n\nスクロール＝ズーム · ドラッグ＝移動 · ⌘0＝全体")
            p.end()
            return
        if self._pixmap:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            target = QRect(int(self._pan.x()), int(self._pan.y()),
                           int(self._pixmap.width() * self._zoom),
                           int(self._pixmap.height() * self._zoom))
            p.drawPixmap(target, self._pixmap)

            # Detail overlay (sharper) within its full-res rect
            if self._detail_pixmap and self._detail_rect_full:
                fx0, fy0, fw, fh = self._detail_rect_full
                sx0, sy0 = self.full_to_screen(fx0, fy0)
                sx1, sy1 = self.full_to_screen(fx0 + fw, fy0 + fh)
                dst = QRect(int(sx0), int(sy0), int(sx1 - sx0), int(sy1 - sy0))
                p.drawPixmap(dst, self._detail_pixmap)

            self._draw_scale_bar(p)
        p.end()

    def _image_screen_rect(self):
        """Visible image rectangle in screen coords, clamped to the widget."""
        left = self._pan.x()
        top = self._pan.y()
        right = left + self._pixmap.width() * self._zoom
        bottom = top + self._pixmap.height() * self._zoom
        return (max(0, left), max(0, top),
                min(self.width(), right), min(self.height(), bottom))

    def _draw_scale_bar(self, p: QPainter):
        umpp_screen = self.um_per_screen_px
        if umpp_screen <= 0:
            return
        if self._custom_scale_bar is not None:
            self._draw_custom_scale_bar(p, umpp_screen)
            return
        max_bar = min(240, self.width() * 0.3)
        length_um, length_px, label = render.nice_scale_bar(umpp_screen, max_bar)
        if length_px <= 0:
            return
        il, it, ir, ib = self._image_screen_rect()
        margin = 14
        # bottom-left, inside the image; lifted above the zoom field so they don't overlap
        x0 = il + margin
        y = min(ib - margin, self.height() - 40)
        x1 = x0 + length_px
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 130)))
        p.drawRect(int(x0 - 8), int(y - 24), int(length_px + 16), 34)
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(3)
        p.setPen(pen)
        p.drawLine(int(x0), int(y), int(x1), int(y))
        p.setFont(QFont(SCALE_BAR_FONT_FAMILY, 11, QFont.Weight.Bold))
        p.drawText(QRect(int(x0 - 8), int(y - 24), int(length_px + 16), 18),
                   Qt.AlignmentFlag.AlignCenter, label)

    def _draw_custom_scale_bar(self, painter: QPainter, umpp_screen: float):
        spec = self._custom_scale_bar
        if not spec.visible:
            self._scale_bar_hit_rect = QRectF()
            self._scale_bar_drag_limits = None
            return
        il, it, ir, ib = self._image_screen_rect()
        max_width = max(1.0, min(300.0, (ir - il) * 0.28))
        if spec.length_um is None:
            length_um, length_px, auto_label = render.nice_scale_bar(
                umpp_screen, max_width)
        else:
            length_um = float(spec.length_um)
            length_px = length_um / umpp_screen
            auto_label = (f"{length_um / 1000:g} mm" if length_um >= 1000
                          else f"{length_um:g} µm")
        if length_px <= 0:
            return
        length_px = min(length_px, max(1.0, ir - il - 16.0))
        label = spec.label or auto_label
        margin = max(5, min(70, int(spec.margin_px)))
        thick = max(1, min(20, int(spec.thickness_px)))
        font_size = max(8, min(24, int(round(spec.font_size_px * 0.7))))
        font = QFont(SCALE_BAR_FONT_FAMILY)
        font.setPixelSize(font_size)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        label_h = metrics.height() if spec.show_label else 0
        gap = max(3, thick // 2)
        total_h = thick + (gap + label_h if spec.show_label else 0)
        text_w = metrics.horizontalAdvance(label) if spec.show_label else 0
        panel_w = max(length_px, text_w)
        pad = max(4, thick)
        # Keep the complete bar panel inside the visible image.  A dragged
        # position is stored as 0..1 so the same relative position is retained
        # when a PNG/PDF/TIFF is exported at a different resolution.
        margin_x = min(margin, max(0.0, (ir - il - panel_w) / 2.0))
        margin_y = min(margin, max(0.0, (ib - it - total_h) / 2.0))
        min_x = il + margin_x
        max_x = max(min_x, ir - margin_x - panel_w)
        min_y = it + margin_y
        max_y = max(min_y, ib - margin_y - total_h)
        anchor_x = getattr(spec, "anchor_x", None)
        anchor_y = getattr(spec, "anchor_y", None)
        if anchor_x is not None and anchor_y is not None:
            x = min_x + max(0.0, min(1.0, float(anchor_x))) * (max_x - min_x)
            y = min_y + max(0.0, min(1.0, float(anchor_y))) * (max_y - min_y)
        else:
            right = "right" in spec.position
            bottom = "bottom" in spec.position
            x = max_x if right else min_x
            y = max_y if bottom else min_y
        self._scale_bar_drag_limits = (min_x, max_x, min_y, max_y)
        self._scale_bar_hit_rect = QRectF(
            x - pad, y - pad, panel_w + 2 * pad, total_h + 2 * pad)
        if spec.background in {"dark", "light"}:
            background = QColor(0, 0, 0, 170) if spec.background == "dark" \
                else QColor(255, 255, 255, 205)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(
                QRect(int(x - pad), int(y - pad), int(panel_w + 2 * pad),
                      int(total_h + 2 * pad)), pad // 2, pad // 2)
        color = QColor(*spec.color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRect(QRect(int(x), int(y), int(length_px), thick))
        if spec.show_label:
            painter.setPen(color)
            painter.drawText(
                QRect(int(x), int(y + thick + gap), int(max(panel_w, 1)), label_h),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
        if self._scale_bar_hover or self._dragging_scale_bar:
            outline = QPen(QColor(59, 130, 246, 220), 1, Qt.PenStyle.DashLine)
            painter.setPen(outline)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self._scale_bar_hit_rect, 4, 4)

    # --- interaction ---
    def _apply_zoom(self, factor, center: QPointF):
        if not self._pixmap or factor <= 0:
            return
        old = self._zoom
        self._zoom = max(0.01, min(200.0, self._zoom * factor))
        if self._zoom == old:
            return
        self._pan = QPointF(
            center.x() - (center.x() - self._pan.x()) * (self._zoom / old),
            center.y() - (center.y() - self._pan.y()) * (self._zoom / old),
        )
        self._sync_zoom_field()
        self.update()
        self._debounce.start()

    def wheelEvent(self, event: QWheelEvent):
        if not self._pixmap:
            return
        dy = event.angleDelta().y()
        if dy == 0:  # pinch/native gestures arrive with y==0 — handled in event()
            return
        # Scale with the reported delta: one mouse-wheel notch (120) ≈ 1.2x, while a
        # trackpad's many small deltas each nudge gently instead of slamming the clamp.
        dy = max(-600, min(600, dy))
        self._apply_zoom(1.0015 ** dy, event.position())

    def event(self, e):
        # macOS trackpad pinch-to-zoom arrives as a native gesture, not a wheel event.
        if e.type() == QEvent.Type.NativeGesture and self._pixmap:
            if e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                self._apply_zoom(1.0 + e.value(), e.position())
                return True
        return super().event(e)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if (self._custom_scale_bar is not None
                    and self._scale_bar_hit_rect.contains(event.position())
                    and self._scale_bar_drag_limits is not None):
                self._dragging_scale_bar = True
                content_origin = self._scale_bar_hit_rect.topLeft() + QPointF(
                    max(4, min(20, int(self._custom_scale_bar.thickness_px))),
                    max(4, min(20, int(self._custom_scale_bar.thickness_px))))
                self._scale_drag_offset = event.position() - content_origin
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            self._dragging = True
            self._drag_start = event.position() - self._pan
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging_scale_bar:
            self._dragging_scale_bar = False
            hover = self._scale_bar_hit_rect.contains(event.position())
            self._scale_bar_hover = hover
            self.setCursor(Qt.CursorShape.OpenHandCursor if hover
                           else Qt.CursorShape.ArrowCursor)
            self.update()
            event.accept()
            return
        self._dragging = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._debounce.start()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        if self._dragging_scale_bar and self._scale_bar_drag_limits is not None:
            min_x, max_x, min_y, max_y = self._scale_bar_drag_limits
            x = max(min_x, min(max_x, pos.x() - self._scale_drag_offset.x()))
            y = max(min_y, min(max_y, pos.y() - self._scale_drag_offset.y()))
            anchor_x = 0.0 if max_x == min_x else (x - min_x) / (max_x - min_x)
            anchor_y = 0.0 if max_y == min_y else (y - min_y) / (max_y - min_y)
            self._custom_scale_bar.anchor_x = anchor_x
            self._custom_scale_bar.anchor_y = anchor_y
            self._custom_scale_bar.position = "custom"
            self.scale_bar_position_changed.emit(anchor_x, anchor_y)
            self.update()
        elif self._dragging:
            self._pan = pos - self._drag_start
            self.update()
        else:
            hover = (self._custom_scale_bar is not None
                     and self._scale_bar_hit_rect.contains(pos))
            if hover != self._scale_bar_hover:
                self._scale_bar_hover = hover
                self.setCursor(Qt.CursorShape.OpenHandCursor if hover
                               else Qt.CursorShape.ArrowCursor)
                self.update()
        fx, fy = self.screen_to_full(pos.x(), pos.y())
        self.cursor_moved.emit(fx, fy)

    def resizeEvent(self, event):
        self._position_overlays()
        if not self._pixmap:
            return
        if self._fitted_once:
            # Keep the user's zoom; just hold the same image point centred.
            old = event.oldSize()
            if old.width() > 0 and old.height() > 0:
                self._pan += QPointF((self.width() - old.width()) / 2,
                                     (self.height() - old.height()) / 2)
            self.update()
            self._debounce.start()
        else:
            self.fit_in_view()

    # --- zoom field overlay ---
    def _position_overlays(self):
        self.zoom_control.move(8, self.height() - self.zoom_control.height() - 8)

    def _sync_zoom_field(self):
        if self._pixmap:
            self.zoom_control.show()
            self.zoom_control.raise_()
        if not self.zoom_edit.hasFocus():
            self.zoom_edit.setText(f"{self.screen_px_per_full_px * 100:.0f}%")

    def _on_zoom_edit(self):
        if not self._pixmap or self._zoom_edit_busy:
            return
        txt = self.zoom_edit.text().strip().rstrip("%").strip()
        try:
            pct = float(txt)
        except ValueError:
            self._sync_zoom_field()
            return
        pct = max(1.0, min(20000.0, pct))
        target_zoom = (pct / 100.0) * self._overview_ds
        center = QPointF(self.width() / 2, self.height() / 2)
        factor = target_zoom / self._zoom if self._zoom else 1.0
        # Release the field itself — clearing focus on the canvas would leave the
        # QLineEdit focused, freezing the readout and re-applying this value later.
        self._zoom_edit_busy = True
        try:
            self.zoom_edit.clearFocus()   # re-fires editingFinished; guarded above
            self.setFocus()
            self._apply_zoom(factor, center)
        finally:
            self._zoom_edit_busy = False
        self._sync_zoom_field()


class DetailWorker(QThread):
    """Loads a full-res region composite for the current viewport."""
    ready = Signal(object, object, int)  # QImage, (x0,y0,w,h), generation

    def __init__(self, channel_paths, channel_views, full_dims, rect_full, detail_ds, gen):
        super().__init__()
        self.channel_paths = channel_paths      # dict ch_id -> (Path, (cw,ch))
        self.channel_views = channel_views      # list of render.ChannelView
        self.full_w, self.full_h = full_dims
        self.rect_full = rect_full               # (x0,y0,x1,y1)
        self.detail_ds = detail_ds
        self.gen = gen

    def run(self):
        try:
            x0, y0, x1, y1 = self.rect_full
            rw = max(1, (x1 - x0) // self.detail_ds)
            rh = max(1, (y1 - y0) // self.detail_ds)
            images = {}
            for cv in self.channel_views:
                if not cv.enabled or cv.ch_id not in self.channel_paths:
                    continue
                path, (cw, ch) = self.channel_paths[cv.ch_id]
                sx = cw / self.full_w
                sy = ch / self.full_h
                cx0, cy0 = int(x0 * sx), int(y0 * sy)
                cx1, cy1 = int(x1 * sx), int(y1 * sy)
                ds = max(1, int(self.detail_ds * sx))
                region = ktf_reader.reconstruct_region(path, cx0, cy0, cx1, cy1, downsample=ds)
                if region.shape != (rh, rw):
                    region = np.array(
                        Image.fromarray(region).resize((rw, rh), Image.Resampling.BILINEAR)
                    )
                images[cv.ch_id] = region
            if not images:
                return
            rgb = render.composite(self.channel_views, images)
            arr = np.ascontiguousarray(rgb)
            h, w, _ = arr.shape
            qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
            # QImage is a reentrant value object and may cross a queued signal;
            # QPixmap is a GUI resource and must be created on the main thread.
            self.ready.emit(qimg, (x0, y0, x1 - x0, y1 - y0), self.gen)
        except Exception as e:
            print(f"Detail load error: {e}")
            self.ready.emit(None, None, self.gen)


class LoadWorker(QThread):
    # Keep QThread.finished() available as the no-argument lifecycle signal.
    # PySide exposes inherited C++ signals strictly, so use a separate payload
    # signal instead of shadowing it with an incompatible signature.
    loaded = Signal(str, object, int)  # ch_id, image, generation

    def __init__(self, path, ch_id, downsample=1, gen=0):
        super().__init__()
        self.path = path
        self.ch_id = ch_id
        self.downsample = downsample
        self.gen = gen

    def run(self):
        try:
            img = ktf_reader.reconstruct_image(self.path, downsample=self.downsample)
            self.loaded.emit(self.ch_id, img, self.gen)
        except Exception as e:
            print(f"Error loading {self.path}: {e}")
            self.loaded.emit(self.ch_id, None, self.gen)


class ChannelControl(QWidget):
    changed = Signal()

    def __init__(self, ch_id, display_name, color, parent=None):
        super().__init__(parent)
        self.ch_id = ch_id
        self.color = color
        self.setMinimumWidth(540)  # keep sliders usable; scroll area scrolls if narrower
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        self.checkbox = QCheckBox(display_name)
        self.checkbox.setChecked(True)
        self.checkbox.setMinimumWidth(130)
        self.checkbox.stateChanged.connect(self.changed.emit)
        layout.addWidget(self.checkbox)

        self.color_btn = QToolButton()
        self.color_btn.setFixedSize(20, 20)
        self._update_color_btn()
        self.color_btn.clicked.connect(self._pick_color)
        layout.addWidget(self.color_btn)

        self.solo_btn = QToolButton()
        self.solo_btn.setText("S")
        self.solo_btn.setCheckable(True)
        self.solo_btn.setFixedSize(20, 20)
        self.solo_btn.setToolTip("Solo (single-channel grayscale)")
        self.solo_btn.toggled.connect(self.changed.emit)
        layout.addWidget(self.solo_btn)

        layout.addWidget(QLabel("min"))
        self.slider_min = self._mk_slider(0, 255, 0)
        layout.addWidget(self.slider_min)
        layout.addWidget(QLabel("max"))
        self.slider_max = self._mk_slider(1, 255, 255)
        layout.addWidget(self.slider_max)
        layout.addWidget(QLabel("γ"))
        self.slider_gamma = self._mk_slider(10, 300, 100)  # gamma*100
        layout.addWidget(self.slider_gamma)

    def _mk_slider(self, lo, hi, val):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.setFixedWidth(70)
        s.valueChanged.connect(self.changed.emit)
        return s

    def _update_color_btn(self):
        r, g, b = self.color
        self.color_btn.setStyleSheet(
            f"background: rgb({r},{g},{b}); border: 1px solid #777; border-radius: 3px;")

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(*self.color), self, "Channel color")
        if c.isValid():
            self.color = (c.red(), c.green(), c.blue())
            self._update_color_btn()
            self.changed.emit()

    def to_view(self) -> render.ChannelView:
        return render.ChannelView(
            ch_id=self.ch_id, color=self.color,
            lo=self.slider_min.value(),
            hi=max(self.slider_max.value(), self.slider_min.value() + 1),
            gamma=self.slider_gamma.value() / 100.0,
            enabled=self.checkbox.isChecked(),
            solo=self.solo_btn.isChecked(),
        )

    def auto_contrast(self, image):
        nz = image[image > 0]
        if len(nz) == 0:
            return
        lo = int(np.percentile(nz, 1))
        hi = int(np.percentile(nz, 99.5))
        self.slider_min.blockSignals(True)
        self.slider_max.blockSignals(True)
        self.slider_min.setValue(lo)
        self.slider_max.setValue(max(hi, lo + 1))
        self.slider_min.blockSignals(False)
        self.slider_max.blockSignals(False)


class WellPlateWidget(QWidget):
    well_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setSpacing(4)

    def set_raw_wells(self, wells: dict):
        """Raw mode has no embedded thumbnails — show the field/Z/channel counts."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not wells:
            return
        rows = sorted({w[0] for w in wells})
        cols = sorted({w[1:] for w in wells})
        for ci, col in enumerate(cols):
            lbl = QLabel(col); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#3c4149; font-weight:bold;")
            self._layout.addWidget(lbl, 0, ci + 1)
        for ri, row in enumerate(rows):
            lbl = QLabel(row); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#3c4149; font-weight:bold;")
            self._layout.addWidget(lbl, ri + 1, 0)
            for ci, col in enumerate(cols):
                wid = f"{row}{col}"
                if wid not in wells:
                    continue
                wt = wells[wid]
                btn = QPushButton(
                    f"{wid}\n{wt.n_tiles} 視野\nZ {len(wt.z_values)}\n{','.join(wt.channels)}")
                btn.setFixedSize(120, 100)
                btn.setCheckable(True)
                btn.setToolTip(f"{wid}: {wt.n_tiles} fields, {len(wt.z_values)} Z, "
                               f"{', '.join(wt.channels)}")
                btn.setStyleSheet("""
                    QPushButton { background:#ffffff; border:1px dashed #9aa1aa; border-radius:4px;
                                  color:#1c1e21; font-size:10px; }
                    QPushButton:hover { border-color:#5b8def; background:#eef3fb; }
                    QPushButton:checked { border:2px solid #2f9e6e; background:#e6f5ee; }
                """)
                btn.clicked.connect(lambda _, w=wid: self._pick_raw(w))
                self._layout.addWidget(btn, ri + 1, ci + 1)

    def _pick_raw(self, well_id):
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            if isinstance(w, QPushButton):
                w.setChecked(w.text().split("\n")[0] == well_id)
        self.well_clicked.emit(well_id)

    def set_wells(self, well_data: dict):
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not well_data:
            return
        rows = sorted(set(w[0] for w in well_data))
        cols = sorted(set(w[1:] for w in well_data))
        for ci, col in enumerate(cols):
            lbl = QLabel(col)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#3c4149; font-weight:bold;")
            self._layout.addWidget(lbl, 0, ci + 1)
        for ri, row in enumerate(rows):
            lbl = QLabel(row)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#3c4149; font-weight:bold;")
            self._layout.addWidget(lbl, ri + 1, 0)
            for ci, col in enumerate(cols):
                wid = f"{row}{col}"
                if wid in well_data:
                    self._layout.addWidget(self._mk_button(wid, well_data[wid]), ri + 1, ci + 1)

    def _mk_button(self, well_id, channels):
        btn = QPushButton()
        btn.setFixedSize(120, 100)
        btn.setToolTip(f"{well_id}: {', '.join(sorted(channels.keys()))}")
        btn.setStyleSheet("""
            QPushButton { background:#ffffff; border:1px solid #c2c7ce; border-radius:4px;
                          color:#1c1e21; font-size:11px; }
            QPushButton:hover { border-color:#5b8def; background:#eef3fb; }
        """)
        first = list(channels.values())[0]
        if first.thumbnail_jpeg:
            qimg = QImage.fromData(first.thumbnail_jpeg)
            if not qimg.isNull():
                pix = QPixmap.fromImage(qimg).scaled(
                    110, 80, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                btn.setIcon(QIcon(pix))
                btn.setIconSize(QSize(110, 80))
        btn.setText(well_id)
        btn.clicked.connect(lambda _, w=well_id: self.well_clicked.emit(w))
        return btn


class WellConditionsTable(QTableWidget):
    """Editable per-well sample-conditions grid with Excel-style copy/paste.

    - Ctrl/Cmd+V pastes tab/newline-separated clipboard data starting at the current
      cell (extra columns are added automatically to fit a wide paste).
    - Ctrl/Cmd+C copies the selection as TSV; Delete/Backspace clears cells.
    - Edits are reported via `edited` so the host can persist them.
    """

    edited = Signal()
    DEFAULT_HEADERS = ["Well", "Sample", "Treatment", "Conc.", "Time", "Notes"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._suppress = False
        self.setColumnCount(len(self.DEFAULT_HEADERS))
        self.setHorizontalHeaderLabels(self.DEFAULT_HEADERS)
        self.verticalHeader().setVisible(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.setStyleSheet("""
            QTableWidget { background:#ffffff; color:#1c1e21; gridline-color:#dfe3e8;
                           selection-background-color:#cfe1fb; selection-color:#0d1117; }
            QHeaderView::section { background:#eceff3; color:#3c4149;
                                   border:1px solid #d6dae0; padding:3px; }
            QTableWidget QLineEdit { background:#ffffff; color:#1c1e21; }
        """)
        self.itemChanged.connect(self._on_item_changed)

        # Column management: double-click a header to rename, right-click for a menu.
        header = self.horizontalHeader()
        header.setSectionsClickable(True)
        header.sectionDoubleClicked.connect(self.rename_column)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._header_menu)

    # --- column management ---
    def _header_text(self, c):
        it = self.horizontalHeaderItem(c)
        return it.text() if it else f"Col{c}"

    def _header_menu(self, pos):
        col = self.horizontalHeader().logicalIndexAt(pos)
        menu = QMenu(self)
        act_rename = menu.addAction("Rename column…")
        act_add = menu.addAction("Add column…")
        act_del = menu.addAction("Delete column")
        if col <= 0:                      # the Well column is fixed
            act_rename.setEnabled(False)
            act_del.setEnabled(False)
        chosen = menu.exec(self.horizontalHeader().mapToGlobal(pos))
        if chosen == act_rename:
            self.rename_column(col)
        elif chosen == act_add:
            self.add_column(after=col)
        elif chosen == act_del:
            self.delete_column(col)

    def rename_column(self, col):
        if col is None or col <= 0:       # never rename "Well"
            return
        current = self._header_text(col)
        name, ok = QInputDialog.getText(self, "Rename column", "Column title:", text=current)
        if not ok:
            return
        name = name.strip() or current
        self.setHorizontalHeaderItem(col, QTableWidgetItem(name))
        self.resizeColumnsToContents()
        self.edited.emit()

    def add_column(self, after=None):
        name, ok = QInputDialog.getText(self, "Add column", "Column title:")
        if not ok:
            return
        name = name.strip() or f"Col{self.columnCount()}"
        at = self.columnCount() if after is None or after < 1 else after + 1
        self._suppress = True
        self.insertColumn(at)
        self.setHorizontalHeaderItem(at, QTableWidgetItem(name))
        for r in range(self.rowCount()):
            self.setItem(r, at, QTableWidgetItem(""))
        self._suppress = False
        self.resizeColumnsToContents()
        self.edited.emit()

    def delete_column(self, col):
        if col is None or col <= 0 or self.columnCount() <= 2:
            return
        name = self._header_text(col)
        if QMessageBox.question(
                self, "Delete column",
                f"Delete column “{name}” and its values for every well?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        self._suppress = True
        self.removeColumn(col)
        self._suppress = False
        self.resizeColumnsToContents()
        self.edited.emit()

    def current_column(self):
        c = self.currentColumn()
        return c if c is not None and c > 0 else None

    def set_wells(self, well_ids, saved: dict):
        """Populate the Well column and restore any saved condition values."""
        self._suppress = True
        headers = saved.get("__headers__") or self.DEFAULT_HEADERS
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setRowCount(len(well_ids))
        for r, wid in enumerate(well_ids):
            well_item = QTableWidgetItem(wid)
            well_item.setFlags(well_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            well_item.setForeground(QColor(29, 78, 216))
            self.setItem(r, 0, well_item)
            row_vals = saved.get(wid, [])
            for c in range(1, self.columnCount()):
                val = row_vals[c - 1] if c - 1 < len(row_vals) else ""
                self.setItem(r, c, QTableWidgetItem(val))
        self.resizeColumnsToContents()
        self._suppress = False

    def to_dict(self) -> dict:
        """Serialize to {well: [col1..], '__headers__': [...]} for persistence."""
        data = {"__headers__": [self.horizontalHeaderItem(c).text()
                                for c in range(self.columnCount())]}
        for r in range(self.rowCount()):
            well_item = self.item(r, 0)
            if not well_item:
                continue
            vals = []
            for c in range(1, self.columnCount()):
                it = self.item(r, c)
                vals.append(it.text() if it else "")
            data[well_item.text()] = vals
        return data

    def _on_item_changed(self, _item):
        if not self._suppress:
            self.edited.emit()

    def keyPressEvent(self, event):
        mod = event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        if mod and event.key() == Qt.Key.Key_V:
            self._paste()
            return
        if mod and event.key() == Qt.Key.Key_C:
            self._copy()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._clear_selection()
            return
        super().keyPressEvent(event)

    def _paste(self):
        text = QApplication.clipboard().text()
        if not text:
            return
        rows = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n").split("\n")
        cur = self.currentIndex()
        r0 = max(0, cur.row())
        c0 = max(1, cur.column())  # never overwrite the Well column
        self._suppress = True
        for dr, line in enumerate(rows):
            cells = line.split("\t")
            r = r0 + dr
            if r >= self.rowCount():
                break
            for dc, val in enumerate(cells):
                c = c0 + dc
                if c >= self.columnCount():  # grow columns to fit wide pastes
                    self.setColumnCount(c + 1)
                    self.setHorizontalHeaderItem(c, QTableWidgetItem(f"Col{c}"))
                it = self.item(r, c)
                if it is None:
                    it = QTableWidgetItem()
                    self.setItem(r, c, it)
                it.setText(val)
        self._suppress = False
        self.resizeColumnsToContents()
        self.edited.emit()

    def _copy(self):
        sel = self.selectedRanges()
        if not sel:
            return
        rng = sel[0]
        lines = []
        for r in range(rng.topRow(), rng.bottomRow() + 1):
            cells = []
            for c in range(rng.leftColumn(), rng.rightColumn() + 1):
                it = self.item(r, c)
                cells.append(it.text() if it else "")
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))

    def _clear_selection(self):
        self._suppress = True
        for it in self.selectedItems():
            if it.column() != 0:
                it.setText("")
        self._suppress = False
        self.edited.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.setMinimumSize(1200, 800)
        self.resize(1500, 950)

        self._experiment = None
        self._channel_images = {}     # ch_id -> overview 2D array
        self._channel_controls = {}   # ch_id -> ChannelControl
        self._channel_info = {}       # ch_id -> KtfInfo
        self._current_well = None
        self._overview_ds = 1
        self._full_dims = (0, 0)
        self._um_per_px_full = 0.0
        self._workers = []
        self._detail_worker = None
        self._pending = 0
        self._gen = 0                 # bumped on every well load; stale worker results discarded
        self._detail_pending = False  # a viewport changed while a detail worker was running
        self._loading_experiment = False
        self._mode = None             # None | "ktf" | "raw" | "mosaic"
        self._raw_experiment = None   # raw model, kept separate from _experiment
        self._current_raw_well = None
        self._mosaic_dataset = None
        self._mosaic_geometry = None
        self._mosaic_images = {}
        self._mosaic_preview_ds = 1
        self._mosaic_metadata_worker = None
        self._mosaic_build_worker = None
        self._mosaic_export_worker = None
        self._mosaic_detail_worker = None
        self._scan_worker = None
        self._pending_root = None     # committed to settings only after a good load
        self._exporting = False       # a synchronous export is running
        self._stitch_worker = None    # asynchronous stitch (own lifetime)
        self._current_composite = None
        self._plate_series_dock = None
        self._plate_series_builder = None

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_plate_series_dock()
        self._apply_style()
        self._update_checker = None
        if QSettings().value("check_updates", "1") == "1":
            QTimer.singleShot(2500, lambda: self._check_updates(quiet=True))

    # ---------- UI ----------
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # left
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        tree_group = QGroupBox("Experiments")
        self.tree_group = tree_group
        tg = QVBoxLayout(tree_group)
        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderLabels(["Name", "Wells", "Channels"])
        self.folder_tree.setColumnWidth(0, 200)
        self.folder_tree.itemDoubleClicked.connect(self._on_experiment_selected)
        tg.addWidget(self.folder_tree)
        ll.addWidget(tree_group)
        well_group = QGroupBox("Well Plate")
        self.well_group = well_group
        wg = QVBoxLayout(well_group)
        well_tabs = QTabWidget()
        self.well_tabs = well_tabs

        self.well_plate = WellPlateWidget()
        self.well_plate.well_clicked.connect(self._dispatch_well_clicked)
        ws = QScrollArea()
        ws.setWidget(self.well_plate)
        ws.setWidgetResizable(True)
        well_tabs.addTab(ws, "Plate")

        cond_tab = QWidget()
        self.cond_tab = cond_tab
        ct = QVBoxLayout(cond_tab)
        ct.setContentsMargins(2, 2, 2, 2)
        self.conditions = WellConditionsTable()
        self.conditions.edited.connect(self._save_conditions)
        ct.addWidget(self.conditions)
        cond_hint = QLabel("Paste from Excel (⌘/Ctrl+V) · double-click or right-click a "
                           "column title to rename · saved per experiment.")
        cond_hint.setWordWrap(True)
        cond_hint.setStyleSheet("color:#6b7280; font-size:10px;")
        ct.addWidget(cond_hint)
        cond_btns = QHBoxLayout()
        for label, tip, slot in [
            ("+ Col", "Add a column after the selected one",
             lambda: self.conditions.add_column(after=self.conditions.current_column())),
            ("Rename", "Rename the selected column",
             lambda: self.conditions.rename_column(self.conditions.current_column())),
            ("− Col", "Delete the selected column",
             lambda: self.conditions.delete_column(self.conditions.current_column())),
        ]:
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            cond_btns.addWidget(b)
        b_clear = QPushButton("全消去")
        b_clear.setToolTip("この実験のサンプル条件をすべて消去します")
        b_clear.clicked.connect(self._clear_conditions)
        cond_btns.addWidget(b_clear)
        cond_btns.addStretch()
        b_csv = QPushButton("Export CSV…")
        b_csv.clicked.connect(self._export_conditions_csv)
        cond_btns.addWidget(b_csv)
        ct.addLayout(cond_btns)
        well_tabs.addTab(cond_tab, "Conditions")

        wg.addWidget(well_tabs)
        ll.addWidget(well_group)
        left.setMaximumWidth(600)
        splitter.addWidget(left)

        # right
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.canvas = ImageCanvas()
        self.canvas.view_changed.connect(self._refresh_detail)
        self.canvas.cursor_moved.connect(self._on_cursor)
        rl.addWidget(self.canvas, stretch=1)

        self.readout = QLabel("")
        self.readout.setFont(QFont("Menlo", 11))
        self.readout.setStyleSheet("color:#1d4ed8; padding:2px 6px;")
        self.readout.setFixedHeight(22)
        rl.addWidget(self.readout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        progress_host = QWidget()
        progress_layout = QHBoxLayout(progress_host)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(6)
        progress_layout.addWidget(self.progress_bar, 1)
        self.btn_cancel_operation = QPushButton("処理を中止")
        self.btn_cancel_operation.setMaximumWidth(110)
        self.btn_cancel_operation.setToolTip("現在の長い処理を安全な区切りで中止します")
        self.btn_cancel_operation.clicked.connect(self._cancel_active_operation)
        self.btn_cancel_operation.hide()
        progress_layout.addWidget(self.btn_cancel_operation)
        rl.addWidget(progress_host)

        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(4, 4, 4, 4)

        ch_group = QGroupBox("Channels")
        cg = QVBoxLayout(ch_group)
        ch_scroll = QScrollArea()
        ch_scroll.setWidgetResizable(True)
        ch_host = QWidget()
        self.channel_layout = QVBoxLayout(ch_host)
        self.channel_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_layout.addStretch()
        ch_scroll.setWidget(ch_host)
        cg.addWidget(ch_scroll)
        bl.addWidget(ch_group, stretch=2)

        meta_group = QGroupBox("Metadata")
        mg = QVBoxLayout(meta_group)
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setFont(QFont("Menlo", 11))
        mg.addWidget(self.meta_text)
        bl.addWidget(meta_group, stretch=1)

        export_group = QGroupBox("Export")
        export_group.setObjectName("quickExportGroup")
        eg = QVBoxLayout(export_group)
        eg.setSpacing(4)
        b1 = QPushButton("表示画像をPNG保存…")
        b1.clicked.connect(lambda: self._export("png"))
        b2 = QPushButton("現在のウェルをTIFF保存…")
        b2.clicked.connect(lambda: self._export("tiff"))
        b3 = QPushButton("全ウェルをTIFF一括保存…")
        b3.clicked.connect(self._export_all_wells)
        self.btn_series_pdf = QPushButton("Stack / time seriesを1つのPDFへ…")
        self.btn_series_pdf.setObjectName("primaryExportButton")
        self.btn_series_pdf.setToolTip(
            "別々に撮影したStack・time pointを選び、1つのプレートPDFにまとめます")
        self.btn_series_pdf.clicked.connect(self._export_plate_pdf)
        self.lbl_series_pdf = QLabel("撮影を開くとシリーズを作成できます")
        self.lbl_series_pdf.setWordWrap(True)
        self.lbl_series_pdf.setStyleSheet("color:#6b7280; font-size:10px;")
        # Stitching belongs to the raw workflow only — it lives in the raw panel.
        for b in (b1, b2, b3, self.btn_series_pdf):
            eg.addWidget(b)
        eg.addWidget(self.lbl_series_pdf)
        eg.addStretch()
        bl.addWidget(export_group)

        bottom.setMinimumHeight(170)
        bottom.setMaximumHeight(200)
        self.ktf_bottom = bottom
        rl.addWidget(bottom, stretch=0)

        # Raw workflow panel — shown instead of the KTF viewer controls
        self.raw_panel = QGroupBox("生画像ワークフロー")
        rp = QVBoxLayout(self.raw_panel)
        self.raw_summary = QLabel("")
        self.raw_summary.setWordWrap(True)
        rp.addWidget(self.raw_summary)
        hint = QLabel("各視野のタイルを貼り合わせて 1 枚のウェル画像にします。"
                      "出力は OME-TIFF（Fiji / QuPath / napari で開けます）と PNG。")
        hint.setWordWrap(True); hint.setStyleSheet("color:#6b7280; font-size:11px;")
        rp.addWidget(hint)
        self.btn_stitch_raw = QPushButton("プレートを貼り合わせて書き出す…")
        self.btn_stitch_raw.clicked.connect(self._stitch_raw_tiles)
        rp.addWidget(self.btn_stitch_raw)
        rp.addStretch()
        self.raw_panel.setMaximumHeight(200)
        self.raw_panel.setVisible(False)
        rl.addWidget(self.raw_panel, stretch=0)

        self.mosaic_workspace = MosaicWorkspace()
        self.mosaic_workspace.build_requested.connect(self._build_mosaic)
        self.mosaic_workspace.export_requested.connect(self._export_mosaic)
        self.mosaic_workspace.views_changed.connect(self._rebuild_mosaic_preview)
        self.mosaic_workspace.scale_bar_changed.connect(self.canvas.set_scale_bar_spec)
        self.canvas.scale_bar_position_changed.connect(
            self.mosaic_workspace.set_scale_bar_position)
        self.mosaic_workspace.setVisible(False)
        rl.addWidget(self.mosaic_workspace, stretch=0)
        splitter.addWidget(right)
        splitter.setSizes([380, 1120])
        self.statusBar().showMessage("画像セットを開いて開始します")

    def _setup_menu(self):
        menu = self.menuBar()
        fm = menu.addMenu("File")
        a = QAction("ワークフローを選ぶ…", self)
        a.setStatusTip("通常画像・プレート画像・.ktfの入口を切り替えます")
        a.triggered.connect(self._show_start_chooser); fm.addAction(a)
        self.act_workflow = a
        a = QAction("画像セットを開く…", self); a.setShortcut(QKeySequence("Ctrl+O"))
        a.setStatusTip("現在のワークフローに合うフォルダを開きます")
        a.triggered.connect(self._open_folder); fm.addAction(a)
        self.act_open = a
        a = QAction("Stack / time seriesを1つのPDFへ…", self)
        a.setStatusTip("複数撮影を選び、1つのプレートPDFにまとめます")
        a.triggered.connect(self._export_plate_pdf); fm.addAction(a)
        self.act_series_pdf = a
        fm.addSeparator()
        a = QAction("Quit", self); a.setShortcut(QKeySequence("Ctrl+Q"))
        a.triggered.connect(self.close); fm.addAction(a)
        hm = menu.addMenu("Help")
        a = QAction("アップデートを確認…", self)
        a.triggered.connect(lambda: self._check_updates(quiet=False))
        hm.addAction(a)
        a = QAction("起動時にアップデートを確認", self)
        a.setCheckable(True)
        a.setChecked(QSettings().value("check_updates", "1") == "1")
        a.toggled.connect(
            lambda on: QSettings().setValue("check_updates", "1" if on else "0"))
        hm.addAction(a)
        hm.addSeparator()
        a = QAction(f"About {APP_NAME}", self)
        a.triggered.connect(self._about)
        hm.addAction(a)
        vm = menu.addMenu("View")
        a = QAction("Fit in View", self); a.setShortcut(QKeySequence("Ctrl+0"))
        a.triggered.connect(self.canvas.fit_in_view); vm.addAction(a)
        a = QAction("Actual Pixels (100%)", self); a.setShortcut(QKeySequence("Ctrl+1"))
        a.triggered.connect(self.canvas.zoom_actual_pixels); vm.addAction(a)
        a = QAction("Auto Brightness/Contrast", self); a.setShortcut(QKeySequence("Ctrl+Shift+A"))
        a.triggered.connect(self._auto_contrast_all); vm.addAction(a)
        self._update_series_export_state()

    def _setup_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        for label, tip, slot in [
            ("開く", "画像セットを開く (⌘O)", self._open_folder),
            ("全体", "画像全体を表示 (⌘0)", self.canvas.fit_in_view),
            ("100%", "Actual pixels (⌘1)", self.canvas.zoom_actual_pixels),
            ("Auto B/C", "Auto brightness/contrast (⇧⌘A)", self._auto_contrast_all),
        ]:
            act = QAction(label, self)
            act.setToolTip(tip)
            act.triggered.connect(slot)
            tb.addAction(act)

    def _setup_plate_series_dock(self):
        builder = PlateSeriesBuilder(dict(self.PDF_QUALITY), self)
        builder.capture_activated.connect(self._activate_series_capture)
        builder.conditions_requested.connect(self._edit_series_conditions)
        builder.export_requested.connect(self._export_plate_pdf_from_builder)
        dock = QDockWidget("Series PDF Builder", self)
        dock.setObjectName("PlateSeriesDock")
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        dock.setWidget(builder)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self._plate_series_builder = builder
        self._plate_series_dock = dock
        dock.visibilityChanged.connect(self._on_plate_series_dock_visibility)
        dock.hide()

    def _on_plate_series_dock_visibility(self, visible):
        # The builder needs a little more vertical space than the normal image
        # workspace. The viewer remains fully usable, and returns to its larger
        # minimum as soon as the builder is closed.
        self.canvas.setMinimumHeight(180 if visible else 300)
        if not visible:
            self._plate_series_builder.flush_edits()

    def _update_series_export_state(self):
        ready = bool(
            self._mode == StartModeDialog.KTF
            and self._experiment
            and self._experiment.get("wells"))
        button = getattr(self, "btn_series_pdf", None)
        if button is not None:
            button.setEnabled(ready)
        action = getattr(self, "act_series_pdf", None)
        if action is not None:
            action.setEnabled(ready)
        hint = getattr(self, "lbl_series_pdf", None)
        if hint is not None:
            hint.setText(
                "複数の撮影フォルダを1つのPDFへ"
                if ready else "撮影を開くとシリーズを作成できます")

    def _check_updates(self, quiet=True):
        """Ask GitHub whether a newer release exists.

        Quiet at startup — it only speaks up when there IS an update, so being
        offline or rate-limited is silent. The Help menu runs it loudly.
        """
        if self._update_checker and self._update_checker.isRunning():
            return
        self._update_checker = UpdateChecker()
        self._update_checker.found.connect(self._on_update_found)
        if not quiet:
            self.statusBar().showMessage("アップデートを確認しています…")
            self._update_checker.finished.connect(self._on_check_finished)
        self._update_checker.start()

    def _on_check_finished(self):
        if not getattr(self, "_update_seen", False):
            self.statusBar().showMessage(
                f"最新版を使用しています（{__version__}）")
        self._update_seen = False

    def _on_update_found(self, version, url):
        self._update_seen = True
        self.statusBar().showMessage(f"新しいバージョン {version} が利用できます")
        box = QMessageBox(self)
        box.setStyleSheet("")
        box.setWindowTitle("アップデートがあります")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(f"{APP_NAME} <b>{version}</b> が公開されています"
                    f"（現在 {__version__}）。")
        go = box.addButton("ダウンロードページを開く", QMessageBox.ButtonRole.ActionRole)
        box.addButton("あとで", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is go and url:
            QDesktopServices.openUrl(QUrl(url))

    def _about(self):
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> {__version__}<br><br>"
            "広範囲Stitching／プレート画像／.ktf ビューア<br>"
            "&copy; 2026 yoshi-koba-lab — All Rights Reserved.<br><br>"
            "UI: PySide6 6.10.3 / Qt 6.10.3<br>"
            "Qt for Python and Qt &copy; The Qt Company Ltd. and contributors; "
            "provided under LGPLv3.<br>"
            "Complete license texts, notices and replacement instructions are included "
            "in <code>THIRD_PARTY_NOTICES.md</code> and <code>licenses/</code>.<br>"
            "<a href='https://doc.qt.io/qtforpython-6/'>Qt for Python</a><br><br>"
            "<a href='https://github.com/yoshi-koba-lab/bz-plate-studio'>"
            "github.com/yoshi-koba-lab/bz-plate-studio</a>")

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#f5f7fa; color:#20252b; }
            QGroupBox { border:1px solid #d2d8e0; border-radius:8px; margin-top:10px;
                        padding-top:14px; font-weight:bold; color:#3f4750;
                        background:#fbfcfd; }
            QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 3px; }
            QTreeWidget, QTextEdit, QTableWidget, QLineEdit, QComboBox {
                background:#ffffff; border:1px solid #cbd3dc; border-radius:6px;
                color:#20252b; selection-background-color:#dcecff; }
            QTreeWidget { selection-background-color:#cfe1fb; selection-color:#0d1117;
                          alternate-background-color:#f7f8fa; }
            QTreeWidget::item { padding:4px 3px; }
            QHeaderView::section { background:#eef1f5; color:#4b535c;
                                   border:none; border-right:1px solid #d9dee5;
                                   border-bottom:1px solid #d1d7de; padding:5px; }
            QLineEdit, QComboBox { padding:5px 7px; }
            QTabBar::tab { background:#e9edf2; color:#48515a; padding:7px 14px;
                           border:1px solid #d0d6de; border-bottom:none;
                           border-top-left-radius:6px; border-top-right-radius:6px; }
            QTabBar::tab:selected { background:#ffffff; color:#0d1117; }
            QSlider::groove:horizontal { height:4px; background:#d3d7dd; border-radius:2px; }
            QSlider::handle:horizontal { width:11px; height:11px; margin:-4px 0;
                                         background:#6b7280; border-radius:5px; }
            QPushButton { background:#ffffff; border:1px solid #c4ccd5; border-radius:7px;
                          padding:7px 11px; color:#20252b; }
            QPushButton:hover { background:#eef3fb; border-color:#5b8def; }
            QPushButton:disabled { background:#f0f1f3; color:#a0a4ab; border-color:#dcdfe4; }
            QPushButton#primaryExportButton { background:#1674c5; border-color:#1265ac;
                                              color:white; font-weight:bold; }
            QPushButton#primaryExportButton:hover { background:#0f67b4; border-color:#0b599d; }
            QPushButton#primaryExportButton:disabled { background:#d9e0e7; color:#8a929b;
                                                       border-color:#cbd2d9; }
            QGroupBox#quickExportGroup QPushButton { padding:4px 8px; }
            QToolButton { background:#ffffff; border:1px solid #c2c7ce; border-radius:3px;
                          color:#1c1e21; }
            QToolButton:checked { background:#ffd8a8; border-color:#e8973a; }
            QStatusBar { background:#edf1f5; color:#4b535c; }
            QProgressBar { background:#dfe3e8; border:none; }
            QProgressBar::chunk { background:#3b82f6; }
            QDockWidget#PlateSeriesDock { color:#27313a; font-weight:bold; }
            QWidget#seriesEditorBar { background:#eef4fb; border:1px solid #cbd9e8;
                                      border-radius:7px; }
        """)

    # ---------- scanning ----------
    # ---------- workflow selection ----------
    def _recent(self, mode):
        s = QSettings()
        if mode == StartModeDialog.KTF:
            return s.value("last_ktf_root", "") or s.value("last_root", "") or ""
        if mode == StartModeDialog.RAW:
            return s.value("last_raw_root", "") or ""
        return s.value("last_mosaic_root", "") or ""

    def _last_export_dir(self) -> str:
        """Where the previous export went, so Save-As opens somewhere useful."""
        d = QSettings().value("last_export_dir", "") or ""
        return d if d and Path(d).is_dir() else str(Path.home())

    def _remember_export_dir(self, folder: Path):
        QSettings().setValue("last_export_dir", str(folder))

    def _show_start_chooser(self):
        if self._busy:
            self.statusBar().showMessage("処理中です — 完了までお待ちください。")
            return
        dlg = StartModeDialog(
            self, last_mosaic=self._recent(StartModeDialog.MOSAIC),
            last_raw=self._recent(StartModeDialog.RAW),
            last_ktf=self._recent(StartModeDialog.KTF))
        if dlg.exec() and dlg.choice:
            self._choose_folder(dlg.choice)
        elif self._mode is None:
            self.statusBar().showMessage(
                "File ▸ ワークフローを選ぶ… から開始してください")

    def _choose_folder(self, mode):
        start = self._recent(mode) or str(Path.home())
        if not Path(start).is_dir():
            start = str(Path.home())
        if mode == StartModeDialog.MOSAIC:
            title = "通常画像セット（.gciを含む撮影フォルダ、またはその親）を選択"
        elif mode == StartModeDialog.RAW:
            title = "プレート画像セット（未貼り合わせタイル、またはその親）を選択"
        else:
            title = ".ktf画像セット（撮影フォルダ、またはその親）を選択"
        f = QFileDialog.getExistingDirectory(self, title, start)
        if not f:
            return
        self._open_path(Path(f), mode)

    def _open_folder(self):
        if self._mode is None:
            self._show_start_chooser()
        else:
            self._choose_folder(self._mode)

    def _open_data_root(self):
        self._open_folder()

    @staticmethod
    def _find_experiment_dirs(folder: Path) -> list:
        """All directories at/under `folder` that directly contain .ktf files.

        Descent stops once a directory with .ktf is found (the tile subfolders
        below it never contain .ktf), so this stays fast even on large trees.
        """
        found = []
        for dirpath, dirnames, filenames in os.walk(folder):
            if any(ktf_reader.is_ktf_file(Path(f)) for f in filenames):
                found.append(Path(dirpath))
                dirnames[:] = []  # prune tile subfolders
        return sorted(found)

    def _open_path(self, folder: Path, mode: str = None):
        """Discover experiments of `mode` under `folder`, off the GUI thread."""
        mode = mode or self._mode or StartModeDialog.KTF
        if self._busy:
            self.statusBar().showMessage("処理中です — 完了までお待ちください。")
            return
        # A whole drive can hold tens of thousands of directories.
        if folder.parent == folder or str(folder).rstrip("/") in ("/Volumes", ""):
            ans = QMessageBox.question(
                self, "ドライブ全体をスキャンしますか",
                f"“{folder}” 全体の走査は数分かかることがあります。\n続けますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return

        self._pending_root = (folder, mode)
        self.statusBar().showMessage(f"スキャン中: {folder} …")
        self.progress_bar.setRange(0, 0)          # indeterminate
        self.progress_bar.show()
        self._show_cancel_operation(True)
        self._set_actions_enabled(False)
        if self._plate_series_builder is not None:
            self._plate_series_builder.setEnabled(False)
        self._scan_worker = ScanWorker(folder, mode)
        self._scan_worker.progress.connect(
            lambda d, n: self.statusBar().showMessage(f"スキャン中 ({n} フォルダ): {d}"))
        self._scan_worker.finished_scan.connect(self._on_scan_done)
        self._scan_worker.start()

    def _set_actions_enabled(self, on: bool):
        for a in (getattr(self, "act_open", None), getattr(self, "act_workflow", None)):
            if a is not None:
                a.setEnabled(on)

    def _clear_loaded_experiment(self):
        self._save_conditions()
        self._experiment = None
        self._raw_experiment = None
        self._mosaic_dataset = None
        self._mosaic_geometry = None
        self._mosaic_images.clear()
        self._mosaic_preview_ds = 1
        self._current_raw_well = None
        self._reset_well_state()
        self.well_plate.set_wells({})
        self.conditions.set_wells([], {})
        if hasattr(self, "mosaic_workspace"):
            self.mosaic_workspace.clear()
        if self._plate_series_builder is not None:
            self._plate_series_builder.clear_active_path()
        self._update_raw_summary()
        self._update_series_export_state()

    def _on_scan_done(self, dirs, errors):
        self.progress_bar.hide()
        self._show_cancel_operation(False)
        self.progress_bar.setRange(0, 1000)
        self._set_actions_enabled(True)
        if self._plate_series_builder is not None:
            self._plate_series_builder.setEnabled(True)
        folder, mode = self._pending_root or (None, None)
        if dirs is None:                      # cancelled
            self.statusBar().showMessage("スキャンを中止しました。")
            return
        label = ({StartModeDialog.KTF: ".ktf",
                  StartModeDialog.RAW: "プレート生画像",
                  StartModeDialog.MOSAIC: "通常画像セット"}.get(mode, "画像"))
        if not dirs:
            self.statusBar().showMessage(
                f"“{folder.name}” の下に {label} の実験が見つかりませんでした"
                + (f"（{errors} 件のフォルダを読めませんでした）" if errors else ""))
            QMessageBox.information(
                self, "実験が見つかりません",
                f"“{folder}” の下に{label}形式の実験はありませんでした。\n\n"
                "別のフォルダを選ぶか、File ▸ ワークフローを選ぶ… で"
                "別のワークフローをお試しください。")
            return
        paths = {_path_key(path) for path in dirs}
        if mode == StartModeDialog.KTF:
            active = self._experiment
        elif mode == StartModeDialog.RAW:
            active = self._raw_experiment
        else:
            active = self._mosaic_dataset
        active_path = (active["path"] if isinstance(active, dict)
                       else getattr(active, "root", None))
        other_loaded = any(model is not None for model in (
            self._experiment if mode != StartModeDialog.KTF else None,
            self._raw_experiment if mode != StartModeDialog.RAW else None,
            self._mosaic_dataset if mode != StartModeDialog.MOSAIC else None))
        if (self._mode is not None and self._mode != mode) or other_loaded or (
                active_path is not None and _path_key(active_path) not in paths):
            self._clear_loaded_experiment()
        self._set_mode(mode)
        self.folder_tree.clear()
        if mode == StartModeDialog.KTF:
            self._populate_tree(folder, dirs)
        elif mode == StartModeDialog.RAW:
            self._populate_raw_tree(folder, dirs)
        else:
            self._populate_mosaic_tree(folder, dirs)
        if len(dirs) == 1:
            self._select_tree_item(dirs[0])
            self._on_experiment_selected(self.folder_tree.currentItem(), 0)

    def _set_mode(self, mode):
        self._mode = mode
        raw = mode == StartModeDialog.RAW
        wide = mode == StartModeDialog.MOSAIC
        if wide:
            self.folder_tree.setHeaderLabels(["Name", "Grid", "Source"])
            # The minimum 1200 px window gives this pane roughly 350 px.  Keep
            # all three fields visible instead of opening with a horizontal
            # scrollbar and a clipped Source column.
            self.folder_tree.setColumnWidth(0, 155)
            self.folder_tree.setColumnWidth(1, 60)
            self.folder_tree.setColumnWidth(2, 105)
        elif raw:
            self.folder_tree.setHeaderLabels(["Name", "Wells", "Source"])
            self.folder_tree.setColumnWidth(0, 200)
            self.folder_tree.setColumnWidth(1, 100)
            self.folder_tree.setColumnWidth(2, 100)
        else:
            self.folder_tree.setHeaderLabels(["Name", "Wells", "Channels"])
            self.folder_tree.setColumnWidth(0, 200)
            self.folder_tree.setColumnWidth(1, 100)
            self.folder_tree.setColumnWidth(2, 100)
        # Conditions apply to both workflows; they are keyed by experiment path, and
        # the table is repopulated per experiment so values can never leak across.
        if hasattr(self, "well_tabs") and self.well_tabs.indexOf(self.cond_tab) < 0:
            self.well_tabs.addTab(self.cond_tab, "Conditions")
        if hasattr(self, "raw_panel"):
            self.raw_panel.setVisible(raw)
        if hasattr(self, "ktf_bottom"):
            self.ktf_bottom.setVisible(not raw and not wide)
        if hasattr(self, "mosaic_workspace"):
            self.mosaic_workspace.setVisible(wide)
        self.canvas.set_scale_bar_spec(
            self.mosaic_workspace.scale_bar if wide else None)
        if hasattr(self, "well_group"):
            self.well_group.setVisible(not wide)
        if hasattr(self, "tree_group"):
            self.tree_group.setTitle("画像セット" if wide else "Experiments")
        if (raw or wide) and self._plate_series_dock is not None:
            self._plate_series_dock.hide()
        self._update_series_export_state()
        title = ("通常画像セット（広範囲Stitching）" if wide else
                 "プレート画像セット" if raw else ".ktf 表示")
        self.canvas.set_empty_message(
            "画像セットをダブルクリックし、「Stitching実行」を押してください"
            if wide else
            "プレート画像セットをダブルクリックし、ウェルを選択してください"
            if raw else
            ".ktf画像セットをダブルクリックし、ウェルを選択してください")
        self.setWindowTitle(f"{APP_NAME} {__version__} — {title}")

    def _commit_root(self):
        """Remember the browse root only once something actually loaded."""
        if not self._pending_root:
            return
        folder, mode = self._pending_root
        key = ({StartModeDialog.KTF: "last_ktf_root",
                StartModeDialog.RAW: "last_raw_root",
                StartModeDialog.MOSAIC: "last_mosaic_root"}[mode])
        QSettings().setValue(key, str(folder))

    def _populate_mosaic_tree(self, root: Path, exp_dirs: list):
        """List GCI-described image sets without opening hundreds of TIFFs."""
        groups = {}
        for directory in exp_dirs:
            try:
                relative = directory.relative_to(root)
            except ValueError:
                relative = Path(directory.name)
            group = relative.parts[0] if len(relative.parts) > 1 else ""
            label = (str(Path(*relative.parts[1:])) if len(relative.parts) > 1
                     else (relative.parts[0] if relative.parts else directory.name))
            if not group and re.fullmatch(r"XY\d+", label, re.IGNORECASE):
                label = f"{directory.parent.name} / {label}"
            groups.setdefault(group, []).append((label, directory))
        for group in sorted(groups, key=_natural_key):
            container = self.folder_tree
            if group:
                parent = QTreeWidgetItem(self.folder_tree)
                parent.setText(0, group)
                container = parent
            for label, directory in sorted(groups[group], key=lambda pair: _natural_key(pair[0])):
                item = QTreeWidgetItem(container)
                item.setText(0, label)
                item.setText(1, "—")
                item.setText(2, ".gci + OME-TIFF")
                item.setToolTip(0, str(directory))
                item.setData(0, Qt.ItemDataRole.UserRole, str(directory))
                item.setData(0, Qt.ItemDataRole.UserRole + 1, StartModeDialog.MOSAIC)
        self.folder_tree.expandAll()
        self.statusBar().showMessage(
            f"“{root.name}” に通常画像セットが {len(exp_dirs)} 件 — ダブルクリックで開きます")

    def _populate_raw_tree(self, root: Path, exp_dirs: list):
        groups = {}
        for d in exp_dirs:
            try:
                rel = d.relative_to(root)
            except ValueError:
                rel = Path(d.name)
            grp = rel.parts[0] if len(rel.parts) > 1 else ""
            label = str(Path(*rel.parts[1:])) if len(rel.parts) > 1 else (
                rel.parts[0] if rel.parts else d.name)
            groups.setdefault(grp, []).append((label, d))
        for grp in sorted(groups):
            container = self.folder_tree
            if grp:
                parent = QTreeWidgetItem(self.folder_tree)
                parent.setText(0, grp)
                container = parent
            for label, d in sorted(groups[grp]):
                item = QTreeWidgetItem(container)
                item.setText(0, label)
                item.setText(1, str(len(_raw_wells_of(d))))
                item.setText(2, "Raw tiles")
                item.setData(0, Qt.ItemDataRole.UserRole, str(d))
                item.setData(0, Qt.ItemDataRole.UserRole + 1, StartModeDialog.RAW)
        self.folder_tree.expandAll()
        self.statusBar().showMessage(
            f"“{root.name}” に生画像の実験が {len(exp_dirs)} 件 — ダブルクリックで開きます")

    def _load_raw_experiment(self, folder: Path):
        """Build the raw model with the stitcher's own reader."""
        self.statusBar().showMessage(f"{folder.name} を読み込み中…")
        QApplication.processEvents()
        try:
            wells = stitcher.discover_wells(folder)
        except Exception as e:
            self.statusBar().showMessage(f"“{folder.name}” を読めませんでした: {e}")
            return False
        structural = set(_raw_wells_of(folder))
        if not wells:
            self.statusBar().showMessage(
                f"“{folder.name}”: タイル名は見つかりましたが、読めるタイル画像がありません")
            return False
        self._experiment = None            # workflow models never coexist
        self._reset_well_state()
        self._raw_experiment = {"name": folder.name, "path": folder, "wells": wells}
        self._current_raw_well = None
        self.well_plate.set_raw_wells(wells)
        saved = self._load_conditions()
        self.conditions.set_wells(sorted(wells), saved)
        unusable = structural - set(wells)
        msg = (f"{folder.name}: {len(wells)} ウェル（生画像）— "
               f"ウェルを選ぶか、そのまま貼り合わせボタンで全ウェルを処理できます")
        if unusable:
            msg += f" / 使用不可: {', '.join(sorted(unusable))}"
        self.statusBar().showMessage(msg)
        self._update_raw_summary()
        self._update_series_export_state()
        self._commit_root()
        return True

    def _load_mosaic_experiment(self, folder: Path):
        """Load metadata for one wide-area scan; registration starts explicitly."""
        if self._busy:
            self.statusBar().showMessage("処理中です — 完了までお待ちください。")
            return False
        self._mosaic_metadata_worker = MosaicMetadataWorker(folder, self)
        self._mosaic_metadata_worker.progress.connect(self._on_mosaic_progress)
        self._mosaic_metadata_worker.ready.connect(self._on_mosaic_metadata_ready)
        self._mosaic_metadata_worker.failed.connect(self._on_mosaic_failed)
        self._mosaic_metadata_worker.finished.connect(self._on_mosaic_worker_finished)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self._show_cancel_operation(True)
        self._set_actions_enabled(False)
        self.mosaic_workspace.setEnabled(False)
        self.statusBar().showMessage(f"{folder.name}: GCI/OMEメタデータを検証中…")
        self._mosaic_metadata_worker.start()
        return True

    def _on_mosaic_progress(self, message: str, value: float):
        self.statusBar().showMessage(message)
        self.progress_bar.setValue(int(max(0.0, min(1.0, value)) * 1000))

    def _on_mosaic_metadata_ready(self, dataset):
        self._save_conditions()
        self._experiment = None
        self._raw_experiment = None
        self._mosaic_dataset = dataset
        self._mosaic_geometry = None
        self._mosaic_images.clear()
        self._mosaic_preview_ds = 1
        self._reset_well_state()
        self.mosaic_workspace.set_dataset(dataset)
        rows, cols = dataset.grid_shape
        py, px = dataset.pixel_size_um_yx
        physical = (f"{px:g} × {py:g} µm/px" if px is not None and py is not None
                    else "未取得")
        self.meta_text.setText("\n".join((
            f"Image set: {dataset.name}",
            f"Grid:      {rows} × {cols} ({len(dataset.tiles)} fields)",
            f"Tile:      {dataset.tile_shape[1]} × {dataset.tile_shape[0]} px",
            f"Channels:  {len(dataset.channels)}",
            f"Pixel:     {physical}",
            f"Source:    {dataset.gci_path}",
            f"Warnings:  {len(dataset.warnings)}",
        )))
        self._select_tree_item(dataset.root)
        current = self.folder_tree.currentItem()
        if current is not None:
            current.setText(1, f"{rows}×{cols}")
            current.setText(2, f"{len(dataset.channels)} ch")
        self._commit_root()
        self.statusBar().showMessage(
            f"{dataset.name}: {len(dataset.tiles)}視野を確認しました — 「Stitching実行」を押してください")

    def _on_mosaic_failed(self, message: str):
        self.statusBar().showMessage(f"通常画像セットを開けませんでした: {message}")
        QMessageBox.critical(
            self, "通常画像セットを開けません",
            f"GCIと元のOME-TIFFを確認できませんでした。\n\n{message}")

    def _on_mosaic_worker_finished(self):
        self.progress_bar.hide()
        self._show_cancel_operation(False)
        self._set_actions_enabled(True)
        self.mosaic_workspace.setEnabled(True)
        sender = self.sender()
        if sender is self._mosaic_metadata_worker:
            self._mosaic_metadata_worker = None
        elif sender is self._mosaic_build_worker:
            self._mosaic_build_worker = None
        elif sender is self._mosaic_export_worker:
            self._mosaic_export_worker = None
        # These workers are parented to MainWindow so dropping the Python
        # attribute alone does not release their dataset/geometry references.
        # Dispose each finished QThread through Qt's event loop.
        if sender is not None:
            sender.deleteLater()

    def _on_raw_well_selected(self, well_id: str):
        self._current_raw_well = well_id
        self._update_raw_summary()

    def _update_raw_summary(self):
        if not hasattr(self, "raw_summary"):
            return
        if not self._raw_experiment:
            self.raw_summary.setText("")
            return
        wells = self._raw_experiment["wells"]
        lines = [f"<b>{self._raw_experiment['name']}</b> — {len(wells)} ウェル"]
        if self._current_raw_well and self._current_raw_well in wells:
            wt = wells[self._current_raw_well]
            lines.append(f"選択中: <b>{self._current_raw_well}</b> — "
                         f"{wt.n_tiles} 視野 · Z {len(wt.z_values)} · "
                         f"{', '.join(wt.channels)}")
        else:
            lines.append("ウェル未選択（全ウェルを処理できます）")
        self.raw_summary.setText("<br>".join(lines))

    def _dispatch_well_clicked(self, well_id: str):
        if self._mode == StartModeDialog.RAW:
            self._on_raw_well_selected(well_id)
        elif self._mode == StartModeDialog.KTF:
            self._on_well_clicked(well_id)

    def _select_tree_item(self, path: Path):
        """Highlight the tree row whose stored path matches `path`."""
        target = _path_key(path)
        stack = [self.folder_tree.topLevelItem(i)
                 for i in range(self.folder_tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item is None:
                continue
            stored = item.data(0, Qt.ItemDataRole.UserRole)
            if stored and _path_key(stored) == target:
                self.folder_tree.setCurrentItem(item)
                self.folder_tree.scrollToItem(item)
                return True
            stack.extend(item.child(i) for i in range(item.childCount()))
        self.folder_tree.setCurrentItem(None)
        self.folder_tree.clearSelection()
        return False

    def _populate_tree(self, root: Path, exp_dirs: list):
        """Show found experiments in a tree, grouped by their parent folder."""
        groups = {}
        for d in exp_dirs:
            try:
                rel = d.relative_to(root)
            except ValueError:
                rel = Path(d.name)
            if len(rel.parts) > 1:
                grp, label = rel.parts[0], str(Path(*rel.parts[1:]))
            else:
                grp, label = "", (rel.parts[0] if rel.parts else d.name)
            groups.setdefault(grp, []).append((label, d))

        for grp in sorted(groups):
            container = self.folder_tree
            if grp:
                parent = QTreeWidgetItem(self.folder_tree)
                parent.setText(0, grp)
                container = parent
            for label, d in sorted(groups[grp]):
                wells, channels = set(), set()
                for kf in (p for p in d.iterdir() if ktf_reader.is_ktf_file(p)):
                    info = ktf_reader.KtfInfo(
                        path=kf, metadata=ktf_reader.KtfMetadata(),
                        header_size=112, footer_offset=0, file_size=0,
                        tile_entry_count=0, tile_byte_size=0)
                    if info.well_id:
                        wells.add(info.well_id)
                    if info.channel_id:
                        channels.add(info.channel_id)
                item = QTreeWidgetItem(container)
                item.setText(0, label)
                item.setText(1, str(len(wells)))
                item.setText(2, ", ".join(sorted(channels)))
                item.setData(0, Qt.ItemDataRole.UserRole, str(d))
                item.setData(0, Qt.ItemDataRole.UserRole + 1, StartModeDialog.KTF)
        self.folder_tree.expandAll()
        self.statusBar().showMessage(
            f"Found {len(exp_dirs)} experiments under “{root.name}” — double-click one to open")

    # backwards-compatible alias used by main() on startup
    def _scan_root(self, root: Path):
        self._open_path(root)

    def _on_experiment_selected(self, item, col):
        # Parent (folder) rows just expand/collapse; only leaf experiments load.
        if item is None:
            return
        if item.childCount() > 0:
            item.setExpanded(not item.isExpanded())
            return
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path:
            return
        mode = item.data(0, Qt.ItemDataRole.UserRole + 1) or StartModeDialog.KTF
        if mode == StartModeDialog.RAW:
            self._load_raw_experiment(Path(path))
        elif mode == StartModeDialog.MOSAIC:
            self._load_mosaic_experiment(Path(path))
        else:
            self._load_experiment(Path(path))

    def _load_experiment(self, folder: Path):
        if self._busy:
            self.statusBar().showMessage("Another operation is running — please wait.")
            return False
        self._loading_experiment = True
        if self._plate_series_builder is not None:
            self._plate_series_builder.setEnabled(False)
        try:
            self.statusBar().showMessage(f"Loading {folder.name}...")
            QApplication.processEvents()
            experiment = ktf_reader.scan_experiment_folder(folder)
        except Exception as e:
            self.statusBar().showMessage(f"Could not read “{folder.name}”: {e}")
            return False
        finally:
            self._loading_experiment = False
            if self._plate_series_builder is not None:
                self._plate_series_builder.setEnabled(True)
        skipped = experiment.get("errors") or []
        if not experiment["wells"]:
            self.statusBar().showMessage(
                f"No readable wells in “{experiment['name']}”"
                + (f" ({len(skipped)} file(s) unreadable)" if skipped else ""))
            return False
        self._save_conditions()
        self._experiment = experiment
        # A new experiment invalidates everything tied to the previous well.
        self._reset_well_state()
        self.well_plate.set_wells(experiment["wells"])
        # populate the sample-conditions table (restore any saved values)
        saved = self._load_conditions()
        self.conditions.set_wells(sorted(experiment["wells"]), saved)

        msg = f"{experiment['name']}: {len(experiment['wells'])} wells"
        if skipped:
            msg += f" — skipped {len(skipped)} unreadable file(s): " + \
                   ", ".join(n for n, _ in skipped[:3]) + ("…" if len(skipped) > 3 else "")
        self._raw_experiment = None       # workflow models never coexist
        self._commit_root()
        self.statusBar().showMessage(msg)
        self._update_series_export_state()
        if self._plate_series_builder is not None:
            self._plate_series_builder.set_active_path(folder)
        return True

    def _reset_well_state(self):
        """Drop everything belonging to the previously displayed well."""
        self._gen += 1
        self._detail_pending = False
        self._workers = [w for w in self._workers if w.isRunning()]
        self._current_well = None
        self._channel_images.clear()
        self._channel_info = {}
        for ctrl in self._channel_controls.values():
            ctrl.setParent(None)
            ctrl.deleteLater()
        self._channel_controls.clear()
        self._full_dims = (0, 0)
        self._um_per_px_full = 0.0
        self._current_composite = None
        self.canvas.set_detail(None, None)
        self.canvas.clear_image()
        self.meta_text.clear()
        self.readout.setText("")
        if self._plate_series_dock is not None and self._plate_series_dock.isVisible():
            self._plate_series_builder.set_channel_summary(self._pdf_channel_summary())

    # ---------- sample conditions ----------
    def _active_experiment(self):
        """Whichever experiment is loaded — conditions belong to both workflows."""
        return self._experiment or self._raw_experiment

    def _conditions_key(self):
        exp = self._active_experiment()
        p = exp["path"] if exp else None
        return f"conditions/{_path_key(p)}" if p else None

    @staticmethod
    def _condition_path_from_key(key):
        suffix = key[len("conditions/"):]
        if suffix.startswith(os.sep) or re.match(r"^[A-Za-z]:", suffix):
            return suffix
        return os.sep + suffix

    def _matching_condition_keys(self, path):
        target = _path_key(path)
        matches = []
        for key in QSettings().allKeys():
            if not key.startswith("conditions/"):
                continue
            try:
                if _path_key(self._condition_path_from_key(key)) == target:
                    matches.append(key)
            except (OSError, RuntimeError, ValueError):
                continue
        return matches

    def _load_conditions_for_path(self, path):
        settings = QSettings()
        primary = f"conditions/{_path_key(path)}"
        raw = settings.value(primary, "")
        if not raw:
            for key in self._matching_condition_keys(path):
                raw = settings.value(key, "")
                if raw:
                    settings.setValue(primary, raw)
                    break
        try:
            return json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            return {}

    def _load_conditions(self) -> dict:
        exp = self._active_experiment()
        if not exp:
            return {}
        return self._load_conditions_for_path(exp["path"])

    def _save_conditions(self):
        """Merge the visible rows into the stored blob.

        Never write the table wholesale: if the folder was opened while some wells
        were unreadable (drive spinning up, partial sync), those wells are absent
        from the table and a plain overwrite would silently delete annotations the
        user had already entered for them.
        """
        key = self._conditions_key()
        if not key:
            return
        stored = self._load_conditions()
        stored.update(self.conditions.to_dict())
        QSettings().setValue(key, json.dumps(stored))

    def _clear_conditions(self):
        exp = self._active_experiment()
        if not exp:
            return
        if QMessageBox.question(
                self, "サンプル条件の消去",
                f"“{exp['name']}” のサンプル条件をすべて消去しますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        key = self._conditions_key()
        if key:
            settings = QSettings()
            for stored_key in set(self._matching_condition_keys(exp["path"])) | {key}:
                settings.remove(stored_key)
        self.conditions.set_wells(sorted(exp["wells"]), {})
        self.statusBar().showMessage(f"{exp['name']}: サンプル条件を消去しました")

    def _export_conditions_csv(self):
        exp = self._active_experiment()
        if not exp:
            return
        default = f"{_safe_base_name(exp['name'])}_conditions.csv"
        path = _ask_save_path(self, "サンプル条件を名前を付けて保存", default, "CSV (*.csv)")
        if path is None:
            return
        t = self.conditions
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow([t.horizontalHeaderItem(c).text() for c in range(t.columnCount())])
            for r in range(t.rowCount()):
                wr.writerow([(t.item(r, c).text() if t.item(r, c) else "")
                             for c in range(t.columnCount())])
        self.statusBar().showMessage(f"Exported conditions to {path}")

    # ---------- well loading ----------
    def _on_well_clicked(self, well_id: str):
        if not self._experiment:
            return
        if self._busy:
            self.statusBar().showMessage("Another operation is running — please wait.")
            return
        channels = self._experiment["wells"].get(well_id, {})
        if not channels:
            return
        # Invalidate any in-flight workers by bumping the generation; their results
        # will be discarded when they finish (safer than QThread.terminate()).
        self._gen += 1
        self._detail_pending = False
        self.canvas.set_detail(None, None)
        self._workers = [w for w in self._workers if w.isRunning()]

        self._current_well = well_id
        self._channel_images.clear()
        self._channel_info = dict(channels)
        for c in self._channel_controls.values():
            c.setParent(None)
            c.deleteLater()
        self._channel_controls.clear()

        max_w = max(i.metadata.width for i in channels.values())
        max_h = max(i.metadata.height for i in channels.values())
        self._full_dims = (max_w, max_h)
        ds = auto_downsample(max_w, max_h)
        self._overview_ds = ds
        # calibration (µm/px at full res) from any channel
        cal = next((i.metadata.um_per_pixel for i in channels.values()
                    if i.metadata.um_per_pixel > 0), 0.0)
        self._um_per_px_full = cal

        self.statusBar().showMessage(f"Loading {well_id} ({len(channels)} ch, 1/{ds})...")
        self.progress_bar.setRange(0, len(channels))
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self._show_metadata(list(channels.values())[0].metadata, ds)

        self._pending = len(channels)
        for ch_id, info in channels.items():
            color = CHANNEL_COLORS.get(ch_id, (200, 200, 200))
            disp = f"{ch_id} · {info.metadata.channel_comment}" if info.metadata.channel_comment else ch_id
            ctrl = ChannelControl(ch_id, disp, color)
            ctrl.changed.connect(self._on_levels_changed)
            self._channel_controls[ch_id] = ctrl
            self.channel_layout.insertWidget(self.channel_layout.count() - 1, ctrl)
            w = LoadWorker(info.path, ch_id, downsample=ds, gen=self._gen)
            w.loaded.connect(self._on_channel_loaded)
            self._workers.append(w)
            w.start()

    def _on_channel_loaded(self, ch_id, image, gen):
        if gen != self._gen:
            return  # stale result from a previously selected well
        self._pending -= 1
        if image is not None:
            self._channel_images[ch_id] = image
            if ch_id in self._channel_controls:
                self._channel_controls[ch_id].auto_contrast(image)
        self.progress_bar.setValue(len(self._channel_images))
        if self._pending <= 0:
            self.progress_bar.hide()
            if not self._channel_images:
                # Every channel failed (drive asleep/ejected, corrupt files) — do not
                # leave the previous well's picture on screen labelled as this one.
                self.canvas.clear_image()
                self.statusBar().showMessage(
                    f"{self._current_well}: could not read any channel "
                    f"(is the drive still connected?)")
                return
            self.statusBar().showMessage(
                f"{self._current_well}: {len(self._channel_images)} channel(s) — scroll to zoom, drag to pan")
            self._rebuild_overview(reset_view=True)
        else:
            self._rebuild_overview(reset_view=False)
        if self._plate_series_dock is not None and self._plate_series_dock.isVisible():
            self._plate_series_builder.set_channel_summary(self._pdf_channel_summary())

    # ---------- compositing ----------
    def _channel_views(self):
        return [self._channel_controls[c].to_view()
                for c in self._channel_controls
                if c in self._channel_images]

    def _rebuild_overview(self, reset_view=False):
        if not self._channel_images:
            return
        max_h = max(i.shape[0] for i in self._channel_images.values())
        max_w = max(i.shape[1] for i in self._channel_images.values())
        # align channel overviews to common size
        aligned = {}
        for ch_id, img in self._channel_images.items():
            if img.shape[:2] != (max_h, max_w):
                img = np.array(Image.fromarray(img).resize((max_w, max_h), Image.Resampling.BILINEAR))
            aligned[ch_id] = img
        rgb = render.composite(self._channel_views(), aligned)
        pix = numpy_to_qpixmap(rgb)
        # Always keep scale/coordinate metadata matched to what is on screen, so a
        # partially-loaded well never inherits the previous well's calibration.
        self.canvas.update_geometry(self._overview_ds, self._full_dims[0],
                                    self._full_dims[1], self._um_per_px_full)
        if reset_view:
            self.canvas.set_overview(pix, self._overview_ds, self._full_dims[0],
                                     self._full_dims[1], self._um_per_px_full)
        else:
            self.canvas.update_overview_pixmap(pix)

    def _on_levels_changed(self):
        self._rebuild_overview(reset_view=False)
        self._refresh_detail()
        if self._plate_series_dock is not None and self._plate_series_dock.isVisible():
            self._plate_series_builder.set_channel_summary(self._pdf_channel_summary())

    def _refresh_detail(self):
        """Load a sharper composite for the current viewport when zoomed in."""
        if self._mode == StartModeDialog.MOSAIC:
            self._refresh_mosaic_detail()
            return
        if not self._channel_images or self._full_dims == (0, 0):
            return
        mag = self.canvas.screen_px_per_full_px  # screen px per full-res px
        # The overview only holds 1 sample per `_overview_ds` full-res pixels, so it is
        # already being upscaled once mag * ds > 1 — that is when detail is needed
        # (not merely above 100%, which left Cmd+1 showing a blurry overview).
        if mag * self._overview_ds <= 1.05:
            self._detail_pending = False
            self.canvas.set_detail(None, None)
            return
        # Only one detail worker at a time; if one is busy, remember to re-run when
        # it finishes so the final viewport always gets a sharp render.
        if self._detail_worker and self._detail_worker.isRunning():
            self._detail_pending = True
            return
        rect = self.canvas.visible_full_rect()
        x0, y0, x1, y1 = rect
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        detail_ds = max(1, round(1.0 / mag))
        # bound region cost
        while ((x1 - x0) // detail_ds) * ((y1 - y0) // detail_ds) > 6_000_000:
            detail_ds += 1

        channel_paths = {ch_id: (info.path, (info.metadata.width, info.metadata.height))
                         for ch_id, info in self._channel_info.items()
                         if ch_id in self._channel_images}
        self._detail_pending = False
        self._detail_worker = DetailWorker(
            channel_paths, self._channel_views(), self._full_dims, rect, detail_ds, self._gen)
        self._detail_worker.ready.connect(self._on_detail_ready)
        self._detail_worker.finished.connect(self._on_detail_finished)
        self._detail_worker.start()

    def _on_detail_ready(self, image, rect_full, gen):
        if gen != self._gen or image is None:
            return  # stale or failed; keep the scaled overview
        self.canvas.set_detail(QPixmap.fromImage(image), rect_full)

    def _on_detail_finished(self):
        # If the viewport moved while this worker ran, render the latest view now.
        if self._detail_pending:
            self._detail_pending = False
            self._refresh_detail()

    def _refresh_mosaic_detail(self):
        if not self._mosaic_dataset or not self._mosaic_geometry or not self._mosaic_images:
            return
        magnification = self.canvas.screen_px_per_full_px
        if magnification * self._mosaic_preview_ds <= 1.05:
            self.canvas.set_detail(None, None)
            self._detail_pending = False
            return
        if self._mosaic_detail_worker is not None and self._mosaic_detail_worker.isRunning():
            self._detail_pending = True
            return
        x0, y0, x1, y1 = self.canvas.visible_full_rect()
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        detail_ds = max(1, round(1.0 / max(magnification, 1e-6)))
        while ((x1 - x0) // detail_ds) * ((y1 - y0) // detail_ds) > 7_000_000:
            detail_ds += 1
        self._detail_pending = False
        self._mosaic_detail_worker = MosaicDetailWorker(
            self._mosaic_dataset, self._mosaic_geometry,
            self.mosaic_workspace.channel_views(),
            str(self.mosaic_workspace.blend.currentData()),
            (x0, y0, x1, y1), detail_ds, self._gen, self)
        self._mosaic_detail_worker.ready.connect(self._on_mosaic_detail_ready)
        self._mosaic_detail_worker.finished.connect(self._on_mosaic_detail_finished)
        self._mosaic_detail_worker.start()

    def _on_mosaic_detail_ready(self, image, rect_full, generation):
        if generation != self._gen or image is None or self._mode != StartModeDialog.MOSAIC:
            return
        self.canvas.set_detail(numpy_to_qpixmap(image), rect_full)

    def _on_mosaic_detail_finished(self):
        worker = self.sender()
        is_current = worker is self._mosaic_detail_worker
        if is_current:
            self._mosaic_detail_worker = None
        if is_current and self._detail_pending:
            self._detail_pending = False
            self._refresh_mosaic_detail()
        if worker is not None:
            worker.deleteLater()

    # ---------- readout ----------
    def _on_cursor(self, fx, fy):
        if self._full_dims == (0, 0):
            return
        fw, fh = self._full_dims
        if not (0 <= fx < fw and 0 <= fy < fh):
            self.readout.setText("")
            return
        parts = [f"px ({int(fx)}, {int(fy)})"]
        if self._mode == StartModeDialog.MOSAIC:
            if self._um_per_px_full > 0:
                parts.append(
                    f"距離 ({fx * self._um_per_px_full / 1000:.3f}, "
                    f"{fy * self._um_per_px_full / 1000:.3f}) mm")
            ds = self._mosaic_preview_ds
            for key, image in self._mosaic_images.items():
                oy, ox = int(fy / ds), int(fx / ds)
                if 0 <= oy < image.shape[0] and 0 <= ox < image.shape[1]:
                    parts.append(f"{key}={image[oy, ox]}")
            self.readout.setText("   ".join(parts))
            return
        # absolute stage position in µm, if calibration + region known
        info0 = next(iter(self._channel_info.values()), None)
        if info0 and self._um_per_px_full > 0:
            m = info0.metadata
            stage_x_um = (m.region_x_nm + fx * m.calibration_nm) / 1000.0
            stage_y_um = (m.region_y_nm + fy * m.calibration_nm) / 1000.0
            if m.region_x_nm or m.region_y_nm:
                parts.append(f"stage ({stage_x_um:.1f}, {stage_y_um:.1f}) µm")
        # per-channel intensity from overview
        ds = self._overview_ds
        for ch_id, img in self._channel_images.items():
            oy, ox = int(fy / ds), int(fx / ds)
            if 0 <= oy < img.shape[0] and 0 <= ox < img.shape[1]:
                parts.append(f"{ch_id}={img[oy, ox]}")
        self.readout.setText("   ".join(parts))

    def _show_metadata(self, meta, ds):
        umpp = meta.um_per_pixel
        lines = [
            f"Size:      {meta.width} × {meta.height} px",
            f"Display:   1/{ds} ({meta.width // ds} × {meta.height // ds})" if ds > 1 else "Display:   full res",
            f"Pixel:     {umpp:g} µm/px" if umpp else "",
            f"FOV:       {meta.width * umpp / 1000:.2f} × {meta.height * umpp / 1000:.2f} mm" if umpp else "",
            f"Lens:      {meta.lens_name}",
            f"NA / WD:   {meta.numerical_aperture:g} / {meta.working_distance_mm:g} mm" if meta.numerical_aperture else "",
            f"Channel:   {meta.channel_comment} ({meta.channel})",
            f"Mode:      {meta.observation_mode}",
            f"Pixel fmt: {meta.pixel_mode}",
            f"Binning:   {meta.binning}",
            f"Exposure:  {meta.exposure_numerator}/{meta.exposure_denominator} s" if meta.exposure_denominator else "",
            f"Patches:   {meta.patch_count}",
        ]
        self.meta_text.setText("\n".join(l for l in lines if l))

    def _auto_contrast_all(self):
        if self._mode == StartModeDialog.MOSAIC:
            for key, row in self.mosaic_workspace.rows.items():
                if key in self._mosaic_images:
                    row.set_auto_levels(self._mosaic_images[key])
            self._rebuild_mosaic_preview()
            return
        for ch_id, ctrl in self._channel_controls.items():
            if ch_id in self._channel_images:
                ctrl.auto_contrast(self._channel_images[ch_id])
        self._rebuild_overview(reset_view=False)
        self._refresh_detail()
        if self._plate_series_dock is not None and self._plate_series_dock.isVisible():
            self._plate_series_builder.set_channel_summary(self._pdf_channel_summary())

    # ---------- wide-area mosaic ----------
    def _build_mosaic(self, reference_channel: str, blend_mode: str):
        if self._busy:
            self.statusBar().showMessage("処理中です — 完了までお待ちください。")
            return
        if not self._mosaic_dataset:
            self.statusBar().showMessage("通常画像セットを開いてください。")
            return
        self._mosaic_build_worker = MosaicBuildWorker(
            self._mosaic_dataset, reference_channel or None, blend_mode,
            preview_side=4096, parent=self)
        self._mosaic_build_worker.progress.connect(self._on_mosaic_progress)
        self._mosaic_build_worker.ready.connect(self._on_mosaic_ready)
        self._mosaic_build_worker.failed.connect(self._on_mosaic_build_failed)
        self._mosaic_build_worker.cancelled.connect(
            lambda: self.statusBar().showMessage("Stitching実行を中止しました。"))
        self._mosaic_build_worker.finished.connect(self._on_mosaic_worker_finished)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self._show_cancel_operation(True)
        self._set_actions_enabled(False)
        self.mosaic_workspace.setEnabled(False)
        self.statusBar().showMessage("Stage座標と画像の継ぎ目から位置を推定中…")
        self._mosaic_build_worker.start()

    def _on_mosaic_build_failed(self, message: str):
        self.statusBar().showMessage(f"Stitchingを実行できませんでした: {message}")
        QMessageBox.critical(
            self, "Stitchingを実行できません",
            f"位置合わせまたは画像読込中に問題が発生しました。\n\n{message}")

    def _on_mosaic_ready(self, geometry, images, downsample: int):
        self._mosaic_geometry = geometry
        self._mosaic_images = dict(images)
        self._mosaic_preview_ds = max(1, int(downsample))
        self.mosaic_workspace.set_geometry(geometry, images)
        self._rebuild_mosaic_preview(reset_view=True)
        self._show_mosaic_metadata()
        self.statusBar().showMessage(
            f"Stitching完了: {geometry.output_shape[1]:,} × {geometry.output_shape[0]:,} px · "
            f"継ぎ目 {geometry.accepted_edges}/{geometry.expected_edges} を画像で確認")

    def _rebuild_mosaic_preview(self, reset_view=False):
        if self._mode != StartModeDialog.MOSAIC or not self._mosaic_images:
            return
        rgb = render.composite(
            self.mosaic_workspace.channel_views(), self._mosaic_images)
        pixmap = numpy_to_qpixmap(rgb)
        self.canvas.set_detail(None, None)
        geometry = self._mosaic_geometry
        pixel_x = self._mosaic_dataset.pixel_size_um_yx[1]
        if geometry is None or pixel_x is None:
            return
        full_h, full_w = geometry.output_shape
        self._full_dims = (full_w, full_h)
        self._overview_ds = self._mosaic_preview_ds
        self._um_per_px_full = float(pixel_x)
        self.canvas.update_geometry(self._overview_ds, full_w, full_h, float(pixel_x))
        if reset_view or self.canvas._pixmap is None:
            self.canvas.set_overview(
                pixmap, self._overview_ds, full_w, full_h, float(pixel_x))
        else:
            self.canvas.update_overview_pixmap(pixmap)
            self._refresh_mosaic_detail()

    def _show_mosaic_metadata(self):
        dataset, geometry = self._mosaic_dataset, self._mosaic_geometry
        if not dataset or not geometry:
            return
        py, px = dataset.pixel_size_um_yx
        fov_x = geometry.output_shape[1] * float(px) / 1000.0
        fov_y = geometry.output_shape[0] * float(py) / 1000.0
        residual = (f"{geometry.residual_p95:.3f} px"
                    if geometry.residual_p95 is not None else "N/A")
        self.meta_text.setText("\n".join((
            f"Image set:  {dataset.name}",
            f"Grid:       {dataset.grid_shape[0]} × {dataset.grid_shape[1]}",
            f"Fields:     {len(dataset.tiles)}",
            f"Channels:   {', '.join(c.label for c in dataset.channels)}",
            f"Mosaic:     {geometry.output_shape[1]:,} × {geometry.output_shape[0]:,} px",
            f"Physical:   {fov_x:.2f} × {fov_y:.2f} mm",
            f"Pixel:      {px:g} × {py:g} µm/px",
            f"Reference:  {geometry.reference_channel}",
            f"Edges:      {geometry.accepted_edges}/{geometry.expected_edges}",
            f"Residual:   p95 {residual}",
            f"GCI:        {dataset.gci_path}",
        )))

    def _export_mosaic(self, kind: str):
        if self._busy:
            self.statusBar().showMessage("処理中です — 完了までお待ちください。")
            return
        if not self._mosaic_dataset or not self._mosaic_geometry:
            self.statusBar().showMessage("先にStitchingを実行してください。")
            return
        extensions = {"ome": ".ome.tif", "png": ".png", "tiff": ".tif", "pdf": ".pdf"}
        filters = {
            "ome": "OME-TIFF (*.ome.tif *.ome.tiff)", "png": "PNG (*.png)",
            "tiff": "TIFF (*.tif *.tiff)", "pdf": "PDF (*.pdf)",
        }
        base = _safe_base_name(self._mosaic_dataset.name) or "mosaic"
        suffix = extensions[kind]
        default = str(Path(self._last_export_dir()) / f"{base}{suffix}")

        def derived(path):
            if kind != "ome":
                return [path]
            return [path, Path(str(path) + ".mosaic-qc.json"),
                    path.with_name(path.stem + "_alignment.csv")]

        path = _ask_save_path(
            self, "Stitching画像を書き出す", default, filters[kind], derived=derived)
        if path is None:
            return
        problem = _writable_problem(path.parent)
        if problem:
            QMessageBox.critical(self, "この場所には保存できません", problem)
            return
        self._remember_export_dir(path.parent)
        self._mosaic_export_worker = MosaicExportWorker(
            self._mosaic_dataset, self._mosaic_geometry, path, kind,
            self.mosaic_workspace.channel_views(),
            str(self.mosaic_workspace.blend.currentData()),
            self.mosaic_workspace.scale_bar,
            self.mosaic_workspace.presentation_max_side,
            self.mosaic_workspace.effective_presentation_dpi(), self)
        self._mosaic_export_worker.progress.connect(self._on_mosaic_progress)
        self._mosaic_export_worker.done.connect(self._on_mosaic_export_done)
        self._mosaic_export_worker.failed.connect(self._on_mosaic_export_failed)
        self._mosaic_export_worker.cancelled.connect(
            lambda: self.statusBar().showMessage("書き出しを中止しました。"))
        self._mosaic_export_worker.finished.connect(self._on_mosaic_worker_finished)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self._show_cancel_operation(True)
        self._set_actions_enabled(False)
        self.mosaic_workspace.setEnabled(False)
        self.statusBar().showMessage("Stitching画像を書き出し中…")
        self._mosaic_export_worker.start()

    def _on_mosaic_export_done(self, path: str):
        self.statusBar().showMessage(f"保存しました: {path}")

    def _on_mosaic_export_failed(self, message: str):
        self.statusBar().showMessage(f"書き出しに失敗しました: {message}")
        QMessageBox.critical(
            self, "Stitching画像を書き出せません", _friendly_error(Exception(message)))

    def _show_cancel_operation(self, visible: bool):
        button = getattr(self, "btn_cancel_operation", None)
        if button is None:
            return
        button.setText("処理を中止")
        button.setEnabled(bool(visible))
        button.setVisible(bool(visible))

    def _cancel_active_operation(self):
        candidates = (
            self._mosaic_export_worker, self._mosaic_build_worker,
            self._mosaic_metadata_worker, self._scan_worker, self._stitch_worker)
        worker = next((item for item in candidates
                       if item is not None and item.isRunning()), None)
        if worker is None or not hasattr(worker, "cancel"):
            return
        worker.cancel()
        self.btn_cancel_operation.setText("中止中…")
        self.btn_cancel_operation.setEnabled(False)
        self.statusBar().showMessage("安全な区切りで処理を中止しています…")

    # ---------- export ----------
    def _export(self, fmt):
        if self._busy:
            self.statusBar().showMessage("Another operation is running — please wait.")
            return
        if fmt == "tiff":
            # _current_well must still belong to the *current* experiment: switching
            # experiments used to leave a stale name here and crash with KeyError.
            wells = self._experiment["wells"] if self._experiment else {}
            if not self._current_well or self._current_well not in wells:
                self.statusBar().showMessage("Select a well first.")
                return
            channels = list(wells[self._current_well])
            # One file per channel is written, not the name typed into the dialog —
            # so the overwrite check has to be made against those names.
            path = _ask_save_path(
                self, "TIFF を名前を付けて保存", f"{self._current_well}.tif", "TIFF (*.tif)",
                derived=lambda p: [p.with_name(f"{p.stem}_{c}.tif") for c in channels])
            if path is None:
                return
            self.statusBar().showMessage("Exporting full-resolution TIFF per channel...")
            self._exporting = True
            QApplication.processEvents()
            try:
                for ch_id, info in wells[self._current_well].items():
                    img = ktf_reader.reconstruct_image(info.path, downsample=1)
                    out = path.with_name(f"{path.stem}_{ch_id}.tif")
                    Image.fromarray(img).save(str(out))
                self.statusBar().showMessage(f"Exported full-res channels to {path.parent}")
            except Exception as e:
                self.statusBar().showMessage(f"TIFF export failed: {e}")
            finally:
                self._exporting = False
        else:
            if not hasattr(self, "_channel_images") or not self._channel_images:
                return
            path = _ask_save_path(self, "PNG を名前を付けて保存",
                                  f"{self._current_well}.png", "PNG (*.png)")
            if path is None:
                return
            path = str(path)
            max_h = max(i.shape[0] for i in self._channel_images.values())
            max_w = max(i.shape[1] for i in self._channel_images.values())
            aligned = {c: (i if i.shape[:2] == (max_h, max_w)
                           else np.array(Image.fromarray(i).resize((max_w, max_h))))
                       for c, i in self._channel_images.items()}
            rgb = render.composite(self._channel_views(), aligned)
            Image.fromarray(rgb).save(path)
            self.statusBar().showMessage(f"Exported view to {path}")

    def _export_all_wells(self):
        if not self._experiment or self._busy:
            return
        wells = self._experiment["wells"]
        dlg = OutputTargetDialog(
            self, "すべてのウェルを名前を付けて保存",
            self._experiment["name"], "{base}_A01_CH1.tif", self._last_export_dir())
        dlg.setStyleSheet("")
        if not dlg.exec():
            return
        out, base = dlg.folder, dlg.base
        self._remember_export_dir(dlg.root)
        if not _confirm_overwrite(self, [out / f"{base}_{wid}_{ch}.tif"
                                         for wid, chans in wells.items() for ch in chans]):
            return
        self._exporting = True
        total = sum(len(c) for c in wells.values())
        done = 0
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        failed = 0
        try:
            for wid, channels in wells.items():
                for ch_id, info in channels.items():
                    self.statusBar().showMessage(f"Exporting {wid}_{ch_id} ({done+1}/{total})...")
                    QApplication.processEvents()
                    try:
                        img = ktf_reader.reconstruct_image(info.path, downsample=1)
                        Image.fromarray(img).save(str(out / f"{base}_{wid}_{ch_id}.tif"))
                    except Exception as e:
                        failed += 1
                        print(f"Export error {wid}/{ch_id}: {e}")
                    done += 1
                    self.progress_bar.setValue(done)
        finally:
            self._exporting = False
            self.progress_bar.hide()
        msg = f"Exported {done - failed}/{total} images to {out}"
        self.statusBar().showMessage(msg + (f" — {failed} failed" if failed else ""))

    # ---------- busy state ----------
    @property
    def _busy(self) -> bool:
        """True while any long operation owns the data (export or stitch)."""
        return self._loading_experiment or self._exporting or (
            self._stitch_worker is not None and self._stitch_worker.isRunning()) or any(
                worker is not None and worker.isRunning() for worker in (
                    self._scan_worker, self._mosaic_metadata_worker,
                    self._mosaic_build_worker, self._mosaic_export_worker))

    def closeEvent(self, event):
        """Don't let a half-written mosaic be left behind on quit."""
        if self._plate_series_builder is not None:
            self._plate_series_builder.flush_edits()
        if self._exporting:
            QMessageBox.information(
                self, "書き出し中です",
                "PDFまたは画像の書き出しが完了するまでお待ちください。")
            event.ignore()
            return
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.cancel()
            self._scan_worker.wait(15000)
            if self._scan_worker.isRunning():
                event.ignore()
                return
        if self._stitch_worker is not None and self._stitch_worker.isRunning():
            ans = QMessageBox.question(
                self, "Stitching in progress",
                "Stitching is still running. Stop it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._stitch_worker.cancel()
            self._stitch_worker.wait(15000)
            if self._stitch_worker.isRunning():
                QMessageBox.information(
                    self, "終了を待っています",
                    "現在のタイル処理が安全に停止するまで、もう少しお待ちください。")
                event.ignore()
                return
        mosaic_workers = [worker for worker in (
            self._mosaic_metadata_worker, self._mosaic_build_worker,
            self._mosaic_export_worker) if worker is not None and worker.isRunning()]
        if mosaic_workers:
            ans = QMessageBox.question(
                self, "Stitching処理中です",
                "通常画像セットの処理が続いています。中止して終了しますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            for worker in mosaic_workers:
                if hasattr(worker, "cancel"):
                    worker.cancel()
                worker.wait(15000)
            if any(worker.isRunning() for worker in mosaic_workers):
                QMessageBox.information(
                    self, "終了を待っています",
                    "安全に停止するまで、もう少しお待ちください。")
                event.ignore()
                return
        if self._mosaic_detail_worker is not None and self._mosaic_detail_worker.isRunning():
            self._mosaic_detail_worker.wait(15000)
            if self._mosaic_detail_worker.isRunning():
                event.ignore()
                return
        passive_workers = [worker for worker in (
            *self._workers, self._detail_worker, self._update_checker)
            if worker is not None and worker.isRunning()]
        for worker in passive_workers:
            worker.wait(15000)
        if any(worker.isRunning() for worker in passive_workers):
            QMessageBox.information(
                self, "読込の終了を待っています",
                "画像または更新情報の読込が完了してから、もう一度終了してください。")
            event.ignore()
            return
        event.accept()

    # ---------- stitching raw tiles ----------
    def _stitch_raw_tiles(self):
        """Stitch the loaded RAW experiment. Never touches the .ktf model."""
        if self._busy:
            self.statusBar().showMessage("処理中です — 完了までお待ちください。")
            return
        if not self._raw_experiment or not self._raw_experiment.get("wells"):
            self.statusBar().showMessage(
                "プレート画像セットを開いてください（File ▸ ワークフローを選ぶ…）")
            return
        wells = self._raw_experiment["wells"]

        dlg = StitchDialog(wells, self, current_well=self._current_raw_well)
        dlg.setStyleSheet("")
        if not dlg.exec():
            return
        if not dlg.all_wells:
            wid = self._current_raw_well
            if not wid or wid not in wells:
                self.statusBar().showMessage("ウェルを選んでください。")
                return
            wells = {wid: wells[wid]}

        default_base = self._raw_experiment["name"]
        if not dlg.all_wells:
            default_base = f"{default_base}_{next(iter(wells))}"
        tgt = OutputTargetDialog(
            self, "貼り合わせた画像を名前を付けて保存", default_base,
            StitchWorker.SAMPLE_NAME.get(dlg.fmt, "{base}_A01.tif"),
            self._last_export_dir())
        tgt.setStyleSheet("")
        if not tgt.exec():
            return
        out, base = tgt.folder, tgt.base
        self._remember_export_dir(tgt.root)
        if not _confirm_overwrite(
                self, StitchWorker.planned_outputs(out, base, sorted(wells), dlg.fmt)):
            return

        self._set_actions_enabled(False)
        self.btn_stitch_raw.setEnabled(False)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self._show_cancel_operation(True)
        cond = self.conditions.to_dict()
        headers = cond.get("__headers__") or list(WellConditionsTable.DEFAULT_HEADERS)
        self._stitch_worker = StitchWorker(
            wells, out, dlg.z_mode, dlg.fmt,
            flatfield=dlg.flatfield, subpixel=dlg.subpixel,
            exp_name=self._raw_experiment["name"], base=base,
            conditions=cond, cond_headers=headers)
        self._stitch_worker.progress.connect(self._on_stitch_progress)
        self._stitch_worker.done.connect(self._on_stitch_done)
        self._stitch_worker.start()

    def _on_stitch_progress(self, msg, frac):
        if msg:
            self.statusBar().showMessage(msg)
        if frac >= 0:                 # message-only updates carry -1
            self.progress_bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))

    def _on_stitch_done(self, ok, failed, out_dir):
        self.progress_bar.hide()
        self._show_cancel_operation(False)
        self._set_actions_enabled(True)
        if hasattr(self, "btn_stitch_raw"):
            self.btn_stitch_raw.setEnabled(True)
        warn = getattr(self._stitch_worker, "warnings", []) or []
        msg = f"Stitched {ok} well(s) into {out_dir}"
        if failed:
            msg += f" — {failed} failed"
        if warn:
            msg += f" — {len(warn)} warning(s)"
        self.statusBar().showMessage(msg)
        box = QMessageBox(self)
        box.setStyleSheet("")
        box.setWindowTitle("スティッチング完了" if ok else "スティッチング失敗")
        box.setIcon(QMessageBox.Icon.Information if ok else QMessageBox.Icon.Critical)
        body = (f"{ok} ウェルを書き出しました。\n出力先: {out_dir}"
                if ok else f"書き出せたウェルがありません。\n出力先: {out_dir}")
        if failed:
            body += f"\n失敗: {failed} ウェル"
        if warn:
            body += "\n\n" + "\n".join(warn[:12]) + ("\n…" if len(warn) > 12 else "")
            if any("uncertain" in x or "inherited" in x for x in warn):
                body += ("\n\n「alignment uncertain」は重なりが小さすぎるか特徴が乏しく"
                         "測定できなかったことを示します — 該当ウェルを確認してください。")
            if any("保存できません" in x or "書き込めません" in x for x in warn):
                body += ("\n\n保存先に書き込めませんでした。別の場所（内蔵ディスクなど）"
                         "を選び直してください。")
        box.setText(body)
        open_btn = box.addButton("出力フォルダを開く", QMessageBox.ButtonRole.ActionRole)
        box.addButton("閉じる", QMessageBox.ButtonRole.AcceptRole)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))

    # ---------- plate contact-sheet PDF ----------
    @staticmethod
    def _load_font(size):
        # Try bare names first (PIL searches system font dirs on each OS), then
        # explicit paths for macOS / Windows / Linux.
        candidates = (
            "Arial.ttf", "arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        for p in candidates:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
        try:
            return ImageFont.load_default(size)
        except Exception:
            return ImageFont.load_default()

    @staticmethod
    def _auto_lohi(arr):
        nz = arr[arr > 0]
        if nz.size == 0:
            return 0, 255
        return int(np.percentile(nz, 1)), max(int(np.percentile(nz, 99.5)), 1)

    def _levels_for(self, ch_id, arr, settings):
        """Window/level to use when exporting a well.

        Prefer the user's on-screen min/max so the PDF matches what they set (and so
        wells stay comparable); fall back to a per-well auto-stretch only when the
        channel has no control (e.g. a well the user never opened).
        """
        s = settings.get(ch_id)
        if s is not None and (s.lo, s.hi) != (0, 255):
            return s.lo, s.hi
        return self._auto_lohi(arr)

    def _render_well_thumb(self, channels, settings, panel_px):
        """Composite a well from its embedded per-channel JPEG thumbnails (fast; no full read)."""
        thumbs = {}
        for ch_id, info in channels.items():
            if not info.thumbnail_jpeg:
                raise OSError(
                    f"{info.path.name} ({ch_id}) に埋め込みサムネイルがありません。"
                    "Draft以外の画質をお試しください")
            try:
                im = Image.open(io.BytesIO(info.thumbnail_jpeg)).convert("L")
                thumbs[ch_id] = np.array(im)
            except Exception as e:
                raise OSError(
                    f"{info.path.name} ({ch_id}) のサムネイルを読み込めません") from e
        if not thumbs:
            raise OSError("このwellのサムネイルを読み込めません")
        max_h = max(a.shape[0] for a in thumbs.values())
        max_w = max(a.shape[1] for a in thumbs.values())
        views, images = [], {}
        for ch_id, arr in thumbs.items():
            if arr.shape != (max_h, max_w):
                arr = np.array(Image.fromarray(arr).resize((max_w, max_h), Image.Resampling.BILINEAR))
            lo, hi = self._levels_for(ch_id, arr, settings)
            s = settings.get(ch_id)
            views.append(render.ChannelView(
                ch_id=ch_id,
                color=s.color if s else CHANNEL_COLORS.get(ch_id, (200, 200, 200)),
                lo=lo, hi=hi,
                gamma=s.gamma if s else 1.0,
                enabled=s.enabled if s else True,
                solo=s.solo if s else False,
            ))
            images[ch_id] = arr
        rgb = render.composite(views, images)
        im = Image.fromarray(rgb)
        im.thumbnail((panel_px, panel_px), Image.Resampling.LANCZOS)
        return im

    def _render_well_full(self, channels, settings, panel_px):
        """Composite a well from full-resolution tiles (sharp; reads the .ktf files).

        panel_px <= 0 means render at the file's original resolution (no downscaling).
        """
        dims = [(info.metadata.width, info.metadata.height) for info in channels.values()]
        max_w = max(w for w, h in dims)
        max_h = max(h for w, h in dims)
        if panel_px is None or panel_px <= 0 or panel_px >= max(max_w, max_h):
            scale = 1.0  # native resolution
        else:
            scale = panel_px / max(max_w, max_h)
        ds = max(1, round(1 / scale)) if scale < 1 else 1
        tw, th = max(1, round(max_w * scale)), max(1, round(max_h * scale))
        views, images = [], {}
        for ch_id, info in channels.items():
            try:
                img = ktf_reader.reconstruct_image(info.path, downsample=ds)
            except Exception as e:
                raise OSError(
                    f"{info.path.name} ({ch_id}) の画像を再構成できません") from e
            if img.shape != (th, tw):
                img = np.array(Image.fromarray(img).resize((tw, th), Image.Resampling.LANCZOS))
            lo, hi = self._levels_for(ch_id, img, settings)
            s = settings.get(ch_id)
            views.append(render.ChannelView(
                ch_id=ch_id,
                color=s.color if s else CHANNEL_COLORS.get(ch_id, (200, 200, 200)),
                lo=lo, hi=hi,
                gamma=s.gamma if s else 1.0,
                enabled=s.enabled if s else True,
                solo=s.solo if s else False,
            ))
            images[ch_id] = img
        if not images:
            raise OSError("このwellの画像を再構成できません")
        return Image.fromarray(render.composite(views, images))

    # ---------- per-well caption from the Conditions tab ----------
    def _conditions_snapshot(self):
        """Current Conditions table as ({well: [values]}, headers)."""
        data = self.conditions.to_dict()
        headers = data.get("__headers__") or list(WellConditionsTable.DEFAULT_HEADERS)
        return data, headers

    def _conditions_snapshot_for(self, path):
        """Conditions for one experiment without changing the visible table."""
        current = self._experiment["path"] if self._experiment else None
        if current is not None and _path_key(current) == _path_key(path):
            return self._conditions_snapshot()
        data = self._load_conditions_for_path(path)
        headers = data.get("__headers__") or list(WellConditionsTable.DEFAULT_HEADERS)
        return data, headers

    @staticmethod
    def _caption_lines(wid, cond, headers):
        """"Header: value" for every non-empty condition cell of this well."""
        lines = []
        for i, val in enumerate(cond.get(wid, [])):
            val = (val or "").strip()
            if not val:
                continue
            head = headers[i + 1] if i + 1 < len(headers) else ""
            lines.append(f"{head}: {val}" if head else val)
        return lines

    def _draw_well_caption(self, canvas, draw, x, y, wid, lines, f_id, f_txt, inset, max_w):
        """Well ID + its sample conditions on a translucent plate, top-left of the image."""
        gap = max(2, inset // 3)
        idb = draw.textbbox((0, 0), wid, font=f_id)
        w_max, h_total = idb[2] - idb[0], idb[3] - idb[1]
        measured = []
        for ln in lines:
            b = draw.textbbox((0, 0), ln, font=f_txt)
            measured.append((ln, b[3] - b[1]))
            w_max = max(w_max, b[2] - b[0])
            h_total += gap + (b[3] - b[1])
        bw = min(max_w, w_max + 2 * inset)
        bh = h_total + 2 * inset
        # RGBA pasted with itself as mask alpha-blends onto the RGB sheet
        plate = Image.new("RGBA", (max(1, bw), max(1, bh)), (0, 0, 0, 150))
        canvas.paste(plate, (x, y), plate)
        ty = y + inset
        draw.text((x + inset, ty), wid, fill=(255, 235, 60), font=f_id)
        ty += (idb[3] - idb[1]) + gap
        for ln, lh in measured:
            draw.text((x + inset, ty), ln, fill=(240, 240, 240), font=f_txt)
            ty += lh + gap

    @staticmethod
    def _well_aspect(wells):
        """height/width of a typical well image (cells are shaped to match)."""
        for chs in wells.values():
            for info in chs.values():
                w, h = info.metadata.width, info.metadata.height
                if w and h:
                    return h / w
        return 1.0

    @staticmethod
    def _max_well_dim(wells):
        best = 0
        for chs in wells.values():
            for info in chs.values():
                best = max(best, info.metadata.width, info.metadata.height)
        return best or 1600

    # label: (source, panel_px, dpi, layout)   panel_px=0 → original resolution
    PDF_QUALITY = {
        "Draft — fast (embedded thumbnails)": ("thumb", 380, 150, "grid"),
        "Standard — ~700 px/well": ("full", 700, 200, "grid"),
        "High — ~1100 px/well": ("full", 1100, 300, "grid"),
        "Ultra — ~1600 px/well": ("full", 1600, 300, "grid"),
        "Maximum — original resolution, all wells on one sheet": ("full", 0, 300, "grid"),
        "Maximum — original resolution, one well per page": ("full", 0, 300, "pages"),
        "Maximum — overview sheet + one well per page": ("full", 0, 300, "both"),
    }

    def _pdf_default_name(self, paths, first_label=""):
        if len(paths) == 1:
            base = _safe_base_name(first_label or paths[0].name)
            return f"{base}_plate.pdf"
        base = _safe_base_name(first_label or paths[0].name) or "plate"
        return f"{base}_plate_series.pdf"

    def _plate_sheet_layout(self, wells, panel):
        rows = sorted(set(w[0] for w in wells))
        cols = sorted(set(w[1:] for w in wells),
                      key=lambda c: int(c) if c.isdigit() else 0)
        native = panel == 0
        if native:
            panel = self._max_well_dim(wells)
        cell_w = panel
        cell_h = max(1, int(round(panel * self._well_aspect(wells))))
        k = panel / 380.0
        pad, hdr, title_h = int(12 * k), int(30 * k), int(48 * k)
        grid_w = hdr + len(cols) * (cell_w + pad) + pad
        grid_h = title_h + hdr + len(rows) * (cell_h + pad) + pad
        return {
            "rows": rows, "cols": cols, "native": native, "panel": panel,
            "cell_w": cell_w, "cell_h": cell_h, "k": k,
            "pad": pad, "hdr": hdr, "title_h": title_h,
            "grid_w": grid_w, "grid_h": grid_h,
            "est_bytes": grid_w * grid_h * 3,
        }

    def _confirm_large_pdf_sheets(self, series, panel, with_pages):
        large = []
        for exp, _conditions in series:
            geom = self._plate_sheet_layout(exp["wells"], panel)
            if geom["est_bytes"] > 1_000_000_000:
                large.append((exp["name"], geom))
        if not large:
            return True
        shown = "\n".join(
            f"  • {name}: {g['grid_w']} × {g['grid_h']} px (~{g['est_bytes'] / 1e9:.1f} GB)"
            for name, g in large[:8])
        if len(large) > 8:
            shown += f"\n  …ほか {len(large) - 8} 撮影"
        note = ("\n\n各撮影のウェル別ページも、この後1ページずつ追加されます。"
                if with_pages else
                "\n\n「one well per page」は大幅に軽量です。")
        ans = QMessageBox.question(
            self, "非常に大きなPDFになります",
            f"大きなコンタクトシートが {len(large)} 枚あります。\n{shown}"
            f"{note}\n\n続けますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return ans == QMessageBox.StandardButton.Yes

    def _iter_plate_series_pages(self, series, settings, source, panel, layout):
        for exp, conditions in series:
            if layout != "pages":
                sheet = self._build_plate_sheet(
                    exp["wells"], settings, source, panel, exp["name"],
                    with_pages=(layout == "both"), conditions=conditions,
                    confirm_large=False)
                if sheet is None:
                    raise RuntimeError(f"{exp['name']} のコンタクトシートを作成できませんでした")
                yield sheet
            if layout in ("pages", "both"):
                yield from self._iter_well_pages(
                    exp["wells"], settings, exp["name"], conditions=conditions)

    @staticmethod
    def _write_pdf_pages(path, pages, dpi, title):
        """Stream raster pages to an atomically committed PDF."""
        iterator = iter(pages)
        page = next(iterator)
        device = QSaveFile(str(path))
        writer = None
        painter = None
        started = False
        committed = False
        try:
            if not device.open(QIODevice.OpenModeFlag.WriteOnly):
                raise OSError(device.errorString() or "PDFの保存先を開けません")
            writer = QPdfWriter(device)
            writer.setResolution(int(dpi))
            writer.setTitle(title)
            writer.setCreator(f"{APP_NAME} {__version__}")
            writer.setPageMargins(
                QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter)
            painter = QPainter()

            def paint_and_release(image):
                """Keep full-page Qt/numpy buffers scoped to exactly one page."""
                nonlocal started
                rgb = image if image.mode == "RGB" else image.convert("RGB")
                try:
                    size = QPageSize(
                        QSizeF(rgb.width * 25.4 / dpi, rgb.height * 25.4 / dpi),
                        QPageSize.Unit.Millimeter)
                    if not writer.setPageSize(size):
                        raise OSError("PDFのページサイズを設定できません")
                    if not started:
                        if not painter.begin(writer):
                            raise OSError("PDF writerを開始できません")
                        started = True
                    elif not writer.newPage():
                        raise OSError("PDFに次のページを追加できません")
                    arr = np.ascontiguousarray(np.asarray(rgb))
                    qimg = QImage(
                        arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 3,
                        QImage.Format.Format_RGB888).copy()
                    painter.drawImage(
                        QRect(0, 0, writer.width(), writer.height()), qimg)
                    del qimg, arr
                finally:
                    if rgb is not image:
                        rgb.close()
                    image.close()

            while True:
                paint_and_release(page)
                page = None
                try:
                    page = next(iterator)
                except StopIteration:
                    break

            if not painter.end():
                raise OSError("PDF writerを終了できません")
            painter = None
            writer = None
            if not device.commit():
                raise OSError(device.errorString() or "PDFを保存できません")
            committed = True
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass
            if painter is not None and painter.isActive():
                painter.end()
            painter = None
            writer = None
            if not committed and device.isOpen():
                device.cancelWriting()
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def _export_plate_pdf(self):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self.statusBar().showMessage("フォルダのスキャン完了後にPDFを書き出してください。")
            return
        if self._busy:
            self.statusBar().showMessage("処理中です — 完了までお待ちください。")
            return
        if not self._experiment or not self._experiment["wells"]:
            QMessageBox.information(
                self, "撮影を開いてください",
                "Series PDF Builderを使う前に、Experimentsから撮影を1件開いてください。")
            return

        current_path = Path(self._experiment["path"])
        if self._plate_series_builder.is_empty:
            self._plate_series_builder.reset(current_path)
        self._plate_series_builder.set_active_path(current_path)
        self._plate_series_builder.set_channel_summary(self._pdf_channel_summary())
        was_hidden = not self._plate_series_dock.isVisible()
        self._plate_series_dock.show()
        self._plate_series_dock.raise_()
        if was_hidden:
            # Show the label editor without sacrificing all plate-detail space.
            # At the minimum window height the builder's own scroll area remains
            # the fallback; normal desktop windows open the complete editor.
            requested = min(540, max(430, int(self.height() * 0.52)))
            self.resizeDocks(
                [self._plate_series_dock], [requested], Qt.Orientation.Vertical)
        self.statusBar().showMessage(
            "Series PDF Builderで撮影を追加・並べ替えできます。行をクリックすると詳細を表示します。")

    def _activate_series_capture(self, path):
        path = Path(path)
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self.statusBar().showMessage("フォルダのスキャン完了後に撮影を切り替えてください。")
            return False
        if self._busy:
            self.statusBar().showMessage("処理中は撮影を切り替えられません。")
            return False
        mode_changed = self._mode != StartModeDialog.KTF
        if mode_changed:
            self._set_mode(StartModeDialog.KTF)
        current = Path(self._experiment["path"]) if self._experiment else None
        if mode_changed or current is None or _path_key(current) != _path_key(path):
            if not self._load_experiment(path):
                return False
        self._select_tree_item(path)
        self.well_tabs.setCurrentIndex(0)
        self._plate_series_builder.set_active_path(path)
        self._plate_series_builder.set_channel_summary(self._pdf_channel_summary())
        return True

    def _edit_series_conditions(self, path):
        if not self._activate_series_capture(path):
            return
        self.well_tabs.setCurrentWidget(self.cond_tab)
        self.statusBar().showMessage(
            "Conditionsを編集しています。変更はこの撮影フォルダに自動保存されます。")

    def _pdf_channel_summary(self):
        views = [ctrl.to_view() for ctrl in self._channel_controls.values()]
        if not views:
            return "全チャンネルを既定色・wellごとの自動レベルで描画"
        enabled = [view.ch_id for view in views if view.enabled]
        shown = ", ".join(enabled) if enabled else "なし"
        return f"現在の表示設定を全撮影に適用（表示: {shown}）"

    def _export_plate_pdf_from_builder(self, selected, quality_spec):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self.statusBar().showMessage("フォルダのスキャン完了後にPDFを書き出してください。")
            return
        if self._busy:
            self.statusBar().showMessage("処理中です — 完了までお待ちください。")
            return
        selected = [(Path(path), str(label)) for path, label in tuple(selected)]
        quality_spec = tuple(quality_spec)
        self._save_conditions()
        current_path = Path(self._experiment["path"]) if self._experiment else None
        current_exp = dict(self._experiment) if self._experiment else None
        current_conditions = self._conditions_snapshot() if self._experiment else None
        settings = {c: ctrl.to_view() for c, ctrl in self._channel_controls.items()}
        self._exporting = True
        self._plate_series_builder.set_exporting(True)
        self._update_series_export_state()
        try:
            self._export_plate_pdf_selection(
                selected, quality_spec, current_path, current_exp,
                current_conditions, settings)
        finally:
            self._exporting = False
            self._plate_series_builder.set_exporting(False)
            self._plate_series_builder.set_channel_summary(self._pdf_channel_summary())
            self._update_series_export_state()
            self.progress_bar.hide()

    def _export_plate_pdf_selection(
            self, selected, choice, current_path, current_exp, current_conditions, settings):
        source, panel, dpi, layout = (
            self.PDF_QUALITY[choice] if isinstance(choice, str) else tuple(choice))

        series, issues = [], []
        self.statusBar().showMessage(f"{len(selected)} 撮影を確認中…")
        for i, (path, label) in enumerate(selected):
            QApplication.processEvents()
            try:
                if (current_path is not None and current_exp is not None
                        and _path_key(path) == _path_key(current_path)):
                    exp = dict(current_exp)
                    conditions = current_conditions
                else:
                    exp = ktf_reader.scan_experiment_folder(path)
                    conditions = self._conditions_snapshot_for(path)
                if not exp["wells"]:
                    raise ValueError("読み取れるウェルがありません")
                exp["name"] = label
                series.append((exp, conditions))
                for filename, error in exp.get("errors") or []:
                    issues.append(f"{label}: {filename} を読み込めません ({error})")
            except Exception as e:
                issues.append(f"{label}: 撮影を含められません ({e})")
            self.statusBar().showMessage(f"撮影を確認中 ({i + 1}/{len(selected)}): {label}")

        if not series:
            QMessageBox.warning(
                self, "書き出せる撮影がありません", "\n".join(issues[:12]))
            return

        ref_structure = {
            well: tuple(sorted(channels))
            for well, channels in series[0][0]["wells"].items()}
        ref_wells = set(ref_structure)
        ref_channels = {
            channel for channels in series[0][0]["wells"].values() for channel in channels}
        for exp, _conditions in series[1:]:
            structure = {
                well: tuple(sorted(per_well))
                for well, per_well in exp["wells"].items()}
            wells = set(structure)
            channels = {
                channel for per_well in exp["wells"].values() for channel in per_well}
            if structure != ref_structure:
                issues.append(
                    f"{exp['name']}: 先頭撮影と構成が異なります "
                    f"(wells {len(wells)}/{len(ref_wells)}, "
                    f"channels {', '.join(sorted(channels)) or '-'} / "
                    f"{', '.join(sorted(ref_channels)) or '-'})")

        if issues:
            shown = "\n".join(f"  • {line}" for line in issues[:10])
            if len(issues) > 10:
                shown += f"\n  …ほか {len(issues) - 10} 件"
            ans = QMessageBox.question(
                self, "撮影データを確認してください",
                f"読取失敗、またはwell/channel構成の差があります。\n{shown}\n\n"
                f"読めた {len(series)} 撮影で続けますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return

        selected_paths = [Path(exp["path"]) for exp, _conditions in series]
        first_label = series[0][0]["name"]
        default_name = self._pdf_default_name(selected_paths, first_label)
        path = _ask_save_path(self, "プレート PDF を名前を付けて保存",
                              default_name, "PDF (*.pdf)")
        if path is None:
            return
        problem = _writable_problem(path.parent)
        if problem:
            QMessageBox.critical(self, "この場所には保存できません", problem)
            return
        if layout != "pages" and not self._confirm_large_pdf_sheets(
                series, panel, with_pages=(layout == "both")):
            self.statusBar().showMessage("PDF書き出しをキャンセルしました。")
            return

        try:
            pages = self._iter_plate_series_pages(
                series, settings, source, panel, layout)
            self._write_pdf_pages(path, pages, dpi, path.stem)
            self._remember_export_dir(path.parent)
            self.statusBar().showMessage(
                f"{len(series)} 撮影を1つのPDFに書き出しました: {path}")
        except StopIteration:
            self.statusBar().showMessage("書き出せるページがありません。")
        except MemoryError:
            self.statusBar().showMessage(
                "メモリ不足です。「one well per page」または低い画質をお試しください。")
        except Exception as e:
            msg = _friendly_error(e)
            self.statusBar().showMessage(f"PDF書き出しに失敗しました: {msg}")
            QMessageBox.warning(self, "PDF書き出しに失敗しました", msg)

    def _build_plate_sheet(self, wells, settings, source, panel, exp_name, with_pages=False,
                           conditions=None, confirm_large=True):
        """Render every well into one contact sheet. Returns the image, or None."""
        cond, headers = conditions if conditions is not None else self._conditions_snapshot()
        geom = self._plate_sheet_layout(wells, panel)
        rows, cols = geom["rows"], geom["cols"]
        native, panel = geom["native"], geom["panel"]
        cell_w, cell_h, k = geom["cell_w"], geom["cell_h"], geom["k"]
        pad, hdr, title_h = geom["pad"], geom["hdr"], geom["title_h"]
        grid_w, grid_h, est_bytes = geom["grid_w"], geom["grid_h"], geom["est_bytes"]

        if confirm_large and est_bytes > 1_000_000_000:
            gb = est_bytes / 1e9
            extra_note = ("\n\nPer-well pages are added afterwards, one at a time."
                          if with_pages else
                          "\n\n(“one well per page” is much lighter.)")
            ans = QMessageBox.question(
                self, "Very large sheet",
                f"This single sheet will be {grid_w} × {grid_h} px "
                f"(~{gb:.1f} GB in memory) and may take several minutes." + extra_note +
                "\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                self.statusBar().showMessage("Export cancelled.")
                return None

        try:
            canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
        except (MemoryError, ValueError) as e:
            self.statusBar().showMessage(f"Sheet too large ({grid_w}×{grid_h}): {e}")
            return None
        draw = ImageDraw.Draw(canvas)
        f_title = self._load_font(int(26 * k))
        f_hdr = self._load_font(int(20 * k))
        f_lbl = self._load_font(int(17 * k))
        f_cond = self._load_font(int(13 * k))

        chans = sorted({c for chs in wells.values() for c in chs})
        draw.text((pad, pad), f"{exp_name}  ·  {len(wells)} wells  ·  {', '.join(chans)}",
                  fill=(0, 0, 0), font=f_title)
        for ci, col in enumerate(cols):
            x = hdr + ci * (cell_w + pad) + cell_w // 2
            draw.text((x, title_h + hdr // 2), col, fill=(0, 0, 0), font=f_hdr, anchor="mm")

        total = len(wells)
        done = 0
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        for ri, row in enumerate(rows):
            y0 = title_h + hdr + ri * (cell_h + pad)
            draw.text((hdr // 2, y0 + cell_h // 2), row, fill=(0, 0, 0), font=f_hdr, anchor="mm")
            for ci, col in enumerate(cols):
                wid = f"{row}{col}"
                if wid not in wells:
                    continue
                x0 = hdr + ci * (cell_w + pad)
                self.statusBar().showMessage(f"Rendering {wid} ({done + 1}/{total})...")
                QApplication.processEvents()
                draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=(10, 10, 10))
                if source == "thumb":
                    im = self._render_well_thumb(wells[wid], settings, panel)
                else:
                    im = self._render_well_full(wells[wid], settings, 0 if native else panel)
                if im is not None:
                    if im.width > cell_w or im.height > cell_h:
                        im.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
                    canvas.paste(im, (x0 + (cell_w - im.width) // 2,
                                      y0 + (cell_h - im.height) // 2))
                    im = None            # release before the next well is decoded
                else:
                    raise OSError(f"{exp_name} / {wid} の画像を描画できません")
                self._draw_well_caption(
                    canvas, draw, x0 + int(6 * k), y0 + int(6 * k), wid,
                    self._caption_lines(wid, cond, headers),
                    f_lbl, f_cond, max(4, int(6 * k)), cell_w - int(12 * k))
                done += 1
                self.progress_bar.setValue(done)
        return canvas

    def _iter_well_pages(self, wells, settings, exp_name, order=None, conditions=None):
        """Yield one original-resolution page per well (lazy: one page in memory)."""
        order = order if order is not None else sorted(wells)
        total = len(order)
        f_lbl = self._load_font(40)
        f_id = self._load_font(46)
        f_cond = self._load_font(34)
        cond, headers = conditions if conditions is not None else self._conditions_snapshot()
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        for i, wid in enumerate(order):
            self.statusBar().showMessage(
                f"Rendering {wid} at full resolution ({i + 1}/{total})...")
            QApplication.processEvents()
            im = self._render_well_full(wells[wid], settings, 0)  # 0 = native
            if im is None:
                raise OSError(f"{exp_name} / {wid} の画像を描画できません")
            band = 56
            page = Image.new("RGB", (im.width, im.height + band), (255, 255, 255))
            page.paste(im, (0, band))
            im = None
            d = ImageDraw.Draw(page)
            d.text((10, 8), f"{exp_name}   {wid}   ({page.width}×{page.height - band})",
                   fill=(0, 0, 0), font=f_lbl)
            self._draw_well_caption(
                page, d, 16, band + 16, wid,
                self._caption_lines(wid, cond, headers),
                f_id, f_cond, 14, page.width - 32)
            self.progress_bar.setValue(i + 1)
            yield page

REPO = "yoshi-koba-lab/bz-plate-studio"


def _newer(remote: str, local: str) -> bool:
    """True when `remote` is a later semantic version than `local`."""
    def parts(v):
        v = v.strip().lstrip("vV").split("+")[0].split("-")[0]
        out = []
        for x in v.split("."):
            try:
                out.append(int(x))
            except ValueError:
                out.append(0)
        return tuple(out + [0, 0, 0])[:3]
    try:
        return parts(remote) > parts(local)
    except Exception:
        return False


class UpdateChecker(QThread):
    """Asks GitHub for the newest release. Silent on any failure."""

    found = Signal(str, str)     # version, html_url

    def run(self):
        import json as _json
        import urllib.request
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{REPO}/releases/latest",
                headers={"Accept": "application/vnd.github+json",
                         "User-Agent": f"{APP_NAME}/{__version__}"})
            with urllib.request.urlopen(req, timeout=6) as r:
                data = _json.loads(r.read().decode("utf-8"))
            tag = (data.get("tag_name") or "").strip()
            if tag and _newer(tag, __version__):
                self.found.emit(tag.lstrip("vV"), data.get("html_url") or "")
        except Exception:
            pass          # offline, rate-limited, no release yet — never bother the user


class StartModeDialog(QDialog):
    """Choose one of the three explicit acquisition workflows.

    The folder shapes overlap enough that guessing can silently choose the wrong
    reader.  The first card is the wide tissue/ordinary XY workflow; plate raw
    tiles and already-stitched KTF files remain separate and unchanged.
    """

    MOSAIC, RAW, KTF = "mosaic", "raw", "ktf"

    def __init__(self, parent=None, last_ktf="", last_raw="", last_mosaic=""):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} — start")
        self.setMinimumWidth(650)
        self.choice = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(12)

        head = QLabel("<span style='font-size:18px'><b>何を読み込みますか？</b></span>")
        lay.addWidget(head)
        intro = QLabel("撮影形式に合う入口を選ぶと、必要な操作だけを表示します。")
        intro.setStyleSheet("color:#64748b;")
        lay.addWidget(intro)

        for number, mode, title, desc, last, btn_text in [
            (1, self.MOSAIC, "通常画像セットを読み込む",
             "組織切片などの広範囲XY撮影を、Stage座標と画像の重なりから高精度に貼り合わせます。",
             last_mosaic, "通常画像セットを選ぶ…"),
            (2, self.RAW, "プレート画像セットを読み込む",
             "ウェルごとの X###Y### 生画像タイルを読み込み、プレート単位で貼り合わせます。",
             last_raw, "プレート画像セットを選ぶ…"),
            (3, self.KTF, ".ktfファイルを読み込む",
             "貼り合わせ済みの .ktf をプレート表示し、チャンネル調整・書き出しを行います。",
             last_ktf, ".ktf画像セットを選ぶ…"),
        ]:
            box = QGroupBox(f"{number}  {title}")
            box.setObjectName(f"{mode}WorkflowCard")
            v = QVBoxLayout(box)
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setStyleSheet("color:#4b5057;")
            v.addWidget(d)
            if last:
                p = QLabel(f"前回: {last}")
                p.setStyleSheet("color:#6b7280; font-size:10px;")
                p.setWordWrap(True)
                v.addWidget(p)
            b = QPushButton(btn_text)
            b.setObjectName(f"{mode}WorkflowButton")
            b.clicked.connect(lambda _, m=mode: self._pick(m))
            v.addWidget(b)
            lay.addWidget(box)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.setStyleSheet("""
            QDialog { background:#f3f6fa; color:#1f2937; }
            QGroupBox#mosaicWorkflowCard, QGroupBox#rawWorkflowCard,
            QGroupBox#ktfWorkflowCard {
                border:2px solid; border-radius:10px;
                margin-top:12px; padding:14px 12px 10px 12px; font-weight:600;
            }
            QGroupBox#mosaicWorkflowCard { background:#eaf3ff; border-color:#6c9ed9; }
            QGroupBox#rawWorkflowCard { background:#edf9f2; border-color:#64a979; }
            QGroupBox#ktfWorkflowCard { background:#f5efff; border-color:#9274c9; }
            QGroupBox#mosaicWorkflowCard::title, QGroupBox#rawWorkflowCard::title,
            QGroupBox#ktfWorkflowCard::title {
                subcontrol-origin:margin; left:14px; padding:0 5px;
            }
            QPushButton { min-height:30px; background:#ffffff; color:#1f2937;
                border:1px solid #b9c4d2; border-radius:7px; padding:2px 12px; }
            QPushButton#mosaicWorkflowButton, QPushButton#rawWorkflowButton,
            QPushButton#ktfWorkflowButton { color:#ffffff; font-weight:700; }
            QPushButton#mosaicWorkflowButton { background:#2878c8; border-color:#1f66ad; }
            QPushButton#mosaicWorkflowButton:hover { background:#1f69b5; }
            QPushButton#rawWorkflowButton { background:#34865a; border-color:#2b704b; }
            QPushButton#rawWorkflowButton:hover { background:#2b764e; }
            QPushButton#ktfWorkflowButton { background:#7651b5; border-color:#62429a; }
            QPushButton#ktfWorkflowButton:hover { background:#6845a5; }
        """)

    def _pick(self, mode):
        self.choice = mode
        self.accept()


class ScanWorker(QThread):
    """Finds candidate experiments off the GUI thread (a drive root can be huge)."""

    progress = Signal(str, int)      # current dir, dirs examined
    finished_scan = Signal(object, int)   # list[Path], unreadable-dir count

    def __init__(self, root: Path, mode: str):
        super().__init__()
        self.root = Path(root)
        self.mode = mode
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        found, seen, errors = [], 0, 0
        try:
            for dirpath, dirnames, filenames in os.walk(self.root, onerror=lambda e: None):
                if self._cancel:
                    self.finished_scan.emit(None, errors)   # None = cancelled
                    return
                seen += 1
                if seen % 250 == 0:
                    self.progress.emit(dirpath, seen)
                d = Path(dirpath)
                if self.mode == StartModeDialog.KTF:
                    if any(ktf_reader.is_ktf_file(Path(f)) for f in filenames):
                        found.append(d)
                        dirnames[:] = []
                elif self.mode == StartModeDialog.RAW:
                    if _is_raw_experiment(d, dirnames):
                        found.append(d)
                        dirnames[:] = []      # its wells/fields are not experiments
                else:
                    if _is_mosaic_experiment(d, filenames):
                        found.append(d)
                        dirnames[:] = []      # GCI owns the XY child sources
        except Exception:
            errors += 1
        self.finished_scan.emit(sorted(found), errors)


def _load_font_static(size):
    for cand in ("Arial.ttf", "arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 r"C:\Windows\Fonts\arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def _draw_caption_static(canvas, draw, x, y, wid, lines, f_id, f_txt, inset, max_w):
    """Well ID + conditions on a translucent plate, top-left of the image."""
    gap = max(2, inset // 3)
    idb = draw.textbbox((0, 0), wid, font=f_id)
    w_max, h_total = idb[2] - idb[0], idb[3] - idb[1]
    measured = []
    for ln in lines:
        b = draw.textbbox((0, 0), ln, font=f_txt)
        measured.append((ln, b[3] - b[1]))
        w_max = max(w_max, b[2] - b[0])
        h_total += gap + (b[3] - b[1])
    bw, bh = min(max_w, w_max + 2 * inset), h_total + 2 * inset
    plate = Image.new("RGBA", (max(1, bw), max(1, bh)), (0, 0, 0, 150))
    canvas.paste(plate, (x, y), plate)
    ty = y + inset
    draw.text((x + inset, ty), wid, fill=(255, 235, 60), font=f_id)
    ty += (idb[3] - idb[1]) + gap
    for ln, lh in measured:
        draw.text((x + inset, ty), ln, fill=(240, 240, 240), font=f_txt)
        ty += lh + gap


_ILLEGAL_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL",
                 *(f"COM{i}" for i in range(1, 10)),
                 *(f"LPT{i}" for i in range(1, 10))}


def _natural_key(text):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", str(text)) if part)


def _safe_base_name(text: str) -> str:
    """A filename fragment that is legal on both macOS and Windows ("" if nothing left).

    The name is typed by the user, so it has to survive a colon or a slash without
    silently landing in a different directory.
    """
    s = _ILLEGAL_NAME.sub("_", (text or "").strip())
    s = s.strip(" .")                    # Windows drops trailing dots and spaces
    if s.upper() in _WIN_RESERVED:
        s += "_"
    return s[:80]


def _confirm_overwrite(parent, paths, title="上書きの確認") -> bool:
    """Ask before replacing files that already exist. False = the user cancelled.

    QFileDialog only guards the one name typed into it. Every export here writes
    something it never sees — an appended extension, one file per channel, a whole
    folder of per-well files — so a second run used to replace the first with no
    warning at all.
    """
    seen, existing = set(), []
    for p in paths:
        p = Path(p)
        if p in seen:
            continue
        seen.add(p)
        try:
            if p.exists():
                existing.append(p)
        except OSError:
            pass
    if not existing:
        return True
    shown = "\n".join(f"　• {p.name}" for p in existing[:12])
    if len(existing) > 12:
        shown += f"\n　…ほか {len(existing) - 12} 件"
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(f"同じ名前のファイルが {len(existing)} 個あります。上書きしますか？")
    box.setInformativeText(
        f"保存先: {existing[0].parent}\n\n"
        f"次のファイルが置き換えられます（元に戻せません）:\n{shown}")
    ow = box.addButton("上書きする", QMessageBox.ButtonRole.DestructiveRole)
    cancel = box.addButton("キャンセル", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(cancel)         # safe default: Return must not destroy data
    box.setEscapeButton(cancel)
    box.exec()
    return box.clickedButton() is ow


def _ask_save_path(parent, title, default_name, filt, derived=None):
    """Save-As whose overwrite check covers the files actually written. None = cancelled.

    Two holes the plain dialog leaves open:
      * the extension is appended *after* Qt's check, so typing `plate` where
        `plate.pdf` already existed was accepted and then replaced it silently;
      * the TIFF export writes `<name>_<channel>.tif` — files Qt never sees.
    `derived(path)` returns the real list of files for the chosen name.
    """
    # A filter can advertise a compound extension (``*.ome.tif``) and more than
    # one legal spelling.  Looking only at ``Path.suffix`` turned ``foo`` into
    # ``foo.tif`` and even turned a valid ``foo.ome.tiff`` into
    # ``foo.ome.tiff.tif``.  Prefer the longest advertised suffix already used by
    # the default name, and accept every suffix shown in the active filter.
    accepted = []
    for suffix in re.findall(r"\*([.][A-Za-z0-9.]+)", filt or ""):
        suffix = suffix.lower()
        if suffix not in accepted:
            accepted.append(suffix)
    accepted.sort(key=len, reverse=True)
    default_lower = Path(default_name).name.lower()
    want = next((suffix for suffix in accepted if default_lower.endswith(suffix)),
                Path(default_name).suffix)
    start = default_name
    while True:
        chosen, _ = QFileDialog.getSaveFileName(parent, title, start, filt)
        if not chosen:
            return None
        path = Path(chosen)
        name_lower = path.name.lower()
        if want and not any(name_lower.endswith(suffix) for suffix in accepted or [want.lower()]):
            path = path.with_name(path.name + want)
        targets = list(derived(path)) if derived else [path]
        # Qt already confirmed the one name the user typed; only ask about the rest.
        targets = [p for p in targets if Path(p) != Path(chosen)]
        if _confirm_overwrite(parent, targets):
            return path
        start = str(path)                # reopen in the same folder, name preselected


class PlateSeriesBuilder(QWidget):
    """Build an ordered PDF series while the main plate viewer stays usable."""

    capture_activated = Signal(object)
    conditions_requested = Signal(object)
    export_requested = Signal(object, object)

    PATH_ROLE = Qt.ItemDataRole.UserRole
    SEARCH_ROLE = Qt.ItemDataRole.UserRole + 1
    NAME_ROLE = Qt.ItemDataRole.UserRole + 2
    TIME_ROLE = Qt.ItemDataRole.UserRole + 3
    STACK_ROLE = Qt.ItemDataRole.UserRole + 4
    WELLS_ROLE = Qt.ItemDataRole.UserRole + 5
    CHANNELS_ROLE = Qt.ItemDataRole.UserRole + 6

    def __init__(self, quality_specs, parent=None):
        super().__init__(parent)
        self._quality_specs = dict(quality_specs)
        self._items = []
        self._path_items = {}
        self._series_items = {}
        self._active_key = None
        self._active_path = None
        self._editor_item = None
        self._last_add_dir = Path.home()
        self._updating = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(8)

        heading = QHBoxLayout()
        title = QLabel("<b>撮影シリーズを作成</b>")
        heading.addWidget(title)
        note = QLabel("撮影をクリックすると、上のプレート詳細が切り替わります")
        note.setStyleSheet("color:#6b7280; font-size:11px;")
        heading.addWidget(note)
        heading.addStretch()
        self.btn_new_series = QPushButton("現在の撮影から作り直す")
        self.btn_new_series.setEnabled(False)
        self.btn_new_series.setToolTip(
            "一覧を現在表示中の撮影1件に戻します。保存済みのラベルとConditionsは残ります")
        self.btn_new_series.clicked.connect(self._reset_to_active)
        heading.addWidget(self.btn_new_series)
        lay.addLayout(heading)

        editor = QWidget()
        editor.setObjectName("seriesEditorBar")
        editor_grid = QGridLayout(editor)
        editor_grid.setContentsMargins(9, 6, 9, 5)
        editor_grid.setHorizontalSpacing(7)
        editor_grid.setVerticalSpacing(2)
        editor_grid.addWidget(QLabel("<b>選択中</b>"), 0, 0)
        self.ed_name = QLineEdit()
        self.ed_time = QLineEdit()
        self.ed_stack = QLineEdit()
        self.ed_time.setPlaceholderText("例: Day 7")
        self.ed_stack.setPlaceholderText("例: Stack 1")
        for column, (label, field) in enumerate((
                ("撮影名", self.ed_name),
                ("Time point", self.ed_time),
                ("Stack", self.ed_stack)), start=1):
            editor_grid.addWidget(QLabel(label), 0, column * 2 - 1)
            field.editingFinished.connect(self._save_editor)
            editor_grid.addWidget(field, 0, column * 2)
        editor_grid.setColumnStretch(2, 3)
        editor_grid.setColumnStretch(4, 2)
        editor_grid.setColumnStretch(6, 2)
        self.btn_conditions = QPushButton("この撮影のConditionsを編集")
        self.btn_conditions.clicked.connect(self._request_conditions)
        editor_grid.addWidget(self.btn_conditions, 0, 7)
        self.lbl_path = QLabel()
        self.lbl_path.setWordWrap(False)
        self.lbl_path.setMinimumWidth(0)
        self.lbl_path.setStyleSheet("color:#4b5563; font-family:Menlo; font-size:9px;")
        editor_grid.addWidget(self.lbl_path, 1, 1, 1, 6)
        self.lbl_editor = QLabel("名前はPDFの撮影タイトルに使われます")
        self.lbl_editor.setStyleSheet("color:#6b7280; font-size:10px;")
        editor_grid.addWidget(self.lbl_editor, 1, 7)
        lay.addWidget(editor)

        panes = QSplitter(Qt.Orientation.Horizontal)
        panes.setChildrenCollapsible(False)

        source_host = QWidget()
        source_lay = QVBoxLayout(source_host)
        source_lay.setContentsMargins(0, 0, 6, 0)
        source_head = QHBoxLayout()
        source_head.addWidget(QLabel("<b>追加済みの撮影</b>"))
        self.btn_add_folder = QPushButton("＋ 撮影フォルダを追加…")
        self.btn_add_folder.clicked.connect(self._add_capture_folder)
        add_visible = QPushButton("絞り込み結果を追加")
        add_visible.setToolTip("現在表示されている撮影をPDFの末尾へ追加します")
        add_visible.clicked.connect(self._include_visible)
        clear_all = QPushButton("すべて外す")
        clear_all.setToolTip("追加済みの撮影は残し、PDFの収録対象からすべて外します")
        clear_all.clicked.connect(self._exclude_all)
        source_head.addStretch()
        source_head.addWidget(self.btn_add_folder)
        source_head.addWidget(add_visible)
        source_head.addWidget(clear_all)
        source_lay.addLayout(source_head)
        self.ed_filter = QLineEdit()
        self.ed_filter.setPlaceholderText("名前・Time point・Stack・KTF名を絞り込み")
        self.ed_filter.textChanged.connect(self._filter_items)
        source_lay.addWidget(self.ed_filter)
        self.source_tree = QTreeWidget()
        self.source_tree.setHeaderLabels(
            ["撮影名", "Time point", "Stack", "Wells", "Channels"])
        self.source_tree.setRootIsDecorated(False)
        self.source_tree.setAlternatingRowColors(True)
        self.source_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.source_tree.setMinimumHeight(55)
        self.source_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.source_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            self.source_tree.header().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents)
        self.source_tree.itemChanged.connect(self._source_item_changed)
        self.source_tree.itemClicked.connect(self._source_item_clicked)
        source_lay.addWidget(self.source_tree, stretch=1)
        panes.addWidget(source_host)

        series_host = QWidget()
        series_lay = QVBoxLayout(series_host)
        series_lay.setContentsMargins(6, 0, 0, 0)
        series_head = QHBoxLayout()
        series_head.addWidget(QLabel("<b>PDFに含める順序</b>"))
        series_head.addStretch()
        self.btn_up = QPushButton("↑ 上へ")
        self.btn_down = QPushButton("↓ 下へ")
        self.btn_remove = QPushButton("外す")
        self.btn_up.clicked.connect(lambda: self._move_current(-1))
        self.btn_down.clicked.connect(lambda: self._move_current(1))
        self.btn_remove.clicked.connect(self._remove_current)
        for button in (self.btn_up, self.btn_down, self.btn_remove):
            series_head.addWidget(button)
        series_lay.addLayout(series_head)
        self.series_tree = QTreeWidget()
        self.series_tree.setHeaderLabels(["順序", "撮影名", "Time point", "Stack"])
        self.series_tree.setRootIsDecorated(False)
        self.series_tree.setAlternatingRowColors(True)
        self.series_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.series_tree.setMinimumHeight(55)
        self.series_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.series_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.series_tree.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        for column in (2, 3):
            self.series_tree.header().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents)
        self.series_tree.itemClicked.connect(self._series_item_clicked)
        series_lay.addWidget(self.series_tree, stretch=1)
        panes.addWidget(series_host)
        panes.setSizes([620, 620])
        panes.setMinimumHeight(85)
        lay.addWidget(panes, stretch=1)

        settings = QGroupBox("PDF書き出し設定")
        settings_grid = QGridLayout(settings)
        settings_grid.setContentsMargins(8, 8, 8, 6)
        settings_grid.setHorizontalSpacing(8)
        settings_grid.setVerticalSpacing(2)
        settings_grid.addWidget(QLabel("画質・ページ構成"), 0, 0)
        self.cmb_quality = QComboBox()
        self.cmb_quality.setFixedHeight(28)
        for label, spec in self._quality_specs.items():
            self.cmb_quality.addItem(label, tuple(spec))
        self.cmb_quality.setCurrentIndex(min(2, self.cmb_quality.count() - 1))
        self.cmb_quality.currentIndexChanged.connect(self._update_export_summary)
        settings_grid.addWidget(self.cmb_quality, 0, 1, 1, 2)
        self.lbl_quality = QLabel()
        self.lbl_quality.setStyleSheet("color:#4b5563; font-size:11px;")
        settings_grid.addWidget(self.lbl_quality, 1, 1, 1, 2)
        settings_grid.addWidget(QLabel("チャンネル表示"), 2, 0)
        self.lbl_channels = QLabel("全チャンネルを既定値で描画")
        self.lbl_channels.setMinimumWidth(0)
        settings_grid.addWidget(self.lbl_channels, 2, 1)
        settings_grid.addWidget(QLabel("Conditions"), 2, 2)
        settings_grid.addWidget(
            QLabel("撮影ごとの保存内容をwellへ印字"), 2, 3)
        settings_grid.addWidget(QLabel("保存先"), 2, 4)
        settings_grid.addWidget(
            QLabel("書き出し時に指定（置換前に確認）"), 2, 5)
        settings_grid.setColumnStretch(1, 4)
        settings_grid.setColumnStretch(3, 2)
        settings_grid.setColumnStretch(5, 2)
        lay.addWidget(settings)

        actions = QHBoxLayout()
        self.lbl_summary = QLabel()
        self.lbl_summary.setStyleSheet("color:#4b5563;")
        actions.addWidget(self.lbl_summary)
        actions.addStretch()
        self.btn_export = QPushButton("シリーズPDFを書き出す")
        self.btn_export.setObjectName("primaryExportButton")
        self.btn_export.clicked.connect(self._request_export)
        actions.addWidget(self.btn_export)
        lay.addLayout(actions)
        self._set_editor_item(None)
        self._update_export_summary()

    @property
    def is_empty(self):
        return not self._items

    @staticmethod
    def _path_key(path):
        return _path_key(path)

    @classmethod
    def _labels_key(cls, path):
        return f"plate_series_labels/{cls._path_key(path)}"

    def _load_labels(self, path):
        raw = QSettings().value(self._labels_key(path), "")
        try:
            data = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            data = {}
        return data if isinstance(data, dict) else {}

    def _store_labels(self, item):
        path = Path(item.data(0, self.PATH_ROLE))
        data = {
            "name": item.data(0, self.NAME_ROLE),
            "time_point": item.data(0, self.TIME_ROLE),
            "stack": item.data(0, self.STACK_ROLE),
        }
        QSettings().setValue(self._labels_key(path), json.dumps(data, ensure_ascii=False))

    def _default_label(self, path):
        return str(Path(path.parent.name) / path.name)

    def _unique_default_label(self, path):
        label = self._default_label(path)
        used = {str(item.data(0, self.NAME_ROLE)) for item in self._items}
        if label not in used:
            return label
        parts = path.parts
        for depth in range(3, len(parts) + 1):
            candidate = str(Path(*parts[-depth:]))
            if candidate not in used:
                return candidate
        return str(path)

    def reset(self, current):
        self._updating = True
        self.source_tree.clear()
        self.series_tree.clear()
        self._items.clear()
        self._path_items.clear()
        self._series_items.clear()
        self._active_key = None
        self._active_path = None
        self._editor_item = None
        self._updating = False
        current = Path(current)
        self._last_add_dir = current.parent
        self.ensure_capture(current, include=True, focus=True)
        self.set_active_path(current)

    def ensure_capture(self, path, include=False, focus=False):
        path = Path(path)
        key = self._path_key(path)
        item = self._path_items.get(key)
        if item is None:
            item = self._add_path_item(path, checked=include, require_ktf=False)
        elif include and item.checkState(0) != Qt.CheckState.Checked:
            item.setCheckState(0, Qt.CheckState.Checked)
        if item is not None and focus:
            self.source_tree.setCurrentItem(item)
            self.source_tree.scrollToItem(item)
            self._set_editor_item(item)
        return item

    def _add_path_item(self, path, checked=True, require_ktf=True):
        path = Path(path)
        key = self._path_key(path)
        existing = self._path_items.get(key)
        if existing is not None:
            existing.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self.source_tree.setCurrentItem(existing)
            self._set_editor_item(existing)
            return existing

        example, wells, channels, search_text = self._source_info(path)
        partial_errors = []
        if require_ktf and not example:
            QMessageBox.warning(
                self, "KTF撮影フォルダではありません",
                f"“{path}” にはKTFファイルが直接入っていません。\n\n"
                "各wellのKTFファイルを直接含む撮影フォルダを選択してください。")
            return None
        if require_ktf:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                experiment = ktf_reader.scan_experiment_folder(path)
            except Exception as e:
                QMessageBox.warning(
                    self, "撮影フォルダを読み込めません",
                    f"“{path}” を読み込めませんでした。\n\n{_friendly_error(e)}")
                return None
            finally:
                QApplication.restoreOverrideCursor()
            if not experiment["wells"]:
                detail = (f"\n\n読取不能なKTF: {len(experiment.get('errors') or [])} 件"
                          if experiment.get("errors") else "")
                QMessageBox.warning(
                    self, "読み取れるKTFがありません",
                    f"“{path}” には読み取れるwell画像がありません。{detail}")
                return None
            wells = len(experiment["wells"])
            channel_ids = {
                channel for per_well in experiment["wells"].values() for channel in per_well}
            channels = ", ".join(sorted(channel_ids))
            partial_errors = experiment.get("errors") or []

        saved = self._load_labels(path)
        name = str(saved.get("name") or self._unique_default_label(path))
        time_point = str(saved.get("time_point") or "")
        stack = str(saved.get("stack") or "")
        self._updating = True
        item = QTreeWidgetItem(self.source_tree)
        item.setText(0, name)
        item.setText(1, time_point)
        item.setText(2, stack)
        item.setText(3, str(wells))
        item.setText(4, channels)
        item.setToolTip(0, f"{path}\nKTF: {example}" + (
            f"\n⚠ 読取不能: {len(partial_errors)} 件" if partial_errors else ""))
        item.setData(0, self.PATH_ROLE, str(path))
        item.setData(0, self.SEARCH_ROLE, f"{path} {search_text}")
        item.setData(0, self.NAME_ROLE, name)
        item.setData(0, self.TIME_ROLE, time_point)
        item.setData(0, self.STACK_ROLE, stack)
        item.setData(0, self.WELLS_ROLE, int(wells))
        item.setData(0, self.CHANNELS_ROLE, channels)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self._items.append(item)
        self._path_items[key] = item
        item.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._updating = False
        if checked:
            self._set_included(item, True)
        self.source_tree.setCurrentItem(item)
        self._set_editor_item(item)
        if partial_errors:
            QMessageBox.warning(
                self, "一部のKTFを読み取れません",
                f"“{path}” は追加しましたが、{len(partial_errors)} 件のKTFを"
                "読み取れません。PDF書き出し前にも確認します。")
        return item

    def _add_capture_folder(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Stack / time point の撮影フォルダを追加", str(self._last_add_dir))
        if not chosen:
            return
        path = Path(chosen)
        self._last_add_dir = path.parent
        self.ed_filter.clear()
        item = self._add_path_item(path, checked=True, require_ktf=True)
        if item is not None:
            self.lbl_editor.setText("追加しました。行をクリックするとプレート詳細を表示します")

    def _reset_to_active(self):
        if self._active_path is None:
            return
        if len(self._items) > 1:
            ans = QMessageBox.question(
                self, "シリーズを作り直す",
                "撮影一覧とPDF順序を、現在表示中の撮影1件に戻しますか？\n\n"
                "保存済みの撮影ラベルとConditionsは削除されません。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return
        self.reset(self._active_path)

    def _source_item_changed(self, item, column):
        if self._updating or column != 0:
            return
        key = self._path_key(item.data(0, self.PATH_ROLE))
        included = item.checkState(0) == Qt.CheckState.Checked
        if included != (key in self._series_items):
            self._set_included(item, included)

    def _set_included(self, source_item, included):
        key = self._path_key(source_item.data(0, self.PATH_ROLE))
        series_item = self._series_items.get(key)
        if included and series_item is None:
            series_item = QTreeWidgetItem(self.series_tree)
            series_item.setData(0, self.PATH_ROLE, source_item.data(0, self.PATH_ROLE))
            self._series_items[key] = series_item
            self._refresh_series_item(source_item)
            self.series_tree.setCurrentItem(series_item)
        elif not included and series_item is not None:
            index = self.series_tree.indexOfTopLevelItem(series_item)
            if index >= 0:
                self.series_tree.takeTopLevelItem(index)
            self._series_items.pop(key, None)
        self._renumber_series()
        self._update_export_summary()

    def _refresh_series_item(self, source_item):
        key = self._path_key(source_item.data(0, self.PATH_ROLE))
        item = self._series_items.get(key)
        if item is None:
            return
        item.setText(1, str(source_item.data(0, self.NAME_ROLE) or ""))
        item.setText(2, str(source_item.data(0, self.TIME_ROLE) or ""))
        item.setText(3, str(source_item.data(0, self.STACK_ROLE) or ""))
        item.setToolTip(1, source_item.toolTip(0))

    def _renumber_series(self):
        for index in range(self.series_tree.topLevelItemCount()):
            self.series_tree.topLevelItem(index).setText(0, str(index + 1))

    def _source_item_clicked(self, item, _column):
        if self._editor_item is not item:
            self._save_editor()
        self._set_editor_item(item)
        self.capture_activated.emit(Path(item.data(0, self.PATH_ROLE)))

    def _series_item_clicked(self, item, _column):
        key = self._path_key(item.data(0, self.PATH_ROLE))
        source = self._path_items.get(key)
        if source is None:
            return
        if self._editor_item is not source:
            self._save_editor()
        self.source_tree.setCurrentItem(source)
        self._set_editor_item(source)
        self.capture_activated.emit(Path(source.data(0, self.PATH_ROLE)))

    def _set_editor_item(self, item):
        self._editor_item = item
        enabled = item is not None
        for field in (self.ed_name, self.ed_time, self.ed_stack):
            field.setEnabled(enabled)
        self.btn_conditions.setEnabled(enabled)
        self.ed_name.setText(str(item.data(0, self.NAME_ROLE) or "") if item else "")
        self.ed_time.setText(str(item.data(0, self.TIME_ROLE) or "") if item else "")
        self.ed_stack.setText(str(item.data(0, self.STACK_ROLE) or "") if item else "")
        self.lbl_path.setText(str(item.data(0, self.PATH_ROLE) or "") if item else "")
        self.lbl_path.setToolTip(str(item.data(0, self.PATH_ROLE) or "") if item else "")
        self.lbl_editor.setText(
            "名前・Time point・StackはPDFの撮影タイトルに使われます"
            if item else "左または右の撮影を選択してください")

    def _save_editor(self):
        item = self._editor_item
        if item is None:
            return
        name = self.ed_name.text().strip()
        if not name:
            name = str(item.data(0, self.NAME_ROLE) or "")
            self.ed_name.setText(name)
            self.lbl_editor.setText("撮影名は空にできません")
            return
        time_point = self.ed_time.text().strip()
        stack = self.ed_stack.text().strip()
        self._updating = True
        item.setData(0, self.NAME_ROLE, name)
        item.setData(0, self.TIME_ROLE, time_point)
        item.setData(0, self.STACK_ROLE, stack)
        item.setText(0, name)
        item.setText(1, time_point)
        item.setText(2, stack)
        self._updating = False
        self._store_labels(item)
        self._refresh_series_item(item)
        self._update_export_summary()
        self.lbl_editor.setText("ラベルを保存しました")

    def _request_conditions(self):
        if self._editor_item is None:
            return
        self._save_editor()
        self.conditions_requested.emit(Path(self._editor_item.data(0, self.PATH_ROLE)))

    def _move_current(self, delta):
        item = self.series_tree.currentItem()
        if item is None:
            return
        index = self.series_tree.indexOfTopLevelItem(item)
        target = index + delta
        if index < 0 or target < 0 or target >= self.series_tree.topLevelItemCount():
            return
        item = self.series_tree.takeTopLevelItem(index)
        self.series_tree.insertTopLevelItem(target, item)
        self.series_tree.setCurrentItem(item)
        self._renumber_series()
        self._update_export_summary()

    def _remove_current(self):
        item = self.series_tree.currentItem()
        if item is None:
            return
        key = self._path_key(item.data(0, self.PATH_ROLE))
        source = self._path_items.get(key)
        if source is not None:
            source.setCheckState(0, Qt.CheckState.Unchecked)

    def _include_visible(self):
        for item in self._items:
            if not item.isHidden():
                item.setCheckState(0, Qt.CheckState.Checked)

    def _exclude_all(self):
        for item in self._items:
            item.setCheckState(0, Qt.CheckState.Unchecked)

    def _filter_items(self, text):
        needle = text.strip().casefold()
        for item in self._items:
            haystack = " ".join([
                item.text(0), item.text(1), item.text(2),
                str(item.data(0, self.SEARCH_ROLE) or "")]).casefold()
            matched = not needle or needle in haystack
            if matched and needle and needle[-1].isdigit():
                matched = re.search(re.escape(needle) + r"(?!\d)", haystack) is not None
            item.setHidden(not matched)

    @staticmethod
    def _source_info(path):
        try:
            names, wells, channels = [], set(), set()
            for p in path.iterdir():
                if not ktf_reader.is_ktf_file(p):
                    continue
                names.append(p.stem)
                for part in p.stem.split("_"):
                    if len(part) >= 2 and part[0].isalpha() and part[1:].isdigit():
                        wells.add(part)
                    if part.startswith("CH"):
                        channels.add(part)
        except OSError:
            return "", 0, "", ""
        example = min(names, key=_natural_key) if names else ""
        return example, len(wells), ", ".join(sorted(channels)), " ".join(names)

    @staticmethod
    def _compose_label(item):
        parts = [str(item.data(0, PlateSeriesBuilder.NAME_ROLE) or "").strip()]
        time_point = str(item.data(0, PlateSeriesBuilder.TIME_ROLE) or "").strip()
        stack = str(item.data(0, PlateSeriesBuilder.STACK_ROLE) or "").strip()
        if time_point:
            parts.append(f"Time point: {time_point}")
        if stack:
            parts.append(f"Stack: {stack}")
        return " · ".join(part for part in parts if part)

    @property
    def selected(self):
        out = []
        for index in range(self.series_tree.topLevelItemCount()):
            series_item = self.series_tree.topLevelItem(index)
            key = self._path_key(series_item.data(0, self.PATH_ROLE))
            source = self._path_items.get(key)
            if source is not None:
                out.append((Path(source.data(0, self.PATH_ROLE)), self._compose_label(source)))
        return out

    @property
    def quality_spec(self):
        return tuple(self.cmb_quality.currentData())

    def _quality_description(self):
        source, panel, dpi, layout = self.quality_spec
        source_text = "埋め込みthumbnail" if source == "thumb" else "full KTF"
        size_text = "原寸" if panel == 0 else f"約{panel} px / well"
        layout_text = {
            "grid": "撮影ごとにoverview 1ページ",
            "pages": "wellごとに1ページ",
            "both": "overview＋well別ページ",
        }[layout]
        return f"{source_text} · {size_text} · {dpi} DPI · {layout_text}"

    def _update_export_summary(self):
        if not hasattr(self, "lbl_summary"):
            return
        selected = self.selected
        wells = sum(int(self._path_items[self._path_key(path)].data(0, self.WELLS_ROLE) or 0)
                    for path, _label in selected)
        layout = self.quality_spec[3] if self.cmb_quality.count() else "grid"
        pages = len(selected) if layout == "grid" else wells
        if layout == "both":
            pages += len(selected)
        self.lbl_summary.setText(
            f"{len(selected)}撮影 · {wells} wells · 推定{pages}ページ")
        self.lbl_quality.setText(self._quality_description())
        self.btn_export.setEnabled(bool(selected))

    def set_channel_summary(self, text):
        self.lbl_channels.setText(text)

    def set_active_path(self, path):
        self._active_path = Path(path)
        self.btn_new_series.setEnabled(True)
        key = self._path_key(path)
        self._active_key = key
        self._updating = True
        for item in self._items:
            item_key = self._path_key(item.data(0, self.PATH_ROLE))
            bold = item_key == key
            for column in range(5):
                font = item.font(column)
                font.setBold(bold)
                item.setFont(column, font)
        self._updating = False
        for item_key, item in self._series_items.items():
            bold = item_key == key
            for column in range(4):
                font = item.font(column)
                font.setBold(bold)
                item.setFont(column, font)

    def clear_active_path(self):
        self._active_path = None
        self._active_key = None
        self.btn_new_series.setEnabled(False)
        self._updating = True
        for item in self._items:
            for column in range(5):
                font = item.font(column)
                font.setBold(False)
                item.setFont(column, font)
        for item in self._series_items.values():
            for column in range(4):
                font = item.font(column)
                font.setBold(False)
                item.setFont(column, font)
        self._updating = False

    def set_exporting(self, exporting):
        self.setEnabled(not exporting)

    def flush_edits(self):
        self._save_editor()

    def _request_export(self):
        self._save_editor()
        selected = tuple(self.selected)
        if not selected:
            QMessageBox.warning(
                self, "撮影が選ばれていません",
                "PDFに含める撮影を1つ以上追加してください。")
            return
        self.export_requested.emit(selected, self.quality_spec)


class OutputTargetDialog(QDialog):
    """Where a multi-file export goes, and what its files are called.

    Picking only a folder gave the user no say in the filenames and quietly
    replaced the previous run. Here the name is typed like any “Save As”, and a
    folder of its own (on by default) keeps two runs from landing on top of each
    other.
    """

    def __init__(self, parent, title, default_base, sample, start_dir=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(600)
        self._sample = sample            # e.g. "{base}_A01.ome.tif"
        lay = QVBoxLayout(self)

        form = QGridLayout()
        form.addWidget(QLabel("保存先:"), 0, 0)
        self.ed_folder = QLineEdit(start_dir)
        self.ed_folder.setPlaceholderText("保存先のフォルダ")
        form.addWidget(self.ed_folder, 0, 1)
        btn = QPushButton("参照…")
        btn.clicked.connect(self._browse)
        form.addWidget(btn, 0, 2)

        form.addWidget(QLabel("名前:"), 1, 0)
        self.ed_name = QLineEdit(_safe_base_name(default_base))
        self.ed_name.setPlaceholderText("ファイル名（この名前で始まる名前で保存されます）")
        form.addWidget(self.ed_name, 1, 1, 1, 2)
        form.setColumnStretch(1, 1)
        lay.addLayout(form)

        self.chk_sub = QCheckBox("この名前のフォルダを作って、その中に保存する")
        self.chk_sub.setChecked(True)
        self.chk_sub.setToolTip(
            "書き出すたびに専用のフォルダができるので、前回の結果を上書きしません。")
        lay.addWidget(self.chk_sub)

        self.lbl_preview = QLabel()
        self.lbl_preview.setWordWrap(True)
        self.lbl_preview.setStyleSheet("color:#6b7280; font-size:11px;")
        lay.addWidget(self.lbl_preview)

        self.box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                    QDialogButtonBox.StandardButton.Cancel)
        self.box.button(QDialogButtonBox.StandardButton.Save).setText("この名前で保存")
        self.box.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        self.box.accepted.connect(self._accept)
        self.box.rejected.connect(self.reject)
        lay.addWidget(self.box)

        self.ed_folder.textChanged.connect(self._refresh)
        self.ed_name.textChanged.connect(self._refresh)
        self.chk_sub.toggled.connect(self._refresh)
        self._refresh()
        self.ed_name.setFocus()
        self.ed_name.selectAll()

    # ---- chosen target ----
    @property
    def base(self) -> str:
        return _safe_base_name(self.ed_name.text())

    @property
    def root(self) -> Path:
        """The folder the user browsed to (without the per-run subfolder)."""
        return Path(self.ed_folder.text().strip()).expanduser()

    @property
    def folder(self) -> Path:
        return self.root / self.base if self.chk_sub.isChecked() else self.root

    # ---- ui ----
    def _browse(self):
        start = self.ed_folder.text().strip()
        if not Path(start or "/").is_dir():
            start = str(Path.home())
        f = QFileDialog.getExistingDirectory(self, "保存先フォルダを選ぶ", start)
        if f:
            self.ed_folder.setText(f)

    def _refresh(self):
        base, root = self.base, self.ed_folder.text().strip()
        ok = bool(base) and bool(root)
        self.box.button(QDialogButtonBox.StandardButton.Save).setEnabled(ok)
        if not ok:
            self.lbl_preview.setText("保存先フォルダと名前を入力してください。")
            return
        self.lbl_preview.setText(
            f"保存例:　{self.folder / self._sample.format(base=base)}")

    def _accept(self):
        root = self.root
        if not root.is_dir():
            QMessageBox.warning(self, "保存先がありません",
                                f"“{root}” は存在しないか、フォルダではありません。")
            return
        problem = _writable_problem(root)
        if problem:
            QMessageBox.critical(self, "この場所には保存できません", problem)
            return
        out = self.folder
        if out != root:
            try:
                out.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                QMessageBox.critical(
                    self, "フォルダを作成できません",
                    f"“{out}” を作成できませんでした（{e.strerror or e}）。")
                return
        self.accept()


def _writable_problem(folder: Path) -> str:
    """Plain-language reason this folder cannot receive output, or "" if it can.

    Checked before stitching: an NTFS volume mounts read-only on macOS, and
    discovering that only after several minutes of work is the worst time.
    """
    try:
        if not folder.is_dir():
            return f"“{folder}” はフォルダではありません。"
        try:
            st = os.statvfs(folder)
            if bool(st.f_flag & getattr(os, "ST_RDONLY", 1)):
                return (f"“{folder}” は読み取り専用でマウントされています。\n\n"
                        "NTFS でフォーマットされた外付けドライブは、macOS では既定で"
                        "読み取り専用になり、書き込みできません。\n"
                        "内蔵ディスクなど、書き込める場所を選び直してください。")
        except (OSError, AttributeError):
            pass
        if not os.access(folder, os.W_OK):
            return (f"“{folder}” に書き込む権限がありません。\n"
                    "別の場所を選ぶか、フォルダの権限をご確認ください。")
        # Never probe with ``touch`` + ``unlink`` on a fixed name: if a user
        # already had that file, the check itself deleted it.  QSaveFile stages a
        # unique neighbour and cancelWriting() removes only that private staging
        # file without ever committing the nominal target.
        probe = QSaveFile(str(folder / ".bz-studio-write-test"))
        probe.setDirectWriteFallback(False)
        if not probe.open(QIODevice.OpenModeFlag.WriteOnly):
            detail = probe.errorString() or "unknown write error"
            return (f"“{folder}” に書き込めませんでした（{detail}）。\n"
                    "別の場所を選び直してください。")
        probe.cancelWriting()
    except Exception as e:
        return f"“{folder}” を確認できませんでした: {e}"
    return ""


def _friendly_error(e: Exception) -> str:
    """Turn a low-level write failure into something a user can act on."""
    txt = str(e)
    if "Read-only file system" in txt or getattr(e, "errno", None) == 30:
        return "保存先が読み取り専用のため書き込めません"
    if "Permission denied" in txt or getattr(e, "errno", None) == 13:
        return "保存先に書き込む権限がありません"
    if "No space left" in txt or getattr(e, "errno", None) == 28:
        return "保存先の空き容量が足りません"
    # never surface the internal .part temp name
    return txt.replace(".part", "")


def _raw_wells_of(folder: Path) -> list:
    """Well folders directly under `folder` that hold X###Y### tile files.

    Purely structural — it never opens a TIFF, so scanning a whole drive stays
    cheap. `.ici` / `.ibc2` / `.gci` never qualify an experiment on their own.
    """
    wells = []
    try:
        entries = list(folder.iterdir())
    except OSError:
        return wells
    for well in entries:
        if not well.is_dir() or well.name.startswith("._"):
            continue
        try:
            positions = list(well.iterdir())
        except OSError:
            continue
        for pos in positions:
            if not pos.is_dir() or not stitcher.POS_RE.fullmatch(pos.name):
                continue
            try:
                hit = any(f.is_file() and not f.name.startswith("._")
                          and stitcher.TILE_RE.search(f.name)
                          for f in pos.iterdir())
            except OSError:
                continue
            if hit:                       # one proven tile is enough
                wells.append(well.name)
                break
    return sorted(wells)


def _is_raw_experiment(folder: Path, dirnames=None) -> bool:
    return bool(_raw_wells_of(folder))


def _is_mosaic_experiment(folder: Path, filenames=None) -> bool:
    """A discovery candidate has exactly one GCI directly inside it."""
    try:
        names = (list(filenames) if filenames is not None
                 else [entry.name for entry in folder.iterdir()])
    except OSError:
        return False
    return sum(str(name).lower().endswith(".gci") for name in names) == 1


#: sequentially-numbered tile, e.g. IPF1_XY01_00457_CHF.bz.ome.tif
_XYSCAN_TILE = re.compile(r"_XY\d+_\d{4,}_CH[\w-]*\.bz\.ome\.tiff?$", re.IGNORECASE)


def _looks_like_xy_scan(folder: Path) -> bool:
    """An XY-scan (tissue-section) capture rather than a well plate.

    Those store tiles as a flat run of sequential numbers under XY01/, with no
    X###Y### position folders, so the plate reader legitimately finds nothing.
    Recognising the shape lets the app say *why* instead of "no experiments".
    """
    # The user may point at the experiment itself or at a parent, so look a few
    # levels down — but bounded, since this runs only to explain an empty result.
    def scan(d: Path, depth: int) -> bool:
        try:
            entries = list(d.iterdir())[:400]
        except OSError:
            return False
        for f in entries:
            if f.is_file() and _XYSCAN_TILE.search(f.name):
                return True
        if depth <= 0:
            return False
        for sub in entries:
            if sub.is_dir() and not sub.name.startswith("._") and scan(sub, depth - 1):
                return True
        return False

    return scan(folder, 3)


class StitchDialog(QDialog):
    """Options for rebuilding whole-well mosaics from the raw BZ-X tiles."""

    def __init__(self, wells, parent=None, current_well=None):
        super().__init__(parent)
        self._current_well = current_well
        self.setWindowTitle("Stitch raw tiles")
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)

        n_tiles = sum(w.n_tiles for w in wells.values())
        n_z = max(len(w.z_values) for w in wells.values())
        planes = sorted({p.label for w in wells.values() for p in w.planes})
        info = QLabel(
            f"<b>{len(wells)} wells</b> · {n_tiles} tiles · "
            f"{len(planes)} channel(s): {', '.join(planes)}"
            + (f" · {n_z} Z slices" if n_z > 1 else ""))
        info.setWordWrap(True)
        lay.addWidget(info)

        form = QGridLayout()
        r = 0
        form.addWidget(QLabel("Wells:"), r, 0)
        self.cmb_wells = QComboBox()
        self.cmb_wells.addItem(f"All wells ({len(wells)})")
        if current_well:
            self.cmb_wells.addItem(f"Selected well only ({current_well})")
        form.addWidget(self.cmb_wells, r, 1); r += 1

        if n_z > 1:
            form.addWidget(QLabel("Z slices:"), r, 0)
            self.cmb_z = QComboBox()
            self.cmb_z.addItems(["Maximum projection", "Average projection", "Middle slice"])
            form.addWidget(self.cmb_z, r, 1); r += 1
        else:
            self.cmb_z = None

        form.addWidget(QLabel("Output:"), r, 0)
        self.cmb_fmt = QComboBox()
        self.cmb_fmt.addItems([
            "OME-TIFF (multi-channel) + PNG preview",
            "OME-TIFF (multi-channel) only",
            "PNG composite only",
            "Separate TIFF per channel",
            "PDF (1 well per page)",
            "PDF contact sheet (all wells on one page)",
            "PDF: contact sheet + one well per page",
            "OME-TIFF + PDF contact sheet",
        ])
        form.addWidget(self.cmb_fmt, r, 1); r += 1
        lay.addLayout(form)

        self.chk_flat = QCheckBox("照明ムラ補正（フラットフィールド）")
        self.chk_flat.setChecked(False)   # off by default: leave the pixels untouched
        self.chk_flat.setToolTip(
            "各チャンネルのタイル群から照明プロファイルを推定し、保存する画素に適用します。\n"
            "既定はオフ（撮影されたままの値）です。同じ明るさの標本が視野内の位置だけで\n"
            "±40 階調ずれるため、ウェル間で強度を比較する場合はオンにしてください。")
        lay.addWidget(self.chk_flat)
        self.chk_sub = QCheckBox("Sub-pixel seam check (slower, writes stitch_qc.csv)")
        self.chk_sub.setChecked(True)
        lay.addWidget(self.chk_sub)

        note = QLabel("Tile offsets are measured from the images themselves, so all "
                      "channels stay registered. Output opens in Fiji / QuPath / napari.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#6b7280; font-size:11px;")
        lay.addWidget(note)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                               QDialogButtonBox.StandardButton.Cancel)
        box.button(QDialogButtonBox.StandardButton.Ok).setText("保存先と名前を指定…")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    @property
    def all_wells(self):
        return self.cmb_wells.currentIndex() == 0

    @property
    def z_mode(self):
        if self.cmb_z is None:
            return "max"
        return ["max", "mean", "middle"][self.cmb_z.currentIndex()]

    @property
    def fmt(self):
        return ["both", "ometiff", "png", "split",
                "pdf_pages", "pdf_sheet", "pdf_sheet_pages",
                "ometiff_pdf"][self.cmb_fmt.currentIndex()]

    @property
    def flatfield(self):
        return self.chk_flat.isChecked()

    @property
    def subpixel(self):
        return self.chk_sub.isChecked()


class StitchWorker(QThread):
    """Runs stitching off the GUI thread."""
    progress = Signal(str, float)          # message, 0..1
    done = Signal(int, int, str)           # ok, failed, out_dir

    #: what one output file looks like, per format — shown in the Save-As preview
    SAMPLE_NAME = {
        "both": "{base}_A01.ome.tif",
        "ometiff": "{base}_A01.ome.tif",
        "png": "{base}_A01.png",
        "split": "{base}_A01_CH1.tif",
        "pdf_pages": "{base}_A01.pdf",
        "pdf_sheet": "{base}_plate.pdf",
        "pdf_sheet_pages": "{base}_plate_and_wells.pdf",
        "ometiff_pdf": "{base}_A01.ome.tif",
    }

    @staticmethod
    def planned_outputs(out: Path, base: str, wids, fmt) -> list:
        """Every file this run will write, so nothing is replaced without asking.

        Kept next to `_write`, which is the only place these names are produced —
        the two must agree or the warning silently misses files.
        """
        out, paths = Path(out), []
        for wid in wids:
            if fmt in ("ometiff", "both", "ometiff_pdf"):
                paths.append(out / f"{base}_{wid}.ome.tif")
            if fmt in ("png", "both"):
                paths.append(out / f"{base}_{wid}.png")
            if fmt == "pdf_pages":
                paths.append(out / f"{base}_{wid}.pdf")
            if fmt == "split":
                # channel labels are only known after stitching, so match on the stem
                try:
                    paths += sorted(out.glob(f"{base}_{wid}_*.tif"))
                except OSError:
                    pass
        if fmt in ("pdf_sheet", "ometiff_pdf"):
            paths.append(out / f"{base}_plate.pdf")
        if fmt == "pdf_sheet_pages":
            paths.append(out / f"{base}_plate_and_wells.pdf")
        paths.append(out / f"{base}_stitch_qc.csv")
        return paths

    def __init__(self, wells, out_dir, z_mode, fmt, flatfield=True, subpixel=True,
                 exp_name="experiment", base="", conditions=None, cond_headers=None):
        super().__init__()
        self.exp_name = exp_name
        self.base = _safe_base_name(base) or _safe_base_name(exp_name) or "stitched"
        self._cond = conditions or {}
        self._cond_headers = cond_headers or []
        self.sheet_panels = {}
        self.well_pages = []
        self.wells = wells
        self.out_dir = Path(out_dir)
        self.z_mode = z_mode
        self.fmt = fmt
        self.flatfield = flatfield
        self.subpixel = subpixel
        self.warnings = []
        self.qc = []
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def conditions_for(self, wid) -> list:
        """"Header: value" lines for this well, from the Conditions tab."""
        out = []
        for i, val in enumerate(self._cond.get(wid, [])):
            val = (val or "").strip()
            if not val:
                continue
            head = self._cond_headers[i + 1] if i + 1 < len(self._cond_headers) else ""
            out.append(f"{head}: {val}" if head else val)
        return out

    def run(self):
        ok = failed = 0
        total = len(self.wells)
        order = sorted(self.wells.items())
        self.qc = []
        prior = None
        # Two passes: the first well that aligns confidently seeds a plate
        # geometry, which later sparse/empty wells inherit instead of abutting.
        deferred = []
        for i, (wid, wt) in enumerate(order):
            if self._cancel:
                break
            base = i / total

            def cb(msg, frac, _b=base):
                # -1 = "message only": never rewind the bar to the well's start
                if msg:
                    self.progress.emit(msg, -1.0)
                elif frac is not None:
                    self.progress.emit("", _b + frac / total)

            try:
                res = stitcher.stitch_well(
                    wt, z_mode=self.z_mode, progress=cb,
                    cancel=lambda: self._cancel,
                    flatfield=self.flatfield, subpixel=self.subpixel, prior=prior)
                geo = res.pop("__geometry__", {}) or {}
                if not res:
                    failed += 1
                    continue
                self.qc.append(geo)
                prior = stitcher.plate_geometry(self.qc) or prior
                if geo.get("step_source", {}).get("x") == "abut" or \
                        geo.get("step_source", {}).get("y") == "abut":
                    # retry later, once some other well has produced a geometry
                    deferred.append((wid, wt))
                    continue
                self._record_warnings(wid, geo)
                self._write(wid, wt, res)
                ok += 1
            except Exception as e:
                print(f"Stitch error {wid}: {e}")
                self.warnings.append(f"{wid}: {_friendly_error(e)}")
                failed += 1

        for wid, wt in deferred:                 # second pass with the plate prior
            if self._cancel:
                break
            self.progress.emit(f"{wid}: retrying with plate geometry…", -1.0)
            try:
                res = stitcher.stitch_well(
                    wt, z_mode=self.z_mode, cancel=lambda: self._cancel,
                    flatfield=self.flatfield, subpixel=self.subpixel, prior=prior)
                geo = res.pop("__geometry__", {}) or {}
                if not res:
                    failed += 1
                    continue
                self.qc.append(geo)
                self._record_warnings(wid, geo)
                self._write(wid, wt, res)
                ok += 1
            except Exception as e:
                print(f"Stitch error {wid}: {e}")
                self.warnings.append(f"{wid}: {_friendly_error(e)}")
                failed += 1

        if self.fmt in ("pdf_sheet", "pdf_sheet_pages", "ometiff_pdf") \
                and self.sheet_panels and not self._cancel:
            self.progress.emit("コンタクトシート PDF を作成中…", -1.0)
            try:
                self._write_contact_sheet()
            except Exception as e:
                print(f"contact sheet failed: {e}")
                self.warnings.append(f"コンタクトシート PDF: {_friendly_error(e)}")
        self._write_qc()
        self.done.emit(ok, failed, str(self.out_dir))

    def _record_warnings(self, wid, geo):
        src = geo.get("step_source", {}) or {}
        if "abut" in src.values():
            self.warnings.append(f"{wid}: geometry unmeasurable — tiles abutted (check this well)")
        elif "plate" in src.values():
            self.warnings.append(f"{wid}: geometry inherited from the plate")
        elif geo.get("low_confidence"):
            self.warnings.append(f"{wid}: tile alignment uncertain")
        r95 = geo.get("residual_p95")
        if r95 is not None and r95 > 2.0:
            self.warnings.append(f"{wid}: seam residual p95 = {r95} px (>2 px)")
        dev = geo.get("step_deviation")
        if dev is not None and dev > 8:
            self.warnings.append(
                f"{wid}: geometry differs from the rest of the plate by {dev} px "
                f"— check this well (seam residual alone cannot detect this)")
        if geo.get("shading_identifiable") is False and self.flatfield:
            self.warnings.append(
                f"{wid}: tiles overlap too much to separate shading from specimen "
                f"— flat-field may have removed real structure")
        if geo.get("unreadable"):
            self.warnings.append(f"{wid}: {len(geo['unreadable'])} unreadable tile(s)")

    def _write_qc(self):
        """Per-well QC table next to the images — the evidence the stitch is sound."""
        if not getattr(self, "qc", None):
            return
        cols = ["well", "tiles", "edges", "src_x", "src_y",
                "step_x_dy", "step_x_dx", "step_y_dy", "step_y_dx",
                "overlap_x", "overlap_y",
                # these three are INTERNAL consistency, not accuracy
                "lattice_residual_median", "lattice_residual_p95",
                "lattice_residual_max",
                "step_deviation_vs_plate", "shading_identifiable",
                "edge_ncc_median", "edge_ncc_min",
                "ambiguous_edges", "flatfield", "low_confidence",
                "width", "height"]
        try:
            with open(self.out_dir / f"{self.base}_stitch_qc.csv", "w", newline="",
                      encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for g in self.qc:
                    src = g.get("step_source", {}) or {}
                    sx = g.get("step_x", ("", ""))
                    sy = g.get("step_y", ("", ""))
                    shape = g.get("shape", ("", ""))
                    row = {
                        **g,
                        "src_x": src.get("x", ""), "src_y": src.get("y", ""),
                        "step_x_dy": sx[0], "step_x_dx": sx[1],
                        "step_y_dy": sy[0], "step_y_dx": sy[1],
                        "height": shape[0], "width": shape[1],
                        "lattice_residual_median": g.get("residual_median", ""),
                        "lattice_residual_p95": g.get("residual_p95", ""),
                        "lattice_residual_max": g.get("residual_max", ""),
                        "step_deviation_vs_plate": g.get("step_deviation", ""),
                    }
                    w.writerow([row.get(c, "") for c in cols])
        except Exception as e:
            print(f"QC write failed: {e}")

    @staticmethod
    def _composite(planes_data):
        """Display-stretched RGB composite of a stitched well."""
        views, images = [], {}
        for p, img in planes_data:
            nz = img[img > 0]
            lo = int(np.percentile(nz, 1)) if nz.size else 0
            hi = max(int(np.percentile(nz, 99.5)) if nz.size else 255, lo + 1)
            color = p.color or CHANNEL_COLORS.get(p.channel, (200, 200, 200))
            views.append(render.ChannelView(p.key, color, lo=lo, hi=hi))
            images[p.key] = img
        return Image.fromarray(render.composite(views, images))

    def _write(self, wid, wt, res):
        planes_data = [(p, img) for (p, img) in res.values()]
        pixel_um = stitcher.tile_pixel_um(wt)
        # Names must stay in step with planned_outputs(), which is what the user
        # was shown and agreed to overwrite.
        if self.fmt in ("ometiff", "both", "ometiff_pdf"):
            stitcher.save_ome_tiff(
                self.out_dir / f"{self.base}_{wid}.ome.tif", planes_data, pixel_um)
        if self.fmt == "split":
            for p, img in planes_data:
                Image.fromarray(img).save(self.out_dir / f"{self.base}_{wid}_{p.label}.tif")
        if self.fmt in ("png", "both"):
            self._composite(planes_data).save(self.out_dir / f"{self.base}_{wid}.png")
        if self.fmt == "pdf_pages":
            im = self._composite(planes_data)
            self._label_and_save_pdf(im, wid, self.out_dir / f"{self.base}_{wid}.pdf")
        if self.fmt == "pdf_sheet_pages":
            # a page per well, appended after the overview sheet in ONE document
            self.well_pages.append((wid, self._page_for(self._composite(planes_data), wid)))
        if self.fmt in ("pdf_sheet", "pdf_sheet_pages", "ometiff_pdf"):
            # keep a downscaled panel; the full sheet is written once at the end
            im = self._composite(planes_data)
            im.thumbnail((self.SHEET_PANEL, self.SHEET_PANEL), Image.Resampling.LANCZOS)
            self.sheet_panels[wid] = im

    #: long edge of each well panel on the contact sheet
    SHEET_PANEL = 1100

    def _label_and_save_pdf(self, im, wid, path):
        """One well per page, with its conditions printed on the image."""
        page = self._page_for(im, wid)
        tmp = Path(str(path) + ".part")
        page.save(tmp, "PDF", resolution=300.0, quality=95, subsampling=0)
        tmp.replace(path)

    def _page_for(self, im, wid):
        """A single well rendered as a labelled page."""
        band = max(48, im.height // 24)
        page = Image.new("RGB", (im.width, im.height + band), (255, 255, 255))
        page.paste(im.convert("RGB"), (0, band))
        d = ImageDraw.Draw(page)
        f_hdr = _load_font_static(max(20, band // 2))
        d.text((10, band // 5), f"{self.exp_name}   {wid}   ({im.width}×{im.height})",
               fill=(0, 0, 0), font=f_hdr)
        lines = self.conditions_for(wid)
        if lines:
            _draw_caption_static(page, d, 16, band + 16, wid, lines,
                                 _load_font_static(max(22, band // 2)),
                                 _load_font_static(max(16, band // 3)), 12,
                                 page.width - 32)
        return page

    def _write_contact_sheet(self):
        """All wells on one page, in plate layout, with their conditions."""
        if not self.sheet_panels:
            return
        wells = sorted(self.sheet_panels)
        rows = sorted({w[0] for w in wells})
        cols = sorted({w[1:] for w in wells}, key=lambda c: int(c) if c.isdigit() else 0)
        cw = max(im.width for im in self.sheet_panels.values())
        ch = max(im.height for im in self.sheet_panels.values())
        k = cw / 380.0
        pad, hdr, title_h = int(12 * k), int(30 * k), int(48 * k)
        W = hdr + len(cols) * (cw + pad) + pad
        H = title_h + hdr + len(rows) * (ch + pad) + pad
        sheet = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        f_title, f_hdr = _load_font_static(int(26 * k)), _load_font_static(int(20 * k))
        f_lbl, f_cond = _load_font_static(int(17 * k)), _load_font_static(int(13 * k))
        d.text((pad, pad), f"{self.exp_name}  ·  {len(wells)} wells",
               fill=(0, 0, 0), font=f_title)
        for ci, col in enumerate(cols):
            d.text((hdr + ci * (cw + pad) + cw // 2, title_h + hdr // 2), col,
                   fill=(0, 0, 0), font=f_hdr, anchor="mm")
        for ri, row in enumerate(rows):
            y0 = title_h + hdr + ri * (ch + pad)
            d.text((hdr // 2, y0 + ch // 2), row, fill=(0, 0, 0), font=f_hdr, anchor="mm")
            for ci, col in enumerate(cols):
                wid = f"{row}{col}"
                if wid not in self.sheet_panels:
                    continue
                x0 = hdr + ci * (cw + pad)
                im = self.sheet_panels[wid]
                d.rectangle([x0, y0, x0 + cw, y0 + ch], fill=(10, 10, 10))
                sheet.paste(im.convert("RGB"),
                            (x0 + (cw - im.width) // 2, y0 + (ch - im.height) // 2))
                _draw_caption_static(sheet, d, x0 + int(6 * k), y0 + int(6 * k), wid,
                                     self.conditions_for(wid), f_lbl, f_cond,
                                     max(4, int(6 * k)), cw - int(12 * k))
        extra = [pg for _, pg in self.well_pages]
        name = (f"{self.base}_plate_and_wells.pdf" if extra
                else f"{self.base}_plate.pdf")
        path = self.out_dir / name
        tmp = Path(str(path) + ".part")
        sheet.save(tmp, "PDF", resolution=300.0, quality=95, subsampling=0,
                   save_all=bool(extra), append_images=extra)
        tmp.replace(path)
        self.well_pages.clear()


class LeftAffirmativeStyle(QProxyStyle):
    """Put Yes/OK on the LEFT of dialogs.

    macOS's native button layout puts the affirmative button on the right; this
    overrides just that style hint (appearance is otherwise untouched) so Yes/OK is
    consistently on the left, as requested.
    """

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.StyleHint.SH_DialogButtonLayout:
            return QDialogButtonBox.ButtonLayout.WinLayout.value
        return super().styleHint(hint, option, widget, returnData)


def _install_excepthook():
    """Show unexpected errors instead of losing them in a windowed process.

    A windowed (console-less) build otherwise gives the user no useful traceback.
    """
    import traceback

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        print(text, file=sys.stderr)
        try:
            QMessageBox.critical(
                None, f"{APP_NAME} — unexpected error",
                f"{exc_type.__name__}: {exc}\n\nThe app will keep running.")
        except Exception:
            pass

    sys.excepthook = hook


def _run_package_smoke(output_dir: Path) -> int:
    """Exercise lazy binary codecs inside a frozen app, then exit.

    PyInstaller cannot discover imagecodecs' importlib-based codec loading from
    static analysis.  CI launches the finished bundle with the private
    ``BZ_STUDIO_PACKAGE_SMOKE`` environment variable so a build that cannot
    decode LZW or write compressed OME-TIFF never reaches a public release.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "package-smoke.json"
    report = {
        "ok": False,
        "app": APP_NAME,
        "version": __version__,
        "qt_binding": "PySide6",
    }
    try:
        import imagecodecs
        import tifffile
        from PySide6 import QtCore

        if not QtCore.__version__:
            raise RuntimeError("PySide6 version is unavailable")
        report["qt_binding_version"] = QtCore.__version__
        report["qt_version"] = QtCore.qVersion()

        source = ((np.arange(64, dtype=np.uint16)[:, None] * 31
                   + np.arange(80, dtype=np.uint16)[None, :] * 17) % 4093)
        lzw_path = output_dir / "codec-lzw.tif"
        tifffile.imwrite(
            lzw_path, source, compression="lzw", predictor=True,
            photometric="minisblack")
        np.testing.assert_array_equal(tifffile.imread(lzw_path), source)

        ome_path = output_dir / "codec-zlib.ome.tif"
        channels = np.stack((source, np.flip(source, axis=1)))
        tifffile.imwrite(
            ome_path, channels, bigtiff=True, ome=True, compression="zlib",
            predictor=True, photometric="minisblack",
            metadata={"axes": "CYX", "Channel": {"Name": ["A", "B"]}})
        with tifffile.TiffFile(ome_path) as check:
            if not check.ome_metadata or not check.series:
                raise RuntimeError("OME metadata validation failed")
            np.testing.assert_array_equal(check.series[0].asarray(), channels)

        # Exercise the real atomic presentation path as well as scientific
        # TIFF codecs.  In particular this catches Windows builds where fsync
        # is attempted on a read-only descriptor.
        presentation_array = np.stack((
            (source % 256).astype(np.uint8),
            (np.flip(source, axis=1) % 256).astype(np.uint8),
            (np.flip(source, axis=0) % 256).astype(np.uint8),
        ), axis=2)
        presentation = Image.fromarray(presentation_array)
        png_path = output_dir / "presentation.png"
        pdf_path = output_dir / "presentation.pdf"
        mosaic_engine.atomic_save_presentation(presentation, png_path, "png")
        mosaic_engine.atomic_save_presentation(presentation, pdf_path, "pdf")
        with Image.open(png_path) as check:
            np.testing.assert_array_equal(np.asarray(check), presentation_array)
        pdf_data = pdf_path.read_bytes()
        if not pdf_data.startswith(b"%PDF-") or b"%%EOF" not in pdf_data[-4096:]:
            raise RuntimeError("presentation PDF validation failed")

        # Touch the functions explicitly so the test proves the delayed modules
        # themselves, rather than only tifffile's public wrapper, are bundled.
        if not callable(imagecodecs.lzw_decode) or not callable(imagecodecs.zlib_encode):
            raise RuntimeError("required imagecodecs functions are unavailable")
        report.update({
            "ok": True, "lzw": lzw_path.name, "ome": ome_path.name,
            "png": png_path.name, "pdf": pdf_path.name,
            "shape": list(channels.shape),
        })
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1


def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("KTFViewer")
    # Automated runs must never write into the real user settings — a test that
    # fills the Conditions table would otherwise persist against a real experiment.
    app.setApplicationName(
        SETTINGS_APP_NAME + " (test)" if os.environ.get("BZPS_TEST") else SETTINGS_APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setStyle(LeftAffirmativeStyle())   # Yes/OK on the left
    _install_excepthook()
    window = MainWindow()
    window.show()
    # Always ask which workflow to start in; the three paths need different discovery.
    QTimer.singleShot(0, window._show_start_chooser)
    sys.exit(app.exec())


if __name__ == "__main__":
    _smoke_dir = os.environ.get("BZ_STUDIO_PACKAGE_SMOKE")
    if _smoke_dir:
        raise SystemExit(_run_package_smoke(Path(_smoke_dir)))
    main()
