#!/usr/bin/env python3
"""KTF Viewer — desktop app for viewing .ktf microscope images.

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

# Make the app self-contained when run from source: point Qt at PyQt6's bundled
# platform plugins. Some Python environments (e.g. non-activated conda) don't set
# this, causing a "could not find the Qt platform plugin" crash on launch.
# Skip entirely when frozen (PyInstaller) — it configures Qt itself, and overriding
# the path there breaks plugin discovery.
if not getattr(sys, "frozen", False):
    try:
        import PyQt6
        _plugins = os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "plugins", "platforms")
        if os.path.isdir(_plugins):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _plugins)
    except Exception:
        pass

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Contact sheets at original resolution legitimately exceed PIL's bomb threshold.
Image.MAX_IMAGE_PIXELS = None

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeWidget, QTreeWidgetItem, QLabel, QScrollArea,
    QSlider, QGroupBox, QCheckBox, QPushButton, QToolButton,
    QFileDialog, QGridLayout, QSizePolicy, QTextEdit,
    QProgressBar, QColorDialog, QToolBar, QInputDialog, QLineEdit,
    QTabWidget, QTableWidget, QTableWidgetItem, QAbstractItemView, QMessageBox, QMenu,
    QProxyStyle, QStyle, QDialogButtonBox, QDialog, QComboBox,
)
from PyQt6.QtCore import (
    Qt, QSize, pyqtSignal, QPoint, QRect, QThread, QTimer, QPointF, QEvent, QSettings, QUrl,
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QAction, QWheelEvent,
    QMouseEvent, QPen, QFont, QKeySequence, QBrush, QDesktopServices,
)

import ktf_reader
import render
import stitcher
from version import __version__, APP_NAME

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

    view_changed = pyqtSignal()          # emitted after zoom/pan settles (debounced)
    cursor_moved = pyqtSignal(float, float)  # full-res image coords under cursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None                # overview composite pixmap
        self._overview_ds = 1              # full-res px per overview px
        self._full_w = 0
        self._full_h = 0
        self._um_per_px_full = 0.0

        self._detail_pixmap = None         # QPixmap covering a sub-rect at higher res
        self._detail_rect_full = None      # QRect in full-res image coords

        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._dragging = False
        self._drag_start = QPointF()
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

        # Editable zoom field, overlaid at the bottom-left corner.
        self.zoom_edit = QLineEdit(self)
        self.zoom_edit.setFixedWidth(66)
        self.zoom_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_edit.setToolTip("Zoom % — type a value and press Enter")
        self.zoom_edit.setStyleSheet(
            "QLineEdit { background: rgba(0,0,0,150); color:#ddd; border:1px solid #555;"
            " border-radius:4px; padding:1px; font-family:Menlo; font-size:11px; }")
        self.zoom_edit.editingFinished.connect(self._on_zoom_edit)
        self.zoom_edit.hide()
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
        self.zoom_edit.hide()
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
        p.fillRect(self.rect(), QColor(24, 24, 24))
        if not self._pixmap:
            p.setPen(QColor(110, 110, 110))
            p.setFont(QFont("Menlo", 13))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Open an experiment folder, then click a well\n\n"
                       "scroll = zoom   ·   drag = pan   ·   ⌘0 = fit")
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
        p.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
        p.drawText(QRect(int(x0 - 8), int(y - 24), int(length_px + 16), 18),
                   Qt.AlignmentFlag.AlignCenter, label)

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
            self._dragging = True
            self._drag_start = event.position() - self._pan
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._debounce.start()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        if self._dragging:
            self._pan = pos - self._drag_start
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
        self.zoom_edit.move(8, self.height() - self.zoom_edit.height() - 8)

    def _sync_zoom_field(self):
        if self._pixmap:
            self.zoom_edit.show()
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
    ready = pyqtSignal(object, object, int)  # QPixmap, (x0,y0,w,h), generation

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
            self.ready.emit(QPixmap.fromImage(qimg), (x0, y0, x1 - x0, y1 - y0), self.gen)
        except Exception as e:
            print(f"Detail load error: {e}")
            self.ready.emit(None, None, self.gen)


class LoadWorker(QThread):
    finished = pyqtSignal(str, object, int)  # ch_id, image, generation

    def __init__(self, path, ch_id, downsample=1, gen=0):
        super().__init__()
        self.path = path
        self.ch_id = ch_id
        self.downsample = downsample
        self.gen = gen

    def run(self):
        try:
            img = ktf_reader.reconstruct_image(self.path, downsample=self.downsample)
            self.finished.emit(self.ch_id, img, self.gen)
        except Exception as e:
            print(f"Error loading {self.path}: {e}")
            self.finished.emit(self.ch_id, None, self.gen)


class ChannelControl(QWidget):
    changed = pyqtSignal()

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
    well_clicked = pyqtSignal(str)

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
            lbl.setStyleSheet("color:#aaa; font-weight:bold;")
            self._layout.addWidget(lbl, 0, ci + 1)
        for ri, row in enumerate(rows):
            lbl = QLabel(row); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#aaa; font-weight:bold;")
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
                    QPushButton { background:#2a2a2a; border:1px dashed #666; border-radius:4px;
                                  color:#ddd; font-size:10px; }
                    QPushButton:hover { border-color:#88f; background:#333; }
                    QPushButton:checked { border:2px solid #6a9; background:#26332e; }
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
            lbl.setStyleSheet("color:#aaa; font-weight:bold;")
            self._layout.addWidget(lbl, 0, ci + 1)
        for ri, row in enumerate(rows):
            lbl = QLabel(row)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#aaa; font-weight:bold;")
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
            QPushButton { background:#2a2a2a; border:1px solid #555; border-radius:4px; color:white; font-size:11px; }
            QPushButton:hover { border-color:#88f; background:#333; }
        """)
        first = list(channels.values())[0]
        if first.thumbnail_jpeg:
            qimg = QImage.fromData(first.thumbnail_jpeg)
            if not qimg.isNull():
                pix = QPixmap.fromImage(qimg).scaled(
                    110, 80, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                btn.setIcon(pix)
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

    edited = pyqtSignal()
    DEFAULT_HEADERS = ["Well", "Sample", "Treatment", "Conc.", "Time", "Notes"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._suppress = False
        self.setColumnCount(len(self.DEFAULT_HEADERS))
        self.setHorizontalHeaderLabels(self.DEFAULT_HEADERS)
        self.verticalHeader().setVisible(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.setStyleSheet("""
            QTableWidget { background:#252525; color:#ddd; gridline-color:#3a3a3a;
                           selection-background-color:#3a5a8a; }
            QHeaderView::section { background:#333; color:#bbb; border:1px solid #444; padding:3px; }
            QTableWidget QLineEdit { background:#1e1e1e; color:#fff; }
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
            well_item.setForeground(QColor(150, 200, 255))
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
        self._mode = None             # None | "ktf" | "raw"
        self._raw_experiment = None   # raw model, kept separate from _experiment
        self._current_raw_well = None
        self._scan_worker = None
        self._pending_root = None     # committed to settings only after a good load
        self._exporting = False       # a synchronous export is running
        self._stitch_worker = None    # asynchronous stitch (own lifetime)
        self._current_composite = None

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._apply_style()

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
        tg = QVBoxLayout(tree_group)
        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderLabels(["Name", "Wells", "Channels"])
        self.folder_tree.setColumnWidth(0, 200)
        self.folder_tree.itemDoubleClicked.connect(self._on_experiment_selected)
        tg.addWidget(self.folder_tree)
        ll.addWidget(tree_group)
        well_group = QGroupBox("Well Plate")
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
        cond_hint.setStyleSheet("color:#888; font-size:10px;")
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
        cond_btns.addStretch()
        b_csv = QPushButton("Export CSV")
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
        self.readout.setStyleSheet("color:#9cf; padding:2px 6px;")
        self.readout.setFixedHeight(22)
        rl.addWidget(self.readout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        rl.addWidget(self.progress_bar)

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
        eg = QVBoxLayout(export_group)
        b1 = QPushButton("Export PNG (view)")
        b1.clicked.connect(lambda: self._export("png"))
        b2 = QPushButton("Export TIFF (full res)")
        b2.clicked.connect(lambda: self._export("tiff"))
        b3 = QPushButton("Export All Wells (TIFF)")
        b3.clicked.connect(self._export_all_wells)
        b4 = QPushButton("Export Plate to PDF")
        b4.setToolTip("All wells arranged as a contact sheet in one PDF (quality selectable)")
        b4.clicked.connect(self._export_plate_pdf)
        b5 = QPushButton("Stitch Raw Tiles…")
        b5.setToolTip("Rebuild whole-well mosaics from the raw tiles "
                      "(no vendor software needed)")
        b5.clicked.connect(self._stitch_raw_tiles)
        for b in (b1, b2, b3, b4, b5):
            eg.addWidget(b)
        eg.addStretch()
        bl.addWidget(export_group)

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
        hint.setWordWrap(True); hint.setStyleSheet("color:#888; font-size:11px;")
        rp.addWidget(hint)
        self.btn_stitch_raw = QPushButton("Stitch Raw Tiles…")
        self.btn_stitch_raw.clicked.connect(self._stitch_raw_tiles)
        rp.addWidget(self.btn_stitch_raw)
        rp.addStretch()
        self.raw_panel.setMaximumHeight(200)
        self.raw_panel.setVisible(False)
        rl.addWidget(self.raw_panel, stretch=0)
        splitter.addWidget(right)
        splitter.setSizes([380, 1120])
        self.statusBar().showMessage("Open an experiment folder to begin")

    def _setup_menu(self):
        menu = self.menuBar()
        fm = menu.addMenu("File")
        a = QAction("Choose Workflow…", self)
        a.setStatusTip("Switch between .ktf viewing and raw-tile stitching")
        a.triggered.connect(self._show_start_chooser); fm.addAction(a)
        self.act_workflow = a
        a = QAction("Open Folder...", self); a.setShortcut(QKeySequence("Ctrl+O"))
        a.setStatusTip("Open a folder for the current workflow")
        a.triggered.connect(self._open_folder); fm.addAction(a)
        self.act_open = a
        fm.addSeparator()
        a = QAction("Quit", self); a.setShortcut(QKeySequence("Ctrl+Q"))
        a.triggered.connect(self.close); fm.addAction(a)
        hm = menu.addMenu("Help")
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

    def _setup_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        for label, tip, slot in [
            ("Open", "Open experiment folder (⌘O)", self._open_folder),
            ("Fit", "Fit image in view (⌘0)", self.canvas.fit_in_view),
            ("100%", "Actual pixels (⌘1)", self.canvas.zoom_actual_pixels),
            ("Auto B/C", "Auto brightness/contrast (⇧⌘A)", self._auto_contrast_all),
        ]:
            act = QAction(label, self)
            act.setToolTip(tip)
            act.triggered.connect(slot)
            tb.addAction(act)

    def _about(self):
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> {__version__}<br><br>"
            "BZ-X (.ktf) viewer and tile stitcher.<br>"
            "&copy; 2026 yoshi-koba-lab — All Rights Reserved.<br><br>"
            "<a href='https://github.com/yoshi-koba-lab/ktf-viewer'>"
            "github.com/yoshi-koba-lab/ktf-viewer</a>")

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#1e1e1e; color:#ddd; }
            QGroupBox { border:1px solid #444; border-radius:4px; margin-top:8px; padding-top:12px; font-weight:bold; color:#aaa; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; }
            QTreeWidget, QTextEdit { background:#252525; border:1px solid #333; }
            QTreeWidget { selection-background-color:#3a5a8a; }
            QSlider::groove:horizontal { height:4px; background:#444; border-radius:2px; }
            QSlider::handle:horizontal { width:11px; height:11px; margin:-4px 0; background:#999; border-radius:5px; }
            QPushButton { background:#3a3a3a; border:1px solid #555; border-radius:4px; padding:6px 10px; color:#ddd; }
            QPushButton:hover { background:#4a4a4a; border-color:#88f; }
            QToolButton { background:#333; border:1px solid #555; border-radius:3px; color:#ddd; }
            QToolButton:checked { background:#c84; border-color:#fa6; }
            QStatusBar { background:#252525; color:#888; }
            QProgressBar { background:#333; border:none; }
            QProgressBar::chunk { background:#58f; }
        """)

    # ---------- scanning ----------
    # ---------- workflow selection ----------
    def _recent(self, mode):
        s = QSettings()
        if mode == StartModeDialog.KTF:
            return s.value("last_ktf_root", "") or s.value("last_root", "") or ""
        return s.value("last_raw_root", "") or ""

    def _show_start_chooser(self):
        if self._busy:
            self.statusBar().showMessage("処理中です — 完了までお待ちください。")
            return
        dlg = StartModeDialog(self, self._recent(StartModeDialog.KTF),
                              self._recent(StartModeDialog.RAW))
        dlg.setStyleSheet("")
        if dlg.exec() and dlg.choice:
            self._choose_folder(dlg.choice)
        elif self._mode is None:
            self.statusBar().showMessage(
                "File ▸ Choose Workflow… から開始してください")

    def _choose_folder(self, mode):
        start = self._recent(mode) or str(Path.home())
        if not Path(start).is_dir():
            start = str(Path.home())
        title = ("Open a .ktf experiment folder (or any folder containing them)"
                 if mode == StartModeDialog.KTF else
                 "Open a raw-tile experiment folder (or any folder containing them)")
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
        self._set_actions_enabled(False)
        self._scan_worker = ScanWorker(folder, mode)
        self._scan_worker.progress.connect(
            lambda d, n: self.statusBar().showMessage(f"スキャン中 ({n} フォルダ): {d}"))
        self._scan_worker.finished_scan.connect(self._on_scan_done)
        self._scan_worker.start()

    def _set_actions_enabled(self, on: bool):
        for a in (getattr(self, "act_open", None), getattr(self, "act_workflow", None)):
            if a is not None:
                a.setEnabled(on)

    def _on_scan_done(self, dirs, errors):
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 1000)
        self._set_actions_enabled(True)
        folder, mode = self._pending_root or (None, None)
        if dirs is None:                      # cancelled
            self.statusBar().showMessage("スキャンを中止しました。")
            return
        label = ".ktf" if mode == StartModeDialog.KTF else "生画像"
        if not dirs:
            self.statusBar().showMessage(
                f"“{folder.name}” の下に {label} の実験が見つかりませんでした"
                + (f"（{errors} 件のフォルダを読めませんでした）" if errors else ""))
            QMessageBox.information(
                self, "実験が見つかりません",
                f"“{folder}” の下に{label}形式の実験はありませんでした。\n\n"
                "別のフォルダを選ぶか、File ▸ Choose Workflow… で"
                "もう一方のワークフローをお試しください。")
            return
        self._set_mode(mode)
        self.folder_tree.clear()
        if mode == StartModeDialog.KTF:
            self._populate_tree(folder, dirs)
        else:
            self._populate_raw_tree(folder, dirs)
        if len(dirs) == 1:
            self._select_tree_item(dirs[0])
            self._on_experiment_selected(self.folder_tree.currentItem(), 0)

    def _set_mode(self, mode):
        self._mode = mode
        raw = mode == StartModeDialog.RAW
        self.folder_tree.setHeaderLabels(
            ["Name", "Wells", "Source"] if raw else ["Name", "Wells", "Channels"])
        # Conditions apply to both workflows; they are keyed by experiment path, and
        # the table is repopulated per experiment so values can never leak across.
        if hasattr(self, "well_tabs") and self.well_tabs.indexOf(self.cond_tab) < 0:
            self.well_tabs.addTab(self.cond_tab, "Conditions")
        if hasattr(self, "raw_panel"):
            self.raw_panel.setVisible(raw)
        if hasattr(self, "ktf_bottom"):
            self.ktf_bottom.setVisible(not raw)
        self.setWindowTitle(f"{APP_NAME} {__version__} — "
                            + ("生画像（スティッチング）" if raw else ".ktf 表示"))

    def _commit_root(self):
        """Remember the browse root only once something actually loaded."""
        if not self._pending_root:
            return
        folder, mode = self._pending_root
        key = "last_ktf_root" if mode == StartModeDialog.KTF else "last_raw_root"
        QSettings().setValue(key, str(folder))

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
        self._experiment = None            # the two models never coexist
        self._reset_well_state()
        self._raw_experiment = {"name": folder.name, "path": folder, "wells": wells}
        self._current_raw_well = None
        self.well_plate.set_raw_wells(wells)
        saved = self._load_conditions()
        self.conditions.set_wells(sorted(wells), saved)
        unusable = structural - set(wells)
        msg = (f"{folder.name}: {len(wells)} ウェル（生画像）— "
               f"ウェルを選ぶか、そのまま Stitch Raw Tiles… で全ウェルを処理できます")
        if unusable:
            msg += f" / 使用不可: {', '.join(sorted(unusable))}"
        self.statusBar().showMessage(msg)
        self._update_raw_summary()
        self._commit_root()
        return True

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
        else:
            self._on_well_clicked(well_id)

    def _select_tree_item(self, path: Path):
        """Highlight the tree row whose stored path matches `path`."""
        target = str(path)
        stack = [self.folder_tree.topLevelItem(i)
                 for i in range(self.folder_tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item is None:
                continue
            if item.data(0, Qt.ItemDataRole.UserRole) == target:
                self.folder_tree.setCurrentItem(item)
                self.folder_tree.scrollToItem(item)
                return
            stack.extend(item.child(i) for i in range(item.childCount()))

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
        else:
            self._load_experiment(Path(path))

    def _load_experiment(self, folder: Path):
        if self._busy:
            self.statusBar().showMessage("Another operation is running — please wait.")
            return
        self.statusBar().showMessage(f"Loading {folder.name}...")
        QApplication.processEvents()
        try:
            experiment = ktf_reader.scan_experiment_folder(folder)
        except Exception as e:
            self.statusBar().showMessage(f"Could not read “{folder.name}”: {e}")
            return
        self._experiment = experiment
        # A new experiment invalidates everything tied to the previous well.
        self._reset_well_state()
        self.well_plate.set_wells(experiment["wells"])
        # populate the sample-conditions table (restore any saved values)
        saved = self._load_conditions()
        self.conditions.set_wells(sorted(experiment["wells"]), saved)

        msg = f"{experiment['name']}: {len(experiment['wells'])} wells"
        skipped = experiment.get("errors") or []
        if skipped:
            msg += f" — skipped {len(skipped)} unreadable file(s): " + \
                   ", ".join(n for n, _ in skipped[:3]) + ("…" if len(skipped) > 3 else "")
        if not experiment["wells"]:
            msg = f"No readable wells in “{experiment['name']}”" + \
                  (f" ({len(skipped)} file(s) unreadable)" if skipped else "")
        else:
            self._raw_experiment = None       # the two models never coexist
            self._commit_root()
        self.statusBar().showMessage(msg)

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

    # ---------- sample conditions ----------
    def _active_experiment(self):
        """Whichever experiment is loaded — conditions belong to both workflows."""
        return self._experiment or self._raw_experiment

    def _conditions_key(self):
        exp = self._active_experiment()
        p = exp["path"] if exp else None
        return f"conditions/{p}" if p else None

    def _load_conditions(self) -> dict:
        key = self._conditions_key()
        if not key:
            return {}
        raw = QSettings().value(key, "")
        try:
            return json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            return {}

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

    def _export_conditions_csv(self):
        exp = self._active_experiment()
        if not exp:
            return
        default = f"{exp['name']}_conditions.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export conditions CSV", default, "CSV (*.csv)")
        if not path:
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
            w.finished.connect(self._on_channel_loaded)
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

    def _refresh_detail(self):
        """Load a sharper composite for the current viewport when zoomed in."""
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

    def _on_detail_ready(self, pixmap, rect_full, gen):
        if gen != self._gen or pixmap is None:
            return  # stale or failed; keep the scaled overview
        self.canvas.set_detail(pixmap, rect_full)

    def _on_detail_finished(self):
        # If the viewport moved while this worker ran, render the latest view now.
        if self._detail_pending:
            self._detail_pending = False
            self._refresh_detail()

    # ---------- readout ----------
    def _on_cursor(self, fx, fy):
        if self._full_dims == (0, 0):
            return
        fw, fh = self._full_dims
        if not (0 <= fx < fw and 0 <= fy < fh):
            self.readout.setText("")
            return
        parts = [f"px ({int(fx)}, {int(fy)})"]
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
        for ch_id, ctrl in self._channel_controls.items():
            if ch_id in self._channel_images:
                ctrl.auto_contrast(self._channel_images[ch_id])
        self._rebuild_overview(reset_view=False)
        self._refresh_detail()

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
            path, _ = QFileDialog.getSaveFileName(
                self, "Export TIFF", f"{self._current_well}.tif", "TIFF (*.tif)")
            if not path:
                return
            self.statusBar().showMessage("Exporting full-resolution TIFF per channel...")
            self._exporting = True
            QApplication.processEvents()
            try:
                for ch_id, info in wells[self._current_well].items():
                    img = ktf_reader.reconstruct_image(info.path, downsample=1)
                    out = Path(path).with_name(f"{Path(path).stem}_{ch_id}.tif")
                    Image.fromarray(img).save(str(out))
                self.statusBar().showMessage(f"Exported full-res channels to {Path(path).parent}")
            except Exception as e:
                self.statusBar().showMessage(f"TIFF export failed: {e}")
            finally:
                self._exporting = False
        else:
            if not hasattr(self, "_channel_images") or not self._channel_images:
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "Export PNG", f"{self._current_well}.png", "PNG (*.png)")
            if not path:
                return
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
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return
        self._exporting = True
        out = Path(folder)
        wells = self._experiment["wells"]
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
                        Image.fromarray(img).save(str(out / f"{wid}_{ch_id}.tif"))
                    except Exception as e:
                        failed += 1
                        print(f"Export error {wid}/{ch_id}: {e}")
                    done += 1
                    self.progress_bar.setValue(done)
        finally:
            self._exporting = False
            self.progress_bar.hide()
        msg = f"Exported {done - failed}/{total} images to {folder}"
        self.statusBar().showMessage(msg + (f" — {failed} failed" if failed else ""))

    # ---------- busy state ----------
    @property
    def _busy(self) -> bool:
        """True while any long operation owns the data (export or stitch)."""
        return self._exporting or (
            self._stitch_worker is not None and self._stitch_worker.isRunning())

    def closeEvent(self, event):
        """Don't let a half-written mosaic be left behind on quit."""
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
        event.accept()

    # ---------- stitching raw tiles ----------
    def _stitch_raw_tiles(self):
        """Stitch the loaded RAW experiment. Never touches the .ktf model."""
        if self._busy:
            self.statusBar().showMessage("処理中です — 完了までお待ちください。")
            return
        if not self._raw_experiment or not self._raw_experiment.get("wells"):
            self.statusBar().showMessage(
                "生画像の実験を開いてください（File ▸ Choose Workflow… ▸ 生画像から始める）")
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

        out = QFileDialog.getExistingDirectory(self, "Save stitched images to")
        if not out:
            return

        self._set_actions_enabled(False)
        self.btn_stitch_raw.setEnabled(False)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self._stitch_worker = StitchWorker(wells, out, dlg.z_mode, dlg.fmt,
                                           flatfield=dlg.flatfield,
                                           subpixel=dlg.subpixel)
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
            body += ("\n\n「alignment uncertain」は重なりが小さすぎるか特徴が乏しく"
                     "測定できなかったことを示します — 該当ウェルを確認してください。")
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
            if info.thumbnail_jpeg:
                try:
                    im = Image.open(io.BytesIO(info.thumbnail_jpeg)).convert("L")
                    thumbs[ch_id] = np.array(im)
                except Exception:
                    pass
        if not thumbs:
            return None
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
            except Exception:
                continue
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
            return None
        return Image.fromarray(render.composite(views, images))

    # ---------- per-well caption from the Conditions tab ----------
    def _conditions_snapshot(self):
        """Current Conditions table as ({well: [values]}, headers)."""
        data = self.conditions.to_dict()
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

    def _export_plate_pdf(self):
        if not self._experiment or not self._experiment["wells"] or self._busy:
            return
        wells = self._experiment["wells"]

        labels = list(self.PDF_QUALITY.keys())
        # Build the picker manually and clear its stylesheet so it uses the native
        # (readable) look instead of inheriting the app's dark theme.
        dlg = QInputDialog(self)
        dlg.setStyleSheet("")
        dlg.setWindowTitle("Plate PDF quality")
        dlg.setLabelText("Higher quality is sharper but reads the .ktf files and takes longer:")
        dlg.setComboBoxItems(labels)
        dlg.setTextValue(labels[2])  # default: High
        dlg.setOption(QInputDialog.InputDialogOption.UseListViewForComboBoxItems, True)
        if not dlg.exec():
            return
        choice = dlg.textValue()
        source, panel, dpi, layout = self.PDF_QUALITY[choice]

        default_name = f"{self._experiment['name']}_plate.pdf"
        path, _ = QFileDialog.getSaveFileName(self, "Export Plate to PDF", default_name, "PDF (*.pdf)")
        if not path:
            return

        settings = {c: ctrl.to_view() for c, ctrl in self._channel_controls.items()}
        exp_name = self._experiment["name"]      # snapshot: processEvents() runs below
        self._exporting = True
        try:
            if layout == "pages":
                self._export_pdf_pages(wells, settings, dpi, path, choice, exp_name)
                return
            sheet = self._build_plate_sheet(wells, settings, source, panel, exp_name,
                                            with_pages=(layout == "both"))
            if sheet is None:
                return
            # Pages are yielded lazily so only one full-resolution well is held at a
            # time on top of the sheet.
            extra = (self._iter_well_pages(wells, settings, exp_name)
                     if layout == "both" else [])
            try:
                sheet.save(path, "PDF", resolution=float(dpi), quality=95, subsampling=0,
                           save_all=(layout == "both"), append_images=extra)
                self.statusBar().showMessage(
                    f"Exported plate PDF ({len(wells)} wells, {choice.split(' —')[0]}"
                    + (" + per-well pages" if layout == "both" else "") + f") to {path}")
            except MemoryError:
                self.statusBar().showMessage(
                    "Out of memory writing the PDF — try “one well per page”.")
            except Exception as e:
                self.statusBar().showMessage(f"PDF export failed: {e}")
        finally:
            self._exporting = False
            self.progress_bar.hide()

    def _build_plate_sheet(self, wells, settings, source, panel, exp_name, with_pages=False):
        """Render every well into one contact sheet. Returns the image, or None."""
        rows = sorted(set(w[0] for w in wells))
        cols = sorted(set(w[1:] for w in wells), key=lambda c: int(c) if c.isdigit() else 0)
        cond, headers = self._conditions_snapshot()

        native = panel == 0
        if native:                       # "all wells on one sheet" at original size
            panel = self._max_well_dim(wells)
        cell_w = panel
        cell_h = max(1, int(round(panel * self._well_aspect(wells))))

        pad, hdr, title_h = 12, 30, 48
        # scale chrome (padding / headers / fonts) with panel size so large panels
        # don't get tiny labels
        k = panel / 380.0
        pad, hdr, title_h = int(pad * k), int(hdr * k), int(title_h * k)
        grid_w = hdr + len(cols) * (cell_w + pad) + pad
        grid_h = title_h + hdr + len(rows) * (cell_h + pad) + pad

        est_bytes = grid_w * grid_h * 3
        if est_bytes > 1_000_000_000:
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
                self._draw_well_caption(
                    canvas, draw, x0 + int(6 * k), y0 + int(6 * k), wid,
                    self._caption_lines(wid, cond, headers),
                    f_lbl, f_cond, max(4, int(6 * k)), cell_w - int(12 * k))
                done += 1
                self.progress_bar.setValue(done)
        return canvas

    def _iter_well_pages(self, wells, settings, exp_name, order=None):
        """Yield one original-resolution page per well (lazy: one page in memory)."""
        order = order if order is not None else sorted(wells)
        total = len(order)
        f_lbl = self._load_font(40)
        f_id = self._load_font(46)
        f_cond = self._load_font(34)
        cond, headers = self._conditions_snapshot()
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        for i, wid in enumerate(order):
            self.statusBar().showMessage(
                f"Rendering {wid} at full resolution ({i + 1}/{total})...")
            QApplication.processEvents()
            im = self._render_well_full(wells[wid], settings, 0)  # 0 = native
            if im is None:
                continue
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

    def _export_pdf_pages(self, wells, settings, dpi, path, choice, exp_name):
        """Maximum quality: one well per page at the file's original resolution."""
        pages = self._iter_well_pages(wells, settings, exp_name)
        try:
            first = next(pages)
        except StopIteration:
            self.statusBar().showMessage("Nothing to export.")
            return
        try:
            first.save(path, "PDF", resolution=float(dpi),
                       save_all=True, append_images=pages,
                       quality=100, subsampling=0)
            self.statusBar().showMessage(
                f"Exported full-resolution PDF ({len(wells)} wells, one per page) to {path} "
                f"— PDF images are JPEG-encoded; use Export TIFF for lossless")
        except MemoryError:
            self.statusBar().showMessage(
                "Out of memory at original resolution — try “Ultra” instead.")
        except Exception as e:
            self.statusBar().showMessage(f"PDF export failed: {e}")


class StartModeDialog(QDialog):
    """Which of the two workflows to start in.

    The two paths need different discovery: a `.ktf` experiment has the mosaics
    already, a raw capture folder has only per-field tiles. Inferring the mode
    from a folder was the old behaviour and it silently hid raw folders, so the
    choice is explicit.
    """

    KTF, RAW = "ktf", "raw"

    def __init__(self, parent=None, last_ktf="", last_raw=""):
        super().__init__(parent)
        self.setWindowTitle("KTF Viewer — start")
        self.setMinimumWidth(560)
        self.choice = None
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        head = QLabel("<b>どちらから始めますか？</b>")
        lay.addWidget(head)

        for mode, title, desc, last, btn_text in [
            (self.KTF, ".ktf から始める（貼り合わせ済み）",
             "顕微鏡が出力した .ktf のモザイクを開いて表示・書き出します。", last_ktf,
             ".ktf フォルダを選ぶ…"),
            (self.RAW, "生画像から始める（未貼り合わせ）",
             "各視野の X###Y### タイルを読み込み、貼り合わせ（スティッチング）します。",
             last_raw, "生画像フォルダを選ぶ…"),
        ]:
            box = QGroupBox(title)
            v = QVBoxLayout(box)
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setStyleSheet("color:#bbb;")
            v.addWidget(d)
            if last:
                p = QLabel(f"前回: {last}")
                p.setStyleSheet("color:#888; font-size:10px;")
                p.setWordWrap(True)
                v.addWidget(p)
            b = QPushButton(btn_text)
            b.clicked.connect(lambda _, m=mode: self._pick(m))
            v.addWidget(b)
            lay.addWidget(box)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _pick(self, mode):
        self.choice = mode
        self.accept()


class ScanWorker(QThread):
    """Finds candidate experiments off the GUI thread (a drive root can be huge)."""

    progress = pyqtSignal(str, int)      # current dir, dirs examined
    finished_scan = pyqtSignal(object, int)   # list[Path], unreadable-dir count

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
                else:
                    if _is_raw_experiment(d, dirnames):
                        found.append(d)
                        dirnames[:] = []      # its wells/fields are not experiments
        except Exception:
            errors += 1
        self.finished_scan.emit(sorted(found), errors)


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
        ])
        form.addWidget(self.cmb_fmt, r, 1); r += 1
        lay.addLayout(form)

        self.chk_flat = QCheckBox("Correct illumination (flat-field) — needed for quantification")
        self.chk_flat.setChecked(True)
        self.chk_flat.setToolTip(
            "Divides out the vignetting, estimated per channel from the tiles.\n"
            "Without it a uniform specimen reads ±40 grey levels depending only\n"
            "on where in the field it fell.")
        lay.addWidget(self.chk_flat)
        self.chk_sub = QCheckBox("Sub-pixel seam check (slower, writes stitch_qc.csv)")
        self.chk_sub.setChecked(True)
        lay.addWidget(self.chk_sub)

        note = QLabel("Tile offsets are measured from the images themselves, so all "
                      "channels stay registered. Output opens in Fiji / QuPath / napari.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888; font-size:11px;")
        lay.addWidget(note)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                               QDialogButtonBox.StandardButton.Cancel)
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Choose folder…")
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
        return ["both", "ometiff", "png", "split"][self.cmb_fmt.currentIndex()]

    @property
    def flatfield(self):
        return self.chk_flat.isChecked()

    @property
    def subpixel(self):
        return self.chk_sub.isChecked()


class StitchWorker(QThread):
    """Runs stitching off the GUI thread."""
    progress = pyqtSignal(str, float)          # message, 0..1
    done = pyqtSignal(int, int, str)           # ok, failed, out_dir

    def __init__(self, wells, out_dir, z_mode, fmt, flatfield=True, subpixel=True):
        super().__init__()
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
                self.warnings.append(f"{wid}: {e}")
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
                self.warnings.append(f"{wid}: {e}")
                failed += 1

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
            with open(self.out_dir / "stitch_qc.csv", "w", newline="",
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

    def _write(self, wid, wt, res):
        planes_data = [(p, img) for (p, img) in res.values()]
        pixel_um = stitcher.tile_pixel_um(wt)
        if self.fmt in ("ometiff", "both"):
            stitcher.save_ome_tiff(self.out_dir / f"{wid}.ome.tif", planes_data, pixel_um)
        if self.fmt == "split":
            for p, img in planes_data:
                Image.fromarray(img).save(self.out_dir / f"{wid}_{p.label}.tif")
        if self.fmt in ("png", "both"):
            views, images = [], {}
            for p, img in planes_data:
                nz = img[img > 0]
                lo = int(np.percentile(nz, 1)) if nz.size else 0
                hi = max(int(np.percentile(nz, 99.5)) if nz.size else 255, lo + 1)
                color = p.color or CHANNEL_COLORS.get(p.channel, (200, 200, 200))
                views.append(render.ChannelView(p.key, color, lo=lo, hi=hi))
                images[p.key] = img
            Image.fromarray(render.composite(views, images)).save(
                self.out_dir / f"{wid}.png")


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
    """Show unexpected errors instead of letting PyQt abort the process.

    An uncaught exception inside a Qt slot makes PyQt6 call qFatal(), which kills a
    windowed (console-less) build with no message at all.
    """
    import traceback

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        print(text, file=sys.stderr)
        try:
            QMessageBox.critical(
                None, "KTF Viewer — unexpected error",
                f"{exc_type.__name__}: {exc}\n\nThe app will keep running.")
        except Exception:
            pass

    sys.excepthook = hook


def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("KTFViewer")
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setStyle(LeftAffirmativeStyle())   # Yes/OK on the left
    _install_excepthook()
    window = MainWindow()
    window.show()
    # Always ask which workflow to start in; the two paths need different discovery.
    QTimer.singleShot(0, window._show_start_chooser)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
