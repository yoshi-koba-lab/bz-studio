"""Registration, rendering, and export for wide-area XY image sets.

The plate stitcher in :mod:`stitcher` deliberately models a small, regular
well grid.  Wide tissue scans have a different contract: GCI metadata supplies
the topology and stage positions, while every informative neighbour overlap
contributes an image-derived correction.  This module keeps those concerns
separate and gives every tile its own solved position.

Nothing in this module imports Qt.  The GUI only supplies immutable settings,
progress callbacks, and cancellation checks.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mosaic


Position = tuple[int, int]
Progress = Optional[Callable[[str, float], None]]
Cancel = Optional[Callable[[], bool]]


class MosaicCancelled(RuntimeError):
    """Raised internally when a long-running mosaic job is cancelled."""


class MosaicExportError(RuntimeError):
    """Raised when a mosaic cannot be exported without risking partial data."""


@dataclass(frozen=True)
class RegistrationEdge:
    """One expected row/column adjacency and its measured translation."""

    tile_a: Position
    tile_b: Position
    axis: str
    stage_delta_yx: tuple[float, float]
    shift_yx: tuple[float, float]
    correction_yx: tuple[float, float]
    ncc: float
    psr: float
    peak_margin: float
    texture: float
    accepted: bool
    reason: str = ""
    weight: float = 0.0
    residual_yx: Optional[tuple[float, float]] = None


@dataclass
class RadiometricModel:
    """Conservative per-tile affine compensation for feathered display output."""

    channel_key: str
    gains: dict[Position, float]
    offsets: dict[Position, float]
    status: str = "identity"
    mode: str = "identity"
    attempted_edges: int = 0
    accepted_edges: int = 0
    components: int = 0
    raw_seam_bias: Optional[float] = None
    corrected_seam_bias: Optional[float] = None
    raw_seam_mad: Optional[float] = None
    corrected_seam_mad: Optional[float] = None
    raw_seam_p95: Optional[float] = None
    corrected_seam_p95: Optional[float] = None
    validation_edges: int = 0
    validation_mode: str = "none"
    new_clip_fraction: float = 0.0
    orientation_qc: dict[str, dict] = field(default_factory=dict)
    edge_qc: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "algorithm": "tilewise-affine-seam/1.0",
            "channel": self.channel_key,
            "status": self.status,
            "mode": self.mode,
            "attempted_edges": self.attempted_edges,
            "accepted_edges": self.accepted_edges,
            "components": self.components,
            "raw_seam_bias_median_dn": self.raw_seam_bias,
            "corrected_seam_bias_median_dn": self.corrected_seam_bias,
            "raw_seam_mad_median_dn": self.raw_seam_mad,
            "corrected_seam_mad_median_dn": self.corrected_seam_mad,
            "raw_seam_p95_median_dn": self.raw_seam_p95,
            "corrected_seam_p95_median_dn": self.corrected_seam_p95,
            "validation_edges": self.validation_edges,
            "validation_mode": self.validation_mode,
            "new_clip_fraction": self.new_clip_fraction,
            "orientation_qc": self.orientation_qc,
            "warning": (
                "Tilewise gain/offset compensation only; spatial vignetting and "
                "flat-field shading are not corrected."
            ),
            "warnings": list(self.warnings),
            "tiles": [
                {"row": p[0], "column": p[1], "gain": self.gains[p],
                 "offset_dn": self.offsets[p]}
                for p in sorted(self.gains)
            ],
            "edges": list(self.edge_qc),
        }


@dataclass
class SpatialFieldModel:
    """Conservative common additive detector field for feathered output.

    ``field_dn`` is intentionally omitted from the JSON sidecar.  It is an
    implementation detail rather than a calibrated flat-field image; the
    compact QC below records the independent evidence used to accept it.
    """

    channel_key: str
    field_dn: Optional[np.ndarray] = field(default=None, repr=False)
    status: str = "identity"
    downsample: int = 1
    split_correlation_row: Optional[float] = None
    split_correlation_column: Optional[float] = None
    split_disagreement_row: Optional[float] = None
    split_disagreement_column: Optional[float] = None
    field_p01_dn: Optional[float] = None
    field_p99_dn: Optional[float] = None
    channel_dynamic_range_dn: Optional[float] = None
    raw_heldout_bias_dn: Optional[float] = None
    corrected_heldout_bias_dn: Optional[float] = None
    raw_heldout_mad_dn: Optional[float] = None
    corrected_heldout_mad_dn: Optional[float] = None
    heldout_edges: int = 0
    new_clip_fraction: float = 0.0
    orientation_qc: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "algorithm": "common-additive-low-frequency-field/1.0",
            "channel": self.channel_key,
            "status": self.status,
            "downsample": self.downsample,
            "split_correlation_row": self.split_correlation_row,
            "split_correlation_column": self.split_correlation_column,
            "split_disagreement_row": self.split_disagreement_row,
            "split_disagreement_column": self.split_disagreement_column,
            "field_p01_dn": self.field_p01_dn,
            "field_p99_dn": self.field_p99_dn,
            "channel_dynamic_range_dn": self.channel_dynamic_range_dn,
            "heldout_edges": self.heldout_edges,
            "raw_heldout_bias_dn": self.raw_heldout_bias_dn,
            "corrected_heldout_bias_dn": self.corrected_heldout_bias_dn,
            "raw_heldout_mad_dn": self.raw_heldout_mad_dn,
            "corrected_heldout_mad_dn": self.corrected_heldout_mad_dn,
            "new_clip_fraction": self.new_clip_fraction,
            "orientation_qc": self.orientation_qc,
            "warning": (
                "Estimated presentation-only additive shading field; this is "
                "not a calibrated flat-field and does not correct multiplicative "
                "vignetting."
            ),
            "warnings": list(self.warnings),
        }


@dataclass
class MosaicGeometry:
    """Solved per-tile geometry and the evidence used to obtain it."""

    origins: dict[Position, tuple[float, float]]
    stage_origins: dict[Position, tuple[float, float]]
    reference_channel: str
    output_shape: tuple[int, int]
    edges: list[RegistrationEdge] = field(default_factory=list)
    offset_yx: tuple[float, float] = (0.0, 0.0)
    warnings: list[str] = field(default_factory=list)
    accepted_edges: int = 0
    expected_edges: int = 0
    residual_median: Optional[float] = None
    residual_p95: Optional[float] = None
    residual_max: Optional[float] = None
    stage_correction_median: Optional[float] = None
    stage_correction_p95: Optional[float] = None
    accepted_graph_components: int = 0
    rejected_inconsistent_edges: int = 0
    radiometric_models: dict[str, RadiometricModel] = field(
        default_factory=dict, repr=False)
    spatial_models: dict[str, SpatialFieldModel] = field(
        default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return {
            "reference_channel": self.reference_channel,
            "output_shape_yx": list(self.output_shape),
            "offset_yx": list(self.offset_yx),
            "accepted_edges": self.accepted_edges,
            "expected_edges": self.expected_edges,
            "residual_median_px": self.residual_median,
            "residual_p95_px": self.residual_p95,
            "residual_max_px": self.residual_max,
            "stage_correction_median_px": self.stage_correction_median,
            "stage_correction_p95_px": self.stage_correction_p95,
            "accepted_graph_components": self.accepted_graph_components,
            "rejected_inconsistent_edges": self.rejected_inconsistent_edges,
            "warnings": list(self.warnings),
            "origins": [
                {"row": p[0], "column": p[1], "origin_yx": list(v),
                 "stage_origin_yx": list(self.stage_origins[p])}
                for p, v in sorted(self.origins.items())
            ],
            "edges": [asdict(edge) for edge in self.edges],
        }


@dataclass
class ScaleBarSpec:
    """Editable presentation settings for PNG/PDF scale bars."""

    visible: bool = True
    length_um: Optional[float] = None
    position: str = "bottom-left"
    color: tuple[int, int, int] = (255, 255, 255)
    thickness_px: int = 8
    font_size_px: int = 32
    margin_px: int = 32
    show_label: bool = True
    label: str = ""
    background: str = "dark"


def _check_cancel(cancel: Cancel):
    if cancel and cancel():
        raise MosaicCancelled("cancelled")


def _emit(progress: Progress, message: str, fraction: float):
    if progress:
        progress(message, float(max(0.0, min(1.0, fraction))))


def _laplacian(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image, dtype=np.float32)
    out = np.zeros_like(a)
    out[1:-1, 1:-1] = (
        -4.0 * a[1:-1, 1:-1]
        + a[:-2, 1:-1] + a[2:, 1:-1]
        + a[1:-1, :-2] + a[1:-1, 2:]
    )
    return out


def _block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Area-average by an integer factor, padding only the final partial block."""

    if factor <= 1:
        return np.asarray(image)
    a = np.asarray(image)
    h, w = a.shape[-2:]
    oh, ow = math.ceil(h / factor), math.ceil(w / factor)
    ph, pw = oh * factor - h, ow * factor - w
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)), mode="edge")
    out = a.astype(np.float32).reshape(oh, factor, ow, factor).mean(axis=(1, 3))
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(a.dtype)
    return out


def _overlap_patches(a: np.ndarray, b: np.ndarray,
                     delta_yx: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Patches that overlap when B is placed at delta relative to A."""

    dy, dx = (int(round(delta_yx[0])), int(round(delta_yx[1])))
    ha, wa = a.shape
    hb, wb = b.shape
    y0, x0 = max(0, dy), max(0, dx)
    y1, x1 = min(ha, dy + hb), min(wa, dx + wb)
    if y1 <= y0 or x1 <= x0:
        return np.empty((0, 0), np.float32), np.empty((0, 0), np.float32)
    return (a[y0:y1, x0:x1],
            b[y0 - dy:y1 - dy, x0 - dx:x1 - dx])


def _parabolic_offset(v0: float, v1: float, v2: float) -> float:
    den = v0 - 2.0 * v1 + v2
    if not np.isfinite(den) or abs(den) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (v0 - v2) / den, -0.5, 0.5))


def _phase_residual(pa: np.ndarray, pb: np.ndarray,
                    max_shift: float = 48.0) -> tuple:
    """Residual translation between already-overlapping patches.

    The search is deliberately limited around the stage prediction.  This avoids
    promoting a repeated tissue pattern one period away to a physically plausible
    neighbour translation.
    """

    if min(pa.shape, default=0) < 48 or pa.shape != pb.shape:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "overlap_too_small"
    texture = float(np.std(_laplacian(pa)))
    if not np.isfinite(texture) or texture < 0.08:
        return 0.0, 0.0, 0.0, 0.0, 0.0, texture, "low_information"

    ds = 2 if min(pa.shape) >= 256 else 1
    a = _block_mean(_laplacian(pa), ds).astype(np.float32)
    b = _block_mean(_laplacian(pb), ds).astype(np.float32)
    a -= float(a.mean())
    b -= float(b.mean())
    win = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    a *= win
    b *= win
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if den < 1e-8:
        return 0.0, 0.0, 0.0, 0.0, 0.0, texture, "low_information"

    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cps = fa * np.conj(fb)
    cps /= np.abs(cps) + 1e-9
    corr = np.fft.irfft2(cps, s=a.shape).real
    requested = max(1, int(math.ceil(max_shift / ds)))
    # Do not include the same circular FFT shift more than once.  Duplicate
    # wrapped positions can put a perfect zero-shift peak at the search border
    # and falsely classify a small overlap as ``search_boundary``.
    lim_y = min(requested, max(1, (corr.shape[0] - 1) // 2))
    lim_x = min(requested, max(1, (corr.shape[1] - 1) // 2))
    ys = np.arange(-lim_y, lim_y + 1, dtype=int)
    xs = np.arange(-lim_x, lim_x + 1, dtype=int)
    search = corr[np.ix_(ys % corr.shape[0], xs % corr.shape[1])]
    iy, ix = np.unravel_index(int(np.argmax(search)), search.shape)
    peak = float(search[iy, ix])

    mask = np.ones(search.shape, dtype=bool)
    mask[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] = False
    side = search[mask]
    side_mean = float(np.mean(side)) if side.size else 0.0
    side_std = float(np.std(side)) if side.size else 0.0
    psr = (peak - side_mean) / max(side_std, 1e-9)
    runner = float(np.max(side)) if side.size else 0.0
    peak_margin = (peak - runner) / max(abs(peak), 1e-9)

    if iy in (0, search.shape[0] - 1) or ix in (0, search.shape[1] - 1):
        return 0.0, 0.0, peak, psr, peak_margin, texture, "search_boundary"
    fy = _parabolic_offset(search[iy - 1, ix], search[iy, ix], search[iy + 1, ix])
    fx = _parabolic_offset(search[iy, ix - 1], search[iy, ix], search[iy, ix + 1])
    return ((ys[iy] + fy) * ds, (xs[ix] + fx) * ds,
            peak, psr, peak_margin, texture, "")


def _aligned_ncc(a: np.ndarray, b: np.ndarray,
                 delta_yx: tuple[float, float]) -> float:
    pa, pb = _overlap_patches(a, b, delta_yx)
    if min(pa.shape, default=0) < 32:
        return 0.0
    pa = _block_mean(_laplacian(pa), 2).astype(np.float32)
    pb = _block_mean(_laplacian(pb), 2).astype(np.float32)
    pa -= float(pa.mean())
    pb -= float(pb.mean())
    den = float(np.sqrt(np.sum(pa * pa) * np.sum(pb * pb)))
    return float(np.sum(pa * pb) / den) if den > 1e-8 else 0.0


class _PlaneCache:
    """Small LRU cache; keeps XY work bounded even for hundreds of fields."""

    def __init__(self, dataset, channel_key: str, max_items: int = 10,
                 downsample: int = 1):
        self.dataset = dataset
        self.channel_key = channel_key
        self.max_items = max(2, int(max_items))
        self.downsample = max(1, int(downsample))
        self._items: OrderedDict[Position, np.ndarray] = OrderedDict()
        self._tiles = {(t.row, t.col): t for t in dataset.tiles}

    def get(self, pos: Position) -> np.ndarray:
        if pos in self._items:
            value = self._items.pop(pos)
            self._items[pos] = value
            return value
        tile = self._tiles[pos]
        value = mosaic.read_plane(self.dataset, tile, self.channel_key)
        if self.downsample > 1:
            value = _block_mean(value, self.downsample)
        self._items[pos] = value
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return value


def choose_reference_channel(dataset) -> str:
    """Choose the most informative original plane from a small stratified sample."""

    if not dataset.channels:
        raise ValueError("the image set has no original channels")
    positions = sorted((t.row, t.col) for t in dataset.tiles)
    take = sorted(set(np.linspace(0, len(positions) - 1,
                                  min(7, len(positions))).astype(int)))
    best = None
    for channel in dataset.channels:
        cache = _PlaneCache(dataset, channel.key, max_items=2, downsample=4)
        scores = []
        for idx in take:
            try:
                scores.append(float(np.std(_laplacian(cache.get(positions[idx])))))
            except Exception:
                continue
        score = float(np.median(scores)) if scores else -1.0
        # Relief/brightfield is an excellent tiebreaker but never overrides data.
        relief = 1 if any(s in channel.label.lower() for s in ("relief", "bright")) else 0
        candidate = (score, relief, channel.key)
        if best is None or candidate > best:
            best = candidate
    if best is None or best[0] < 0:
        raise ValueError("none of the source channels could be read")
    return best[2]


def _measure_edge(a: np.ndarray, b: np.ndarray, pos_a: Position, pos_b: Position,
                  stage_delta: tuple[float, float], axis: str,
                  max_shift: float) -> RegistrationEdge:
    # Overlap extraction necessarily uses integer array indices.  The phase
    # residual is therefore relative to this rounded placement, not to the
    # original sub-pixel stage delta.
    sampled_delta = (float(round(stage_delta[0])), float(round(stage_delta[1])))
    pa, pb = _overlap_patches(a, b, sampled_delta)
    ry, rx, _peak, psr, margin, texture, reason = _phase_residual(
        pa, pb, max_shift=max_shift)
    if reason:
        shift = (float(stage_delta[0]), float(stage_delta[1]))
    else:
        shift = (sampled_delta[0] + ry, sampled_delta[1] + rx)
    correction = (shift[0] - stage_delta[0], shift[1] - stage_delta[1])
    ncc = _aligned_ncc(a, b, shift) if not reason else 0.0
    accepted = not reason and np.isfinite(ncc) and ncc >= 0.08 and psr >= 4.0
    if not accepted and not reason:
        reason = "weak_match"
    # Ambiguous edges remain available at low weight; the global graph, stage
    # prior, and robust loss decide whether they are consistent with other edges.
    confidence = max(0.0, min(1.0, (ncc - 0.04) / 0.45))
    confidence *= max(0.15, min(1.0, psr / 12.0))
    confidence *= max(0.20, min(1.0, (margin + 0.02) / 0.25))
    return RegistrationEdge(
        tile_a=pos_a, tile_b=pos_b, axis=axis,
        stage_delta_yx=(float(stage_delta[0]), float(stage_delta[1])),
        shift_yx=(float(shift[0]), float(shift[1])),
        correction_yx=(float(correction[0]), float(correction[1])),
        ncc=float(ncc), psr=float(psr),
        peak_margin=float(margin), texture=float(texture), accepted=accepted,
        reason=reason, weight=float(max(0.02, confidence) if accepted else 0.0),
    )


def _affine_fit(origins: dict[Position, tuple[float, float]]) -> tuple[dict, np.ndarray]:
    pos = sorted(origins)
    design = np.array([[1.0, p[0], p[1]] for p in pos], dtype=np.float64)
    values = np.array([origins[p] for p in pos], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    fit = design @ coef
    return {p: tuple(fit[i]) for i, p in enumerate(pos)}, coef


def _calibrated_prior(stage_origins: dict[Position, tuple[float, float]],
                      edges: list[RegistrationEdge]) -> dict[Position, tuple[float, float]]:
    stage_fit, coef = _affine_fit(stage_origins)
    x_edges = [e for e in edges if e.accepted and e.axis == "x"]
    y_edges = [e for e in edges if e.accepted and e.axis == "y"]
    stage_step_y = tuple(coef[1])
    stage_step_x = tuple(coef[2])
    step_x = tuple(np.median(np.array([e.shift_yx for e in x_edges]), axis=0)) \
        if x_edges else stage_step_x
    step_y = tuple(np.median(np.array([e.shift_yx for e in y_edges]), axis=0)) \
        if y_edges else stage_step_y
    r0 = min(p[0] for p in stage_origins)
    c0 = min(p[1] for p in stage_origins)
    out = {}
    for p, stage in stage_origins.items():
        residual = np.asarray(stage) - np.asarray(stage_fit[p])
        nominal = ((p[0] - r0) * np.asarray(step_y)
                   + (p[1] - c0) * np.asarray(step_x))
        out[p] = tuple(nominal + residual)
    return out


def _linear_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Use sparse LSQR when available, with a NumPy fallback for source installs."""

    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import lsqr
        return np.asarray(lsqr(csr_matrix(matrix), rhs, atol=1e-8, btol=1e-8,
                               iter_lim=max(500, matrix.shape[1] * 4))[0])
    except ImportError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _solve_positions(prior: dict[Position, tuple[float, float]],
                     edges: list[RegistrationEdge], prior_weight: float = 0.035) -> dict:
    pos = sorted(prior)
    index = {p: i for i, p in enumerate(pos)}
    used = [e for e in edges if e.accepted and e.tile_a in index and e.tile_b in index]
    robust = np.ones(len(used), dtype=np.float64)
    solution = np.array([prior[p] for p in pos], dtype=np.float64)

    for _ in range(5):
        rows, by, bx = [], [], []
        for k, edge in enumerate(used):
            w = math.sqrt(max(1e-6, edge.weight * robust[k]))
            row = np.zeros(len(pos), dtype=np.float64)
            row[index[edge.tile_a]] = -w
            row[index[edge.tile_b]] = w
            rows.append(row)
            by.append(w * edge.shift_yx[0])
            bx.append(w * edge.shift_yx[1])
        pw = math.sqrt(prior_weight)
        for p in pos:
            row = np.zeros(len(pos), dtype=np.float64)
            row[index[p]] = pw
            rows.append(row)
            by.append(pw * prior[p][0])
            bx.append(pw * prior[p][1])
        # Fix translation without forcing the far field back toward raw stage pitch.
        anchor = pos[0]
        row = np.zeros(len(pos), dtype=np.float64)
        row[index[anchor]] = 10.0
        rows.append(row)
        by.append(10.0 * prior[anchor][0])
        bx.append(10.0 * prior[anchor][1])
        mat = np.vstack(rows)
        solution[:, 0] = _linear_solve(mat, np.asarray(by))
        solution[:, 1] = _linear_solve(mat, np.asarray(bx))
        if not used:
            break
        residual = np.array([
            np.hypot(*((solution[index[e.tile_b]] - solution[index[e.tile_a]])
                       - np.asarray(e.shift_yx))) for e in used
        ])
        med = float(np.median(residual))
        scale = max(0.35, 1.4826 * float(np.median(np.abs(residual - med))))
        limit = max(1.5, med + 2.5 * scale)
        robust = np.minimum(1.0, limit / np.maximum(residual, 1e-9))

    return {p: tuple(solution[index[p]]) for p in pos}


def estimate_geometry(dataset, reference_channel: Optional[str] = None,
                      progress: Progress = None, cancel: Cancel = None,
                      max_stage_residual_px: float = 48.0) -> MosaicGeometry:
    """Measure neighbour translations and solve one robust position per tile."""

    reference_channel = reference_channel or choose_reference_channel(dataset)
    keys = {c.key for c in dataset.channels}
    if reference_channel not in keys:
        raise ValueError(f"unknown reference channel: {reference_channel}")
    stage = mosaic.stage_origins(dataset)
    positions = sorted((t.row, t.col) for t in dataset.tiles)
    available = set(positions)
    expected = []
    for p in positions:
        for q, axis in [((p[0], p[1] + 1), "x"), ((p[0] + 1, p[1]), "y")]:
            if q in available:
                expected.append((p, q, axis))
    if not expected:
        raise ValueError("the image set does not contain adjacent tiles")

    cache = _PlaneCache(dataset, reference_channel, max_items=14)
    edges = []
    for i, (a_pos, b_pos, axis) in enumerate(expected):
        _check_cancel(cancel)
        try:
            a = cache.get(a_pos)
            b = cache.get(b_pos)
            delta = tuple(np.asarray(stage[b_pos]) - np.asarray(stage[a_pos]))
            edge = _measure_edge(a, b, a_pos, b_pos, delta, axis,
                                 max_stage_residual_px)
        except Exception as exc:
            delta = tuple(np.asarray(stage[b_pos]) - np.asarray(stage[a_pos]))
            edge = RegistrationEdge(
                a_pos, b_pos, axis, delta, delta, (0.0, 0.0), 0.0, 0.0,
                0.0, 0.0, False, f"read_error:{exc}", 0.0)
        edges.append(edge)
        _emit(progress, f"位置合わせ {i + 1}/{len(expected)}", (i + 1) / len(expected) * 0.78)

    accepted = [e for e in edges if e.accepted]
    prior = _calibrated_prior(stage, edges)
    solved = _solve_positions(prior, edges) if accepted else prior
    rejected_inconsistent = 0
    # IRLS limits the influence of a gross loop-inconsistent match, but leaving
    # that measurement labelled "accepted" makes QC misleading.  Remove only
    # clear global outliers and solve once more with the remaining graph.
    for _ in range(8):
        if len(accepted) < 3:
            break
        norms = []
        for edge in accepted:
            delta = np.asarray(solved[edge.tile_b]) - np.asarray(solved[edge.tile_a])
            norms.append(float(np.linalg.norm(delta - np.asarray(edge.shift_yx))))
        median = float(np.median(norms))
        sigma = max(0.15, 1.4826 * float(np.median(np.abs(np.asarray(norms) - median))))
        threshold = min(5.0, max(2.5, median + 6.0 * sigma))
        worst = int(np.argmax(norms))
        if norms[worst] <= threshold:
            break
        # A contradictory cycle does not identify which of its edges is wrong.
        # Remove one worst edge, then re-solve instead of discarding the entire
        # cycle in a single pass.
        bad_ids = {id(accepted[worst])}
        edges = [
            RegistrationEdge(**{
                **asdict(edge), "accepted": False,
                "reason": "global_inconsistent", "weight": 0.0,
            }) if id(edge) in bad_ids else edge
            for edge in edges
        ]
        rejected_inconsistent += len(bad_ids)
        accepted = [edge for edge in edges if edge.accepted]
        prior = _calibrated_prior(stage, edges)
        solved = _solve_positions(prior, edges) if accepted else prior
    min_y = min(v[0] for v in solved.values())
    min_x = min(v[1] for v in solved.values())
    origins = {p: (v[0] - min_y, v[1] - min_x) for p, v in solved.items()}
    stage_norm = {p: (v[0] - min_y, v[1] - min_x) for p, v in stage.items()}
    th, tw = dataset.tile_shape
    # Origins locate pixel centres.  The last source pixel contributes through
    # ``origin + size - 0.5`` (exclusive), matching ChunkRenderer's half-pixel
    # support.  Counting to ``origin + size`` adds a guaranteed blank outer
    # row/column for origins whose fractional part is <= 0.5.
    height = int(math.ceil(max(v[0] + th - 0.5 for v in origins.values())))
    width = int(math.ceil(max(v[1] + tw - 0.5 for v in origins.values())))

    residuals = []
    final_edges = []
    for edge in edges:
        residual = None
        if edge.accepted:
            d = ((origins[edge.tile_b][0] - origins[edge.tile_a][0]) - edge.shift_yx[0],
                 (origins[edge.tile_b][1] - origins[edge.tile_a][1]) - edge.shift_yx[1])
            residual = (float(d[0]), float(d[1]))
            residuals.append(float(np.hypot(*d)))
        final_edges.append(RegistrationEdge(**{**asdict(edge), "residual_yx": residual}))
    stage_corr = [float(np.hypot(origins[p][0] - stage_norm[p][0],
                                 origins[p][1] - stage_norm[p][1])) for p in origins]
    warnings = []
    ratio = len(accepted) / len(expected)
    component_count = len(positions)
    if not accepted:
        warnings.append("画像から位置を検証できなかったため、Stage座標のみで配置しました。")
    else:
        neighbours = {p: set() for p in positions}
        for edge in accepted:
            neighbours[edge.tile_a].add(edge.tile_b)
            neighbours[edge.tile_b].add(edge.tile_a)
        unseen = set(positions)
        component_count = 0
        while unseen:
            component_count += 1
            pending = [unseen.pop()]
            while pending:
                current = pending.pop()
                linked = neighbours[current] & unseen
                unseen.difference_update(linked)
                pending.extend(linked)
        if component_count > 1:
            warnings.append(
                f"画像位置合わせが {component_count} 個の非連結領域に分かれたため、"
                "領域間はStage座標で配置しました。"
            )
        if ratio < 0.5:
            warnings.append(
                f"画像で検証できた継ぎ目は {len(accepted)}/{len(expected)} です。"
            )
    if residuals and float(np.percentile(residuals, 95)) > 2.0:
        warnings.append("位置合わせ残差の95パーセンタイルが2 pxを超えています。")
    if residuals and float(max(residuals)) > 5.0:
        warnings.append("位置合わせ残差が5 pxを超える継ぎ目があります。")
    if rejected_inconsistent:
        warnings.append(
            f"グローバル配置と整合しない継ぎ目 {rejected_inconsistent} 本を除外しました。"
        )
    _emit(progress, "全タイルの位置を最適化しました", 1.0)
    return MosaicGeometry(
        origins=origins, stage_origins=stage_norm,
        reference_channel=reference_channel, output_shape=(height, width),
        edges=final_edges, offset_yx=(float(min_y), float(min_x)), warnings=warnings,
        accepted_edges=len(accepted), expected_edges=len(expected),
        residual_median=(float(np.median(residuals)) if residuals else None),
        residual_p95=(float(np.percentile(residuals, 95)) if residuals else None),
        residual_max=(float(max(residuals)) if residuals else None),
        stage_correction_median=float(np.median(stage_corr)),
        stage_correction_p95=float(np.percentile(stage_corr, 95)),
        accepted_graph_components=component_count,
        rejected_inconsistent_edges=rejected_inconsistent,
    )


# ---------------------------------------------------------------- radiometry

def _identity_radiometric_model(dataset, channel_key: str,
                                reason: str = "") -> RadiometricModel:
    positions = [(tile.row, tile.col) for tile in dataset.tiles]
    return RadiometricModel(
        channel_key=channel_key,
        gains={p: 1.0 for p in positions},
        offsets={p: 0.0 for p in positions},
        status="identity",
        components=len(positions),
        warnings=[reason] if reason else [],
    )


def _identity_spatial_model(channel_key: str, reason: str = "") -> SpatialFieldModel:
    return SpatialFieldModel(
        channel_key=channel_key, status="identity",
        warnings=[reason] if reason else [],
    )


def _resize_spatial_field(field_dn: np.ndarray,
                          shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resize a small detector field to one source-plane shape."""

    source = np.asarray(field_dn, np.float32)
    if tuple(source.shape) == tuple(shape):
        return source
    yy = np.linspace(0.0, max(0, source.shape[0] - 1), shape[0], dtype=np.float32)
    xx = np.linspace(0.0, max(0, source.shape[1] - 1), shape[1], dtype=np.float32)
    return _bilinear_patch(source, yy, xx)


def _common_overlap_samples(cache: _PlaneCache, geometry: MosaicGeometry,
                            tile_a: Position, tile_b: Position,
                            downsample: int,
                            field_dn: Optional[np.ndarray] = None,
                            max_samples: int = 30000,
                            return_2d: bool = False
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Sample two tiles at identical global coordinates with bilinear sampling.

    Registration produces fractional origins.  Rounding an overlap rectangle
    before comparing a low-frequency gradient can make a sub-pixel geometry
    residual look like illumination shading, so radiometric validation uses the
    same continuous placement as the feather renderer.
    """

    a = cache.get(tile_a)
    b = cache.get(tile_b)
    oa = np.asarray(geometry.origins[tile_a], np.float64) / downsample
    ob = np.asarray(geometry.origins[tile_b], np.float64) / downsample
    margin = 2.0

    def symmetric_grid(origin_a, origin_b, size_a, size_b):
        # The phase halfway between both fractional origins gives A and B the
        # same interpolation blur (opposite sub-pixel phases).  It is invariant
        # when an edge is reversed, unlike sampling one tile on integer pixels.
        fractional_delta = (origin_b - origin_a) - round(origin_b - origin_a)
        phase = origin_a + 0.5 * fractional_delta
        lower = max(origin_a + margin, origin_b + margin)
        upper = min(origin_a + size_a - 1.0 - margin,
                    origin_b + size_b - 1.0 - margin)
        first = int(math.ceil(lower - phase))
        last = int(math.floor(upper - phase))
        if last < first:
            return np.empty(0, np.float32)
        return (np.arange(first, last + 1, dtype=np.float32)
                + np.float32(phase))

    yy = symmetric_grid(oa[0], ob[0], a.shape[0], b.shape[0])
    xx = symmetric_grid(oa[1], ob[1], a.shape[1], b.shape[1])
    if yy.size == 0 or xx.size == 0:
        raise ValueError("radiometric overlap is empty")
    pa = _bilinear_patch(a, yy - np.float32(oa[0]), xx - np.float32(oa[1]))
    pb = _bilinear_patch(b, yy - np.float32(ob[0]), xx - np.float32(ob[1]))
    if field_dn is not None:
        field = (field_dn if tuple(field_dn.shape) == tuple(a.shape)
                 else _resize_spatial_field(field_dn, a.shape))
        pa = pa - _bilinear_patch(
            field, yy - np.float32(oa[0]), xx - np.float32(oa[1]))
        pb = pb - _bilinear_patch(
            field, yy - np.float32(ob[0]), xx - np.float32(ob[1]))
    count = pa.size
    stride = max(1, int(math.ceil(math.sqrt(count / max_samples))))
    pa, pb = pa[::stride, ::stride], pb[::stride, ::stride]
    if return_2d:
        return pa, pb
    return pa.ravel(), pb.ravel()


def _low_frequency_field(cache: _PlaneCache,
                         positions: list[Position]) -> np.ndarray:
    """Robust common additive field after removing each tile's scalar median."""

    from scipy.ndimage import gaussian_filter

    centered = []
    for position in positions:
        plane = cache.get(position).astype(np.float32)
        centered.append(plane - np.float32(np.median(plane)))
    field_dn = np.median(np.stack(centered), axis=0).astype(np.float32)
    sigma = max(1.5, min(field_dn.shape) / 20.0)
    field_dn = gaussian_filter(field_dn, sigma=sigma, mode="nearest")
    field_dn -= np.float32(np.median(field_dn))
    return field_dn.astype(np.float32, copy=False)


def _field_agreement(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    aa, bb = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if float(np.std(aa)) < 1e-8 or float(np.std(bb)) < 1e-8:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(aa, bb)[0, 1])
    span = max(0.5, 0.5 * sum(
        float(np.percentile(field, 95) - np.percentile(field, 5))
        for field in (a, b)
    ))
    disagreement = float(np.sqrt(np.mean((aa - bb) ** 2)) / span)
    return correlation, disagreement


def _robust_channel_dynamic(cache: _PlaneCache,
                            positions: list[Position],
                            cancel: Cancel = None,
                            progress: Progress = None) -> float:
    ranges = []
    for index, position in enumerate(positions):
        _check_cancel(cancel)
        plane = np.asarray(cache.get(position), np.float32)
        finite = plane[np.isfinite(plane)]
        if finite.size:
            ranges.append(float(np.percentile(finite, 99) - np.percentile(finite, 1)))
        if index % 8 == 0 or index + 1 == len(positions):
            _emit(progress, "チャンネルの輝度範囲を検証中",
                  (index + 1) / max(1, len(positions)))
    return float(np.median(ranges)) if ranges else 0.0


def estimate_spatial_field_model(dataset, geometry: MosaicGeometry,
                                 channel_key: str, progress: Progress = None,
                                 cancel: Cancel = None) -> SpatialFieldModel:
    """Estimate a strictly gated additive low-frequency presentation field.

    The estimator deliberately does not claim calibrated flat-field recovery.
    Four independent row/column parity subsets must reproduce the same detector
    pattern, and each candidate is evaluated on overlaps from the opposite
    subset before any source pixels are changed.
    """

    model = _identity_spatial_model(channel_key)
    dtype = _channel_dtype(dataset, channel_key)
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        model.warnings.append("unsupported dtype; common field disabled")
        return model
    positions = sorted(geometry.origins)
    if len(positions) < 24:
        model.warnings.append("too few tiles for independent common-field splits")
        return model
    eligible = [
        edge for edge in geometry.edges
        if edge.accepted and edge.residual_yx is not None
        and float(np.hypot(*edge.residual_yx)) <= 2.0
    ]
    if not eligible or not any(edge.axis == "x" for edge in eligible) \
            or not any(edge.axis == "y" for edge in eligible):
        model.warnings.append("both overlap orientations are required")
        return model

    downsample = max(1, int(math.ceil(max(dataset.tile_shape) / 128.0)))
    model.downsample = downsample
    cache = _PlaneCache(dataset, channel_key, max_items=len(positions),
                        downsample=downsample)
    channel_dynamic = _robust_channel_dynamic(
        cache, positions, cancel=cancel,
        progress=(lambda message, value: _emit(
            progress, message, 0.18 * value)))
    model.channel_dynamic_range_dn = channel_dynamic
    groups = {
        "row0": [p for p in positions if p[0] % 2 == 0],
        "row1": [p for p in positions if p[0] % 2 == 1],
        "col0": [p for p in positions if p[1] % 2 == 0],
        "col1": [p for p in positions if p[1] % 2 == 1],
    }
    if min(map(len, groups.values())) < 12:
        model.warnings.append("too few tiles in a row/column split")
        return model
    try:
        fields = {}
        for index, (name, selected) in enumerate(groups.items()):
            _check_cancel(cancel)
            fields[name] = _low_frequency_field(cache, selected)
            _emit(progress, "低周波fieldのsplit再現性を検証中",
                  0.18 + 0.22 * (index + 1) / len(groups))
    except MosaicCancelled:
        raise
    except Exception as exc:
        model.warnings.append(f"common-field read failed: {exc}")
        return model

    row_corr, row_disagreement = _field_agreement(fields["row0"], fields["row1"])
    col_corr, col_disagreement = _field_agreement(fields["col0"], fields["col1"])
    model.split_correlation_row = row_corr
    model.split_correlation_column = col_corr
    model.split_disagreement_row = row_disagreement
    model.split_disagreement_column = col_disagreement
    if (min(row_corr, col_corr) < 0.93
            or max(row_disagreement, col_disagreement) > 0.20):
        model.warnings.append("independent split fields were not reproducible")
        return model

    from scipy.ndimage import gaussian_filter
    field_dn = np.median(np.stack(list(fields.values())), axis=0).astype(np.float32)
    field_dn = gaussian_filter(
        field_dn, sigma=max(1.0, min(field_dn.shape) / 28.0), mode="nearest")
    field_dn -= np.float32(np.median(field_dn))
    p01, p05, p95, p99 = map(float, np.percentile(field_dn, (1, 5, 95, 99)))
    model.field_p01_dn, model.field_p99_dn = p01, p99
    full_scale = float(np.iinfo(dtype).max)
    if p95 - p05 < 0.5:
        model.warnings.append("common field was below the measurable amplitude")
        return model
    amplitude_limit = min(0.02 * full_scale,
                          max(0.5, 0.25 * channel_dynamic))
    if p99 - p01 > amplitude_limit:
        model.warnings.append("common field exceeded the safe amplitude bound")
        return model

    seam_records = []
    evaluations = [
        ("row0", "x", 1), ("row1", "x", 0),
        ("col0", "y", 1), ("col1", "y", 0),
    ]
    for field_name, axis, heldout_parity in evaluations:
        validation_field = fields[field_name]
        for edge in eligible:
            _check_cancel(cancel)
            parity = edge.tile_a[0] if axis == "x" else edge.tile_a[1]
            if edge.axis != axis or parity % 2 != heldout_parity:
                continue
            try:
                raw_a, raw_b = _common_overlap_samples(
                    cache, geometry, edge.tile_a, edge.tile_b, downsample)
                corrected_a, corrected_b = _common_overlap_samples(
                    cache, geometry, edge.tile_a, edge.tile_b, downsample,
                    validation_field)
            except Exception:
                continue
            raw_delta = raw_a - raw_b
            corrected_delta = corrected_a - corrected_b
            raw_center = float(np.median(raw_delta))
            corrected_center = float(np.median(corrected_delta))
            seam_records.append({
                "axis": axis,
                "raw_bias": abs(raw_center),
                "corrected_bias": abs(corrected_center),
                "raw_mad": 1.4826 * float(
                    np.median(np.abs(raw_delta - raw_center))),
                "corrected_mad": 1.4826 * float(
                    np.median(np.abs(corrected_delta - corrected_center))),
                "raw_p95": float(np.percentile(np.abs(raw_delta), 95)),
                "corrected_p95": float(
                    np.percentile(np.abs(corrected_delta), 95)),
            })
        _emit(progress, "独立overlapでfieldを検証中",
              0.40 + 0.28 * (evaluations.index(
                  (field_name, axis, heldout_parity)) + 1) / len(evaluations))
    model.heldout_edges = len(seam_records)
    if len(seam_records) < 12:
        model.warnings.append("too few held-out overlaps")
        return model
    model.raw_heldout_bias_dn = float(np.median(
        [record["raw_bias"] for record in seam_records]))
    model.corrected_heldout_bias_dn = float(np.median(
        [record["corrected_bias"] for record in seam_records]))
    model.raw_heldout_mad_dn = float(np.median(
        [record["raw_mad"] for record in seam_records]))
    model.corrected_heldout_mad_dn = float(np.median(
        [record["corrected_mad"] for record in seam_records]))
    model.orientation_qc = _orientation_seam_qc(seam_records)
    if not _orientation_seams_improve_safely(model.orientation_qc, 0.20):
        model.warnings.append("held-out seams did not improve safely")
        return model

    full_field = _resize_spatial_field(field_dn, dataset.tile_shape)
    clipped = 0
    unsaturated = 0
    tiles = {(tile.row, tile.col): tile for tile in dataset.tiles}
    try:
        for index, position in enumerate(positions):
            _check_cancel(cancel)
            plane = np.asarray(
                mosaic.read_plane(dataset, tiles[position], channel_key),
                dtype=np.float32)
            valid = np.isfinite(plane) & (plane > 0.0) & (plane < full_scale)
            corrected = plane - full_field
            clipped += int(np.count_nonzero(
                valid & ((corrected < 0.0) | (corrected > full_scale))))
            unsaturated += int(np.count_nonzero(valid))
            if index % 8 == 0 or index + 1 == len(positions):
                _emit(progress, "fieldのclippingを検証中",
                      0.68 + 0.32 * (index + 1) / len(positions))
    except MosaicCancelled:
        raise
    except Exception as exc:
        model.warnings.append(f"common-field clipping preflight failed: {exc}")
        return model
    model.new_clip_fraction = clipped / max(1, unsaturated)
    if model.new_clip_fraction > 1e-4:
        model.warnings.append("common field would create too many clipped pixels")
        return model

    model.field_dn = field_dn
    model.status = "applied"
    return model


def _cached_spatial_model(dataset, geometry: MosaicGeometry,
                          channel_key: str, progress: Progress = None,
                          cancel: Cancel = None) -> SpatialFieldModel:
    cached = geometry.spatial_models.get(channel_key)
    if cached is not None:
        return cached
    try:
        model = estimate_spatial_field_model(
            dataset, geometry, channel_key, progress=progress, cancel=cancel)
    except MosaicCancelled:
        raise
    except Exception as exc:
        model = _identity_spatial_model(
            channel_key, f"common-field estimation failed: {exc}")
    geometry.spatial_models[channel_key] = model
    return model


def _weighted_tls(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """Robust orthogonal affine fit ``y = slope*x + intercept``."""

    weights = np.ones(x.size, np.float64)
    slope, intercept = 1.0, float(np.median(y - x))
    scale = 0.0
    for _ in range(6):
        total = float(weights.sum())
        mx = float(np.sum(weights * x) / total)
        my = float(np.sum(weights * y) / total)
        dx, dy = x - mx, y - my
        vx = float(np.sum(weights * dx * dx) / total)
        vy = float(np.sum(weights * dy * dy) / total)
        cov = float(np.sum(weights * dx * dy) / total)
        if cov <= 1e-12:
            raise ValueError("non-positive overlap covariance")
        slope = float((vy - vx + math.sqrt((vy - vx) ** 2 + 4.0 * cov ** 2))
                      / (2.0 * cov))
        intercept = float(np.median(y - slope * x))
        residual = (y - (slope * x + intercept)) / math.sqrt(1.0 + slope * slope)
        center = float(np.median(residual))
        scale = max(1e-6, 1.4826 * float(np.median(np.abs(residual - center))))
        weights = np.minimum(1.0, 2.5 * scale / np.maximum(np.abs(residual - center), 1e-9))
    corr = cov / max(1e-12, math.sqrt(vx * vy))
    inlier = float(np.mean(np.abs(residual - center) <= 3.0 * scale))
    return slope, intercept, float(corr), inlier


def _radiometric_overlap(cache: _PlaneCache, geometry: MosaicGeometry,
                         tile_a: Position, tile_b: Position,
                         downsample: int, max_samples: int = 30000,
                         field_dn: Optional[np.ndarray] = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Sample a common global grid, optionally after common-field removal."""

    return _common_overlap_samples(
        cache, geometry, tile_a, tile_b, downsample,
        field_dn=field_dn, max_samples=max_samples)


def _radiometric_pair_fit(a: np.ndarray, b: np.ndarray,
                          full_scale: float) -> dict:
    """Fit raw A as ``slope * raw B + intercept`` with conservative gates."""

    aa = np.asarray(a, np.float64) / full_scale
    bb = np.asarray(b, np.float64) / full_scale
    finite = np.isfinite(aa) & np.isfinite(bb)
    saturation = float(np.mean(finite & ((aa >= 1.0) | (bb >= 1.0))))
    valid = finite & (aa >= 0.0) & (bb >= 0.0) & (aa < 1.0) & (bb < 1.0)
    aa, bb = aa[valid], bb[valid]
    result = {"samples": int(aa.size), "saturation_fraction": saturation}
    if aa.size < 1024:
        return {**result, "accepted": False, "reason": "too_few_samples"}
    a05, a95 = np.percentile(aa, (5, 95))
    b05, b95 = np.percentile(bb, (5, 95))
    dynamic_a, dynamic_b = float(a95 - a05), float(b95 - b05)
    signal = (aa > a05 + 0.10 * dynamic_a) | (bb > b05 + 0.10 * dynamic_b)
    signal_fraction = float(np.mean(signal))
    result.update(dynamic_a=dynamic_a, dynamic_b=dynamic_b,
                  signal_fraction=signal_fraction)
    if saturation > 0.05:
        return {**result, "accepted": False, "reason": "saturated_overlap"}
    if min(dynamic_a, dynamic_b) < 0.012 or signal_fraction < 0.05:
        return {**result, "accepted": False, "reason": "low_information"}
    try:
        slope, intercept, corr, inlier = _weighted_tls(aa, bb)
    except ValueError as exc:
        return {**result, "accepted": False, "reason": str(exc)}
    result.update(
        slope=float(slope), intercept_normalized=float(intercept),
        correlation=float(corr), inlier_fraction=float(inlier),
        median_a=float(np.median(aa)), median_b=float(np.median(bb)),
    )
    if corr < 0.72 or inlier < 0.60:
        return {**result, "accepted": False, "reason": "unstable_affine_fit"}
    if not (0.80 <= slope <= 1.25) or abs(intercept) > 0.05:
        return {**result, "accepted": False, "reason": "affine_fit_out_of_bounds"}
    return {**result, "accepted": True, "reason": ""}


def _solve_scalar_graph(positions: list[Position], constraints: list[tuple],
                        prior_weight: float = 0.04) -> dict[Position, float]:
    index = {p: i for i, p in enumerate(positions)}
    robust = np.ones(len(constraints), np.float64)
    solution = np.zeros(len(positions), np.float64)
    value_floor = (0.05 * float(np.median(np.abs([item[2] for item in constraints])))
                   if constraints else 0.0)
    for _ in range(5):
        rows, rhs = [], []
        for k, (a, b, value, confidence) in enumerate(constraints):
            weight = math.sqrt(max(1e-6, confidence * robust[k]))
            row = np.zeros(len(positions), np.float64)
            row[index[a]], row[index[b]] = -weight, weight
            rows.append(row)
            rhs.append(weight * value)
        prior = math.sqrt(prior_weight)
        for p in positions:
            row = np.zeros(len(positions), np.float64)
            row[index[p]] = prior
            rows.append(row)
            rhs.append(0.0)
        solution = _linear_solve(np.vstack(rows), np.asarray(rhs))
        if not constraints:
            break
        residual = np.asarray([
            (solution[index[b]] - solution[index[a]]) - value
            for a, b, value, _ in constraints
        ])
        scale = max(1e-4, value_floor,
                    1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        robust = np.minimum(1.0, 2.5 * scale / np.maximum(np.abs(residual), 1e-9))
    return {p: float(solution[index[p]]) for p in positions}


def _orientation_seam_qc(records: list[dict]) -> dict[str, dict]:
    """Aggregate seam evidence per acquisition axis, never across axes."""

    output = {}
    for axis in ("x", "y"):
        selected = [record for record in records if record["axis"] == axis]
        if not selected:
            continue
        output[axis] = {
            "edges": len(selected),
            "raw_bias_dn": float(np.median([r["raw_bias"] for r in selected])),
            "corrected_bias_dn": float(np.median(
                [r["corrected_bias"] for r in selected])),
            "raw_mad_dn": float(np.median([r["raw_mad"] for r in selected])),
            "corrected_mad_dn": float(np.median(
                [r["corrected_mad"] for r in selected])),
            "raw_p95_dn": float(np.median([r["raw_p95"] for r in selected])),
            "corrected_p95_dn": float(np.median(
                [r["corrected_p95"] for r in selected])),
        }
    return output


def _orientation_seams_improve_safely(qc: dict[str, dict],
                                      minimum_improvement: float = 0.20) -> bool:
    if not qc:
        return False
    improved = False
    for values in qc.values():
        raw_bias = values["raw_bias_dn"]
        corrected_bias = values["corrected_bias_dn"]
        # One quarter of one DN allows sub-pixel interpolation noise without
        # allowing a visible one-DN seam to be hidden by the other orientation.
        absolute_tolerance = 0.25
        if corrected_bias > 1.02 * raw_bias + absolute_tolerance:
            return False
        if (values["corrected_mad_dn"]
                > 1.02 * values["raw_mad_dn"] + absolute_tolerance):
            return False
        if (values["corrected_p95_dn"]
                > 1.02 * values["raw_p95_dn"] + absolute_tolerance):
            return False
        if (raw_bias >= 0.25
                and corrected_bias <= (1.0 - minimum_improvement) * raw_bias):
            improved = True
    return improved


def _offset_pair_summary(a: np.ndarray, b: np.ndarray,
                         full_scale: float) -> dict:
    """Train/validate one additive seam offset on disjoint spatial blocks."""

    aa, bb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if aa.ndim != 2 or aa.shape != bb.shape or min(aa.shape, default=0) < 8:
        return {"accepted": False, "reason": "offset_overlap_too_small"}
    finite = np.isfinite(aa) & np.isfinite(bb)
    saturation = float(np.mean(
        finite & ((aa >= full_scale) | (bb >= full_scale))))
    valid = finite & (aa >= 0.0) & (bb >= 0.0) \
        & (aa < full_scale) & (bb < full_scale)
    yy, xx = np.indices(aa.shape)
    block = max(4, min(12, min(aa.shape) // 6))
    checker = ((yy // block + xx // block) & 1) == 0
    train = valid & checker
    heldout = valid & ~checker
    result = {
        "samples_train": int(np.count_nonzero(train)),
        "samples_heldout": int(np.count_nonzero(heldout)),
        "saturation_fraction": saturation,
        "block_size": int(block),
    }
    if result["samples_train"] < 512 or result["samples_heldout"] < 512:
        return {**result, "accepted": False, "reason": "too_few_offset_samples"}
    if saturation > 0.05:
        return {**result, "accepted": False, "reason": "saturated_overlap"}
    delta_train = (aa - bb)[train].astype(np.float64)
    delta_heldout = (aa - bb)[heldout].astype(np.float64)
    center_train = float(np.median(delta_train))
    center_heldout = float(np.median(delta_heldout))
    mad_train = 1.4826 * float(np.median(np.abs(delta_train - center_train)))
    mad_heldout = 1.4826 * float(np.median(
        np.abs(delta_heldout - center_heldout)))
    consistency_limit = max(0.5, 0.25 * (mad_train + mad_heldout))
    result.update(
        constraint_dn=center_train,
        heldout_center_dn=center_heldout,
        train_mad_dn=mad_train,
        heldout_mad_dn=mad_heldout,
        split_difference_dn=abs(center_train - center_heldout),
    )
    if abs(center_train - center_heldout) > consistency_limit:
        return {**result, "accepted": False, "reason": "offset_split_disagreement"}
    return {**result, "accepted": True, "reason": ""}


def _estimate_offset_only_model(dataset, geometry: MosaicGeometry,
                                channel_key: str,
                                eligible: list[RegistrationEdge],
                                cache: _PlaneCache, downsample: int,
                                spatial_model: SpatialFieldModel,
                                spatial_field: Optional[np.ndarray],
                                full_scale: float,
                                prior_warnings: list[str],
                                progress: Progress = None,
                                cancel: Cancel = None) -> RadiometricModel:
    """Conservative additive-only fallback for low-information channels."""

    positions = sorted(geometry.origins)
    model = _identity_radiometric_model(dataset, channel_key)
    model.attempted_edges = len(eligible)
    model.mode = "offset-only"
    model.warnings = list(prior_warnings) + [
        "tile affine rejected; evaluated offset-only fallback"
    ]
    dynamic = _robust_channel_dynamic(
        cache, positions, cancel=cancel,
        progress=(lambda message, value: _emit(
            progress, message, 0.12 * value)))
    offset_bound = min(0.02 * full_scale, max(1.0, 0.25 * dynamic))
    fits, edge_qc = [], []
    for index, edge in enumerate(eligible):
        _check_cancel(cancel)
        try:
            pa, pb = _common_overlap_samples(
                cache, geometry, edge.tile_a, edge.tile_b, downsample,
                field_dn=spatial_field, return_2d=True)
            fit = _offset_pair_summary(pa, pb, full_scale)
        except Exception as exc:
            fit = {"accepted": False,
                   "reason": f"offset_read_or_overlap_error:{exc}"}
        if fit.get("accepted") and abs(fit["constraint_dn"]) > 2.0 * offset_bound:
            fit = {**fit, "accepted": False,
                   "reason": "offset_pair_out_of_bounds"}
        fit.update(tile_a=edge.tile_a, tile_b=edge.tile_b, axis=edge.axis)
        edge_qc.append({
            **{key: value for key, value in fit.items()
               if key not in {"tile_a", "tile_b"}},
            "row_a": edge.tile_a[0], "column_a": edge.tile_a[1],
            "row_b": edge.tile_b[0], "column_b": edge.tile_b[1],
        })
        if fit.get("accepted"):
            fits.append(fit)
        if index % 8 == 0 or index + 1 == len(eligible):
            _emit(progress, "offset-only overlapを推定中",
                  0.12 + 0.28 * (index + 1) / max(1, len(eligible)))
    model.edge_qc = edge_qc
    if not fits:
        model.warnings.append("all offset-only overlap fits were rejected")
        return model

    constraints = []
    for index, fit in enumerate(fits):
        _check_cancel(cancel)
        stability = 1.0 / (1.0 + fit["split_difference_dn"]
                           + 0.15 * fit["heldout_mad_dn"])
        sample_weight = min(1.0, fit["samples_train"] / 4096.0)
        constraints.append((fit["tile_a"], fit["tile_b"],
                            fit["constraint_dn"],
                            max(0.05, stability * sample_weight)))
    offsets = None
    chosen_prior = None
    for prior_weight in (0.04, 0.08, 0.12, 0.20, 0.30, 0.50, 1.0, 2.0):
        candidate = _solve_scalar_graph(
            positions, constraints, prior_weight=prior_weight)
        if all(abs(candidate[position]) <= offset_bound for position in positions):
            offsets, chosen_prior = candidate, prior_weight
            break
    if offsets is None:
        model.warnings.append("no bounded offset-only graph solution")
        return model
    if chosen_prior > 0.04:
        model.warnings.append(
            f"offset identity regularization increased to {chosen_prior:g}")

    adjacency = {position: set() for position in positions}
    for fit in fits:
        adjacency[fit["tile_a"]].add(fit["tile_b"])
        adjacency[fit["tile_b"]].add(fit["tile_a"])
    unseen, components = set(positions), []
    while unseen:
        component = {unseen.pop()}
        pending = list(component)
        while pending:
            linked = adjacency[pending.pop()] & unseen
            unseen.difference_update(linked)
            component.update(linked)
            pending.extend(linked)
        components.append(component)
    minimum_component = 8 if len(positions) >= 24 else 2
    inactive = set().union(*(
        component for component in components if len(component) < minimum_component
    )) if components else set()
    for position in inactive:
        offsets[position] = 0.0
    if inactive:
        model.warnings.append(
            f"{len(inactive)} tiles in small offset components retained identity")

    seam_records = []
    for index, fit in enumerate(fits):
        _check_cancel(cancel)
        try:
            pa, pb = _common_overlap_samples(
                cache, geometry, fit["tile_a"], fit["tile_b"], downsample,
                field_dn=spatial_field, return_2d=True)
        except Exception:
            continue
        finite = np.isfinite(pa) & np.isfinite(pb)
        yy, xx = np.indices(pa.shape)
        block = fit["block_size"]
        heldout = finite & (((yy // block + xx // block) & 1) == 1)
        delta = (pa - pb)[heldout].astype(np.float64)
        if delta.size < 256:
            continue
        adjusted = delta + offsets[fit["tile_a"]] - offsets[fit["tile_b"]]
        raw_center = float(np.median(delta))
        adjusted_center = float(np.median(adjusted))
        seam_records.append({
            "axis": fit["axis"],
            "raw_bias": abs(raw_center),
            "corrected_bias": abs(adjusted_center),
            "raw_mad": 1.4826 * float(np.median(np.abs(delta - raw_center))),
            "corrected_mad": 1.4826 * float(
                np.median(np.abs(adjusted - adjusted_center))),
            "raw_p95": float(np.percentile(np.abs(delta), 95)),
            "corrected_p95": float(np.percentile(np.abs(adjusted), 95)),
        })
        if index % 8 == 0 or index + 1 == len(fits):
            _emit(progress, "offset-only holdoutを検証中",
                  0.40 + 0.25 * (index + 1) / max(1, len(fits)))
    if not seam_records:
        model.warnings.append("offset-only held-out pixels were unavailable")
        return model
    model.validation_mode = "checkerboard-spatial-blocks"
    model.validation_edges = len(seam_records)
    model.components = len(components)
    model.raw_seam_bias = float(np.median(
        [record["raw_bias"] for record in seam_records]))
    model.corrected_seam_bias = float(np.median(
        [record["corrected_bias"] for record in seam_records]))
    model.raw_seam_mad = float(np.median(
        [record["raw_mad"] for record in seam_records]))
    model.corrected_seam_mad = float(np.median(
        [record["corrected_mad"] for record in seam_records]))
    model.raw_seam_p95 = float(np.median(
        [record["raw_p95"] for record in seam_records]))
    model.corrected_seam_p95 = float(np.median(
        [record["corrected_p95"] for record in seam_records]))
    model.orientation_qc = _orientation_seam_qc(seam_records)
    if not _orientation_seams_improve_safely(model.orientation_qc, 0.20):
        model.warnings.append("offset-only held-out seams did not improve safely")
        return model

    full_field = (
        _resize_spatial_field(spatial_model.field_dn, dataset.tile_shape)
        if spatial_model.status == "applied" and spatial_model.field_dn is not None
        else None
    )
    tiles = {(tile.row, tile.col): tile for tile in dataset.tiles}
    clipped = 0
    unsaturated = 0
    try:
        for index, position in enumerate(positions):
            _check_cancel(cancel)
            plane = np.asarray(
                mosaic.read_plane(dataset, tiles[position], channel_key),
                dtype=np.float32)
            corrected = plane - full_field if full_field is not None else plane
            corrected = corrected + np.float32(offsets[position])
            valid = np.isfinite(plane) & (plane > 0.0) & (plane < full_scale)
            clipped += int(np.count_nonzero(
                valid & ((corrected < 0.0) | (corrected > full_scale))))
            unsaturated += int(np.count_nonzero(valid))
            if index % 8 == 0 or index + 1 == len(positions):
                _emit(progress, "offset-only clippingを検証中",
                      0.65 + 0.35 * (index + 1) / len(positions))
    except MosaicCancelled:
        raise
    except Exception as exc:
        model.warnings.append(f"offset-only clipping preflight failed: {exc}")
        return model
    model.new_clip_fraction = clipped / max(1, unsaturated)
    if model.new_clip_fraction > 1e-4:
        model.warnings.append("offset-only correction would create clipped pixels")
        return model

    model.offsets = offsets
    model.accepted_edges = len(fits)
    if any(abs(value) > 1e-3 for value in offsets.values()):
        model.status = "applied"
    return model


def estimate_radiometric_model(dataset, geometry: MosaicGeometry,
                               channel_key: str, progress: Progress = None,
                               cancel: Cancel = None) -> RadiometricModel:
    """Estimate bounded tilewise gain/offset for feathered display mosaics."""

    positions = sorted(geometry.origins)
    model = _identity_radiometric_model(dataset, channel_key)
    dtype = _channel_dtype(dataset, channel_key)
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        model.warnings.append("unsupported dtype; radiometric compensation disabled")
        return model
    full_scale = float(np.iinfo(dtype).max)
    eligible = []
    for edge in geometry.edges:
        residual = (float(np.hypot(*edge.residual_yx))
                    if edge.residual_yx is not None else math.inf)
        if edge.accepted and residual <= 2.0:
            eligible.append(edge)
    model.attempted_edges = len(eligible)
    if not eligible:
        model.warnings.append("no geometrically verified overlaps")
        return model
    downsample = 4 if min(dataset.tile_shape) >= 256 else max(1, min(dataset.tile_shape) // 64)
    cache = _PlaneCache(dataset, channel_key, max_items=max(2, len(positions)),
                        downsample=downsample)
    spatial_model = _cached_spatial_model(
        dataset, geometry, channel_key,
        progress=(lambda message, value: _emit(
            progress, message, 0.45 * value)), cancel=cancel)
    spatial_field = (
        _resize_spatial_field(spatial_model.field_dn, cache.get(positions[0]).shape)
        if spatial_model.status == "applied" and spatial_model.field_dn is not None
        else None
    )

    def offset_fallback(warnings):
        return _estimate_offset_only_model(
            dataset, geometry, channel_key, eligible, cache, downsample,
            spatial_model, spatial_field, full_scale, list(warnings),
            progress=(lambda message, value: _emit(
                progress, message, 0.45 + 0.55 * value)), cancel=cancel)

    fits = []
    edge_qc = []
    for index, edge in enumerate(eligible):
        _check_cancel(cancel)
        try:
            pa, pb = _radiometric_overlap(
                cache, geometry, edge.tile_a, edge.tile_b, downsample,
                field_dn=spatial_field)
            fit = _radiometric_pair_fit(pa, pb, full_scale)
        except Exception as exc:
            fit = {"accepted": False, "reason": f"read_or_overlap_error:{exc}",
                   "samples": 0}
        fit.update(tile_a=edge.tile_a, tile_b=edge.tile_b, axis=edge.axis)
        edge_qc.append({
            **{k: v for k, v in fit.items()
               if k not in {"tile_a", "tile_b"}},
            "row_a": edge.tile_a[0], "column_a": edge.tile_a[1],
            "row_b": edge.tile_b[0], "column_b": edge.tile_b[1],
        })
        if fit.get("accepted"):
            fits.append(fit)
        if index % 8 == 0 or index + 1 == len(eligible):
            _emit(progress, "tile affine overlapを推定中",
                  0.45 + 0.20 * (index + 1) / len(eligible))
    model.edge_qc = edge_qc
    if not fits:
        model.warnings.append("all radiometric overlap fits were rejected")
        return offset_fallback(model.warnings)

    def edge_key(fit):
        a, b = sorted((fit["tile_a"], fit["tile_b"]))
        return fit["axis"], a, b

    # Whole-edge holdout prevents the graph from validating itself on the same
    # overlap equations.  Small synthetic/diagnostic graphs retain all edges;
    # they cannot support an independent graph split and are covered by the
    # direction-invariant pair test instead.
    holdout = []
    if len(fits) >= 12:
        for axis in ("x", "y"):
            oriented = sorted(
                (fit for fit in fits if fit["axis"] == axis), key=edge_key)
            holdout.extend(oriented[2::5])
    holdout_keys = {edge_key(fit) for fit in holdout}
    training = [fit for fit in fits if edge_key(fit) not in holdout_keys]
    if not training:
        model.warnings.append("radiometric holdout left no training edges")
        return offset_fallback(model.warnings)
    for qc in edge_qc:
        qa = (qc["row_a"], qc["column_a"])
        qb = (qc["row_b"], qc["column_b"])
        a, b = sorted((qa, qb))
        qc["validation_holdout"] = (qc.get("axis"), a, b) in holdout_keys

    def confidence(fit):
        return max(0.05, min(1.0, fit["correlation"] ** 2
                            * fit["inlier_fraction"]
                            * min(1.0, fit["samples"] / 4096.0)))

    gain_constraints = [
        (fit["tile_a"], fit["tile_b"], math.log(fit["slope"]), confidence(fit))
        for fit in training
    ]
    channel_dynamic = (spatial_model.channel_dynamic_range_dn
                       if spatial_model.channel_dynamic_range_dn is not None
                       else _robust_channel_dynamic(cache, positions))
    max_log_gain = math.log(1.10)
    max_offset = min(0.02 * full_scale, max(1.0, 0.25 * channel_dynamic))
    log_gain, gains, offsets = {}, {}, {}
    chosen_prior = None
    for prior_weight in (0.04, 0.08, 0.12, 0.20, 0.30, 0.50, 1.0):
        candidate_log = _solve_scalar_graph(
            positions, gain_constraints, prior_weight=prior_weight)
        candidate_gains = {p: math.exp(candidate_log[p]) for p in positions}
        offset_constraints = [
            (fit["tile_a"], fit["tile_b"],
             candidate_gains[fit["tile_a"]]
             * fit["intercept_normalized"] * full_scale,
             confidence(fit))
            for fit in training
        ]
        candidate_offsets = _solve_scalar_graph(
            positions, offset_constraints, prior_weight=prior_weight)
        if (all(abs(candidate_log[p]) <= max_log_gain for p in positions)
                and all(abs(candidate_offsets[p]) <= max_offset for p in positions)):
            log_gain, gains, offsets = (
                candidate_log, candidate_gains, candidate_offsets)
            chosen_prior = prior_weight
            break
    if chosen_prior is None:
        model.warnings.append("no bounded graph solution; identity retained")
        return offset_fallback(model.warnings)
    if chosen_prior > 0.04:
        model.warnings.append(
            f"identity regularization increased to {chosen_prior:g} for safe bounds")

    adjacency = {p: set() for p in positions}
    for fit in training:
        adjacency[fit["tile_a"]].add(fit["tile_b"])
        adjacency[fit["tile_b"]].add(fit["tile_a"])
    unseen = set(positions)
    components = []
    while unseen:
        component = {unseen.pop()}
        pending = list(component)
        while pending:
            linked = adjacency[pending.pop()] & unseen
            unseen.difference_update(linked)
            component.update(linked)
            pending.extend(linked)
        components.append(component)

    # Tiny islands cannot distinguish an exposure change from one unusual piece
    # of tissue.  Keep them native instead of propagating one bridge fit.
    minimum_component = 8 if len(positions) >= 24 else 2
    inactive = set().union(*(
        component for component in components if len(component) < minimum_component
    )) if components else set()
    if inactive:
        for position in inactive:
            gains[position], offsets[position] = 1.0, 0.0
        model.warnings.append(
            f"{len(inactive)} tiles in small radiometric components retained identity")

    validation = holdout if holdout else training
    model.validation_mode = "whole-edge-holdout" if holdout else "small-graph-pair"
    seam_records = []
    for index, fit in enumerate(validation):
        _check_cancel(cancel)
        a, b = fit["tile_a"], fit["tile_b"]
        try:
            pa, pb = _radiometric_overlap(
                cache, geometry, a, b, downsample, field_dn=spatial_field)
        except Exception:
            continue
        finite = np.isfinite(pa) & np.isfinite(pb)
        pa, pb = pa[finite].astype(np.float64), pb[finite].astype(np.float64)
        if pa.size < 256:
            continue
        raw_delta = pa - pb
        adjusted_delta = gains[a] * pa + offsets[a] - gains[b] * pb - offsets[b]
        raw_center = float(np.median(raw_delta))
        adjusted_center = float(np.median(adjusted_delta))
        seam_records.append({
            "axis": fit["axis"],
            "raw_bias": abs(raw_center),
            "corrected_bias": abs(adjusted_center),
            "raw_mad": 1.4826 * float(
                np.median(np.abs(raw_delta - raw_center))),
            "corrected_mad": 1.4826 * float(
                np.median(np.abs(adjusted_delta - adjusted_center))),
            "raw_p95": float(np.percentile(np.abs(raw_delta), 95)),
            "corrected_p95": float(np.percentile(np.abs(adjusted_delta), 95)),
        })
        if index % 4 == 0 or index + 1 == len(validation):
            _emit(progress, "tile affine holdoutを検証中",
                  0.65 + 0.10 * (index + 1) / len(validation))

    model.validation_edges = len(seam_records)
    model.components = len(components)
    if not seam_records:
        model.warnings.append("radiometric validation overlaps were unavailable")
        return offset_fallback(model.warnings)
    model.raw_seam_bias = float(np.median(
        [record["raw_bias"] for record in seam_records]))
    model.corrected_seam_bias = float(np.median(
        [record["corrected_bias"] for record in seam_records]))
    model.raw_seam_mad = float(np.median(
        [record["raw_mad"] for record in seam_records]))
    model.corrected_seam_mad = float(np.median(
        [record["corrected_mad"] for record in seam_records]))
    model.raw_seam_p95 = float(np.median(
        [record["raw_p95"] for record in seam_records]))
    model.corrected_seam_p95 = float(np.median(
        [record["corrected_p95"] for record in seam_records]))
    model.orientation_qc = _orientation_seam_qc(seam_records)
    safe_validation = _orientation_seams_improve_safely(
        model.orientation_qc, 0.20)
    if not safe_validation:
        model.warnings.append(
            "independent seam validation did not improve; tile affine retained identity")
        return offset_fallback(model.warnings)

    # Combined common-field + tile-affine preflight is exact and streaming.
    # A presentation correction that creates new clipped source values is not
    # accepted merely because a seam statistic improved.
    full_field = (
        _resize_spatial_field(spatial_model.field_dn, dataset.tile_shape)
        if spatial_model.status == "applied" and spatial_model.field_dn is not None
        else None
    )
    tiles = {(tile.row, tile.col): tile for tile in dataset.tiles}
    clipped = 0
    unsaturated = 0
    try:
        for index, position in enumerate(positions):
            _check_cancel(cancel)
            plane = np.asarray(
                mosaic.read_plane(dataset, tiles[position], channel_key),
                dtype=np.float32)
            source = plane - full_field if full_field is not None else plane
            adjusted = source * np.float32(gains[position]) + np.float32(offsets[position])
            valid = np.isfinite(plane) & (plane > 0.0) & (plane < full_scale)
            clipped += int(np.count_nonzero(
                valid & ((adjusted < 0.0) | (adjusted > full_scale))))
            unsaturated += int(np.count_nonzero(valid))
            if index % 8 == 0 or index + 1 == len(positions):
                _emit(progress, "tile affine clippingを検証中",
                      0.75 + 0.25 * (index + 1) / len(positions))
    except MosaicCancelled:
        raise
    except Exception as exc:
        model.warnings.append(f"combined clipping preflight failed: {exc}")
        return offset_fallback(model.warnings)
    model.new_clip_fraction = clipped / max(1, unsaturated)
    if model.new_clip_fraction > 1e-4:
        model.warnings.append(
            "tile affine would create too many clipped pixels; identity retained")
        return offset_fallback(model.warnings)

    model.gains = gains
    model.offsets = offsets
    model.accepted_edges = len(training)
    if training and any(abs(gains[p] - 1.0) > 1e-4 or abs(offsets[p]) > 1e-3
                    for p in positions):
        model.status = "applied"
        model.mode = "affine"
    return model


def _cached_radiometric_model(dataset, geometry: MosaicGeometry,
                              channel_key: str, progress: Progress = None,
                              cancel: Cancel = None) -> RadiometricModel:
    cached = geometry.radiometric_models.get(channel_key)
    if cached is not None:
        return cached
    try:
        model = estimate_radiometric_model(
            dataset, geometry, channel_key, progress=progress, cancel=cancel)
    except MosaicCancelled:
        raise
    except Exception as exc:
        model = _identity_radiometric_model(
            dataset, channel_key, f"radiometric estimation failed: {exc}")
    geometry.radiometric_models[channel_key] = model
    return model


# ---------------------------------------------------------------- rendering

def _bilinear_patch(source: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Sample a 2-D source at separable floating-point coordinates."""

    h, w = source.shape
    iy = np.clip(np.floor(yy).astype(np.int64), 0, max(0, h - 2))
    ix = np.clip(np.floor(xx).astype(np.int64), 0, max(0, w - 2))
    fy = (yy - iy)[:, None].astype(np.float32)
    fx = (xx - ix)[None, :].astype(np.float32)
    a = source[np.ix_(iy, ix)].astype(np.float32)
    b = source[np.ix_(iy, np.minimum(ix + 1, w - 1))].astype(np.float32)
    c = source[np.ix_(np.minimum(iy + 1, h - 1), ix)].astype(np.float32)
    d = source[np.ix_(np.minimum(iy + 1, h - 1),
                      np.minimum(ix + 1, w - 1))].astype(np.float32)
    return (a * (1.0 - fy) * (1.0 - fx)
            + b * (1.0 - fy) * fx + c * fy * (1.0 - fx) + d * fy * fx)


def _nearest_patch(source: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Sample native source values without inventing intermediate intensities."""

    iy = np.clip(np.floor(yy + 0.5).astype(np.int64), 0, source.shape[0] - 1)
    ix = np.clip(np.floor(xx + 0.5).astype(np.int64), 0, source.shape[1] - 1)
    return source[np.ix_(iy, ix)]


class ChunkRenderer:
    """Render arbitrary output blocks without allocating the full mosaic."""

    def __init__(self, dataset, geometry: MosaicGeometry, channel_key: str,
                 *, downsample: int = 1, blend_mode: str = "feather",
                 cache_items: int = 12, output_dtype=None,
                 radiometric_correction: bool = True,
                 progress: Progress = None, cancel: Cancel = None):
        if blend_mode not in {"feather", "nearest"}:
            raise ValueError("blend_mode must be feather or nearest")
        self.dataset = dataset
        self.geometry = geometry
        self.channel_key = channel_key
        self.downsample = max(1, int(downsample))
        self.blend_mode = blend_mode
        self.cache = _PlaneCache(dataset, channel_key, max_items=cache_items,
                                 downsample=self.downsample)
        self.origins = {
            p: (v[0] / self.downsample, v[1] / self.downsample)
            for p, v in geometry.origins.items()
        }
        th, tw = dataset.tile_shape
        self.source_shape = (math.ceil(th / self.downsample),
                             math.ceil(tw / self.downsample))
        self.output_shape = (math.ceil(geometry.output_shape[0] / self.downsample),
                             math.ceil(geometry.output_shape[1] / self.downsample))
        self.output_dtype = np.dtype(output_dtype or self._channel_dtype())
        if blend_mode == "feather" and radiometric_correction:
            _check_cancel(cancel)
            self.spatial_model = _cached_spatial_model(
                dataset, geometry, channel_key,
                progress=(lambda message, value: _emit(
                    progress, message, 0.50 * value)), cancel=cancel)
            self.spatial_field = (
                _resize_spatial_field(self.spatial_model.field_dn, self.source_shape)
                if self.spatial_model.status == "applied"
                and self.spatial_model.field_dn is not None else None
            )
            self.radiometric_model = _cached_radiometric_model(
                dataset, geometry, channel_key,
                progress=(lambda message, value: _emit(
                    progress, message, 0.50 + 0.50 * value)), cancel=cancel)
        else:
            # Quantitative nearest output must neither estimate nor apply any
            # presentation radiometry; native source DN remain untouched.
            self.spatial_model = _identity_spatial_model(channel_key)
            self.spatial_field = None
            self.radiometric_model = _identity_radiometric_model(
                dataset, channel_key)
        # ``origins`` locate source pixel centres.  A border pixel therefore
        # contributes over half a pixel beyond the first/last centre.  Using
        # only [origin, origin + size - 1] drops one outer row/column whenever
        # a fractional origin lies on the corresponding side of an output
        # pixel centre (including the native-DN scientific nearest path).
        # Divide the native half-pixel support bounds themselves.  Building
        # bounds from ``ceil(tile_size / downsample)`` overextends partial
        # source blocks and can leave the true far edge outside the box.
        self._boxes = {
            p: ((native[0] - 0.5) / self.downsample,
                (native[1] - 0.5) / self.downsample,
                (native[0] + th - 0.5) / self.downsample,
                (native[1] + tw - 0.5) / self.downsample)
            for p, native in geometry.origins.items()
        }

    def _channel_dtype(self):
        for channel in self.dataset.channels:
            if channel.key == self.channel_key:
                return channel.dtype
        return self.dataset.dtype

    def render_block(self, y0: int, x0: int, height: int, width: int) -> np.ndarray:
        y1, x1 = min(self.output_shape[0], y0 + height), min(self.output_shape[1], x0 + width)
        bh, bw = max(0, y1 - y0), max(0, x1 - x0)
        if bh == 0 or bw == 0:
            return np.zeros((0, 0), self.output_dtype)
        acc = np.zeros((bh, bw), np.float32)
        weight = np.zeros((bh, bw), np.float32)
        best = np.full((bh, bw), -1.0, np.float32) if self.blend_mode == "nearest" else None
        sh, sw = self.source_shape

        for pos, (box_y0, box_x0, box_y1, box_x1) in self._boxes.items():
            gy0 = max(y0, int(math.ceil(box_y0)))
            gx0 = max(x0, int(math.ceil(box_x0)))
            # The high side of a pixel-support box is exclusive.  ``ceil`` is
            # the first integer coordinate outside that half-open interval.
            gy1 = min(y1, int(math.ceil(box_y1)))
            gx1 = min(x1, int(math.ceil(box_x1)))
            if gy1 <= gy0 or gx1 <= gx0:
                continue
            source = self.cache.get(pos)
            ty0, tx0 = self.origins[pos]
            yy = np.arange(gy0, gy1, dtype=np.float32) - ty0
            xx = np.arange(gx0, gx1, dtype=np.float32) - tx0
            if self.blend_mode == "nearest":
                patch = _nearest_patch(source, yy, xx)
            else:
                # Edge support extends half a pixel past the border centre;
                # replicate the border value instead of extrapolating it.
                sample_yy = np.clip(yy, 0.0, max(0, sh - 1))
                sample_xx = np.clip(xx, 0.0, max(0, sw - 1))
                patch = _bilinear_patch(source, sample_yy, sample_xx)
            if self.blend_mode == "feather":
                if self.spatial_field is not None:
                    patch = patch - _bilinear_patch(
                        self.spatial_field, sample_yy, sample_xx)
                gain = self.radiometric_model.gains.get(pos, 1.0)
                offset = self.radiometric_model.offsets.get(pos, 0.0)
                if gain != 1.0 or offset != 0.0:
                    patch = patch * np.float32(gain) + np.float32(offset)
            fy = np.minimum(yy + 1.0, sh - yy)
            fx = np.minimum(xx + 1.0, sw - xx)
            fw = np.minimum(fy[:, None], fx[None, :]).astype(np.float32)
            fw /= max(1.0, min(sh, sw) / 2.0)
            oy0, ox0 = gy0 - y0, gx0 - x0
            oy1, ox1 = oy0 + patch.shape[0], ox0 + patch.shape[1]
            finite = np.isfinite(patch)
            if self.blend_mode == "nearest":
                view = best[oy0:oy1, ox0:ox1]
                use = finite & (fw > view)
                target = acc[oy0:oy1, ox0:ox1]
                target[use] = patch[use]
                view[use] = fw[use]
            else:
                valid_w = np.where(finite, fw, 0.0)
                acc[oy0:oy1, ox0:ox1] += np.where(finite, patch, 0.0) * valid_w
                weight[oy0:oy1, ox0:ox1] += valid_w

        if self.blend_mode == "nearest":
            out = np.where(best >= 0.0, acc, 0.0)
        else:
            out = np.divide(acc, weight, out=np.zeros_like(acc), where=weight > 0.0)
        if np.issubdtype(self.output_dtype, np.integer):
            info = np.iinfo(self.output_dtype)
            return np.clip(np.rint(out), info.min, info.max).astype(self.output_dtype)
        return out.astype(self.output_dtype)

    def tiles(self, tile_size: int = 512, *, progress: Progress = None,
              cancel: Cancel = None, progress_range=(0.0, 1.0)) -> Iterable[np.ndarray]:
        h, w = self.output_shape
        ny, nx = math.ceil(h / tile_size), math.ceil(w / tile_size)
        total = max(1, ny * nx)
        lo, hi = progress_range
        done = 0
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                _check_cancel(cancel)
                block = self.render_block(y, x, min(tile_size, h - y),
                                          min(tile_size, w - x))
                done += 1
                _emit(progress, "モザイクを描画中", lo + (hi - lo) * done / total)
                yield block


def render_preview_channels(dataset, geometry: MosaicGeometry,
                            channel_keys: Optional[list[str]] = None,
                            max_side: int = 4096, blend_mode: str = "feather",
                            progress: Progress = None, cancel: Cancel = None,
                            radiometric_correction: bool = True) -> tuple[dict, int]:
    """Render bounded channel mosaics for the interactive viewer."""

    channel_keys = channel_keys or [c.key for c in dataset.channels]
    ds = max(1, int(math.ceil(max(geometry.output_shape) / max(512, max_side))))
    images = {}
    count = max(1, len(channel_keys))
    for ci, key in enumerate(channel_keys):
        # Model estimation/preflight dominates large presentation datasets, so
        # reserve most of each channel's progress range for renderer setup.
        init_fraction = 0.70 if (
            blend_mode == "feather" and radiometric_correction) else 0.0
        renderer = ChunkRenderer(dataset, geometry, key, downsample=ds,
                                 blend_mode=blend_mode, cache_items=14,
                                 radiometric_correction=radiometric_correction,
                                 progress=(lambda message, value, ci=ci: _emit(
                                     progress,
                                     f"プレビュ {ci + 1}/{count}: {message}",
                                     (ci + init_fraction * value) / count)),
                                 cancel=cancel)
        _emit(progress, key, (ci + init_fraction) / count)
        h, w = renderer.output_shape
        out = np.zeros((h, w), renderer.output_dtype)
        block = 768
        ny, nx = math.ceil(h / block), math.ceil(w / block)
        total = max(1, ny * nx)
        done = 0
        for y in range(0, h, block):
            for x in range(0, w, block):
                _check_cancel(cancel)
                part = renderer.render_block(y, x, min(block, h - y), min(block, w - x))
                out[y:y + part.shape[0], x:x + part.shape[1]] = part
                done += 1
                frac = (ci + init_fraction
                        + (1.0 - init_fraction) * done / total) / count
                _emit(progress, f"プレビュー {ci + 1}/{count}: {key}", frac)
        images[key] = out
    return images, ds


# ---------------------------------------------------------------- presentation export

def _font(size: int):
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=max(8, int(size)))
        except Exception:
            continue
    return ImageFont.load_default()


def nice_scale_length(pixel_um: float, width_px: int, fraction: float = 0.20) -> float:
    if pixel_um <= 0 or width_px <= 0:
        return 0.0
    target = pixel_um * width_px * fraction
    exponent = 10.0 ** math.floor(math.log10(max(target, 1e-9)))
    return max(v * exponent for v in (1.0, 2.0, 5.0) if v * exponent <= target)


def draw_scale_bar(image: Image.Image, pixel_um: float,
                   spec: Optional[ScaleBarSpec] = None) -> Image.Image:
    """Burn an editable scale bar into a presentation image copy."""

    spec = spec or ScaleBarSpec()
    out = image.convert("RGB").copy()
    if not spec.visible or pixel_um <= 0:
        return out
    length_um = float(nice_scale_length(pixel_um, out.width)
                      if spec.length_um is None else spec.length_um)
    if not math.isfinite(length_um) or length_um <= 0:
        raise ValueError("scale-bar length must be finite and positive")
    length_px = max(1, int(round(length_um / pixel_um)))
    margin = max(0, int(spec.margin_px))
    thick = max(1, int(spec.thickness_px))
    font = _font(spec.font_size_px)
    if spec.label:
        text = spec.label
    elif length_um >= 1000 and abs(length_um / 1000 - round(length_um / 1000)) < 1e-8:
        text = f"{length_um / 1000:g} mm"
    else:
        text = f"{length_um:g} µm"
    probe = ImageDraw.Draw(out)
    bbox = probe.textbbox((0, 0), text, font=font) if spec.show_label else (0, 0, 0, 0)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    total_h = thick + (max(4, thick) + text_h if spec.show_label else 0)
    available_w = out.width - 2 * margin
    available_h = out.height - 2 * margin
    if length_px > available_w:
        raise ValueError(
            f"scale bar ({length_px} px) exceeds available image width "
            f"({max(0, available_w)} px)"
        )
    if max(length_px, text_w) > available_w or total_h > available_h:
        raise ValueError("scale-bar label or height does not fit inside the image")
    pos = spec.position.lower()
    x = out.width - margin - length_px if "right" in pos else margin
    y = out.height - margin - total_h if "bottom" in pos else margin
    pad = max(4, thick)
    if spec.background in {"dark", "light"}:
        bg = (0, 0, 0) if spec.background == "dark" else (255, 255, 255)
        probe.rounded_rectangle(
            (x - pad, y - pad, x + max(length_px, text_w) + pad,
             y + total_h + pad), radius=max(3, pad // 2), fill=bg)
    color = tuple(int(np.clip(c, 0, 255)) for c in spec.color)
    probe.rectangle((x, y, x + length_px, y + thick - 1), fill=color)
    if spec.show_label:
        probe.text((x, y + thick + max(4, thick)), text, fill=color, font=font)
    return out


def atomic_save_presentation(image: Image.Image, path: Path, fmt: str,
                             *, dpi: int = 300, cancel: Cancel = None):
    """Save PNG, TIFF, or PDF without replacing a good destination on failure."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part",
                                     dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        _check_cancel(cancel)
        if fmt.lower() == "pdf":
            image.convert("RGB").save(temp, "PDF", resolution=float(dpi), quality=95,
                                      subsampling=0)
        elif fmt.lower() in {"tif", "tiff"}:
            image.save(temp, "TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
        else:
            image.save(temp, "PNG", compress_level=6, dpi=(dpi, dpi))
        if fmt.lower() == "pdf":
            size = temp.stat().st_size
            with temp.open("rb") as handle:
                header = handle.read(8)
                handle.seek(max(0, size - 4096))
                trailer = handle.read()
            if size < 100 or not header.startswith(b"%PDF-") or b"%%EOF" not in trailer:
                raise MosaicExportError("PDF validation failed")
        else:
            with Image.open(temp) as check:
                check.verify()
        # Pillow compression itself is not interruptible.  A cancellation that
        # arrives during that codec call is honoured here, before the staged
        # file can replace an existing destination.
        _check_cancel(cancel)
        # Windows rejects fsync on a read-only descriptor.  Reopen the staged
        # file read/write even though no more bytes are changed, then make the
        # completed presentation durable before the atomic replacement.
        with temp.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


# ---------------------------------------------------------------- scientific OME-TIFF

def _pyramid_levels(shape: tuple[int, int], maximum: int = 5) -> list[int]:
    levels = []
    factor = 2
    while len(levels) < maximum and max(math.ceil(shape[0] / factor),
                                        math.ceil(shape[1] / factor)) > 900:
        levels.append(factor)
        factor *= 2
    if not levels and max(shape) > 1400:
        levels.append(2)
    return levels


def _level_tiles(array: np.memmap, tile_size: int,
                 cancel: Cancel = None) -> Iterable[np.ndarray]:
    for channel in range(array.shape[0]):
        for y in range(0, array.shape[1], tile_size):
            for x in range(0, array.shape[2], tile_size):
                _check_cancel(cancel)
                yield np.asarray(array[channel, y:y + tile_size, x:x + tile_size])


def _codec_options() -> dict:
    try:
        import imagecodecs  # noqa: F401
        return {"compression": "zlib", "compressionargs": {"level": 6},
                "predictor": True, "maxworkers": 2}
    except ImportError:
        # Source checkouts remain usable, but the packaged app requires imagecodecs.
        return {"compression": None, "predictor": False, "maxworkers": 1}


def _channel_dtype(dataset, key: str) -> np.dtype:
    for channel in dataset.channels:
        if channel.key == key:
            return np.dtype(channel.dtype)
    return np.dtype(dataset.dtype)


def write_pyramidal_ome(dataset, geometry: MosaicGeometry, path: Path,
                        channel_keys: Optional[list[str]] = None,
                        *, blend_mode: str = "nearest", tile_size: int = 512,
                        progress: Progress = None, cancel: Cancel = None,
                        qc_extra: Optional[dict] = None,
                        radiometric_correction: bool = False) -> Path:
    """Stage and validate BigTIFF, QC JSON, and alignment CSV as one export set."""

    import tifffile

    path = Path(path)
    channel_keys = ([c.key for c in dataset.channels]
                    if channel_keys is None else list(channel_keys))
    if not channel_keys:
        raise MosaicExportError("no channels selected")
    if len(channel_keys) != len(set(channel_keys)):
        raise MosaicExportError("duplicate channels selected")
    channel_map = {c.key: c for c in dataset.channels}
    unknown = [k for k in channel_keys if k not in channel_map]
    if unknown:
        raise MosaicExportError(f"unknown channels: {', '.join(unknown)}")
    promoted = np.result_type(*[_channel_dtype(dataset, k) for k in channel_keys])
    # Normalise byte order as TIFF pages are decoded into native arrays.  Only
    # the two lossless instrument depths have an integer scientific contract;
    # other numeric inputs are deliberately promoted to float32.
    if promoted.kind == "u" and promoted.itemsize == 1:
        out_dtype = np.dtype("uint8")
    elif promoted.kind == "u" and promoted.itemsize == 2:
        out_dtype = np.dtype("uint16")
    else:
        out_dtype = np.dtype("float32")
    h, w = geometry.output_shape
    if h <= 0 or w <= 0:
        raise MosaicExportError("mosaic output dimensions must be positive")
    factors = _pyramid_levels((h, w))
    # Every internal render-block boundary must also be a downsample boundary.
    # Otherwise independently reducing adjacent blocks overlaps or skips pixels
    # at the largest pyramid factor (for example tile_size=144, factor=32).
    alignment = max(16, factors[-1] if factors else 16)
    tile_size = max(128, int(math.ceil(tile_size / alignment) * alignment))
    pixel_y, pixel_x = dataset.pixel_size_um_yx
    if any(value is None or not np.isfinite(value) or value <= 0
           for value in (pixel_y, pixel_x)):
        raise MosaicExportError(
            "finite positive PhysicalSizeX/Y is required for OME-TIFF export"
        )
    export_id = uuid.uuid4().hex
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.stem}-staging-", dir=path.parent))
    temp_tiff = staging / path.name
    sidecar_path = Path(str(path) + ".mosaic-qc.json")
    temp_sidecar = staging / sidecar_path.name
    alignment_path = path.with_name(path.stem + "_alignment.csv")
    temp_alignment = staging / alignment_path.name
    pyramid_maps = []
    try:
        for li, factor in enumerate(factors):
            shape = (len(channel_keys), math.ceil(h / factor), math.ceil(w / factor))
            mm = np.memmap(staging / f"level-{li + 1}.dat", mode="w+",
                           dtype=out_dtype, shape=shape)
            pyramid_maps.append(mm)

        tiles_per_channel = math.ceil(h / tile_size) * math.ceil(w / tile_size)
        total_base_tiles = len(channel_keys) * tiles_per_channel
        initialization_units = (
            tiles_per_channel
            if blend_mode == "feather" and radiometric_correction else 0
        )
        total_base_units = len(channel_keys) * (
            initialization_units + tiles_per_channel)
        counter = 0
        radiometric_models = {}
        spatial_models = {}

        def base_tiles():
            nonlocal counter
            for ci, key in enumerate(channel_keys):
                channel_base = ci * (initialization_units + tiles_per_channel)
                renderer = ChunkRenderer(dataset, geometry, key, blend_mode=blend_mode,
                                         cache_items=14, output_dtype=out_dtype,
                                         radiometric_correction=radiometric_correction,
                                         progress=(lambda message, value,
                                                   channel_base=channel_base: _emit(
                                             progress, message,
                                             0.82 * (channel_base
                                                     + initialization_units * value)
                                             / max(1, total_base_units))),
                                         cancel=cancel)
                radiometric_models[key] = renderer.radiometric_model
                spatial_models[key] = renderer.spatial_model
                channel_counter = 0
                for y in range(0, h, tile_size):
                    for x in range(0, w, tile_size):
                        _check_cancel(cancel)
                        block = renderer.render_block(y, x, min(tile_size, h - y),
                                                      min(tile_size, w - x))
                        for mm, factor in zip(pyramid_maps, factors):
                            small = _block_mean(block, factor).astype(out_dtype, copy=False)
                            sy, sx = y // factor, x // factor
                            ey = min(mm.shape[1], sy + small.shape[0])
                            ex = min(mm.shape[2], sx + small.shape[1])
                            mm[ci, sy:ey, sx:ex] = small[:ey - sy, :ex - sx]
                        counter += 1
                        channel_counter += 1
                        _emit(progress, "フル解像度OME-TIFFを書き出し中",
                              0.82 * (channel_base + initialization_units
                                      + channel_counter)
                              / max(1, total_base_units))
                        yield block

        names = [channel_map[k].label for k in channel_keys]
        metadata = {
            "axes": "CYX",
            "Channel": {"Name": names},
            "PhysicalSizeX": float(pixel_x), "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": float(pixel_y), "PhysicalSizeYUnit": "µm",
            "Description": "Wide-area mosaic generated by BZ Studio",
            "MapAnnotation": {
                "Namespace": "openmicroscopy.org/PyramidResolution",
                "ExportId": export_id,
                **{str(i + 1): f"{math.ceil(w / f)} {math.ceil(h / f)}"
                   for i, f in enumerate(factors)},
            },
        }
        options = {
            "photometric": "minisblack", "tile": (tile_size, tile_size),
            "resolutionunit": "CENTIMETER", **_codec_options(),
        }
        with tifffile.TiffWriter(temp_tiff, bigtiff=True, ome=True) as writer:
            writer.write(
                base_tiles(), shape=(len(channel_keys), h, w), dtype=out_dtype,
                subifds=len(factors),
                resolution=(1e4 / float(pixel_x), 1e4 / float(pixel_y)),
                metadata=metadata, **options)
            for li, (factor, mm) in enumerate(zip(factors, pyramid_maps)):
                mm.flush()
                _check_cancel(cancel)
                writer.write(
                    _level_tiles(mm, tile_size, cancel), shape=mm.shape,
                    dtype=out_dtype,
                    subfiletype=1,
                    resolution=(1e4 / (float(pixel_x) * factor),
                                1e4 / (float(pixel_y) * factor)),
                    metadata=None, **options)
                _emit(progress, f"ピラミッド {li + 1}/{len(factors)} を書き出し中",
                      0.82 + 0.13 * (li + 1) / max(1, len(factors)))
        if counter != total_base_tiles:
            raise MosaicExportError("OME-TIFF validation failed: incomplete base image")

        # Reopen before replacing a destination. A truncated but syntactically
        # accepted writer result must never displace a known-good image.
        with tifffile.TiffFile(temp_tiff) as check:
            if not check.series:
                raise MosaicExportError("OME-TIFF validation failed: no image series")
            if not check.ome_metadata or export_id not in check.ome_metadata:
                raise MosaicExportError("OME-TIFF validation failed: missing OME metadata")
            series = check.series[0]
            base_shapes = {(len(channel_keys), h, w)}
            base_axes = {"CYX"}
            if len(channel_keys) == 1:
                # tifffile/OME legitimately squeezes a singleton C dimension.
                base_shapes.add((h, w))
                base_axes.add("YX")
            if tuple(series.shape) not in base_shapes or series.axes not in base_axes:
                raise MosaicExportError("OME-TIFF validation failed: unexpected shape")
            if len(series.levels) != len(factors) + 1:
                raise MosaicExportError("OME-TIFF validation failed: incomplete pyramid")
            if np.dtype(series.dtype) != out_dtype:
                raise MosaicExportError("OME-TIFF validation failed: unexpected dtype")
            for level, factor in zip(series.levels[1:], factors):
                lh, lw = math.ceil(h / factor), math.ceil(w / factor)
                level_shapes = {(len(channel_keys), lh, lw)}
                level_axes = {"CYX"}
                if len(channel_keys) == 1:
                    level_shapes.add((lh, lw))
                    level_axes.add("YX")
                if (tuple(level.shape) not in level_shapes
                        or level.axes not in level_axes
                        or np.dtype(level.dtype) != out_dtype):
                    raise MosaicExportError(
                        "OME-TIFF validation failed: invalid pyramid level"
                    )

        _check_cancel(cancel)

        qc = {
            "schema_version": "bz-studio-mosaic/1.0",
            "export_id": export_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(dataset.root), "gci": str(dataset.gci_path),
            "output": str(path), "dataset": dataset.name,
            "alignment_csv": str(alignment_path),
            "grid_shape_rc": list(dataset.grid_shape),
            "tile_shape_yx": list(dataset.tile_shape),
            "pixel_size_um_yx": list(dataset.pixel_size_um_yx),
            "dtype": str(out_dtype), "channels": names,
            "blend_mode": blend_mode, "pyramid_factors": factors,
            "radiometric_compensation": {
                "enabled_for_feather_only": bool(
                    radiometric_correction and blend_mode == "feather"),
                "nearest_preserves_native_values": blend_mode == "nearest",
                "channels": {key: radiometric_models[key].to_dict()
                             for key in channel_keys},
                "common_additive_field": {
                    key: spatial_models[key].to_dict() for key in channel_keys
                },
            },
            "geometry": geometry.to_dict(),
            "dataset_warnings": list(getattr(dataset, "warnings", [])),
        }
        if qc_extra:
            qc["presentation"] = qc_extra
        temp_sidecar.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
        json.loads(temp_sidecar.read_text(encoding="utf-8"))
        _write_edge_csv_contents(temp_alignment, geometry, export_id=export_id)
        _check_cancel(cancel)
        # Multiple directory entries cannot be committed with one POSIX rename.
        # Commit the two small audit files first and restore both if a later
        # replace fails.  The TIFF is last, so no fallible write follows it.
        auxiliary = [(temp_sidecar, sidecar_path), (temp_alignment, alignment_path)]
        backups = {}
        for _, target in auxiliary:
            backup = staging / f"previous-{target.name}"
            if target.is_file():
                shutil.copy2(target, backup)
                backups[target] = backup
            else:
                backups[target] = None
        committed = []
        try:
            for temp_file, target in auxiliary:
                os.replace(temp_file, target)
                committed.append(target)
            os.replace(temp_tiff, path)
        except Exception as commit_exc:
            rollback_errors = []
            for target in reversed(committed):
                try:
                    backup = backups[target]
                    if backup is not None:
                        os.replace(backup, target)
                    else:
                        target.unlink(missing_ok=True)
                except Exception as rollback_exc:
                    rollback_errors.append(f"{target.name}: {rollback_exc}")
            if rollback_errors:
                raise MosaicExportError(
                    "export commit failed and audit rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from commit_exc
            raise
        # A presentation-only callback must not turn a completed, consistent
        # commit into a reported export failure.
        try:
            _emit(progress, "OME-TIFFの検証が完了しました", 1.0)
        except Exception:
            pass
        return path
    except MosaicCancelled:
        raise
    except Exception as exc:
        raise MosaicExportError(str(exc)) from exc
    finally:
        for mm in pyramid_maps:
            try:
                mm.flush()
            except Exception:
                pass
            try:
                mmap = getattr(mm, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            except Exception:
                pass
        pyramid_maps.clear()
        shutil.rmtree(staging, ignore_errors=True)


def _write_edge_csv_contents(path: Path, geometry: MosaicGeometry,
                             export_id: str = "") -> None:
    columns = [
        "export_id", "axis", "row_a", "column_a", "row_b", "column_b",
        "accepted", "reason",
        "stage_dy", "stage_dx", "measured_dy", "measured_dx", "correction_dy",
        "correction_dx", "ncc", "psr", "peak_margin", "texture", "weight",
        "residual_dy", "residual_dx",
    ]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for edge in geometry.edges:
            row = {
                "export_id": export_id,
                "axis": edge.axis, "row_a": edge.tile_a[0],
                "column_a": edge.tile_a[1], "row_b": edge.tile_b[0],
                "column_b": edge.tile_b[1], "accepted": int(edge.accepted),
                "reason": edge.reason, "stage_dy": edge.stage_delta_yx[0],
                "stage_dx": edge.stage_delta_yx[1], "measured_dy": edge.shift_yx[0],
                "measured_dx": edge.shift_yx[1], "correction_dy": edge.correction_yx[0],
                "correction_dx": edge.correction_yx[1], "ncc": edge.ncc,
                "psr": edge.psr, "peak_margin": edge.peak_margin,
                "texture": edge.texture, "weight": edge.weight,
                "residual_dy": edge.residual_yx[0] if edge.residual_yx else "",
                "residual_dx": edge.residual_yx[1] if edge.residual_yx else "",
            }
            writer.writerow(row)


def write_edge_csv(path: Path, geometry: MosaicGeometry, export_id: str = ""):
    """Write an auditable, atomic edge table beside the JSON QC."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part",
                                     dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        _write_edge_csv_contents(temp, geometry, export_id=export_id)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
