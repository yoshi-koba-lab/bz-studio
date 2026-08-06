"""Stitch BZ-X raw tiles into whole-well mosaics.

The instrument writes every field of view as a standard OME-TIFF under
``<experiment>/<well>/X###Y###/``.  This module reassembles them without the
vendor's analysis software:

  * tile offsets are recovered from the images themselves (phase correlation on
    the overlapping strips), so no proprietary metadata is needed;
  * one geometry is measured per well and reused for every channel and Z slice,
    which keeps channels perfectly registered;
  * tiles are feather-blended, so seams are smoother than the Analyzer's.

Nothing here touches Qt, so it can be unit-tested and scripted.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageSequence

# <prefix>_<well>_X###Y###[_Z###]_<CH…>.bz.ome.tif
TILE_RE = re.compile(
    r"_X(?P<x>\d+)Y(?P<y>\d+)(?:_Z(?P<z>\d+))?_(?P<ch>CH[\w-]+)\.bz\.ome\.tiff?$",
    re.IGNORECASE,
)
POS_RE = re.compile(r"^X(\d+)Y(\d+)$", re.IGNORECASE)


@dataclass
class Plane:
    """One stitchable image plane: a channel (and its sub-channel page)."""
    channel: str          # file-level channel tag, e.g. "CH4" or "CHF"
    page: int             # page index inside the OME-TIFF
    name: str             # human name from OME metadata, e.g. "DAPI"
    color: Optional[tuple] = None   # (r,g,b) if the metadata provides one

    @property
    def key(self) -> str:
        return f"{self.channel}-{self.page}" if self.page else self.channel

    @property
    def label(self) -> str:
        return self.name or self.key


@dataclass
class WellTiles:
    well: str
    files: dict = field(default_factory=dict)   # (x, y, z, channel) -> Path
    positions: list = field(default_factory=list)
    z_values: list = field(default_factory=list)
    channels: list = field(default_factory=list)
    planes: list = field(default_factory=list)  # list[Plane]
    tile_shape: tuple = (0, 0)

    @property
    def n_tiles(self) -> int:
        return len(self.positions)


# ---------------------------------------------------------------- discovery

def _ome_planes(path: Path, channel: str) -> list:
    """Channel names/colours for each page of a tile file."""
    names, colors = [], []
    try:
        with Image.open(path) as im:
            n_pages = getattr(im, "n_frames", 1)
            desc = im.tag_v2.get(270, "") if hasattr(im, "tag_v2") else ""
    except Exception:
        return [Plane(channel, 0, channel)]
    if desc:
        try:
            root = ET.fromstring(desc)
            ns = {"o": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
            find = ".//o:Channel" if ns else ".//Channel"
            for el in root.findall(find, ns) if ns else root.findall(find):
                names.append(el.get("Name") or "")
                raw = el.get("Color")
                colors.append(_decode_color(raw) if raw is not None else None)
        except ET.ParseError:
            pass
    planes = []
    for i in range(n_pages):
        nm = names[i] if i < len(names) else ""
        planes.append(Plane(channel, i, nm or f"{channel}[{i}]",
                            colors[i] if i < len(colors) else None))
    return planes


def _decode_color(raw) -> Optional[tuple]:
    """OME stores channel colour as a signed 32-bit RGBA int."""
    try:
        v = int(raw) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return None
    r, g, b = (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF
    return (r, g, b) if (r or g or b) else None


def scan_well(well_dir: Path) -> Optional[WellTiles]:
    """Index every tile file under a well directory."""
    well_dir = Path(well_dir)
    wt = WellTiles(well=well_dir.name)
    zs, chans = set(), set()
    for pos_dir in sorted(well_dir.iterdir()):
        if not pos_dir.is_dir() or not POS_RE.match(pos_dir.name):
            continue
        for f in sorted(pos_dir.iterdir()):
            if f.name.startswith("._"):
                continue
            m = TILE_RE.search(f.name)
            if not m:
                continue
            x, y = int(m.group("x")), int(m.group("y"))
            z = int(m.group("z")) if m.group("z") else 0
            ch = m.group("ch").upper()
            wt.files[(x, y, z, ch)] = f
            zs.add(z)
            chans.add(ch)
    if not wt.files:
        return None
    wt.positions = sorted({(x, y) for (x, y, _, _) in wt.files})
    wt.z_values = sorted(zs)
    wt.channels = sorted(chans)
    # Enumerate planes per channel, trying further files if a sample won't open.
    for ch in wt.channels:
        samples = [p for (x, y, z, c), p in wt.files.items() if c == ch]
        planes = None
        for s in samples[:5]:
            planes = _ome_planes(s, ch)
            if planes:
                break
        wt.planes.extend(planes or [Plane(ch, 0, ch)])
    # Tile shape: probe files until one opens — a single unreadable tile must not
    # take down the well (let alone the whole experiment).
    for f in wt.files.values():
        try:
            with Image.open(f) as im:
                wt.tile_shape = (im.height, im.width)
            break
        except Exception:
            continue
    if wt.tile_shape == (0, 0):
        return None                    # nothing in this well could be read
    return wt


def tile_pixel_um(wt: WellTiles) -> float:
    """Physical pixel size (µm) from a tile's OME metadata, 0 if unknown."""
    try:
        path = next(iter(wt.files.values()))
        with Image.open(path) as im:
            desc = im.tag_v2.get(270, "") if hasattr(im, "tag_v2") else ""
        m = re.search(r'PhysicalSizeX="([\d.]+)"', desc or "")
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def discover_wells(experiment_dir: Path) -> dict:
    """{well_id: WellTiles} for every well that has raw tiles."""
    experiment_dir = Path(experiment_dir)
    out = {}
    for d in sorted(experiment_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            wt = scan_well(d)
        except Exception as e:      # one bad well must not block the other 15
            print(f"skipping well {d.name}: {e}")
            continue
        if wt is not None:
            out[d.name] = wt
    return out


# ---------------------------------------------------------------- alignment

def _read_plane(path: Path, page: int = 0) -> np.ndarray:
    """Native-depth pixels of one page.

    Deliberately avoids PIL's convert("L"): it *clips* 16-bit data (anything with a
    non-zero high byte becomes 255) and applies luma weights to colour tiles, both
    of which would silently destroy intensities.
    """
    with Image.open(path) as im:
        if page:
            im.seek(page)
        arr = np.asarray(im)
    if arr.ndim == 3:                      # pseudo-coloured tile → intensity
        arr = arr.max(axis=2)
    if arr.dtype not in (np.uint8, np.uint16):
        arr = arr.astype(np.float32)
    return arr


def _boxblur(a: np.ndarray, k: int) -> np.ndarray:
    if k < 2:
        return a
    pad = np.pad(a.astype(np.float32), k, mode="edge")
    c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    h, w = a.shape
    s = (c[2 * k:2 * k + h, 2 * k:2 * k + w] - c[0:h, 2 * k:2 * k + w]
         - c[2 * k:2 * k + h, 0:w] + c[0:h, 0:w])
    return s / float((2 * k) ** 2)


def _prep_score(img: np.ndarray, ds: int = 4) -> np.ndarray:
    """Small, high-passed copy used to rank candidate shifts.

    The vignetting is the same in every tile, so plain correlation rewards
    whichever candidate overlaps most — it is matching the illumination, not the
    specimen. Removing the low frequencies makes the score reflect real detail.
    """
    small = img[::ds, ::ds].astype(np.float32)
    k = max(2, min(small.shape) // 16)
    return small - _boxblur(small, k)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two equally-shaped patches."""
    a = a.astype(np.float32) - float(a.mean())
    b = b.astype(np.float32) - float(b.mean())
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-9 else 0.0


def _score_shift(a: np.ndarray, b: np.ndarray, dy: int, dx: int) -> tuple:
    """(ncc, overlap_area) for placing b at (dy,dx) relative to a."""
    th, tw = a.shape
    y0, x0 = max(0, dy), max(0, dx)
    y1, x1 = min(th, dy + th), min(tw, dx + tw)
    if y1 - y0 < 16 or x1 - x0 < 16:
        return -1.0, 0
    pa = a[y0:y1, x0:x1]
    pb = b[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return _ncc(pa, pb), (y1 - y0) * (x1 - x0)


def _subpixel(a: np.ndarray, b: np.ndarray, dy: int, dx: int,
              span: float = 1.0, steps: int = 21) -> tuple:
    """Refine an integer shift to sub-pixel by maximising NCC on the overlap.

    A local 1-D search per axis around the integer winner, using bilinear
    resampling. Integer-only placement quantises every step, and the error is
    coherent across the lattice, so recovering the fraction is worth the cost.
    """
    th, tw = a.shape
    y0, x0 = max(0, dy), max(0, dx)
    y1, x1 = min(th, dy + th), min(tw, dx + tw)
    if y1 - y0 < 32 or x1 - x0 < 32:
        return float(dy), float(dx), 0.0
    pa = a[y0:y1, x0:x1].astype(np.float32)

    def score(fy, fx):
        sy, sx = dy + fy, dx + fx
        yy = np.arange(y0, y1, dtype=np.float32) - sy
        xx = np.arange(x0, x1, dtype=np.float32) - sx
        iy = np.clip(np.floor(yy).astype(int), 0, th - 2)
        ix = np.clip(np.floor(xx).astype(int), 0, tw - 2)
        wy = (yy - iy)[:, None]
        wx = (xx - ix)[None, :]
        b00 = b[np.ix_(iy, ix)].astype(np.float32)
        b01 = b[np.ix_(iy, ix + 1)].astype(np.float32)
        b10 = b[np.ix_(iy + 1, ix)].astype(np.float32)
        b11 = b[np.ix_(iy + 1, ix + 1)].astype(np.float32)
        pb = (b00 * (1 - wy) * (1 - wx) + b01 * (1 - wy) * wx
              + b10 * wy * (1 - wx) + b11 * wy * wx)
        return _ncc(pa, pb)

    best = (0.0, 0.0, score(0.0, 0.0))
    for axis in (0, 1):
        grid = np.linspace(-span, span, steps)
        for v in grid:
            fy = v if axis == 0 else best[0]
            fx = v if axis == 1 else best[1]
            s = score(fy, fx)
            if s > best[2]:
                best = (fy, fx, s)
    return dy + best[0], dx + best[1], best[2]


def _phase_corr(a: np.ndarray, b: np.ndarray) -> tuple:
    """(dy, dx, peak) aligning b onto a."""
    a = a.astype(np.float32) - float(a.mean())
    b = b.astype(np.float32) - float(b.mean())
    A = np.fft.rfft2(a)
    B = np.fft.rfft2(b)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-9
    r = np.fft.irfft2(R, s=a.shape)
    dy, dx = np.unravel_index(int(np.argmax(r)), r.shape)
    peak = float(r.max())
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return int(dy), int(dx), peak


#: strip fractions tried when searching for the overlap
_FRACS = (0.12, 0.20, 0.30, 0.45, 0.60)
#: below this NCC a pair is treated as unmeasurable
MIN_PAIR_SCORE = 0.15
#: the winner must beat the best DISTINCT rival by this fraction of its own score,
#: otherwise the pair is ambiguous (periodic specimen) and does not vote
AMBIGUITY_MARGIN = 0.12
#: shifts closer than this are the same candidate
_TOL = 6


def _pair_candidates(a: np.ndarray, b: np.ndarray, axis: str, top: int = 6) -> list:
    """Best (dy, dx, score) placing neighbour b relative to a.

    Phase correlation is run on strips of several widths, because a strip only
    resolves overlaps near its own width: the peak wraps against the *strip*, so
    one fixed fraction silently returns a shift that is off by exactly one strip
    width when the real overlap is outside its band. Every candidate (including
    its ±1 strip aliases) is then scored by normalised cross-correlation over the
    overlap it implies, and the genuinely best one wins.
    """
    th, tw = a.shape
    cands = set()
    for frac in _FRACS:
        if axis == "x":
            ov = max(8, min(tw - 1, int(tw * frac)))
            dy, dxl, _ = _phase_corr(a[:, tw - ov:], b[:, :ov])
            for k in (-1, 0, 1):
                cands.add((dy, tw - ov + dxl + k * ov))
        else:
            ov = max(8, min(th - 1, int(th * frac)))
            dyl, dx, _ = _phase_corr(a[th - ov:, :], b[:ov, :])
            for k in (-1, 0, 1):
                cands.add((th - ov + dyl + k * ov, dx))
    # Rank candidates on small, high-passed copies: cheap, and blind to the
    # illumination pattern that is identical in every tile.
    DS = 2
    sa, sb = _prep_score(a, DS), _prep_score(b, DS)
    sh, sw = sa.shape
    scored = []
    min_area = 0.01 * sh * sw
    for (dy, dx) in cands:
        # a neighbour must be a forward step that still overlaps
        if axis == "x" and not (0 < dx <= tw):
            continue
        if axis == "y" and not (0 < dy <= th):
            continue
        score, area = _score_shift(sa, sb, int(round(dy / DS)), int(round(dx / DS)))
        if area < min_area:
            continue
        scored.append((dy, dx, score))
    scored.sort(key=lambda t: -t[2])
    if not scored:
        return []
    # Uniqueness test. A 1-D periodic specimen (stripes, gratings, a ruled slide)
    # makes every shift one period apart score almost identically, so the winner is
    # meaningless even though its NCC is ~0.99. If the best distinct rival is nearly
    # as good, this pair cannot decide the step — drop it rather than let it vote.
    by, bx, bs = scored[0]
    rival = next((s for (dy, dx, s) in scored[1:]
                  if abs(dy - by) > _TOL or abs(dx - bx) > _TOL), None)
    if rival is not None and bs > 0 and (bs - rival) < AMBIGUITY_MARGIN * bs:
        return []
    return scored[:top]


def _consensus(pair_cands: list, tol: int = 6) -> tuple:
    """Pick the shift best supported by *all* pairs.

    Every neighbour pair in a plate shares the same step, so a candidate that
    several pairs agree on beats one that merely wins a single noisy pair. This
    is what rescues thin overlaps, where one pair alone is easy to fool.
    """
    pairs = [c for c in pair_cands if c]
    if not pairs:
        return None, 0.0
    best = None
    for cands in pairs:
        for (vy, vx, _) in cands:
            total, votes, agree = 0.0, 0, []
            for other in pairs:
                m = max((s for (dy, dx, s) in other
                         if abs(dy - vy) <= tol and abs(dx - vx) <= tol), default=None)
                if m is not None:
                    total += m
                    votes += 1
                    agree.extend([(dy, dx) for (dy, dx, s) in other
                                  if abs(dy - vy) <= tol and abs(dx - vx) <= tol and s == m])
            if best is None or (votes, total) > (best[0], best[1]):
                best = (votes, total, agree)
    if best is None or not best[2]:
        return None, 0.0
    arr = np.array(best[2])
    return tuple(np.median(arr, axis=0).astype(int)), best[1] / max(1, best[0])


#: physically plausible overlap fraction for a tiled acquisition. Anything outside
#: this is a correlation alias, not a real stage move — 1-D periodic specimens
#: (stripes, gratings) otherwise produce a confident, unflagged wrong step.
OVERLAP_RANGE = (0.03, 0.75)


def _plausible(dy: int, dx: int, axis: str, th: int, tw: int) -> bool:
    lo, hi = OVERLAP_RANGE
    if axis == "x":
        return tw * (1 - hi) <= dx <= tw * (1 - lo)
    return th * (1 - hi) <= dy <= th * (1 - lo)


def estimate_steps(images: dict, prior: dict = None) -> tuple:
    """Measure (dy,dx) per +1 step in X and in Y, with confidences.

    Correlation runs on overlapping strips only: the vignetting is identical in
    every tile, so correlating whole tiles pins the peak at zero shift.

    `prior` (optional) is a {'step_x','step_y'} geometry measured elsewhere — in
    practice the plate consensus. It is used ONLY when this well cannot measure a
    direction itself; every well of a plate shares one stage program, so that is a
    far better fallback than abutting the tiles at zero overlap (which duplicates
    a strip of specimen at every seam).
    """
    th, tw = next(iter(images.values())).shape
    hx, hy = [], []
    for (x, y), a in images.items():
        b = images.get((x + 1, y))
        if b is not None:
            hx.append([c for c in _pair_candidates(a, b, "x")
                       if _plausible(c[0], c[1], "x", th, tw)])
        b = images.get((x, y + 1))
        if b is not None:
            hy.append([c for c in _pair_candidates(a, b, "y")
                       if _plausible(c[0], c[1], "y", th, tw)])

    step_x, px = _consensus(hx)
    step_y, py = _consensus(hy)
    src_x = src_y = "measured"
    if step_x is None or px < MIN_PAIR_SCORE:
        if prior and prior.get("step_x") is not None:
            step_x, src_x = tuple(prior["step_x"]), "plate"
        else:
            step_x, src_x = (0, tw), "abut"       # last resort, always reported
        px = px or 0.0
    if step_y is None or py < MIN_PAIR_SCORE:
        if prior and prior.get("step_y") is not None:
            step_y, src_y = tuple(prior["step_y"]), "plate"
        else:
            step_y, src_y = (th, 0), "abut"
        py = py or 0.0
    return step_x, step_y, px, py, {"x": src_x, "y": src_y}


def plate_geometry(per_well: list) -> dict:
    """Consensus geometry for a plate from the wells that measured confidently.

    Measured across 8 wells of one real plate: step_x dx sd = 0.5 px,
    step_y dy sd = 0.8 px — i.e. one stage program, so a well that cannot see
    enough specimen can safely inherit it.
    """
    xs = [g["step_x"] for g in per_well if g.get("peak_x", 0) >= MIN_PAIR_SCORE]
    ys = [g["step_y"] for g in per_well if g.get("peak_y", 0) >= MIN_PAIR_SCORE]
    out = {"step_x": None, "step_y": None, "n_x": len(xs), "n_y": len(ys)}
    if xs:
        out["step_x"] = tuple(np.median(np.array(xs), axis=0).astype(int))
        out["spread_x"] = float(np.std(np.array(xs), axis=0).max())
    if ys:
        out["step_y"] = tuple(np.median(np.array(ys), axis=0).astype(int))
        out["spread_y"] = float(np.std(np.array(ys), axis=0).max())
    return out


def tile_origins(positions, step_x, step_y) -> dict:
    """Top-left pixel of every tile, normalised so the mosaic starts at (0,0)."""
    x0 = min(p[0] for p in positions)
    y0 = min(p[1] for p in positions)
    raw = {}
    for (x, y) in positions:
        i, j = x - x0, y - y0
        raw[(x, y)] = (i * step_x[0] + j * step_y[0],
                       i * step_x[1] + j * step_y[1])
    miny = min(v[0] for v in raw.values())
    minx = min(v[1] for v in raw.values())
    return {k: (v[0] - miny, v[1] - minx) for k, v in raw.items()}


def mosaic_size(origins, tile_shape) -> tuple:
    th, tw = tile_shape
    h = max(v[0] for v in origins.values()) + th
    w = max(v[1] for v in origins.values()) + tw
    return h, w


def _box(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over a (2k+1)-ish square, via summed-area table."""
    if k < 1:
        return a
    pad = np.pad(a.astype(np.float32), k, mode="edge")
    c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    h, w = a.shape
    s = (c[2 * k:2 * k + h, 2 * k:2 * k + w] - c[0:h, 2 * k:2 * k + w]
         - c[2 * k:2 * k + h, 0:w] + c[0:h, 0:w])
    return s / float((2 * k) ** 2)


def estimate_flatfield(tiles: list, smooth_frac: float = 0.12,
                       lo_pct: float = 25.0) -> np.ndarray:
    """Multiplicative shading field (mean 1) estimated from the tiles themselves.

    With only 21-60 fields — and a specimen that can fill most of every field —
    a per-pixel *median* still contains specimen. BaSiC's low-rank/sparse
    decomposition is the field standard but wants a larger, more diverse stack.
    What survives at this tile count is a low-percentile ("darkest common
    background") estimate: at each pixel take a low percentile across tiles, which
    tracks the illumination floor rather than the objects, then smooth heavily,
    since real shading is smooth by construction.

    Returns a field to DIVIDE by; it is clipped so a bad estimate cannot blow up
    the corrected image.
    """
    stack = np.stack([t.astype(np.float32) for t in tiles])
    prof = np.percentile(stack, lo_pct, axis=0)
    k = max(3, int(min(prof.shape) * smooth_frac))
    prof = _box(prof, k)
    m = float(prof.mean())
    if not np.isfinite(m) or m <= 1e-6:
        return np.ones_like(prof)
    prof /= m
    return np.clip(prof, 0.2, 5.0).astype(np.float32)


def apply_flatfield(img: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Divide out the shading, staying in float32.

    Deliberately does NOT clip or re-quantise: a value pushed above the dtype
    maximum must keep its true magnitude until the very end, or a max-projection
    over Z would pick the clipped value and a mean-projection would be biased.
    Quantisation happens once, in `blend`.
    """
    f = np.asarray(field, dtype=np.float32)
    # A non-finite or non-positive field would turn pixels into inf/NaN/negatives
    # that then poison the whole mosaic; refuse it rather than propagate.
    if f.shape != img.shape or not np.all(np.isfinite(f)) or f.min() <= 0:
        f = np.where(np.isfinite(f) & (f > 0), f, 1.0).astype(np.float32) \
            if f.shape == img.shape else np.ones_like(img, dtype=np.float32)
    return img.astype(np.float32) / f


def _feather(shape) -> np.ndarray:
    th, tw = shape
    fy = np.minimum(np.arange(th), th - 1 - np.arange(th)).astype(np.float32) + 1.0
    fx = np.minimum(np.arange(tw), tw - 1 - np.arange(tw)).astype(np.float32) + 1.0
    w = np.minimum.outer(fy, fx)
    return w / w.max()


def blend(images: dict, origins: dict, tile_shape, out_shape,
          out_dtype=None) -> np.ndarray:
    """Feather-blended mosaic; the single point where the result is quantised.

    Tiles may arrive as float32 (flat-fielded) — accumulate in float and round
    once at the end, so nothing is clipped before Z-projection or fusion.
    """
    th, tw = tile_shape
    H, W = out_shape
    acc = np.zeros((H, W), np.float32)
    wgt = np.zeros((H, W), np.float32)
    feather = _feather(tile_shape)
    dtype = np.dtype(out_dtype) if out_dtype is not None \
        else next(iter(images.values())).dtype
    for key, img in images.items():
        oy, ox = origins[key]
        h = min(th, H - oy, img.shape[0])
        w = min(tw, W - ox, img.shape[1])
        if h <= 0 or w <= 0:
            continue
        patch = img[:h, :w].astype(np.float32)
        fw = feather[:h, :w]
        good = np.isfinite(patch)          # a bad pixel must not poison the seam
        acc[oy:oy + h, ox:ox + w] += np.where(good, patch, 0.0) * fw
        wgt[oy:oy + h, ox:ox + w] += np.where(good, fw, 0.0)
    out = np.divide(acc, wgt, out=np.zeros_like(acc), where=wgt > 0)
    if dtype == np.uint16:
        return np.clip(np.rint(out), 0, 65535).astype(np.uint16)
    if dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out.astype(np.float32)


# ---------------------------------------------------------------- stitching

def _z_reduce(stack: list, mode: str) -> np.ndarray:
    if len(stack) == 1:
        return stack[0]
    arr = np.stack(stack)
    if mode == "max":
        return arr.max(axis=0)
    if mode == "mean":
        # Stay in float: rounding here and again after flat-fielding loses a DN
        # (uint16 1000,1001 -> 1000 -> 5000 instead of 5002).
        return arr.mean(axis=0, dtype=np.float32)
    return arr[len(arr) // 2]          # "middle"


def reference_plane(wt: WellTiles) -> Plane:
    """Plane used to measure the geometry — brightfield first, it has most texture."""
    for p in wt.planes:
        if p.channel.upper() == "CH4":
            return p
    return wt.planes[0]


def stitch_well(wt: WellTiles, planes=None, z_mode: str = "max",
                progress=None, cancel=None, flatfield: bool = True,
                prior: dict = None, subpixel: bool = True) -> dict:
    """Stitch a well. Returns {plane_key: (Plane, mosaic)} plus "__geometry__".

    The geometry is measured once on the reference plane and reused, so all
    channels land on exactly the same grid.

    flatfield : divide out the illumination profile, estimated per channel from
        the tiles themselves, BEFORE projection and blending. Without it a uniform
        specimen reads +-40 DN depending only on where it fell in the field.
    prior : plate-level geometry to fall back on (see estimate_steps).
    subpixel : refine the per-edge shifts to sub-pixel and report the residuals.
    """
    planes = planes or wt.planes
    ref = reference_plane(wt)

    bad = []

    def load(plane, pos):
        x, y = pos
        stack = []
        for z in wt.z_values:
            f = wt.files.get((x, y, z, plane.channel))
            if f is None:
                continue
            try:
                stack.append(_read_plane(f, plane.page))
            except Exception as e:     # skip the slice, keep the well
                bad.append((f.name, str(e)))
        if not stack:
            return None
        shapes = {s.shape for s in stack}
        if len(shapes) > 1:            # ragged Z stack — keep the common size
            h = min(s.shape[0] for s in stack)
            w = min(s.shape[1] for s in stack)
            stack = [s[:h, :w] for s in stack]
        return _z_reduce(stack, z_mode)

    if progress:
        progress(f"{wt.well}: measuring tile overlap…", 0.0)
    ref_imgs = {}
    for i, pos in enumerate(wt.positions):
        if cancel and cancel():
            return {}
        img = load(ref, pos)
        if img is not None:
            ref_imgs[pos] = img
        if progress:
            progress(None, 0.15 * (i + 1) / max(1, len(wt.positions)))
    if not ref_imgs:
        return {}

    # Shading is corrected on the reference plane too, so alignment sees the same
    # pixels that will be written.
    ref_dtype = next(iter(ref_imgs.values())).dtype   # only the reference plane's
    plane_dtypes = {}
    fields = {}
    if flatfield and len(ref_imgs) >= 4:
        fields[ref.key] = estimate_flatfield(list(ref_imgs.values()))
        ref_imgs = {k: apply_flatfield(v, fields[ref.key]) for k, v in ref_imgs.items()}

    step_x, step_y, px, py, src = estimate_steps(ref_imgs, prior=prior)
    origins = tile_origins(wt.positions, step_x, step_y)
    out_shape = mosaic_size(origins, wt.tile_shape)
    edges = _edge_report(ref_imgs, step_x, step_y, subpixel=subpixel)

    result = {}
    for pi, plane in enumerate(planes):
        if cancel and cancel():
            return result
        if progress:
            progress(f"{wt.well}: stitching {plane.label}…", None)
        if plane.key == ref.key:
            imgs = ref_imgs                       # already flat-fielded
        else:
            imgs = {pos: img for pos in wt.positions
                    if (img := load(plane, pos)) is not None}
            if not imgs:
                continue
            plane_dtypes[plane.key] = next(iter(imgs.values())).dtype
            if flatfield and len(imgs) >= 4:
                # estimate per channel: each has its own illumination path
                fields[plane.key] = estimate_flatfield(list(imgs.values()))
                imgs = {k: apply_flatfield(v, fields[plane.key]) for k, v in imgs.items()}
        if not imgs:
            continue
        # Each plane keeps ITS OWN depth: a uint8 brightfield reference must not
        # force a uint16 fluorescence plane down to 8 bits.
        pdtype = ref_dtype if plane.key == ref.key else plane_dtypes.get(plane.key, ref_dtype)
        result[plane.key] = (plane, blend(imgs, origins, wt.tile_shape, out_shape,
                                          out_dtype=pdtype))
        if progress:
            progress(None, 0.15 + 0.85 * (pi + 1) / max(1, len(planes)))
    result["__geometry__"] = {
        "well": wt.well,
        "step_x": step_x, "step_y": step_y, "peak_x": px, "peak_y": py,
        "step_source": src,
        "shape": out_shape, "tiles": len(ref_imgs),
        "overlap_x": wt.tile_shape[1] - abs(step_x[1]),
        "overlap_y": wt.tile_shape[0] - abs(step_y[0]),
        "unreadable": bad,
        "flatfield": bool(fields),
        # Independent cross-check: does this well's own geometry match the rest of
        # the plate? A detector-fixed artifact makes every edge agree (residual ~0)
        # while placing the tiles wrongly — only a comparison against an outside
        # reference can catch that.
        "step_deviation": (None if not prior or prior.get("step_x") is None else
                           round(float(max(
                               abs(step_x[0] - prior["step_x"][0]),
                               abs(step_x[1] - prior["step_x"][1]),
                               abs(step_y[0] - prior["step_y"][0]),
                               abs(step_y[1] - prior["step_y"][1]))), 1)),
        # Shading and specimen are only separable when tiles show mostly new
        # content; below ~40% turnover the estimate absorbs real structure.
        "shading_identifiable": (abs(step_x[1]) / max(1, wt.tile_shape[1])) >= 0.4,
        "low_confidence": (px < MIN_PAIR_SCORE or py < MIN_PAIR_SCORE
                           or src["x"] != "measured" or src["y"] != "measured"),
        **edges,
    }
    return result


def _edge_report(images: dict, step_x, step_y, subpixel: bool = True) -> dict:
    """Per-edge QC: how far each measured neighbour shift sits from the lattice.

    IMPORTANT — this measures INTERNAL CONSISTENCY, not accuracy. It is computed
    with the same estimator that produced the lattice, so if every edge locks onto
    the same wrong feature (a detector-fixed artifact, a repeated pattern) the
    residual is small while the placement is wrong. A small residual is necessary
    but not sufficient; `step_deviation` below is the independent cross-check.
    Field convention for the honest case: sub-pixel median, p95 <= ~1 px.
    """
    res, scores, n_amb = [], [], 0
    for (x, y), a in images.items():
        for nb, axis, step in [((x + 1, y), "x", step_x), ((x, y + 1), "y", step_y)]:
            b = images.get(nb)
            if b is None:
                continue
            c = _pair_candidates(a, b, axis)
            if not c:
                n_amb += 1
                continue
            dy, dx, sc = c[0]
            if subpixel:
                fy, fx, sc2 = _subpixel(a, b, dy, dx)
                if sc2 > 0:
                    dy, dx, sc = fy, fx, sc2
            res.append(float(np.hypot(dy - step[0], dx - step[1])))
            scores.append(float(sc))
    if not res:
        return {"edges": 0, "residual_median": None, "residual_p95": None,
                "edge_ncc_median": None, "ambiguous_edges": n_amb}
    r = np.array(res)
    return {
        "edges": len(res),
        "residual_median": round(float(np.median(r)), 2),
        "residual_p95": round(float(np.percentile(r, 95)), 2),
        "residual_max": round(float(r.max()), 2),
        "edge_ncc_median": round(float(np.median(scores)), 3),
        "edge_ncc_min": round(float(np.min(scores)), 3),
        "ambiguous_edges": n_amb,
    }


# ---------------------------------------------------------------- output

def save_ome_tiff(path: Path, planes_data: list, pixel_um: float = 0.0):
    """Write a multi-channel OME-TIFF (channel names preserved)."""
    import tifffile
    arr = np.stack([img for _, img in planes_data])          # (C, Y, X)
    meta = {"axes": "CYX",
            "Channel": {"Name": [p.label for p, _ in planes_data]}}
    kwargs = {}
    if pixel_um:
        meta["PhysicalSizeX"] = pixel_um
        meta["PhysicalSizeY"] = pixel_um
        meta["PhysicalSizeXUnit"] = "µm"
        meta["PhysicalSizeYUnit"] = "µm"
        kwargs["resolution"] = (1e4 / pixel_um, 1e4 / pixel_um)
    # Write to a temporary file and move it into place, so a failed or cancelled
    # write can never truncate a previously good result.
    tmp = Path(str(path) + ".part")
    last = None
    for comp in ("zlib", "lzw", None):   # zlib needs a recent imagecodecs
        try:
            tifffile.imwrite(str(tmp), arr, photometric="minisblack",
                             compression=comp, metadata=meta, **kwargs)
            tmp.replace(path)
            return
        except Exception as e:           # missing codec → try the next one
            last = e
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not write {path}: {last}")
