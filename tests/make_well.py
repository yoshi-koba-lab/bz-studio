"""Synthetic BZ-X-like tile sets with GROUND TRUTH, for validating the stitcher.

Parameters match what was measured on the real instrument:
  tile 1832x1374, step_x=(-3,1294), step_y=(970,2)  (~29% overlap),
  a 5x5 grid with the corners missing (21 tiles), strong vignetting
  (a uniform specimen reads 47-152 for a true value of 120).
Because the scene, the lattice and the shading are known exactly, geometry and
intensity can both be scored against truth rather than against vendor output.
"""
from __future__ import annotations

import numpy as np

TILE_H, TILE_W = 1374, 1832
STEP_X = (-3, 1294)
STEP_Y = (970, 2)
# corners dropped, like the real plate scans
GRID = [(x, y) for x in range(5) for y in range(5)
        if not (x in (0, 4) and y in (0, 4))]


def vignette(shape=(TILE_H, TILE_W), strength=0.55, cx=0.35, cy=0.5):
    """Multiplicative shading field, normalised to mean 1."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    v = (1.0 - strength) + strength * np.exp(
        -(((xx - w * cx) / (w * 0.55)) ** 2 + ((yy - h * cy) / (h * 0.8)) ** 2))
    return (v / v.mean()).astype(np.float32)


def specimen(h, w, seed=0, kind="organoid"):
    """A scene with structure at several scales (like cells + debris)."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 60.0, np.float32)
    if kind == "flat":                       # uniform target for intensity tests
        return np.full((h, w), 120.0, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    # a big round object
    r = np.hypot((xx - w * 0.55) / (w * 0.3), (yy - h * 0.5) / (h * 0.42))
    base += 70.0 * np.clip(1.2 - r, 0, 1)
    # punctate detail
    for _ in range(4000):
        cy_, cx_ = rng.integers(0, h), rng.integers(0, w)
        rad = int(rng.integers(3, 14))
        y0, y1 = max(0, cy_ - rad), min(h, cy_ + rad)
        x0, x1 = max(0, cx_ - rad), min(w, cx_ + rad)
        base[y0:y1, x0:x1] += rng.normal(25, 10)
    base += rng.normal(0, 3, (h, w))
    return np.clip(base, 0, 255).astype(np.float32)


def build_well(grid=None, step_x=STEP_X, step_y=STEP_Y, jitter=0, shading=True,
               seed=0, kind="organoid", noise=1.5, tile_shape=(TILE_H, TILE_W)):
    """Return (tiles, truth_scene, origins, shade).

    tiles   : {(x,y): uint8 tile}
    truth   : the un-vignetted scene the tiles were cut from
    origins : {(x,y): (row, col)} ground-truth placement inside `truth`
    shade   : the multiplicative field applied to every tile
    """
    grid = grid or GRID
    th, tw = tile_shape
    rng = np.random.default_rng(seed)
    raw = {}
    for (x, y) in grid:
        raw[(x, y)] = (x * step_x[0] + y * step_y[0],
                       x * step_x[1] + y * step_y[1])
    miny = min(v[0] for v in raw.values())
    minx = min(v[1] for v in raw.values())
    origins = {k: (v[0] - miny, v[1] - minx) for k, v in raw.items()}
    H = max(v[0] for v in origins.values()) + th
    W = max(v[1] for v in origins.values()) + tw
    truth = specimen(H, W, seed=seed, kind=kind)
    shade = vignette((th, tw)) if shading else np.ones((th, tw), np.float32)

    tiles = {}
    actual = {}          # where each tile was really cut from (origins + jitter)
    for k, (oy, ox) in origins.items():
        jy = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
        jx = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
        sy = int(np.clip(oy + jy, 0, H - th))
        sx = int(np.clip(ox + jx, 0, W - tw))
        actual[k] = (sy, sx)
        patch = truth[sy:sy + th, sx:sx + tw] * shade
        if noise:
            patch = patch + rng.normal(0, noise, patch.shape)
        tiles[k] = np.clip(patch, 0, 255).astype(np.uint8)
    return tiles, truth, actual, shade


def sparse_well(**kw):
    """A well whose specimen covers only a small part — alignment should fail."""
    kw.setdefault("kind", "sparse")
    tiles, truth, origins, shade = build_well(**kw)
    # blank out everything except a small central patch, like an empty well
    H, W = truth.shape
    mask = np.zeros((H, W), bool)
    mask[int(H * 0.45):int(H * 0.55), int(W * 0.45):int(W * 0.55)] = True
    flat = np.where(mask, truth, 35.0).astype(np.float32)
    out = {}
    for k, (oy, ox) in origins.items():
        out[k] = np.clip(flat[oy:oy + TILE_H, ox:ox + TILE_W] * shade, 0, 255).astype(np.uint8)
    return out, flat, origins, shade
