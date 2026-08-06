"""Pure rendering helpers for the KTF viewer — no Qt, so they're unit-testable."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelView:
    """Display settings for one channel."""
    ch_id: str
    color: tuple  # (r, g, b) 0-255
    lo: int = 0
    hi: int = 255
    gamma: float = 1.0
    enabled: bool = True
    solo: bool = False


def apply_levels(gray: np.ndarray, lo: int, hi: int, gamma: float) -> np.ndarray:
    """Window/level + gamma. Returns float array in [0,1]."""
    hi = max(hi, lo + 1)
    scaled = np.clip((gray.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-3:
        scaled = np.power(scaled, 1.0 / gamma)
    return scaled


def composite(channels: list, images: dict) -> np.ndarray:
    """Blend channels into an RGB uint8 image.

    Args:
        channels: list of ChannelView
        images: dict ch_id -> 2D uint8 array (all same shape)
    Rules:
        - If any channel is soloed, only soloed channels are shown.
        - A soloed channel renders in grayscale (white) for a clean single-channel look.
    """
    visible = [c for c in channels if c.enabled and c.ch_id in images]
    soloed = [c for c in visible if c.solo]
    if soloed:
        visible = soloed

    if not visible:
        # find any image for shape
        any_img = next(iter(images.values()), None)
        if any_img is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return np.zeros((*any_img.shape[:2], 3), dtype=np.uint8)

    h, w = images[visible[0].ch_id].shape[:2]
    acc = np.zeros((h, w, 3), dtype=np.float32)

    single_solo = len(visible) == 1 and visible[0].solo
    for c in visible:
        scaled = apply_levels(images[c.ch_id], c.lo, c.hi, c.gamma)
        color = (255, 255, 255) if single_solo else c.color
        acc[:, :, 0] += scaled * (color[0] / 255.0)
        acc[:, :, 1] += scaled * (color[1] / 255.0)
        acc[:, :, 2] += scaled * (color[2] / 255.0)

    return np.clip(acc * 255.0, 0, 255).astype(np.uint8)


def nice_scale_bar(um_per_px: float, max_bar_px: float) -> tuple:
    """Choose a 'nice' scale bar length (1/2/5 x 10^n µm) that fits within max_bar_px.

    Returns (length_um, length_px, label). label uses µm or mm as appropriate.
    """
    if um_per_px <= 0 or max_bar_px <= 0:
        return 0.0, 0.0, ""
    max_um = max_bar_px * um_per_px
    # Largest 1/2/5 x 10^n <= max_um
    exp = math.floor(math.log10(max_um))
    best = None
    for mant in (1, 2, 5):
        for e in (exp - 1, exp, exp + 1):
            cand = mant * (10 ** e)
            if cand <= max_um:
                if best is None or cand > best:
                    best = cand
    if best is None:
        best = max_um
    length_px = best / um_per_px
    if best >= 1000:
        label = f"{best / 1000:g} mm"
    else:
        label = f"{best:g} µm"
    return best, length_px, label
