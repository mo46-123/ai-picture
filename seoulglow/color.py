"""Korean-style color grading (韩系调色): a small, parametric "Lightroom" in NumPy.

A Korean look is mostly about *air*: bright exposure with protected highlights,
low contrast, slightly lifted blacks, gentle desaturation, pink/peach skin,
minty greens, and sometimes a dreamy glow or film grain.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from . import ops
from .masks import smoothstep


@dataclass(frozen=True)
class Grade:
    exposure: float = 0.0        # soft exposure (EV-like), highlights never clip
    contrast: float = 0.0        # [-0.5, 0.5] S-curve around mid-gray
    highlights: float = 0.0      # [-1, 1]
    shadows: float = 0.0         # [-1, 1]
    temperature: float = 0.0     # [-1, 1]  (−) cool / (+) warm
    tint: float = 0.0            # [-1, 1]  (−) green / (+) pink-magenta
    saturation: float = 0.0      # [-1, 1]
    vibrance: float = 0.0        # [-1, 1]  boosts muted colors more than vivid ones
    fade: float = 0.0            # [0, 0.3] lifted blacks / milky whites
    shadow_tint: tuple[float, float] = (0.0, 0.0)     # (hue°, strength)
    highlight_tint: tuple[float, float] = (0.0, 0.0)  # (hue°, strength)
    green_shift: float = 0.0     # degrees to rotate greens toward mint/cyan
    green_sat: float = 0.0       # [-1, 1] saturation change for greens
    glow: float = 0.0            # [0, 1] bloom + soft focus (韩剧柔光)
    grain: float = 0.0           # [0, 1] film grain


def _tone_contrast(x: np.ndarray, c: float) -> np.ndarray:
    c = float(np.clip(c, -0.5, 0.5))
    return x + c * 4.0 * (x - 0.5) * x * (1.0 - x)


def _hue_chroma(hue: float) -> np.ndarray:
    """A zero-luminance color offset for the given hue (used for split toning)."""
    rgb = np.array(colorsys.hsv_to_rgb((hue % 360.0) / 360.0, 1.0, 1.0), np.float32)
    chroma = rgb - float(rgb @ ops.LUMA)
    return chroma / max(float(np.abs(chroma).max()), 1e-6)


def _greens(img: np.ndarray, shift: float, sat: float) -> np.ndarray:
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)  # H in [0, 360)
    h, s = hsv[..., 0], hsv[..., 1]
    band = smoothstep(55.0, 85.0, h) * (1.0 - smoothstep(150.0, 180.0, h)) * smoothstep(0.08, 0.25, s)
    hsv[..., 0] = (h + shift * band) % 360.0
    hsv[..., 1] = np.clip(s * (1.0 + sat * band), 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _glow(img: np.ndarray, amount: float) -> np.ndarray:
    h, w = img.shape[:2]
    small_scale = 0.25
    small = cv2.resize(img, (max(1, int(w * small_scale)), max(1, int(h * small_scale))),
                       interpolation=cv2.INTER_AREA)
    lum = ops.luminance(small)
    bright = small * smoothstep(0.5, 1.0, lum)[..., None]
    sigma = 0.012 * max(h, w) * small_scale
    bloom = cv2.resize(ops.gaussian(bright, sigma), (w, h), interpolation=cv2.INTER_LINEAR)
    out = ops.screen(img, bloom * (0.55 * amount))
    half = cv2.resize(out, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    soft = cv2.resize(ops.gaussian(half, 0.00125 * max(h, w)), (w, h), interpolation=cv2.INTER_LINEAR)
    return ops.lerp(out, np.maximum(out, soft), 0.35 * amount)  # "lighten" halation


def _grain(img: np.ndarray, amount: float, seed: int) -> np.ndarray:
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w)).astype(np.float32)
    noise = ops.gaussian(noise, 0.6) * 1.6
    lum = ops.luminance(img)
    visibility = 1.0 - 0.6 * np.abs(2.0 * lum - 1.0)  # most visible in midtones
    return img + (noise * visibility * 0.035 * amount)[..., None]


def pixel_ops(img: np.ndarray, g: Grade) -> np.ndarray:
    """Every per-pixel part of the grade (no neighborhood ops), so it can be baked into a LUT."""
    x = np.clip(img.astype(np.float32), 0.0, 1.0)

    if g.exposure:
        x = 1.0 - (1.0 - x) ** (2.0 ** g.exposure)

    if g.temperature or g.tint:
        t, ti = g.temperature, g.tint
        gains = np.array([1 + 0.10 * t + 0.03 * ti, 1 - 0.06 * ti, 1 - 0.10 * t + 0.03 * ti], np.float32)
        gains /= float(gains @ ops.LUMA)
        x = x * gains

    if g.contrast:
        x = _tone_contrast(np.clip(x, 0.0, 1.0), g.contrast)

    if g.highlights or g.shadows:
        lum = np.clip(ops.luminance(x), 0.0, 1.0)
        delta = g.shadows * 0.28 * (1.0 - lum) ** 3 + g.highlights * 0.22 * lum ** 3
        x = x + delta[..., None]

    if g.saturation or g.vibrance:
        lum = ops.luminance(x)[..., None]
        chroma = x - lum
        r, gch, b = x[..., 0], x[..., 1], x[..., 2]
        spread = np.maximum(np.maximum(r, gch), b) - np.minimum(np.minimum(r, gch), b)
        spread = np.clip(spread * 2.0, 0.0, 1.0)[..., None]
        factor = 1.0 + g.saturation + g.vibrance * (1.0 - spread)
        x = lum + chroma * np.maximum(factor, 0.0)

    if g.green_shift or g.green_sat:
        x = _greens(np.clip(x, 0.0, 1.0), g.green_shift, g.green_sat)

    for (hue, strength), weight_fn in ((g.shadow_tint, lambda l: (1.0 - l) ** 2),
                                       (g.highlight_tint, lambda l: l ** 2)):
        if strength:
            lum = np.clip(ops.luminance(x), 0.0, 1.0)
            x = x + (weight_fn(lum) * 0.12 * strength)[..., None] * _hue_chroma(hue)

    if g.fade:
        f = float(np.clip(g.fade, 0.0, 0.3))
        x = f + np.clip(x, 0.0, 1.0) * (1.0 - f - 0.35 * f)

    return np.clip(x, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 3D LUT: compile the per-pixel ops once per grade, then apply with two remaps
# --------------------------------------------------------------------------- #
LUT_SIZE = 33


@lru_cache(maxsize=64)
def compile_lut(g: Grade, size: int = LUT_SIZE) -> np.ndarray:
    """Return the LUT as a 2D image of shape (size, size*size, 3).

    Row = green index, column = blue_index*size + red_index, so one
    ``cv2.remap`` performs bilinear interpolation in (red, green) for a fixed
    blue slice; two slices are blended for full trilinear interpolation.
    """
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, gr, r = np.meshgrid(axis, axis, axis, indexing="ij")  # [b, g, r]
    grid = np.stack([r, gr, b], axis=-1).reshape(1, -1, 3)
    out = pixel_ops(grid, g).reshape(size, size, size, 3)   # [b, g, r, c]
    return np.ascontiguousarray(out.transpose(1, 0, 2, 3).reshape(size, size * size, 3))


def apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    size = lut.shape[0]
    x = np.clip(img, 0.0, 1.0).astype(np.float32) * (size - 1)
    r, gch, b = x[..., 0], x[..., 1], x[..., 2]
    b0 = np.floor(b)
    fb = (b - b0)[..., None]
    b0 = np.minimum(b0, size - 1)
    b1 = np.minimum(b0 + 1, size - 1)
    s0 = cv2.remap(lut, b0 * size + r, gch, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    s1 = cv2.remap(lut, b1 * size + r, gch, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return s0 + (s1 - s0) * fb


def to_cube(g: Grade, title: str = "SeoulGlow", size: int = LUT_SIZE) -> str:
    """Export the per-pixel part of a grade as an Adobe/Resolve ``.cube`` 3D LUT."""
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, gr, r = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([r, gr, b], axis=-1).reshape(1, -1, 3)   # red changes fastest
    out = pixel_ops(grid, g).reshape(-1, 3)
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    lines += [f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in out]
    return "\n".join(lines) + "\n"


def apply_grade(img: np.ndarray, g: Grade, amount: float = 1.0, seed: int = 7) -> np.ndarray:
    """Apply a :class:`Grade` (LUT + glow + grain) and blend it with the input by ``amount``."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img
    x = apply_lut(img, compile_lut(g))
    if g.glow:
        x = _glow(x, g.glow)
    if g.grain:
        x = _grain(x, g.grain, seed)
    x = np.clip(x, 0.0, 1.0)
    return x if amount >= 1.0 else ops.lerp(img, x, amount)
