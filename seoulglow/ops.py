"""Small numeric helpers shared by the effects (float32 RGB images in [0, 1])."""

from __future__ import annotations

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    return img @ LUMA


def screen(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - base) * (1.0 - layer)


def lerp(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    if isinstance(t, np.ndarray) and t.ndim == a.ndim - 1:
        t = t[..., None]
    return a + (b - a) * t


def to_lab(img: np.ndarray) -> np.ndarray:
    """float RGB [0,1] → Lab with L in [0,100] and a/b roughly in [-127,127]."""
    return cv2.cvtColor(np.clip(img, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2Lab)


def from_lab(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB), 0.0, 1.0)


def rgb255_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    px = np.array([[rgb]], np.float32) / 255.0
    return cv2.cvtColor(px, cv2.COLOR_RGB2Lab)[0, 0]


def box(img: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_smooth(img: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Self-guided filter (He et al.) per channel: edge-preserving, O(N) in radius."""
    mean = box(img, radius)
    var = np.maximum(box(img * img, radius) - mean * mean, 0.0)
    a = var / (var + eps)
    b = mean * (1.0 - a)
    return box(a, radius) * img + box(b, radius)


def gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; large sigmas run on a downsampled copy (same look, ~10× faster)."""
    if sigma < 0.3:
        return img
    if sigma <= 4.0:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    h, w = img.shape[:2]
    factor = int(sigma // 2)
    sw, sh = max(1, w // factor), max(1, h // factor)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / factor, sigmaY=sigma / factor)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
