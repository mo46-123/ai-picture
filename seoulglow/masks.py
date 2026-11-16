"""Soft masks for facial regions (skin, lips, eyes...) computed from landmarks.

Everything is float32 in [0, 1] and has the same height/width as the crop it
belongs to. Masks are built lazily, so an effect only pays for what it uses.
"""

from __future__ import annotations

from functools import cached_property

import cv2
import numpy as np

from . import face as F
from . import ops

_SHIFT = 2  # sub-pixel precision for cv2 drawing (coordinates are multiplied by 2**_SHIFT)


def fill_poly(shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    poly = np.round(np.asarray(pts) * (1 << _SHIFT)).astype(np.int32)
    cv2.fillPoly(mask, [poly], 255, lineType=cv2.LINE_AA, shift=_SHIFT)
    return mask.astype(np.float32) / 255.0


def polyline(shape: tuple[int, int], pts: np.ndarray, thickness: float, closed: bool = False) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    poly = np.round(np.asarray(pts) * (1 << _SHIFT)).astype(np.int32)
    cv2.polylines(mask, [poly], closed, 255, thickness=max(1, int(round(thickness))),
                  lineType=cv2.LINE_AA, shift=_SHIFT)
    return mask.astype(np.float32) / 255.0


def blur(mask: np.ndarray, sigma: float) -> np.ndarray:
    return ops.gaussian(mask, sigma)


def dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    r = int(round(radius))
    if r < 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask, kernel)


def erode(mask: np.ndarray, radius: float) -> np.ndarray:
    r = int(round(radius))
    if r < 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(mask, kernel)


def close_holes(mask: np.ndarray, radius: float) -> np.ndarray:
    """Morphological closing done at low resolution (fast even for big radii).

    Fills holes smaller than ~2·radius (blemishes, moles, pores) while large
    non-skin areas (hair, glasses frames) stay excluded.
    """
    if radius < 1.0:
        return mask
    h, w = mask.shape
    factor = max(1.0, radius / 4.0)
    sw, sh = max(1, int(round(w / factor))), max(1, int(round(h / factor)))
    small = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
    r = max(1, int(round(radius / factor)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE)
    closed = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.maximum(mask, closed)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gaussian_ellipse(shape: tuple[int, int], center: np.ndarray, sx: float, sy: float,
                     angle_deg: float = 0.0, out: np.ndarray | None = None, weight: float = 1.0) -> np.ndarray:
    """Add an anisotropic Gaussian blob (sigma sx along the rotated x-axis, sy along y)."""
    h, w = shape
    if out is None:
        out = np.zeros(shape, np.float32)
    sx, sy = max(float(sx), 0.5), max(float(sy), 0.5)
    reach = 3.0 * max(sx, sy)
    cx, cy = float(center[0]), float(center[1])
    x0, x1 = int(max(0, np.floor(cx - reach))), int(min(w, np.ceil(cx + reach) + 1))
    y0, y1 = int(max(0, np.floor(cy - reach))), int(min(h, np.ceil(cy + reach) + 1))
    if x0 >= x1 or y0 >= y1:
        return out
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    a = np.radians(angle_deg)
    u = dx * np.cos(a) + dy * np.sin(a)
    v = -dx * np.sin(a) + dy * np.cos(a)
    blob = np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2)).astype(np.float32)
    out[y0:y1, x0:x1] += weight * blob
    return out


class FaceRegions:
    """Lazily computed soft masks for a single face inside an image crop."""

    def __init__(self, img: np.ndarray, face: F.Face) -> None:
        self.img = img
        self.face = face
        self.shape = img.shape[:2]
        self.iod = face.iod

    # -- hard-ish geometric regions ----------------------------------------- #
    @cached_property
    def face_poly(self) -> np.ndarray:
        return fill_poly(self.shape, self.face.pts(F.FACE_OVAL))

    @cached_property
    def _below_chin(self) -> np.ndarray:
        """Distance below the chin along the face's down axis, in units of 1.2·IOD."""
        h, w = self.shape
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        chin, y = self.face.pt(F.CHIN), self.face.y_axis
        return ((xs - chin[0]) * y[0] + (ys - chin[1]) * y[1]) / (1.2 * self.iod)

    @cached_property
    def neck_poly(self) -> np.ndarray:
        """Neck area that fades out gradually below the chin (no visible edge)."""
        poly = fill_poly(self.shape, self.face.neck_polygon())
        return poly * (1.0 - smoothstep(0.05, 0.9, self._below_chin))

    @cached_property
    def neck_long(self) -> np.ndarray:
        """A longer, gentler neck region for tone changes, so face and neck stay matched."""
        poly = fill_poly(self.shape, self.face.neck_polygon(depth=2.6))
        return poly * (1.0 - smoothstep(0.4, 1.9, self._below_chin))

    @cached_property
    def face_poly_wide(self) -> np.ndarray:
        """Face oval pushed outward by ~0.25·IOD, so tone changes also reach ears/hairline."""
        oval = self.face.pts(F.FACE_OVAL)
        center = oval.mean(axis=0)
        vec = oval - center
        norms = np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-6)
        return fill_poly(self.shape, oval + vec / norms * 0.25 * self.iod)

    @cached_property
    def nose_tip(self) -> np.ndarray:
        """Lower nose (tip + nostrils): excluded from spot removal so nostrils survive."""
        m = fill_poly(self.shape, self.face.pts(F.NOSE_LOWER))
        return blur(dilate(m, 0.03 * self.iod), 0.02 * self.iod)

    @cached_property
    def lips_outer(self) -> np.ndarray:
        return fill_poly(self.shape, self.face.pts(F.LIPS_OUTER))

    @cached_property
    def inner_mouth(self) -> np.ndarray:
        return fill_poly(self.shape, self.face.pts(F.LIPS_INNER))

    @cached_property
    def lips(self) -> np.ndarray:
        """Lip area only (mouth opening removed), lightly feathered."""
        raw = np.clip(self.lips_outer - self.inner_mouth, 0.0, 1.0)
        return blur(raw, max(0.8, 0.07 * self.face.lip_thickness))

    @cached_property
    def lip_inner_proximity(self) -> np.ndarray:
        """≈1 along the line where the lips meet, fading toward the lip edges (for 咬唇妆)."""
        line = polyline(self.shape, self.face.pts(F.LIPS_INNER), thickness=1, closed=True)
        inv = np.where(line > 0.5, 0, 255).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        scale = 0.75 * self.face.lip_thickness
        return np.exp(-((dist / scale) ** 2)).astype(np.float32)

    @cached_property
    def eyes(self) -> np.ndarray:
        m = np.maximum(fill_poly(self.shape, self.face.pts(F.RIGHT_EYE)),
                       fill_poly(self.shape, self.face.pts(F.LEFT_EYE)))
        return blur(m, 0.012 * self.iod)

    @cached_property
    def features(self) -> np.ndarray:
        """Eyes, brows and lips (dilated): regions skin smoothing must not touch."""
        eyes = np.maximum(fill_poly(self.shape, self.face.pts(F.RIGHT_EYE)),
                          fill_poly(self.shape, self.face.pts(F.LEFT_EYE)))
        brows = np.maximum(fill_poly(self.shape, self.face.pts(F.RIGHT_BROW)),
                           fill_poly(self.shape, self.face.pts(F.LEFT_BROW)))
        m = np.maximum.reduce([
            dilate(eyes, 0.07 * self.iod),
            dilate(brows, 0.05 * self.iod),
            dilate(self.lips_outer, 0.025 * self.iod),
        ])
        return blur(m, 0.03 * self.iod)

    @cached_property
    def features_tight(self) -> np.ndarray:
        """Eyeballs, brows and lips only: what tone changes must not touch.

        Using the dilated ``features`` here would leave an un-brightened ring
        around the eyes ("panda eyes").
        """
        eyes = np.maximum(fill_poly(self.shape, self.face.pts(F.RIGHT_EYE)),
                          fill_poly(self.shape, self.face.pts(F.LEFT_EYE)))
        brows = np.maximum(fill_poly(self.shape, self.face.pts(F.RIGHT_BROW)),
                           fill_poly(self.shape, self.face.pts(F.LEFT_BROW)))
        m = np.maximum.reduce([dilate(eyes, 0.01 * self.iod), dilate(brows, 0.045 * self.iod), self.lips_outer])
        return blur(m, 0.015 * self.iod)

    # -- adaptive skin color model ------------------------------------------ #
    def _skin_samples(self) -> np.ndarray:
        f, iod = self.face, self.iod
        sample = np.zeros(self.shape, np.float32)
        for side in ("right", "left"):
            c = f.eye_center(side) + f.y_axis * 0.5 * iod + f.outward(side) * 0.03 * iod
            gaussian_ellipse(self.shape, c, 0.11 * iod, 0.09 * iod, f.angle_deg, out=sample)
        gaussian_ellipse(self.shape, f.brow_mid - f.y_axis * 0.3 * iod, 0.22 * iod, 0.08 * iod,
                         f.angle_deg, out=sample)
        bridge = (f.pt(6) + f.pt(195)) / 2.0
        gaussian_ellipse(self.shape, bridge, 0.045 * iod, 0.12 * iod, f.angle_deg, out=sample)
        return (sample > 0.5) & (self.face_poly > 0.5) & (self.features < 0.05)

    @cached_property
    def _skin_distance(self) -> tuple[np.ndarray, np.ndarray]:
        """(Mahalanobis² distance to the sampled skin chroma, luma) for every pixel."""
        ycc = cv2.cvtColor(self.img.astype(np.float32), cv2.COLOR_RGB2YCrCb)
        y, crcb = ycc[..., 0], ycc[..., 1:3]
        samples = crcb[self._skin_samples()]
        if samples.shape[0] >= 40:
            mu = samples.mean(axis=0)
            cov = np.cov(samples, rowvar=False) + np.eye(2) * 2e-5
            # One round of outlier rejection (specular highlights, stray hair...)
            d2 = np.einsum("ni,ij,nj->n", samples - mu, np.linalg.inv(cov), samples - mu)
            keep = samples[d2 < 6.0]
            if keep.shape[0] >= 30:
                mu = keep.mean(axis=0)
                cov = np.cov(keep, rowvar=False) + np.eye(2) * 2e-5
        else:  # generic skin chroma range as a fallback
            mu = np.array([0.60, 0.42], np.float32)
            cov = np.diag([0.035 ** 2, 0.04 ** 2]).astype(np.float32)
        diff = crcb - mu.astype(np.float32)
        inv = np.linalg.inv(cov).astype(np.float32)
        d2 = (diff[..., 0] ** 2 * inv[0, 0] + 2 * diff[..., 0] * diff[..., 1] * inv[0, 1]
              + diff[..., 1] ** 2 * inv[1, 1])
        return d2, y

    def _probability(self, tolerance: float) -> np.ndarray:
        d2, y = self._skin_distance
        prob = np.exp(-0.5 * d2 / (tolerance ** 2)).astype(np.float32)
        prob *= smoothstep(0.06, 0.18, y)
        prob = close_holes(prob, 0.06 * self.iod)  # blemishes are skin too!
        return blur(prob, 0.012 * self.iod)

    @cached_property
    def skin_probability(self) -> np.ndarray:
        return self._probability(2.2)

    @cached_property
    def skin_probability_loose(self) -> np.ndarray:
        """More tolerant: also accepts shadowed or warmer skin (neck, ears, jaw shadow)."""
        return self._probability(3.4)

    @cached_property
    def skin(self) -> np.ndarray:
        """Tight skin mask (face + neck): used for smoothing, spots, glow and blush."""
        region = blur(np.maximum(self.face_poly, self.neck_poly), 0.04 * self.iod)
        mask = region * np.clip(self.skin_probability * 1.25, 0.0, 1.0) * (1.0 - self.features)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    @cached_property
    def skin_wide(self) -> np.ndarray:
        """Wider, softer skin mask for brightening / skin tone, to avoid visible edges."""
        region = blur(np.maximum(self.face_poly_wide, self.neck_long), 0.18 * self.iod)
        mask = region * np.clip(self.skin_probability_loose * 1.3, 0.0, 1.0) * (1.0 - self.features_tight)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)
