"""Face reshaping (瘦脸 / 大眼) with local image warps.

Both warps use backward mapping (for every output pixel, compute where to sample
from) and only touch a region of interest, so they are fast and leave the rest
of the photo untouched.

* Eyes: a radial magnification with a smooth falloff (A. Gustafsson,
  "Interactive Image Warping", 1993 — the classic "liquify" formula).
* Face: a single smooth displacement field from Gaussian RBFs on the jaw line,
  with zero-shift anchors on the inner face. Applying many small local warps one
  after another makes the jaw wavy; one combined field keeps the contour clean.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import face as F
from .face import Face


def _roi(shape, cx: float, cy: float, radius: float):
    h, w = shape[:2]
    x0, x1 = int(max(0, np.floor(cx - radius))), int(min(w, np.ceil(cx + radius) + 1))
    y0, y1 = int(max(0, np.floor(cy - radius))), int(min(h, np.ceil(cy + radius) + 1))
    return x0, y0, x1, y1


def scale_warp(img: np.ndarray, center: np.ndarray, radius: float, strength: float) -> np.ndarray:
    """Magnify the content near ``center`` (strength 0 → none, ~0.3 → strong)."""
    if radius < 2 or strength <= 0.0:
        return img
    cx, cy = float(center[0]), float(center[1])
    x0, y0, x1, y1 = _roi(img.shape, cx, cy, radius)
    if x0 >= x1 or y0 >= y1:
        return img
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    t = np.clip((dx * dx + dy * dy) / (radius * radius), 0.0, 1.0)
    k = (1.0 - strength * (1.0 - t) ** 2).astype(np.float32)
    out = img.copy()
    out[y0:y1, x0:x1] = cv2.remap(img, cx + dx * k, cy + dy * k, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
    return out


def displacement_warp(img: np.ndarray, centers: np.ndarray, shifts: np.ndarray, sigma: float,
                      roi: tuple[int, int, int, int], step: int = 4) -> np.ndarray:
    """Warp with one smooth field built from Gaussian RBFs.

    D(x) = Σ φᵢ(x)·sᵢ / max(Σ φᵢ(x), 1): near the control points the field equals a
    smooth weighted average of their shifts; far away it decays to zero. Points with
    a zero shift act as anchors that keep features (eyes, nose, mouth) in place.
    The field is evaluated on a coarse grid and upsampled (it is very smooth).
    """
    x0, y0, x1, y1 = roi
    if x1 - x0 < 2 or y1 - y0 < 2:
        return img
    gx = np.arange(x0, x1 + step, step, dtype=np.float32)
    gy = np.arange(y0, y1 + step, step, dtype=np.float32)
    xs, ys = np.meshgrid(gx, gy)
    num_x = np.zeros_like(xs)
    num_y = np.zeros_like(xs)
    den = np.zeros_like(xs)
    inv = 1.0 / (2.0 * sigma * sigma)
    for (cx, cy), (sx, sy) in zip(centers, shifts):
        phi = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) * inv)
        num_x += phi * sx
        num_y += phi * sy
        den += phi
    den = np.maximum(den, 1.0)
    w, h = x1 - x0, y1 - y0
    # Grid node j sits at x0 + j·step, so resizing must map node centers to pixel centers.
    dx = cv2.resize(num_x / den, (len(gx) * step, len(gy) * step), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(num_y / den, (len(gx) * step, len(gy) * step), interpolation=cv2.INTER_LINEAR)
    off = step // 2
    dx, dy = dx[off:off + h, off:off + w], dy[off:off + h, off:off + w]
    pys, pxs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    out = img.copy()
    out[y0:y1, x0:x1] = cv2.remap(img, pxs - dx, pys - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def _slim_profile(rel: float) -> float:
    """How much the contour moves at height ``rel`` (0 = eye line, 1 = chin).

    One smooth bump: nothing at the temples, most at the lower cheek, about half at
    the chin. A smooth profile keeps the outline a single clean curve; per-point
    weights that peak at the jaw corner turn it into straight segments with a kink.
    """
    if rel <= 0.0:
        return 0.0
    if rel < 0.68:
        return float(np.sin(0.5 * np.pi * rel / 0.68) ** 1.6)
    t = min(1.0, (rel - 0.68) / 0.32)
    return float(1.0 - 0.5 * (0.5 - 0.5 * np.cos(np.pi * t)))


def slim_face(img: np.ndarray, face: Face, strength: float) -> np.ndarray:
    """瘦脸: shrink the lower face toward its center, like scaling the outline down."""
    if strength <= 0.0:
        return img
    iod, y = face.iod, face.y_axis
    magnitude = 0.15 * iod * float(np.clip(strength, 0.0, 1.5))
    chin_drop = max(float(np.dot(face.pt(F.CHIN) - face.eye_mid, y)), 0.5 * iod)
    center = face.eye_mid + y * 0.55 * chin_drop            # around the base of the nose
    oval = face.pts(F.FACE_OVAL)
    dense = np.vstack([oval, (oval + np.roll(oval, -1, axis=0)) / 2.0])   # landmarks + midpoints
    centers, shifts = [], []
    for p in dense:
        weight = _slim_profile(float(np.dot(p - face.eye_mid, y)) / chin_drop)
        if weight <= 0.0:
            continue
        direction = center - p
        direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
        centers.append(p)
        shifts.append(direction * magnitude * weight)
    for index in F.SLIM_ANCHORS:  # keep the inner face still
        centers.append(face.pt(index))
        shifts.append(np.zeros(2))
    sigma = 0.36 * iod     # wide and smooth: the whole lower face moves together, no kinks
    bx0, by0, bx1, by1 = face.bbox
    pad = 3.0 * sigma
    h, w = img.shape[:2]
    roi = (int(max(0, bx0 - pad)), int(max(0, by0 - pad)), int(min(w, bx1 + pad)), int(min(h, by1 + pad)))
    return displacement_warp(img, np.array(centers), np.array(shifts), sigma, roi)


def slim_nose(img: np.ndarray, face: Face, strength: float) -> np.ndarray:
    """瘦鼻: pull the nose wings toward the nose's center line; tip and cheeks stay put."""
    if strength <= 0.0:
        return img
    iod, x = face.iod, face.x_axis
    tip = face.pt(F.NOSE_TIP)
    centers, shifts = [], []
    for index in F.NOSE_LOWER:
        p = face.pt(index)
        lateral = float(np.dot(p - tip, x))
        if abs(lateral) < 0.04 * iod:
            continue
        share = min(1.0, abs(lateral) / (0.18 * iod))
        centers.append(p)
        shifts.append(-np.sign(lateral) * x * 0.07 * iod * float(np.clip(strength, 0.0, 1.5)) * share)
    if not centers:
        return img
    for index in (F.NOSE_TIP, 1, 2, 6, 0, 13):
        centers.append(face.pt(index))
        shifts.append(np.zeros(2))
    for side in (-1.0, 1.0):          # cheeks beside the nose don't move
        for dy in (-0.12, 0.12):
            centers.append(tip + x * side * 0.5 * iod + face.y_axis * dy * iod)
            shifts.append(np.zeros(2))
    pts = np.array(centers)
    sigma = 0.09 * iod
    pad = 3.0 * sigma
    h, w = img.shape[:2]
    roi = (int(max(0, pts[:, 0].min() - pad)), int(max(0, pts[:, 1].min() - pad)),
           int(min(w, pts[:, 0].max() + pad)), int(min(h, pts[:, 1].max() + pad)))
    return displacement_warp(img, pts, np.array(shifts), sigma, roi, step=2)


def enlarge_eyes(img: np.ndarray, face: Face, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return img
    amount = 0.22 * float(np.clip(strength, 0.0, 1.5))
    for side in ("right", "left"):
        radius = 0.95 * face.eye_width(side)
        img = scale_warp(img, face.eye_center(side), radius, amount)
    return img


def enlarge_iris(img: np.ndarray, face: Face, strength: float) -> np.ndarray:
    """眼瞳放大: magnify just the iris, like a larger-diameter contact lens."""
    if strength <= 0.0 or face.landmarks.shape[0] < 478:
        return img
    for center, edge in ((468, 469), (473, 474)):
        r = float(np.linalg.norm(face.pt(center) - face.pt(edge)))
        if r >= 4.0:
            img = scale_warp(img, face.pt(center), 1.9 * r, 0.17 * float(np.clip(strength, 0.0, 1.5)))
    return img


def plump_lips(img: np.ndarray, face: Face, strength: float) -> np.ndarray:
    """丰唇: a gentle magnification centered on the mouth."""
    if strength <= 0.0:
        return img
    center = (face.pt(F.MOUTH_RIGHT) + face.pt(F.MOUTH_LEFT) + face.pt(F.UPPER_LIP_TOP) + face.pt(F.LOWER_LIP_BOTTOM)) / 4.0
    return scale_warp(img, center, 0.75 * face.mouth_width, 0.16 * float(np.clip(strength, 0.0, 1.5)))
