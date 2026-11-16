"""A drawn test face with matching MediaPipe-style landmarks.

Lets the pipeline be tested end-to-end (masks, makeup, warps, API) without a
real photo or the face model. Geometry roughly follows real proportions.
"""

from __future__ import annotations

import cv2
import numpy as np

from seoulglow import face as F

W, H = 720, 900
CX, CY = 360.0, 430.0      # face ellipse center
A, B = 170.0, 225.0        # face ellipse semi-axes
EYE_Y = 380.0
EYE_DX = 64.0              # half inter-ocular distance
EYE_W, EYE_H = 58.0, 22.0
MOUTH_Y = 548.0
MOUTH_W = 100.0
SKIN = (226, 182, 158)
LIP = (186, 104, 108)


def _eye(lower_ids, upper_ids, cx, outer_on_left):
    ox = cx - EYE_W / 2 if outer_on_left else cx + EYE_W / 2
    ix = cx + EYE_W / 2 if outer_on_left else cx - EYE_W / 2
    pts = {}
    for i, idx in enumerate(lower_ids):            # outer → inner
        t = i / (len(lower_ids) - 1)
        pts[idx] = (ox + t * (ix - ox), EYE_Y + EYE_H / 2 * np.sin(np.pi * t))
    for i, idx in enumerate(upper_ids):            # inner → outer
        t = (i + 1) / (len(upper_ids) + 1)
        pts[idx] = (ix + t * (ox - ix), EYE_Y - EYE_H / 2 * np.sin(np.pi * t))
    return pts


def _brow(lower_ids, upper_ids, cx, outer_on_left):
    ox = cx - 42 if outer_on_left else cx + 42
    ix = cx + 34 if outer_on_left else cx - 34
    pts = {}
    for i, idx in enumerate(lower_ids):            # outer → inner
        t = i / (len(lower_ids) - 1)
        pts[idx] = (ox + t * (ix - ox), 346 - 6 * np.sin(np.pi * t))
    for i, idx in enumerate(upper_ids):            # inner → outer
        t = i / (len(upper_ids) - 1)
        pts[idx] = (ix + t * (ox - ix), 333 - 7 * np.sin(np.pi * t))
    return pts


def _lips():
    rx, lx = CX - MOUTH_W / 2, CX + MOUTH_W / 2
    pts = {}
    lower_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    upper_outer = [409, 270, 269, 267, 0, 37, 39, 40, 185]
    lower_inner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    upper_inner = [415, 310, 311, 312, 13, 82, 81, 80, 191]
    for i, idx in enumerate(lower_outer):
        t = i / 10
        pts[idx] = (rx + t * (lx - rx), MOUTH_Y + 19 * np.sin(np.pi * t))
    for i, idx in enumerate(upper_outer):
        t = (i + 1) / 10
        dip = 3.0 * np.exp(-((t - 0.5) / 0.08) ** 2)  # cupid's bow
        pts[idx] = (lx + t * (rx - lx), MOUTH_Y - 15 * np.sin(np.pi * t) + dip)
    irx, ilx = rx + 6, lx - 6
    for i, idx in enumerate(lower_inner):
        t = i / 10
        pts[idx] = (irx + t * (ilx - irx), MOUTH_Y + 1.0 * np.sin(np.pi * t))
    for i, idx in enumerate(upper_inner):
        t = (i + 1) / 10
        pts[idx] = (ilx + t * (irx - ilx), MOUTH_Y - 1.0 * np.sin(np.pi * t))
    return pts


def synthetic_landmarks() -> np.ndarray:
    lm = np.tile(np.array([CX, CY]), (478, 1)).astype(np.float64)
    for k, idx in enumerate(F.FACE_OVAL):
        theta = np.radians(-90 + k * 10)
        lm[idx] = (CX + A * np.cos(theta), CY + B * np.sin(theta))
    pts = {}
    pts.update(_eye(F.RIGHT_EYE[:9], F.RIGHT_EYE[9:], CX - EYE_DX, outer_on_left=True))
    pts.update(_eye(F.LEFT_EYE[:9], F.LEFT_EYE[9:], CX + EYE_DX, outer_on_left=False))
    pts.update(_brow(F.RIGHT_BROW[:5], F.RIGHT_BROW[5:], CX - EYE_DX, outer_on_left=True))
    pts.update(_brow(F.LEFT_BROW[:5], F.LEFT_BROW[5:], CX + EYE_DX, outer_on_left=False))
    pts.update(_lips())
    for idx, y in zip([168, 6, 197, 195, 5, 4, 1], [380, 400, 420, 440, 458, 470, 476]):
        pts[idx] = (CX, y)
    nose = {2: (0, 482), 97: (-12, 480), 326: (12, 480), 98: (-24, 476), 327: (24, 476),
            64: (-32, 468), 294: (32, 468), 48: (-30, 455), 278: (30, 455), 115: (-26, 445),
            344: (26, 445), 220: (-18, 440), 440: (18, 440), 45: (-10, 450), 275: (10, 450)}
    for idx, (dx, y) in nose.items():
        pts[idx] = (CX + dx, y)
    for idx, (x, y) in pts.items():
        lm[idx] = (x, y)
    return lm


def synthetic_face_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    img[:] = np.array([0.80, 0.84, 0.88], np.float32) * (1 - 0.15 * yy[..., None])
    cv2.circle(img, (95, 140), 80, (0.35, 0.62, 0.30), -1)            # a "plant" for green tests
    cv2.ellipse(img, (int(CX), int(CY + 40)), (int(A + 30), int(B + 20)), 0, 180, 360,
                (0.18, 0.12, 0.10), -1)                                  # hair
    skin = np.array(SKIN, np.float32) / 255
    cv2.rectangle(img, (int(CX - 85), int(CY + 150)), (int(CX + 85), H), (skin * 0.93).tolist(), -1)
    cv2.rectangle(img, (0, H - 120), (W, H), (0.92, 0.92, 0.95), -1)    # shirt
    cv2.ellipse(img, (int(CX), int(CY)), (int(A), int(B)), 0, 0, 360, skin.tolist(), -1)

    # skin texture + a few blemishes, so smoothing has something to do
    noise = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 1.2) * 0.035
    face = np.zeros((H, W), np.float32)
    cv2.ellipse(face, (int(CX), int(CY)), (int(A), int(B)), 0, 0, 360, 1.0, -1)
    img += (noise * face)[..., None]
    for _ in range(14):
        x = int(CX + rng.uniform(-0.6, 0.6) * A)
        y = int(CY + rng.uniform(0.0, 0.6) * B)
        cv2.circle(img, (x, y), int(rng.integers(3, 6)), (0.78, 0.55, 0.52), -1)

    lm = synthetic_landmarks()
    to_int = lambda idx: np.round(lm[idx]).astype(np.int32)  # noqa: E731
    for poly in (F.RIGHT_BROW, F.LEFT_BROW):
        cv2.fillPoly(img, [to_int(poly)], (0.25, 0.18, 0.15), lineType=cv2.LINE_AA)
    for poly, cx in ((F.RIGHT_EYE, CX - EYE_DX), (F.LEFT_EYE, CX + EYE_DX)):
        cv2.fillPoly(img, [to_int(poly)], (0.93, 0.90, 0.88), lineType=cv2.LINE_AA)
        cv2.circle(img, (int(cx), int(EYE_Y)), 10, (0.22, 0.15, 0.12), -1, lineType=cv2.LINE_AA)
    cv2.line(img, (int(CX - 6), 462), (int(CX + 6), 462), (0.70, 0.52, 0.45), 3, cv2.LINE_AA)
    cv2.fillPoly(img, [to_int(F.LIPS_OUTER)], (np.array(LIP) / 255).tolist(), lineType=cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def synthetic_face() -> F.Face:
    return F.Face(synthetic_landmarks())


class FakeAnalyzer:
    """Stands in for MediaPipe: returns the synthetic face for any image."""

    def __init__(self, faces=None):
        self.faces = faces

    def detect(self, rgb):
        if self.faces is not None:
            return list(self.faces)
        h, w = rgb.shape[:2]
        lm = synthetic_landmarks() * np.array([w / W, h / H])
        return [F.Face(lm)]
