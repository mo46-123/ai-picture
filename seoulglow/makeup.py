"""Korean-style makeup: gradient lips (咬唇妆), lip gloss (水光唇), blush (腮红),
aegyo-sal (卧蚕), brighter eyes (亮眼) and teeth whitening.

Colors are applied in Lab space: hue/chroma move toward the target shade while
the original lightness variation is kept, so lip lines and skin texture survive.
"""

from __future__ import annotations

import numpy as np

from . import face as F
from . import ops
from .face import Face
from .masks import FaceRegions, blur, gaussian_ellipse, polyline, smoothstep


def lips(img: np.ndarray, face: Face, regions: FaceRegions, color: tuple[int, int, int],
         intensity: float, gradient: float = 0.5, gloss: float = 0.0) -> np.ndarray:
    if intensity <= 0.0:
        return img
    mask = regions.lips
    if float(mask.sum()) < 4.0:
        return img
    g = float(np.clip(gradient, 0.0, 1.0))
    weight = mask * ((1.0 - g) + g * regions.lip_inner_proximity) * float(np.clip(intensity, 0.0, 1.0))

    lab = ops.to_lab(img)
    target = ops.rgb255_to_lab(color)
    mean_l = float((lab[..., 0] * mask).sum() / max(float(mask.sum()), 1e-6))
    lab[..., 1] += (target[1] - lab[..., 1]) * weight * 0.9
    lab[..., 2] += (target[2] - lab[..., 2]) * weight * 0.9
    lab[..., 0] += (target[0] - mean_l) * weight * 0.55
    out = ops.from_lab(lab)

    if gloss > 0.0:
        out = _lip_gloss(out, face, regions, gloss)
    return out


def _lip_gloss(img: np.ndarray, face: Face, regions: FaceRegions, amount: float) -> np.ndarray:
    shape, ang, mw = regions.shape, face.angle_deg, face.mouth_width
    lower_h = float(np.linalg.norm(face.pt(F.LOWER_LIP_BOTTOM) - face.pt(F.LOWER_LIP_TOP)))
    upper_h = float(np.linalg.norm(face.pt(F.UPPER_LIP_TOP) - face.pt(F.UPPER_LIP_BOTTOM)))
    spec = np.zeros(shape, np.float32)
    c_low = face.pt(F.LOWER_LIP_TOP) + 0.42 * (face.pt(F.LOWER_LIP_BOTTOM) - face.pt(F.LOWER_LIP_TOP))
    gaussian_ellipse(shape, c_low, 0.13 * mw, max(0.18 * lower_h, 0.8), ang, out=spec, weight=1.0)
    c_up = face.pt(F.UPPER_LIP_BOTTOM) + 0.45 * (face.pt(F.UPPER_LIP_TOP) - face.pt(F.UPPER_LIP_BOTTOM))
    gaussian_ellipse(shape, c_up, 0.07 * mw, max(0.16 * upper_h, 0.8), ang, out=spec, weight=0.45)
    lum = ops.luminance(img)
    mask = regions.lips
    mean = float((lum * mask).sum() / max(float(mask.sum()), 1e-6))
    texture = 0.55 + 0.9 * np.clip((lum - mean) * 4.0 + 0.5, 0.0, 1.0)  # follow natural lip texture
    spec = np.clip(spec, 0.0, 1.0) * mask * texture
    return np.clip(ops.screen(img, spec[..., None] * (0.55 * amount)), 0.0, 1.0)


def blush(img: np.ndarray, face: Face, regions: FaceRegions, color: tuple[int, int, int],
          intensity: float, height: float = 0.55, spread: float = 1.0) -> np.ndarray:
    """Round, diffused blush on the apples of the cheeks (韩系苹果肌腮红).

    ``height`` places the blush between the eye line (0) and the nose tip (1);
    K-drama "under-eye blush" sits higher (~0.45).
    """
    if intensity <= 0.0:
        return img
    iod, shape = face.iod, regions.shape
    weight = np.zeros(shape, np.float32)
    for side in ("right", "left"):
        center = (face.eye_center(side) + face.y_axis * (height * face.eye_to_nose)
                  + face.outward(side) * 0.07 * iod)
        gaussian_ellipse(shape, center, 0.19 * iod * spread, 0.13 * iod * spread, face.angle_deg, out=weight)
    weight = np.clip(weight, 0.0, 1.0) * regions.skin * float(np.clip(intensity, 0.0, 1.0))

    lab = ops.to_lab(img)
    target = ops.rgb255_to_lab(color)
    w = weight * 0.6
    lab[..., 1] += (target[1] - lab[..., 1]) * w
    lab[..., 2] += (target[2] - lab[..., 2]) * w
    lab[..., 0] += (target[0] - lab[..., 0]) * w * 0.15
    return ops.from_lab(lab)


def aegyo_sal(img: np.ndarray, face: Face, regions: FaceRegions, amount: float) -> np.ndarray:
    """卧蚕: a soft highlight band under the lower lash line plus a faint crease below."""
    if amount <= 0.0:
        return img
    iod, shape = face.iod, regions.shape
    band = np.zeros(shape, np.float32)
    crease = np.zeros(shape, np.float32)
    for side, lid in (("right", F.RIGHT_EYE_LOWER_LID), ("left", F.LEFT_EYE_LOWER_LID)):
        pts = face.pts(lid)
        offset = face.y_axis * (0.45 * face.eye_height(side) + 0.035 * iod)
        band = np.maximum(band, polyline(shape, pts + offset, thickness=0.055 * iod))
        crease = np.maximum(crease, polyline(shape, pts + offset + face.y_axis * 0.06 * iod,
                                             thickness=0.025 * iod))
    keep = regions.face_poly * (1.0 - regions.eyes)
    band = blur(band, 0.022 * iod) * keep
    crease = blur(crease, 0.02 * iod) * keep * (1.0 - band)
    lab = ops.to_lab(img)
    lab[..., 0] += (6.5 * band - 2.5 * crease) * amount
    lab[..., 2] += 1.0 * band * amount
    return ops.from_lab(lab)


def brighten_eyes(img: np.ndarray, regions: FaceRegions, amount: float) -> np.ndarray:
    """亮眼: a touch more contrast in the iris and whiter, less red sclera."""
    if amount <= 0.0:
        return img
    mask = regions.eyes
    if float(mask.sum()) < 4.0:
        return img
    lab = ops.to_lab(img)
    lightness, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    mean_l = float((lightness * mask).sum() / max(float(mask.sum()), 1e-6))
    lab[..., 0] = lightness + ((lightness - mean_l) * 0.3 + 4.0) * mask * amount
    chroma = np.sqrt(a * a + b * b)
    sclera = mask * smoothstep(50.0, 72.0, lightness) * (1.0 - smoothstep(10.0, 26.0, chroma))
    lab[..., 1] = a * (1.0 - 0.5 * sclera * amount)
    lab[..., 2] = b * (1.0 - 0.5 * sclera * amount)
    return ops.from_lab(lab)


def whiten_teeth(img: np.ndarray, face: Face, regions: FaceRegions, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return img
    inner = regions.inner_mouth
    if float(inner.sum()) < 0.002 * face.iod ** 2:  # mouth closed
        return img
    mask = blur(inner, 0.01 * face.iod)
    lab = ops.to_lab(img)
    teeth = mask * smoothstep(42.0, 68.0, lab[..., 0]) * amount
    lab[..., 2] *= 1.0 - 0.6 * teeth
    lab[..., 1] *= 1.0 - 0.3 * teeth
    lab[..., 0] += 7.0 * teeth
    return ops.from_lab(lab)
