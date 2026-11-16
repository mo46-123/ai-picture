"""Korean-style makeup: gradient lips (咬唇妆), lip gloss (水光唇), blush (腮红),
soft brows (韩系眉), eyeshadow (眼影), eyeliner (内眼线), aegyo-sal (卧蚕),
brighter eyes (亮眼) and teeth whitening.

Colors are applied in Lab space: hue/chroma move toward the target shade while
the original lightness variation is kept, so lip lines and skin texture survive.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import face as F
from . import ops
from .face import Face
from .masks import FaceRegions, blur, dilate, fill_poly, gaussian_ellipse, polyline, smoothstep


def lips(img: np.ndarray, face: Face, regions: FaceRegions, color: tuple[int, int, int],
         intensity: float, gradient: float = 0.5, gloss: float = 0.0) -> np.ndarray:
    if intensity <= 0.0:
        return img
    mask = regions.lips
    if float(mask.sum()) < 4.0:
        return img
    g = float(np.clip(gradient, 0.0, 1.0))
    weight = mask * ((1.0 - g) + g * regions.lip_inner_proximity) * float(np.clip(intensity, 0.0, 1.0))
    weight *= _corner_fade(face, regions.shape)

    lab = ops.to_lab(img)
    target = ops.rgb255_to_lab(color)
    lightness = lab[..., 0]
    mean_l = float((lightness * mask).sum() / max(float(mask.sum()), 1e-6))
    # A tint, not paint. Real lipstick is darker and less saturated where the lip is in
    # shadow, so the target chroma follows each pixel's own lightness.
    shade = np.clip(lightness / max(float(target[0]), 1.0), 0.4, 1.1)
    lab[..., 1] += (target[1] * shade - lab[..., 1]) * weight * 0.72
    lab[..., 2] += (target[2] * shade - lab[..., 2]) * weight * 0.72
    # Never lighten: lifting lip lightness toward a pale shade makes lips look chalky white.
    lab[..., 0] += (min(float(target[0]) - mean_l, 0.0) * 0.3 - 2.0) * weight
    out = ops.from_lab_where(img, lab, weight)

    if gloss > 0.0:
        out = _lip_gloss(out, face, regions, gloss)
    return out


def _corner_fade(face: Face, shape: tuple[int, int]) -> np.ndarray:
    """Fade color out toward the mouth corners (landmarks tend to overshoot there)."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.full((h, w), np.inf, np.float32)
    for idx in (F.MOUTH_RIGHT, F.MOUTH_LEFT):
        cx, cy = face.pt(idx)
        dist = np.minimum(dist, np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2))
    return smoothstep(0.0, 0.16 * face.mouth_width, dist)


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
    # Kept subtle and tinted with the lip's own color: a white specular patch looks fake.
    return np.clip(ops.screen(img, spec[..., None] * img * (0.3 * amount)), 0.0, 1.0)


def blush(img: np.ndarray, face: Face, regions: FaceRegions, color: tuple[int, int, int],
          intensity: float, height: float = 0.55, spread: float = 1.0, nose: float = 0.0) -> np.ndarray:
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
        gaussian_ellipse(shape, center, 0.24 * iod * spread, 0.16 * iod * spread, face.angle_deg, out=weight)
    if nose > 0.0:  # a touch on the nose tip (the Korean "flushed" look)
        gaussian_ellipse(shape, face.pt(F.NOSE_TIP), 0.07 * iod, 0.055 * iod, face.angle_deg, out=weight, weight=0.7 * nose)
    weight = np.clip(weight, 0.0, 1.0) * regions.skin * float(np.clip(intensity, 0.0, 1.0))

    lab = ops.to_lab(img)
    target = ops.rgb255_to_lab(color)
    # Cheeks that are already flushed (sun, natural redness) need far less blush.
    total = float(weight.sum())
    if total > 1.0:
        cheek_a = float((lab[..., 1] * weight).sum() / total)
        weight = weight * float(np.clip((float(target[1]) - cheek_a) / 14.0, 0.25, 1.0))
    w = weight * 0.38  # a diffused flush; much stronger reads as a rash on real photos
    lab[..., 1] += (target[1] - lab[..., 1]) * w
    lab[..., 2] += (target[2] - lab[..., 2]) * w
    lab[..., 0] += (target[0] - lab[..., 0]) * w * 0.15
    return ops.from_lab_where(img, lab, w)


def brows(img: np.ndarray, face: Face, regions: FaceRegions, color: tuple[int, int, int],
          amount: float, shape: float = 0.0) -> np.ndarray:
    """韩系眉: fill and soften the brows toward a soft brown (only ever darkens)."""
    if amount <= 0.0:
        return img
    iod = face.iod
    mask = np.maximum(fill_poly(regions.shape, face.pts(F.RIGHT_BROW)),
                      fill_poly(regions.shape, face.pts(F.LEFT_BROW)))
    mask = blur(dilate(mask, 0.012 * iod), 0.018 * iod) * float(np.clip(amount, 0.0, 1.0))
    # shape > 0: straight "一字眉" — fill the hollow under the arch so the lower edge is flat.
    # shape < 0: a slightly lifted, longer tail for a cooler, more defined brow.
    extra = np.zeros(regions.shape, np.float32)
    for side, brow in (("right", F.RIGHT_BROW), ("left", F.LEFT_BROW)):
        lower, upper = face.pts(brow[:5]), face.pts(brow[5:])      # outer→inner, inner→outer
        if shape > 0.0:
            chord = np.linspace(lower[-1], lower[0], 5) + face.y_axis * 0.012 * iod
            extra = np.maximum(extra, fill_poly(regions.shape, np.vstack([lower, chord])))
        elif shape < 0.0:
            tip = upper[-1] + face.outward(side) * 0.06 * iod - face.y_axis * 0.025 * iod
            extra = np.maximum(extra, fill_poly(regions.shape, np.vstack([upper[-2], tip, lower[0], lower[1]])))
    if shape != 0.0:
        mask = np.maximum(mask, blur(extra, 0.012 * iod) * float(np.clip(amount * abs(shape), 0.0, 1.0)))
    lab = ops.to_lab(img)
    target = ops.rgb255_to_lab(color)
    lightness = lab[..., 0]
    lab[..., 0] = lightness - np.clip(lightness - target[0], 0.0, None) * mask * 0.45
    lab[..., 1] += (target[1] - lab[..., 1]) * mask * 0.35
    lab[..., 2] += (target[2] - lab[..., 2]) * mask * 0.35
    return ops.from_lab_where(img, lab, mask)


def eyeshadow(img: np.ndarray, face: Face, regions: FaceRegions, color: tuple[int, int, int],
              amount: float, outer: float = 0.0) -> np.ndarray:
    """眼影: a soft wash on the upper lid, strongest at the lash line, with a small outer wing."""
    if amount <= 0.0:
        return img
    iod, y, shape = face.iod, face.y_axis, regions.shape
    region = np.zeros(shape, np.float32)
    lines = np.zeros(shape, np.uint8)
    for side, lid in (("right", F.RIGHT_UPPER_LID), ("left", F.LEFT_UPPER_LID)):
        pts = face.pts(lid)                               # outer → inner
        heights = np.linspace(0.16, 0.09, len(pts))[:, None] * iod
        top = pts - y[None, :] * heights
        wing = pts[0] + face.outward(side) * 0.07 * iod - y * 0.06 * iod
        region = np.maximum(region, fill_poly(shape, np.vstack([wing[None], pts, top[::-1]])))
        cv2.polylines(lines, [np.round(pts).astype(np.int32)], False, 255, 1)
    dist = cv2.distanceTransform(np.where(lines > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    near_lashes = np.exp(-((dist / (0.07 * iod)) ** 2)).astype(np.float32)
    weight = blur(region * (0.35 + 0.65 * near_lashes), 0.025 * iod) * (1.0 - regions.eyes)
    weight *= float(np.clip(amount, 0.0, 1.0))
    lab = ops.to_lab(img)
    target = ops.rgb255_to_lab(color)
    lab[..., 1] += (target[1] - lab[..., 1]) * weight * 0.42
    lab[..., 2] += (target[2] - lab[..., 2]) * weight * 0.42
    lab[..., 0] -= lab[..., 0] * 0.07 * weight
    if outer > 0.0:  # deepen the outer third for a soft smoky corner
        smoke = np.zeros(shape, np.float32)
        for side, lid in (("right", F.RIGHT_UPPER_LID), ("left", F.LEFT_UPPER_LID)):
            c = face.pt(lid[1]) + face.outward(side) * 0.02 * iod - y * 0.03 * iod
            gaussian_ellipse(shape, c, 0.09 * iod, 0.05 * iod, face.angle_deg, out=smoke)
        smoke = np.clip(smoke, 0.0, 1.0) * (1.0 - regions.eyes) * float(np.clip(amount * outer, 0.0, 1.0))
        lab[..., 0] -= lab[..., 0] * 0.16 * smoke
        lab[..., 1] += (target[1] - lab[..., 1]) * smoke * 0.3
        lab[..., 2] += (target[2] - lab[..., 2]) * smoke * 0.3
        weight = np.maximum(weight, smoke)
    return ops.from_lab_where(img, lab, weight)


def eyeliner(img: np.ndarray, face: Face, regions: FaceRegions, amount: float,
             tail: float = 0.045, drop: float = -0.018) -> np.ndarray:
    """内眼线: a fine dark line along the outer part of the upper lash line with a short tail."""
    if amount <= 0.0:
        return img
    iod, y, shape = face.iod, face.y_axis, regions.shape
    mask = np.zeros(shape, np.float32)
    for side, lid in (("right", F.RIGHT_UPPER_LID), ("left", F.LEFT_UPPER_LID)):
        pts = face.pts(lid)[:7] - y * 0.004 * iod        # outer three quarters of the lid
        # drop > 0 pulls the tail down (soft "puppy" eye), drop < 0 lifts it (sharper, cooler)
        end = pts[0] + face.outward(side) * tail * iod + y * drop * iod
        mask = np.maximum(mask, polyline(shape, np.vstack([end[None], pts]), thickness=max(1.0, 0.016 * iod)))
    mask = blur(mask, max(0.5, 0.005 * iod)) * float(np.clip(amount, 0.0, 1.0))
    lab = ops.to_lab(img)
    lab[..., 0] -= np.clip(lab[..., 0] - 12.0, 0.0, None) * mask * 0.55
    lab[..., 1] *= 1.0 - 0.4 * mask
    lab[..., 2] *= 1.0 - 0.4 * mask
    return ops.from_lab_where(img, lab, mask)


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
    return ops.from_lab_where(img, lab, band + crease)


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
    return ops.from_lab_where(img, lab, mask)


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
    return ops.from_lab_where(img, lab, teeth)
