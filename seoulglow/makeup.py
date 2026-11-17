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


def _lab(color) -> np.ndarray:
    """Accept a shade as an (r, g, b) tuple or as an already light-adapted Lab triple."""
    return color if isinstance(color, np.ndarray) else ops.rgb255_to_lab(color)


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
    target = _lab(color)
    lightness = lab[..., 0]
    mean_l = float((lightness * mask).sum() / max(float(mask.sum()), 1e-6))
    # A tint, not paint. Real lipstick is darker and less saturated where the lip is in
    # shadow, so the target chroma follows each pixel's own lightness.
    shade = np.clip(lightness / max(float(target[0]), 1.0), 0.4, 1.1) ** 1.5   # dark lips take less color
    lab[..., 1] += (target[1] * shade - lab[..., 1]) * weight * 0.82
    lab[..., 2] += (target[2] * shade - lab[..., 2]) * weight * 0.82
    # Never lighten: lifting lip lightness toward a pale shade makes lips look chalky white.
    # Darken toward the shade freely, but lighten only lips that are much darker than it
    # (lips in shadow) and never by much: lifting normal lips makes them chalky.
    gap = float(target[0]) - mean_l
    lab[..., 0] += (gap * 0.3 if gap < 0.0 else min(max(gap - 6.0, 0.0) * 0.7, 14.0)) * weight
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
          intensity: float, height: float = 0.55, spread: float = 1.0, nose: float = 0.0,
          outward: float = 0.07) -> np.ndarray:
    """Round, diffused blush on the cheeks (韩系腮红).

    ``height`` places the blush between the eye line (0) and the nose tip (1);
    ``outward`` moves it from under the pupil toward the cheekbone. The Korean
    "under-eye" blush sits high and wide: a soft cloud from below the outer half of
    the eye out to the edge of the face, not a dot next to the nose.
    """
    if intensity <= 0.0:
        return img
    iod, shape = face.iod, regions.shape
    lab = ops.to_lab(img)
    target = _lab(color)
    weight = np.zeros(shape, np.float32)
    for side in ("right", "left"):
        center = (face.eye_center(side) + face.y_axis * (height * face.eye_to_nose)
                  + face.outward(side) * outward * iod)
        one = gaussian_ellipse(shape, center, 0.26 * iod * spread, 0.17 * iod * spread, face.angle_deg)
        one = np.clip(one, 0.0, 1.0) * regions.skin
        # A cheek that is already flushed needs less; judged per cheek so both end up alike.
        total = float(one.sum())
        if total > 1.0:
            cheek_a = float((lab[..., 1] * one).sum() / total)
            one = one * float(np.clip((float(target[1]) - cheek_a) / 12.0, 0.45, 1.0))
        weight = np.maximum(weight, one)
    if nose > 0.0:  # a touch on the nose tip (the Korean "flushed" look)
        tip = gaussian_ellipse(shape, face.pt(F.NOSE_TIP), 0.07 * iod, 0.055 * iod, face.angle_deg)
        weight = np.maximum(weight, np.clip(tip, 0.0, 1.0) * regions.skin * 0.7 * nose)
    w = weight * float(np.clip(intensity, 0.0, 1.0)) * 0.5
    # Color only: real blush tints the skin, it does not darken it into a bruise.
    lab[..., 1] += (target[1] - lab[..., 1]) * w
    lab[..., 2] += (target[2] - lab[..., 2]) * w
    return ops.from_lab_where(img, lab, w)


def brows(img: np.ndarray, face: Face, regions: FaceRegions, color: tuple[int, int, int],
          amount: float, shape: float = 0.0) -> np.ndarray:
    """韩系眉: fill and soften the brows toward a soft brown (only ever darkens)."""
    if amount <= 0.0:
        return img
    iod = face.iod
    mask = np.maximum(fill_poly(regions.shape, face.pts(F.RIGHT_BROW)),
                      fill_poly(regions.shape, face.pts(F.LEFT_BROW)))
    mask = blur(dilate(mask, 0.004 * iod), 0.016 * iod) * float(np.clip(amount, 0.0, 1.0))
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
    target = _lab(color)
    lightness = lab[..., 0]
    # Reinforce the brow hair that is there; bare skin (or a fringe covering the brow)
    # gets only a faint fill, otherwise the brow looks drawn on with a marker.
    skin_l = float((lightness * regions.skin).sum() / max(float(regions.skin.sum()), 1e-6))
    mask = mask * (0.15 + 0.85 * smoothstep(4.0, 18.0, skin_l - ops.gaussian(lightness, 0.012 * iod)))
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
    target = _lab(color)
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


def lashes(img: np.ndarray, face: Face, regions: FaceRegions, amount: float) -> np.ndarray:
    """睫毛: short fine strokes fanning out from the outer two thirds of the upper lash line."""
    iod = face.iod
    if amount <= 0.0 or iod < 90.0:       # too few pixels to draw individual lashes
        return img
    shape = regions.shape
    canvas = np.zeros(shape, np.uint8)
    up = -face.y_axis
    for side, lid in (("right", F.RIGHT_UPPER_LID), ("left", F.LEFT_UPPER_LID)):
        pts = face.pts(lid)[:6]                          # outer corner → past the middle
        out = face.outward(side)
        for i in range(len(pts) - 1):
            for t in (0.0, 0.5):
                base = pts[i] * (1 - t) + pts[i + 1] * t
                lean = max(0.0, 1.0 - (i + t) / 4.0)     # outer lashes sweep outward
                tip = base + (up * (0.054 - 0.012 * (i + t) / 5.0) + out * 0.035 * lean) * iod
                cv2.line(canvas, tuple(np.round(base * 4).astype(int)), tuple(np.round(tip * 4).astype(int)),
                         255, 1, cv2.LINE_AA, shift=2)
    mask = canvas.astype(np.float32) / 255.0 * float(np.clip(amount, 0.0, 1.0))
    lab = ops.to_lab(img)
    lab[..., 0] -= np.clip(lab[..., 0] - 10.0, 0.0, None) * mask * 0.6
    lab[..., 1] *= 1.0 - 0.5 * mask
    lab[..., 2] *= 1.0 - 0.5 * mask
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
    lab[..., 0] += (8.0 * band - 3.5 * crease) * amount
    lab[..., 2] += 1.0 * band * amount
    return ops.from_lab_where(img, lab, band + crease)


def brighten_eyes(img: np.ndarray, regions: FaceRegions, amount: float, face: Face | None = None) -> np.ndarray:
    """亮眼: open up dark eyes, add contrast and a catchlight, whiten the sclera.

    Eyes in shade come out as black slits; retouchers lift them until the iris and
    the white can be told apart, then add a small catchlight so they look alive.
    """
    if amount <= 0.0:
        return img
    mask = regions.eyes
    if float(mask.sum()) < 4.0:
        return img
    lab = ops.to_lab(img)
    lightness, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    mean_l = float((lightness * mask).sum() / max(float(mask.sum()), 1e-6))
    # Eyes lost in shadow: lift the lighter parts (whites, iris) but leave pupils and
    # lashes dark, otherwise the whole eye turns into a pale smear.
    rescue = float(np.clip(40.0 - mean_l, 0.0, 14.0)) * smoothstep(mean_l - 6.0, mean_l + 12.0, lightness)
    lifted = lightness + ((lightness - mean_l) * 0.35 + 4.0 + rescue) * mask * amount
    lab[..., 0] = lifted
    chroma = np.sqrt(a * a + b * b)
    sclera = mask * smoothstep(46.0, 66.0, lifted) * (1.0 - smoothstep(16.0, 34.0, chroma))
    lab[..., 1] = a * (1.0 - 0.8 * sclera * min(1.0, amount))
    lab[..., 2] = b * (1.0 - 0.8 * sclera * min(1.0, amount))
    out = ops.from_lab_where(img, lab, mask)
    if face is not None and face.landmarks.shape[0] >= 478:        # iris centers: 468 (right), 473 (left)
        spark = np.zeros(regions.shape, np.float32)
        for center, edge in ((468, 469), (473, 474)):
            r = float(np.linalg.norm(face.pt(center) - face.pt(edge)))
            if r < 5.0:      # iris too small: a drawn catchlight would just be a white dot
                continue
            c = face.pt(center) + np.array([-0.3, -0.35]) * r
            gaussian_ellipse(regions.shape, c, 0.22 * r, 0.22 * r, out=spark)
        spark = np.clip(spark, 0.0, 1.0) * mask * (0.55 * amount)
        out = np.clip(ops.screen(out, spark[..., None] * np.ones(3, np.float32)), 0.0, 1.0)
    return out


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


def iris_lens(img: np.ndarray, face: Face, regions: FaceRegions, amount: float) -> np.ndarray:
    """美瞳: a lighter, slightly grey iris with a dark outer ring and bigger catchlights."""
    if amount <= 0.0 or face.landmarks.shape[0] < 478:
        return img
    shape = regions.shape
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    iris = np.zeros(shape, np.float32)
    ring = np.zeros(shape, np.float32)
    pupil = np.zeros(shape, np.float32)
    spark = np.zeros(shape, np.float32)
    for center, edge in ((468, 469), (473, 474)):
        c = face.pt(center)
        r = float(np.linalg.norm(c - face.pt(edge)))
        if r < 5.0:
            continue
        d = np.sqrt((xs - c[0]) ** 2 + (ys - c[1]) ** 2) / r
        # the band between pupil and rim is what a colored lens changes
        pupil = np.maximum(pupil, 1.0 - smoothstep(0.26, 0.42, d))
        iris = np.maximum(iris, smoothstep(0.38, 0.6, d) * (1.0 - smoothstep(0.82, 1.0, d)))
        ring = np.maximum(ring, smoothstep(0.82, 0.96, d) * (1.0 - smoothstep(1.0, 1.12, d)))
        gaussian_ellipse(shape, c + np.array([-0.32, -0.36]) * r, 0.17 * r, 0.17 * r, out=spark)
        gaussian_ellipse(shape, c + np.array([0.36, 0.34]) * r, 0.09 * r, 0.09 * r, out=spark, weight=0.5)
    eye = regions.eyes
    k = float(np.clip(amount, 0.0, 1.0))
    iris, ring = iris * eye * k, np.maximum(ring, pupil * 0.8) * eye * k    # rim and pupil stay deep
    lab = ops.to_lab(img)
    lightness = lab[..., 0]
    # dark irises are lifted toward a mid grey; an iris that is already light is left alone
    lab[..., 0] = lightness + np.clip(40.0 - lightness, -30.0, 20.0) * iris * 0.7 - np.clip(lightness - 14.0, 0.0, None) * ring * 0.6
    lab[..., 1] += (-1.0 - lab[..., 1]) * iris * 0.45      # toward a cool grey
    lab[..., 2] += (-9.0 - lab[..., 2]) * iris * 0.45
    weight = np.maximum(iris, ring)
    out = ops.from_lab_where(img, lab, weight)
    spark = np.clip(spark, 0.0, 1.0) * eye * (0.6 * k)
    return np.clip(ops.screen(out, spark[..., None] * np.ones(3, np.float32)), 0.0, 1.0)
