"""Skin effects: spot removal (祛痘), smoothing (磨皮), brightening (美白),
glass-skin glow (水光) and matte finish (雾面)."""

from __future__ import annotations

import cv2
import numpy as np

from . import face as F
from . import ops
from .face import Face
from .imageio import to_uint8
from .masks import FaceRegions, blur, fill_poly, gaussian_ellipse, polyline, smoothstep


def spot_map(img: np.ndarray, regions: FaceRegions) -> np.ndarray:
    """Soft map of small blemishes: spots darker and/or redder than the skin around them."""
    iod = regions.iod
    h, w = img.shape[:2]
    # Spots are several pixels wide on big faces, so half resolution finds them just as well.
    scale = 0.5 if iod > 120 else 1.0
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    lab = ops.to_lab(small)
    lightness, a = lab[..., 0] / 100.0, lab[..., 1]
    sigma = max(1.5, 0.035 * iod * scale)
    local_l = ops.gaussian(lightness, 1.6 * sigma)
    darker = local_l - ops.gaussian(lightness, 0.25 * sigma)
    redder = (ops.gaussian(a, 0.25 * sigma) - ops.gaussian(a, 1.6 * sigma)) / 20.0
    score = np.clip(smoothstep(0.02, 0.06, darker) + 0.6 * smoothstep(0.12, 0.45, redder), 0.0, 1.0)
    # Broad dark areas are shading (nasolabial folds, jaw shadow), not spots.
    broad = smoothstep(0.03, 0.08, ops.gaussian(lightness, 5.0 * sigma) - local_l)
    score = blur(score * (1.0 - broad), 0.01 * iod * scale)
    if scale < 1.0:
        score = cv2.resize(score, (w, h), interpolation=cv2.INTER_LINEAR)
    return score * regions.skin * (1.0 - regions.nose_tip)


def remove_spots(img: np.ndarray, regions: FaceRegions, strength: float) -> np.ndarray:
    """祛痘: replace detected spots with a median-filtered version of the surrounding skin."""
    if strength <= 0.0:
        return img
    weight = np.clip(spot_map(img, regions) * 1.6, 0.0, 1.0) * float(np.clip(strength * 1.5, 0.0, 1.0))
    ys, xs = np.nonzero(weight > 0.02)
    if ys.size == 0:
        return img
    k = max(3, int(round(0.09 * regions.iod)) | 1)
    h, w = img.shape[:2]   # heal only the box that actually contains spots
    y0, y1 = max(0, int(ys.min()) - k), min(h, int(ys.max()) + k + 1)
    x0, x1 = max(0, int(xs.min()) - k), min(w, int(xs.max()) + k + 1)
    sub = img[y0:y1, x0:x1]
    healed = cv2.medianBlur(to_uint8(sub), k).astype(np.float32) / 255.0
    out = img.copy()
    out[y0:y1, x0:x1] = ops.lerp(sub, healed, weight[y0:y1, x0:x1])
    return out


def smooth(img: np.ndarray, regions: FaceRegions, strength: float, texture_keep: float = 0.4) -> np.ndarray:
    """Remove blemishes and unevenness while keeping fine pore texture.

    1. A self-guided filter flattens mid-frequency detail (spots, redness, bumps)
       but preserves strong edges (nose, jaw, hairline).
    2. The finest detail layer (pores) is added back so skin doesn't look plastic.
    3. The result is blended in through the soft skin mask.
    """
    if strength <= 0.0:
        return img
    iod = regions.iod
    radius = max(2, int(round(0.05 * iod)))
    eps = (0.02 + 0.035 * min(strength, 1.0)) ** 2
    base = ops.guided_smooth(img, radius, eps)
    # Frequency separation: everything finer than ~1% of the eye distance is pore
    # texture and is kept almost entirely; only the uneven mid frequencies are flattened.
    fine = img - ops.gaussian(img, max(0.8, 0.011 * iod))
    smoothed = base + fine * float(np.clip(texture_keep, 0.0, 1.0))
    weight = regions.skin * float(np.clip(strength, 0.0, 1.0))
    return np.clip(ops.lerp(img, smoothed, weight), 0.0, 1.0)


def _cheek_reference(lab: np.ndarray, face: Face, regions: FaceRegions) -> tuple[float, float, float] | None:
    """Mean Lab of clean cheek skin, the target that under-eye skin is lifted toward."""
    iod = face.iod
    ref = np.zeros(regions.shape, np.float32)
    for side in ("right", "left"):
        c = face.eye_center(side) + face.y_axis * 0.55 * iod + face.outward(side) * 0.05 * iod
        gaussian_ellipse(regions.shape, c, 0.1 * iod, 0.07 * iod, face.angle_deg, out=ref)
    w = ref * regions.skin
    total = float(w.sum())
    if total < 20.0:
        return None
    return tuple(float((lab[..., k] * w).sum() / total) for k in range(3))  # type: ignore[return-value]


def under_eye(img: np.ndarray, face: Face, regions: FaceRegions, amount: float) -> np.ndarray:
    """去黑眼圈: lift the dark, purplish band under each eye toward the cheek's tone."""
    if amount <= 0.0:
        return img
    iod, y = face.iod, face.y_axis
    band = np.zeros(regions.shape, np.float32)
    for lid in (F.RIGHT_LOWER_LID, F.LEFT_LOWER_LID):
        pts = face.pts(lid)
        top = pts + y * 0.035 * iod
        bottom = pts[::-1] + y * 0.27 * iod
        band = np.maximum(band, fill_poly(regions.shape, np.vstack([top, bottom])))
    band = blur(band, 0.05 * iod) * (1.0 - regions.eyes) * regions.face_poly * (1.0 - regions.fine_lines)
    lab = ops.to_lab(img)
    ref = _cheek_reference(lab, face, regions)
    if ref is None:
        return img
    sigma = 0.03 * iod
    low_l = ops.gaussian(lab[..., 0], sigma)
    lift = np.clip(ref[0] - low_l, 0.0, 30.0)
    w = band * float(np.clip(amount, 0.0, 1.5))
    lab[..., 0] += lift * w * 0.9
    lab[..., 1] += (ref[1] - ops.gaussian(lab[..., 1], sigma)) * w * 0.5
    lab[..., 2] += (ref[2] - ops.gaussian(lab[..., 2], sigma)) * w * 0.5
    return ops.from_lab_where(img, lab, w)


def soften_folds(img: np.ndarray, face: Face, regions: FaceRegions, amount: float) -> np.ndarray:
    """淡化法令纹: fill the dark crease running from the nose wing to the mouth corner."""
    if amount <= 0.0:
        return img
    iod, x, y = face.iod, face.x_axis, face.y_axis
    tip = face.pt(F.NOSE_TIP)
    nose = face.pts(F.NOSE_LOWER)
    lateral = (nose - tip) @ x
    mask = np.zeros(regions.shape, np.float32)
    for side, corner in (("right", F.MOUTH_RIGHT), ("left", F.MOUTH_LEFT)):
        out = face.outward(side)
        wing = nose[int(np.argmin(lateral))] if side == "right" else nose[int(np.argmax(lateral))]
        start = wing + out * 0.05 * iod - y * 0.03 * iod
        end = face.pt(corner) + out * 0.06 * iod + y * 0.06 * iod
        ctrl = (start + end) / 2.0 + out * 0.05 * iod
        t = np.linspace(0.0, 1.0, 14)[:, None]
        curve = (1 - t) ** 2 * start + 2 * (1 - t) * t * ctrl + t ** 2 * end
        mask = np.maximum(mask, polyline(regions.shape, curve, thickness=0.11 * iod))
    mask = blur(mask, 0.03 * iod) * regions.skin
    lab = ops.to_lab(img)
    lightness = lab[..., 0]
    sigma = 0.03 * iod
    crease = np.clip(ops.gaussian(lightness, 3.0 * sigma) - ops.gaussian(lightness, 0.6 * sigma), 0.0, 20.0)
    lab[..., 0] = lightness + crease * mask * float(np.clip(amount, 0.0, 1.5))
    return ops.from_lab_where(img, lab, mask)


NOMINAL_SKIN = (66.0, 13.0, 17.0)   # Lab of evenly lit light skin


def relight(img: np.ndarray, regions: FaceRegions, strength: float = 1.0) -> np.ndarray:
    """补光: bring a face that is in shade or under colored light back to normal light.

    A retoucher's first move on a dim, blue-grey face is to lift its exposure and
    neutralize the cast; makeup put on top of uncorrected shade looks like paint.
    Well-lit faces are left alone.
    """
    if strength <= 0.0:
        return img
    core = regions.skin
    total = float(core.sum())
    if total < 50.0:
        return img
    lab = ops.to_lab(img)
    mean = [float((lab[..., c] * core).sum() / total) for c in range(3)]
    gain = float(np.clip(NOMINAL_SKIN[0] / max(mean[0], 1.0) - 1.0, 0.0, 0.5))
    shift_a = float(np.clip(0.65 * (NOMINAL_SKIN[1] - mean[1]), -4.0, 10.0))
    shift_b = float(np.clip(0.65 * (NOMINAL_SKIN[2] - mean[2]), -6.0, 14.0))
    if gain < 0.03 and abs(shift_a) < 1.5 and abs(shift_b) < 1.5:
        return img
    w = regions.face_light * float(np.clip(strength, 0.0, 1.0))
    lab[..., 0] *= 1.0 + gain * w
    lab[..., 1] += shift_a * w
    lab[..., 2] += shift_b * w
    return ops.from_lab_where(img, lab, w)


def brighten(img: np.ndarray, regions: FaceRegions, amount: float,
             tone_a: float = 0.0, tone_b: float = 0.0) -> np.ndarray:
    """Lift skin lightness with a log curve and nudge its hue (a: +pink, b: -yellow)."""
    if amount <= 0.0 and tone_a == 0.0 and tone_b == 0.0:
        return img
    mask = regions.skin_wide
    lab = ops.to_lab(img)
    lightness = lab[..., 0] / 100.0
    if amount > 0.0:
        # Gentle on purpose: real Korean retouching brightens skin a few L* units,
        # it doesn't turn it white (amount 0.3 ≈ +5 L* in the midtones).
        beta = 1.0 + 1.6 * float(np.clip(amount, 0.0, 1.5))
        lifted = np.log1p((beta - 1.0) * lightness) / np.log(beta)
        lab[..., 0] = (lightness + (lifted - lightness) * mask) * 100.0
    lab[..., 1] += tone_a * mask
    lab[..., 2] += tone_b * mask
    return ops.from_lab_where(img, lab, mask)


def glass_glow(img: np.ndarray, face: Face, regions: FaceRegions, amount: float,
               reference: np.ndarray | None = None) -> np.ndarray:
    """水光肌: a dewy, slightly oily shine rather than a flat white glow.

    1. The skin's own highlights are taken from ``reference`` (the photo before
       smoothing, which would otherwise flatten them) and amplified, so the shine
       sits exactly where light really hits and keeps its sparkle.
    2. Small, tight specular spots are added on the high points of the face
       (nose bridge and tip, cheekbones, cupid's bow, chin, forehead).
    3. Midtones get a touch more contrast: wet-looking skin is glossy, not pale.
    """
    if amount <= 0.0:
        return img
    iod, shape, ang = face.iod, regions.shape, face.angle_deg
    y = face.y_axis
    skin = regions.skin
    lum = ops.luminance(img)

    # 1) natural highlights, amplified
    ref_lum = ops.luminance(reference) if reference is not None else lum
    local = ops.gaussian(ref_lum, 0.09 * iod)
    natural = np.clip(ref_lum - local, 0.0, 0.25) * skin * smoothstep(0.35, 0.7, ref_lum)

    # 2) placed specular spots (tight cores)
    spots = np.zeros(shape, np.float32)
    bridge = (face.pt(6) + face.pt(197) + face.pt(195)) / 3.0
    gaussian_ellipse(shape, bridge, 0.022 * iod, 0.13 * iod, ang, out=spots, weight=1.0)
    gaussian_ellipse(shape, face.pt(F.NOSE_TIP) - y * 0.03 * iod, 0.035 * iod, 0.03 * iod, ang, out=spots, weight=0.9)
    for side in ("right", "left"):
        c = face.eye_center(side) + y * 0.33 * iod + face.outward(side) * 0.2 * iod
        tilt = ang + (-22.0 if side == "left" else 22.0)
        gaussian_ellipse(shape, c, 0.11 * iod, 0.04 * iod, tilt, out=spots, weight=0.9)
    gaussian_ellipse(shape, face.brow_mid - y * 0.36 * iod, 0.13 * iod, 0.05 * iod, ang, out=spots, weight=0.6)
    gaussian_ellipse(shape, face.pt(152) - y * 0.17 * iod, 0.06 * iod, 0.03 * iod, ang, out=spots, weight=0.5)
    gaussian_ellipse(shape, face.pt(0) - y * 0.035 * iod, 0.035 * iod, 0.014 * iod, ang, out=spots, weight=0.6)
    # Shine needs light to reflect: spots stay faint on skin that is in shadow.
    spots = np.clip(spots, 0.0, 1.0) ** 1.5 * skin * smoothstep(0.3, 0.75, lum)

    shine = natural * (2.2 * amount) + spots * (0.2 * amount)
    tint = np.array([1.0, 0.985, 0.97], np.float32)
    out = ops.screen(img, np.clip(shine, 0.0, 0.6)[..., None] * tint)

    # 3) glossy midtone contrast on skin
    mid = float((lum * skin).sum() / max(float(skin.sum()), 1e-6))
    out = out + ((out - mid) * (0.10 * amount)) * skin[..., None]
    return np.clip(out, 0.0, 1.0)


def matte(img: np.ndarray, regions: FaceRegions, amount: float) -> np.ndarray:
    """奶油雾面: compress shiny highlights on the skin for a soft, velvety finish."""
    if amount <= 0.0:
        return img
    skin = regions.skin
    lum = ops.luminance(img)
    core = lum[skin > 0.6]
    if core.size < 50:
        return img
    threshold = float(np.percentile(core, 80))
    over = np.clip(lum - threshold, 0.0, None)
    target = lum - over * 0.75 * amount * skin
    ratio = np.where(lum > 1e-4, target / np.maximum(lum, 1e-4), 1.0)
    return np.clip(img * ratio[..., None], 0.0, 1.0)
