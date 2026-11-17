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


def even_out(img: np.ndarray, regions: FaceRegions, strength: float) -> np.ndarray:
    """匀肤: what a retoucher does for acne and acne scars.

    * Color: redness is a chroma problem, so a*/b* are pulled toward their local
      average over the skin. Breakouts lose their red without touching texture.
    * Relief: pimples and pits are bumps a few percent of the eye distance wide. That
      band of the lightness is turned down (inverted high-pass), while pores (finer)
      and the face's shading (broader) are left alone.
    """
    if strength <= 0.0:
        return img
    iod = regions.iod
    w = regions.skin * float(np.clip(strength, 0.0, 1.0))
    if float(w.sum()) < 20.0:
        return img
    lab = ops.to_lab(img)
    sigma = 0.1 * iod
    norm = np.maximum(ops.gaussian(regions.skin, sigma), 1e-3)
    for c in (1, 2):
        local = ops.gaussian(lab[..., c] * regions.skin, sigma) / norm
        lab[..., c] += (local - lab[..., c]) * w * 0.85
    lightness = lab[..., 0]
    band = ops.gaussian(lightness, max(0.8, 0.012 * iod)) - ops.gaussian(lightness, 0.07 * iod)
    lab[..., 0] = lightness - np.clip(band, -12.0, 12.0) * w * 0.6
    # Small dark dots (blackheads, tiny scabs, pit shadows) are finer than that band.
    # Only their dark side is filled in, so the bright sparkle of pores survives.
    dots = np.clip(lab[..., 0] - ops.gaussian(lab[..., 0], max(0.8, 0.012 * iod)), -15.0, 0.0)
    lab[..., 0] -= dots * w * 0.8
    return ops.from_lab_where(img, lab, w)


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


NOMINAL_SKIN = (68.0, 13.0, 17.0)   # Lab of evenly lit light skin


def relight(img: np.ndarray, regions: FaceRegions, strength: float = 1.0,
            target_l: float | None = None) -> np.ndarray:
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
    goal = NOMINAL_SKIN[0] if target_l is None else target_l
    gain = float(np.clip(goal / max(mean[0], 1.0) - 1.0, 0.0, 0.7))
    shift_a = float(np.clip(0.65 * (NOMINAL_SKIN[1] - mean[1]), -4.0, 10.0))
    shift_b = float(np.clip(0.65 * (NOMINAL_SKIN[2] - mean[2]), -6.0, 14.0))
    if gain < 0.03 and abs(shift_a) < 1.5 and abs(shift_b) < 1.5:
        return img
    w = regions.face_light * float(np.clip(strength, 0.0, 1.0))
    lab[..., 0] *= 1.0 + gain * w * (1.0 - smoothstep(62.0, 88.0, lab[..., 0]))   # never blow out lit areas
    lab[..., 1] += shift_a * w
    lab[..., 2] += shift_b * w
    if gain > 0.15:
        # A flatly lifted face looks pasted on. Add a soft warm key light from the side
        # that was already brighter, so the face keeps its shape.
        face = regions.face
        h, wd = regions.shape
        ys, xs = np.mgrid[0:h, 0:wd].astype(np.float32)
        side = ((xs - face.eye_mid[0]) * face.x_axis[0] + (ys - face.eye_mid[1]) * face.x_axis[1]) / regions.iod
        left = float((lab[..., 0] * core * (side > 0)).sum() / max(float((core * (side > 0)).sum()), 1.0))
        right = float((lab[..., 0] * core * (side < 0)).sum() / max(float((core * (side < 0)).sum()), 1.0))
        key = smoothstep(-0.6, 0.9, side if left >= right else -side) * w * min(1.0, gain / 0.4)
        lab[..., 0] *= 1.0 + 0.14 * key * (1.0 - smoothstep(62.0, 88.0, lab[..., 0]))
        lab[..., 2] += 2.5 * key
    return ops.from_lab_where(img, lab, w)


def _masked_blur(x: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian average of ``x`` over the pixels where ``weight`` is set (hair and
    background around the face don't leak into the face's averages)."""
    return ops.gaussian(x * weight, sigma) / np.maximum(ops.gaussian(weight, sigma), 1e-3)


def _red_lightness(img: np.ndarray) -> np.ndarray:
    """L*-like lightness of the red channel alone."""
    r = img[..., 0]
    lin = np.where(r <= 0.04045, r / 12.92, ((r + 0.055) / 1.055) ** 2.4)
    return (116.0 * np.cbrt(lin) - 16.0).astype(np.float32)


def _under_eye_band(face: Face, regions: FaceRegions) -> np.ndarray:
    iod, y = face.iod, face.y_axis
    band = np.zeros(regions.shape, np.float32)
    for lid in (F.RIGHT_LOWER_LID, F.LEFT_LOWER_LID):
        pts = face.pts(lid)
        band = np.maximum(band, fill_poly(regions.shape, np.vstack([pts + y * 0.02 * iod, pts[::-1] + y * 0.27 * iod])))
    return blur(band, 0.05 * iod)


def _cheek_zone(face: Face, shape: tuple[int, int]) -> np.ndarray:
    """Soft mask of the cheeks: below the eyes, beside the nose and the mouth."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    tip, x, y = face.pt(F.NOSE_TIP), face.x_axis, face.y_axis
    across = np.abs((xs - tip[0]) * x[0] + (ys - tip[1]) * x[1]) / face.iod
    down = ((xs - face.eye_mid[0]) * y[0] + (ys - face.eye_mid[1]) * y[1]) / face.iod
    return (smoothstep(0.2, 0.36, across) * smoothstep(0.15, 0.32, down) * (1.0 - smoothstep(0.95, 1.15, down))).astype(np.float32)


def relief(img: np.ndarray, original: np.ndarray, face: Face, regions: FaceRegions,
           amount: float, boost: float = 1.2, cheeks: float = 1.0) -> np.ndarray:
    """立体: give the face back its own light and shade.

    Brightening, whitening and evening the skin each lift the shadows a little more
    than the lights; together they flatten the face into a pale, even disc. Here the
    face's large-scale light and shade (the cheekbones against the hollows below them,
    the nose against its sides, the jaw, the eye sockets, the temples) is measured on
    the photo before retouching and put back, scaled to the new brightness and a little
    stronger. Blemish-sized detail is far below this scale and stays gone; the band
    under the eyes is left out so dark circles don't come back.
    """
    if amount <= 0.0:
        return img
    iod = face.iod
    w = regions.skin
    total = float(w.sum())
    if total < 50.0:
        return img
    lab = ops.to_lab(img)
    now = lab[..., 0]
    # Light and shade are read from the red channel, as retouchers do: shading darkens
    # every channel alike, but red blemishes (acne, marks, flushed patches) are nearly
    # invisible in red, so they are not mistaken for the face's shape and brought back.
    before, current = _red_lightness(original), _red_lightness(img)
    small, large = 0.08 * iod, 0.4 * iod
    shape_before = _masked_blur(before, w, small) - _masked_blur(before, w, large)
    shape_now = _masked_blur(current, w, small) - _masked_blur(current, w, large)
    # A shading ratio shows up larger in L* on brighter skin: scale by (L*+16).
    m_before = float((before * w).sum() / total)
    m_now = float((current * w).sum() / total)
    scale = float(np.clip((m_now + 16.0) / max(m_before + 16.0, 1.0), 0.8, 2.0))
    goal = np.clip(shape_before * scale * boost, -25.0, 25.0)
    zone = w * (1.0 - _under_eye_band(face, regions)) * float(np.clip(amount, 0.0, 1.5))
    if cheeks < 1.0:   # softer on the cheeks than on the nose, jaw and eye sockets
        zone = zone * (1.0 - (1.0 - cheeks) * _cheek_zone(face, regions.shape))
    lab[..., 0] = now + (goal - shape_now) * zone
    return ops.from_lab_where(img, lab, zone)


def brighten(img: np.ndarray, regions: FaceRegions, amount: float,
             tone_a: float = 0.0, tone_b: float = 0.0) -> np.ndarray:
    """Lift skin lightness with a log curve and nudge its hue (a: +pink, b: -yellow)."""
    if amount <= 0.0 and tone_a == 0.0 and tone_b == 0.0:
        return img
    mask = regions.skin_wide
    lab = ops.to_lab(img)
    # One complexion for the whole face. The color model drops whatever is not typical
    # skin (around the brows, the ears, a shaded jaw); left alone, those parts keep the
    # photo's old color and show as yellow-brown patches next to the newly fair skin.
    # So everything on the face that is not dark hair, an eye or the lips is included.
    face = regions.face
    total = float(regions.skin.sum())
    skin_l = float((lab[..., 0] * regions.skin).sum() / total) if total > 1.0 else 60.0
    not_hair = smoothstep(skin_l - 36.0, skin_l - 14.0, lab[..., 0])
    # A fringe is mostly hair with slivers of skin showing through: leave all of it alone.
    # Brows are a thin band, so the skin around them still counts as face.
    hairy = ops.gaussian((lab[..., 0] < skin_l - 24.0).astype(np.float32), 0.07 * face.iod)
    not_hair = not_hair * (1.0 - smoothstep(0.3, 0.55, hairy))
    keep_out = blur(np.maximum(regions.eyes, regions.lips_outer), 0.012 * face.iod)
    mask = np.maximum(mask, regions.face_light * not_hair * (1.0 - np.clip(keep_out * 1.5, 0.0, 1.0)))
    if total > 1.0:
        # 肤色统一: pull every part of the face toward the same hue, so redness, sallow
        # corners and colored shadows stop reading as separate patches.
        unify = 0.5 * mask * float(np.clip(amount / 0.5, 0.0, 1.0))
        for c in (1, 2):
            mean_c = float((lab[..., c] * regions.skin).sum() / total)
            lab[..., c] += (mean_c - lab[..., c]) * unify
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


# Landmarks around the cheekbone: (outer, under-eye, inner, inner-lower, outer-lower, inner-upper)
_CHEEK = {"right": (117, 118, 119, 101, 50, 100), "left": (346, 347, 348, 330, 280, 329)}
_NOSE_WING = {"right": (129, 64), "left": (358, 294)}
_EAR_FRONT = {"right": (93, 132), "left": (323, 361)}


def cheekbone_patch(face: Face, side: str) -> tuple[np.ndarray, float, float, float]:
    """Center, sigmas and angle of the lit patch on one cheekbone.

    Placed from landmarks rather than from the eye position, so it sits on the
    cheekbone itself and follows the face in three-quarter views: the near cheek is
    wide and its patch broad, the far cheek is foreshortened and its patch narrow.
    """
    outer, under_eye, inner, inner_low, outer_low, inner_up = (face.pt(i) for i in _CHEEK[side])
    center = (outer + under_eye + inner_low + outer_low) / 4.0     # the cheekbone's high point
    width = float(np.linalg.norm(outer - inner_up))
    height = float(np.linalg.norm(under_eye - outer_low))
    axis = inner_low - outer
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    if side == "left":                      # same direction convention on both cheeks
        angle += 180.0
    iod = face.iod
    return center, max(0.3 * width, 0.05 * iod), max(0.45 * height, 0.04 * iod), angle


def _bridge_line(face: Face, shape: tuple[int, int]) -> np.ndarray:
    """The upper and middle nose bridge (not the tip), as a soft band."""
    line = polyline(shape, face.pts([168, 6, 197, 195]), thickness=0.08 * face.iod)
    return blur(line, 0.03 * face.iod)


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
    # The photo's own shine along the nose bridge reads as a long thin stripe once it is
    # amplified; on a retouched face the brightest point of the nose is its tip.
    natural = natural * (1.0 - 0.75 * _bridge_line(face, shape))

    # 2) placed specular spots (tight cores)
    spots = np.zeros(shape, np.float32)
    bridge = (face.pt(6) + face.pt(197) + face.pt(195)) / 3.0
    gaussian_ellipse(shape, bridge, 0.02 * iod, 0.12 * iod, ang, out=spots, weight=0.25)
    gaussian_ellipse(shape, face.pt(F.NOSE_TIP) - y * 0.025 * iod, 0.045 * iod, 0.04 * iod, ang, out=spots, weight=1.0)
    gaussian_ellipse(shape, face.brow_mid - y * 0.36 * iod, 0.13 * iod, 0.05 * iod, ang, out=spots, weight=0.6)
    gaussian_ellipse(shape, face.pt(152) - y * 0.17 * iod, 0.06 * iod, 0.03 * iod, ang, out=spots, weight=0.5)
    gaussian_ellipse(shape, face.pt(0) - y * 0.035 * iod, 0.035 * iod, 0.014 * iod, ang, out=spots, weight=0.6)
    # Shine needs light to reflect: spots stay faint on skin that is in shadow.
    spots = np.clip(spots, 0.0, 1.0) ** 1.5 * skin * smoothstep(0.3, 0.75, lum)
    # The cheekbone gets a broad, soft sheen (a lit patch), not a thin streak.
    sheen = np.zeros(shape, np.float32)
    for side in ("right", "left"):
        c, sx, sy, a = cheekbone_patch(face, side)
        gaussian_ellipse(shape, c, 0.75 * sx, 0.75 * sy, a, out=sheen)
    sheen = np.clip(sheen, 0.0, 1.0) * skin * smoothstep(0.3, 0.75, lum)

    shine = natural * (1.8 * amount) + spots * (0.42 * amount) + sheen * (0.2 * amount)
    tint = np.array([1.0, 0.985, 0.97], np.float32)
    out = ops.screen(img, np.clip(shine, 0.0, 0.6)[..., None] * tint)

    # 3) glossy midtone contrast on skin
    mid = float((lum * skin).sum() / max(float(skin.sum()), 1e-6))
    out = out + ((out - mid) * (0.10 * amount)) * skin[..., None]
    return np.clip(out, 0.0, 1.0)


def sculpt(img: np.ndarray, face: Face, regions: FaceRegions, amount: float) -> np.ndarray:
    """修容高光: shape the face with light instead of brightening it evenly.

    * The nose: the middle of the bridge is toned down and the brightest point is a
      small soft patch on the tip; the nose wings keep a light, slightly pink shadow.
    * The cheekbones get a broad lit patch, and the skin darkens gradually from there
      toward the ears; the jaw turn keeps a soft shadow.
    """
    if amount <= 0.0:
        return img
    iod, shape, ang, y = face.iod, regions.shape, face.angle_deg, face.y_axis
    light = np.zeros(shape, np.float32)
    cheek = np.zeros(shape, np.float32)
    shade = np.zeros(shape, np.float32)
    wing = np.zeros(shape, np.float32)
    tip = face.pt(F.NOSE_TIP)
    gaussian_ellipse(shape, tip - y * 0.025 * iod, 0.05 * iod, 0.045 * iod, ang, out=light, weight=0.85)
    gaussian_ellipse(shape, (face.pt(6) + face.pt(195)) / 2.0, 0.025 * iod, 0.14 * iod, ang, out=light, weight=0.12)
    for side in ("right", "left"):
        out = face.outward(side)
        c, sx, sy, a = cheekbone_patch(face, side)
        gaussian_ellipse(shape, c, 0.85 * sx, 0.85 * sy, a, out=cheek, weight=0.6)    # soft edges
        gaussian_ellipse(shape, c, 0.5 * sx, 0.5 * sy, a, out=cheek, weight=0.45)     # on its high point
        p, q = (face.pt(i) for i in _NOSE_WING[side])
        gaussian_ellipse(shape, (p + q) / 2.0 + out * 0.015 * iod, 0.04 * iod, 0.065 * iod, ang, out=wing)
        # the hollow under the cheekbone: a soft shadow from the front of the ear toward
        # the corner of the mouth, fading out halfway (the main line of contouring)
        start = face.pt(_EAR_FRONT[side][0])
        v = face.pt(F.MOUTH_RIGHT if side == "right" else F.MOUTH_LEFT) - start
        length = max(float(np.linalg.norm(v)), 1e-6)
        gaussian_ellipse(shape, start + 0.3 * v, 0.22 * length, 0.055 * iod,
                         float(np.degrees(np.arctan2(v[1], v[0]))), out=shade, weight=0.6)
    tip_dir = tip
    for index in (132, 361, 58, 288, 172, 397, 136, 365, 150, 379):   # in front of the ears, the jaw turn
        jaw = face.pt(index)
        inward = tip_dir - jaw
        jaw = jaw + inward / max(float(np.linalg.norm(inward)), 1e-6) * 0.06 * iod
        gaussian_ellipse(shape, jaw, 0.12 * iod, 0.12 * iod, ang, out=shade, weight=0.7)
    # gradual falloff from the cheekbones toward the sides of the face, below the eyes
    inside = (regions.face_poly > 0.5).astype(np.uint8)
    depth = cv2.distanceTransform(inside, cv2.DIST_L2, 3) / max(iod, 1.0)
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    below = ((xs - face.eye_mid[0]) * y[0] + (ys - face.eye_mid[1]) * y[1]) / max(iod, 1.0)
    side_fall = (1.0 - smoothstep(0.0, 0.42, depth)) ** 1.5 * smoothstep(-0.05, 0.4, below)
    shade = np.maximum(np.clip(shade, 0.0, 1.0), side_fall)
    # From each cheekbone the light drops off steadily toward the front of the ear, so
    # the lit patch reads as the high point of the cheek rather than a stripe on it.
    ear_fall = np.zeros(shape, np.float32)
    for side in ("right", "left"):
        c, _, _, _ = cheekbone_patch(face, side)
        ear = (face.pt(_EAR_FRONT[side][0]) + face.pt(_EAR_FRONT[side][1])) / 2.0
        v = ear - c
        reach = max(float(np.linalg.norm(v)), 1e-6)
        u = v / reach
        t = ((xs - c[0]) * u[0] + (ys - c[1]) * u[1]) / reach
        across = ((xs - c[0]) * -u[1] + (ys - c[1]) * u[0]) / max(iod, 1.0)
        ear_fall = np.maximum(ear_fall, smoothstep(0.08, 0.6, t) * np.exp(-0.5 * (across / 0.4) ** 2))
    ear_fall = ear_fall * smoothstep(-0.05, 0.3, below)
    skin = regions.skin
    light = np.clip(light, 0.0, 1.0) * skin
    cheek = np.clip(cheek, 0.0, 1.0) * skin * (1.0 - light)
    lit = np.maximum(light, cheek)
    wing = np.clip(wing, 0.0, 1.0) * skin * (1.0 - lit)
    shade = shade * skin * (1.0 - lit)
    ear_fall = ear_fall * skin * (1.0 - lit)
    lab = ops.to_lab(img)
    lightness = lab[..., 0]
    # tone down the long thin highlight on the middle of the bridge (keep the tip)
    bridge = _bridge_line(face, shape) * skin * (1.0 - np.clip(lit * 2.0, 0.0, 1.0))
    excess = np.clip(lightness - ops.gaussian(lightness, 0.08 * iod), 0.0, None)
    lightness = lightness - excess * bridge * min(1.0, 0.8 * amount)
    # The cheekbones get a restrained light: on a face that is not very bright yet, a
    # strong one turns into two pale patches. Depth comes from the shadows instead.
    lab[..., 0] = lightness + ((100.0 - lightness) * (0.32 * light + 0.17 * cheek)
                               - lightness * (0.13 * shade + 0.12 * ear_fall + 0.09 * wing)) * amount
    lab[..., 1] += 4.0 * wing * amount            # shadows on skin are a little pink, not grey
    return ops.from_lab_where(img, lab, np.maximum.reduce([light, cheek, shade, ear_fall, wing, bridge]))


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


def hair_depth(img: np.ndarray, regions: FaceRegions, amount: float) -> np.ndarray:
    """Deep hair with light on the strands instead of a flat grey-blue haze.

    Everything dark around the face that is not skin (hair, roots under a hat brim,
    the brim itself) is pushed down, while the highlights that already run along
    the strands are kept and strengthened, so the hair reads as black and glossy.
    """
    if amount <= 0.0:
        return img
    iod = regions.iod
    lum = ops.luminance(img)
    zone = 1.0 - smoothstep(0.3, 0.5, ops.gaussian(lum, 0.05 * iod))          # broadly dark areas
    zone = zone * (1.0 - np.clip(regions.skin * 2.5, 0.0, 1.0)) * (1.0 - np.clip(regions.features * 1.5, 0.0, 1.0))
    # Only what can be hair: not skin-colored (ears, a shaded jaw), not itself light,
    # and close to neutral, so a dark colored wall or dress keeps its color.
    lab = ops.to_lab(img)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    chroma = chroma * np.where(lab[..., 2] < -2.0, 0.4, 1.0)      # a blue cast on black hair is haze, not color
    zone = zone * (1.0 - np.clip(regions.skin_probability_loose * 2.0, 0.0, 1.0))
    zone = zone * (1.0 - smoothstep(0.45, 0.65, lum)) * (1.0 - smoothstep(14.0, 26.0, chroma))
    zone = ops.gaussian(zone, 0.01 * iod)
    zone = zone * float(np.clip(amount, 0.0, 1.0))
    base = ops.gaussian(lum, 0.045 * iod)
    strands = np.clip(lum - base, 0.0, 0.2)                                   # light along the hair
    gain = 0.55 + 0.45 * smoothstep(0.18, 0.55, lum)                              # darks down, lights stay
    out = img * (1.0 + (gain - 1.0) * zone)[..., None]
    out = out + (strands * 1.5 * zone)[..., None] * np.array([0.96, 0.98, 1.0], np.float32)
    grey = ops.luminance(out)[..., None]
    out = out + (grey - out) * (0.3 * zone)[..., None]                       # take the blue out of the haze
    return np.clip(out, 0.0, 1.0)
