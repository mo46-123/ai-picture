"""The local retouching pipeline: detect → skin → makeup → reshape → color grade."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import cv2
import numpy as np

from . import color, makeup, ops, reshape, skin
from . import face as F
from .config import Settings, get_settings
from .face import Face, get_analyzer
from .imageio import decode_image, encode_jpeg, to_float, to_uint8
from .masks import FaceRegions, blur, dilate, fill_poly, gaussian_ellipse, smoothstep
from .presets import LIP_SHADES, Preset, get_preset

WORK_IOD = 200.0   # faces larger than this (eye distance, px) are retouched at reduced size
MIN_IOD_PX = 16.0  # faces smaller than this are too tiny for landmark-based edits

_UNSET: Any = object()


def _clamp01(value: Any, default: float | None) -> float | None:
    if value is None or value == "":
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return min(1.0, max(0.0, v))


@dataclass
class Adjustments:
    """User-facing controls (all 0..1). ``None`` means "use the preset default"."""

    intensity: float = 1.0
    smooth: float | None = None
    whiten: float | None = None
    slim: float | None = None
    eyes: float | None = None
    nose: float | None = None
    makeup: float | None = None
    filter: float | None = None
    lip: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Adjustments":
        lip = data.get("lip")
        lip = str(lip) if lip not in (None, "") else None
        if lip is not None and lip != "none" and lip not in LIP_SHADES:
            lip = None
        return cls(
            intensity=_clamp01(data.get("intensity"), 1.0),
            smooth=_clamp01(data.get("smooth"), None),
            whiten=_clamp01(data.get("whiten"), None),
            slim=_clamp01(data.get("slim"), None),
            eyes=_clamp01(data.get("eyes"), None),
            nose=_clamp01(data.get("nose"), None),
            makeup=_clamp01(data.get("makeup"), None),
            filter=_clamp01(data.get("filter"), None),
            lip=lip,
        )


@dataclass
class Params:
    """Fully resolved effect strengths for one run."""

    preset: Preset
    smooth: float
    whiten: float
    tone_scale: float
    slim: float
    eyes: float
    nose: float
    makeup: float
    filter: float
    cleanup: float          # dark circles / folds follow the 磨皮 slider
    lip_rgb: tuple[int, int, int] | None
    dark_scene: float = 0.0  # 0 = normal light, 1 = photo taken in a dark room (set by process)


def resolve(preset: Preset, adj: Adjustments) -> Params:
    k = adj.intensity

    def pick(user: float | None, default: float) -> float:
        return (default if user is None else user) * k

    whiten = pick(adj.whiten, preset.whiten)
    tone_scale = min(1.5, whiten / preset.whiten) if preset.whiten > 0 else k
    if adj.lip == "none":
        lip_rgb = None
    else:
        lip_rgb = LIP_SHADES[adj.lip or preset.lip].rgb
    smooth = pick(adj.smooth, preset.smooth)
    return Params(
        preset=preset,
        smooth=smooth,
        whiten=whiten,
        tone_scale=tone_scale,
        slim=pick(adj.slim, preset.slim),
        eyes=pick(adj.eyes, preset.eyes),
        nose=pick(adj.nose, preset.nose),
        makeup=pick(adj.makeup, preset.default_makeup),
        filter=pick(adj.filter, preset.default_filter),
        cleanup=min(1.5, smooth / preset.smooth) if preset.smooth > 0 else k,
        lip_rgb=lip_rgb,
    )


def _local(img: np.ndarray, face: Face, indices: list[int], pad: float, fn) -> np.ndarray:
    """Run ``fn`` on a small box around some landmarks (lips, eyes...) instead of the
    whole face crop. Purely geometric masks are cheap to rebuild for the sub-box."""
    pts = face.pts(indices)
    h, w = img.shape[:2]
    x0, y0 = int(max(0, pts[:, 0].min() - pad)), int(max(0, pts[:, 1].min() - pad))
    x1, y1 = int(min(w, pts[:, 0].max() + pad + 1)), int(min(h, pts[:, 1].max() + pad + 1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return img
    sub = np.ascontiguousarray(img[y0:y1, x0:x1])
    sub_face = face.shifted(-x0, -y0)
    img[y0:y1, x0:x1] = fn(sub, sub_face, FaceRegions(sub, sub_face))
    return img


NOMINAL_SKIN = np.array(skin.NOMINAL_SKIN, np.float32)   # Lab of evenly lit light skin


def _skin_light(img: np.ndarray, regions: FaceRegions) -> np.ndarray:
    """Mean Lab of the face's skin: tells us how bright and what color the light is."""
    w = regions.skin
    total = float(w.sum())
    if total < 50.0:
        return NOMINAL_SKIN
    lab = ops.to_lab(img)
    return np.array([float((lab[..., k] * w).sum() / total) for k in range(3)], np.float32)


def _adapt(rgb: tuple[int, int, int], light: np.ndarray) -> np.ndarray:
    """Re-light a makeup shade for this photo.

    Shades are designed for an evenly lit face. In shade or under colored light the
    same absolute color looks painted on (orange blush on a blue-grey face), so the
    shade is expressed as an offset from the skin tone and dimmed with the light.
    """
    target = ops.rgb255_to_lab(rgb).astype(np.float32)
    k = float(np.clip(light[0] / NOMINAL_SKIN[0], 0.45, 1.1))
    base = NOMINAL_SKIN[1:] + 0.7 * (light[1:] - NOMINAL_SKIN[1:])
    return np.array([target[0] * k, base[0] + (target[1] - NOMINAL_SKIN[1]) * k,
                     base[1] + (target[2] - NOMINAL_SKIN[2]) * k], np.float32)


DIM_FACE_L = 56.0     # faces darker than this (L*) are underexposed
LIFT_TARGET_L = 54.0  # the whole-photo lift brings the face up to about here; relight does the rest
MAX_LIFT = 2.4        # at most ~1.3 EV


def _to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055).astype(np.float32)


def _face_lightness(img: np.ndarray, face: Face) -> float:
    """Typical L* of a face's skin, sampled on the cheekbones and the forehead."""
    x0, y0, x1, y1 = face.roi(img.shape, margin=0.05)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return float(NOMINAL_SKIN[0])
    lightness = ops.to_lab(np.ascontiguousarray(img[y0:y1, x0:x1]))[..., 0]
    f = face.shifted(-x0, -y0)
    weight = np.zeros(lightness.shape, np.float32)
    for side in ("right", "left"):
        c, sx, sy, a = skin.cheekbone_patch(f, side)
        gaussian_ellipse(lightness.shape, c, sx, sy, a, out=weight)
    gaussian_ellipse(lightness.shape, f.brow_mid - f.y_axis * 0.3 * f.iod, 0.15 * f.iod, 0.07 * f.iod,
                     f.angle_deg, out=weight)
    sample = lightness[weight > 0.5]
    return float(np.median(sample)) if sample.size > 20 else float(np.median(lightness))


def _lift_dark_scene(img: np.ndarray, faces: list[Face], strength: float) -> tuple[np.ndarray, float]:
    """补光 for photos taken in the dark.

    A retoucher brightens a dark photo as a whole and then fine-tunes the face. Lifting
    only the face leaves a bright oval floating in a dark picture, which reads as a
    mask. So when the face and the scene are both dark, most of the lift goes to the
    whole photo, in linear light like a longer exposure (blacks stay black, highlights
    roll off softly); the face relight only finishes the job. A backlit face in a
    bright scene is left to the local relight.

    Returns the image and how dark the scene is (0 = normal light, 1 = a dark room).
    """
    if not faces or strength <= 0.0:
        return img, 0.0
    face_l = _face_lightness(img, max(faces, key=lambda f: f.iod))
    if face_l >= DIM_FACE_L:
        return img, 0.0
    h, w = img.shape[:2]
    small = cv2.resize(img, (256, max(1, round(256 * h / w))), interpolation=cv2.INTER_AREA)
    bright = float(np.percentile(ops.to_lab(small)[..., 0], 95))
    share = 1.0 - float(smoothstep(60.0, 80.0, np.float32(bright)))
    dark_scene = (1.0 - float(smoothstep(40.0, DIM_FACE_L, np.float32(face_l)))) * share

    def lum(l: float) -> float:
        return ((l + 16.0) / 116.0) ** 3

    gain = min(MAX_LIFT, (lum(LIFT_TARGET_L) / lum(face_l)) ** (share * float(np.clip(strength, 0.0, 1.0))))
    if gain < 1.03:
        return img, dark_scene
    lin = _to_linear(img) * gain
    knee = 0.7                                   # soft shoulder: lamps and windows don't clip
    lin = np.where(lin > knee, knee + (1.0 - knee) * (1.0 - np.exp(-(lin - knee) / (1.0 - knee))), lin)
    return _to_srgb(lin), dark_scene


def _beautify_face(img: np.ndarray, face: Face, p: Params) -> np.ndarray:
    pr, iod = p.preset, face.iod
    m = p.makeup * face.frontalness   # no makeup on faces turned away from the camera
    regions = FaceRegions(img, face)
    original = img
    relit = skin.relight(img, regions, min(1.0, p.whiten / max(pr.whiten, 1e-6)),
                         target_l=float(NOMINAL_SKIN[0]) - 8.0 * p.dark_scene)   # a dark photo stays a little darker
    if relit is not img:   # the skin model must be rebuilt on the corrected colors
        img, regions = relit, FaceRegions(relit, face)
    img = skin.remove_spots(img, regions, p.smooth)
    img = skin.smooth(img, regions, p.smooth, pr.texture)
    img = skin.even_out(img, regions, p.smooth)
    img = skin.under_eye(img, face, regions, pr.dark_circles * p.cleanup * face.frontalness)
    img = skin.soften_folds(img, face, regions, pr.folds * p.cleanup * face.frontalness)
    img = skin.brighten(img, regions, p.whiten, pr.tone_a * p.tone_scale, pr.tone_b * p.tone_scale)
    img = skin.relief(img, original, face, regions, pr.relief, boost=1.35)
    light = _skin_light(img, regions)                     # measure again, after brightening
    dim = float(np.clip(light[0] / 62.0, 0.35, 1.0))     # darkening makeup is eased off on dim faces
    blush_c, brow_c, shadow_c = _adapt(pr.blush_rgb, light), _adapt(pr.brow_rgb, light), _adapt(pr.shadow_rgb, light)
    lip_c = _adapt(p.lip_rgb, light) if p.lip_rgb is not None else None
    img = skin.matte(img, regions, pr.matte * m)
    img = skin.glass_glow(img, face, regions, pr.glow * m, reference=original)
    img = skin.sculpt(img, face, regions, pr.sculpt * m)
    img = makeup.blush(img, face, regions, blush_c, pr.blush * m, pr.blush_height, pr.blush_spread,
                       nose=pr.blush_nose, outward=pr.blush_out)

    eyes = F.RIGHT_EYE + F.LEFT_EYE
    brows = F.RIGHT_BROW + F.LEFT_BROW
    img = _local(img, face, brows, 0.12 * iod, lambda s, f, r: makeup.brows(
        s, f, r, brow_c, pr.brow * m * dim, shape=pr.brow_shape))
    img = _local(img, face, eyes + brows, 0.15 * iod, lambda s, f, r: makeup.eyeshadow(
        s, f, r, shadow_c, pr.shadow * m * dim, outer=pr.shadow_outer))
    img = _local(img, face, eyes, 0.12 * iod, lambda s, f, r: makeup.eyeliner(
        s, f, r, pr.liner * m * dim, tail=pr.liner_tail, drop=pr.liner_drop))
    img = _local(img, face, eyes, 0.15 * iod, lambda s, f, r: makeup.lashes(s, f, r, pr.lash * m * dim))
    if p.lip_rgb is not None:
        img = _local(img, face, F.LIPS_OUTER, 0.15 * iod, lambda s, f, r: makeup.lips(
            s, f, r, lip_c, pr.lip_intensity * m, pr.lip_gradient, pr.lip_gloss * m))
    img = _local(img, face, eyes, 0.35 * iod, lambda s, f, r: makeup.aegyo_sal(s, f, r, pr.aegyo * m))
    img = _local(img, face, eyes, 0.08 * iod, lambda s, f, r: makeup.brighten_eyes(
        s, r, pr.eye_brighten * m, face=f, catchlight=pr.lens <= 0.0))   # the lens step handles catchlights
    img = _local(img, face, eyes, 0.08 * iod, lambda s, f, r: makeup.iris_lens(s, f, r, pr.lens * m, skin_l=float(light[0])))
    # Hair, a hat brim, a dark collar: whatever was dark and is not skin stays dark.
    # Corrections aimed at the face spill onto them as a grey haze otherwise.
    # Not the features, though: in a dim photo the lids, lash lines, lips and nostrils are
    # dark too, and pulling them back would leave dark rings in a brightened face.
    was = ops.luminance(original)
    eye_zone = dilate(np.maximum(fill_poly(regions.shape, face.pts(F.RIGHT_EYE)),
                                 fill_poly(regions.shape, face.pts(F.LEFT_EYE))), 0.14 * iod)
    nose_zone = fill_poly(regions.shape, cv2.convexHull(face.pts(F.NOSE_LOWER).astype(np.float32)).reshape(-1, 2))
    features = blur(np.maximum.reduce([eye_zone, dilate(regions.lips_outer, 0.06 * iod), dilate(nose_zone, 0.03 * iod)]),
                    0.03 * iod)
    dark = (1.0 - smoothstep(0.1, 0.28, was)) * (1.0 - np.clip(regions.skin * 2.0, 0.0, 1.0)) * (1.0 - features)
    img = img + (np.minimum(original, img) - img) * dark[..., None]
    img = skin.hair_depth(img, regions, min(1.0, p.filter / max(pr.default_filter, 1e-6)) * pr.hair
                          * (1.0 - 0.6 * p.dark_scene))
    img = _local(img, face, F.LIPS_INNER, 0.06 * iod, lambda s, f, r: makeup.whiten_teeth(
        s, f, r, pr.teeth * min(1.0, m + 0.3)))
    return img


def process(rgb: np.ndarray, preset_id: str | None, adj: Adjustments | None,
            faces: list[Face]) -> np.ndarray:
    """Run the full pipeline on an RGB uint8 image with pre-detected faces."""
    p = resolve(get_preset(preset_id), adj or Adjustments())
    img = to_float(rgb)
    usable = [f for f in faces if f.iod >= MIN_IOD_PX]
    img, dark_scene = _lift_dark_scene(img, usable, min(1.0, p.whiten / max(p.preset.whiten, 1e-6)))
    p = replace(p, dark_scene=dark_scene)

    for face in usable:  # appearance edits, each inside its own crop
        x0, y0, x1, y1 = face.roi(img.shape)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        local = face.shifted(-x0, -y0)
        if face.iod > WORK_IOD * 1.15:
            # Big close-up: compute the retouch on a smaller copy and add the *change*
            # back at full size. The photo's own fine detail is untouched, and the
            # cost no longer grows with the size of the face.
            scale = WORK_IOD / face.iod
            small = cv2.resize(crop, (max(8, round(crop.shape[1] * scale)), max(8, round(crop.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)
            sx, sy = small.shape[1] / crop.shape[1], small.shape[0] / crop.shape[0]
            delta = _beautify_face(small.copy(), Face(local.landmarks * np.array([sx, sy])), p) - small
            delta = cv2.resize(delta, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
            img[y0:y1, x0:x1] = np.clip(crop + delta, 0.0, 1.0)
        else:
            img[y0:y1, x0:x1] = _beautify_face(crop, local, p)

    for face in usable:  # geometry last, so makeup moves together with the face
        k = face.frontalness
        img = reshape.slim_face(img, face, p.slim * k)
        img = reshape.slim_nose(img, face, p.nose * k)
        img = reshape.enlarge_eyes(img, face, p.eyes * k)
        img = reshape.enlarge_iris(img, face, p.eyes * k)
        img = reshape.plump_lips(img, face, p.preset.lip_plump * min(1.0, p.makeup / max(p.preset.default_makeup, 1e-6)) * k)

    grade = p.preset.grade
    if p.dark_scene > 0.0:
        # In a dark photo the background is already black; deepening the shadows much
        # more would cut the brightened face out of it.
        grade = replace(grade, shadows=grade.shadows * (1.0 - 0.4 * p.dark_scene) if grade.shadows < 0 else grade.shadows)
    img = color.apply_grade(img, grade, p.filter)
    # Retouching and the soft grade cost a little crispness; give it back. Faces get a
    # second, finer pass (feathered, so no visible box) to make eyes, brows and lips crisp.
    h, w = img.shape[:2]
    img = ops.unsharp(img, max(0.8, 0.0011 * max(h, w)), 0.35 * p.filter)
    for face in usable:
        x0, y0, x1, y1 = face.roi(img.shape, margin=0.2)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        fx0, fy0, fx1, fy1 = face.bbox
        feather = gaussian_ellipse(crop.shape[:2], np.array([(fx0 + fx1) / 2 - x0, (fy0 + fy1) / 2 - y0]),
                                   0.45 * (fx1 - fx0), 0.45 * (fy1 - fy0), face.angle_deg)
        sharp = ops.unsharp(crop, max(0.7, 0.012 * face.iod), 0.3 * min(1.0, face.iod / 140.0))
        img[y0:y1, x0:x1] = crop + (sharp - crop) * np.clip(feather * 1.4, 0.0, 1.0)[..., None]
    return to_uint8(img)


@dataclass
class RetouchResult:
    jpeg: bytes
    width: int
    height: int
    faces: int
    elapsed_ms: int
    notice: str | None = None
    extra: dict = field(default_factory=dict)


def retouch_array(rgb: np.ndarray, preset_id: str | None = None, adj: Adjustments | None = None,
                  settings: Settings | None = None, analyzer: Any = _UNSET) -> tuple[np.ndarray, int, str | None]:
    """Detect faces and run the pipeline. Returns (image, face_count, notice_for_user)."""
    if analyzer is _UNSET:
        analyzer = get_analyzer(settings or get_settings())
    faces = analyzer.detect(rgb) if analyzer is not None else []
    notice = None
    if analyzer is None:
        notice = "人脸模型还没下载，这次只加了韩系滤镜（运行 scripts/download_models.py 即可解锁美颜）"
    elif not faces:
        notice = "没有检测到人脸，这次只加了韩系滤镜"
    elif all(f.iod < MIN_IOD_PX for f in faces):
        notice = "人脸太小啦，这次只加了韩系滤镜；换张近一点的照片效果更好"
    return process(rgb, preset_id, adj, faces), len(faces), notice


def retouch_bytes(data: bytes, preset_id: str | None = None, adj: Adjustments | None = None,
                  settings: Settings | None = None, analyzer: Any = _UNSET,
                  max_side: int | None = None) -> RetouchResult:
    settings = settings or get_settings()
    start = time.perf_counter()
    side = min(max_side or settings.max_side, settings.max_side)
    rgb = decode_image(data, side)
    out, faces, notice = retouch_array(rgb, preset_id, adj, settings, analyzer)
    jpeg = encode_jpeg(out, settings.jpeg_quality)
    elapsed = int((time.perf_counter() - start) * 1000)
    return RetouchResult(jpeg=jpeg, width=out.shape[1], height=out.shape[0],
                         faces=faces, elapsed_ms=elapsed, notice=notice)
