"""The local retouching pipeline: detect → skin → makeup → reshape → color grade."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from . import color, makeup, reshape, skin
from . import face as F
from .config import Settings, get_settings
from .face import Face, get_analyzer
from .imageio import decode_image, encode_jpeg, to_float, to_uint8
from .masks import FaceRegions
from .presets import LIP_SHADES, Preset, get_preset

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
    makeup: float
    filter: float
    lip_rgb: tuple[int, int, int] | None


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
    return Params(
        preset=preset,
        smooth=pick(adj.smooth, preset.smooth),
        whiten=whiten,
        tone_scale=tone_scale,
        slim=pick(adj.slim, preset.slim),
        eyes=pick(adj.eyes, preset.eyes),
        makeup=pick(adj.makeup, preset.default_makeup),
        filter=pick(adj.filter, preset.default_filter),
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


def _beautify_face(img: np.ndarray, face: Face, p: Params) -> np.ndarray:
    pr, m, iod = p.preset, p.makeup, face.iod
    regions = FaceRegions(img, face)
    img = skin.remove_spots(img, regions, p.smooth)
    img = skin.smooth(img, regions, p.smooth, pr.texture)
    img = skin.brighten(img, regions, p.whiten, pr.tone_a * p.tone_scale, pr.tone_b * p.tone_scale)
    img = skin.matte(img, regions, pr.matte * m)
    img = skin.glass_glow(img, face, regions, pr.glow * m)
    img = makeup.blush(img, face, regions, pr.blush_rgb, pr.blush * m, pr.blush_height, pr.blush_spread)

    eyes = F.RIGHT_EYE + F.LEFT_EYE
    if p.lip_rgb is not None:
        img = _local(img, face, F.LIPS_OUTER, 0.15 * iod, lambda s, f, r: makeup.lips(
            s, f, r, p.lip_rgb, pr.lip_intensity * m, pr.lip_gradient, pr.lip_gloss * m))
    img = _local(img, face, eyes, 0.35 * iod, lambda s, f, r: makeup.aegyo_sal(s, f, r, pr.aegyo * m))
    img = _local(img, face, eyes, 0.08 * iod, lambda s, f, r: makeup.brighten_eyes(s, r, pr.eye_brighten * m))
    img = _local(img, face, F.LIPS_INNER, 0.06 * iod, lambda s, f, r: makeup.whiten_teeth(
        s, f, r, pr.teeth * min(1.0, m + 0.3)))
    return img


def process(rgb: np.ndarray, preset_id: str | None, adj: Adjustments | None,
            faces: list[Face]) -> np.ndarray:
    """Run the full pipeline on an RGB uint8 image with pre-detected faces."""
    p = resolve(get_preset(preset_id), adj or Adjustments())
    img = to_float(rgb)
    usable = [f for f in faces if f.iod >= MIN_IOD_PX]

    for face in usable:  # appearance edits, each inside its own crop
        x0, y0, x1, y1 = face.roi(img.shape)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        img[y0:y1, x0:x1] = _beautify_face(crop, face.shifted(-x0, -y0), p)

    for face in usable:  # geometry last, so makeup moves together with the face
        img = reshape.slim_face(img, face, p.slim)
        img = reshape.enlarge_eyes(img, face, p.eyes)

    img = color.apply_grade(img, p.preset.grade, p.filter)
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
