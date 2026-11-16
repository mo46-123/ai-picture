"""High-level "AI 韩系写真" flow with automatic fallback to the local pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..config import Settings, get_settings
from ..engine import _UNSET, Adjustments, retouch_array
from ..imageio import InvalidImageError, decode_image, encode_jpeg
from .base import ProviderError
from .label import add_ai_badge, aigc_metadata
from .prompts import build_prompt
from .providers import get_provider

UPLOAD_SIDE = 1536  # big enough for the models, small enough to upload fast


@dataclass
class AIResult:
    jpeg: bytes
    width: int
    height: int
    provider: dict | None
    fallback: bool
    notice: str | None
    elapsed_ms: int


def _for_upload(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[:2]
    scale = UPLOAD_SIDE / max(h, w)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return encode_jpeg(rgb, 90)


def ai_portrait(data: bytes, style: str, provider_name: str | None = None,
                settings: Settings | None = None, analyzer: Any = _UNSET) -> AIResult:
    settings = settings or get_settings()
    start = time.perf_counter()
    rgb = decode_image(data, settings.max_side)

    def local(notice: str) -> AIResult:
        out, _, extra = retouch_array(rgb, style, Adjustments(), settings, analyzer)
        msg = notice if not extra else f"{notice}；{extra}"
        return AIResult(encode_jpeg(out, settings.jpeg_quality), out.shape[1], out.shape[0],
                        None, True, msg, int((time.perf_counter() - start) * 1000))

    provider = get_provider(provider_name or settings.ai_provider)
    if provider is None:
        return local("还没有配置大模型 API Key，已自动使用本地韩系精修（配置方法见 README）")

    prompt, negative = build_prompt(style, "en" if provider.name == "gemini" else "zh")
    try:
        result = provider.edit(_for_upload(rgb), prompt, negative)
        out = decode_image(result, settings.max_side)
    except (ProviderError, InvalidImageError) as exc:
        return local(f"AI 写真暂时不可用（{exc}），已为你生成本地韩系精修")

    comment = exif = None
    if settings.ai_label:
        out = add_ai_badge(out)
        comment, exif = aigc_metadata(provider.name, provider.model)
    jpeg = encode_jpeg(out, settings.jpeg_quality, comment=comment, exif=exif)
    return AIResult(jpeg, out.shape[1], out.shape[0], provider.describe(), False, None,
                    int((time.perf_counter() - start) * 1000))
