"""Labels for AI-generated images.

China's《人工智能生成合成内容标识办法》(in force since 2025-09-01) asks AI
services to add both an explicit label (visible text/badge) and an implicit one
(metadata). This module adds a small corner badge plus an "AIGC" JSON record in
the JPEG comment and EXIF. Check the official standard before a real launch.
"""

from __future__ import annotations

import json
import os
import uuid
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


@lru_cache(maxsize=8)
def _font(size: int) -> tuple[ImageFont.ImageFont, bool]:
    """Return (font, supports_chinese)."""
    paths = [os.getenv("AI_LABEL_FONT", "")] + FONT_CANDIDATES
    for path in paths:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size), True
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size), False
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default(), False


def add_ai_badge(rgb: np.ndarray, text_zh: str = "AI生成") -> np.ndarray:
    """Draw a small translucent rounded badge in the bottom-right corner."""
    img = Image.fromarray(rgb).convert("RGBA")
    w, h = img.size
    size = max(12, int(min(w, h) * 0.028))
    font, has_cjk = _font(size)
    text = text_zh if has_cjk else "AI"
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad_x, pad_y = int(size * 0.55), int(size * 0.3)
    bw, bh = right - left + 2 * pad_x, bottom - top + 2 * pad_y
    margin = int(size * 0.8)
    x0, y0 = w - bw - margin, h - bh - margin
    draw.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=bh // 2, fill=(0, 0, 0, 96))
    draw.text((x0 + pad_x - left, y0 + pad_y - top), text, font=font, fill=(255, 255, 255, 235))
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))


def aigc_metadata(provider: str, model: str) -> tuple[str, bytes]:
    """Return (jpeg_comment, exif_bytes) carrying an implicit AIGC label."""
    record = {"AIGC": {"Label": "1", "ContentProducer": "SeoulGlow", "ProduceID": uuid.uuid4().hex,
                       "Provider": provider, "Model": model}}
    comment = json.dumps(record, ensure_ascii=False)
    exif = Image.Exif()
    exif[0x010E] = "AI-generated image (SeoulGlow)"   # ImageDescription
    exif[0x0131] = "SeoulGlow"                         # Software
    return comment, exif.tobytes()
