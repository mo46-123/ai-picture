"""Decoding / encoding helpers shared by the CLI, the web API and the AI providers."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps

try:  # Optional: lets the server accept iPhone HEIC uploads if pillow-heif is installed.
    from pillow_heif import register_heif_opener  # type: ignore

    register_heif_opener()
except Exception:  # pragma: no cover - optional dependency
    pass


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a decodable image."""


def decode_image(data: bytes, max_side: int = 2048) -> np.ndarray:
    """Bytes → RGB uint8 array, with EXIF rotation applied and the long side capped."""
    if not data:
        raise InvalidImageError("图片是空的")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, rgba)
        img = img.convert("RGB")
    except Exception as exc:  # PIL raises many different exception types
        raise InvalidImageError("无法识别这张图片，请上传 JPG / PNG / WebP 格式") from exc

    w, h = img.size
    if min(w, h) < 32:
        raise InvalidImageError("图片太小啦，换一张清晰一点的吧")
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8).copy()


def to_float(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def encode_jpeg(rgb: np.ndarray, quality: int = 92, comment: str | None = None, exif: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    kwargs: dict = {"quality": quality, "optimize": True}
    if comment:
        kwargs["comment"] = comment.encode("utf-8")
    if exif:
        kwargs["exif"] = exif
    Image.fromarray(rgb).save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


def to_data_url(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
