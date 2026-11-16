"""Common plumbing for generative image-editing providers."""

from __future__ import annotations

import base64
import os
from typing import Any

import requests


class ProviderError(RuntimeError):
    """An error whose message is safe and useful to show to the end user."""


class ImageEditProvider:
    """Base class: one image + one instruction in, one edited image out (bytes)."""

    name: str = ""
    label: str = ""
    env_keys: tuple[str, ...] = ()   # API-key variables, checked in order
    default_model: str = ""

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None,
                 timeout: float = 150.0) -> None:
        self.api_key = api_key
        self.model = model or self.default_model
        self.base_url = (base_url or self.default_base_url()).rstrip("/")
        self.timeout = timeout

    # -- configuration ------------------------------------------------------ #
    @classmethod
    def default_base_url(cls) -> str:  # pragma: no cover - overridden
        return ""

    @classmethod
    def from_env(cls) -> "ImageEditProvider | None":
        key = next((os.environ[k].strip() for k in cls.env_keys if os.getenv(k, "").strip()), "")
        if not key:
            return None
        prefix = cls.name.upper()
        try:
            timeout = float(os.getenv("AI_TIMEOUT", "150") or 150)
        except ValueError:
            timeout = 150.0
        return cls(key, model=os.getenv(f"{prefix}_MODEL") or None,
                   base_url=os.getenv(f"{prefix}_BASE_URL") or None, timeout=timeout)

    def describe(self) -> dict:
        return {"name": self.name, "label": self.label, "model": self.model}

    # -- the one method subclasses implement -------------------------------- #
    def edit(self, image_jpeg: bytes, prompt: str, negative_prompt: str = "") -> bytes:
        raise NotImplementedError

    # -- HTTP helpers (kept tiny so tests can monkeypatch them) ------------- #
    def _post_json(self, url: str, headers: dict, body: dict) -> dict:
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=(10, self.timeout))
        except requests.Timeout as exc:
            raise ProviderError(f"{self.label} 响应超时，请稍后再试") from exc
        except requests.RequestException as exc:
            raise ProviderError(f"连接 {self.label} 失败：{exc.__class__.__name__}") from exc
        try:
            data: Any = resp.json()
        except ValueError:
            data = {}
        if resp.status_code >= 400:
            raise ProviderError(f"{self.label} 返回错误（HTTP {resp.status_code}）：{self._error_message(data)}")
        if not isinstance(data, dict):
            raise ProviderError(f"{self.label} 返回了无法解析的内容")
        return data

    def _get_bytes(self, url: str) -> bytes:
        try:
            resp = requests.get(url, timeout=(10, 60))
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"下载 {self.label} 生成的图片失败") from exc
        return resp.content

    @staticmethod
    def _error_message(data: Any) -> str:
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])[:300]
            if data.get("message"):
                return str(data["message"])[:300]
        return "未知错误"

    @staticmethod
    def _data_uri(image_jpeg: bytes) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
