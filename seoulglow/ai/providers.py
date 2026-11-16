"""Concrete providers. Each one is a thin, dependency-free HTTP client.

| name        | service                         | reachable from mainland China |
|-------------|---------------------------------|-------------------------------|
| dashscope   | 阿里云百炼 · 千问图像编辑        | yes                           |
| volcengine  | 火山引擎方舟 · 豆包 Seedream     | yes                           |
| gemini      | Google Gemini · Nano Banana     | needs a proxy                 |

Model names change often — every default can be overridden with
``<NAME>_MODEL`` / ``<NAME>_BASE_URL`` environment variables.
"""

from __future__ import annotations

import base64
import os

from .base import ImageEditProvider, ProviderError


class DashScopeProvider(ImageEditProvider):
    """Alibaba Cloud Model Studio (百炼), multimodal-generation endpoint."""

    name = "dashscope"
    label = "阿里云百炼 · 千问图像"
    env_keys = ("DASHSCOPE_API_KEY",)
    default_model = "qwen-image-2.1-pro"

    @classmethod
    def default_base_url(cls) -> str:
        workspace = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
        if workspace:  # newer consoles hand out workspace-scoped endpoints
            region = os.getenv("DASHSCOPE_REGION", "cn-beijing").strip() or "cn-beijing"
            return f"https://{workspace}.{region}.maas.aliyuncs.com/api/v1"
        return "https://dashscope.aliyuncs.com/api/v1"

    def edit(self, image_jpeg: bytes, prompt: str, negative_prompt: str = "") -> bytes:
        body: dict = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": [
                {"image": self._data_uri(image_jpeg)},
                {"text": prompt},
            ]}]},
            # prompt_extend=False: our prompt is already precise, and rewriting it
            # tends to change the person's face.
            "parameters": {"n": 1, "prompt_extend": False, "watermark": False},
        }
        if negative_prompt:
            body["parameters"]["negative_prompt"] = negative_prompt[:500]
        data = self._post_json(
            f"{self.base_url}/services/aigc/multimodal-generation/generation",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            body,
        )
        try:
            content = data["output"]["choices"][0]["message"]["content"]
            url = next(item["image"] for item in content if isinstance(item, dict) and item.get("image"))
        except (KeyError, IndexError, TypeError, StopIteration) as exc:
            raise ProviderError(f"{self.label} 没有返回图片：{self._error_message(data)}") from exc
        return self._get_bytes(url)


class VolcengineProvider(ImageEditProvider):
    """Volcano Engine Ark (火山方舟) Seedream, OpenAI-style images endpoint."""

    name = "volcengine"
    label = "火山引擎 · 豆包 Seedream"
    env_keys = ("ARK_API_KEY", "VOLCENGINE_API_KEY")
    default_model = "doubao-seedream-4-5-251128"

    @classmethod
    def default_base_url(cls) -> str:
        return "https://ark.cn-beijing.volces.com/api/v3"

    @classmethod
    def from_env(cls):
        provider = super().from_env()
        if provider is not None:  # ARK_* is the console's own naming
            provider.model = os.getenv("ARK_MODEL") or os.getenv("VOLCENGINE_MODEL") or provider.model
            provider.base_url = (os.getenv("ARK_BASE_URL") or provider.base_url).rstrip("/")
        return provider

    def edit(self, image_jpeg: bytes, prompt: str, negative_prompt: str = "") -> bytes:
        body = {
            "model": self.model,
            "prompt": prompt,
            "image": self._data_uri(image_jpeg),
            "size": os.getenv("ARK_SIZE", "2K"),
            "response_format": "b64_json",
            "watermark": False,
            "sequential_image_generation": "disabled",
        }
        data = self._post_json(
            f"{self.base_url}/images/generations",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            body,
        )
        try:
            item = data["data"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.label} 没有返回图片：{self._error_message(data)}") from exc
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"])
        if item.get("url"):
            return self._get_bytes(item["url"])
        raise ProviderError(f"{self.label} 没有返回图片")


class GeminiProvider(ImageEditProvider):
    """Google Gemini image models (Nano Banana family) via generateContent."""

    name = "gemini"
    label = "Google Gemini · Nano Banana"
    env_keys = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    default_model = "gemini-3.1-flash-image"

    @classmethod
    def default_base_url(cls) -> str:
        return "https://generativelanguage.googleapis.com/v1beta"

    def edit(self, image_jpeg: bytes, prompt: str, negative_prompt: str = "") -> bytes:
        text = prompt if not negative_prompt else f"{prompt}\nAvoid: {negative_prompt}"
        body = {"contents": [{"role": "user", "parts": [
            {"text": text},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_jpeg).decode("ascii")}},
        ]}]}
        data = self._post_json(
            f"{self.base_url}/models/{self.model}:generateContent",
            {"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            body,
        )
        candidates = data.get("candidates") or []
        for candidate in candidates:
            for part in (candidate.get("content") or {}).get("parts") or []:
                if part.get("thought"):
                    continue  # interim "thinking" images
                blob = part.get("inlineData") or part.get("inline_data")
                if blob and blob.get("data"):
                    return base64.b64decode(blob["data"])
        reason = ((data.get("promptFeedback") or {}).get("blockReason")
                  or (candidates[0].get("finishReason") if candidates else None) or "未知原因")
        raise ProviderError(f"{self.label} 没有返回图片（{reason}），换张照片或稍后再试")


PROVIDERS: dict[str, type[ImageEditProvider]] = {
    "dashscope": DashScopeProvider,
    "volcengine": VolcengineProvider,
    "gemini": GeminiProvider,
}
AUTO_ORDER = ["dashscope", "volcengine", "gemini"]  # China-reachable services first


def get_provider(name: str | None = "auto") -> ImageEditProvider | None:
    name = (name or "auto").strip().lower()
    if name in {"none", "off", "local", ""}:
        return None
    if name == "auto":
        for candidate in AUTO_ORDER:
            provider = PROVIDERS[candidate].from_env()
            if provider is not None:
                return provider
        return None
    cls = PROVIDERS.get(name)
    return cls.from_env() if cls else None


def available_providers() -> list[dict]:
    found = []
    for name in AUTO_ORDER:
        provider = PROVIDERS[name].from_env()
        if provider is not None:
            found.append(provider.describe())
    return found
