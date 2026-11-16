import base64
import io

import numpy as np
import pytest
from PIL import Image

import synthetic as S
from seoulglow.ai import ai_portrait, available_providers, get_provider
from seoulglow.ai.base import ProviderError
from seoulglow.ai.label import add_ai_badge, aigc_metadata
from seoulglow.ai.prompts import build_prompt
from seoulglow.ai.providers import DashScopeProvider, GeminiProvider, VolcengineProvider
from seoulglow.imageio import encode_jpeg
from seoulglow.presets import PRESETS


def png_bytes(size=(640, 800), color=(240, 200, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def jpeg(face_rgb):
    return encode_jpeg(face_rgb)


def test_auto_picks_china_reachable_providers_first(monkeypatch):
    assert get_provider() is None and available_providers() == []
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert get_provider().name == "gemini"
    monkeypatch.setenv("ARK_API_KEY", "a")
    assert get_provider().name == "volcengine"
    monkeypatch.setenv("DASHSCOPE_API_KEY", "d")
    assert get_provider().name == "dashscope"
    assert [p["name"] for p in available_providers()] == ["dashscope", "volcengine", "gemini"]
    assert get_provider("none") is None and get_provider("gemini").name == "gemini"


def test_model_and_endpoint_overrides(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "d")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws123")
    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen-image-edit-plus")
    p = get_provider("dashscope")
    assert p.model == "qwen-image-edit-plus"
    assert p.base_url == "https://ws123.cn-beijing.maas.aliyuncs.com/api/v1"


def test_dashscope_request_and_response(monkeypatch):
    seen = {}
    provider = DashScopeProvider("key")
    monkeypatch.setattr(provider, "_post_json", lambda url, headers, body: seen.update(url=url, body=body) or {
        "output": {"choices": [{"message": {"content": [{"image": "https://x/out.png"}]}}]}})
    monkeypatch.setattr(provider, "_get_bytes", lambda url: b"PNG" if url == "https://x/out.png" else b"")
    assert provider.edit(b"jpeg", "prompt", "neg") == b"PNG"
    content = seen["body"]["input"]["messages"][0]["content"]
    assert content[0]["image"].startswith("data:image/jpeg;base64,") and content[1]["text"] == "prompt"
    assert seen["body"]["parameters"]["prompt_extend"] is False
    assert seen["url"].endswith("/services/aigc/multimodal-generation/generation")


def test_volcengine_request_and_response(monkeypatch):
    seen = {}
    provider = VolcengineProvider("key")
    out = png_bytes()
    monkeypatch.setattr(provider, "_post_json", lambda url, headers, body: seen.update(body=body) or {
        "data": [{"b64_json": base64.b64encode(out).decode()}]})
    assert provider.edit(b"jpeg", "prompt") == out
    assert seen["body"]["response_format"] == "b64_json" and seen["body"]["watermark"] is False


def test_gemini_skips_thought_images(monkeypatch):
    provider = GeminiProvider("key")
    final, draft = png_bytes(color=(1, 2, 3)), png_bytes(color=(9, 9, 9))
    monkeypatch.setattr(provider, "_post_json", lambda url, headers, body: {"candidates": [{"content": {"parts": [
        {"inlineData": {"data": base64.b64encode(draft).decode()}, "thought": True},
        {"text": "here you go"},
        {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(final).decode()}},
    ]}}]})
    assert provider.edit(b"jpeg", "prompt") == final


def test_gemini_reports_blocked_requests(monkeypatch):
    provider = GeminiProvider("key")
    monkeypatch.setattr(provider, "_post_json", lambda *a: {"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(ProviderError, match="SAFETY"):
        provider.edit(b"jpeg", "prompt")


def test_provider_failure_falls_back_to_local(monkeypatch, jpeg, settings):
    monkeypatch.setenv("ARK_API_KEY", "a")

    def boom(self, *args):
        raise ProviderError("火山引擎 返回错误（HTTP 401）：bad key")

    monkeypatch.setattr(VolcengineProvider, "_post_json", boom)
    result = ai_portrait(jpeg, "kdrama", settings=settings, analyzer=S.FakeAnalyzer())
    assert result.fallback is True and "bad key" in result.notice


def test_successful_ai_result_is_labeled(monkeypatch, jpeg, settings):
    monkeypatch.setenv("ARK_API_KEY", "a")
    monkeypatch.setattr(VolcengineProvider, "_post_json", lambda self, *a: {
        "data": [{"b64_json": base64.b64encode(png_bytes()).decode()}]})
    result = ai_portrait(jpeg, "glass_skin", settings=settings, analyzer=S.FakeAnalyzer())
    assert result.fallback is False and result.provider["name"] == "volcengine"
    img = Image.open(io.BytesIO(result.jpeg))
    assert b'"AIGC"' in img.info.get("comment", b"")


@pytest.mark.parametrize("style", sorted(PRESETS))
def test_prompts_always_protect_identity(style):
    zh, neg = build_prompt(style, "zh")
    en, _ = build_prompt(style, "en")
    assert "同一个人" in zh and "same person" in en and "换脸" in neg


def test_badge_changes_only_the_corner():
    img = np.full((400, 300, 3), 200, np.uint8)
    out = add_ai_badge(img)
    assert out.shape == img.shape
    assert np.array_equal(out[:200], img[:200]) and not np.array_equal(out[-40:, -120:], img[-40:, -120:])
    comment, exif = aigc_metadata("volcengine", "m")
    assert '"Label": "1"' in comment and exif
