import base64
import dataclasses
import io

import pytest
from PIL import Image
from starlette.testclient import TestClient

from seoulglow.imageio import encode_jpeg
from seoulglow.web.app import RateLimiter, create_app


@pytest.fixture
def client(settings, fake_analyzer):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def jpeg(face_rgb):
    return encode_jpeg(face_rgb)


def decode_data_url(url):
    header, b64 = url.split(",", 1)
    assert header == "data:image/jpeg;base64"
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_health_and_styles(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["face_model"] is True and health["ai_providers"] == []
    styles = client.get("/api/styles").json()
    assert [p["id"] for p in styles["presets"]][0] == styles["default"]
    assert styles["max_upload_mb"] > 0


def test_index_page_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "SeoulGlow" in page.text
    assert client.get("/app.js").status_code == 200


def test_retouch_returns_an_image(client, jpeg):
    r = client.post("/api/retouch", files={"image": ("a.jpg", jpeg, "image/jpeg")},
                    data={"style": "kdrama", "slim": "0.8", "lip": "rose", "preview": "1"})
    body = r.json()
    assert r.status_code == 200 and body["faces"] == 1 and body["notice"] is None
    assert decode_data_url(body["image"]).size == (body["width"], body["height"])


def test_retouch_rejects_missing_or_bad_files(client):
    assert client.post("/api/retouch", data={"style": "kdrama"}).status_code == 400
    r = client.post("/api/retouch", files={"image": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "图片" in r.json()["error"]


def test_upload_size_limit(settings, fake_analyzer):
    small = dataclasses.replace(settings, max_upload_mb=1.0)
    with TestClient(create_app(small)) as c:
        r = c.post("/api/retouch", files={"image": ("big.jpg", b"0" * (1024 * 1024 + 10), "image/jpeg")})
    assert r.status_code == 413


def test_ai_without_keys_falls_back_to_local(client, jpeg):
    body = client.post("/api/ai", files={"image": ("a.jpg", jpeg, "image/jpeg")}, data={"style": "id_photo"}).json()
    assert body["fallback"] is True and body["provider"] is None and "API Key" in body["notice"]


def test_unknown_style_uses_the_default(client, jpeg):
    r = client.post("/api/retouch", files={"image": ("a.jpg", jpeg, "image/jpeg")}, data={"style": "???"})
    assert r.status_code == 200


def test_rate_limiter():
    limiter = RateLimiter("2/minute")
    assert limiter.allow("1.2.3.4") and limiter.allow("1.2.3.4")
    assert not limiter.allow("1.2.3.4")
    assert limiter.allow("5.6.7.8")
    unlimited = RateLimiter("0/hour")
    assert all(unlimited.allow("x") for _ in range(100))
