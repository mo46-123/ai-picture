from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import FakeAnalyzer, synthetic_face, synthetic_face_image  # noqa: E402

from seoulglow import face as face_module  # noqa: E402
from seoulglow.config import get_settings  # noqa: E402
from seoulglow.imageio import to_float  # noqa: E402
from seoulglow.masks import FaceRegions  # noqa: E402

AI_KEYS = ["DASHSCOPE_API_KEY", "ARK_API_KEY", "VOLCENGINE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]


@pytest.fixture(autouse=True)
def _no_ai_keys(monkeypatch):
    """Tests never talk to real AI services, whatever the developer's environment holds."""
    for key in AI_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session")
def face_rgb():
    return synthetic_face_image()


@pytest.fixture(scope="session")
def face():
    return synthetic_face()


@pytest.fixture
def img(face_rgb):
    return to_float(face_rgb)


@pytest.fixture
def regions(img, face):
    return FaceRegions(img, face)


@pytest.fixture
def settings(tmp_path):
    """Settings that never download anything and point at a missing model."""
    return dataclasses.replace(get_settings(), face_model_path=tmp_path / "missing.task",
                               auto_download_model=False, ai_provider="auto")


@pytest.fixture
def fake_analyzer():
    analyzer = FakeAnalyzer()
    face_module.set_analyzer(analyzer)
    yield analyzer
    face_module.set_analyzer(None)
