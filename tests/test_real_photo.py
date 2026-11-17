"""Integration test on a real photo with the real MediaPipe model.

Skipped when the model hasn't been downloaded (python scripts/download_models.py).
CI downloads it, so this runs on every push.
"""

from pathlib import Path

import numpy as np
import pytest

from seoulglow import face as F
from seoulglow.config import get_settings
from seoulglow.engine import Adjustments, process
from seoulglow.imageio import decode_image

SAMPLE = Path(__file__).parent / "data" / "sample_portrait.jpg"
MODEL = get_settings().face_model_path

pytestmark = pytest.mark.skipif(not MODEL.is_file(), reason="face model not downloaded")


@pytest.fixture(scope="module")
def analyzer():
    return F.FaceAnalyzer(MODEL)


@pytest.fixture(scope="module")
def photo():
    return decode_image(SAMPLE.read_bytes(), 1280)


def test_detects_one_face_with_full_landmarks(analyzer, photo):
    faces = analyzer.detect(photo)
    assert len(faces) == 1
    f = faces[0]
    assert f.landmarks.shape == (478, 2)
    assert 40 < f.iod < 400
    assert f.eye_right[0] < f.eye_left[0]           # subject's right eye is on the image's left
    assert f.pt(F.CHIN)[1] > f.pt(F.NOSE_TIP)[1] > f.eye_mid[1]


def test_retouch_changes_the_face_not_the_background(analyzer, photo):
    faces = analyzer.detect(photo)
    out = process(photo, "glass_skin", Adjustments(filter=0.0), faces)
    diff = np.abs(out.astype(int) - photo.astype(int)).sum(axis=2)
    x0, y0, x1, y1 = (int(v) for v in faces[0].bbox)
    inside = diff[y0:y1, x0:x1].mean()
    corner = diff[:40, :40].mean()
    assert inside > 4 and corner < 1


def _darken(rgb: np.ndarray, ev: float) -> np.ndarray:
    """The same photo taken in a dark room: less light, warmer lamp."""
    x = rgb.astype(np.float32) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4) * (2.0 ** ev) * np.array([1.2, 1.0, 0.75])
    out = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.clip(lin, 0, 1) ** (1 / 2.4) - 0.055)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def test_dark_photo_is_brightened_as_a_whole_not_just_the_face(analyzer, photo):
    from seoulglow.engine import _face_lightness, _lift_dark_scene

    faces = analyzer.detect(photo)
    lifted, dark_scene = _lift_dark_scene(photo.astype(np.float32) / 255.0, faces, 1.0)
    assert dark_scene == 0.0 and np.array_equal(lifted, photo.astype(np.float32) / 255.0)  # normal light: untouched

    dark = _darken(photo, -2.2)
    dark_faces = analyzer.detect(dark)
    assert dark_faces
    img = dark.astype(np.float32) / 255.0
    lifted, dark_scene = _lift_dark_scene(img, dark_faces, 1.0)
    assert dark_scene > 0.5
    assert _face_lightness(lifted, dark_faces[0]) > _face_lightness(img, dark_faces[0]) + 8.0
    corner = (slice(0, 40), slice(0, 40))      # the background is brightened too
    assert lifted[corner].mean() > img[corner].mean() * 1.3

    out = process(dark, "glass_skin", Adjustments(), dark_faces)
    assert out.shape == dark.shape and out.dtype == np.uint8
