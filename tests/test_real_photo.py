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
