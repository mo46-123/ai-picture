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
    assert lifted[corner].mean() > img[corner].mean() * 1.15

    out = process(dark, "glass_skin", Adjustments(), dark_faces)
    assert out.shape == dark.shape and out.dtype == np.uint8


def test_rounder_faces_get_more_slimming_than_oval_ones(analyzer, photo):
    from seoulglow import reshape

    face = analyzer.detect(photo)[0]
    center = face.pt(F.NOSE_TIP)

    def widened(factor):
        pts = face.landmarks.copy()
        pts[:, 0] = center[0] + (pts[:, 0] - center[0]) * factor
        return F.Face(pts)

    narrow, normal, wide = (reshape.needs(widened(k))[0] for k in (0.85, 1.0, 1.2))
    assert 0.35 <= narrow <= normal <= wide <= 1.0
    assert wide > narrow + 0.2


def _lamp_from_one_side(rgb: np.ndarray, face, ev: float = -1.2, ambient: float = 0.18) -> np.ndarray:
    """The same photo under a lamp to one side: one half of the face lit, the other in shade."""
    x = rgb.astype(np.float32) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    h, w = x.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nose = face.pt(F.NOSE_TIP)
    s = ((xs - nose[0]) * face.x_axis[0] + (ys - nose[1]) * face.x_axis[1]) / (0.9 * face.iod)
    key = np.clip(0.5 + 0.5 * s, 0, 1)
    key = key * key * (3 - 2 * key)
    lin = lin * (2.0 ** ev) * (ambient + (1 - ambient) * key)[..., None]
    out = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.clip(lin, 0, 1) ** (1 / 2.4) - 0.055)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def _half_gap(rgb: np.ndarray, face) -> float:
    """Lightness difference between the two cheeks (L*)."""
    from seoulglow import ops
    from seoulglow.skin import cheekbone_patch

    lab = ops.to_lab(rgb.astype(np.float32) / 255.0)
    values = []
    for side in ("right", "left"):
        c = cheekbone_patch(face, side)[0]
        x, y, r = int(c[0]), int(c[1]), max(2, int(0.05 * face.iod))
        values.append(float(lab[y - r:y + r, x - r:x + r, 0].mean()))
    return abs(values[0] - values[1])


def test_light_from_one_side_is_evened_out(analyzer, photo):
    from seoulglow.engine import _balance_faces

    face = analyzer.detect(photo)[0]
    lamp = _lamp_from_one_side(photo, face)
    faces = analyzer.detect(lamp)
    assert faces
    before = _half_gap(lamp, faces[0])
    assert before > 15.0                                     # the simulation is strongly one-sided
    balanced = (np.clip(_balance_faces(lamp.astype(np.float32) / 255.0, faces, 1.0), 0, 1) * 255 + 0.5).astype(np.uint8)
    assert _half_gap(balanced, faces[0]) < 0.6 * before

    # the full pipeline no longer turns the lit half white while the other stays dark
    out = process(lamp, "glass_skin", Adjustments(), faces)
    assert _half_gap(out, faces[0]) < before


def test_evenly_lit_face_is_not_rebalanced(analyzer, photo):
    from seoulglow.engine import _balance_faces

    faces = analyzer.detect(photo)
    img = photo.astype(np.float32) / 255.0
    balanced = _balance_faces(img.copy(), faces, 1.0)
    assert float(np.abs(balanced - img).mean()) < 0.01
