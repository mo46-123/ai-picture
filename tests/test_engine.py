import io

import numpy as np
import pytest
from PIL import Image

import synthetic as S
from seoulglow import engine
from seoulglow.engine import Adjustments, process, resolve, retouch_bytes
from seoulglow.imageio import InvalidImageError, decode_image, encode_jpeg
from seoulglow.presets import LIP_SHADES, PRESETS, catalog, get_preset


def test_adjustments_parsing_clamps_and_ignores_garbage():
    adj = Adjustments.from_mapping({"intensity": "2", "smooth": "-1", "slim": "0.4", "eyes": "abc",
                                    "nose": "nan", "lip": "not-a-shade"})
    assert adj.intensity == 1.0 and adj.smooth == 0.0 and adj.slim == 0.4
    assert adj.eyes is None and adj.nose is None and adj.lip is None
    assert Adjustments.from_mapping({"lip": "none"}).lip == "none"
    assert Adjustments.from_mapping({}).intensity == 1.0


def test_resolve_uses_preset_defaults_and_overrides():
    preset = PRESETS["kdrama"]
    p = resolve(preset, Adjustments())
    assert p.slim == preset.slim and p.nose == preset.nose and p.filter == preset.default_filter
    assert p.lip_rgb == LIP_SHADES[preset.lip].rgb
    p = resolve(preset, Adjustments(intensity=0.5, slim=1.0, lip="none"))
    assert p.slim == 0.5 and p.eyes == preset.eyes * 0.5 and p.lip_rgb is None


def test_unknown_preset_falls_back_to_default():
    assert get_preset("nope").id == get_preset(None).id


def test_catalog_lists_every_slider():
    data = catalog()
    assert len(data["presets"]) == 5 and len(data["lip_shades"]) == 5
    for p in data["presets"]:
        assert set(p["defaults"]) == {"smooth", "whiten", "slim", "eyes", "nose", "makeup", "filter", "lip"}


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_every_preset_changes_the_face_more_than_the_background(face_rgb, face, preset_id):
    out = process(face_rgb, preset_id, Adjustments(filter=0.0), [face])
    assert out.shape == face_rgb.shape and out.dtype == np.uint8
    diff = np.abs(out.astype(int) - face_rgb.astype(int)).sum(axis=2)
    face_change = diff[int(S.CY - 60):int(S.CY + 120), int(S.CX - 100):int(S.CX + 100)].mean()
    corner_change = diff[20:120, S.W - 120:S.W - 20].mean()
    assert face_change > 3 * max(corner_change, 0.5)


def test_no_faces_still_applies_the_filter(face_rgb):
    out = process(face_rgb, "cream_skin", None, [])
    assert np.abs(out.astype(int) - face_rgb.astype(int)).mean() > 1


def test_tiny_faces_are_skipped(face_rgb, face):
    tiny = face.shifted(0, 0)
    tiny.landmarks = tiny.landmarks * 0.05
    out = process(face_rgb, "glass_skin", Adjustments(filter=0.0), [engine.Face(tiny.landmarks)])
    assert np.array_equal(out, face_rgb)


def test_retouch_bytes_with_and_without_a_model(face_rgb, settings):
    data = encode_jpeg(face_rgb)
    result = retouch_bytes(data, "glass_skin", None, settings, analyzer=S.FakeAnalyzer())
    assert result.faces == 1 and result.notice is None
    assert Image.open(io.BytesIO(result.jpeg)).size == (S.W, S.H)
    result = retouch_bytes(data, "glass_skin", None, settings, analyzer=None)
    assert result.faces == 0 and "模型" in result.notice


def test_preview_size_is_capped(face_rgb, settings):
    big = np.asarray(Image.fromarray(face_rgb).resize((1440, 1800)))
    result = retouch_bytes(encode_jpeg(big), "kdrama", None, settings, analyzer=S.FakeAnalyzer(), max_side=1280)
    assert max(result.width, result.height) == 1280


def test_decode_rejects_garbage_and_tiny_images():
    with pytest.raises(InvalidImageError):
        decode_image(b"definitely not an image")
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, "PNG")
    with pytest.raises(InvalidImageError):
        decode_image(buf.getvalue())


def test_decode_flattens_transparency_onto_white():
    buf = io.BytesIO()
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(buf, "PNG")
    assert decode_image(buf.getvalue()).min() == 255
