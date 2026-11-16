import numpy as np
import pytest

from seoulglow.color import Grade, apply_grade, apply_lut, compile_lut, pixel_ops, to_cube
from seoulglow.presets import PRESETS


def gradient_image(size=128):
    gx, gy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    return np.dstack([gx, gy, (gx + gy) / 2]).astype(np.float32)


def test_neutral_grade_is_identity():
    img = gradient_image()
    out = pixel_ops(img, Grade())
    assert np.allclose(out, img, atol=1e-6)


def test_zero_amount_returns_input_unchanged():
    img = gradient_image()
    assert apply_grade(img, PRESETS["kdrama"].grade, amount=0.0) is img


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_lut_matches_direct_math(preset_id):
    """The 33³ LUT must be visually identical to computing every pixel directly."""
    img = gradient_image(160)
    grade = PRESETS[preset_id].grade
    error = np.abs(apply_lut(img, compile_lut(grade)) - pixel_ops(img, grade))
    assert error.max() * 255 < 4.0
    assert error.mean() * 255 < 0.5


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_grades_stay_in_range_and_brighten(preset_id):
    img = gradient_image() * 0.8
    out = apply_grade(img, PRESETS[preset_id].grade)
    assert out.shape == img.shape and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out.mean() > img.mean()  # every Korean look is brighter than the input


def test_temperature_direction():
    gray = np.full((8, 8, 3), 0.5, np.float32)
    warm = pixel_ops(gray, Grade(temperature=0.5))[0, 0]
    cool = pixel_ops(gray, Grade(temperature=-0.5))[0, 0]
    assert warm[0] > warm[2] and cool[2] > cool[0]


def test_cube_export_format():
    text = to_cube(PRESETS["glass_skin"].grade, title="t", size=9)
    lines = text.strip().splitlines()
    assert lines[1] == "LUT_3D_SIZE 9"
    body = [line for line in lines if line[0].isdigit()]
    assert len(body) == 9 ** 3
    first = [float(v) for v in body[0].split()]
    assert len(first) == 3 and all(0.0 <= v <= 1.0 for v in first)
