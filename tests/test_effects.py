"""Every effect: no-op at zero strength, valid output, and the change lands where it should."""

import numpy as np
import pytest

import synthetic as S
from seoulglow import makeup, ops, reshape, skin
from seoulglow.imageio import to_float


def region_mean(img, x0, y0, x1, y1):
    return img[int(y0):int(y1), int(x0):int(x1)].reshape(-1, 3).mean(axis=0)


def lab_mean(img, x0, y0, x1, y1):
    return ops.to_lab(img)[int(y0):int(y1), int(x0):int(x1)].reshape(-1, 3).mean(axis=0)


def check_valid(out, ref):
    assert out.shape == ref.shape and out.dtype == np.float32
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


EFFECTS = [
    lambda i, f, r, a: skin.remove_spots(i, r, a),
    lambda i, f, r, a: skin.smooth(i, r, a),
    lambda i, f, r, a: skin.under_eye(i, f, r, a),
    lambda i, f, r, a: skin.soften_folds(i, f, r, a),
    lambda i, f, r, a: skin.brighten(i, r, a),
    lambda i, f, r, a: skin.glass_glow(i, f, r, a),
    lambda i, f, r, a: skin.matte(i, r, a),
    lambda i, f, r, a: makeup.lips(i, f, r, (236, 112, 108), a),
    lambda i, f, r, a: makeup.blush(i, f, r, (240, 156, 156), a),
    lambda i, f, r, a: makeup.brows(i, f, r, (120, 92, 80), a),
    lambda i, f, r, a: makeup.eyeshadow(i, f, r, (232, 146, 150), a),
    lambda i, f, r, a: makeup.eyeliner(i, f, r, a),
    lambda i, f, r, a: makeup.aegyo_sal(i, f, r, a),
    lambda i, f, r, a: makeup.brighten_eyes(i, r, a),
    lambda i, f, r, a: makeup.whiten_teeth(i, f, r, a),
    lambda i, f, r, a: reshape.slim_face(i, f, a),
    lambda i, f, r, a: reshape.slim_nose(i, f, a),
    lambda i, f, r, a: reshape.enlarge_eyes(i, f, a),
]


@pytest.mark.parametrize("effect", EFFECTS)
def test_zero_strength_is_identity(effect, img, face, regions):
    assert np.array_equal(effect(img, face, regions, 0.0), img)


@pytest.mark.parametrize("effect", EFFECTS)
def test_output_is_valid(effect, img, face, regions):
    check_valid(effect(img, face, regions, 0.8), img)


def test_spot_removal_reduces_blemish_contrast(img, regions):
    """Measured on clean skin only (lips and nose are features, not spots)."""
    skin_only = regions.skin > 0.9
    before = ops.luminance(img)[skin_only].std()
    after = ops.luminance(skin.remove_spots(img, regions, 1.0))[skin_only].std()
    assert after < before * 0.8


def test_lab_effects_leave_untouched_pixels_exactly_alone(img, face, regions):
    out = makeup.blush(img, face, regions, (240, 156, 156), 1.0)
    assert np.array_equal(out[:150], img[:150])      # far above the cheeks


def test_smoothing_reduces_skin_noise_but_not_edges(img, regions):
    patch = (S.CX + 40, S.CY - 70, S.CX + 100, S.CY - 30)   # forehead/cheek, no blemishes
    out = skin.smooth(img, regions, 1.0, texture_keep=0.3)
    assert out[int(patch[1]):int(patch[3]), int(patch[0]):int(patch[2])].std() < \
        img[int(patch[1]):int(patch[3]), int(patch[0]):int(patch[2])].std()
    # the face outline stays sharp: background next to the jaw is untouched
    assert np.allclose(out[60, 60], img[60, 60])


def test_brighten_lifts_skin_only(img, regions):
    out = skin.brighten(img, regions, 0.6)
    cheek = (S.CX - 110, S.CY - 20, S.CX - 70, S.CY + 20)
    assert lab_mean(out, *cheek)[0] > lab_mean(img, *cheek)[0] + 3
    assert np.allclose(region_mean(out, 20, 300, 60, 340), region_mean(img, 20, 300, 60, 340), atol=1e-3)


def test_under_eye_brightens_the_band_below_the_eyes(img, face, regions):
    darkened = img.copy()
    band = (int(S.CX - S.EYE_DX - 20), int(S.EYE_Y + 16), int(S.CX - S.EYE_DX + 20), int(S.EYE_Y + 30))
    darkened[band[1]:band[3], band[0]:band[2]] *= 0.85          # fake dark circles
    out = skin.under_eye(darkened, face, regions, 1.0)
    assert lab_mean(out, *band)[0] > lab_mean(darkened, *band)[0] + 2


def test_lips_move_toward_the_shade(img, face, regions):
    target = (214, 58, 78)
    out = makeup.lips(img, face, regions, target, 1.0, gradient=0.0)
    lip = (S.CX - 15, S.MOUTH_Y + 6, S.CX + 15, S.MOUTH_Y + 14)
    a_before, a_after = lab_mean(img, *lip)[1], lab_mean(out, *lip)[1]
    assert a_after > a_before + 3                                 # redder
    assert np.allclose(region_mean(out, S.CX - 10, S.CY - 60, S.CX + 10, S.CY - 40),
                       region_mean(img, S.CX - 10, S.CY - 60, S.CX + 10, S.CY - 40), atol=1e-4)


def test_brows_only_darken(img, face, regions):
    out = makeup.brows(img, face, regions, (120, 92, 80), 1.0)
    assert (ops.luminance(out) <= ops.luminance(img) + 1e-4).all()


def test_slim_face_narrows_the_jaw(img, face):
    out = reshape.slim_face(img, face, 1.0)
    row = int(S.CY + 0.7 * S.B)

    def face_width(im):
        diff = np.abs(im[row] - np.array(S.SKIN) / 255.0).sum(axis=1)
        xs = np.nonzero(diff < 0.12)[0]
        return xs.max() - xs.min()

    assert face_width(out) < face_width(img) - 6


def test_enlarge_eyes_makes_eyes_wider(img, face):
    out = reshape.enlarge_eyes(img, face, 1.0)
    y = int(S.EYE_Y)
    x0, x1 = int(S.CX - S.EYE_DX - 45), int(S.CX - S.EYE_DX + 45)

    def eye_width(im):
        bright = np.nonzero(ops.luminance(im[y, x0:x1]) > 0.85)[0]   # the white of the eye
        return bright.max() - bright.min()

    assert eye_width(out) > eye_width(img)


def test_displacement_warp_moves_content_in_the_shift_direction():
    canvas = np.zeros((120, 120, 3), np.float32)
    canvas[58:62, 58:62] = 1.0
    out = reshape.displacement_warp(canvas, np.array([[60.0, 60.0]]), np.array([[6.0, 0.0]]),
                                    sigma=15.0, roi=(0, 0, 120, 120), step=2)
    ys, xs = np.nonzero(out[..., 0] > 0.5)
    assert 64 <= xs.mean() <= 68 and abs(ys.mean() - 59.5) < 1.5


def test_effects_work_on_float_input_from_uint8(face_rgb, face):
    out = reshape.slim_nose(to_float(face_rgb), face, 0.5)
    check_valid(out, to_float(face_rgb))
