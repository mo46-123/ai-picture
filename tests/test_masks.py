import numpy as np

import synthetic as S
from seoulglow import face as F
from seoulglow.masks import close_holes, dilate, fill_poly, gaussian_ellipse


def at(mask, x, y):
    return float(mask[int(round(y)), int(round(x))])


def test_fill_poly_and_dilate():
    square = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], float)
    m = fill_poly((40, 40), square)
    assert m[20, 20] == 1.0 and m[5, 5] == 0.0
    grown = dilate(m, 4)  # large-radius path (distance transform)
    assert grown[20, 7] == 1.0 and grown[20, 2] == 0.0


def test_close_holes_fills_small_holes_only():
    m = np.ones((100, 100), np.float32)
    m[48:52, 48:52] = 0.0       # a blemish-sized hole
    m[:, :20] = 0.0             # a large non-skin area
    out = close_holes(m, 6)
    assert out[50, 50] > 0.9
    assert out[50, 5] < 0.1


def test_gaussian_ellipse_peak_and_falloff():
    out = gaussian_ellipse((60, 60), np.array([30.0, 30.0]), 6, 3)
    assert abs(out[30, 30] - 1.0) < 1e-6
    assert out[30, 42] < out[30, 36] < out[30, 30]   # along x
    assert out[36, 30] < out[30, 36]                 # narrower along y


def test_face_geometry(face):
    assert abs(face.iod - 2 * S.EYE_DX) < 1.0
    assert np.allclose(face.x_axis, [1, 0], atol=1e-6)
    assert np.allclose(face.y_axis, [0, 1], atol=1e-6)
    x0, y0, x1, y1 = face.roi((S.H, S.W))
    assert x0 >= 0 and y0 >= 0 and x1 <= S.W and y1 <= S.H


def test_skin_mask_covers_skin_and_skips_features(regions):
    skin = regions.skin
    cheek_x, cheek_y = S.CX - 90, S.CY + 40
    assert at(skin, cheek_x, cheek_y) > 0.7                    # cheek
    assert at(skin, S.CX - S.EYE_DX, S.EYE_Y) < 0.1            # eye
    assert at(skin, S.CX, S.MOUTH_Y + 8) < 0.1                 # lips
    assert at(skin, 40, 40) < 0.05                             # background


def test_blemishes_count_as_skin(regions):
    """The color model rejects spots; closing must add them back so they get smoothed."""
    assert regions.skin_probability.min() >= 0.0
    face_area = regions.face_poly > 0.99
    covered = (regions.skin_probability > 0.5) & face_area & (regions.features < 0.05)
    assert covered.sum() / (face_area & (regions.features < 0.05)).sum() > 0.97


def test_lip_masks(regions, face):
    lips = regions.lips
    assert at(lips, S.CX, S.MOUTH_Y + 10) > 0.8                # lower lip
    assert at(lips, S.CX, S.MOUTH_Y - 40) < 0.05               # above the mouth
    prox = regions.lip_inner_proximity
    assert at(prox, S.CX, S.MOUTH_Y) > at(prox, S.CX, S.MOUTH_Y + 16)


def test_wide_skin_mask_reaches_neck(regions):
    neck_point = (S.CX, S.CY + S.B + 40)
    assert at(regions.skin_wide, *neck_point) > at(regions.skin, *neck_point)
    assert len(F.NOSE_LOWER) == 16


def test_frontal_face_is_frontal_and_turned_face_is_not(face):
    assert face.frontalness == 1.0
    turned = face.landmarks.copy()
    turned[F.NOSE_TIP, 0] += 0.6 * face.iod          # nose far off the eye midpoint = profile
    assert F.Face(turned).frontalness == 0.0


def test_thin_lines_near_the_eyes_are_protected(img, face):
    """A glasses frame must not be smoothed away as if it were a blemish."""
    import cv2
    from seoulglow.masks import FaceRegions
    with_frame = img.copy()
    y = int(S.EYE_Y + 34)
    cv2.line(with_frame, (int(S.CX - 120), y), (int(S.CX + 120), y), (0.2, 0.2, 0.22), 2)
    regions = FaceRegions(with_frame, face)
    assert regions.fine_lines[y, int(S.CX - 100)] > 0.8
    assert regions.skin[y, int(S.CX - 100)] < 0.2
    assert FaceRegions(img, face).fine_lines[int(S.CY + 60), int(S.CX - 90)] < 0.05   # plain cheek
