"""Face landmark detection (MediaPipe Face Landmarker) and face geometry helpers.

All landmark indices follow MediaPipe's 468/478-point Face Mesh topology.
"Right"/"left" always mean the *subject's* right/left: in a normal (non-mirrored)
photo the subject's right eye appears on the left side of the image.
"""

from __future__ import annotations

import atexit
import logging
import shutil
import threading
import urllib.request
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Landmark index groups (ordered as closed polygons)
# --------------------------------------------------------------------------- #
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37,
              39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82,
              81, 80, 191]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_EYE_LOWER_LID = [7, 163, 144, 145, 153, 154, 155]   # corners excluded
LEFT_EYE_LOWER_LID = [249, 390, 373, 374, 380, 381, 382]
# Full lid contours, ordered from the outer corner to the inner corner.
RIGHT_UPPER_LID = [33, 246, 161, 160, 159, 158, 157, 173, 133]
LEFT_UPPER_LID = [263, 466, 388, 387, 386, 385, 384, 398, 362]
RIGHT_LOWER_LID = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LEFT_LOWER_LID = [263, 249, 390, 373, 374, 380, 381, 382, 362]
RIGHT_BROW = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70]
LEFT_BROW = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300]
NOSE_BRIDGE = [168, 6, 197, 195, 5]
NOSE_LOWER = [98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64]

NOSE_TIP = 4
CHIN = 152
UPPER_LIP_TOP, UPPER_LIP_BOTTOM = 0, 13
LOWER_LIP_TOP, LOWER_LIP_BOTTOM = 14, 17
MOUTH_RIGHT, MOUTH_LEFT = 61, 291
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 159, 145, 33, 133
LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_OUTER, LEFT_EYE_INNER = 386, 374, 263, 362
RIGHT_FACE_EDGE, LEFT_FACE_EDGE = 234, 454

# Face-slimming contour: (landmark, inward share, upward share) of the slimming shift.
# Jaw corners move in the most (V-line); the chin moves up a little (small face).
JAW_CONTOUR_RIGHT = [(234, 0.30, 0.0), (93, 0.55, 0.0), (132, 0.80, 0.0), (58, 0.95, 0.05),
                     (172, 1.00, 0.12), (136, 0.90, 0.18), (150, 0.70, 0.22), (149, 0.48, 0.24),
                     (176, 0.28, 0.25), (148, 0.12, 0.25)]
JAW_CONTOUR_LEFT = [(454, 0.30, 0.0), (323, 0.55, 0.0), (361, 0.80, 0.0), (288, 0.95, 0.05),
                    (397, 1.00, 0.12), (365, 0.90, 0.18), (379, 0.70, 0.22), (378, 0.48, 0.24),
                    (400, 0.28, 0.25), (377, 0.12, 0.25)]
CHIN_LIFT = [(152, 0.0, 0.25)]
# Zero-shift anchors so the warp never drags the eyes, nose or mouth. The mouth corners
# are deliberately not anchors: they sit close to the jaw and pinning them leaves a
# bulge in the contour right beside the chin.
SLIM_ANCHORS = [4, 1, 6, 168, 0, 17, 33, 133, 263, 362, 10, 151]
NECK_TOP = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]

MIN_LANDMARKS = 468


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


@dataclass
class Face:
    """One detected face. ``landmarks`` is an (N, 2) array of pixel coordinates (x, y)."""

    landmarks: np.ndarray

    def __post_init__(self) -> None:
        self.landmarks = np.asarray(self.landmarks, dtype=np.float64)
        if self.landmarks.ndim != 2 or self.landmarks.shape[0] < MIN_LANDMARKS:
            raise ValueError(f"expected >= {MIN_LANDMARKS} landmarks, got {self.landmarks.shape}")

    # -- basic access ------------------------------------------------------- #
    def pt(self, index: int) -> np.ndarray:
        return self.landmarks[index]

    def pts(self, indices: list[int]) -> np.ndarray:
        return self.landmarks[indices]

    def shifted(self, dx: float, dy: float) -> "Face":
        return Face(self.landmarks + np.array([dx, dy]))

    # -- face-aligned coordinate frame -------------------------------------- #
    @cached_property
    def eye_right(self) -> np.ndarray:
        return self.pts(RIGHT_EYE).mean(axis=0)

    @cached_property
    def eye_left(self) -> np.ndarray:
        return self.pts(LEFT_EYE).mean(axis=0)

    @cached_property
    def eye_mid(self) -> np.ndarray:
        return (self.eye_right + self.eye_left) / 2.0

    @cached_property
    def iod(self) -> float:
        """Inter-ocular distance in pixels: the scale unit for every effect."""
        return max(float(np.linalg.norm(self.eye_left - self.eye_right)), 1.0)

    @cached_property
    def x_axis(self) -> np.ndarray:
        """Unit vector from the subject's right eye to the left eye (≈ image →)."""
        return _unit(self.eye_left - self.eye_right)

    @cached_property
    def y_axis(self) -> np.ndarray:
        """Unit vector pointing from the eyes toward the chin (≈ image ↓)."""
        x = self.x_axis
        y = np.array([-x[1], x[0]])
        if float(np.dot(self.pt(CHIN) - self.eye_mid, y)) < 0:
            y = -y
        return y

    @cached_property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.x_axis[1], self.x_axis[0])))

    def outward(self, side: str) -> np.ndarray:
        """Direction pointing away from the face's center line on the given side."""
        return -self.x_axis if side == "right" else self.x_axis

    def eye_center(self, side: str) -> np.ndarray:
        return self.eye_right if side == "right" else self.eye_left

    def eye_width(self, side: str) -> float:
        a, b = (RIGHT_EYE_OUTER, RIGHT_EYE_INNER) if side == "right" else (LEFT_EYE_OUTER, LEFT_EYE_INNER)
        return float(np.linalg.norm(self.pt(a) - self.pt(b)))

    def eye_height(self, side: str) -> float:
        a, b = (RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM) if side == "right" else (LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
        return float(np.linalg.norm(self.pt(a) - self.pt(b)))

    @cached_property
    def frontalness(self) -> float:
        """1.0 for a face looking at the camera, falling to 0.0 for a profile.

        Landmarks on the far side of a turned face are guesses, so makeup and
        reshaping are faded out (a brow drawn on a profile lands on the temple).
        """
        # Frontal: the nose tip sits under the midpoint of the eyes. Turned: it slides
        # sideways while the eyes move closer together in the image.
        offset = abs(float(np.dot(self.pt(NOSE_TIP) - self.eye_mid, self.x_axis))) / self.iod
        t = float(np.clip((offset - 0.28) / (0.55 - 0.28), 0.0, 1.0))
        return 1.0 - t * t * (3.0 - 2.0 * t)

    @cached_property
    def brow_mid(self) -> np.ndarray:
        return (self.pts(RIGHT_BROW).mean(axis=0) + self.pts(LEFT_BROW).mean(axis=0)) / 2.0

    @cached_property
    def eye_to_nose(self) -> float:
        """Vertical (face-frame) distance from the eye line to the nose tip."""
        return max(float(np.dot(self.pt(NOSE_TIP) - self.eye_mid, self.y_axis)), 0.3 * self.iod)

    @cached_property
    def mouth_width(self) -> float:
        return float(np.linalg.norm(self.pt(MOUTH_LEFT) - self.pt(MOUTH_RIGHT)))

    @cached_property
    def lip_thickness(self) -> float:
        upper = np.linalg.norm(self.pt(UPPER_LIP_TOP) - self.pt(UPPER_LIP_BOTTOM))
        lower = np.linalg.norm(self.pt(LOWER_LIP_TOP) - self.pt(LOWER_LIP_BOTTOM))
        return max(float(upper + lower) / 2.0, 1.0)

    # -- bounding boxes ----------------------------------------------------- #
    @cached_property
    def bbox(self) -> tuple[float, float, float, float]:
        oval = self.pts(FACE_OVAL)
        return (float(oval[:, 0].min()), float(oval[:, 1].min()),
                float(oval[:, 0].max()), float(oval[:, 1].max()))

    def neck_polygon(self, depth: float = 1.8) -> np.ndarray:
        """Jaw line extended downward (``depth`` × IOD) into a trapezoid covering the neck."""
        top = self.pts(NECK_TOP)  # runs from the right jaw (172) via the chin to the left jaw (397)
        down, inward = self.y_axis * depth * self.iod, self.x_axis * 0.22 * self.iod
        below_left_jaw = self.pt(397) + down - inward
        below_right_jaw = self.pt(172) + down + inward
        return np.vstack([top, below_left_jaw, below_right_jaw])

    def roi(self, shape: tuple[int, ...], margin: float = 0.8) -> tuple[int, int, int, int]:
        """Integer crop (x0, y0, x1, y1) covering face + neck with a margin, clipped to image.

        The margin must exceed the reach of the softest mask (wide skin mask + blur),
        otherwise brightening would stop at a visible edge at the crop border.
        """
        pts = np.vstack([self.pts(FACE_OVAL), self.neck_polygon(depth=2.6)])
        m = margin * self.iod
        h, w = shape[:2]
        x0 = int(max(0, np.floor(pts[:, 0].min() - m)))
        y0 = int(max(0, np.floor(pts[:, 1].min() - m)))
        x1 = int(min(w, np.ceil(pts[:, 0].max() + m)))
        y1 = int(min(h, np.ceil(pts[:, 1].max() + m)))
        return x0, y0, x1, y1


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def ensure_model(path: Path, url: str, timeout: float = 60.0) -> bool:
    """Make sure the .task model exists locally, downloading it once if needed."""
    if path.is_file() and path.stat().st_size > 100_000:
        return True
    if not url:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        log.info("Downloading face model from %s", url)
        with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as fh:
            shutil.copyfileobj(response, fh)
        if tmp.stat().st_size < 100_000:
            raise IOError("downloaded file is too small to be a model")
        tmp.replace(path)
        return True
    except Exception as exc:  # network errors, HTTP errors, disk errors...
        log.warning("Face model download failed: %s", exc)
        tmp.unlink(missing_ok=True)
        return False


class FaceAnalyzer:
    """Thread-safe wrapper around MediaPipe's FaceLandmarker (IMAGE mode)."""

    def __init__(self, model_path: Path, max_faces: int = 6, min_confidence: float = 0.5) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        atexit.register(self.close)  # MediaPipe's own __del__ is noisy at interpreter shutdown

    def close(self) -> None:
        landmarker, self._landmarker = getattr(self, "_landmarker", None), None
        if landmarker is None:
            return
        try:
            landmarker.close()
        except Exception:  # at interpreter exit MediaPipe's executor may already be gone
            pass
        finally:
            landmarker.close = lambda: None  # stop MediaPipe's __del__ from closing (and failing) again

    def detect(self, rgb: np.ndarray) -> list[Face]:
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        with self._lock:
            result = self._landmarker.detect(image)
        faces: list[Face] = []
        for landmarks in result.face_landmarks:
            if len(landmarks) < MIN_LANDMARKS:
                continue
            arr = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float64)
            faces.append(Face(arr))
        faces.sort(key=lambda f: f.bbox[0])
        return faces


_analyzer: FaceAnalyzer | None = None
_analyzer_error: str | None = None
_analyzer_lock = threading.Lock()


def get_analyzer(settings=None) -> FaceAnalyzer | None:
    """Lazily build the shared analyzer. Returns None if the model is unavailable."""
    global _analyzer, _analyzer_error
    if _analyzer is not None:
        return _analyzer
    with _analyzer_lock:
        if _analyzer is not None:
            return _analyzer
        if settings is None:
            from .config import get_settings

            settings = get_settings()
        path = settings.face_model_path
        url = settings.face_model_url if settings.auto_download_model else ""
        if not ensure_model(path, url):
            _analyzer_error = (
                f"人脸模型不存在：{path}。请运行 `python scripts/download_models.py` 下载，"
                "或设置 FACE_MODEL_URL 使用镜像地址。"
            )
            return None
        try:
            _analyzer = FaceAnalyzer(path, max_faces=settings.max_faces)
            _analyzer_error = None
        except Exception as exc:
            log.exception("Failed to load face model")
            _analyzer_error = f"人脸模型加载失败：{exc}"
            return None
        return _analyzer


def analyzer_status() -> tuple[bool, str | None]:
    return _analyzer is not None, _analyzer_error


def set_analyzer(analyzer) -> None:
    """Inject a custom analyzer (used by tests and for alternative detectors)."""
    global _analyzer, _analyzer_error
    _analyzer, _analyzer_error = analyzer, None
