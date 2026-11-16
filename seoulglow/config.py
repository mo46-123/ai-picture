"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FACE_MODEL_URL_DEFAULT = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE per line). Never overrides real env vars."""
    path = path or ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    face_model_path: Path
    face_model_url: str
    auto_download_model: bool
    max_side: int
    max_upload_mb: float
    jpeg_quality: int
    max_faces: int
    ai_provider: str
    ai_label: bool
    host: str
    port: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    models_dir = Path(os.getenv("SEOULGLOW_MODELS_DIR", str(ROOT / "models")))
    return Settings(
        face_model_path=Path(os.getenv("FACE_MODEL_PATH", str(models_dir / "face_landmarker.task"))),
        face_model_url=os.getenv("FACE_MODEL_URL", FACE_MODEL_URL_DEFAULT),
        auto_download_model=_env_bool("AUTO_DOWNLOAD_MODEL", True),
        max_side=max(256, _env_int("MAX_SIDE", 2048)),
        max_upload_mb=max(1.0, _env_float("MAX_UPLOAD_MB", 15.0)),
        jpeg_quality=min(100, max(60, _env_int("JPEG_QUALITY", 92))),
        max_faces=max(1, _env_int("MAX_FACES", 6)),
        ai_provider=os.getenv("AI_PROVIDER", "auto").strip().lower() or "auto",
        ai_label=_env_bool("AI_LABEL", True),
        host=os.getenv("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
    )
