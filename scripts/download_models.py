"""Download the MediaPipe face landmark model (~3.6 MB) into ./models/.

    python scripts/download_models.py
    python scripts/download_models.py --url https://your-mirror/face_landmarker.task

storage.googleapis.com is not reachable from mainland China. In that case,
download the file elsewhere (or from a mirror) and put it at
models/face_landmarker.task, or pass --url / set FACE_MODEL_URL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seoulglow.config import FACE_MODEL_URL_DEFAULT, get_settings  # noqa: E402
from seoulglow.face import ensure_model  # noqa: E402


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=settings.face_model_url or FACE_MODEL_URL_DEFAULT)
    parser.add_argument("--out", type=Path, default=settings.face_model_path)
    args = parser.parse_args()

    if ensure_model(args.out, args.url):
        print(f"✓ 人脸模型已就绪: {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
        return 0
    print("✗ 下载失败。国内网络请用镜像地址：--url <镜像链接>，或手动把 face_landmarker.task 放进 models/ 文件夹。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
