"""Run the web app:  python -m seoulglow.web  [--host 0.0.0.0] [--port 8000]"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from ..config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="SeoulGlow 韩系修图 web app")
    parser.add_argument("--host", default=settings.host, help="use 0.0.0.0 to open it from your phone on the same Wi-Fi")
    parser.add_argument("--port", type=int, default=settings.port)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"\n  SeoulGlow 已启动：http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}\n")
    uvicorn.run("seoulglow.web.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
