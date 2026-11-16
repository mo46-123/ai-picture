"""Command line:  python -m seoulglow INPUT.jpg [-s kdrama] [-o out.jpg] [--all]

Handy for batch work and for checking the pipeline on real photos without the web page.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import get_settings
from .engine import Adjustments, retouch_bytes
from .face import analyzer_status, get_analyzer
from .presets import PRESETS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m seoulglow", description="SeoulGlow 韩系修图（命令行）")
    parser.add_argument("input", type=Path, help="照片路径")
    parser.add_argument("-s", "--style", default="glass_skin", choices=sorted(PRESETS), help="韩系风格")
    parser.add_argument("-o", "--output", type=Path, help="输出路径（默认：原文件名_风格.jpg）")
    parser.add_argument("--all", action="store_true", help="把 5 种风格都输出一遍")
    parser.add_argument("--intensity", type=float, default=1.0, help="整体强度 0~1")
    for key, label in [("smooth", "磨皮"), ("whiten", "美白"), ("slim", "瘦脸"), ("eyes", "大眼"),
                       ("makeup", "妆容"), ("filter", "滤镜")]:
        parser.add_argument(f"--{key}", type=float, help=f"{label} 0~1（默认用风格预设）")
    parser.add_argument("--lip", help="口红色号：peach / milktea / strawberry / rose / nude / none")
    args = parser.parse_args(argv)

    data = args.input.read_bytes()
    settings = get_settings()
    get_analyzer(settings)
    loaded, error = analyzer_status()
    if not loaded:
        print(f"提示：{error}", file=sys.stderr)

    adj = Adjustments.from_mapping(vars(args))
    styles = sorted(PRESETS) if args.all else [args.style]
    for style in styles:
        result = retouch_bytes(data, style, adj, settings)
        out = args.output if (args.output and not args.all) else args.input.with_name(f"{args.input.stem}_{style}.jpg")
        out.write_bytes(result.jpeg)
        note = f"（{result.notice}）" if result.notice else ""
        print(f"✓ {PRESETS[style].name}: {out}  {result.faces} 张脸，{result.elapsed_ms} ms{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
