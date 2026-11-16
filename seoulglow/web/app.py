"""HTTP API + the mobile web page (Starlette).

Routes
------
GET  /api/health    service status (face model loaded? AI providers configured?)
GET  /api/styles    presets, lipstick shades and what features are available
POST /api/retouch   local Korean-style retouch   (multipart: image + options)
POST /api/ai        generative "AI 写真"         (multipart: image + style)
GET  /              the web page (static files)

Uploaded photos are processed in memory and never written to disk.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .. import __version__
from ..ai import ai_portrait, available_providers
from ..config import Settings, get_settings
from ..engine import Adjustments, retouch_bytes
from ..face import analyzer_status, get_analyzer
from ..imageio import InvalidImageError, to_data_url
from ..presets import DEFAULT_PRESET, PRESETS, catalog

log = logging.getLogger("seoulglow")
STATIC_DIR = Path(__file__).parent / "static"
PREVIEW_SIDE = 1280


class RateLimiter:
    """Tiny sliding-window limiter per client IP (protects paid AI quota)."""

    def __init__(self, spec: str) -> None:
        count, _, period = (spec or "10/hour").partition("/")
        seconds = {"minute": 60, "hour": 3600, "day": 86400}.get(period.strip().lower(), 3600)
        try:
            self.limit = max(0, int(count))
        except ValueError:
            self.limit = 10
        self.window = seconds
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit == 0:
            return True  # 0 = unlimited
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "?")


def create_app(settings: Settings | None = None) -> Starlette:
    settings = settings or get_settings()
    workers = asyncio.Semaphore(int(os.getenv("PROCESS_CONCURRENCY", "0") or 0) or (os.cpu_count() or 2))
    ai_limiter = RateLimiter(os.getenv("AI_RATE_LIMIT", "10/hour"))
    limit_bytes = int(settings.max_upload_mb * 1024 * 1024)

    async def read_upload(request: Request):
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit_bytes + 1024 * 1024:
            return None, None, _error(f"图片太大了，请换一张小于 {settings.max_upload_mb:g} MB 的照片", 413)
        try:
            form = await request.form(max_files=1, max_fields=24)
        except Exception:
            return None, None, _error("上传格式不对，请重新选择照片", 400)
        upload = form.get("image")
        if not isinstance(upload, UploadFile):
            return None, None, _error("请先选择一张照片", 400)
        data = await upload.read(limit_bytes + 1)
        if len(data) > limit_bytes:
            return None, None, _error(f"图片太大了，请换一张小于 {settings.max_upload_mb:g} MB 的照片", 413)
        return data, form, None

    def style_of(form) -> str:
        style = str(form.get("style") or DEFAULT_PRESET)
        return style if style in PRESETS else DEFAULT_PRESET

    async def health(request: Request) -> JSONResponse:
        loaded, error = analyzer_status()
        return JSONResponse({"status": "ok", "version": __version__, "face_model": loaded,
                             "face_model_error": error, "ai_providers": available_providers()})

    async def styles(request: Request) -> JSONResponse:
        payload = catalog()
        loaded, error = analyzer_status()
        payload["face_model"] = loaded
        payload["face_model_error"] = error
        payload["ai_providers"] = available_providers()
        payload["max_upload_mb"] = settings.max_upload_mb
        return JSONResponse(payload)

    async def retouch(request: Request) -> JSONResponse:
        data, form, err = await read_upload(request)
        if err:
            return err
        adj = Adjustments.from_mapping(form)
        preview = str(form.get("preview", "")) in {"1", "true", "yes"}
        try:
            async with workers:
                result = await run_in_threadpool(
                    retouch_bytes, data, style_of(form), adj, settings,
                    max_side=PREVIEW_SIDE if preview else None)
        except InvalidImageError as exc:
            return _error(str(exc), 400)
        except Exception:
            log.exception("retouch failed")
            return _error("修图失败了，请换一张照片再试", 500)
        return JSONResponse({
            "image": to_data_url(result.jpeg), "width": result.width, "height": result.height,
            "faces": result.faces, "ms": result.elapsed_ms, "notice": result.notice,
        })

    async def ai(request: Request) -> JSONResponse:
        data, form, err = await read_upload(request)
        if err:
            return err
        if available_providers() and not ai_limiter.allow(_client_ip(request)):
            return _error("AI 写真用得太频繁啦，请稍后再试（本地精修不受限制）", 429)
        provider = str(form.get("provider") or "") or None
        try:
            async with workers:
                result = await run_in_threadpool(ai_portrait, data, style_of(form), provider, settings)
        except InvalidImageError as exc:
            return _error(str(exc), 400)
        except Exception:
            log.exception("ai portrait failed")
            return _error("AI 写真失败了，请稍后再试", 500)
        return JSONResponse({
            "image": to_data_url(result.jpeg), "width": result.width, "height": result.height,
            "provider": result.provider, "fallback": result.fallback, "notice": result.notice,
            "ms": result.elapsed_ms,
        })

    @asynccontextmanager
    async def lifespan(app):
        # Load (and if needed download) the face model in the background so the
        # server starts instantly and the first photo doesn't wait for it.
        threading.Thread(target=get_analyzer, args=(settings,), daemon=True).start()
        yield

    routes = [
        Route("/api/health", health),
        Route("/api/styles", styles),
        Route("/api/retouch", retouch, methods=["POST"]),
        Route("/api/ai", ai, methods=["POST"]),
        Mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static"),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


app = create_app()
