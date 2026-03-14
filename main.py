"""
yt-dlp FastAPI Backend
Replaces Piped/Invidious with a direct yt-dlp integration.
Endpoints:
  GET /metadata/{video_id}  - title, description, thumbnails, formats
  GET /stream/{video_id}    - direct playable URL (m3u8 / mp4)
"""

import asyncio
import json
import subprocess
import sys
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="yt-dlp Proxy API",
    description="Lightweight API that wraps yt-dlp to serve YouTube metadata and stream URLs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

YT_URL_TEMPLATE = "https://www.youtube.com/watch?v={video_id}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class YtDlpError(Exception):
    """Raised when yt-dlp returns a non-zero exit code."""
    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


def _classify_error(stderr: str) -> tuple[int, str]:
    """
    Map common yt-dlp error strings to appropriate HTTP status codes
    and user-friendly messages.
    """
    lower = stderr.lower()
    if "private video" in lower:
        return 403, "This video is private and cannot be accessed."
    if "age" in lower and ("restricted" in lower or "confirmation" in lower):
        return 451, "This video is age-restricted. Authentication (cookies) may be required."
    if "video unavailable" in lower or "has been removed" in lower:
        return 404, "This video is unavailable or has been removed."
    if "copyright" in lower:
        return 451, "This video is blocked due to a copyright claim in your region."
    if "premieres" in lower or "upcoming" in lower:
        return 425, "This video has not premiered yet."
    return 500, f"yt-dlp error: {stderr.strip()[:300]}"


async def _run_ytdlp(*args: str) -> str:
    """
    Run yt-dlp as a subprocess and return stdout.
    Raises YtDlpError on non-zero exit.
    """
    cmd = [sys.executable, "-m", "yt_dlp", *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise YtDlpError("yt-dlp subprocess failed", stderr.decode(errors="replace"))

    return stdout.decode(errors="replace")


async def _dump_json(video_id: str) -> dict[str, Any]:
    """Fetch full video metadata via --dump-json."""
    raw = await _run_ytdlp(
        "--dump-json",
        "--no-playlist",
        "--no-warnings",
        YT_URL_TEMPLATE.format(video_id=video_id),
    )
    return json.loads(raw)


async def _get_url(video_id: str, format_selector: str) -> str:
    """Fetch the direct stream URL via --get-url."""
    raw = await _run_ytdlp(
        "--get-url",
        "--no-playlist",
        "--no-warnings",
        "-f", format_selector,
        YT_URL_TEMPLATE.format(video_id=video_id),
    )
    # --get-url may return multiple lines (audio + video); return the first non-empty line
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        raise YtDlpError("yt-dlp returned no URL")
    return lines[0]


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class Thumbnail(BaseModel):
    url: str
    width: int | None = None
    height: int | None = None
    id: str | None = None


class Format(BaseModel):
    format_id: str
    ext: str
    resolution: str | None = None
    fps: float | None = None
    vcodec: str | None = None
    acodec: str | None = None
    filesize: int | None = None
    tbr: float | None = None          # total bitrate (kbps)
    url: str | None = None            # only populated if caller requests it


class MetadataResponse(BaseModel):
    video_id: str
    title: str
    description: str | None
    duration: int | None              # seconds
    upload_date: str | None           # YYYYMMDD
    uploader: str | None
    uploader_url: str | None
    view_count: int | None
    like_count: int | None
    thumbnails: list[Thumbnail]
    formats: list[Format]


class StreamResponse(BaseModel):
    video_id: str
    url: str
    format_id: str
    ext: str
    is_hls: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
async def root():
    return {"status": "ok", "service": "yt-dlp proxy"}


@app.get(
    "/metadata/{video_id}",
    response_model=MetadataResponse,
    summary="Get video metadata",
)
async def get_metadata(video_id: str):
    """
    Returns title, description, thumbnails, and all available formats
    for the given YouTube video ID.
    """
    try:
        data = await _dump_json(video_id)
    except YtDlpError as exc:
        code, msg = _classify_error(exc.stderr)
        raise HTTPException(status_code=code, detail=msg)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="yt-dlp returned invalid JSON.")

    # Parse thumbnails
    thumbnails: list[Thumbnail] = [
        Thumbnail(
            url=t.get("url", ""),
            width=t.get("width"),
            height=t.get("height"),
            id=t.get("id"),
        )
        for t in (data.get("thumbnails") or [])
        if t.get("url")
    ]

    # Parse formats — omit raw URLs by default (they expire quickly)
    formats: list[Format] = [
        Format(
            format_id=f.get("format_id", ""),
            ext=f.get("ext", ""),
            resolution=f.get("resolution") or (
                f"{f['width']}x{f['height']}"
                if f.get("width") and f.get("height") else None
            ),
            fps=f.get("fps"),
            vcodec=f.get("vcodec"),
            acodec=f.get("acodec"),
            filesize=f.get("filesize"),
            tbr=f.get("tbr"),
        )
        for f in (data.get("formats") or [])
    ]

    return MetadataResponse(
        video_id=video_id,
        title=data.get("title", ""),
        description=data.get("description"),
        duration=data.get("duration"),
        upload_date=data.get("upload_date"),
        uploader=data.get("uploader"),
        uploader_url=data.get("uploader_url"),
        view_count=data.get("view_count"),
        like_count=data.get("like_count"),
        thumbnails=thumbnails,
        formats=formats,
    )


@app.get(
    "/stream/{video_id}",
    response_model=StreamResponse,
    summary="Get direct stream URL",
)
async def get_stream(
    video_id: str,
    quality: str = Query(
        default="best",
        description=(
            "yt-dlp format selector. "
            "Examples: 'best', 'bestvideo+bestaudio', '22', 'bestvideo[height<=720]+bestaudio'"
        ),
    ),
    prefer_hls: bool = Query(
        default=True,
        description="Prefer HLS (m3u8) streams when available.",
    ),
):
    """
    Returns the direct playable URL for a given video.

    - Use `quality=best` for the highest available quality.
    - Use `quality=bestvideo[height<=1080]+bestaudio/best` for merged streams.
    - Set `prefer_hls=true` to prefer HLS/m3u8 (better for in-browser playback).
    """
    # Build format selector
    if prefer_hls:
        # Try HLS first, fall back to the caller's quality preference
        format_selector = f"({quality})[protocol^=m3u8]/({quality})[protocol=m3u8_native]/{quality}"
    else:
        format_selector = quality

    try:
        url = await _get_url(video_id, format_selector)
    except YtDlpError as exc:
        # If HLS preference caused a failure, retry without HLS filter
        if prefer_hls:
            try:
                url = await _get_url(video_id, quality)
            except YtDlpError as exc2:
                code, msg = _classify_error(exc2.stderr)
                raise HTTPException(status_code=code, detail=msg)
        else:
            code, msg = _classify_error(exc.stderr)
            raise HTTPException(status_code=code, detail=msg)

    is_hls = "m3u8" in url or ".m3u8" in url

    # Determine format_id and ext from the URL heuristically
    ext = "m3u8" if is_hls else ("mp4" if ".mp4" in url else "unknown")

    return StreamResponse(
        video_id=video_id,
        url=url,
        format_id=quality,
        ext=ext,
        is_hls=is_hls,
    )
