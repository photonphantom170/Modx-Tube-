"""
yt-dlp FastAPI Backend — ModxTube
Endpoints:
  GET /                          - health check
  GET /metadata/{video_id}       - video metadata via yt-dlp
  GET /stream/{video_id}         - direct stream URL via yt-dlp
  GET /search?q=...              - search videos (proxies Piped)
  GET /trending                  - trending videos (proxies Piped)
  GET /comments/{video_id}       - video comments (proxies Piped)
  GET /streams/{video_id}        - video details/recs (proxies Piped)
  GET /channel/{channel_id}      - channel info (proxies Piped)
"""

import asyncio
import json
import sys
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="ModxTube API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

YT_URL_TEMPLATE = "https://www.youtube.com/watch?v={video_id}"

PIPED_INSTANCES = [
    "https://pipedapi.kavin.rocks",
    "https://piped-api.garudalinux.org",
    "https://api.piped.projectsegfau.lt",
    "https://pipedapi.adminforge.de",
    "https://api.piped.yt",
    "https://pipedapi.tokhmi.xyz",
    "https://piped.video/api",
    "https://pipedapi.leptons.xyz",
]

# ---------------------------------------------------------------------------
# Piped proxy helper  (server-side — no CORS issues)
# ---------------------------------------------------------------------------

async def piped_fetch(path: str) -> Any:
    """Try each Piped instance until one succeeds. Runs server-side so no CORS."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        last_err: Exception | None = None
        for base in PIPED_INSTANCES:
            try:
                r = await client.get(
                    base + path,
                    headers={"Accept": "application/json"},
                    follow_redirects=True,
                )
                if r.status_code == 200:
                    return r.json()
                last_err = Exception(f"HTTP {r.status_code} from {base}")
            except Exception as e:
                last_err = e
                continue
    raise HTTPException(status_code=502, detail=f"All Piped instances failed: {last_err}")


# ---------------------------------------------------------------------------
# yt-dlp helpers
# ---------------------------------------------------------------------------

class YtDlpError(Exception):
    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


def _classify_error(stderr: str) -> tuple[int, str]:
    lower = stderr.lower()
    if "private video" in lower:
        return 403, "This video is private."
    if "age" in lower and ("restricted" in lower or "confirmation" in lower):
        return 451, "This video is age-restricted."
    if "video unavailable" in lower or "has been removed" in lower:
        return 404, "This video is unavailable or has been removed."
    if "copyright" in lower:
        return 451, "This video is blocked due to copyright."
    if "premieres" in lower or "upcoming" in lower:
        return 425, "This video has not premiered yet."
    return 500, f"yt-dlp error: {stderr.strip()[:300]}"


async def _run_ytdlp(*args: str) -> str:
    cmd = [sys.executable, "-m", "yt_dlp", *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise YtDlpError("yt-dlp failed", stderr.decode(errors="replace"))
    return stdout.decode(errors="replace")


async def _dump_json(video_id: str) -> dict[str, Any]:
    raw = await _run_ytdlp(
        "--dump-json", "--no-playlist", "--no-warnings",
        YT_URL_TEMPLATE.format(video_id=video_id),
    )
    return json.loads(raw)


async def _get_url(video_id: str, fmt: str) -> str:
    raw = await _run_ytdlp(
        "--get-url", "--no-playlist", "--no-warnings",
        "-f", fmt, YT_URL_TEMPLATE.format(video_id=video_id),
    )
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        raise YtDlpError("yt-dlp returned no URL")
    return lines[0]


# ---------------------------------------------------------------------------
# Pydantic models
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
    tbr: float | None = None


class MetadataResponse(BaseModel):
    video_id: str
    title: str
    description: str | None
    duration: int | None
    upload_date: str | None
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

@app.get("/")
async def root():
    return {"status": "ok", "service": "ModxTube API"}


@app.get("/metadata/{video_id}", response_model=MetadataResponse)
async def get_metadata(video_id: str):
    try:
        data = await _dump_json(video_id)
    except YtDlpError as exc:
        code, msg = _classify_error(exc.stderr)
        raise HTTPException(status_code=code, detail=msg)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="yt-dlp returned invalid JSON.")

    thumbnails = [
        Thumbnail(url=t.get("url",""), width=t.get("width"), height=t.get("height"), id=t.get("id"))
        for t in (data.get("thumbnails") or []) if t.get("url")
    ]
    formats = [
        Format(
            format_id=f.get("format_id",""), ext=f.get("ext",""),
            resolution=f.get("resolution") or (
                f"{f['width']}x{f['height']}" if f.get("width") and f.get("height") else None
            ),
            fps=f.get("fps"), vcodec=f.get("vcodec"), acodec=f.get("acodec"),
            filesize=f.get("filesize"), tbr=f.get("tbr"),
        )
        for f in (data.get("formats") or [])
    ]
    return MetadataResponse(
        video_id=video_id,
        title=data.get("title",""),
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


@app.get("/stream/{video_id}", response_model=StreamResponse)
async def get_stream(
    video_id: str,
    quality: str = Query(default="best"),
    prefer_hls: bool = Query(default=True),
):
    fmt = (
        f"({quality})[protocol^=m3u8]/({quality})[protocol=m3u8_native]/{quality}"
        if prefer_hls else quality
    )
    try:
        url = await _get_url(video_id, fmt)
    except YtDlpError as exc:
        if prefer_hls:
            try:
                url = await _get_url(video_id, quality)
            except YtDlpError as exc2:
                code, msg = _classify_error(exc2.stderr)
                raise HTTPException(status_code=code, detail=msg)
        else:
            code, msg = _classify_error(exc.stderr)
            raise HTTPException(status_code=code, detail=msg)

    is_hls = "m3u8" in url
    ext = "m3u8" if is_hls else ("mp4" if ".mp4" in url else "unknown")
    return StreamResponse(video_id=video_id, url=url, format_id=quality, ext=ext, is_hls=is_hls)


# ── Piped proxy routes (browser calls these; we call Piped server-side) ──────

@app.get("/search")
async def search(
    q: str = Query(...),
    nextpage: str = Query(default=""),
):
    path = f"/search?q={quote(q)}&filter=videos"
    if nextpage:
        path += f"&nextpage={quote(nextpage)}"
    return await piped_fetch(path)


@app.get("/trending")
async def trending(region: str = Query(default="US")):
    return await piped_fetch(f"/trending?region={region}")


@app.get("/comments/{video_id}")
async def comments(video_id: str):
    return await piped_fetch(f"/comments/{video_id}")


@app.get("/streams/{video_id}")
async def streams_piped(video_id: str):
    return await piped_fetch(f"/streams/{video_id}")


@app.get("/channel/{channel_id}/playlists")
async def channel_playlists(channel_id: str):
    return await piped_fetch(f"/channel/{channel_id}/playlists")


@app.get("/channel/{channel_id}/videos")
async def channel_videos(channel_id: str, nextpage: str = Query(default="")):
    path = f"/nextpage/channel/{channel_id}"
    if nextpage:
        path += f"?nextpage={quote(nextpage)}"
    return await piped_fetch(path)


@app.get("/channel/{channel_id}")
async def channel(channel_id: str):
    return await piped_fetch(f"/channel/{channel_id}")
