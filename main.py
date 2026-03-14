"""
ModxTube Backend — v3.0
- /search    → yt-dlp (primary) with Piped fallback
- /trending  → yt-dlp (primary) with Piped fallback
- /metadata  → yt-dlp only
- /stream    → yt-dlp only
- /comments  → Piped (yt-dlp has no comment support)
- /streams   → Piped (for recommendations)
- /channel   → Piped (for channel pages)
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

app = FastAPI(title="ModxTube API", version="3.0.0")

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
        return 451, "Blocked due to copyright."
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


def _parse_ytdlp_entry(entry: dict) -> dict:
    """Convert a yt-dlp flat playlist entry into the same shape as a Piped item."""
    vid_id = entry.get("id", "")
    dur = entry.get("duration") or 0
    mn, sc = divmod(int(dur), 60)
    hr, mn = divmod(mn, 60)
    dur_fmt = (
        f"{hr}:{mn:02d}:{sc:02d}" if hr else f"{mn}:{sc:02d}"
    ) if dur else ""
    upload_ts = entry.get("timestamp") or entry.get("release_timestamp")
    uploaded = str(upload_ts * 1000) if upload_ts else ""
    return {
        "url": f"/watch?v={vid_id}",
        "title": entry.get("title", ""),
        "uploaderName": entry.get("uploader") or entry.get("channel", ""),
        "uploaderUrl": f"/channel/{entry.get('channel_id', '')}",
        "thumbnail": (
            entry.get("thumbnail")
            or f"https://i.ytimg.com/vi/{vid_id}/mqdefault.jpg"
        ),
        "duration": dur,
        "views": entry.get("view_count") or 0,
        "uploaded": uploaded,
        "shortDescription": entry.get("description", ""),
    }


# ---------------------------------------------------------------------------
# Piped proxy helper  (fallback only — server-side, no CORS)
# ---------------------------------------------------------------------------

async def piped_fetch(path: str) -> Any:
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
                last_err = Exception(f"HTTP {r.status_code}")
            except Exception as e:
                last_err = e
        raise HTTPException(
            status_code=502,
            detail=f"All Piped instances failed: {last_err}",
        )


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
# Routes — health
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "service": "ModxTube API", "version": "3.0.0"}


# ---------------------------------------------------------------------------
# Routes — yt-dlp metadata & stream
# ---------------------------------------------------------------------------

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
        Thumbnail(url=t.get("url", ""), width=t.get("width"),
                  height=t.get("height"), id=t.get("id"))
        for t in (data.get("thumbnails") or []) if t.get("url")
    ]
    formats = [
        Format(
            format_id=f.get("format_id", ""), ext=f.get("ext", ""),
            resolution=f.get("resolution") or (
                f"{f['width']}x{f['height']}"
                if f.get("width") and f.get("height") else None
            ),
            fps=f.get("fps"), vcodec=f.get("vcodec"), acodec=f.get("acodec"),
            filesize=f.get("filesize"), tbr=f.get("tbr"),
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
    return StreamResponse(video_id=video_id, url=url,
                          format_id=quality, ext=ext, is_hls=is_hls)


# ---------------------------------------------------------------------------
# Routes — SEARCH via yt-dlp (primary) + Piped fallback
# ---------------------------------------------------------------------------

@app.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    nextpage: str = Query(default="", description="Pagination token (Piped format)"),
):
    """
    Search YouTube. Uses yt-dlp as primary source.
    Falls back to Piped if yt-dlp fails or nextpage token is provided
    (yt-dlp doesn't support Piped-style pagination tokens).
    """
    # If a Piped nextpage token is given, delegate to Piped directly
    if nextpage:
        try:
            path = f"/search?q={quote(q)}&filter=videos&nextpage={quote(nextpage)}"
            return await piped_fetch(path)
        except HTTPException:
            return {"items": [], "nextpage": None}

    # Primary: yt-dlp search
    try:
        raw = await _run_ytdlp(
            "--dump-json",
            "--flat-playlist",
            "--no-warnings",
            "--playlist-end", "20",
            f"ytsearch20:{q}",
        )
        entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
        items = [_parse_ytdlp_entry(e) for e in entries if e.get("id")]
        if items:
            return {"items": items, "nextpage": None}
    except Exception:
        pass

    # Fallback: Piped
    try:
        return await piped_fetch(f"/search?q={quote(q)}&filter=videos")
    except HTTPException:
        return {"items": [], "nextpage": None}


# ---------------------------------------------------------------------------
# Routes — TRENDING via yt-dlp (primary) + Piped fallback
# ---------------------------------------------------------------------------

@app.get("/trending")
async def trending(region: str = Query(default="US")):
    """
    Trending videos. Uses yt-dlp as primary source.
    Falls back to Piped if yt-dlp fails.
    """
    # Primary: yt-dlp — YouTube trending playlist
    # YT trending playlist IDs by region: PLrEnWoR732-BHrPp_Pm8_VleD68f9s14-  (global)
    # We use ytsearch with a broad trending query as yt-dlp has no native trending
    try:
        raw = await _run_ytdlp(
            "--dump-json",
            "--flat-playlist",
            "--no-warnings",
            "--playlist-end", "30",
            # YouTube's trending page as a playlist
            "https://www.youtube.com/feed/trending",
        )
        entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
        items = [_parse_ytdlp_entry(e) for e in entries if e.get("id")]
        if items:
            return items  # Piped returns a plain array for trending
    except Exception:
        pass

    # Fallback: Piped
    try:
        return await piped_fetch(f"/trending?region={region}")
    except HTTPException:
        return []


# ---------------------------------------------------------------------------
# Routes — Piped proxy (comments, recommendations, channel pages)
# ---------------------------------------------------------------------------

@app.get("/comments/{video_id}")
async def comments(video_id: str):
    """Video comments — Piped only (yt-dlp has no comment support)."""
    try:
        return await piped_fetch(f"/comments/{video_id}")
    except HTTPException:
        return {"comments": [], "nextpage": None}


@app.get("/streams/{video_id}")
async def streams_piped(video_id: str):
    """Video details + recommendations via Piped."""
    try:
        return await piped_fetch(f"/streams/{video_id}")
    except HTTPException:
        return {"relatedStreams": [], "title": "", "description": ""}


@app.get("/channel/{channel_id}/playlists")
async def channel_playlists(channel_id: str):
    try:
        return await piped_fetch(f"/channel/{channel_id}/playlists")
    except HTTPException:
        return {"playlists": []}


@app.get("/channel/{channel_id}/videos")
async def channel_videos(channel_id: str, nextpage: str = Query(default="")):
    try:
        path = f"/nextpage/channel/{channel_id}"
        if nextpage:
            path += f"?nextpage={quote(nextpage)}"
        return await piped_fetch(path)
    except HTTPException:
        return {"relatedStreams": [], "nextpage": None}


@app.get("/channel/{channel_id}")
async def channel(channel_id: str):
    try:
        return await piped_fetch(f"/channel/{channel_id}")
    except HTTPException:
        return {"name": "", "description": "", "relatedStreams": []}
