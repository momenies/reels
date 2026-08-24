"""HTTP API and the studio UI.

    uvicorn server.app:app --host 0.0.0.0 --port 8000

Uploads stream to disk in chunks rather than buffering — a two-hour podcast is
routinely a couple of gigabytes, and reading one into memory is how a worker
box with four other jobs on it dies.
"""

from __future__ import annotations

import asyncio
import io
import json
import mimetypes
import os
import re
import shutil
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    Depends, FastAPI, Form, HTTPException, Request, Response, UploadFile, File,
)
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from pipeline import __version__
from pipeline.captions import STYLES
from pipeline.config import ROOT, settings
from pipeline.log import get, setup

from .store import ACTIVE, CANCELED, QUEUED, UPLOADING, Store
from .worker import Worker, sweep

log = get("api")

WEB_DIR = ROOT / "web"
CHUNK = 4 * 1024 * 1024

#: Containers ffmpeg reads reliably and browsers actually produce.
ALLOWED_SUFFIXES = {
    ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".ts", ".flv",
    ".mp3", ".m4a", ".wav", ".aac",
}

state: dict = {}


def store() -> Store:
    return state["store"]


def jobs_dir() -> Path:
    return state["jobs_dir"]


# --- auth -------------------------------------------------------------------

def require_key(request: Request) -> None:
    """Optional shared-secret gate.

    Off by default so a local clone just works; set REELS_API_KEY and every
    /api route needs the header. This is deliberately not user accounts —
    it is the lock you put on a single-tenant deployment before it is on the
    open internet.
    """
    expected = settings().api_key
    if not expected:
        return
    got = request.headers.get("x-api-key") or request.query_params.get("key", "")
    if got != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


# --- lifecycle --------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup()
    cfg = settings(refresh=True)
    data_dir = Path(cfg.data_dir)
    state["jobs_dir"] = data_dir / "jobs"
    state["jobs_dir"].mkdir(parents=True, exist_ok=True)
    state["store"] = Store(data_dir / "jobs.db")
    state["worker"] = Worker(
        state["store"], state["jobs_dir"], concurrency=cfg.job_workers
    )
    state["worker"].start()
    state["sweeper"] = asyncio.create_task(_sweeper())
    log.info("data dir %s, %d worker(s)", data_dir, cfg.job_workers)
    try:
        yield
    finally:
        state["sweeper"].cancel()
        state["worker"].stop()


async def _sweeper() -> None:
    """Age out finished jobs once an hour."""
    cfg = settings()
    while True:
        try:
            await asyncio.sleep(3600)
            await asyncio.to_thread(sweep, store(), jobs_dir(), cfg.retention_hours)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("retention sweep failed")


app = FastAPI(
    title="reels-engine",
    version=__version__,
    description="Long video in, vertical captioned clips out.",
    lifespan=lifespan,
)


# --- meta -------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    ok = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    return {
        "ok": ok,
        "version": __version__,
        "ffmpeg": ok,
        "queue": store().counts() if "store" in state else {},
        "auth": bool(settings().api_key),
    }


@app.get("/api/styles")
def caption_styles() -> dict:
    return {
        "styles": [
            {"name": s.name, "label": s.label, "size": s.size,
             "primary": _css(s.primary), "highlight": _css(s.highlight),
             "uppercase": s.uppercase, "boxed": s.border_style == 3}
            for s in STYLES.values()
        ]
    }


def _css(ass_colour: str) -> str:
    """ASS is &HAABBGGRR; the UI needs #RRGGBB."""
    m = re.fullmatch(r"&H([0-9A-Fa-f]{2})?([0-9A-Fa-f]{6})", ass_colour)
    if not m:
        return "#ffffff"
    bgr = m.group(2)
    return f"#{bgr[4:6]}{bgr[2:4]}{bgr[0:2]}"


# --- jobs -------------------------------------------------------------------

@app.post("/api/jobs", dependencies=[Depends(require_key)])
async def create_job(
    file: UploadFile = File(...),
    lang: str = Form(""),
    clips: int = Form(5),
    min_score: int = Form(60),
    style: str = Form("pop"),
    faces: bool = Form(True),
    hook: bool = Form(False),
    model: str = Form(""),
) -> JSONResponse:
    name = Path(file.filename or "upload.mp4").name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"{suffix or 'that file type'} is not a video we can read. "
                   f"Try {', '.join(sorted(ALLOWED_SUFFIXES)[:6])}…",
        )

    options = {
        "lang": lang, "clips": clips, "min_score": min_score, "style": style,
        "faces": faces, "hook": hook, "model": model,
    }
    if style not in STYLES:
        raise HTTPException(status_code=400, detail=f"unknown caption style {style!r}")

    # UPLOADING, not QUEUED: a worker polls every second, and one that claims
    # the job before the bytes land runs ffprobe on a partial file. The
    # resulting "moov atom not found" is indistinguishable from a corrupt
    # upload, so the job is only published once the file is complete.
    job = store().create(
        filename=name, size_bytes=0, options=options, status=UPLOADING
    )
    dest_dir = jobs_dir() / job.id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"source{suffix}"
    partial = dest_dir / f"source{suffix}.part"

    limit = settings().max_upload_mb * 1024 * 1024
    written = 0
    try:
        with partial.open("wb") as out:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file is larger than the {settings().max_upload_mb} MB limit",
                    )
                out.write(chunk)
    except HTTPException:
        shutil.rmtree(dest_dir, ignore_errors=True)
        store().delete(job.id)
        raise
    except Exception:
        shutil.rmtree(dest_dir, ignore_errors=True)
        store().delete(job.id)
        log.exception("upload failed for job %s", job.id)
        raise HTTPException(status_code=500, detail="the upload could not be saved")
    finally:
        await file.close()

    if written == 0:
        shutil.rmtree(dest_dir, ignore_errors=True)
        store().delete(job.id)
        raise HTTPException(status_code=400, detail="the uploaded file is empty")

    # Rename last: until this line there is no `source.*` for a worker to find,
    # and after it the file is whole. Then publish the job to the queue.
    partial.replace(dest)
    store().update(job.id, size_bytes=written)
    store().mark_queued(job.id)
    log.info("job %s queued — %s (%.1f MB)", job.id, name, written / 1e6)
    fresh = store().get(job.id)
    return JSONResponse(fresh.to_dict(), status_code=201)


@app.get("/api/jobs", dependencies=[Depends(require_key)])
def list_jobs(limit: int = 50, offset: int = 0) -> dict:
    limit = max(1, min(200, limit))
    jobs = store().list(limit=limit, offset=max(0, offset))
    return {"jobs": [j.to_dict() for j in jobs], "counts": store().counts()}


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_key)])
def get_job(job_id: str) -> dict:
    return _job_or_404(job_id).to_dict()


@app.delete("/api/jobs/{job_id}", dependencies=[Depends(require_key)])
def delete_job(job_id: str) -> dict:
    job = _job_or_404(job_id)
    if job.status in ACTIVE and not store().cancel(job.id):
        # It is already running; ffmpeg owns the files until it is finished.
        raise HTTPException(
            status_code=409,
            detail="this job is already rendering — wait for it to finish, then delete it",
        )
    store().delete(job.id)
    shutil.rmtree(jobs_dir() / job.id, ignore_errors=True)
    return {"deleted": job.id}


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_key)])
def cancel_job(job_id: str) -> dict:
    job = _job_or_404(job_id)
    if job.status != QUEUED:
        raise HTTPException(status_code=409, detail=f"job is {job.status}, not queued")
    store().cancel(job.id)
    return {"status": CANCELED}


@app.get("/api/jobs/{job_id}/events", dependencies=[Depends(require_key)])
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    """Server-sent events: one frame per change, then the stream closes.

    Polling the store rather than pushing from the worker is on purpose —
    it means a browser that reconnects, or a second tab, sees the same
    truth without the worker knowing how many listeners exist.
    """
    _job_or_404(job_id)

    async def stream():
        last = None
        idle = 0.0
        while True:
            if await request.is_disconnected():
                return
            job = await asyncio.to_thread(store().get, job_id)
            if job is None:
                yield _sse({"error": "job no longer exists"}, event="gone")
                return
            payload = job.to_dict()
            if payload != last:
                last = payload
                idle = 0.0
                yield _sse(payload)
            if job.status not in ACTIVE:
                yield _sse({"status": job.status}, event="end")
                return
            idle += 0.5
            if idle >= 15:
                idle = 0.0
                yield ": keepalive\n\n"   # keeps proxies from closing the stream
            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse(data: dict, event: str | None = None) -> str:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data, ensure_ascii=False)}\n\n"


# --- files ------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/files/{name}", dependencies=[Depends(require_key)])
def job_file(job_id: str, name: str, request: Request):
    job = _job_or_404(job_id)
    path = _safe_output_path(job.id, name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{name} is not in this job")
    media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    # FileResponse handles Range itself, which is what lets the gallery scrub
    # a clip without downloading it whole.
    return FileResponse(path, media_type=media, headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/jobs/{job_id}/download", dependencies=[Depends(require_key)])
def download_all(job_id: str):
    """Every clip in the job as one zip, built on the fly."""
    job = _job_or_404(job_id)
    out_dir = jobs_dir() / job.id / "out"
    files = [out_dir / c["file"] for c in job.clips if (out_dir / c["file"]).exists()]
    if not files:
        raise HTTPException(status_code=404, detail="this job has no clips to download")

    def generate():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            # ZIP_STORED, not DEFLATE: H.264 does not compress, and deflating
            # a gigabyte of it just burns CPU the renderer needs.
            for path in files:
                zf.write(path, arcname=path.name)
            manifest = out_dir / "clips.json"
            if manifest.exists():
                zf.write(manifest, arcname="clips.json")
        buf.seek(0)
        while chunk := buf.read(CHUNK):
            yield chunk

    stem = Path(job.filename).stem[:40] or "clips"
    return StreamingResponse(
        generate(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{_ascii(stem)}-clips.zip"'},
    )


@app.get("/api/jobs/{job_id}/source", dependencies=[Depends(require_key)])
def job_source(job_id: str):
    """The original upload — the studio previews it while the job runs."""
    job = _job_or_404(job_id)
    for path in sorted((jobs_dir() / job.id).glob("source.*")):
        if path.is_file():
            return FileResponse(
                path, media_type=mimetypes.guess_type(path.name)[0] or "video/mp4"
            )
    raise HTTPException(status_code=404, detail="the source file is no longer on disk")


# --- helpers ----------------------------------------------------------------

def _job_or_404(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{6,32}", job_id or ""):
        raise HTTPException(status_code=404, detail="no such job")
    job = store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job


def _safe_output_path(job_id: str, name: str) -> Path:
    """Resolve a requested filename inside the job's output directory only.

    The name comes from a URL, so ``../../etc/passwd`` and an absolute path
    both have to bounce off this before touching the filesystem.
    """
    base = (jobs_dir() / job_id / "out").resolve()
    candidate = (base / name).resolve()
    # A strict descendant only: `..`, an absolute path and an empty name all
    # resolve onto or above the directory itself, and none of them names a clip.
    if base not in candidate.parents:
        raise HTTPException(status_code=404, detail="no such file")
    return candidate


def _ascii(text: str) -> str:
    """Content-Disposition is a latin-1 header; Arabic titles are not."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return cleaned or "clips"


# --- the studio -------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> Response:
    page = WEB_DIR / "index.html"
    if not page.exists():  # pragma: no cover - only if web/ was not shipped
        return HTMLResponse("<h1>reels-engine</h1><p>API is up. See /docs.</p>")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    icon = WEB_DIR / "favicon.svg"
    if icon.exists():
        return FileResponse(icon, media_type="image/svg+xml")
    return Response(status_code=204)
