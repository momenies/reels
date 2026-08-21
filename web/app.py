"""A local web UI for the pipeline: drop a video in, get clips out.

Deliberately a single-worker queue rather than a job runner. One clip render
saturates the CPU already, so running two at once only makes both slower, and
a single worker means the captured pipeline output belongs to exactly one job.

This is the local tool, not the product. It binds to 127.0.0.1, trusts the
person at the keyboard, and keeps every job in its own directory under
``jobs/``. Do not expose it to a network you do not control: it accepts
uploads and spends money on the Claude call for every one of them.
"""

from __future__ import annotations

import contextlib
import io
import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline import fetch
from pipeline import run as pipeline_run

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "jobs"
STATIC = Path(__file__).resolve().parent / "static"

UPLOAD_CHUNK = 1 << 20
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


@dataclass
class Job:
    id: str
    name: str
    options: dict
    url: str = ""
    state: str = "queued"          # queued | running | done | failed | empty
    log: list[str] = field(default_factory=list)
    clips: list[dict] = field(default_factory=list)
    error: str = ""
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "state": self.state,
            "options": self.options,
            "log": self.log,
            "clips": self.clips,
            "error": self.error,
            "elapsed": round((self.finished or time.time()) - (self.started or self.created), 1),
        }


JOBS: dict[str, Job] = {}
ORDER: list[str] = []
WORK: "queue.Queue[str]" = queue.Queue()
LOCK = threading.Lock()


class _LogStream(io.TextIOBase):
    """Collects the pipeline's prints into a job's log, line by line."""

    def __init__(self, job: Job):
        self.job = job
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                with LOCK:
                    self.job.log.append(line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            with LOCK:
                self.job.log.append(self._buf.strip())
            self._buf = ""


def _worker() -> None:
    while True:
        job_id = WORK.get()
        job = JOBS.get(job_id)
        if job is None:
            continue
        job.state, job.started = "running", time.time()
        stream = _LogStream(job)
        try:
            with contextlib.redirect_stdout(stream):
                source = job.dir / "source" / job.name
                if job.url:
                    source = fetch.download(job.url, job.dir / "source")
                    job.name = source.name
                clips = pipeline_run.process(
                    source,
                    job.dir / "out",
                    lang=job.options["lang"] or None,
                    clips=job.options["clips"],
                    min_score=job.options["min_score"],
                    model=job.options["model"],
                    faces=job.options["faces"],
                )
            stream.flush()
            job.clips = clips
            job.state = "done" if clips else "empty"
        except Exception as exc:
            stream.flush()
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            with LOCK:
                job.log.extend(traceback.format_exc().strip().splitlines()[-12:])
        finally:
            job.finished = time.time()
            WORK.task_done()


app = FastAPI(title="reels-engine")


@app.on_event("startup")
def _start() -> None:
    JOBS_DIR.mkdir(exist_ok=True)
    threading.Thread(target=_worker, daemon=True).start()


@app.post("/api/jobs")
async def create_job(
    video: UploadFile | None = None,
    url: str = Form(""),
    lang: str = Form(""),
    clips: int = Form(5),
    min_score: int = Form(60),
    model: str = Form("small"),
    faces: bool = Form(True),
) -> JSONResponse:
    url = url.strip()
    if url and video is not None and video.filename:
        raise HTTPException(400, "send a file or a link, not both")
    if not url and (video is None or not video.filename):
        raise HTTPException(400, "no video: upload a file or paste a link")

    if url:
        if not url.startswith(("http://", "https://")):
            raise HTTPException(400, "the link must start with http:// or https://")
        if not fetch.available():
            raise HTTPException(400, "yt-dlp is not installed: "
                                     "pip install -r requirements-web.txt")
        name = url
    else:
        name = Path(video.filename).name
        if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
            raise HTTPException(400, f"{name!r} is not a video file "
                                     f"({', '.join(sorted(VIDEO_SUFFIXES))})")

    job = Job(
        id=uuid.uuid4().hex[:12],
        name=name,
        url=url,
        options={"lang": lang.strip(), "clips": max(1, min(clips, 10)),
                 "min_score": max(0, min(min_score, 100)), "model": model,
                 "faces": bool(faces)},
    )
    src = job.dir / "source"
    src.mkdir(parents=True, exist_ok=True)
    if not url:
        with open(src / name, "wb") as fh:      # stream: videos do not fit in RAM
            while chunk := await video.read(UPLOAD_CHUNK):
                fh.write(chunk)

    JOBS[job.id] = job
    ORDER.insert(0, job.id)
    WORK.put(job.id)
    return JSONResponse(job.as_dict())


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    return JSONResponse([
        {k: v for k, v in JOBS[i].as_dict().items() if k != "log"}
        for i in ORDER if i in JOBS
    ])


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return JSONResponse(job.as_dict())


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.state in {"queued", "running"}:
        raise HTTPException(409, "job is still running")
    shutil.rmtree(job.dir, ignore_errors=True)
    JOBS.pop(job_id, None)
    if job_id in ORDER:
        ORDER.remove(job_id)
    return JSONResponse({"deleted": job_id})


@app.get("/media/{job_id}/{filename}")
def media(job_id: str, filename: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    # resolve and confine: never serve outside this job's output directory
    base = (job.dir / "out").resolve()
    path = (base / Path(filename).name).resolve()
    if not path.is_file() or base not in path.parents:
        raise HTTPException(404, "no such file")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="reels-engine web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"reels-engine UI -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
