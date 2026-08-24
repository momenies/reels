"""Background job runner.

A pool of threads pulls queued jobs out of SQLite and runs the pipeline on
them. Threads rather than processes because the expensive stages — ffmpeg and
CTranslate2 — release the GIL and spend their time in native code; the Python
frames are only bookkeeping.

Progress is written straight to the job row. That is a write every few hundred
milliseconds per job, which SQLite in WAL mode absorbs without complaint, and
it means progress survives a page refresh, a reconnect, or a second browser
tab, none of which an in-memory queue gives you.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

from pipeline.captions import DEFAULT_STYLE
from pipeline.log import get
from pipeline.reframe import ProbeError
from pipeline.run import Options, process
from pipeline.score import ScoringError
from pipeline.transcribe import TranscriptionError

from .store import DONE, ERROR, Store

log = get("worker")

#: Progress rows are cheap but not free; this is how often one job is allowed
#: to write. Fine-grained enough that a 24-clip render still animates.
WRITE_INTERVAL = 0.4

#: How the five pipeline stages map onto one 0-1 bar for the customer.
#: Transcription dominates wall clock on CPU, so it owns most of the bar.
STAGE_WEIGHTS = (
    ("probe", 0.02),
    ("transcribe", 0.55),
    ("score", 0.08),
    ("render", 0.35),
)


def overall_progress(stage: str, pct: float) -> float:
    """Collapse (stage, fraction) into a single monotonic 0-1 value."""
    if stage == "done":
        return 1.0
    before = 0.0
    for name, weight in STAGE_WEIGHTS:
        if name == stage:
            return min(1.0, before + weight * max(0.0, min(1.0, pct)))
        before += weight
    return min(1.0, before)


def options_from(raw: dict) -> Options:
    """Build pipeline options from whatever the API was handed.

    Values are clamped rather than rejected: a customer who asks for 500 clips
    should get 20, not a validation error.
    """
    def num(key, default, lo, hi, cast=int):
        try:
            return max(lo, min(hi, cast(raw.get(key, default))))
        except (TypeError, ValueError):
            return default

    lang = (raw.get("lang") or "").strip().lower() or None
    if lang in ("auto", "detect"):
        lang = None

    return Options(
        lang=lang,
        clips=num("clips", 5, 1, 20),
        min_score=num("min_score", 60, 0, 100),
        model=(raw.get("model") or None),
        faces=bool(raw.get("faces", True)),
        style=(raw.get("style") or DEFAULT_STYLE),
        hook=bool(raw.get("hook", False)),
    )


class Worker:
    """Owns the worker threads and the polling loop."""

    def __init__(self, store: Store, jobs_dir: Path, concurrency: int = 1,
                 poll_seconds: float = 1.0):
        self.store = store
        self.jobs_dir = Path(jobs_dir)
        self.concurrency = max(1, concurrency)
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        requeued = self.store.requeue_stale()
        if requeued:
            log.info("requeued %d job(s) interrupted by a restart", requeued)
        for i in range(self.concurrency):
            t = threading.Thread(target=self._loop, name=f"reels-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("started %d worker thread(s)", self.concurrency)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    # --- the loop ----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.store.claim_next()
            except Exception:
                log.exception("could not claim a job")
                job = None
            if job is None:
                self._stop.wait(self.poll_seconds)
                continue
            try:
                self.run_job(job.id)
            except Exception:
                log.exception("job %s crashed outside its own error handling", job.id)
                self.store.finish(job.id, ERROR, error="internal error", message="failed")

    def run_job(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        job_dir = self.jobs_dir / job_id
        source = _find_source(job_dir)
        if source is None:
            self.store.finish(
                job_id, ERROR, message="failed",
                error="the uploaded file is no longer on disk",
            )
            return

        last_write = [0.0]

        def on_progress(stage: str, pct: float, message: str) -> None:
            now = time.monotonic()
            final = stage == "done" or pct >= 1.0
            if not final and now - last_write[0] < WRITE_INTERVAL:
                return
            last_write[0] = now
            self.store.update(
                job_id,
                stage=stage,
                progress=overall_progress(stage, pct),
                message=message,
            )

        started = time.time()
        try:
            result = process(
                source, job_dir / "out", options_from(job.options),
                on_progress=on_progress,
            )
        except (ProbeError, TranscriptionError, ScoringError) as exc:
            # The three failures a customer can actually act on.
            log.warning("job %s failed: %s", job_id, exc)
            self.store.finish(job_id, ERROR, message="failed", error=str(exc),
                              elapsed=time.time() - started)
            return
        except Exception as exc:
            log.exception("job %s failed unexpectedly", job_id)
            self.store.finish(job_id, ERROR, message="failed",
                              error=f"unexpected error: {exc}",
                              elapsed=time.time() - started)
            return

        if not result.clips:
            self.store.finish(
                job_id, DONE, message="no clip cleared the bar", progress=1.0,
                stage="done", clips=[], elapsed=result.seconds,
                source_duration=result.duration,
                error="; ".join(result.errors),
            )
            return

        self.store.finish(
            job_id, DONE,
            message=f"{len(result.clips)} clip(s) ready",
            progress=1.0,
            stage="done",
            clips=result.clips,
            elapsed=result.seconds,
            source_duration=result.duration,
            error="; ".join(result.errors),
        )
        log.info("job %s done — %d clip(s) in %.0fs", job_id, len(result.clips), result.seconds)


def _find_source(job_dir: Path) -> Path | None:
    if not job_dir.exists():
        return None
    for path in sorted(job_dir.glob("source.*")):
        # `.part` is an upload still in flight — never hand one to ffprobe.
        if path.is_file() and path.suffix != ".part":
            return path
    return None


def sweep(store: Store, jobs_dir: Path, retention_hours: int) -> int:
    """Delete finished jobs and their media once they age out.

    Video is the expensive thing to keep. Without this the disk fills, and a
    full disk fails jobs in ways that look like pipeline bugs.
    """
    if retention_hours <= 0:
        return 0
    cutoff = time.time() - retention_hours * 3600
    removed = 0
    for job in store.older_than(cutoff):
        shutil.rmtree(Path(jobs_dir) / job.id, ignore_errors=True)
        store.delete(job.id)
        removed += 1
    if removed:
        log.info("swept %d job(s) older than %dh", removed, retention_hours)
    return removed
