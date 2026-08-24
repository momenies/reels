"""Job persistence.

SQLite rather than Redis: a single worker box is the shipping configuration,
and a job store that survives a restart without an extra service to operate
is worth more at this stage than horizontal scale. The schema and the access
pattern (one row per job, status transitions guarded by the database) port to
Postgres without changes when a second worker box shows up.

Every writer opens its own connection — SQLite objects are not safe to share
across threads, and the worker pool is threads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: The upload is still streaming in. Deliberately not claimable: a worker that
#: picks a job up mid-upload runs ffprobe on a partial file and fails it with
#: "moov atom not found", which looks exactly like a corrupt source.
UPLOADING = "uploading"
QUEUED = "queued"
PROCESSING = "processing"
DONE = "done"
ERROR = "error"
CANCELED = "canceled"

ACTIVE = (UPLOADING, QUEUED, PROCESSING)
TERMINAL = (DONE, ERROR, CANCELED)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    filename     TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL DEFAULT 0,
    stage        TEXT NOT NULL DEFAULT '',
    progress     REAL NOT NULL DEFAULT 0,
    message      TEXT NOT NULL DEFAULT '',
    options      TEXT NOT NULL DEFAULT '{}',
    clips        TEXT NOT NULL DEFAULT '[]',
    error        TEXT NOT NULL DEFAULT '',
    source_duration REAL NOT NULL DEFAULT 0,
    elapsed      REAL NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);
"""


@dataclass
class Job:
    id: str
    status: str
    filename: str
    size_bytes: int = 0
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    options: dict = field(default_factory=dict)
    clips: list = field(default_factory=list)
    error: str = ""
    source_duration: float = 0.0
    elapsed: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(
            id=row["id"],
            status=row["status"],
            filename=row["filename"],
            size_bytes=row["size_bytes"],
            stage=row["stage"],
            progress=row["progress"],
            message=row["message"],
            options=json.loads(row["options"] or "{}"),
            clips=json.loads(row["clips"] or "[]"),
            error=row["error"],
            source_duration=row["source_duration"],
            elapsed=row["elapsed"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "message": self.message,
            "options": self.options,
            "clips": self.clips,
            "error": self.error,
            "source_duration": self.source_duration,
            "elapsed": round(self.elapsed, 1),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
        }


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            # WAL lets the SSE readers poll while the worker writes progress.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    # --- writes ------------------------------------------------------------

    def create(
        self,
        filename: str,
        size_bytes: int,
        options: dict,
        status: str = QUEUED,
    ) -> Job:
        now = time.time()
        job = Job(
            id=uuid.uuid4().hex[:12],
            status=status,
            filename=filename,
            size_bytes=size_bytes,
            options=options,
            message="queued" if status == QUEUED else status,
            created_at=now,
            updated_at=now,
        )
        self._conn().execute(
            "INSERT INTO jobs (id, status, filename, size_bytes, options, message,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (job.id, job.status, job.filename, job.size_bytes,
             json.dumps(job.options, ensure_ascii=False), job.message, now, now),
        )
        return job

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        for key in ("options", "clips"):
            if key in fields:
                fields[key] = json.dumps(fields[key], ensure_ascii=False)
        fields["updated_at"] = time.time()
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._conn().execute(
            f"UPDATE jobs SET {sets} WHERE id = ?", (*fields.values(), job_id)
        )

    def mark_queued(self, job_id: str) -> bool:
        """Hand a finished upload to the workers. Only from UPLOADING."""
        return bool(
            self._conn().execute(
                "UPDATE jobs SET status = ?, message = ?, updated_at = ?"
                " WHERE id = ? AND status = ?",
                (QUEUED, "queued", time.time(), job_id, UPLOADING),
            ).rowcount
        )

    def claim_next(self) -> Job | None:
        """Atomically take the oldest queued job.

        The UPDATE ... WHERE status='queued' is the lock: two workers racing
        for the same row means one of them changes zero rows and moves on.
        """
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at LIMIT 1",
                (QUEUED,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            now = time.time()
            changed = conn.execute(
                "UPDATE jobs SET status = ?, started_at = ?, updated_at = ?,"
                " message = ? WHERE id = ? AND status = ?",
                (PROCESSING, now, now, "starting", row["id"], QUEUED),
            ).rowcount
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return self.get(row["id"]) if changed else None

    def finish(self, job_id: str, status: str, **fields: Any) -> None:
        self.update(job_id, status=status, finished_at=time.time(), **fields)

    def cancel(self, job_id: str) -> bool:
        """Only a job that has not started rendering can be cancelled cleanly.

        An upload still in flight counts: the request is abandoned and the
        partial file is removed with the job.
        """
        now = time.time()
        changed = self._conn().execute(
            "UPDATE jobs SET status = ?, finished_at = ?, updated_at = ?,"
            " message = ? WHERE id = ? AND status IN (?, ?)",
            (CANCELED, now, now, "canceled", job_id, QUEUED, UPLOADING),
        ).rowcount
        return bool(changed)

    def delete(self, job_id: str) -> bool:
        return bool(
            self._conn().execute("DELETE FROM jobs WHERE id = ?", (job_id,)).rowcount
        )

    def requeue_stale(self) -> int:
        """A process killed mid-job leaves rows stuck in 'processing'.

        On startup they go back in the queue: the source file is still on
        disk, and re-running is cheap because the transcript is cached.
        """
        changed = self._conn().execute(
            "UPDATE jobs SET status = ?, message = ?, progress = 0, stage = ''"
            " WHERE status = ?",
            (QUEUED, "requeued after restart", PROCESSING),
        ).rowcount
        # An upload interrupted by the restart left a truncated file behind;
        # there is nothing to resume, so fail it rather than feed it to ffprobe.
        self._conn().execute(
            "UPDATE jobs SET status = ?, message = ?, error = ?,"
            " finished_at = ?, updated_at = ? WHERE status = ?",
            (ERROR, "failed", "the upload was interrupted by a restart",
             time.time(), time.time(), UPLOADING),
        )
        return changed

    # --- reads -------------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        row = self._conn().execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return Job.from_row(row) if row else None

    def list(self, limit: int = 50, offset: int = 0) -> list[Job]:
        rows = self._conn().execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [Job.from_row(r) for r in rows]

    def counts(self) -> dict[str, int]:
        rows = self._conn().execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def older_than(self, cutoff: float) -> Iterable[Job]:
        rows = self._conn().execute(
            "SELECT * FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?",
            (cutoff,),
        ).fetchall()
        return [Job.from_row(r) for r in rows]
