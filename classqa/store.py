"""Persistence behind small interfaces so the backend can be swapped.

Prototype: one SQLite file (WAL mode, thread-safe via a lock) holding
  jobs         - the job state machine
  checkpoints  - idempotent per-stage outputs keyed by (content_hash, stage, version)
  quota        - per-day, per-model request counters for the daily quota guard
Production: jobs -> Postgres, checkpoints -> Postgres/object storage with lifecycle TTL,
quota -> Redis INCR with expiry (see DESIGN.md)."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class JobStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    TRANSCRIBING = "transcribing"
    SCORING = "scoring"
    DONE = "done"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.NEEDS_REVIEW, JobStatus.FAILED)


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    source_type: str = "path"          # upload | link | path
    source_ref: str = ""               # file name or URL (never file contents)
    content_hash: str | None = None
    progress: float = 0.0
    message: str = "Queued"
    error_code: str | None = None
    error_message: str | None = None
    result: dict | None = None
    usage: dict = field(default_factory=dict)
    review_reasons: list[str] = field(default_factory=list)
    options: dict = field(default_factory=dict)
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @staticmethod
    def new(**kw) -> "Job":
        return Job(id=uuid.uuid4().hex[:12], **kw)


class JobStore(Protocol):
    def create(self, job: Job) -> Job: ...
    def get(self, job_id: str) -> Job | None: ...
    def update(self, job_id: str, **fields: Any) -> Job: ...
    def list(self, limit: int = 200) -> list[Job]: ...


class CheckpointStore(Protocol):
    def get_checkpoint(self, content_hash: str, stage: str, version: str) -> Any | None: ...
    def put_checkpoint(self, content_hash: str, stage: str, version: str, payload: Any) -> None: ...
    def purge_older_than(self, seconds: float) -> int: ...


class QuotaStore(Protocol):
    def incr(self, model: str) -> int: ...
    def count(self, model: str | None = None) -> int: ...


_JSON_FIELDS = {"result", "usage", "review_reasons", "options"}


def default_db_path() -> Path:
    return Path(os.environ.get("CLASSQA_DB", Path(__file__).resolve().parent.parent / ".cache" / "classqa.db"))


class SQLiteStore(JobStore, CheckpointStore, QuotaStore):
    def __init__(self, path: str | Path | None = None):
        self.path = str(path or default_db_path())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, status TEXT, source_type TEXT, source_ref TEXT, content_hash TEXT,
              progress REAL, message TEXT, error_code TEXT, error_message TEXT, result TEXT, usage TEXT,
              review_reasons TEXT, options TEXT, attempts INTEGER, created_at REAL, updated_at REAL);
            CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
            CREATE TABLE IF NOT EXISTS checkpoints (
              content_hash TEXT, stage TEXT, version TEXT, payload TEXT, created_at REAL,
              PRIMARY KEY (content_hash, stage, version));
            CREATE TABLE IF NOT EXISTS quota (day TEXT, model TEXT, n INTEGER, PRIMARY KEY (day, model));
            """)

    # ---- jobs
    def _row(self, r: sqlite3.Row) -> Job:
        d = dict(r)
        for k in _JSON_FIELDS:
            d[k] = json.loads(d[k]) if d[k] else ({} if k in ("usage", "options") else ([] if k == "review_reasons" else None))
        d["status"] = JobStatus(d["status"])
        return Job(**d)

    def create(self, job: Job) -> Job:
        d = asdict(job)
        d["status"] = job.status.value
        for k in _JSON_FIELDS:
            d[k] = json.dumps(d[k]) if d[k] is not None else None
        with self._lock:
            self._db.execute(f"INSERT INTO jobs ({','.join(d)}) VALUES ({','.join('?' * len(d))})", list(d.values()))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(r) if r else None

    def update(self, job_id: str, **fields: Any) -> Job:
        fields["updated_at"] = time.time()
        vals = []
        for k, v in fields.items():
            if k == "status":
                v = JobStatus(v).value
            elif k in _JSON_FIELDS:
                v = json.dumps(v) if v is not None else None
            vals.append(v)
        with self._lock:
            self._db.execute(f"UPDATE jobs SET {','.join(f'{k}=?' for k in fields)} WHERE id=?", [*vals, job_id])
        return self.get(job_id)  # type: ignore[return-value]

    def list(self, limit: int = 200) -> list[Job]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    # ---- checkpoints
    def get_checkpoint(self, content_hash: str, stage: str, version: str) -> Any | None:
        with self._lock:
            r = self._db.execute("SELECT payload FROM checkpoints WHERE content_hash=? AND stage=? AND version=?",
                                 (content_hash, stage, version)).fetchone()
        return json.loads(r["payload"]) if r else None

    def put_checkpoint(self, content_hash: str, stage: str, version: str, payload: Any) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?)",
                             (content_hash, stage, version, json.dumps(payload, default=str), time.time()))

    def purge_older_than(self, seconds: float) -> int:
        with self._lock:
            cur = self._db.execute("DELETE FROM checkpoints WHERE created_at < ?", (time.time() - seconds,))
        return cur.rowcount

    # ---- quota
    @staticmethod
    def _day() -> str:
        # Gemini free-tier quotas reset at midnight Pacific; UTC day is a close-enough conservative proxy
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def incr(self, model: str) -> int:
        with self._lock:
            self._db.execute("INSERT INTO quota VALUES (?,?,1) ON CONFLICT(day, model) DO UPDATE SET n=n+1",
                             (self._day(), model))
            return self._db.execute("SELECT n FROM quota WHERE day=? AND model=?", (self._day(), model)).fetchone()["n"]

    def count(self, model: str | None = None) -> int:
        with self._lock:
            if model:
                r = self._db.execute("SELECT n FROM quota WHERE day=? AND model=?", (self._day(), model)).fetchone()
                return r["n"] if r else 0
            r = self._db.execute("SELECT COALESCE(SUM(n),0) AS n FROM quota WHERE day=?", (self._day(),)).fetchone()
            return r["n"]
