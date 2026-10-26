"""Job runner: a background worker thread that pulls jobs from an in-process queue,
runs the staged pipeline, persists status/progress/usage, and always cleans up media.

On Streamlit Community Cloud (single container) this runs inside the app process;
in production the same `process()` function is the body of a Cloud Tasks / Celery
worker (see DESIGN.md)."""
from __future__ import annotations

import logging
import os
import queue
import shutil
import threading
import time
import traceback
from pathlib import Path

from . import media, obs
from .errors import ClassQAError
from .llm import LLMClient
from .pipeline import JobContext, run
from .store import Job, JobStatus, SQLiteStore

log = logging.getLogger("classqa.worker")

DATA_DIR = Path(os.environ.get("CLASSQA_DATA_DIR", Path(__file__).resolve().parent.parent / ".cache"))
CHECKPOINT_TTL_S = float(os.environ.get("CLASSQA_CHECKPOINT_TTL_S", str(7 * 24 * 3600)))


def job_dir(job_id: str) -> Path:
    d = DATA_DIR / "jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


class JobRunner:
    def __init__(self, store: SQLiteStore | None = None, llm: LLMClient | None = None, *, workers: int = 1,
                 start: bool = True):
        obs.setup_logging()
        self.store = store or SQLiteStore()
        self.llm = llm or LLMClient(quota=self.store)
        self.q: queue.Queue[str] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        purged = self.store.purge_older_than(CHECKPOINT_TTL_S)
        if purged:
            obs.event(log, "checkpoints_purged", n=purged)
        self._recover()
        if start:
            for i in range(workers):
                t = threading.Thread(target=self._loop, name=f"classqa-worker-{i}", daemon=True)
                t.start()
                self._threads.append(t)

    def _recover(self) -> None:
        """Jobs left running by a crashed/restarted process are failed explicitly (media is gone)."""
        stale = time.time() - float(os.environ.get("CLASSQA_STALE_JOB_S", "900"))
        for j in self.store.list(500):
            if not j.status.terminal and j.updated_at < stale:
                self.store.update(j.id, status=JobStatus.FAILED, error_code="worker_restarted",
                                  error_message="The app restarted while this job was running. Retry it - "
                                                "finished stages are reused.")
                shutil.rmtree(DATA_DIR / "jobs" / j.id, ignore_errors=True)

    # ---- API used by UI / CLI
    def submit(self, *, source_type: str, source_ref: str, media_path: Path | None = None,
               vtt_text: str | None = None, options: dict | None = None, job_id: str | None = None) -> Job:
        job = Job.new(source_type=source_type, source_ref=source_ref, options=options or {})
        if job_id:
            job.id = job_id
        self.store.create(job)
        d = job_dir(job.id)
        opts = dict(job.options)
        if media_path is not None:
            media_path = Path(media_path)
            if source_type == "path":
                # local file owned by the caller (CLI / batch): read in place, never deleted by us
                opts["media_path"] = str(media_path.resolve())
            else:
                # uploads are written into the job's private dir and deleted after the attempt
                if media_path.parent != d:
                    shutil.move(str(media_path), d / media_path.name)
                opts["media_file"] = media_path.name
        if vtt_text:
            (d / "transcript.vtt").write_text(vtt_text, encoding="utf-8")
        self.store.update(job.id, options=opts)
        self.q.put(job.id)
        return self.store.get(job.id)

    def retry(self, job_id: str) -> None:
        self.store.update(job_id, status=JobStatus.QUEUED, message="Re-queued", error_code=None, error_message=None)
        self.q.put(job_id)

    def wait(self, job_id: str, timeout: float = 3600, poll: float = 1.0) -> Job:
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.store.get(job_id)
            if j and j.status.terminal:
                return j
            time.sleep(poll)
        raise TimeoutError(job_id)

    def stop(self) -> None:
        self._stop.set()

    # ---- worker
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.process(job_id)
            finally:
                self.q.task_done()

    def process(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        assert job is not None
        tok = obs.job_id_var.set(job_id)
        usage = obs.Usage()
        utok = obs.usage_var.set(usage)
        d = job_dir(job_id)
        t0 = time.monotonic()
        ctx = None
        self.store.update(job_id, status=JobStatus.EXTRACTING, attempts=job.attempts + 1, message="Starting")
        try:
            media_path = None
            if job.source_type == "link":
                with obs.stage_timer(log, "download"):
                    self.store.update(job_id, message="Downloading recording", progress=0.02)
                    media_path = media.download_from_link(job.source_ref, d)
            elif job.options.get("media_path"):
                media_path = Path(job.options["media_path"])
                if not media_path.exists():
                    raise ClassQAError(f"File not found: {media_path.name}")
            elif job.options.get("media_file"):
                media_path = d / job.options["media_file"]
                if not media_path.exists():
                    raise ClassQAError("The uploaded file is no longer available (media is deleted after each "
                                       "attempt for privacy). Upload it again - finished stages will be reused.")
            vtt_path = d / "transcript.vtt"
            vtt = vtt_path.read_text(encoding="utf-8") if vtt_path.exists() else None

            def progress(f: float, msg: str, status: str) -> None:
                self.store.update(job_id, progress=round(f, 3), message=msg, status=JobStatus(status),
                                  usage=usage.to_dict())

            ctx = JobContext(job_id=job_id, media_path=media_path, vtt_text=vtt, work_dir=d, llm=self.llm,
                             checkpoints=self.store, progress=progress,
                             self_consistency=int(job.options.get("self_consistency", 1)),
                             use_audio_excerpts=bool(job.options.get("audio_excerpts", True)))
            report = run(ctx)
            report["job"] = {"id": job_id, "content_hash": ctx.content_hash, "seconds": round(time.monotonic() - t0, 1),
                             "usage": usage.to_dict()}
            status = JobStatus.NEEDS_REVIEW if report["needs_review"] else JobStatus.DONE
            job = self.store.update(job_id, status=status, progress=1.0, result=report, content_hash=ctx.content_hash,
                                    review_reasons=report["review_reasons"], usage=usage.to_dict(),
                                    message="Needs human review" if report["needs_review"] else "Done")
            obs.event(log, "job_finished", status=status.value, seconds=round(time.monotonic() - t0, 1),
                      **{k: v for k, v in usage.to_dict().items() if k != "by_model"})
        except ClassQAError as e:
            job = self.store.update(job_id, status=JobStatus.FAILED, error_code=e.code, error_message=e.user_message,
                                    usage=usage.to_dict(), message="Failed",
                                    content_hash=ctx.content_hash if ctx else None)
            obs.event(log, "job_failed", logging.WARNING, code=e.code, detail=e.detail[:300])
        except Exception as e:  # noqa: BLE001 - last-resort guard: never leave a job running
            job = self.store.update(job_id, status=JobStatus.FAILED, error_code="internal_error",
                                    error_message="Unexpected internal error - the team has the logs.",
                                    usage=usage.to_dict(), message="Failed")
            obs.event(log, "job_crashed", logging.ERROR, error=str(e)[:300], tb=traceback.format_exc()[-1500:])
        finally:
            self._cleanup(job_id, ctx)
            obs.job_id_var.reset(tok)
            obs.usage_var.reset(utok)
        return job

    def _cleanup(self, job_id: str, ctx: JobContext | None) -> None:
        """Privacy: media never outlives the job attempt (local dir + any Gemini Files uploads)."""
        if ctx is not None:
            for name in ctx.uploaded_files:
                try:
                    self.llm.backend.delete(name)  # type: ignore[attr-defined]
                except Exception as e:  # noqa: BLE001 - Gemini also auto-deletes files after 48 h
                    obs.event(log, "file_delete_failed", logging.WARNING, file=name, error=str(e)[:120])
        shutil.rmtree(DATA_DIR / "jobs" / job_id, ignore_errors=True)
        obs.event(log, "media_deleted")


def run_sync(media_path: Path | None, vtt_text: str | None = None, *, runner: "JobRunner | None" = None,
             **options) -> Job:
    """Run one job in the calling thread (eval harness, tests)."""
    runner = runner or JobRunner(start=False)
    job = runner.submit(source_type="path", source_ref=media_path.name if media_path else "transcript",
                        media_path=media_path, vtt_text=vtt_text, options=options)
    runner.q.get_nowait()
    return runner.process(job.id)


def ops_summary(store: SQLiteStore, limit: int = 500) -> dict:
    jobs = store.list(limit)
    by_status: dict[str, int] = {}
    reasons: dict[str, int] = {}
    costs, secs, toks = [], [], []
    for j in jobs:
        by_status[j.status.value] = by_status.get(j.status.value, 0) + 1
        if j.status == JobStatus.FAILED:
            reasons[j.error_code or "unknown"] = reasons.get(j.error_code or "unknown", 0) + 1
        if j.status in (JobStatus.DONE, JobStatus.NEEDS_REVIEW) and j.result:
            u = j.result.get("job", {}).get("usage", {})
            costs.append(u.get("est_cost_usd", 0))
            toks.append(u.get("tokens_in", 0) + u.get("tokens_out", 0))
            secs.append(j.result.get("job", {}).get("seconds", 0))
    avg = (lambda xs: round(sum(xs) / len(xs), 4) if xs else None)
    return {"jobs": len(jobs), "by_status": by_status, "failure_reasons": reasons,
            "avg_cost_usd_per_class": avg(costs), "avg_seconds_per_class": avg(secs), "avg_tokens_per_class": avg(toks),
            "requests_today": store.count()}
