"""Job model, worker, checkpoints/resume, cleanup, partial results - fake LLM, synthetic media."""
import shutil

import pytest
from conftest import FakeBackend, assessment_json

from classqa import worker
from classqa.store import Job, JobStatus


@pytest.fixture
def runner(store, make_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "DATA_DIR", tmp_path / "data")

    def _make(backend=None, **kw):
        return worker.JobRunner(store=store, llm=make_llm(backend, **kw), start=False)
    return _make


def _run(r, path, **opts):
    j = r.submit(source_type="path", source_ref=path.name, media_path=path, options=opts)
    r.q.get_nowait()
    return r.process(j.id)


def test_store_roundtrip(store):
    j = store.create(Job.new(source_type="upload", source_ref="a.mp4"))
    store.update(j.id, status=JobStatus.SCORING, result={"x": 1}, review_reasons=["r"])
    g = store.get(j.id)
    assert g.status == JobStatus.SCORING and g.result == {"x": 1} and g.review_reasons == ["r"]


def test_happy_path_end_to_end(runner, media_dir):
    b = FakeBackend(assessment=[assessment_json(score=5, quote="today we learn about forks in chess",
                                                holistic="Excellent")])
    r = runner(b)
    j = _run(r, media_dir / "good.mp4")
    assert j.status in (JobStatus.DONE, JobStatus.NEEDS_REVIEW), (j.error_code, j.error_message)
    assert j.result["rating"] == "Excellent"
    assert j.result["evidence_check"]["exact"] >= 1
    assert j.usage["calls"] == 2 and j.usage["est_cost_usd"] > 0
    assert set(j.usage["stage_seconds"]) >= {"validate", "transcript", "score"}
    assert not (worker.DATA_DIR / "jobs" / j.id).exists()           # media/work dir deleted
    assert (media_dir / "good.mp4").exists()                        # caller-owned file untouched


def test_checkpoints_make_rerun_free(runner, media_dir):
    b = FakeBackend()
    r = runner(b)
    _run(r, media_dir / "good.mp4")
    n = len(b.calls)
    j2 = _run(r, media_dir / "good.mp4")
    assert len(b.calls) == n and j2.status != JobStatus.FAILED     # idempotent: no new API calls


def test_failed_scoring_then_retry_resumes_from_transcript(runner, media_dir):
    b = FakeBackend(script=[None])  # placeholder replaced below
    b.script = []
    r = runner(b, max_attempts=1)
    # first attempt: transcription succeeds, scoring returns garbage twice -> structured failure
    b.assessment = ["garbage", "still garbage"]
    j = _run(r, media_dir / "good.mp4")
    assert j.status == JobStatus.FAILED and j.error_code == "malformed_output"
    transcribe_calls = sum(1 for _, s in b.calls if s == "TranscriptChunk")
    b.assessment, b._a = [assessment_json()], 0
    j2 = _run(r, media_dir / "good.mp4")
    assert j2.status != JobStatus.FAILED
    assert sum(1 for _, s in b.calls if s == "TranscriptChunk") == transcribe_calls   # resumed


@pytest.mark.parametrize("name,code", [("silent.mp4", "silent_audio"), ("corrupt.mp4", "corrupt_media"),
                                       ("noaudio.mp4", "no_audio"), ("short.mp4", "media_too_short")])
def test_bad_media_fails_before_any_api_call(runner, media_dir, name, code):
    b = FakeBackend()
    j = _run(runner(b), media_dir / name)
    assert j.status == JobStatus.FAILED and j.error_code == code and j.error_message
    assert b.calls == []


def test_audio_only_degrades_gracefully(runner, media_dir):
    j = _run(runner(), media_dir / "audio_only.mp3")
    assert j.status == JobStatus.NEEDS_REVIEW
    assert "video_frames" in j.result["missing_signals"]


def test_llm_outage_is_a_clean_failure(runner, media_dir):
    b = FakeBackend(script=[RuntimeError("503 UNAVAILABLE")] * 50)
    j = _run(runner(b), media_dir / "good.mp4")
    assert j.status == JobStatus.FAILED and j.error_code in ("llm_unavailable", "circuit_open")


def test_self_consistency_disagreement_needs_review(runner, media_dir):
    b = FakeBackend(assessment=[assessment_json(score=5, holistic="Excellent"),
                                assessment_json(score=2, holistic="Poor")])
    j = _run(runner(b), media_dir / "good.mp4", self_consistency=2)
    assert j.status == JobStatus.NEEDS_REVIEW and j.result["self_consistency"]["disagree"]


def test_upload_media_deleted_even_on_failure(runner, media_dir, tmp_path):
    up = tmp_path / "upload.mp4"
    shutil.copy(media_dir / "corrupt.mp4", up)
    r = runner()
    j = r.submit(source_type="upload", source_ref="upload.mp4", media_path=up)
    r.q.get_nowait()
    j = r.process(j.id)
    assert j.status == JobStatus.FAILED and not (worker.DATA_DIR / "jobs" / j.id).exists()


def test_ops_summary(runner, media_dir, store):
    r = runner()
    _run(r, media_dir / "good.mp4")
    _run(r, media_dir / "corrupt.mp4")
    s = worker.ops_summary(store)
    assert s["by_status"].get("failed") == 1 and s["failure_reasons"] == {"corrupt_media": 1}
    assert s["avg_cost_usd_per_class"] is not None


def test_frame_extraction_failure_degrades(runner, media_dir, monkeypatch):
    from classqa import media
    from classqa.errors import CorruptMedia

    def boom(*a, **k):
        raise CorruptMedia("frame grab failed")
    monkeypatch.setattr(media, "frame_jpeg", boom)
    j = _run(runner(), media_dir / "good.mp4")
    assert j.status == JobStatus.NEEDS_REVIEW and "video_frames" in j.result["missing_signals"]


def test_unexpected_bug_is_contained(runner, media_dir, monkeypatch):
    from classqa import metrics

    def bug(*a, **k):
        raise KeyError("oops")
    monkeypatch.setattr(metrics, "compute", bug)
    j = _run(runner(), media_dir / "good.mp4")
    assert j.status == JobStatus.FAILED and j.error_code == "internal_error"
    assert not (worker.DATA_DIR / "jobs" / j.id).exists()


def test_restart_recovery_marks_stale_jobs_failed(store, make_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "DATA_DIR", tmp_path / "data")
    j = store.create(Job.new(source_type="link", source_ref="https://x/y.mp4"))
    store.update(j.id, status=JobStatus.TRANSCRIBING)
    store._db.execute("UPDATE jobs SET updated_at=0 WHERE id=?", (j.id,))
    worker.JobRunner(store=store, llm=make_llm(), start=False)
    assert store.get(j.id).error_code == "worker_restarted"


def test_background_thread_processes_queue(store, make_llm, tmp_path, monkeypatch, media_dir):
    monkeypatch.setattr(worker, "DATA_DIR", tmp_path / "data")
    r = worker.JobRunner(store=store, llm=make_llm(), start=True)
    j = r.submit(source_type="path", source_ref="good.mp4", media_path=media_dir / "good.mp4")
    done = r.wait(j.id, timeout=120, poll=0.2)
    r.stop()
    assert done.status in (JobStatus.DONE, JobStatus.NEEDS_REVIEW)
