"""UI smoke tests with Streamlit's AppTest (no network, no API calls)."""
import json
from pathlib import Path

import pytest
from conftest import assessment_json

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def ui_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("CLASSQA_DB", str(tmp_path / "ui.sqlite"))
    monkeypatch.setenv("CLASSQA_DATA_DIR", str(tmp_path / "data"))
    import streamlit as st
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _report():
    from classqa.pipeline import build_report
    from classqa.schemas import Assessment
    from classqa.transcript import from_vtt_text
    from test_quality import VTT

    class Ctx:
        missing_signals, models_used = ["video_frames"], {}
    tr = from_vtt_text(VTT)
    from classqa import metrics
    return build_report([(Assessment.model_validate_json(assessment_json(quote="Today we learn about forks")), "m")],
                        metrics.compute(tr), {}, {}, tr, Ctx())


def test_app_renders_finished_and_failed_jobs(ui_env):
    from streamlit.testing.v1 import AppTest

    from classqa.store import Job, JobStatus, SQLiteStore
    s = SQLiteStore()
    ok = s.create(Job.new(source_type="link", source_ref="https://x/y.mp4"))
    s.update(ok.id, status=JobStatus.NEEDS_REVIEW, result=json.loads(json.dumps(_report())), progress=1.0)
    bad = s.create(Job.new(source_type="link", source_ref="https://x/z.mp4"))
    s.update(bad.id, status=JobStatus.FAILED, error_code="drive_permission", error_message="Not shared")

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["job_id"] = ok.id
    at.run()
    assert not at.exception
    assert any("Needs human review" in w.value for w in at.warning)
    at.session_state["job_id"] = bad.id
    at.run()
    assert not at.exception and any("drive_permission" in e.value for e in at.error)


def test_ops_page_renders(ui_env):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "pages" / "1_Ops.py"), default_timeout=60)
    at.run()
    assert not at.exception
