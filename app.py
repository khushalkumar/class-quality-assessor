"""Streamlit front-end. Thin UI over the processing core: it submits jobs to the
background worker and polls their status; all processing lives in `classqa/`."""
from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from classqa import llm, media
from classqa.rubric import DIMENSIONS
from classqa.store import JobStatus
from classqa.worker import job_dir
from ui.report_view import render
from ui.runtime import get_runner

st.set_page_config(page_title="8x8 Class Quality Assessor", page_icon="♟️", layout="wide")

UPLOAD_RECOMMENDED_MB = 200
STEPS = [("queued", "Queued"), ("extracting", "Download & checks"), ("transcribing", "Transcribe"),
         ("scoring", "Score"), ("done", "Report")]

st.title("♟️ 8x8 Class Quality Assessor")
st.caption("Give a recorded online class (Drive/URL link or upload). The app transcribes it, measures interaction, "
           "looks at the video, and rates it **Poor / Average / Good / Excellent** against a rubric derived from "
           "8x8's own reviewer feedback — with timestamped evidence for every observation. Uncertain results are "
           "routed to **human review** instead of being presented as certain.")

if not llm.get_api_key():
    st.error("GEMINI_API_KEY is not configured. Add it to `.streamlit/secrets.toml` or the environment.")
    st.stop()
runner = get_runner()

with st.sidebar:
    st.header("How it works")
    st.markdown("""
1. **Validate** the file (format, size, duration, audio present and not silent) before any AI spend.
2. **Transcribe** with Gemini in 20-min chunks (or use your Zoom `.vtt`).
3. **Measure** talk-time, questions, student turns, participation, monologues, silences.
4. **Look & listen**: 12 frames + speaker crops, 4 short audio excerpts.
5. **Score** 6 rubric dimensions with quoted evidence; quotes are **checked against the transcript**.
6. **Label** by a fixed weighted score; borderline / low-confidence / incomplete results → **needs review**.
""")
    st.dataframe(pd.DataFrame([{"Dimension": d["name"], "Weight": f"{int(d['weight']*100)}%"} for d in DIMENSIONS]),
                 hide_index=True, width="stretch")
    st.caption("Privacy: media is processed in a per-job temp folder, sent only to the Gemini API, and deleted when "
               "the job ends (success or failure).")
    st.divider()
    st.subheader("Recent jobs")
    for j in runner.store.list(8):
        label = j.result["rating"] if j.result else j.status.value
        if st.button(f"{j.source_ref[:28] or j.id} · {label}", key=f"open_{j.id}"):
            st.session_state["job_id"] = j.id

tab_link, tab_up = st.tabs(["🔗 Link (recommended for big files)", "📤 Upload"])
with tab_link:
    link = st.text_input("Google Drive file link or direct URL to a video/audio file",
                         placeholder="https://drive.google.com/file/d/<id>/view?usp=sharing")
    st.caption("Share the Drive file as **Anyone with the link → Viewer**. The server streams the file to disk "
               f"with progress and resumes dropped connections (limit {media.MAX_BYTES // 1024 ** 2} MB, "
               f"{media.MAX_DURATION_S / 3600:.0f} h).")
with tab_up:
    up = st.file_uploader("Class recording (video or audio)", type=[e.strip(".") for e in sorted(media.SUPPORTED_EXT)])
    st.caption(f"Browser uploads above ~{UPLOAD_RECOMMENDED_MB} MB can stall on slow or proxied connections — use the "
               "Link tab for full-length recordings.")
vtt = st.file_uploader("Optional: Zoom transcript (.vtt) — faster and more accurate speaker labels", type=["vtt"])
with st.expander("Options"):
    sc = st.checkbox("Self-consistency: score twice and flag disagreement (2× scoring cost)", value=False)

if st.button("Assess class", type="primary", disabled=not (up or link.strip())):
    vtt_text = vtt.getvalue().decode("utf-8", errors="ignore") if vtt else None
    opts = {"self_consistency": 2 if sc else 1}
    if link.strip() and not up:
        job = runner.submit(source_type="link", source_ref=link.strip(), vtt_text=vtt_text, options=opts)
    else:
        jid = uuid.uuid4().hex[:12]
        name = Path(up.name).name
        if Path(name).suffix.lower() not in media.SUPPORTED_EXT:
            st.error(f"Unsupported file type '{Path(name).suffix}'.")
            st.stop()
        dst = job_dir(jid) / name
        with open(dst, "wb") as f:
            for chunk in iter(lambda: up.read(8 << 20), b""):
                f.write(chunk)
        job = runner.submit(source_type="upload", source_ref=name, media_path=dst, vtt_text=vtt_text,
                            options=opts, job_id=jid)
    st.session_state["job_id"] = job.id


def _stepper(status: str) -> str:
    order = [s for s, _ in STEPS]
    cur = order.index(status) if status in order else (len(order) - 1 if status in ("needs_review",) else -1)
    out = []
    for i, (_, label) in enumerate(STEPS):
        out.append(("✅ " if i < cur or status in ("done", "needs_review") else "⏳ " if i == cur else "▫️ ") + label)
    return "  →  ".join(out)


@st.fragment(run_every=2.0)
def job_panel(job_id: str) -> None:
    j = runner.store.get(job_id)
    if j is None:
        st.error("Job not found.")
        return
    if j.status.terminal:
        if st.session_state.get("_rendered") != (job_id, j.status.value):
            st.session_state["_rendered"] = (job_id, j.status.value)
            st.rerun()   # leave the fragment so the full report renders once
        return
    st.markdown(f"**Job `{j.id}`** — {_stepper(j.status.value)}")
    st.progress(min(j.progress, 1.0), j.message or j.status.value)
    u = j.usage or {}
    if u.get("calls"):
        st.caption(f"{u['calls']} Gemini calls so far · ~${u.get('est_cost_usd', 0):.4f} (est.)")


if "job_id" in st.session_state:
    jid = st.session_state["job_id"]
    j = runner.store.get(jid)
    if j and not j.status.terminal:
        job_panel(jid)
    elif j and j.status == JobStatus.FAILED:
        st.error(f"**The assessment failed** ({j.error_code}): {j.error_message}")
        hints = {
            "drive_permission": "Open the Drive file → Share → General access: *Anyone with the link*.",
            "drive_quota": "Drive limits downloads of popular files; wait a while or upload a copy.",
            "daily_quota": "The daily AI budget is used up; the job can be retried after the reset.",
            "llm_unavailable": "Gemini was busy on every configured model. Retry in a few minutes — finished "
                               "stages (e.g. the transcript) are reused, so a retry is cheap.",
            "circuit_open": "Repeated AI failures paused requests briefly. Retry in a few minutes.",
            "worker_restarted": "The app restarted during the job. Retry — finished stages are reused.",
        }
        if j.error_code in hints:
            st.info(hints[j.error_code])
        if j.source_type == "link" and st.button("Retry job"):
            runner.retry(jid)
            st.rerun()
        elif j.source_type == "upload":
            st.caption("For privacy the upload was deleted; re-upload to retry (finished stages are reused).")
    elif j and j.result:
        render(j.result)
