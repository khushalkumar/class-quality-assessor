"""Streamlit front-end: upload a class recording (or paste a link) and get a
Poor / Average / Good / Excellent rating with evidence-backed observations."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from classqa import llm, media, pipeline
from classqa.report import to_markdown
from classqa.rubric import DIMENSIONS

st.set_page_config(page_title="8x8 Class Quality Assessor", page_icon="♟️", layout="wide")

COLORS = {"Poor": "#DC2626", "Average": "#D97706", "Good": "#2563EB", "Excellent": "#059669"}
NAMES = {d["key"]: d["name"] for d in DIMENSIONS}

st.markdown("""
<style>
.badge {border-radius:18px;padding:22px 28px;color:white;text-align:center}
.badge h1 {color:white;margin:0;font-size:3rem}
.badge p {margin:0;opacity:.9}
.ev {font-size:.92rem;margin:2px 0 6px 0}
.ts {font-family:monospace;background:#EEF2FF;color:#3730A3;padding:1px 6px;border-radius:6px}
</style>""", unsafe_allow_html=True)

st.title("♟️ 8x8 Class Quality Assessor")
st.caption("Upload a recorded online class (or share a link). The app transcribes it, measures interaction, looks at "
           "the video, and rates the class **Poor / Average / Good / Excellent** against a rubric derived from 8x8's "
           "own reviewer feedback — with timestamped evidence for every observation.")

if not llm.get_api_key():
    st.error("GEMINI_API_KEY is not configured. Add it to `.streamlit/secrets.toml` or the environment.")
    st.stop()

with st.sidebar:
    st.header("How it works")
    st.markdown("""
1. **Audio** → Gemini transcribes in 10-min chunks with speaker roles (or use your Zoom `.vtt`).
2. **Metrics** → talk-time split, questions, student turns, participation spread, monologues, silences, praise, homework, A/V-problem mentions.
3. **Video** → 12 sampled frames (camera, face, branded background, lighting, screen share) + pixel checks.
4. **Tone** → 3 short audio excerpts for teacher energy & audio quality.
5. **Rubric** → Gemini scores 6 dimensions (1–5) with quoted evidence; the final label is a fixed weighted score, calibrated on the labelled sample classes.
""")
    st.markdown("**Rubric weights**")
    st.dataframe(pd.DataFrame([{"Dimension": d["name"], "Weight": f"{int(d['weight']*100)}%"} for d in DIMENSIONS]),
                 hide_index=True, use_container_width=True)
    st.caption("Privacy: the recording is processed in a temporary folder and sent only to the Gemini API; "
               "nothing is stored after the report is produced.")

tab_up, tab_link = st.tabs(["📤 Upload recording", "🔗 Link"])
with tab_up:
    up = st.file_uploader("Class recording (video or audio)", type=[e.strip(".") for e in sorted(media.SUPPORTED_EXT)])
with tab_link:
    link = st.text_input("Google Drive file link or direct URL to a video/audio file",
                         placeholder="https://drive.google.com/file/d/.../view?usp=sharing")
vtt = st.file_uploader("Optional: Zoom transcript (.vtt) — faster; otherwise the audio is transcribed",
                       type=["vtt"])
run = st.button("Assess class", type="primary", disabled=not (up or link.strip()))


def render(rep: dict) -> None:
    a, met = rep["assessment"], rep["metrics"]["transcript"]
    c1, c2 = st.columns([1, 2.3])
    with c1:
        st.markdown(f"<div class='badge' style='background:{COLORS[rep['rating']]}'><p>Overall rating</p>"
                    f"<h1>{rep['rating']}</h1><p>score {rep['weighted_score']:.2f} / 5 · confidence "
                    f"{rep['confidence']}</p></div>", unsafe_allow_html=True)
        if rep["holistic_rating"] != rep["rating"]:
            st.caption(f"The model's holistic view was **{rep['holistic_rating']}** — borderline case.")
    with c2:
        st.markdown(f"**Summary.** {a['class_summary']}")
        st.markdown(f"**Why.** {a['verdict']}")
        for f in a.get("red_flags", []):
            (st.error if f["severity"] == "major" else st.warning)(f"**{f['severity'].title()} flag** `{f['timestamp']}` — {f['issue']}")
        if rep.get("capped_by_red_flag"):
            st.caption("Rating capped at Average because of a major red flag.")

    st.subheader("Rubric scores")
    dims = a["dimensions"]
    cc1, cc2 = st.columns([1.2, 1])
    with cc1:
        labels = [NAMES.get(d["key"], d["key"]) for d in dims]
        fig = go.Figure(go.Bar(x=[d["score"] for d in dims], y=labels, orientation="h",
                               marker_color=[COLORS[["Poor", "Poor", "Average", "Good", "Excellent"][d["score"] - 1]] for d in dims],
                               text=[f"{d['score']}/5" for d in dims], textposition="outside"))
        fig.update_layout(xaxis=dict(range=[0, 5.5], dtick=1), height=300, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Duration", f"{met['duration_min']} min")
        m2.metric("Teacher talk", f"{met['teacher_talk_pct']}%")
        m3.metric("Student turns / 10 min", met["student_turns_per_10min"])
        m4, m5, m6 = st.columns(3)
        m4.metric("Questions / 10 min", met["teacher_questions_per_10min"])
        m5.metric("Children who spoke", met["distinct_students_spoke"])
        m6.metric("Longest monologue", f"{met['longest_teacher_monologue_min']} min")
        v = a["visual"]
        st.caption(f"📷 Camera: **{v['teacher_camera_on']}** · face: **{v['teacher_face_visible']}** · background: "
                   f"**{v['branded_background']}** · lighting: **{v['lighting']}**")

    st.subheader("Evidence by dimension")
    for d in dims:
        with st.expander(f"{NAMES.get(d['key'], d['key'])} — {d['score']}/5: {d['summary']}"):
            for e in d["evidence"]:
                st.markdown(f"<div class='ev'><span class='ts'>{e['timestamp']}</span> {e['observation']}"
                            f"<br><i>“{e['quote']}”</i></div>", unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        st.subheader("✅ Strengths")
        for p in a["strengths"]:
            st.markdown(f"**{p['title']}** <span class='ts'>{p['timestamp']}</span><br>{p['detail']}",
                        unsafe_allow_html=True)
    with s2:
        st.subheader("🎯 Improve next class")
        for p in a["improvements"]:
            st.markdown(f"**{p['title']}** <span class='ts'>{p['timestamp']}</span><br>{p['detail']}<br>"
                        f"💡 _{p['coaching_tip']}_", unsafe_allow_html=True)

    with st.expander("All objective metrics"):
        st.json(rep["metrics"])
    with st.expander("Visual check notes"):
        st.write(a["visual"]["notes"])
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Report (Markdown)", to_markdown(rep), "class_report.md", "text/markdown")
    d2.download_button("⬇️ Report (JSON)", json.dumps(rep, indent=2), "class_report.json", "application/json")


if run:
    work = media.tmpdir()
    bar = st.progress(0.0, "Starting")
    try:
        if up is not None:
            path = work / Path(up.name).name
            with open(path, "wb") as f:
                for chunk in iter(lambda: up.read(8 << 20), b""):
                    f.write(chunk)
        else:
            bar.progress(0.02, "Downloading recording from link")
            path = media.download_from_link(link, work)
        if path.suffix.lower() not in media.SUPPORTED_EXT:
            raise media.MediaError(f"Unsupported file type '{path.suffix}'.")
        vtt_text = vtt.getvalue().decode("utf-8", errors="ignore") if vtt else None
        rep = pipeline.assess(path, vtt_text, progress=lambda f, m: bar.progress(min(f, 1.0), m))
        bar.progress(1.0, "Done")
        st.session_state["report"] = rep
    except (media.MediaError, ValueError) as e:
        st.error(str(e))
    except llm.LLMError as e:
        st.error(f"The AI service is busy or failed: {e}. Please try again in a minute.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Unexpected error: {e}")
    finally:
        media.cleanup(work)

if "report" in st.session_state:
    render(st.session_state["report"])
