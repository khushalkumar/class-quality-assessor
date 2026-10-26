"""Report card rendering (pure presentation; takes a finished report dict)."""
from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st

from classqa.report import to_markdown
from classqa.rubric import DIMENSIONS

COLORS = {"Poor": "#DC2626", "Average": "#D97706", "Good": "#2563EB", "Excellent": "#059669"}
NAMES = {d["key"]: d["name"] for d in DIMENSIONS}
QBADGE = {"exact": "✅ verified", "approximate": "≈ near-verbatim", "unverified": "⚠️ not in transcript",
          "nonverbal": "👁️ visual/audio"}

CSS = """
<style>
.badge {border-radius:18px;padding:22px 28px;color:white;text-align:center}
.badge h1 {color:white;margin:0;font-size:3rem}
.badge p {margin:0;opacity:.9}
.ev {font-size:.92rem;margin:2px 0 6px 0}
.ts {font-family:monospace;background:#EEF2FF;color:#3730A3;padding:1px 6px;border-radius:6px}
.qs {font-size:.75rem;color:#6B7280}
</style>"""


def render(rep: dict) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    a, met = rep["assessment"], rep["metrics"]["transcript"]
    if rep.get("needs_review"):
        st.warning("**Needs human review** — the rating is shown, but please confirm before acting on it:\n\n"
                   + "\n".join(f"- {r}" for r in rep.get("review_reasons", [])))
    c1, c2 = st.columns([1, 2.3])
    with c1:
        st.markdown(f"<div class='badge' style='background:{COLORS[rep['rating']]}'><p>Overall rating</p>"
                    f"<h1>{rep['rating']}</h1><p>score {rep['weighted_score']:.2f} / 5 · confidence "
                    f"{rep['confidence']}</p></div>", unsafe_allow_html=True)
        if rep["holistic_rating"] != rep["rating"]:
            st.caption(f"The model's holistic view was **{rep['holistic_rating']}**.")
        sc = rep.get("self_consistency", {})
        if sc.get("runs", 1) > 1:
            st.caption(f"Self-consistency: {sc['runs']} runs → {', '.join(sc['ratings'])}")
    with c2:
        st.markdown(f"**Summary.** {a['class_summary']}")
        st.markdown(f"**Why.** {a['verdict']}")
        for f in a.get("red_flags", []):
            (st.error if f["severity"] == "major" else st.warning)(
                f"**{f['severity'].title()} flag** `{f['timestamp']}` — {f['issue']}")
        if rep.get("capped_by_red_flag"):
            st.caption("Rating capped at Average because of a major red flag.")
        if rep.get("missing_signals"):
            st.info("Assessed without: " + ", ".join(m.replace("_", " ") for m in rep["missing_signals"])
                    + ". Related checks are marked unclear.")

    st.subheader("Rubric scores")
    dims = a["dimensions"]
    cc1, cc2 = st.columns([1.2, 1])
    with cc1:
        labels = [NAMES.get(d["key"], d["key"]) for d in dims]
        fig = go.Figure(go.Bar(x=[d["score"] for d in dims], y=labels, orientation="h",
                               marker_color=[COLORS[["Poor", "Poor", "Average", "Good", "Excellent"][d["score"] - 1]]
                                             for d in dims],
                               text=[f"{d['score']}/5" for d in dims], textposition="outside"))
        fig.update_layout(xaxis=dict(range=[0, 5.5], dtick=1), height=300, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
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
        ec = rep.get("evidence_check")
        if ec:
            st.caption(f"🔎 Evidence check: {ec['exact'] + ec['approximate']} quotes verified in the transcript, "
                       f"{ec['unverified']} removed as not found, {ec['nonverbal']} visual/audio observations.")

    st.subheader("Evidence by dimension")
    for d in dims:
        with st.expander(f"{NAMES.get(d['key'], d['key'])} — {d['score']}/5: {d['summary']}"):
            for e in d["evidence"]:
                qs = QBADGE.get(e.get("quote_status", ""), "")
                st.markdown(f"<div class='ev'><span class='ts'>{e['timestamp']}</span> {e['observation']}"
                            f"<br><i>“{e['quote']}”</i> <span class='qs'>{qs}</span></div>", unsafe_allow_html=True)

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
    job = rep.get("job", {})
    u = job.get("usage", {})
    v = rep.get("versions", {})
    st.caption(f"Model {rep.get('model')} · transcript {rep.get('transcript_source')}"
               + (f" ({rep.get('transcription_model')})" if rep.get("transcription_model") else "")
               + f" · prompt {rep.get('prompt_version')} · rubric {v.get('rubric', '?')} · pipeline {v.get('pipeline', '?')}"
               + (f" · job {job.get('id')} · {job.get('seconds')} s · {u.get('calls', 0)} Gemini calls · "
                  f"~${u.get('est_cost_usd', 0):.4f} (est.)" if job else ""))
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Report (Markdown)", to_markdown(rep), "class_report.md", "text/markdown")
    d2.download_button("⬇️ Report (JSON)", json.dumps(rep, indent=2), "class_report.json", "application/json")
