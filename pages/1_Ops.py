"""Ops dashboard: throughput, failure reasons, cost and latency per class, quota and breaker state."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from classqa.worker import ops_summary
from ui.runtime import get_runner

st.set_page_config(page_title="Ops · 8x8 Class Quality", page_icon="📊", layout="wide")
st.title("📊 Ops & metrics")
runner = get_runner()
s = ops_summary(runner.store)

c = st.columns(5)
c[0].metric("Jobs (last 500)", s["jobs"])
c[1].metric("Needs review", s["by_status"].get("needs_review", 0))
c[2].metric("Failed", s["by_status"].get("failed", 0))
c[3].metric("Avg cost / class (est.)", f"${s['avg_cost_usd_per_class']:.4f}" if s["avg_cost_usd_per_class"] else "–")
c[4].metric("Avg time / class", f"{s['avg_seconds_per_class']:.0f} s" if s["avg_seconds_per_class"] else "–")
st.caption(f"Gemini requests today: {s['requests_today']} / {runner.llm.daily_request_limit} (daily guard) · "
           f"circuit breakers: {runner.llm.breaker.state() or 'all closed'}")

left, right = st.columns(2)
if s["by_status"]:
    left.plotly_chart(px.bar(x=list(s["by_status"]), y=list(s["by_status"].values()), labels={"x": "status", "y": "jobs"},
                          title="Jobs by status"), width="stretch")
if s["failure_reasons"]:
    right.plotly_chart(px.bar(x=list(s["failure_reasons"]), y=list(s["failure_reasons"].values()),
                          labels={"x": "error code", "y": "jobs"}, title="Failure reasons"), width="stretch")

rows = []
for j in runner.store.list(200):
    u = (j.result or {}).get("job", {}).get("usage", {}) or j.usage or {}
    rows.append({"job": j.id, "status": j.status.value, "source": j.source_type,
                 "rating": (j.result or {}).get("rating"), "confidence": (j.result or {}).get("confidence"),
                 "error": j.error_code, "calls": u.get("calls"), "tokens": (u.get("tokens_in", 0) + u.get("tokens_out", 0)),
                 "est_cost_usd": u.get("est_cost_usd"), "seconds": (j.result or {}).get("job", {}).get("seconds"),
                 "review_reasons": "; ".join(j.review_reasons or []),
                 "created": pd.to_datetime(j.created_at, unit="s")})
if rows:
    st.subheader("Jobs")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Stage latency (s), recent finished jobs")
    lat = [{"job": j.id, **(j.result or {}).get("job", {}).get("usage", {}).get("stage_seconds", {})}
           for j in runner.store.list(50) if j.result]
    if lat:
        st.dataframe(pd.DataFrame(lat), hide_index=True, width="stretch")
else:
    st.info("No jobs yet.")
