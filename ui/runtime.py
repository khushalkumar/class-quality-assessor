"""Process-wide singletons for the Streamlit UI (one worker per container)."""
from __future__ import annotations

import os

import streamlit as st

from classqa import llm
from classqa.worker import JobRunner


@st.cache_resource(show_spinner=False)
def get_runner() -> JobRunner:
    key = llm.get_api_key()
    if key:
        os.environ.setdefault("GEMINI_API_KEY", key)   # worker thread has no Streamlit context
    return JobRunner(workers=int(os.environ.get("CLASSQA_WORKERS", "1")))
