"""Thin, defensive wrapper around the google-genai SDK:
retry with exponential backoff on 429/5xx, model fallback chain, structured
JSON output via Pydantic schemas, a tiny on-disk cache, and a call counter."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import types
from pydantic import BaseModel

PRIMARY_MODEL = os.environ.get("CLASSQA_MODEL", "gemini-3-flash-preview")
FALLBACK_MODELS = [m for m in os.environ.get(
    "CLASSQA_FALLBACK_MODELS", "gemini-3.1-pro-preview,gemini-flash-latest,gemini-3.1-flash-lite-preview").split(",") if m]
CACHE_DIR = Path(os.environ.get("CLASSQA_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache"))

STATS = {"calls": 0, "failed_calls": 0, "prompt_tokens": 0, "output_tokens": 0, "cache_hits": 0}


class LLMError(RuntimeError):
    pass


def get_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - no secrets file / not in streamlit
        return None


_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        key = get_api_key()
        if not key:
            raise LLMError("GEMINI_API_KEY is not configured (env var or Streamlit secrets).")
        _client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=600_000))
    return _client


def _cache_key(parts: list[Any]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:32]


def _retryable(e: Exception) -> bool:
    """Retry everything except clear client errors (bad request / auth)."""
    s = str(e)
    return not any(c in s for c in ("400 ", "401 ", "403 ", "INVALID_ARGUMENT", "PERMISSION_DENIED", "API key not valid"))


def generate_json(contents: list, schema: type[BaseModel], *, system: str | None = None,
                  cache_parts: list | None = None, thinking: str = "low", max_attempts: int = 6,
                  models: list[str] | None = None, on_retry: Callable[[str], None] | None = None) -> BaseModel:
    """Call Gemini with a response schema; return a validated Pydantic object."""
    ck = None
    if cache_parts is not None:
        ck = _cache_key([schema.__name__, system or "", *cache_parts])
        f = CACHE_DIR / f"{ck}.json"
        if f.exists():
            STATS["cache_hits"] += 1
            return schema.model_validate_json(f.read_text())

    chain = models or [PRIMARY_MODEL, *FALLBACK_MODELS]
    last: Exception | None = None
    for model in chain:
        for attempt in range(max_attempts):
            try:
                cfg = types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                    thinking_config=types.ThinkingConfig(thinking_level=thinking),
                    max_output_tokens=32768,
                )
                STATS["calls"] += 1
                resp = client().models.generate_content(model=model, contents=contents, config=cfg)
                um = resp.usage_metadata
                if um:
                    STATS["prompt_tokens"] += um.prompt_token_count or 0
                    STATS["output_tokens"] += (um.candidates_token_count or 0) + (um.thoughts_token_count or 0)
                obj = resp.parsed if isinstance(resp.parsed, schema) else schema.model_validate_json(resp.text or "")
                if ck:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    (CACHE_DIR / f"{ck}.json").write_text(obj.model_dump_json())
                return obj
            except Exception as e:  # noqa: BLE001
                STATS["failed_calls"] += 1
                last = e
                msg = str(e)
                # model unavailable / zero quota for this model -> move to next model
                if "404" in msg or "limit: 0" in msg or "PerDay" in msg:
                    break
                if not _retryable(e):
                    raise LLMError(f"Gemini request failed: {msg[:300]}") from e
                wait = min(90, (2 ** attempt) * 3 + random.uniform(0, 2))
                if on_retry:
                    on_retry(f"{model}: {msg[:80]}... retrying in {wait:.0f}s")
                time.sleep(wait)
    raise LLMError(f"Gemini unavailable after retries/fallbacks: {str(last)[:300]}")


def audio_part(data: bytes, mime: str = "audio/mp3") -> types.Part:
    return types.Part.from_bytes(data=data, mime_type=mime)


def image_part(data: bytes) -> types.Part:
    return types.Part.from_bytes(data=data, mime_type="image/jpeg")
