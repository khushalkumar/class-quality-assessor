"""Gemini access layer with explicit failure handling.

- exponential backoff with full jitter on 429 / 5xx / timeouts / dropped connections
- model fallback chain; a model that reports a *daily* quota error is skipped for the day
- per-model circuit breaker (N consecutive failures -> open for a cooldown)
- structured output validated against a Pydantic schema, ONE repair retry on malformed JSON
- per-job token/cost budget and a global daily request guard
- deterministic-as-possible settings (fixed seed; temperature configurable)
The transport is injectable (`backend`) so every path is unit-tested without API calls."""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ValidationError

from . import obs
from .errors import AuthError, BudgetExceeded, CircuitOpen, DailyQuotaExceeded, LLMUnavailable, SchemaError

log = logging.getLogger("classqa.llm")

# Frozen after evaluation (eval/EVALUATION.md): gemini-3.5-flash-lite gave the best complete run and has a usable
# free-tier quota; larger flash models share a ~20 requests/day free-tier pool and returned many 429/503s.
PRIMARY_MODEL = os.environ.get("CLASSQA_MODEL", "gemini-3.5-flash-lite")
TRANSCRIBE_MODEL = os.environ.get("CLASSQA_TRANSCRIBE_MODEL", "gemini-3.1-flash-lite-preview")
FALLBACK_MODELS = [m for m in os.environ.get(
    "CLASSQA_FALLBACK_MODELS",
    "gemini-3.1-flash-lite-preview,gemini-flash-lite-latest,gemini-3-flash-preview").split(",") if m]

SEED = int(os.environ.get("CLASSQA_SEED", "1234"))
# Gemini 3 guidance is to keep the default temperature (1.0); lower values can cause looping. Configurable.
TEMPERATURE = float(os.environ["CLASSQA_TEMPERATURE"]) if os.environ.get("CLASSQA_TEMPERATURE") else None
INLINE_LIMIT = 15 * 1024 * 1024   # bytes; larger media goes through the Files API (and is deleted afterwards)


def get_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - no secrets file / not running under streamlit
        return None


# ---------------------------------------------------------------- transport
@dataclass
class RawResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class Backend(Protocol):
    def generate(self, model: str, contents: list, *, system: str | None, schema: type[BaseModel],
                 thinking: str) -> RawResponse: ...


class GeminiBackend:
    def __init__(self, api_key: str | None = None, timeout_s: int = 600):
        from google import genai
        from google.genai import types

        self._types = types
        key = api_key or get_api_key()
        if not key:
            raise AuthError("GEMINI_API_KEY is not configured (Streamlit secrets or environment).")
        self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=timeout_s * 1000))

    def generate(self, model, contents, *, system, schema, thinking) -> RawResponse:
        t = self._types
        cfg = t.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json", response_schema=schema,
            thinking_config=t.ThinkingConfig(thinking_level=thinking), max_output_tokens=32768, seed=SEED,
            **({"temperature": TEMPERATURE} if TEMPERATURE is not None else {}))
        r = self._client.models.generate_content(model=model, contents=contents, config=cfg)
        um = r.usage_metadata
        return RawResponse(r.text or "", (um.prompt_token_count or 0) if um else 0,
                           ((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0)

    # Files API for media above the inline limit; always paired with delete().
    def upload(self, path: str, mime: str):
        return self._client.files.upload(file=path, config={"mime_type": mime})

    def delete(self, name: str) -> None:
        self._client.files.delete(name=name)


# ---------------------------------------------------------------- classification
def classify(e: Exception) -> str:
    """-> 'auth' | 'missing_model' | 'daily_quota' | 'rate_limit' | 'transient' | 'bad_request'"""
    s = str(e)
    if any(x in s for x in ("API key not valid", "PERMISSION_DENIED", "401", "403 ")):
        return "auth"
    if "404" in s or "NOT_FOUND" in s:
        return "missing_model"
    if "429" in s or "RESOURCE_EXHAUSTED" in s:
        return "daily_quota" if ("per day" in s.lower() or "PerDay" in s or "limit: 0" in s) else "rate_limit"
    if "400" in s or "INVALID_ARGUMENT" in s:
        return "bad_request"
    return "transient"   # 500/503/504, timeouts, disconnects


class CircuitBreaker:
    def __init__(self, threshold: int = 5, cooldown_s: float = 120.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown_s, clock
        self._fails: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def allow(self, model: str) -> bool:
        with self._lock:
            return self.clock() >= self._open_until.get(model, 0.0)

    def success(self, model: str) -> None:
        with self._lock:
            self._fails[model] = 0

    def failure(self, model: str, *, hard: bool = False) -> None:
        with self._lock:
            n = self._fails.get(model, 0) + 1
            self._fails[model] = n
            if hard or n >= self.threshold:
                # hard = daily quota exhausted: keep open for a long time
                self._open_until[model] = self.clock() + (6 * 3600 if hard else self.cooldown)
                self._fails[model] = 0
                obs.event(log, "circuit_open", logging.WARNING, model=model, hard=hard)

    def state(self) -> dict:
        now = self.clock()
        return {m: ("open" if t > now else "closed") for m, t in self._open_until.items()}


REPAIR_PROMPT = ("Your previous answer was not valid JSON for the required schema ({err}). "
                 "Return ONLY the corrected JSON object, nothing else.")


class LLMClient:
    def __init__(self, backend: Backend | None = None, *, quota=None, breaker: CircuitBreaker | None = None,
                 max_attempts: int = 5, base_delay: float = 2.0, max_delay: float = 60.0,
                 sleep: Callable[[float], None] = time.sleep,
                 daily_request_limit: int | None = None,
                 job_token_budget: int | None = None, job_cost_budget_usd: float | None = None):
        self._backend = backend
        self.quota = quota
        self.breaker = breaker or CircuitBreaker()
        self.max_attempts, self.base_delay, self.max_delay, self.sleep = max_attempts, base_delay, max_delay, sleep
        self.daily_request_limit = daily_request_limit if daily_request_limit is not None else int(
            os.environ.get("CLASSQA_DAILY_REQUEST_LIMIT", "400"))
        self.job_token_budget = job_token_budget if job_token_budget is not None else int(
            os.environ.get("CLASSQA_JOB_TOKEN_BUDGET", "2000000"))
        self.job_cost_budget = job_cost_budget_usd if job_cost_budget_usd is not None else float(
            os.environ.get("CLASSQA_JOB_COST_BUDGET_USD", "0.50"))

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = GeminiBackend()
        return self._backend

    # ---- guards
    def _check_guards(self) -> None:
        u = obs.usage_var.get()
        if u is not None and (u.tokens_in + u.tokens_out > self.job_token_budget or u.cost_usd > self.job_cost_budget):
            raise BudgetExceeded(
                f"This job hit its budget ({u.tokens_in + u.tokens_out:,} tokens, ~${u.cost_usd:.3f}). "
                "It was stopped to protect cost; very long recordings may need a higher budget.")
        if self.quota is not None and self.quota.count() >= self.daily_request_limit:
            raise DailyQuotaExceeded(
                f"Daily request limit reached ({self.daily_request_limit} Gemini calls today). "
                "New jobs will run after the quota resets.")

    def _backoff(self, attempt: int) -> float:
        # exponential backoff with full jitter
        return random.uniform(0, min(self.max_delay, self.base_delay * (2 ** attempt)))

    # ---- main entry
    def generate_json(self, contents: list, schema: type[BaseModel], *, system: str | None = None,
                      thinking: str = "low", models: list[str] | None = None,
                      on_retry: Callable[[str], None] | None = None) -> tuple[BaseModel, str]:
        """Return (validated object, model that answered)."""
        chain = list(dict.fromkeys(models or [PRIMARY_MODEL, *FALLBACK_MODELS]))
        last: Exception | None = None
        tried_any = False
        for model in chain:
            if not self.breaker.allow(model):
                obs.event(log, "model_skipped_circuit_open", model=model)
                continue
            tried_any = True
            repaired = False
            attempt = 0
            msgs = list(contents)
            while attempt < self.max_attempts:
                self._check_guards()
                u = obs.usage_var.get()
                try:
                    if self.quota is not None:
                        self.quota.incr(model)
                    t0 = time.monotonic()
                    raw = self.backend.generate(model, msgs, system=system, schema=schema, thinking=thinking)
                    if u is not None:
                        u.add(model, raw.tokens_in, raw.tokens_out)
                    obs.event(log, "llm_call", model=model, schema=schema.__name__, seconds=round(time.monotonic() - t0, 2),
                              tokens_in=raw.tokens_in, tokens_out=raw.tokens_out,
                              est_cost_usd=round(obs.est_cost(model, raw.tokens_in, raw.tokens_out), 5))
                except Exception as e:  # noqa: BLE001
                    if u is not None:
                        u.failed_calls += 1
                    kind = classify(e)
                    last = e
                    obs.event(log, "llm_error", logging.WARNING, model=model, kind=kind, attempt=attempt + 1, error=str(e)[:200])
                    if kind == "auth":
                        raise AuthError("The Gemini API key was rejected. Check the GEMINI_API_KEY secret.") from e
                    if kind in ("missing_model", "bad_request"):
                        self.breaker.failure(model, hard=kind == "missing_model")
                        break
                    if kind == "daily_quota":
                        self.breaker.failure(model, hard=True)
                        break
                    self.breaker.failure(model)
                    if not self.breaker.allow(model):
                        break
                    wait = self._backoff(attempt)
                    if on_retry:
                        on_retry(f"{model} busy ({kind}); retrying in {wait:.0f}s")
                    self.sleep(wait)
                    attempt += 1
                    continue
                # validate
                try:
                    obj = schema.model_validate_json(raw.text)
                    self.breaker.success(model)
                    return obj, model
                except ValidationError as ve:
                    last = ve
                    obs.event(log, "schema_invalid", logging.WARNING, model=model, repaired=repaired, error=str(ve)[:200])
                    if repaired:
                        raise SchemaError("The model returned malformed output twice; the job was stopped rather "
                                          "than guessing.", detail=str(ve)[:500]) from ve
                    repaired = True
                    msgs = [*contents, "Previous answer:\n" + raw.text[:20000],
                            REPAIR_PROMPT.format(err=str(ve).splitlines()[0][:200])]
                    continue
        if not tried_any:
            raise CircuitOpen("All Gemini models are temporarily paused after repeated failures. Try again in a few minutes.")
        raise LLMUnavailable("Gemini is busy or out of quota on every configured model. The job can be retried later.",
                             detail=str(last)[:500])


def audio_part(data: bytes, mime: str = "audio/mp3") -> Any:
    from google.genai import types
    return types.Part.from_bytes(data=data, mime_type=mime)


def image_part(data: bytes) -> Any:
    from google.genai import types
    return types.Part.from_bytes(data=data, mime_type="image/jpeg")
