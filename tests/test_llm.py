"""Gemini failure handling with a fake transport."""
import pytest
from conftest import FakeBackend, assessment_json

from classqa import obs
from classqa.errors import AuthError, BudgetExceeded, CircuitOpen, DailyQuotaExceeded, LLMUnavailable, SchemaError
from classqa.llm import CircuitBreaker, classify
from classqa.schemas import Assessment

M = ["model-a", "model-b"]


def E(msg):
    return RuntimeError(msg)


def test_classify():
    assert classify(E("429 RESOURCE_EXHAUSTED requests per day limit exceeded")) == "daily_quota"
    assert classify(E("429 RESOURCE_EXHAUSTED per minute")) == "rate_limit"
    assert classify(E("503 UNAVAILABLE")) == "transient"
    assert classify(E("Server disconnected without sending a response.")) == "transient"
    assert classify(E("400 API key not valid")) == "auth"
    assert classify(E("404 NOT_FOUND model")) == "missing_model"


def test_rate_limit_backoff_then_success(make_llm):
    sleeps = []
    b = FakeBackend(script=[E("429 RESOURCE_EXHAUSTED per minute"), E("503 UNAVAILABLE")])
    llm = make_llm(b, sleep=sleeps.append)
    obj, model = llm.generate_json(["x"], Assessment, models=M)
    assert model == "model-a" and len(sleeps) == 2 and all(s >= 0 for s in sleeps)


def test_backoff_has_jitter_and_cap(make_llm):
    llm = make_llm(max_delay=10, base_delay=2)
    vals = {round(llm._backoff(6), 3) for _ in range(20)}
    assert len(vals) > 1 and max(vals) <= 10


def test_fallback_when_primary_keeps_failing(make_llm):
    b = FakeBackend(script=[E("503 UNAVAILABLE")] * 3)   # breaker threshold 3 -> open -> next model
    obj, model = make_llm(b).generate_json(["x"], Assessment, models=M)
    assert model == "model-b"


def test_daily_quota_skips_model_for_the_day(make_llm):
    b = FakeBackend(script=[E("429 RESOURCE_EXHAUSTED Generate content requests per day limit exceeded")])
    llm = make_llm(b)
    _, model = llm.generate_json(["x"], Assessment, models=M)
    assert model == "model-b" and not llm.breaker.allow("model-a")
    _, model = llm.generate_json(["x"], Assessment, models=M)   # still skipped on the next job
    assert model == "model-b"


def test_circuit_open_everywhere(make_llm):
    t = [0.0]
    br = CircuitBreaker(threshold=1, cooldown_s=60, clock=lambda: t[0])
    llm = make_llm(FakeBackend(script=[E("503")] * 2), breaker=br)
    with pytest.raises(LLMUnavailable):
        llm.generate_json(["x"], Assessment, models=M)
    with pytest.raises(CircuitOpen):
        llm.generate_json(["x"], Assessment, models=M)
    t[0] = 61   # cooldown passed -> half-open, call succeeds
    assert llm.generate_json(["x"], Assessment, models=M)[1] == "model-a"


def test_auth_error_not_retried(make_llm):
    b = FakeBackend(script=[E("400 INVALID_ARGUMENT API key not valid")])
    with pytest.raises(AuthError):
        make_llm(b).generate_json(["x"], Assessment, models=M)
    assert len(b.calls) == 1


def test_malformed_json_repaired_once(make_llm):
    b = FakeBackend(script=['{"class_summary": "oops"'])
    obj, _ = make_llm(b).generate_json(["x"], Assessment, models=M)
    assert obj.class_summary and len(b.calls) == 2


def test_malformed_json_twice_is_structured_failure(make_llm):
    b = FakeBackend(script=["not json", '{"also": "wrong"}'])
    with pytest.raises(SchemaError) as e:
        make_llm(b).generate_json(["x"], Assessment, models=M)
    assert e.value.code == "malformed_output"


def test_out_of_range_score_rejected_by_schema(make_llm):
    bad = assessment_json(score=9)
    b = FakeBackend(script=[bad, bad])
    with pytest.raises(SchemaError):
        make_llm(b).generate_json(["x"], Assessment, models=M)


def test_job_budget_guard(make_llm):
    u = obs.Usage(tokens_in=5_000_000)
    tok = obs.usage_var.set(u)
    try:
        with pytest.raises(BudgetExceeded):
            make_llm(job_token_budget=1_000_000).generate_json(["x"], Assessment, models=M)
    finally:
        obs.usage_var.reset(tok)


def test_daily_quota_guard(make_llm, store):
    llm = make_llm(daily_request_limit=2)
    llm.generate_json(["x"], Assessment, models=M)
    llm.generate_json(["x"], Assessment, models=M)
    assert store.count() == 2
    with pytest.raises(DailyQuotaExceeded):
        llm.generate_json(["x"], Assessment, models=M)


def test_usage_and_cost_accounted(make_llm):
    u = obs.Usage()
    tok = obs.usage_var.set(u)
    try:
        make_llm().generate_json(["x"], Assessment, models=["gemini-3.5-flash-lite"])
    finally:
        obs.usage_var.reset(tok)
    assert u.calls == 1 and u.tokens_in == 30000 and u.cost_usd > 0
