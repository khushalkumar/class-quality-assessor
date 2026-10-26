"""Observability: structured JSON logs, per-stage timing, token/cost accounting.

Cost figures are ESTIMATES from a static price table (USD per 1M tokens) and are
only used for budgeting and dashboards - check current vendor pricing."""
from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

# Estimated list prices (USD / 1M tokens, input incl. audio/image, output incl. thinking).
PRICES = {
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite-preview": (0.10, 0.40),
    "gemini-flash-lite-latest": (0.10, 0.40),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3.6-flash": (0.50, 3.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}
DEFAULT_PRICE = (0.50, 3.00)

job_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_id", default=None)


def est_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = PRICES.get(model, DEFAULT_PRICE)
    return tokens_in / 1e6 * pin + tokens_out / 1e6 * pout


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name,
             "msg": record.getMessage(), "job_id": getattr(record, "job_id", None) or job_id_var.get()}
        extra = getattr(record, "fields", None)
        if extra:
            d.update(extra)
        return json.dumps(d, default=str)


_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger("classqa")
    root.handlers[:] = [h]
    root.setLevel(os.environ.get("CLASSQA_LOG_LEVEL", "INFO"))
    root.propagate = False
    _configured = True


def event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})


@dataclass
class Usage:
    """Per-job accounting, accumulated across all LLM calls of the job."""
    calls: int = 0
    failed_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    by_model: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)   # stage -> seconds
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, model: str, tin: int, tout: int) -> None:
        with self._lock:
            self._add(model, tin, tout)

    def _add(self, model: str, tin: int, tout: int) -> None:
        self.calls += 1
        self.tokens_in += tin
        self.tokens_out += tout
        c = est_cost(model, tin, tout)
        self.cost_usd += c
        m = self.by_model.setdefault(model, {"calls": 0, "tokens_in": 0, "tokens_out": 0})
        m["calls"] += 1
        m["tokens_in"] += tin
        m["tokens_out"] += tout

    def to_dict(self) -> dict:
        return {"calls": self.calls, "failed_calls": self.failed_calls, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out, "est_cost_usd": round(self.cost_usd, 5),
                "by_model": self.by_model, "stage_seconds": {k: round(v, 2) for k, v in self.stages.items()}}


usage_var: contextvars.ContextVar[Usage | None] = contextvars.ContextVar("usage", default=None)


@contextmanager
def stage_timer(logger: logging.Logger, stage: str):
    t0 = time.monotonic()
    event(logger, "stage_start", stage=stage)
    ok = False
    try:
        yield
        ok = True
    finally:
        dt = time.monotonic() - t0
        u = usage_var.get()
        if u is not None:
            u.stages[stage] = u.stages.get(stage, 0.0) + dt
        event(logger, "stage_end", stage=stage, seconds=round(dt, 2), ok=ok)
