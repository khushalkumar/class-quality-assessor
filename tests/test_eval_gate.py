"""CI regression gate (offline, no API key): re-score the frozen, anonymised eval outputs with the current rubric
and fail if agreement with the 8x8 labels drops below the floors in eval/frozen/gate.json."""
import json
import sys
from pathlib import Path

import pytest

from classqa import evalmetrics as em
from classqa import rubric

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
import advanced_eval  # noqa: E402

GATE = json.loads((EVAL / "frozen" / "gate.json").read_text())


def gate_failures(res: dict) -> list[str]:
    bad = []
    for split in ("in_sample", "leave_one_out"):
        for k, floor in GATE[split].items():
            if res[split][k] < floor:
                bad.append(f"{split}.{k} = {res[split][k]} < {floor}")
    q = res["quotes"]
    if q["verified_rate"] < GATE["quote_verified_rate"]:
        bad.append(f"quote verified rate {q['verified_rate']} < {GATE['quote_verified_rate']}")
    if q["unverified"] > GATE["max_unverified_quotes"]:
        bad.append(f"unverified quotes {q['unverified']} > {GATE['max_unverified_quotes']}")
    return bad


def test_frozen_outputs_are_anonymised():
    raw = (EVAL / "frozen" / f"{GATE['tag']}.json").read_text()
    data = json.loads(raw)
    allowed = {"id", "actual", "dimension_scores", "capped", "holistic", "confidence", "major_flags", "quotes",
               "model", "prompt_version"}
    for c in data["classes"]:
        assert set(c) <= allowed              # numbers and labels only - no text fields
    assert len(data["classes"]) == 12 and sum(c["actual"] is not None for c in data["classes"]) == 10


def test_regression_gate():
    res = advanced_eval.compute(GATE["tag"])
    assert res["n"] == 10
    assert not gate_failures(res), "eval regression: " + "; ".join(gate_failures(res))


def test_gate_catches_a_bad_rubric_change(monkeypatch):
    # sanity check that the gate has teeth: a sloppy threshold change must fail it
    monkeypatch.setattr(rubric, "THRESHOLDS", [(3.0, "Excellent"), (2.6, "Good"), (2.2, "Average"), (0.0, "Poor")])
    assert gate_failures(advanced_eval.compute(GATE["tag"]))


# ------------------------------------------------------------------ metric unit tests
def test_quadratic_kappa_known_values():
    R = rubric.RATINGS
    assert em.quadratic_kappa(R, R) == pytest.approx(1.0)
    assert em.quadratic_kappa(R, R[::-1]) == pytest.approx(-1.0)
    # hand-computed: one adjacent error in four -> 1 - (1/9) / (20/9 * 1/4 ...) ; check against direct formula
    a, p = ["Poor", "Average", "Good", "Excellent"], ["Poor", "Average", "Good", "Good"]
    assert 0.8 < em.quadratic_kappa(a, p) < 1.0


def test_confusion_and_counts():
    a = ["Poor", "Poor", "Good", "Excellent"]
    p = ["Poor", "Average", "Excellent", "Excellent"]
    m = em.confusion(a, p)
    assert m[0] == [1, 1, 0, 0] and m[2] == [0, 0, 0, 1] and m[3] == [0, 0, 0, 1]
    assert em.exact(a, p) == 2 and em.within_one(a, p) == 4 and em.mae_levels(a, p) == 0.5


def test_label_respects_cap_and_loo_does_not_peek():
    assert em.label(4.0, (2.75, 3.35, 3.45)) == "Excellent"
    assert em.label(4.0, (2.75, 3.35, 3.45), capped=True) == rubric.MAJOR_FLAG_CAP
    rows = [{"actual": "Poor", "score": 2.0, "capped": False}, {"actual": "Poor", "score": 2.1, "capped": False},
            {"actual": "Excellent", "score": 4.0, "capped": False}, {"actual": "Excellent", "score": 4.1, "capped": False},
            {"actual": "Excellent", "score": 2.05, "capped": False}]   # outlier: only predictable if it peeks
    assert em.leave_one_out(rows)[-1] == "Poor"


def test_spearman():
    assert em.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert em.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
