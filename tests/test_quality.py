"""Trust layer: quote verification, self-consistency, review routing, aggregation."""
import json

from conftest import assessment_json

from classqa import quality, rubric
from classqa.pipeline import build_report
from classqa.schemas import Assessment
from classqa.transcript import from_vtt_text, render

VTT = """WEBVTT

1
00:00:30.000 --> 00:00:33.000
Ms Tutor: Good evening everyone! Welcome, Alice. How are you?

2
00:00:34.000 --> 00:00:35.000
Alice Parent: I'm fine ma'am.

3
00:00:40.000 --> 00:00:48.000
Ms Tutor: Today we learn about forks. Bob, what is a fork in chess? Can you tell me?

4
00:00:50.000 --> 00:00:53.000
Bob K: It attacks two pieces at once.

5
00:40:00.000 --> 00:40:06.000
Ms Tutor: For homework, solve five fork puzzles. See you next class, bye!
"""


def test_pseudonymisation():
    text = render(from_vtt_text(VTT))
    assert "Alice" not in text and "Bob" not in text and "Tutor" not in text and "[Student 2]" in text


def test_quote_verification():
    tr = render(from_vtt_text(VTT))
    corpus = quality._norm(tr)
    toks = corpus.split()
    assert quality.verify_quote("Today we learn about forks.", corpus, toks) == "exact"
    assert quality.verify_quote("[student], what is a fork in chess?", corpus, toks) == "exact"
    assert quality.verify_quote("For homework solve 5 fork puzzles", corpus, toks) == "approximate"
    assert quality.verify_quote("You must always castle early, children", corpus, toks) == "unverified"
    assert quality.verify_quote("(visual)", corpus, toks) == "nonverbal"


def test_hallucinated_quotes_removed_and_counted():
    a = json.loads(assessment_json(quote="The teacher shouted at everyone loudly today"))
    stats = quality.verify_evidence(a, render(from_vtt_text(VTT)))
    ev = a["dimensions"][0]["evidence"][0]
    assert ev["quote_status"] == "unverified" and "removed" in ev["quote"] and ev["unverified_quote"]
    assert stats["unverified_rate"] == 1.0


def test_merge_runs_detects_disagreement():
    hi = {k: 5 for k in rubric.WEIGHTS}
    lo = {k: 2 for k in rubric.WEIGHTS}
    mean, ratings, disagree = quality.merge_runs([hi, lo])
    assert disagree and ratings == ["Excellent", "Poor"] and mean["energy_warmth"] == 3.5
    assert not quality.merge_runs([hi, hi])[2]


def test_borderline():
    assert quality.borderline(3.42) is not None and quality.borderline(3.40, margin=0.1) is not None
    assert quality.borderline(4.5) is None


def _report(*runs, missing=()):
    class Ctx:
        missing_signals = list(missing)
        models_used = {}
    tr = from_vtt_text(VTT)
    return build_report([(Assessment.model_validate_json(r), "m") for r in runs], {}, {}, {}, tr, Ctx())


def test_confident_result_auto_accepted():
    r = _report(assessment_json(score=5, quote="Today we learn about forks", holistic="Excellent"))
    assert r["rating"] == "Excellent" and not r["needs_review"], r["review_reasons"]
    assert r["versions"]["rubric"].startswith("r-")


def test_major_flag_caps_and_routes_to_review():
    r = _report(assessment_json(score=5, quote="Today we learn about forks", holistic="Excellent", flag="major"))
    assert r["rating"] == "Average" and r["capped_by_red_flag"] and r["needs_review"]


def test_disagreeing_runs_go_to_review():
    r = _report(assessment_json(score=5, holistic="Excellent"), assessment_json(score=2, holistic="Poor"))
    assert r["needs_review"] and r["confidence"] == "Low" and r["self_consistency"]["disagree"]


def test_missing_video_goes_to_review():
    r = _report(assessment_json(score=5, quote="Today we learn about forks", holistic="Excellent"),
                missing=["video_frames"])
    assert r["needs_review"] and any("video" in x for x in r["review_reasons"])


def test_thresholds():
    assert rubric.to_rating(rubric.weighted_score({k: 5 for k in rubric.WEIGHTS})) == "Excellent"
    assert rubric.to_rating(rubric.weighted_score({k: 3 for k in rubric.WEIGHTS})) == "Average"
    assert rubric.to_rating(rubric.weighted_score({k: 1 for k in rubric.WEIGHTS})) == "Poor"
