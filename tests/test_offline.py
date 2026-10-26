"""Offline tests (no API calls, no real data): parsing, pseudonymisation, metrics,
aggregation and report rendering on a synthetic class."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classqa import metrics, pipeline, rubric  # noqa: E402
from classqa.report import to_markdown  # noqa: E402
from classqa.schemas import Assessment  # noqa: E402
from classqa.transcript import from_vtt_text, render  # noqa: E402

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
00:00:54.000 --> 00:00:57.000
Ms Tutor: Excellent, Bob! Very good. Sorry, I can't hear you, Alice, you are on mute.

6
00:40:00.000 --> 00:40:06.000
Ms Tutor: For homework, solve five fork puzzles. See you next class, bye!
"""


def test_parse_and_pseudonymise():
    tr = from_vtt_text(VTT)
    assert [s.speaker for s in tr.segments[:2]] == ["Teacher", "Student 1"]
    text = render(tr)
    assert "Alice" not in text and "Bob" not in text and "Tutor" not in text
    assert "[Student 2]" in text


def test_metrics():
    m = metrics.compute(from_vtt_text(VTT), duration_s=2410)
    assert m["teacher_questions"] >= 3
    assert m["distinct_students_spoke"] == 2
    assert m["homework_mentions"] == 1
    assert m["tech_issue_mentions"] >= 1
    assert m["distinct_students_addressed_by_name"] == 2


def test_rating_thresholds():
    assert rubric.to_rating(rubric.weighted_score({k: 5 for k in rubric.WEIGHTS})) == "Excellent"
    assert rubric.to_rating(rubric.weighted_score({k: 3 for k in rubric.WEIGHTS})) == "Average"
    assert rubric.to_rating(rubric.weighted_score({k: 1 for k in rubric.WEIGHTS})) == "Poor"


def _assessment(score=5, flag="minor"):
    ev = [{"timestamp": "00:40", "observation": "x", "quote": "y"}]
    return Assessment.model_validate({
        "class_summary": "s",
        "visual": {"teacher_camera_on": "throughout", "teacher_face_visible": "clearly",
                   "branded_background": "8x8 branded", "lighting": "good", "screen_share_used": True, "notes": "n"},
        "reviewer_concerns": [],
        "dimensions": [{"key": k, "score": score, "summary": "s", "evidence": ev} for k in rubric.WEIGHTS],
        "strengths": [{"title": "t", "detail": "d", "timestamp": "00:01"}],
        "improvements": [{"title": "t", "detail": "d", "timestamp": "00:01", "coaching_tip": "c"}],
        "red_flags": [{"issue": "i", "timestamp": "01:00", "severity": flag}],
        "holistic_rating": "Excellent", "confidence": "High", "verdict": "v"})


def test_major_flag_caps_rating_and_report_renders():
    tr = from_vtt_text(VTT)
    m = metrics.compute(tr)
    rep = pipeline.build_report(_assessment(5, "minor"), m, {}, {}, tr, has_video=True)
    assert rep["rating"] == "Excellent"
    rep = pipeline.build_report(_assessment(5, "major"), m, {}, {}, tr, has_video=True)
    assert rep["rating"] == "Average" and rep["capped_by_red_flag"]
    md = to_markdown(rep)
    assert "Class quality report" in md and "capped" in md
