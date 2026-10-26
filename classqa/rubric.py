"""The assessment rubric.

Derived from (a) 8x8's own reviewer feedback on the labelled sample classes and
(b) contrasting the transcripts/recordings of Excellent vs Poor classes.
What 8x8 reviewers repeatedly reward or penalise:
  * rapport building + a proper intro before content; time for new joiners to introduce themselves
  * calling on every child individually ("consider everyone in the class")
  * energetic, pleasant, happy interaction
  * homework given; a proper conclusion
  * professional setup: 8x8 branded background, camera on, face visible, good light, no A/V issues
Weights follow how often each theme decides a label in the feedback
(energy/pleasantness, rapport/intro and setup separate the top from the bottom;
individual questioning is present in almost every class, so it discriminates less).
Weights are deliberately simple and were sanity-checked on the labelled set
(see EVALUATION.md) - not fitted per-sample, to stay general on unseen classes.
"""
from __future__ import annotations

RATINGS = ["Poor", "Average", "Good", "Excellent"]

DIMENSIONS = [
    {
        "key": "rapport_opening",
        "name": "Rapport & opening",
        "weight": 0.20,
        "what": "Warm greeting of each child, teacher/self introduction, time for new joiners to introduce "
                "themselves, small talk or a hook before content, clear statement of today's topic.",
        "anchors": {
            1: "Goes straight into content/revision ('let's start', 'what did we do last class') with no real rapport; new joiners ignored.",
            3: "Says hello/good evening and states the topic, but no personal conversation with the children and no self-introduction time for new joiners.",
            5: "Greets each child by name as they join, genuine small talk (how was your day/school), new joiners introduce themselves, topic framed with a hook.",
        },
    },
    {
        "key": "individual_engagement",
        "name": "Individual engagement & participation",
        "weight": 0.15,
        "what": "Does the teacher call on children individually by name, involve EVERY child (not just the most "
                "vocal), let them answer/think aloud, make moves/solve puzzles themselves? Student talk share, "
                "questions per 10 min and student turns per 10 min are the objective proxies.",
        "anchors": {
            1: "Lecture; few questions; one or two children answer; others silent for long stretches.",
            3: "Regular questions but mostly open to the room; some children rarely involved.",
            5: "Constant back-and-forth; every child is individually asked and responds; children do the thinking.",
        },
    },
    {
        "key": "teaching_clarity",
        "name": "Teaching clarity & checking understanding",
        "weight": 0.10,
        "what": "Clear, age-appropriate explanations; correct content; uses the board/screen/examples well; "
                "checks understanding (asks why, asks children to explain); corrects mistakes constructively; sensible pacing.",
        "anchors": {
            1: "Confusing or rushed explanations; wrong answers simply dismissed ('no') with no explanation.",
            3: "Mostly clear; some explanation; limited checking of understanding.",
            5: "Very clear, structured explanations; mistakes turned into learning; understanding checked often.",
        },
    },
    {
        "key": "energy_warmth",
        "name": "Energy, warmth & encouragement",
        "weight": 0.25,
        "what": "Energetic, cheerful, pleasant tone suitable for children; praise and encouragement; patience; "
                "positive classroom management. Judge tone from the audio excerpts when provided.",
        "anchors": {
            1: "Flat, low-energy, curt or unprofessional language; wrong answers met with a bare 'no'; children sound disengaged.",
            3: "Calm, polite and patient, but neutral/instructional energy - this is the DEFAULT for a competent tutor. Routine 'very good' / 'correct' does not raise the score.",
            5: "Consistently animated, joyful voice; laughter and fun; specific, enthusiastic praise; children clearly enjoy it. Reserve 5 for this.",
        },
    },
    {
        "key": "structure_closure",
        "name": "Lesson structure, closure & homework",
        "weight": 0.10,
        "what": "Logical flow (recap -> new concept -> practice/puzzles -> summary); proper conclusion; "
                "homework/practice assigned and explained; next class previewed.",
        "anchors": {
            1: "No discernible structure; class just stops; no homework.",
            3: "Some structure; brief ending; homework missing or only mentioned in passing.",
            5: "Clear arc with recap and summary; homework clearly assigned; warm goodbye.",
        },
    },
    {
        "key": "professional_setup",
        "name": "Professional setup & A/V quality",
        "weight": 0.20,
        "what": "Teacher camera ON with face clearly visible and well lit, 8x8 branded virtual background, clear audio "
                "(no repeated 'can't hear you'/'not clear'), stable connection, starts on time, screen share readable.",
        "anchors": {
            1: "Camera off/face not visible for long periods, no 8x8 background, or repeated audio/video problems.",
            3: "Mostly fine but with noticeable issues (dim lighting, occasional audio complaints, late start).",
            5: "Branded background, face visible and well lit throughout, clear audio, punctual, crisp screen share.",
        },
    },
]

WEIGHTS = {d["key"]: d["weight"] for d in DIMENSIONS}
assert abs(sum(WEIGHTS.values()) - 1) < 1e-9

# weighted 1-5 score -> rating. Calibrated on the labelled set (EVALUATION.md).
THRESHOLDS = [(3.95, "Excellent"), (3.35, "Good"), (2.75, "Average"), (0.0, "Poor")]

# a "major" red flag (e.g. unprofessional language, camera off most of the class, class unusable due to
# A/V problems) caps the rating, mirroring how 8x8 treats conduct/quality issues.
MAJOR_FLAG_CAP = "Average"


def rubric_text() -> str:
    out = []
    for i, d in enumerate(DIMENSIONS, 1):
        out.append(f"{i}. {d['key']} - {d['name']} (weight {int(d['weight']*100)}%)\n   Look for: {d['what']}")
        for s, a in d["anchors"].items():
            out.append(f"   score {s}: {a}")
    out.append("   (scores 2 and 4 are in-between levels)")
    return "\n".join(out)


def weighted_score(scores: dict[str, float]) -> float:
    return round(sum(WEIGHTS[k] * float(scores[k]) for k in WEIGHTS if k in scores)
                 / sum(WEIGHTS[k] for k in WEIGHTS if k in scores), 2)


def to_rating(score: float) -> str:
    for th, r in THRESHOLDS:
        if score >= th:
            return r
    return "Poor"
