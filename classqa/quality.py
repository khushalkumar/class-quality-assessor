"""Trust layer between the model and the reviewer:
  * verify every evidence quote against the transcript (flag/strip hallucinated quotes)
  * merge self-consistency runs and detect disagreement
  * decide whether a result is safe to auto-accept or must go to human review"""
from __future__ import annotations

import os
import re
from difflib import SequenceMatcher

from . import rubric

NONVERBAL = {"(visual)", "(audio)", "visual", "audio", ""}
_label = re.compile(r"\[[^\]]*\]")
_punct = re.compile(r"[^\w\s]")


def _norm(s: str) -> str:
    s = _label.sub(" ", s.lower().replace("’", "'"))
    s = _punct.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def verify_quote(quote: str, corpus_norm: str, corpus_tokens: list[str], threshold: float = 0.8) -> str:
    """-> 'nonverbal' | 'exact' | 'approximate' | 'unverified'"""
    q = quote.strip().strip('"“”')
    if q.lower() in NONVERBAL:
        return "nonverbal"
    parts = [_norm(p) for p in re.split(r"\.\.\.|…", q)]
    parts = [p for p in parts if len(p.split()) >= 2] or [_norm(q)]
    if all(p and p in corpus_norm for p in parts):
        return "exact"
    # fuzzy: best window match for the longest part
    p = max(parts, key=len)
    toks = p.split()
    n = len(toks)
    if n == 0:
        return "nonverbal"
    best = 0.0
    first = set(toks[:3])
    for i in range(0, max(len(corpus_tokens) - n + 1, 1)):
        if corpus_tokens[i] not in first and i % 3:
            continue  # cheap pruning
        best = max(best, SequenceMatcher(None, toks, corpus_tokens[i:i + n]).ratio())
        if best >= threshold:
            return "approximate"
    return "unverified"


def verify_evidence(assessment: dict, transcript_text: str) -> dict:
    """Annotate each evidence item with `quote_status`; blank unverified quotes. Returns stats."""
    corpus = _norm(transcript_text)
    tokens = corpus.split()
    stats = {"exact": 0, "approximate": 0, "unverified": 0, "nonverbal": 0}
    for d in assessment.get("dimensions", []):
        for ev in d.get("evidence", []):
            st = verify_quote(ev.get("quote", ""), corpus, tokens)
            stats[st] += 1
            ev["quote_status"] = st
            if st == "unverified":
                ev["unverified_quote"] = ev.get("quote", "")
                ev["quote"] = "(quote not found in transcript - removed)"
    verbal = stats["exact"] + stats["approximate"] + stats["unverified"]
    stats["unverified_rate"] = round(stats["unverified"] / verbal, 2) if verbal else 0.0
    return stats


def merge_runs(score_sets: list[dict[str, int]]) -> tuple[dict[str, float], list[str], bool]:
    """Average dimension scores across self-consistency runs.
    Returns (mean scores, per-run ratings, disagreement flag)."""
    keys = rubric.WEIGHTS.keys()
    mean = {k: round(sum(s.get(k, 3) for s in score_sets) / len(score_sets), 2) for k in keys}
    ratings = [rubric.to_rating(rubric.weighted_score(s)) for s in score_sets]
    idx = [rubric.RATINGS.index(r) for r in ratings]
    return mean, ratings, (max(idx) - min(idx)) >= 1


# Text inside a transcript that addresses the AI grader rather than the class. The transcript is untrusted data:
# the label is computed in code from dimension scores, quotes are verified, and these hits force human review.
_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)"
    r"|(system|developer)\s+prompt"
    r"|you\s+are\s+(now\s+)?(an?\s+)?(ai|assistant|language\s+model|chatgpt|gemini)\b"
    r"|\b(rate|score|grade|mark)\s+(this|the)\s+(class|lesson|teacher|session)\s+(as\s+)?(excellent|good|5|five|top)"
    r"|(give|assign)\s+(this|the)\s+(class|lesson|teacher)\s+(a\s+)?(rating|score)\s+of"
    r"|\bnew\s+instructions\s*:", re.I)


def injection_markers(transcript_text: str) -> int:
    """Number of instruction-like phrases aimed at an AI grader found in the transcript."""
    return len(_INJECTION.findall(transcript_text))


BORDERLINE_MARGIN = float(os.environ.get("CLASSQA_BORDERLINE_MARGIN", "0.05"))


def borderline(score: float, margin: float | None = None) -> float | None:
    """Distance to the nearest threshold if within margin, else None."""
    margin = BORDERLINE_MARGIN if margin is None else margin
    d = min(abs(score - th) for th, _ in rubric.THRESHOLDS if th > 0)
    return round(d, 2) if round(d, 3) < margin else None


def review_reasons(*, confidence: str, rating: str, holistic: str, weighted: float, run_ratings: list[str],
                   quote_stats: dict, missing_signals: list[str], major_flags: int,
                   capped: bool = False, injection_hits: int = 0) -> list[str]:
    reasons = []
    if injection_hits:
        reasons.append(f"The transcript contains {injection_hits} instruction-like phrase(s) addressed to an AI "
                       "grader (possible prompt injection) - check the rating by hand.")
    if confidence == "Low":
        reasons.append("Model confidence is low.")
    gap = abs(rubric.RATINGS.index(rating) - rubric.RATINGS.index(holistic))
    if gap >= 2:
        reasons.append(f"Rubric score ({rating}) and the model's holistic view ({holistic}) differ by {gap} levels.")
    if len(set(run_ratings)) > 1:
        reasons.append(f"Independent scoring runs disagreed: {', '.join(run_ratings)}.")
    b = borderline(weighted)
    if b is not None:
        reasons.append(f"Score {weighted:.2f} is within {b:.2f} of a rating boundary.")
    if quote_stats.get("unverified_rate", 0) > 0.34:
        reasons.append(f"{quote_stats['unverified']} evidence quotes could not be found in the transcript.")
    for m in missing_signals:
        if m in ("video_frames", "audio_excerpts"):
            reasons.append(f"Assessed without {m.replace('_', ' ')} - visual/tone checks are incomplete.")
    if major_flags:
        reasons.append("A major red flag capped the rating - please confirm it." if capped
                       else "A major red flag was raised - please confirm it.")
    return reasons
