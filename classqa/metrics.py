"""Objective, reproducible interaction metrics computed from a transcript.
These are fed to the LLM as grounding *and* shown to the user, so every
qualitative claim can be cross-checked against a number."""
from __future__ import annotations

import re
from collections import Counter

from .transcript import Transcript, fmt_ts

PRAISE = re.compile(r"\b(good job|well done|very good|excellent|awesome|great|super|brilliant|fantastic|"
                    r"perfect|amazing|nice|good (?:try|answer|move|work)|correct|right answer|clap|bravo|wonderful)\b", re.I)
TECH = re.compile(r"(can'?t hear|cannot hear|not audible|inaudible|voice (?:is )?(?:breaking|cutting|not clear)|"
                  r"(?:not|isn'?t) clear|network|internet|lag|frozen|freez|disconnect|reconnect|unmute|"
                  r"you(?:'re| are) (?:on )?mute|be (?:a little )?louder|echo|screen (?:is )?(?:not|isn'?t) (?:visible|showing)|"
                  r"can'?t see|cannot see|not visible|camera (?:is )?off|turn on (?:your|the) camera|audio)", re.I)
HOMEWORK = re.compile(r"\b(home ?work|hw|assignment|practice (?:at home|this)|worksheet|task for (?:you|next))\b", re.I)
GREETING = re.compile(r"\b(how are you|welcome|good (?:morning|evening|afternoon)|my name|introduce|"
                      r"introduction|tell (?:me|us) about yourself|what did you do)\b", re.I)
CLOSURE = re.compile(r"\b(bye|see you|next class|next session|recap|summary|summari[sz]e|today we (?:learn|saw|did)|"
                     r"what did we learn|any (?:doubts|questions)|thank you)\b", re.I)


def compute(tr: Transcript, duration_s: float | None = None) -> dict:
    segs = tr.segments
    if not segs:
        return {"error": "empty transcript"}
    dur = duration_s or segs[-1].end
    mins = max(dur / 60, 1e-6)
    t_segs = [s for s in segs if s.role == "teacher"]
    s_segs = [s for s in segs if s.role == "student"]

    def talk(ss):
        return sum(max(s.end - s.start, 0.3) for s in ss)

    t_talk, s_talk = talk(t_segs), talk(s_segs)
    t_words = sum(len(s.text.split()) for s in t_segs)
    s_words = sum(len(s.text.split()) for s in s_segs)

    # longest teacher-only stretch: time between student turns while teacher keeps talking
    longest_mono, mono_at, run_start, last_end = 0.0, None, None, None
    for s in segs:
        if s.role == "teacher":
            if run_start is None:
                run_start = s.start
            last_end = s.end
            if last_end - run_start > longest_mono:
                longest_mono, mono_at = last_end - run_start, run_start
        elif s.role == "student":
            run_start = None

    # longest gap with no speech at all (after class starts)
    gaps = [(b.start - a.end, a.end) for a, b in zip(segs, segs[1:], strict=False)]
    longest_gap, gap_at = max(gaps, default=(0.0, None))

    turns = Counter(s.speaker for s in s_segs)
    n_spoke = len(turns)
    balance = round(min(turns.values()) / max(turns.values()), 2) if n_spoke > 1 else (1.0 if n_spoke else 0.0)
    addressed = Counter(lab for s in t_segs for lab in s.names_mentioned)

    def hits(pattern, pool, limit=8):
        out = [(fmt_ts(s.start), s.text[:120]) for s in pool if pattern.search(s.text)]
        return len(out), [f"[{t}] {x}" for t, x in out[:limit]]

    first_t = t_segs[0].start if t_segs else None
    n_tech, tech_ex = hits(TECH, segs)
    n_hw, hw_ex = hits(HOMEWORK, t_segs, 5)
    n_praise, _ = hits(PRAISE, t_segs)
    opening = [s for s in t_segs if s.start < (first_t or 0) + 300]
    closing = [s for s in t_segs if s.start > segs[-1].end - 360]
    n_greet, _ = hits(GREETING, opening)
    n_close, close_ex = hits(CLOSURE, closing, 4)
    t_questions = sum(s.text.count("?") for s in t_segs)
    # first substantive content: first teacher line >= 12 words
    first_content = next((s.start for s in t_segs if len(s.text.split()) >= 12), None)

    return {
        "duration_min": round(dur / 60, 1),
        "transcript_source": tr.source,
        "teacher_first_speaks_at": fmt_ts(first_t) if first_t is not None else None,
        "rapport_window_min": round((first_content - first_t) / 60, 1) if first_content and first_t is not None else None,
        "teacher_talk_pct": round(100 * t_talk / (t_talk + s_talk + 1e-6), 1),
        "student_talk_pct": round(100 * s_talk / (t_talk + s_talk + 1e-6), 1),
        "student_word_share_pct": round(100 * s_words / (t_words + s_words + 1e-6), 1),
        "teacher_questions": t_questions,
        "teacher_questions_per_10min": round(10 * t_questions / mins, 1),
        "student_turns": len(s_segs),
        "student_turns_per_10min": round(10 * len(s_segs) / mins, 1),
        "distinct_students_spoke": n_spoke,
        "participation_balance": balance,          # min/max turns across students (1 = even)
        "distinct_students_addressed_by_name": len(addressed),
        "name_addresses_per_10min": round(10 * sum(addressed.values()) / mins, 1),
        "praise_per_10min": round(10 * n_praise / mins, 1),
        "longest_teacher_monologue_min": round(longest_mono / 60, 1),
        "longest_teacher_monologue_at": fmt_ts(mono_at) if mono_at is not None else None,
        "longest_silence_min": round(longest_gap / 60, 1),
        "longest_silence_at": fmt_ts(gap_at) if gap_at is not None else None,
        "greeting_lines_in_opening": n_greet,
        "closure_lines_in_last_6min": n_close,
        "closure_examples": close_ex,
        "homework_mentions": n_hw,
        "homework_examples": hw_ex,
        "tech_issue_mentions": n_tech,
        "tech_issue_examples": tech_ex,
    }
