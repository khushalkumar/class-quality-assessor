# Evaluation

**Data.** 12 sample 8x8 class recordings (38–90 min Zoom recordings of children's chess classes), each with a Zoom
`.vtt`, plus 8x8 reviewer feedback. Labels: Excellent ×2, Good ×2, Average ×2, Weak ×2 (→ *Poor*),
"Known quality issues" ×2 (→ *Poor*), Demo ×2 (unlabelled; reported, not scored). IDs below are anonymised
(EX/GO/AV/PO/QI/DE); no names, recordings or transcripts are in this repo.

**Harness.** `eval/run_eval.py` runs the real pipeline. In `vtt` mode the provided Zoom transcript replaces Gemini
transcription (saves quota), **but the mp4 is still used**: 12 frames + 12 top-right speaker crops, ffmpeg silence
and blank-frame statistics, and four 90-second audio excerpts go to the model. `full` mode ignores the `.vtt` and
transcribes the audio with Gemini (the path the hosted app uses). `eval/rescore.py` re-applies the current
aggregation to saved outputs with no API calls.

Metrics: **exact** label match and **within one level** on the 10 labelled classes.

## What the feedback says matters (rubric source)

| Theme in 8x8 feedback | Mentioned for | Rubric dimension (weight) |
|---|---|---|
| Energetic, pleasant, "happier" interaction | missing in Average/Weak/Quality-issue | Energy & warmth (25%) |
| Rapport building, intro, time for new joiners to introduce themselves | praised in Excellent/Good, missing in Weak/QI | Rapport & opening (20%) |
| 8x8 background, face visible, light, camera on, quality issues | flagged in Average/Weak/QI | Professional setup & A/V (20%) |
| Asking each child individually, "considers everyone" | present in almost **all** classes, incl. Weak | Individual engagement (15%) |
| Clarity | Average | Teaching clarity (10%) |
| Homework, conclusion | Good/Average/Weak | Structure, closure & homework (10%) |

**"Quality issue" classes** are not a transcript problem. The feedback for them is about *no 8x8 background*,
*teacher's face not visible*, *no rapport* and the teacher having been *told about quality issues several times*.
A transcript-only model rated them Good/Excellent (v2). They can only be caught from the media: the frames +
upscaled speaker-thumbnail crops (background, face, camera), audio excerpts (tone, audio quality), and the rule
that a *major* red flag caps the rating. After v3 both QI classes are rated Poor in every complete run.

## Runs

| Run | Model (assessment) | Change | Exact | Within one | Notes |
|---|---|---|---|---|---|
| v1 | gemini-3-flash-preview | first rubric (engagement 25%), 1 class smoke test | 1/1 | 1/1 | evidence quality good |
| v2 | gemini-3-flash-preview | full run; timestamp/verdict/privacy prompt fixes | **4/10** | 7/10 | PO-2 Good, QI-1 Excellent, QI-2 Good: over-generous, setup unseen |
| v3 | gemini-3-flash-preview | speaker crops; removed brightness metric; feedback-derived weights; strict anchors; concerns-before-scores; major-flag cap | 2/3 (partial) | – | stopped: network disconnects + camera-"off" false positives |
| v4 | mixed (3-flash → lite fallback) | Zoom-recording artefact rules; strict "major" definition | – | – | invalid: daily quota hit mid-run, models mixed → cache made model-aware |
| v4-lite | gemini-3.1-flash-lite-preview | same prompt | 2/10 | 7/10 | scores compressed to 2–3; EX ≈ GO |
| v4-g36 | gemini-3.6-flash | same prompt | 2/4 (partial) | 3/4 | 8 of 12 calls failed (503 then 20/day free-tier 429) |
| **v4-l35** | **gemini-3.5-flash-lite** | same prompt, initial thresholds 3.95/3.35/2.75 | **5/10** | **9/10** | complete, 1 transient failure (retried) |
| **v4-l35 + recal.** | same outputs | Excellent cut 3.95 → 3.45 (no new calls) | **7/10** | **9/10** | leave-one-out threshold fit: **5/10** |

Frozen default: v4 prompt, `gemini-3.5-flash-lite` for assessment, `gemini-3.1-flash-lite-preview` for
transcription, recalibrated thresholds. The honest generalisation estimate is the leave-one-out number (5/10
exact, 9/10 within one), not the in-sample 7/10.

### Final configuration, per class (v4-l35, recalibrated)

| id | actual | predicted | weighted score | rapport | engagement | clarity | energy | closure | setup |
|---|---|---|---|---|---|---|---|---|---|
| EX-1 | Excellent | Excellent | 3.65 | 3 | 4 | 4 | 3 | 3 | 5 |
| EX-2 | Excellent | Excellent | 3.45 | 3 | 4 | 4 | 3 | 3 | 4 |
| GO-1 | Good | **Poor** | 2.05 | 2 | 3 | 2 | 2 | 1 | 2 |
| GO-2 | Good | Good | 3.40 | 2 | 3 | 4 | 3 | 4 | 5 |
| AV-1 | Average | Average | 2.75 | 2 | 3 | 2 | 2 | 2 | 5 |
| AV-2 | Average | **Poor** | 2.20 | 2 | 2 | 2 | 2 | 2 | 3 |
| PO-1 | Poor | Poor | 2.55 | 1 | 3 | 4 | 2 | 2 | 4 |
| PO-2 | Poor | **Average** | 2.90 | 2 | 3 | 3 | 3 | 2 | 4 |
| QI-1 | Poor | Poor | 2.70 | 2 | 3 | 3 | 3 | 2 | 3 |
| QI-2 | Poor | Poor | 2.40 | 1 | 4 | 3 | 2 | 2 | 3 |
| DE-1 | demo | Poor | 2.20 | | | | | | |
| DE-2 | demo | Excellent | 3.60 | | | | | | |

Misses: **GO-1** is a recording "part" where ~12 minutes is a pre-recorded video and ~11 minutes an unmoderated
tournament; on this segment alone the model's criticism is defensible, but the label is about the teacher overall.
**AV-2 / PO-2** are one level off at the Average/Poor boundary, where the feedback itself differs only in nuance.

## End-to-end (real mp4, no transcript; Gemini transcription)

FULL_RESULTS_PLACEHOLDER

## Qualitative agreement with reviewer feedback

The specific observations usually match what 8x8 reviewers wrote, e.g. EX-2: model flagged *no homework assigned*
(feedback: "can … share HWs") while crediting the branded background and individual questioning; PO-1: *late,
abrupt start straight into revision, new joiner not welcomed, flat tone, student audio complaints* (feedback:
"directly go to revision… without proper rapport building, should give time for new joiners"); QI-1: *a space-themed
virtual background instead of the 8x8 one, abrupt start without greeting, no homework* (feedback: "no background,
need a better happy interaction"); QI-2: *abrupt opening with no greeting or introduction for new students*, plus a
major flag for unprofessional language (feedback: "directly entered into class without proper rapport building";
the face check came back "unclear" rather than "not visible" — a partial miss).

## Signals that turned out misleading

- **Frame brightness** – dark Lichess/Zoom themes read as "poor lighting" → removed from model input.
- **Missing teacher thumbnail** – Zoom recordings often show only the shared screen; the model called this
  "camera off" → explicit artefact rule; "camera off" only when the teacher's own tile is black.
- **Long silences** – usually children solving puzzles or playing a tournament, not dead air.
- **"Asks children individually"** – present in nearly every class, so weighted down.
- **Transcript-derived counts** depend on the transcriber: on EX-2 student turns/10 min were 99 (Zoom VTT) vs 43
  (Gemini), because Gemini merges short replies. The model is told to interpret, not threshold, these numbers.

## Limitations of this evaluation

10 labelled classes, 2 per label; thresholds tuned on the same data (hence leave-one-out reported). LLM runs vary
by up to ~0.4 on the weighted score between runs/models, enough to move borderline classes by one level. Free-tier
quotas (20 requests/day on the larger flash models) prevented repeated runs and self-consistency voting.
