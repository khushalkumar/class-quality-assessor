"""End-to-end pipeline: recording (+ optional transcript) -> Assessment report."""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from . import llm, media, metrics as M, rubric
from .schemas import Assessment, TranscriptChunk
from .transcript import Transcript, build_transcript, fmt_ts, from_vtt_text, render

PROMPT_VERSION = "v4"
CHUNK_S = 600          # transcription chunk length
N_FRAMES = 12          # frames shown to the model
EXCERPT_S = 90         # length of each audio excerpt for tone/energy judgement

Progress = Callable[[float, str], None]

SYSTEM = """You are an experienced academic quality reviewer at 8x8 (Eight Times Eight), an online chess academy \
for children aged roughly 5-15. Classes run on Zoom: a tutor teaches a small group using a shared chess board \
(lichess etc.), puzzles and discussion. You review recorded classes and rate them Poor / Average / Good / Excellent \
for the academic team, the way 8x8's own reviewers do.

8x8 reviewers care most about: rapport building and a proper intro before content (including letting new joiners \
introduce themselves); individually involving EVERY child by name; energetic, pleasant, happy communication; clear \
teaching; a proper conclusion with homework; and professional setup (8x8 branded background, camera on, face \
visible, good lighting, no audio/video problems).

Rules:
- Ground every claim in evidence: cite timestamps (MM:SS) and short verbatim quotes from the transcript, or frame \
timestamps for visual claims. Never invent quotes. If something cannot be observed, say so instead of guessing.
- Be specific to this class; avoid generic statements like "the teacher was engaging".
- Use the objective metrics as grounding, but interpret them (e.g. a quiet puzzle-solving period is not dead air; \
a transcript may mislabel speakers).
- Calibrate against 8x8's bar, which is high: asking children questions individually is EXPECTED in every class \
and is not by itself a sign of quality. What separates classes is (a) genuinely energetic, happy, pleasant \
communication, (b) real rapport building and a proper intro before content (including time for new joiners to \
introduce themselves), (c) professional setup (8x8 branded background, face clearly visible and lit), and \
(d) a proper conclusion with homework. A calm, polite, competent tutor who jumps into content, has neutral energy \
and skips homework is Average at best. Be sceptical: list concerns first, then score. A score of 4-5 needs \
explicit positive evidence; absence of evidence is a 3 or lower.
- Recording artefacts are not teacher faults: Zoom recordings often show ONLY the shared screen, so a missing \
teacher thumbnail during screen share does NOT mean the camera is off. Say "camera off" only when the teacher's own \
tile is visible as a black tile / name card. Quiet periods while children play a game or solve puzzles are not \
dead air. Pre-class waiting before the teacher arrives is not part of the lesson.
- Timestamps are always class time. For audio excerpts, add the offset inside the excerpt to the excerpt's start time.
- Privacy: children are minors. NEVER output a child's or teacher's real name, even inside quotes - replace any \
name with [student] / [teacher] or keep the [Student N] labels.
- The `verdict` explains the overall quality and the decisive evidence; do not name a rating label in it (the \
final label is computed from your dimension scores).
"""


def _file_key(path: Path) -> str:
    h = hashlib.sha256()
    st = path.stat()
    h.update(f"{st.st_size}".encode())
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
        f.seek(max(st.st_size - (4 << 20), 0))
        h.update(f.read())
    return h.hexdigest()[:24]


def _parse_rel(ts: str) -> float | None:
    p = [int(x) for x in re.findall(r"\d+", ts)[:3]]
    if not p:
        return None
    if len(p) == 3:
        return p[0] * 3600 + p[1] * 60 + p[2]
    if len(p) == 2:
        return p[0] * 60 + p[1]
    return float(p[0])


# ---------------------------------------------------------------- transcription
TRANSCRIBE_PROMPT = """Transcribe this audio clip from an online children's chess class (Zoom).
- One entry per utterance (a sentence or two). `start` is the time from the start of THIS clip, MM:SS.
- speaker: "Teacher" for the adult tutor; for a child use "Student: <first name>" when the name is clear \
from context (e.g. the teacher just called on them), otherwise "Student". A parent/other adult: "Other adult".
- If speech is not English, translate it to English and prefix the text with "(translated)".
- Mark audio problems inline, e.g. "[inaudible]", "[audio breaking up]", "[long silence]".
- Do not summarise; transcribe everything that is said, including chess moves (e.g. "Queen to h5")."""


def transcribe(media_path: Path, duration: float, work: Path, progress: Progress) -> Transcript:
    fk = _file_key(media_path)
    starts = list(range(0, int(duration), CHUNK_S))

    def one(i_start):
        i, s = i_start
        clip = media.extract_audio(media_path, work / f"chunk_{i}.mp3", start=s, duration=CHUNK_S)
        data = clip.read_bytes()
        res = llm.generate_json([llm.audio_part(data), TRANSCRIBE_PROMPT], TranscriptChunk,
                                cache_parts=["transcribe", PROMPT_VERSION, fk, s, CHUNK_S], thinking="low")
        return s, res

    raw, langs, done = [], [], 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        for s, res in ex.map(one, enumerate(starts)):
            done += 1
            progress(0.15 + 0.45 * done / len(starts), f"Transcribed {done}/{len(starts)} audio chunks")
            langs.append(res.language)
            for seg in res.segments:
                rel = _parse_rel(seg.start)
                if rel is None or rel > CHUNK_S + 30:
                    continue
                raw.append([s + rel, None, seg.speaker.strip() or "Unknown", seg.text.strip()])
    raw.sort(key=lambda r: r[0])
    for a, b in zip(raw, raw[1:] + [None]):
        est = a[0] + max(1.0, len(a[3].split()) / 2.5)
        a[1] = min(est, b[0]) if b else est
    teacher = "Teacher"
    cues = [(a, max(b, a + 0.3), spk, t) for a, b, spk, t in raw]
    lang = max(set(langs), key=langs.count) if langs else "unknown"
    return build_transcript(cues, "gemini", teacher_hint=teacher, language=lang)


# ---------------------------------------------------------------- assessment
def _excerpt_windows(tr: Transcript, duration: float) -> list[tuple[str, float]]:
    first = next((s.start for s in tr.segments if s.role == "teacher"), 60.0)
    return [("opening", max(first - 5, 0)), ("first third", duration * 0.33), ("second third", duration * 0.66),
            ("closing", max(duration - EXCERPT_S - 60, 0))]


def assess(media_path: str | Path | None = None, vtt_text: str | None = None, *,
           progress: Progress | None = None, use_audio_excerpts: bool = True,
           n_frames: int = N_FRAMES) -> dict:
    progress = progress or (lambda f, m: None)
    if media_path is None and not vtt_text:
        raise ValueError("Provide a recording or a transcript.")
    work = media.tmpdir()
    try:
        info = None
        sig_audio, sig_video, frames, excerpts = {}, {}, [], []
        if media_path is not None:
            media_path = Path(media_path)
            progress(0.03, "Reading recording")
            info = media.probe(media_path)

        # 1. transcript
        if vtt_text:
            progress(0.1, "Parsing provided transcript")
            tr = from_vtt_text(vtt_text)
            if len(tr.segments) < 20:
                if media_path is None:
                    raise ValueError("The transcript has too few lines to assess.")
                progress(0.12, "Provided transcript looks unusable - transcribing audio instead")
                tr = transcribe(media_path, info.duration_s, work, progress)
        else:
            progress(0.12, "Transcribing audio with Gemini (speaker-labelled)")
            tr = transcribe(media_path, info.duration_s, work, progress)
        if not tr.segments:
            raise ValueError("No speech could be transcribed from this recording.")
        duration = info.duration_s if info else tr.segments[-1].end

        # 2. objective metrics
        progress(0.62, "Computing interaction metrics")
        met = M.compute(tr, duration)
        if media_path is not None:
            progress(0.66, "Analysing audio signal and video frames")
            sig_audio = media.audio_signal_metrics(media_path)
            if info.has_video:
                sig_video = media.visual_signal_metrics(media_path, duration)
                for t in media.sample_times(duration, n_frames):
                    jpg = media.frame_jpeg(media_path, t, width=1280)
                    if jpg:
                        frames.append((t, jpg, media.corner_crop_jpeg(media_path, t)))
            if use_audio_excerpts:
                for name, s in _excerpt_windows(tr, duration):
                    p = media.extract_audio(media_path, work / f"ex_{name}.mp3", start=s, duration=EXCERPT_S)
                    excerpts.append((name, s, p.read_bytes()))

        # 3. multimodal rubric assessment
        progress(0.75, "Assessing against the 8x8 rubric with Gemini")
        all_metrics = {"transcript_metrics": met, "audio_signal": sig_audio, "video_signal": sig_video}
        contents: list = [
            "## Rubric (score each dimension 1-5)\n" + rubric.rubric_text(),
            "## Objective metrics (computed automatically; teacher = speaker with most words)\n"
            + json.dumps(all_metrics, indent=1),
        ]
        if frames:
            contents.append(f"## {len(frames)} video frames sampled evenly across the class")
            contents.append("Each frame is followed by an upscaled crop of its top-right corner, where Zoom shows the "
                            "speaker's thumbnail during screen share - use the crops to judge the teacher's camera, "
                            "face, background and lighting. Judge 'lighting' on the teacher's face, not on the dark "
                            "app/board theme.")
            for t, jpg, crop in frames:
                contents += [f"Frame at {fmt_ts(t)}:", llm.image_part(jpg)]
                if crop:
                    contents += [f"Top-right crop at {fmt_ts(t)}:", llm.image_part(crop)]
        else:
            contents.append("## No video available - visual checks must be 'unclear'; lower your confidence.")
        if excerpts:
            contents.append("## Audio excerpts (judge the teacher's tone, energy, warmth and audio quality)")
            for name, s, data in excerpts:
                contents += [f"Audio excerpt '{name}' starting at {fmt_ts(s)} ({EXCERPT_S}s):", llm.audio_part(data)]
        contents.append(f"## Transcript (source: {tr.source}; speaker names pseudonymised)\n" + render(tr))
        contents.append("Assess this class now. Score all six dimensions with timestamped evidence, then give "
                        "strengths, improvements with coaching tips, red flags, a holistic rating and confidence.")

        ck = ["assess", PROMPT_VERSION, rubric.rubric_text(), json.dumps(all_metrics, sort_keys=True),
              render(tr), [hashlib.md5(j).hexdigest() for _, j, _c in frames],
              [hashlib.md5(d).hexdigest() for *_, d in excerpts]]
        a: Assessment = llm.generate_json(contents, Assessment, system=SYSTEM, cache_parts=ck, thinking="high")

        # 4. deterministic aggregation
        progress(0.95, "Building report")
        return build_report(a, met, sig_audio, sig_video, tr, has_video=bool(frames))
    finally:
        media.cleanup(work)


def build_report(a: Assessment, met: dict, sig_audio: dict, sig_video: dict, tr: Transcript,
                 has_video: bool) -> dict:
    scores = {d.key: d.score for d in a.dimensions}
    for k in rubric.WEIGHTS:                      # model omitted a dimension -> neutral
        scores.setdefault(k, 3)
    ws = rubric.weighted_score(scores)
    rating = rubric.to_rating(ws)
    major = [f for f in a.red_flags if f.severity == "major"]
    capped = False
    if major and rubric.RATINGS.index(rating) > rubric.RATINGS.index(rubric.MAJOR_FLAG_CAP):
        rating, capped = rubric.MAJOR_FLAG_CAP, True
    gap = abs(rubric.RATINGS.index(rating) - rubric.RATINGS.index(a.holistic_rating))
    conf = a.confidence
    order = ["Low", "Medium", "High"]
    if gap >= 1:
        conf = order[max(order.index(conf) - gap, 0)]
    if not has_video and conf == "High":
        conf = "Medium"
    return {
        "rating": rating,
        "weighted_score": ws,
        "holistic_rating": a.holistic_rating,
        "capped_by_red_flag": capped,
        "confidence": conf,
        "dimension_scores": scores,
        "assessment": a.model_dump(),
        "metrics": {"transcript": met, "audio_signal": sig_audio, "video_signal": sig_video},
        "transcript_source": tr.source,
        "language": tr.language,
        "prompt_version": PROMPT_VERSION,
        "model": llm.PRIMARY_MODEL,
    }
