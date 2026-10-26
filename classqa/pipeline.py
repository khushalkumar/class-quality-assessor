"""Processing core: staged, checkpointed, UI-independent.

Stages (job status in brackets):
  download   [extracting]   fetch link sources into the job's private work dir
  validate   [extracting]   ffprobe + size/duration/audio checks - before any API spend
  signals    [extracting]   ffmpeg silence/loudness + blank-frame statistics; silent audio is rejected
  transcript [transcribing] Zoom .vtt or Gemini transcription in 20-min chunks       (checkpointed)
  metrics    [scoring]      objective interaction metrics from the transcript
  media      [scoring]      12 frames + speaker crops, 4 audio excerpts (degrade gracefully if they fail)
  score      [scoring]      n Gemini rubric runs (self-consistency)                 (each run checkpointed)
  finalize   [done | needs_review] quote verification, deterministic label, review routing

Checkpoints are keyed by (content hash of the input, stage, version) so a retried job resumes from the
last finished stage and re-running the same recording costs nothing."""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import media, obs, quality, rubric
from . import metrics as M
from .errors import ClassQAError, MediaError
from .llm import FALLBACK_MODELS, INLINE_LIMIT, PRIMARY_MODEL, TRANSCRIBE_MODEL, LLMClient, audio_part, image_part
from .prompts import PROMPT_VERSION, SYSTEM, TRANSCRIBE_PROMPT
from .schemas import Assessment, TranscriptChunk
from .transcript import Transcript, build_transcript, fmt_ts, from_vtt_text, render

log = logging.getLogger("classqa.pipeline")

PIPELINE_VERSION = "2.0"
CHUNK_S = 1200         # transcription chunk length (s)
N_FRAMES = 12          # frames shown to the model
EXCERPT_S = 90         # length of each audio excerpt for tone/energy judgement
MIN_SEGMENTS = 20      # below this a transcript is considered unusable

Progress = Callable[[float, str, str], None]   # (fraction, message, status)


class NullCheckpoints:
    def get_checkpoint(self, *a):
        return None

    def put_checkpoint(self, *a):
        pass


@dataclass
class JobContext:
    job_id: str
    media_path: Path | None
    vtt_text: str | None
    work_dir: Path
    llm: LLMClient
    checkpoints: object = field(default_factory=NullCheckpoints)
    progress: Progress = lambda f, m, s: None
    self_consistency: int = 1
    use_audio_excerpts: bool = True
    content_hash: str = ""
    missing_signals: list[str] = field(default_factory=list)
    uploaded_files: list[str] = field(default_factory=list)
    models_used: dict = field(default_factory=dict)


def content_hash(path: Path | None, vtt_text: str | None) -> str:
    h = hashlib.sha256()
    if path is not None:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
    if vtt_text:
        h.update(b"vtt:" + vtt_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:32]


def _parse_rel(ts: str) -> float | None:
    p = [int(x) for x in re.findall(r"\d+", ts)[:3]]
    if not p:
        return None
    if len(p) == 3:
        return p[0] * 3600 + p[1] * 60 + p[2]
    if len(p) == 2:
        return p[0] * 60 + p[1]
    return float(p[0])


def _ck(ctx: JobContext, stage: str, version: str, compute: Callable[[], object]):
    """Idempotent stage: return checkpoint if present, else compute and store."""
    hit = ctx.checkpoints.get_checkpoint(ctx.content_hash, stage, version)
    if hit is not None:
        obs.event(log, "checkpoint_hit", stage=stage)
        return hit
    val = compute()
    ctx.checkpoints.put_checkpoint(ctx.content_hash, stage, version, val)
    return val


# ---------------------------------------------------------------- transcription
def _media_part(ctx: JobContext, path: Path, mime: str = "audio/mp3"):
    data = path.read_bytes()
    if len(data) <= INLINE_LIMIT or not hasattr(ctx.llm.backend, "upload"):
        return audio_part(data, mime)
    f = ctx.llm.backend.upload(str(path), mime)          # Files API; deleted in the worker's cleanup
    ctx.uploaded_files.append(f.name)
    return f


def transcribe(ctx: JobContext, duration: float) -> Transcript:
    starts = list(range(0, int(duration), CHUNK_S))
    done = [0]

    def one(i_start):
        i, s = i_start
        clip = media.extract_audio(ctx.media_path, ctx.work_dir / f"chunk_{i}.mp3", start=s, duration=CHUNK_S)

        def call():
            obj, model = ctx.llm.generate_json(
                [_media_part(ctx, clip), TRANSCRIBE_PROMPT], TranscriptChunk, thinking="low",
                models=[TRANSCRIBE_MODEL, PRIMARY_MODEL, *FALLBACK_MODELS])
            ctx.models_used["transcription"] = model
            return obj.model_dump()

        res = TranscriptChunk.model_validate(_ck(ctx, f"transcribe_chunk_{s}", f"{PROMPT_VERSION}/{CHUNK_S}", call))
        clip.unlink(missing_ok=True)
        done[0] += 1
        ctx.progress(0.15 + 0.45 * done[0] / len(starts), f"Transcribed {done[0]}/{len(starts)} audio chunks",
                     "transcribing")
        return s, res

    raw, langs = [], []
    with ThreadPoolExecutor(max_workers=3) as ex:
        # copy contextvars (job id, usage accounting) into each worker thread
        futs = [ex.submit(contextvars.copy_context().run, one, x) for x in enumerate(starts)]
        for s, res in (f.result() for f in futs):
            langs.append(res.language)
            for seg in res.segments:
                rel = _parse_rel(seg.start)
                if rel is None or rel > CHUNK_S + 30:
                    continue
                raw.append([s + rel, None, seg.speaker.strip() or "Unknown", seg.text.strip()])
    raw.sort(key=lambda r: r[0])
    for a, b in zip(raw, raw[1:] + [None], strict=False):
        est = a[0] + max(1.0, len(a[3].split()) / 2.5)
        a[1] = min(est, b[0]) if b else est
    cues = [(a, max(b, a + 0.3), spk, t) for a, b, spk, t in raw]
    lang = max(set(langs), key=langs.count) if langs else "unknown"
    return build_transcript(cues, "gemini", teacher_hint="Teacher", language=lang)


def _excerpt_windows(tr: Transcript, duration: float) -> list[tuple[str, float]]:
    first = next((s.start for s in tr.segments if s.role == "teacher"), 60.0)
    return [("opening", max(first - 5, 0)), ("first third", duration * 0.33), ("second third", duration * 0.66),
            ("closing", max(duration - EXCERPT_S - 60, 0))]


def build_contents(tr: Transcript, all_metrics: dict, frames: list, excerpts: list) -> list:
    """Assessment prompt contents (unchanged from the evaluated v4 prompt)."""
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
            contents += [f"Frame at {fmt_ts(t)}:", image_part(jpg)]
            if crop:
                contents += [f"Top-right crop at {fmt_ts(t)}:", image_part(crop)]
    else:
        contents.append("## No video available - visual checks must be 'unclear'; lower your confidence.")
    if excerpts:
        contents.append("## Audio excerpts (judge the teacher's tone, energy, warmth and audio quality)")
        for name, s, data in excerpts:
            contents += [f"Audio excerpt '{name}' starting at {fmt_ts(s)} ({EXCERPT_S}s):", audio_part(data)]
    contents.append(f"## Transcript (source: {tr.source}; speaker names pseudonymised)\n" + render(tr))
    contents.append("Assess this class now. Score all six dimensions with timestamped evidence, then give "
                    "strengths, improvements with coaching tips, red flags, a holistic rating and confidence.")
    return contents


# ---------------------------------------------------------------- the staged run
def run(ctx: JobContext) -> dict:
    P = ctx.progress
    if ctx.media_path is None and not ctx.vtt_text:
        raise MediaError("Provide a recording or a transcript.")
    ctx.content_hash = ctx.content_hash or content_hash(ctx.media_path, ctx.vtt_text)
    info = None
    sig_audio, sig_video, frames, excerpts = {}, {}, [], []

    if ctx.media_path is not None:
        with obs.stage_timer(log, "validate"):
            P(0.04, "Validating recording", "extracting")
            info = media.probe(ctx.media_path)
            if not info.has_video:
                ctx.missing_signals.append("video_frames")
        with obs.stage_timer(log, "signals"):
            P(0.08, "Checking audio levels", "extracting")
            sig_audio = _ck(ctx, "audio_signal", "1", lambda: media.audio_signal_metrics(ctx.media_path))
            media.check_not_silent(sig_audio)
            if info.has_video:
                try:
                    sig_video = _ck(ctx, "video_signal", "1",
                                    lambda: media.visual_signal_metrics(ctx.media_path, info.duration_s))
                except ClassQAError:
                    ctx.missing_signals.append("video_signal")

    with obs.stage_timer(log, "transcript"):
        tr = None
        if ctx.vtt_text:
            P(0.12, "Parsing provided transcript", "transcribing")
            tr = from_vtt_text(ctx.vtt_text)
            if len(tr.segments) < MIN_SEGMENTS:
                if ctx.media_path is None:
                    raise MediaError("The transcript has too few lines to assess.")
                P(0.13, "Provided transcript looks unusable - transcribing audio instead", "transcribing")
                tr = None
        if tr is None:
            if ctx.media_path is None:
                raise MediaError("No transcript or recording to transcribe.")
            P(0.14, "Transcribing audio with Gemini (speaker-labelled)", "transcribing")
            tr = transcribe(ctx, info.duration_s)
        if len(tr.segments) < MIN_SEGMENTS:
            raise MediaError(f"Only {len(tr.segments)} speech segments were found - the recording has too little "
                             "speech to assess a class.")
    duration = info.duration_s if info else tr.segments[-1].end

    with obs.stage_timer(log, "metrics"):
        P(0.62, "Computing interaction metrics", "scoring")
        met = M.compute(tr, duration)

    if ctx.media_path is not None:
        with obs.stage_timer(log, "media"):
            P(0.66, "Sampling video frames and audio excerpts", "scoring")
            if info.has_video:
                try:
                    for t in media.sample_times(duration, N_FRAMES):
                        jpg = media.frame_jpeg(ctx.media_path, t, width=1280)
                        if jpg:
                            frames.append((t, jpg, media.corner_crop_jpeg(ctx.media_path, t)))
                except ClassQAError:
                    frames = []
                if not frames and "video_frames" not in ctx.missing_signals:
                    ctx.missing_signals.append("video_frames")
            if ctx.use_audio_excerpts:
                try:
                    for name, s in _excerpt_windows(tr, duration):
                        p = media.extract_audio(ctx.media_path, ctx.work_dir / f"ex_{name}.mp3", start=s,
                                                duration=EXCERPT_S)
                        excerpts.append((name, s, p.read_bytes()))
                        p.unlink(missing_ok=True)
                except ClassQAError:
                    excerpts = []
                    ctx.missing_signals.append("audio_excerpts")
    else:
        ctx.missing_signals += ["video_frames", "audio_excerpts"]

    all_metrics = {"transcript_metrics": met, "audio_signal": sig_audio, "video_signal": sig_video}
    contents = build_contents(tr, all_metrics, frames, excerpts)
    score_version = f"{PROMPT_VERSION}/{rubric.rubric_version()}/{PRIMARY_MODEL}/{len(frames)}/{len(excerpts)}"

    runs: list[tuple[Assessment, str]] = []
    with obs.stage_timer(log, "score"):
        for i in range(max(1, ctx.self_consistency)):
            P(0.72 + 0.2 * i / ctx.self_consistency,
              f"Scoring against the 8x8 rubric (run {i + 1}/{ctx.self_consistency})", "scoring")

            def call():
                obj, model = ctx.llm.generate_json(contents, Assessment, system=SYSTEM, thinking="high")
                return {"assessment": obj.model_dump(), "model": model}

            d = _ck(ctx, f"score_{i}", score_version, call)
            runs.append((Assessment.model_validate(d["assessment"]), d["model"]))

    with obs.stage_timer(log, "finalize"):
        P(0.95, "Verifying evidence and building report", "scoring")
        return build_report(runs, met, sig_audio, sig_video, tr, ctx)


def build_report(runs: list[tuple[Assessment, str]], met: dict, sig_audio: dict, sig_video: dict,
                 tr: Transcript, ctx: JobContext | None = None) -> dict:
    a, model = runs[0]
    score_sets = [{k: 3 for k in rubric.WEIGHTS} | {d.key: d.score for d in r.dimensions} for r, _ in runs]
    scores, run_ratings, disagree = quality.merge_runs(score_sets)
    ws = rubric.weighted_score(scores)
    rating = rubric.to_rating(ws)
    major = [f for r, _ in runs for f in r.red_flags if f.severity == "major"]
    capped = False
    if major and rubric.RATINGS.index(rating) > rubric.RATINGS.index(rubric.MAJOR_FLAG_CAP):
        rating, capped = rubric.MAJOR_FLAG_CAP, True
    gap = abs(rubric.RATINGS.index(rating) - rubric.RATINGS.index(a.holistic_rating))
    order = ["Low", "Medium", "High"]
    conf = a.confidence
    if gap >= 1:
        conf = order[max(order.index(conf) - gap, 0)]
    missing = list(dict.fromkeys(ctx.missing_signals if ctx else []))
    if "video_frames" in missing and conf == "High":
        conf = "Medium"
    if disagree:
        conf = "Low"

    adump = a.model_dump()
    qstats = quality.verify_evidence(adump, render(tr, max_chars=10 ** 9))
    reasons = quality.review_reasons(confidence=conf, rating=rating, holistic=a.holistic_rating, weighted=ws,
                                     run_ratings=run_ratings, quote_stats=qstats, missing_signals=missing,
                                     major_flags=len(major), capped=capped)
    return {
        "rating": rating,
        "weighted_score": ws,
        "holistic_rating": a.holistic_rating,
        "capped_by_red_flag": capped,
        "confidence": conf,
        "needs_review": bool(reasons),
        "review_reasons": reasons,
        "dimension_scores": scores,
        "self_consistency": {"runs": len(runs), "ratings": run_ratings, "disagree": disagree,
                             "per_run_scores": score_sets},
        "evidence_check": qstats,
        "missing_signals": missing,
        "assessment": adump,
        "metrics": {"transcript": met, "audio_signal": sig_audio, "video_signal": sig_video},
        "transcript_source": tr.source,
        "language": tr.language,
        "versions": {"pipeline": PIPELINE_VERSION, "prompt": PROMPT_VERSION, "rubric": rubric.rubric_version(),
                     "thresholds": [t for t, _ in rubric.THRESHOLDS]},
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "transcription_model": (ctx.models_used.get("transcription") if ctx else None) if tr.source == "gemini" else None,
    }
