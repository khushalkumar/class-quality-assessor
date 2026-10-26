# Design: 8x8 Class Quality Assessor

*Status: prototype running in production-shaped form (Streamlit Community Cloud). Author: Khushal Kumar.*

## 1. Problem and goals

8x8 teaches children chess on Zoom. Its academic team reviews recorded classes by hand: a reviewer watches the
class, then writes a label (Excellent / Good / Average / Weak) and a few lines of feedback. That takes about one
class-length of human time per class and doesn't scale to hundreds of classes a week.

**Goals**
1. Rate a recorded class **Poor / Average / Good / Excellent** and explain the rating with **specific, timestamped,
   verifiable evidence** that a reviewer or teacher can act on.
2. Be **trustworthy**: never present an uncertain result as certain. Route doubtful cases to a human.
3. Be **operable at 8x8 scale**: hundreds of classes a week, predictable cost, explicit failures, resumable work,
   and observable behaviour.
4. **Protect children's data**: no media kept beyond the job, no names in outputs, one approved model vendor.

**Non-goals (for this version)**
- Replacing reviewers. The system triages and drafts; humans decide on borderline cases and on anything that
  affects a teacher's pay or employment.
- Live, in-class feedback. This is batch/offline only.
- Training a custom model. With 10 labelled classes that would overfit; a prompted multimodal model plus a
  calibrated, code-computed label is the right tool at this data size.

## 2. Architecture

```mermaid
flowchart TB
  subgraph Client
    UI[Streamlit UI<br/>submit + poll + report]
    OPS[Ops page<br/>status, failures, cost]
    CLI[CLI / batch<br/>folder or link list]
  end
  subgraph Core["Processing core (classqa/)"]
    Q[(Job queue)]
    W[Worker thread(s)]
    P[Staged pipeline]
    LLM[LLM client<br/>backoff+jitter, fallback chain,<br/>circuit breaker, schema repair,<br/>budget + daily quota guard]
    T[Trust layer<br/>quote verification,<br/>self-consistency, review routing]
  end
  subgraph Storage
    DB[(SQLite: jobs, checkpoints, quota)]
    TMP[/per-job temp dir<br/>deleted after each attempt/]
  end
  G[Gemini API]
  UI -->|submit| Q
  CLI -->|submit| Q
  Q --> W --> P
  P <--> DB
  P <--> TMP
  P --> LLM --> G
  P --> T
  UI -->|poll| DB
  OPS --> DB
```

The **UI is a thin client**. `app.py` submits a job and polls its status with an auto-refreshing fragment. All
processing lives in `classqa/` and runs in a **background worker thread**. The CLI drives the same worker, so bulk
runs and the app share one code path, one set of guards and one set of checkpoints.

### Job state machine

`queued → extracting → transcribing → scoring → done | needs_review | failed`

- `needs_review` is a **successful** result that a human should confirm (reasons are stored with the job).
- `failed` always carries a stable `error_code` (for dashboards) and a plain-language `error_message` (for users).
- If a job is left non-terminal because the process restarted, it is marked failed (`worker_restarted`) on the next
  start. A retry resumes from the checkpoints.

### Stages and checkpoints (data flow)

| Stage | Status | Input → output | Checkpointed |
|---|---|---|---|
| download | extracting | link → file streamed to the job dir (1 MB chunks, size cap, HTTP Range resume) | – |
| validate | extracting | file → ffprobe facts; rejects bad input **before any API spend** | – |
| signals | extracting | ffmpeg silence/loudness + blank-frame stats; silent audio rejected | yes |
| transcript | transcribing | Zoom `.vtt` or Gemini transcription in 20-min chunks (parallel ×3) | **per chunk** |
| metrics | scoring | transcript → talk-time, questions, turns, monologues, silences, praise, homework, A/V complaints | – (cheap) |
| media | scoring | 12 frames + top-right speaker crops, 4×90 s audio excerpts (optional; degrade if they fail) | – (local, cheap) |
| score | scoring | everything above → `n` rubric runs (structured JSON) | **per run** |
| finalize | done / needs_review | quote verification, merge runs, deterministic label, review reasons | – |

The checkpoint key is **(sha256 of the input bytes + transcript, stage, version)**. The version includes the
prompt version, the rubric hash, the model and the number of frames and excerpts. Consequences:
- a **retry resumes** from the last finished stage (tested: a scoring failure followed by a retry re-uses the
  transcript and makes zero transcription calls);
- **re-submitting the same recording costs nothing**;
- changing the prompt or rubric automatically invalidates only the stages it affects.
- Checkpoints hold derived data only: pseudonymised transcript JSON, metrics, model output. They are purged after
  a TTL (default 7 days).

## 3. Key decisions and trade-offs

| Decision | Alternatives considered | Why this one | Cost of the choice |
|---|---|---|---|
| **Gemini multimodal** (audio + frames + transcript in one call) | transcript-only LLM; Whisper + GPT; custom classifier | 8x8's feedback is largely visual and tonal (branded background, face visible, "energetic and pleasant"). Transcript-only scoring rated the "quality issue" classes Good/Excellent (eval v2). One vendor handles audio, images and text, which is also simpler for privacy approval. | Vendor lock-in (the transport is behind a `Backend` interface); free-tier quotas are tiny. |
| **Rubric derived from 8x8's feedback**, weighted by how often a theme decides a label | generic teaching rubric (e.g. CLASS); let the model decide freely | Aligns with what this client actually rewards. Example: "asks each child individually" appears even in weak classes, so it carries less weight. | Small sample; weights need re-checking as labelled data grows. |
| **Code computes the label** from the per-dimension scores (fixed weights and thresholds, red-flag cap) | ask the model for the label | Auditable, stable, tunable without re-prompting. The model's holistic label is kept as a cross-check and its disagreement lowers confidence. | Thresholds must be calibrated, and are now calibrated on 10 classes (leave-one-out reported). |
| **Human review queue** (`needs_review`) | always output a label; refuse when unsure | A wrong confident label harms teachers and erodes trust. A flagged label still saves most of the reviewer's time, because the evidence is pre-assembled. | Some reviewer load. Routing sent 6/10 eval classes to review; the margin is configurable to match reviewer capacity. |
| **Quote verification** against the transcript | trust the model | Hallucinated evidence is the fastest way to lose reviewer trust; checking quotes is cheap and deterministic. | Paraphrases count as "approximate", and audio-only claims can't be verified textually (they're labelled). |
| **Cheap model for both stages** (`gemini-3.5-flash-lite` scoring, `gemini-3.1-flash-lite-preview` transcription) | larger flash or pro models | Best *complete* eval run, roughly 10× cheaper, and usable quota. Bigger models hit the 20 requests/day free-tier pool mid-eval. | Compressed score range; the fallback chain includes bigger models. |
| **SQLite + in-process worker** | Postgres + Redis + Cloud Run now | Runs on Streamlit Cloud's single container. Interfaces (`JobStore`, `CheckpointStore`, `QuotaStore`, `Backend`) make the swap mechanical. | One container = one worker pool, and data is lost if the container is recycled (jobs fail explicitly). |
| **Link-first input** for big files | browser upload only | Browser uploads of long recordings stall on proxied or slow connections (seen live: a 54 MB upload never finished). Server-side streaming download has progress, a size cap and resume. | Users must share the Drive file publicly ("anyone with the link"). |

## 4. Failure modes

Every row is covered by a unit test in `tests/` (fakes, no API calls) unless marked *(manual)*.

| Failure | Detection | Handling | What the user sees |
|---|---|---|---|
| Unsupported file type / Drive folder link / HTML page instead of a file | extension check, URL pattern, `content-type` | reject before download/API | "Unsupported file type…" / "That is a folder link — share the file" |
| Corrupt or truncated media | ffprobe non-zero / unparseable / zero duration | `failed: corrupt_media`, no API spend | "This file could not be read — it looks corrupt" |
| Too large / too long / too short | size cap (2 GB, checked again while streaming), duration limits (60 s – 3 h) | reject early | exact limit in the message |
| No audio track | ffprobe streams | `failed: no_audio` | "This file has no audio track" |
| Silent audio (muted capture) | ffmpeg volumedetect (max < −45 dB) | `failed: silent_audio` | "The audio track is silent…" |
| No video track (audio-only) | ffprobe | **degrade**: score without visuals, `missing_signals` recorded, confidence capped, → `needs_review` | blue info box: "Assessed without video frames" |
| Frames or excerpts fail to extract | exception in the media stage | degrade as above | same |
| Large browser upload stalls *(manual, seen live)* | upload never completes | UI recommends the Link path; `maxUploadSize` 500 MB; guidance under the uploader | "Uploads above ~200 MB can stall — use the Link tab" |
| Drive file not shared | HTTP 401/403/404, HTML sign-in page, gdown "cannot retrieve public link" | `failed: drive_permission` (not retried) | how to set "Anyone with the link" |
| Drive download quota ("too many users") / HTTP 429 | gdown message / status | `failed: drive_quota` (retryable) | "Try later or upload a copy" |
| Dropped connection mid-download | exception while streaming | resume with HTTP Range (3 retries, backoff) | progress continues |
| Gemini 429 per-minute, 503, timeouts, disconnects | error classification | exponential backoff with **full jitter** (capped), then next model | progress message "model busy, retrying" |
| Gemini daily quota exhausted | 429 "per day" | model's breaker opened for hours; fall back to next model | transparent; if all exhausted → `failed: llm_unavailable` + retry hint |
| Repeated failures on one model | per-model **circuit breaker** (5 consecutive → open 120 s) | skip that model; all open → `failed: circuit_open` | "AI requests paused briefly, retry in a few minutes" |
| Invalid API key | 400/401/403 "API key not valid" | fail fast, no retries (`llm_auth`) | "Check the GEMINI_API_KEY secret" |
| Malformed JSON / schema violation (e.g. score 9) | Pydantic validation against the response schema | **one repair retry** with the error, then `failed: malformed_output` | "The model returned malformed output twice; stopped rather than guessing" |
| Hallucinated evidence quote | quote not found (exact or ≥0.8 fuzzy) in the transcript | quote removed and flagged; >⅓ unverified → `needs_review` | "⚠️ not in transcript" badge, evidence-check summary |
| Low confidence, holistic vs rubric gap ≥2, score within 0.05 of a threshold, self-consistency disagreement, major red flag | `quality.review_reasons` | `needs_review` with the reasons listed | yellow "Needs human review" banner listing the reasons |
| Runaway cost on one job | per-job token/USD budget (default 2M tokens / $0.50) | `failed: budget_exceeded` | message with tokens and cost |
| Team-wide quota exhaustion | daily request counter (SQLite; Redis in prod) | `failed: daily_quota` before calling | "Daily limit reached; retry after reset" |
| Process restart mid-job | non-terminal job older than 15 min at startup | `failed: worker_restarted`; retry resumes from checkpoints | retry button (link jobs) |
| Unexpected bug | catch-all in the worker | `failed: internal_error`, traceback in the JSON logs; media still deleted | generic message, nothing leaks |

## 5. Reliability of the rating

- **Self-consistency (optional, n=2):** both runs are scored independently and their dimension scores averaged. If
  the runs disagree by one level or more, confidence is set to Low and the job goes to `needs_review`. It's off by
  default because it doubles scoring cost; recommended for anything that feeds HR decisions.
- **Deterministic settings:** fixed seed (`CLASSQA_SEED`). Temperature is left at the vendor default because
  Gemini 3 guidance warns that low temperatures can cause looping; it's configurable. The label itself comes from
  code, so model variance only moves the dimension scores.
- **Versioning for audit:** every result stores the pipeline version, prompt version (`v4`), rubric hash (weights,
  anchors, thresholds, cap), the models that answered, and the job's token usage. A rating can always be traced to
  the exact logic that produced it.

## 6. Observability

- **Structured JSON logs** (`classqa.obs`). Every line carries `job_id`. Events: `stage_start`/`stage_end` (with
  seconds), `llm_call` (model, schema, latency, tokens, estimated cost), `llm_error` (classified kind, attempt),
  `circuit_open`, `schema_invalid`, `checkpoint_hit`, `job_finished`/`job_failed`, `media_deleted`.
- **Per-job usage** stored with the result: calls, failed calls, tokens in/out, estimated USD, per-model breakdown,
  per-stage seconds.
- **Ops page** (`pages/1_Ops.py`): jobs by status, failure reasons by code, average cost/time/tokens per class,
  requests today vs the daily guard, circuit-breaker state, recent jobs table, and per-stage latency.

## 7. Privacy

- Media lives only in a per-job temp folder. It's deleted when the attempt ends, on success **or** failure
  (tested). Uploads are written there directly; link downloads are streamed there, never held in memory.
  Files a caller owns (CLI paths) are read in place and never modified.
- Media goes to the Gemini API inline by default. Anything above 15 MB uses the Files API and is **deleted
  explicitly** after the job (Gemini also expires files after 48 h).
- Zoom display names (and their occurrences in the text) are replaced with `Teacher` / `Student N` before the
  transcript leaves the transcript stage. The model is instructed never to output names. Names spoken aloud can
  still appear in transcripts, so checkpoints are TTL-purged (7 days) and live outside the repo.
- Nothing from the dataset (recordings, transcripts, feedback, names) is in git; CI and tests use synthetic media.
- **Vendor approval:** the brief asks that recordings be processed only with services the organiser permits.
  Gemini must be confirmed with Humaniti/8x8 before real use. In production that means a paid Gemini API or
  Vertex AI project with data-use terms that exclude training on customer data, and a region chosen to match
  8x8's data-residency needs.

## 8. Scaling and cost (estimates)

Measured on the real end-to-end job (38-min class, public Drive link, no `.vtt`) through the job path: 3
successful Gemini calls (2 transcription chunks + 1 scoring) plus 5 failed 503s handled by backoff/fallback,
**~101k input + ~17k output tokens, 122 s wall time**. Breakdown: download 5 s, signals 12 s, transcription 64 s,
media 7 s, scoring 34 s. Eval runs (`.vtt` path) used ~45–60k input tokens per scoring call.

| Item | Per class (60 min, est.) | Per 1,000 classes (est.) |
|---|---|---|
| Tokens (transcription 3 chunks + 1 scoring) | ~150k in / ~25k out | ~150M in / ~25M out |
| Cost, flash-lite models (≈$0.10 / $0.40 per 1M) | **~$0.025** | **~$25** |
| Cost, same with self-consistency n=2 | ~$0.035 | ~$35 |
| Cost, larger flash model for scoring (≈$0.50 / $3.00) | ~$0.10 | ~$100 |
| Wall time, one worker | ~3 min | ~50 h serial → ~5 h with 10 workers |
| Gemini requests | ~4 | ~4,000 (needs paid-tier rate limits) |

*Prices are list-price assumptions in `classqa/obs.py`; check current pricing. The ops page reports the running
average from real jobs.*

**Path to production scale** (interfaces already in place):
- `JobStore` → **Postgres** (jobs, results, review decisions); `CheckpointStore` → Postgres + **object storage**
  (GCS/S3) with a lifecycle rule for the TTL; `QuotaStore` → **Redis** `INCR` with expiry, shared by all workers.
- Queue → **Cloud Tasks / Pub/Sub** (or Celery + Redis). `JobRunner.process(job_id)` is already the task body.
  Workers on **Cloud Run jobs**, autoscaled on queue depth. Concurrency is capped per model to respect RPM/TPM.
- Media: signed-URL uploads straight to object storage (bypassing the web tier), so the 54 MB-upload problem goes
  away. Workers stream from the bucket. Bucket lifecycle deletes media after 24 h.
- Ingest: a Zoom cloud-recording webhook creates a job per finished class, using Zoom's own transcript when
  available (cheaper, better speaker labels).
- Review: a reviewer queue for `needs_review` jobs. Reviewer decisions are stored and become new labelled data
  for re-calibrating thresholds every month.

## 9. Evaluation and its limits

Details in `eval/EVALUATION.md`. Final configuration on the 10 labelled classes: **7/10 exact, 9/10 within one
level** (leave-one-out threshold fit: 5/10 exact). Both "known quality issue" classes are rated Poor, caught by the
media checks rather than the transcript. Trust layer on the same outputs (offline, no API calls): 141/144 textual
quotes verified in the transcript (3 removed); 6/10 classes routed to review, catching 2 of the 3 misclassified,
and 3 of the 4 auto-accepted classes were correct.

Limits: 2 labelled classes per label, thresholds tuned on the same data, run-to-run variance about ±0.4 on the
weighted score, the visual background check is unstable on at least one class, and Gemini transcripts merge short
replies (so turn counts differ from Zoom's).

## 10. Roadmap

1. **Dedicated visual check:** a narrow prompt (or 8x8-logo template matching) for background, face and camera,
   the biggest remaining error source.
2. **More labelled data from the review queue:** recalibrate weights and thresholds; report a confusion matrix
   and inter-rater agreement against human reviewers, not just against labels.
3. **Use Zoom metadata** (participant join/leave times, camera state) instead of inferring it from pixels.
4. **Production infrastructure** as in §8, with signed-URL uploads and webhook ingest.
5. **Per-teacher trends:** rating over time, recurring coaching themes, and alerts on repeated red flags.
6. **Prompt/model A/B harness:** run candidate prompts on the labelled set in CI (nightly, paid key) and block a
   prompt change that lowers accuracy.
