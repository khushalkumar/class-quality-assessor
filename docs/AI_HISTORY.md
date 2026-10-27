# AI interaction history — 8x8 Class Quality Assessor

*Khushal Kumar · HUMANITI AI build challenge. Personal data (children's and teachers' names), credentials and
unrelated material are redacted. Class recordings were processed only through the Gemini API.*

## Tools used
- **AI coding agent** (Grok Bot, on a sandboxed Linux box): research, code generation, running ffmpeg and the
  evaluation harness, debugging. I steered it with a design brief and reviewed results between iterations.
- **Gemini API** (google-genai SDK) inside the product: transcription (`gemini-3.1-flash-lite-preview`) and
  multimodal rubric assessment (`gemini-3.5-flash-lite` after evaluation, with a fallback chain).

## 1. Brief and design prompt (to the coding agent)
> Build a hosted Streamlit app for 8x8 (children's online classes): user gives a class recording (upload or link);
> app rates Poor / Average / Good / Excellent and explains why with specific, evidence-backed observations.
> Pipeline: ffmpeg audio (+ sampled frames) → Gemini → timestamped transcript with speaker turns → objective metrics
> (talk-time, questions, student responses, monologues, silences, tech-issue mentions…) → rubric-based assessment
> with timestamped quoted evidence → rating + confidence + strengths + improvements + coaching tips.
> Derive the rubric from feedbacks.pdf and the labelled transcripts; calibrate so it reproduces the labels; use
> structured JSON output. Write an eval script over all labelled samples and iterate; don't overfit.
> Keep recordings confidential; never commit data or keys.

## 2. Exploration (agent, local only)
- Read the brief and `feedbacks.pdf`. Key finding: 8x8's reviewers judge **how the class looks and sounds** as much as
  what is said — 8x8 branded background, teacher camera on / face visible, lighting, "energetic and pleasant
  interaction", rapport building and an intro before content, time for new joiners to introduce themselves,
  individually asking each child, homework, a proper conclusion, and "known quality issues".
- Probed the 12 recordings (38–90 min, 1080p Zoom, chess on Lichess) and looked at sampled frames: during screen
  share the teacher is a tiny thumbnail in the top-right corner, or not in the recording at all.
- Computed transcript metrics for every labelled class *before* any LLM work. Student turns per 10 min separated
  Excellent from the rest; "asks children individually" did not (the feedback notes it even for weak classes).

## 3. API reality check
- `gemini-2.5-*` returned 404 for this key ("no longer available to new users"); `gemini-3.1-pro-preview` has a zero
  free-tier quota. `gemini-3-flash-preview` worked but hit its **daily free-tier request limit (~20/day)** during
  evaluation; `gemini-3.6-flash` and `gemini-3.5-flash` share the same ~20/day pool. Decision: transcription on
  the high-quota flash-lite model, assessment on the best model that could finish a complete eval run, a fallback chain across models, an on-disk cache keyed by file hash + prompt version + model, and retry/backoff.

## 4. Evaluation-driven iterations (details and tables in EVALUATION.md)
| Round | Change | Result on 10 labelled classes |
|---|---|---|
| v1 | First rubric (engagement-heavy weights), transcript + 12 frames + 3 audio excerpts; smoke test on 1 class | Report quality good (specific timestamps, quotes) |
| v2 | Full run. Fixes: excerpt-relative timestamps, verdict must not contradict the computed label, no names in quotes | 4/10 exact, 7/10 within one. "Quality issue" and weak classes rated Good/Excellent |
| v3 | Diagnosed with the agent: (a) teacher's background/face invisible in 1280px frames → add **upscaled top-right crops**; (b) my "dark frame %" metric made the model call dark Lichess themes "poor lighting" → **removed**; (c) reweighted rubric to the themes that decide labels in the feedback (energy 25%, rapport 20%, setup 20%, engagement 15%, clarity 10%, closure/HW 10%); (d) stricter anchors ("calm and polite = 3"); (e) model lists concerns before scoring; (f) **major red flag caps the rating at Average** | partial run interrupted (network disconnects → retry logic fixed) |
| v4 | Model called the camera "off" whenever the recording showed only the shared screen → prompt explains Zoom recording artefacts; "major" flag strictly defined | invalid run: daily quota hit, models mixed → model-aware cache |
| v4-lite | Same prompt on gemini-3.1-flash-lite-preview | 2/10 exact, 7/10 within one; compressed scores → used for transcription only |
| v4-g36 | Same prompt on gemini-3.6-flash | 2/4 before 503s and the daily 429 stopped it |
| v4-l35 | Same prompt on gemini-3.5-flash-lite | **5/10 exact, 9/10 within one**; both quality-issue classes Poor |
| recal. | Moved only the Excellent threshold 3.95 → 3.45 on saved scores (no API calls); checked with leave-one-out | 7/10 in-sample, 5/10 leave-one-out, 9/10 within one |
| e2e | Full path on 3 real mp4s without the Zoom transcript (Gemini transcription) | all completed; EX-2 Good, PO-2 Average, QI-1 Good (background misread: unstable visual check, documented) |

I stopped iterating there because the submission was due; the next step would be a dedicated visual check for
the 8x8 background and face.

## 5. Key prompts inside the product (abridged)
- **Transcription** (per 20-min audio chunk, JSON schema): "Transcribe this audio clip from an online children's chess
  class… speaker: 'Teacher' / 'Student: <first name>' / 'Student'… translate non-English with '(translated)'… mark
  [inaudible] / [audio breaking up]… do not summarise."
- **Assessment system prompt**: reviewer persona at 8x8; what 8x8 reviewers value (from the feedback); evidence rules
  (timestamps + verbatim quotes, never invent); recording-artefact rules; calibration rules ("asking individually is
  expected… a calm, polite tutor who jumps into content and skips homework is Average at best… 4–5 needs explicit
  positive evidence"); privacy rule (never output names).
- **Inputs**: rubric with anchors, objective metrics JSON, 12 frames + top-right crops, 4×90 s audio excerpts,
  pseudonymised transcript. **Output**: Pydantic schema (visual check → concerns → 6 dimension scores with evidence →
  strengths → improvements with coaching tips → red flags with severity → holistic rating, confidence, verdict).

## 6. Debugging with AI
- `Server disconnected without sending a response` was treated as non-retryable → retry everything except 400/401/403.
- Free-tier daily 429s silently mixed models in one eval run → cache is now model-aware, eval pins one model,
  the log records which model answered.
- Zoom VTT speaker names are often parents' accounts; teacher = speaker with most words; names in text are replaced
  with `[Student N]` labels before sending to the assessor.

## 7. Production-design phase (after the first deploy)
Goal from me: make the deployed app read like a production system (job model, failure handling, observability,
privacy, tests, CI, a real design doc), without spending quota on full eval reruns. Work was done on the branch
`production-design` in small commits, then merged to main.

| Step | What I asked / what the agent did | Outcome |
|---|---|---|
| Split | Core in `classqa/` with no Streamlit imports; `ui/` + `app.py` submit jobs and poll with `st.fragment(run_every=2)` | UI never blocks; the CLI `assess`/`batch` commands use the same JobRunner |
| Job model | `Job` + `JobStatus` (queued, extracting, transcribing, scoring, done, failed, needs_review) in SQLite behind `JobStore`/`CheckpointStore`/`QuotaStore` protocols | Swappable for Postgres/object storage/Redis (DESIGN.md §8) |
| Checkpoints | Every stage (signals, each 20-min transcription chunk, each scoring run) is keyed by sha256 + prompt/rubric/model version | Retries and re-submits skip finished work |
| Failure handling | Typed errors with codes and user messages; error classification (auth, missing model, daily quota, rate limit, transient); backoff with full jitter; fallback chain; per-model circuit breaker; one schema-repair retry; per-job token/cost budget; daily request guard | Each failure tested with a fake backend or synthetic ffmpeg media |
| Trust layer | Quote verification against the transcript (exact/fuzzy, unverified quotes removed); self-consistency merge; review routing with reasons | Offline on saved eval outputs: 141/144 quotes verified, 6/10 routed to review, 2/3 misses caught |
| Observability | JSON logs with job_id, stage, latency, tokens, cost; Ops page | Measured cost of the real e2e job: ~$0.017, 122 s |
| Privacy | Per-job scratch dir deleted on done/failed; Gemini Files deleted after use; checkpoints hold derived JSON only, 7-day TTL; vendor-approval note | Documented in README and DESIGN.md §7 |
| Tests/CI | Fakes (FakeBackend, synthetic mp4s: good, no audio, silent, short, audio-only, corrupt); AppTest for the UI and Ops page; ruff; GitHub Actions | 61 offline tests, ruff clean |

**Bugs the tests caught (with the agent):**
- Usage from transcription threads was not counted, because `contextvars` don't propagate into a
  `ThreadPoolExecutor`. Fixed with `copy_context().run` and a lock on the usage object.
- The borderline check used float `<=`, so a score exactly 0.05 from a threshold was sometimes routed. Changed to
  a strict comparison after rounding.
- The review reason said a red flag "capped the rating" even when the label was already below the cap. The
  wording now depends on whether it actually capped.

**Live issue I reported:** a 54 MB upload on Streamlit Cloud stalled for over 10 minutes, twice. I asked the agent to
make the Link path the recommended one. It added a streaming download from Drive's `usercontent` endpoint in 1 MB
chunks, with a progress callback, a size cap, HTTP Range resume (3 retries), Drive 403/404 → permission error,
429 → quota error, and gdown as a fallback. The UI now puts the Link tab first with sharing instructions and the
limits, and `maxUploadSize` is 500 MB. The real Drive download of that file took 5 s.

**One real end-to-end job** (no full rerun, since the scoring logic was unchanged): AV-1 from a public Drive link,
no `.vtt`. The job finished as Poor (2.30; actual label Average). 5 HTTP 503s on the transcription model opened the
breaker, transcription fell back to another model, and the job completed. 3 successful calls, ~$0.017.
- Final review pass: pydantic validation errors include the raw input, which could hold transcript text. Schema-error logs and repair prompts now carry only field locations and error types.

## 8. Advanced evals (after submission)

I asked for agreement statistics, stability, robustness tests, an LLM judge for feedback quality, and a CI
regression gate, with main green at every commit.

| Step | What the agent did | What came out |
|---|---|---|
| κ + confusion | `classqa/evalmetrics.py` (pure Python): quadratic weighted κ, confusion matrix, scripted leave-one-out threshold fit | κ 0.786 in-sample, 0.762 leave-one-out. The scripted leave-one-out gives 6/10; the earlier hand fit gave 5/10, and I kept 5/10 as the headline |
| Per-dimension | Checked the labels first: overall-only. So no per-dimension accuracy; instead Spearman vs label/holistic and cross-model consistency | Rapport is most aligned (0.80); energy is weak (0.29) despite its 25% weight; setup is least reproducible across models |
| Stability | `eval/stability.py`: fresh checkpoint store per repeat, `.vtt` transcript, scoring only; 3 classes × 3 repeats, run twice (fixed seed, varied seed) | Fixed seed: identical. Varied seed: flip rate 0.67, EX-1 Average↔Excellent |
| Robustness | `tests/test_robustness.py`: truncated / garbled / silent / Hindi / prompt-injection inputs with fake backends | 13 tests; added a prompt-injection tripwire (0 hits on the 12 real transcripts) |
| Judge | `eval/judge.py` with a negative control | Real feedback 3.7–4.9 per criterion, controls 1; grounding is the weakest |
| Gate | `tests/test_eval_gate.py` + `eval/frozen/gate.json`, inside the existing pytest step | Fails CI on a regression; a test proves a bad threshold change trips it |

**Bugs and design flaws this work found:**
- **Self-consistency was a no-op.** Every run used the same fixed seed, so "independent" runs were replays and
  could never disagree. The stability run made this obvious. Extra runs now use distinct seeds, with a unit test.
- **Error classification read status codes out of message text.** A `401`/`403` substring anywhere in the message
  counted as an auth error. One judge call failed as "AuthError" during a quota-heavy run, though I could not
  reproduce the exact message. Classification now uses the SDK's numeric `code` first, with word-boundary
  matching as a fallback.
- **Local schema errors were retried as "transient".** My judge schema used `Literal[1..5]`, which the SDK cannot
  convert, and the client retried it 5 times per model. Local `ValidationError`s are now `bad_request` and are
  not retried.
- **The headline numbers are one draw.** A rerun today with the production seed moved EX-1 from 3.65 to 3.10
  (Excellent → Average). EVALUATION.md now says so. The fix (n ≥ 3 median by default) changes scoring logic, so I
  put it on the roadmap instead of shipping it without a re-eval.
