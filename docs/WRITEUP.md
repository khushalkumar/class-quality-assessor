# Brief explanation: 8x8 Class Quality Assessor (Khushal Kumar)

**Design.** The Streamlit UI only submits jobs and polls them. A worker
runs a staged pipeline, and each job moves through queued, extracting, transcribing, scoring, then done, failed
or needs_review in a SQLite store behind an interface. Every expensive stage is checkpointed under the file's hash plus the prompt, rubric and model versions,
so retries never pay twice. DESIGN.md covers the Postgres, object
storage and queue version, and the cost: about $0.02–0.03 per class, or about $25 per 1,000 (estimates).

**Failure handling.** Each failure gets its own code and a plain message in the UI: corrupt or too-long media, no
audio or silent audio, Drive permission or quota errors. Gemini 429/503s get backoff with jitter, a model fallback
chain and a circuit breaker. Malformed JSON gets one repair retry. There is a per-job budget
and a daily quota guard. In the real end-to-end test, five 503s were absorbed by fallback without the user seeing
anything. A live 54 MB browser upload stalled, so the Link tab is now the recommended path: the server streams the
file from Drive with progress, a size check and resume.

**Assessment.** The rubric comes from 8x8's feedback, which weighs how a class looks and sounds. Gemini gets the transcript, metrics, frames, crops of the teacher's tile and audio excerpts. It
scores six dimensions with quotes, and code decides the label. Final eval: 7/10 exact, 9/10 within one level
(5/10 exact with leave-one-out). Quotes are checked against the transcript, and the 2% not found are removed.
Boundary scores, red flags and model disagreement go to human review; that flags 2 of the 3 misses.

**Limits.** Ten labelled classes is very little, and review routing is too eager (6/10). Visual checks vary
between runs. Media is deleted after each job, and real use needs 8x8 to approve the vendor.
