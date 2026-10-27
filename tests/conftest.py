"""Shared fakes and synthetic fixtures. No real API calls, no real class data."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classqa import rubric  # noqa: E402
from classqa.llm import CircuitBreaker, LLMClient, RawResponse  # noqa: E402
from classqa.store import SQLiteStore  # noqa: E402

QUOTE = "today we learn about forks in chess"


def transcript_chunk_json(n: int = 40) -> str:
    segs = []
    for i in range(n):
        t = f"{(i * 20) // 60:02d}:{(i * 20) % 60:02d}"
        if i % 2 == 0:
            text = "Good evening everyone, " + QUOTE if i == 0 else f"Can you find the best move number {i}?"
            segs.append({"start": t, "speaker": "Teacher", "text": text})
        else:
            segs.append({"start": t, "speaker": f"Student: Kid{i % 3}", "text": f"Knight to f{i % 8 + 1}"})
    return json.dumps({"language": "English", "segments": segs})


def assessment_json(score: int = 4, quote: str = QUOTE, holistic: str = "Good", confidence: str = "High",
                    flag: str | None = None) -> str:
    ev = [{"timestamp": "00:00", "observation": "Teacher states the topic", "quote": quote},
          {"timestamp": "00:20", "observation": "Board shown", "quote": "(visual)"}]
    return json.dumps({
        "class_summary": "Synthetic class.",
        "visual": {"teacher_camera_on": "throughout", "teacher_face_visible": "clearly",
                   "branded_background": "8x8 branded", "lighting": "good", "screen_share_used": True, "notes": "n"},
        "reviewer_concerns": ["none"],
        "dimensions": [{"key": k, "score": score, "summary": "s", "evidence": ev} for k in rubric.WEIGHTS],
        "strengths": [{"title": "t", "detail": "d", "timestamp": "00:01"}],
        "improvements": [{"title": "t", "detail": "d", "timestamp": "00:01", "coaching_tip": "c"}],
        "red_flags": ([{"issue": "i", "timestamp": "01:00", "severity": flag}] if flag else []),
        "holistic_rating": holistic, "confidence": confidence, "verdict": "v"})


class FakeBackend:
    """Scriptable transport. `script` is a list consumed per call: Exception instances are raised,
    strings are returned as response text; when exhausted, valid JSON for the schema is returned."""

    def __init__(self, script=None, assessment=None):
        self.script = list(script or [])
        self.calls = []
        self.assessment = assessment or [assessment_json()]
        self.deleted = []
        self._a = 0

    def generate(self, model, contents, *, system, schema, thinking, seed=None):
        self.calls.append((model, schema.__name__))
        self.seeds = [*getattr(self, "seeds", []), seed]
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return RawResponse(item, 100, 10)
        if schema.__name__ == "TranscriptChunk":
            return RawResponse(transcript_chunk_json(), 1000, 500)
        text = self.assessment[min(self._a, len(self.assessment) - 1)]
        self._a += 1
        return RawResponse(text, 30000, 3000)

    def upload(self, path, mime):
        class F:
            name = "files/fake"
        return F()

    def delete(self, name):
        self.deleted.append(name)


@pytest.fixture
def store():
    return SQLiteStore(":memory:")


@pytest.fixture
def make_llm(store):
    def _make(backend=None, **kw):
        kw.setdefault("sleep", lambda s: None)
        kw.setdefault("breaker", CircuitBreaker(threshold=3, cooldown_s=60))
        return LLMClient(backend or FakeBackend(), quota=store, **kw)
    return _make


def _ffmpeg(args, out):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, str(out)], check=True)
    return out


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    v = ["-f", "lavfi", "-i", "testsrc=size=320x180:rate=5"]
    tone = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000"]
    _ffmpeg([*v, *tone, "-t", "90", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"], d / "good.mp4")
    _ffmpeg([*v, "-t", "90", "-c:v", "libx264", "-pix_fmt", "yuv420p"], d / "noaudio.mp4")
    _ffmpeg([*v, "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "90", "-shortest", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-c:a", "aac"], d / "silent.mp4")
    _ffmpeg([*v, *tone, "-t", "10", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"], d / "short.mp4")
    _ffmpeg([*tone, "-t", "90", "-c:a", "libmp3lame"], d / "audio_only.mp3")
    (d / "corrupt.mp4").write_bytes(b"\x00\x01not really a video" * 1000)
    (d / "notes.txt").write_text("hello")
    return d
