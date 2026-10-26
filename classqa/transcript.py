"""Transcript handling: parse Zoom-style WebVTT, identify the teacher,
pseudonymise speakers, render a compact timestamped transcript for the LLM."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field

TS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})|(\d{1,2}):(\d{2})[.,](\d{1,3})")


@dataclass
class Segment:
    start: float
    end: float
    speaker: str          # pseudonymised label: "Teacher", "Student 1", ...
    role: str             # "teacher" | "student" | "unknown"
    text: str
    names_mentioned: list[str] = field(default_factory=list)  # student labels addressed in this line


@dataclass
class Transcript:
    segments: list[Segment]
    source: str                      # "vtt" | "gemini"
    language: str = "unknown"
    n_students: int = 0

    def to_dict(self) -> dict:
        return {"source": self.source, "language": self.language, "n_students": self.n_students,
                "segments": [asdict(s) for s in self.segments]}


def _ts(s: str) -> float:
    m = TS.search(s)
    if not m:
        raise ValueError(s)
    if m.group(1):
        h, mi, se, ms = m.group(1, 2, 3, 4)
    else:
        h, (mi, se, ms) = "0", m.group(5, 6, 7)
    return int(h) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000


def fmt_ts(t: float) -> str:
    t = int(max(t, 0))
    return f"{t // 3600:d}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


def parse_vtt(text: str) -> list[tuple[float, float, str | None, str]]:
    """Return raw cues (start, end, speaker_or_None, text)."""
    cues = []
    blocks = re.split(r"\n\s*\n", text.replace("\r", ""))
    for b in blocks:
        lines = [ln for ln in b.strip().split("\n") if ln.strip()]
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        a, _, z = lines[idx].partition("-->")
        try:
            start, end = _ts(a), _ts(z)
        except ValueError:
            continue
        body = " ".join(lines[idx + 1:]).strip()
        body = re.sub(r"<[^>]+>", "", body)
        if not body:
            continue
        m = re.match(r"^([^:]{1,60}):\s+(.*)$", body)
        spk, txt = (m.group(1).strip(), m.group(2).strip()) if m else (None, body)
        cues.append((start, end, spk, txt))
    return cues


def _name_tokens(name: str) -> list[str]:
    toks = re.findall(r"[A-Za-z]{3,}", name)
    stop = {"the", "mam", "sir", "maam", "student", "teacher", "iphone", "ipad", "galaxy", "android", "laptop"}
    return [t.lower() for t in toks if t.lower() not in stop]


def build_transcript(raw: list[tuple[float, float, str | None, str]], source: str,
                     teacher_hint: str | None = None, extra_student_names: dict[str, list[str]] | None = None,
                     language: str = "unknown") -> Transcript:
    """Assign roles (teacher = speaker with the most words unless hinted),
    replace real names with labels in speakers AND text, and record which
    student labels the teacher addressed by name in each line."""
    # carry speaker forward for un-prefixed cues
    filled, last = [], None
    for s, e, spk, t in raw:
        spk = spk or last or "Unknown"
        last = spk
        filled.append((s, e, spk, t))
    words = defaultdict(int)
    for _, _, spk, t in filled:
        words[spk] += len(t.split())
    if not words:
        return Transcript([], source, language)
    teacher = teacher_hint if teacher_hint in words else max(words, key=words.get)

    labels: dict[str, str] = {teacher: "Teacher"}
    k = 0
    for _, _, spk, _ in filled:
        if spk not in labels:
            if spk.lower().startswith("unknown"):
                labels[spk] = "Unknown"
                continue
            k += 1
            labels[spk] = f"Student {k}"

    # name token -> label, for pseudonymising text and tracking name use
    tok2label: dict[str, str] = {}
    for spk, lab in labels.items():
        for tok in _name_tokens(spk):
            tok2label.setdefault(tok, lab)
    for lab, names in (extra_student_names or {}).items():
        for n in names:
            for tok in _name_tokens(n):
                tok2label.setdefault(tok, lab)
    pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(tok2label, key=len, reverse=True))) + r")\b",
                     re.I) if tok2label else None

    segs = []
    for s, e, spk, t in filled:
        lab = labels.get(spk, "Unknown")
        role = "teacher" if lab == "Teacher" else ("unknown" if lab == "Unknown" else "student")
        mentioned = []
        if pat:
            def rep(m):
                l2 = tok2label[m.group(1).lower()]
                if l2 != "Teacher":
                    mentioned.append(l2)
                return f"[{l2}]"
            t = pat.sub(rep, t)
        segs.append(Segment(round(s, 2), round(e, 2), lab, role, t, sorted(set(mentioned))))
    return Transcript(segs, source, language, n_students=k)


def from_vtt_text(text: str) -> Transcript:
    return build_transcript(parse_vtt(text), "vtt")


def render(tr: Transcript, max_chars: int = 160_000) -> str:
    """Compact '[mm:ss] Speaker: text' lines, merging consecutive same-speaker cues."""
    lines, cur = [], None
    for s in tr.segments:
        if cur and cur[1] == s.speaker and s.start - cur[2] < 4 and len(cur[3]) < 600:
            cur[3] += " " + s.text
            cur[2] = s.end
        else:
            if cur:
                lines.append(f"[{fmt_ts(cur[0])}] {cur[1]}: {cur[3]}")
            cur = [s.start, s.speaker, s.end, s.text]
    if cur:
        lines.append(f"[{fmt_ts(cur[0])}] {cur[1]}: {cur[3]}")
    out = "\n".join(lines)
    if len(out) > max_chars:  # keep beginning and end (intro + closure matter most), trim the middle
        half = max_chars // 2
        out = out[:half] + "\n[... middle of transcript trimmed for length ...]\n" + out[-half:]
    return out


def from_dict(d: dict) -> Transcript:
    return Transcript([Segment(**s) for s in d["segments"]], d["source"], d.get("language", "unknown"),
                      d.get("n_students", 0))
