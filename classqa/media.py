"""Media helpers: download, probe, audio extraction, frame sampling, cheap
signal-level metrics. Everything here is local (ffmpeg) - no API calls."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXT = {".mp3", ".m4a", ".wav", ".ogg", ".aac", ".flac", ".opus"}
SUPPORTED_EXT = VIDEO_EXT | AUDIO_EXT


class MediaError(RuntimeError):
    """User-facing media problem (bad link, unsupported file, ffmpeg failure)."""


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as e:  # ffmpeg missing
        raise MediaError("ffmpeg/ffprobe not found - install ffmpeg (packages.txt on Streamlit Cloud).") from e
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"Media processing timed out: {' '.join(cmd[:3])}...") from e


# ---------------------------------------------------------------- download
def download_from_link(url: str, dest_dir: str | Path) -> Path:
    """Download a recording from a Google Drive share link or a direct URL."""
    url = url.strip()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if "drive.google.com" in url or "docs.google.com" in url:
        import gdown

        m = re.search(r"/d/([\w-]{20,})", url) or re.search(r"[?&]id=([\w-]{20,})", url)
        if not m:
            raise MediaError("Could not find a file id in that Google Drive link (folder links are not supported - share the file).")
        out = gdown.download(id=m.group(1), output=str(dest_dir) + os.sep, quiet=True, fuzzy=True)
        if not out or not Path(out).exists():
            raise MediaError("Google Drive download failed - make sure the file is shared as 'Anyone with the link'.")
        return Path(out)

    import requests

    try:
        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            if "text/html" in ctype:
                raise MediaError("That link returned a web page, not a media file. Use a direct file link or a Google Drive file link.")
            name = Path(url.split("?")[0]).name or "recording.mp4"
            if Path(name).suffix.lower() not in SUPPORTED_EXT:
                name += ".mp4"
            out = dest_dir / name
            with open(out, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    except MediaError:
        raise
    except Exception as e:  # noqa: BLE001
        raise MediaError(f"Download failed: {e}") from e
    return out


# ---------------------------------------------------------------- probe
@dataclass
class MediaInfo:
    duration_s: float
    has_video: bool
    has_audio: bool
    width: int | None = None
    height: int | None = None


def probe(path: str | Path) -> MediaInfo:
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
              "-of", "json", str(path)], timeout=120)
    if r.returncode != 0:
        raise MediaError("Could not read this file - is it a valid audio/video recording?")
    data = json.loads(r.stdout or b"{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    dur = float(data.get("format", {}).get("duration") or 0)
    if not a:
        raise MediaError("This file has no audio track - a class recording needs audio to be assessed.")
    if dur < 60:
        raise MediaError(f"Recording is only {dur:.0f}s long - too short to assess a class.")
    return MediaInfo(dur, v is not None, True, v and v.get("width"), v and v.get("height"))


# ---------------------------------------------------------------- audio
def extract_audio(path: str | Path, out_path: str | Path, start: float | None = None,
                  duration: float | None = None, bitrate: str = "32k") -> Path:
    """Mono 16 kHz MP3 - ~14 MB per hour, plenty for speech understanding."""
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.2f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.2f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-b:a", bitrate, str(out_path)]
    r = _run(cmd)
    if r.returncode != 0 or not Path(out_path).exists():
        raise MediaError("Audio extraction failed: " + r.stderr.decode(errors="ignore")[-300:])
    return Path(out_path)


def audio_signal_metrics(path: str | Path, noise_db: int = -35, min_silence: float = 8.0) -> dict:
    """Dead-air and loudness stats via ffmpeg silencedetect/volumedetect."""
    r = _run(["ffmpeg", "-v", "info", "-i", str(path), "-vn", "-ac", "1",
              "-af", f"silencedetect=noise={noise_db}dB:d={min_silence},volumedetect", "-f", "null", "-"])
    err = r.stderr.decode(errors="ignore")
    durs = [float(x) for x in re.findall(r"silence_duration: ([\d.]+)", err)]
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    mean_db = re.search(r"mean_volume: ([-\d.]+) dB", err)
    max_db = re.search(r"max_volume: ([-\d.]+) dB", err)
    longest_i = int(np.argmax(durs)) if durs else None
    return {
        "dead_air_total_s": round(sum(durs), 1),
        "dead_air_segments": len(durs),
        "longest_dead_air_s": round(max(durs), 1) if durs else 0.0,
        "longest_dead_air_at_s": round(starts[longest_i], 1) if longest_i is not None and longest_i < len(starts) else None,
        "mean_volume_db": float(mean_db.group(1)) if mean_db else None,
        "max_volume_db": float(max_db.group(1)) if max_db else None,
    }


# ---------------------------------------------------------------- frames
def frame_jpeg(path: str | Path, t: float, width: int = 768) -> bytes | None:
    r = _run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
              "-vf", f"scale={width}:-2", "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "-"], timeout=120)
    return r.stdout if r.returncode == 0 and r.stdout else None


def corner_crop_jpeg(path: str | Path, t: float, out_w: int = 768) -> bytes | None:
    """Zoom places the active speaker's thumbnail in the top-right corner during screen
    share. At full-frame resolution the model cannot see the teacher's face/background
    there, so we send an upscaled crop of that corner alongside each frame."""
    r = _run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
              "-vf", f"crop=iw*0.28:ih*0.30:iw*0.72:0,scale={out_w}:-2", "-q:v", "4",
              "-f", "image2pipe", "-vcodec", "mjpeg", "-"], timeout=120)
    return r.stdout if r.returncode == 0 and r.stdout else None


def _frame_gray_small(path: str | Path, t: float, w: int = 96, h: int = 54) -> np.ndarray | None:
    r = _run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
              "-vf", f"scale={w}:{h}", "-pix_fmt", "gray", "-f", "rawvideo", "-"], timeout=120)
    if r.returncode != 0 or len(r.stdout) != w * h:
        return None
    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(h, w)


def sample_times(duration: float, n: int, margin: float = 0.02) -> list[float]:
    lo, hi = duration * margin, duration * (1 - margin)
    return [lo + (hi - lo) * i / max(n - 1, 1) for i in range(n)]


def visual_signal_metrics(path: str | Path, duration: float, n: int = 40) -> dict:
    """Cheap pixel statistics over n evenly spaced frames.
    - near-blank frames (mostly black) => camera off / nothing shared
    - very dark frames => poor lighting
    - static share between consecutive samples => long idle screens"""
    blanks, darks, means, prev, static = 0, 0, [], None, 0
    got = 0
    for t in sample_times(duration, n):
        g = _frame_gray_small(path, t)
        if g is None:
            continue
        got += 1
        m = float(g.mean())
        means.append(m)
        if (g < 30).mean() > 0.85:
            blanks += 1
        elif m < 60:
            darks += 1
        if prev is not None and np.abs(g.astype(int) - prev.astype(int)).mean() < 1.0:
            static += 1
        prev = g
    if not got:
        return {}
    return {
        "frames_sampled": got,
        "blank_frame_pct": round(100 * blanks / got, 1),
        "unchanged_consecutive_frame_pct": round(100 * static / max(got - 1, 1), 1),
    }


def tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="classqa_"))


def cleanup(p: str | Path) -> None:
    shutil.rmtree(p, ignore_errors=True)
