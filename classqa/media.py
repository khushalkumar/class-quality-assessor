"""Media helpers: download, probe, audio extraction, frame sampling, cheap
signal-level metrics. Everything here is local (ffmpeg) - no API calls."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXT = {".mp3", ".m4a", ".wav", ".ogg", ".aac", ".flac", ".opus"}
SUPPORTED_EXT = VIDEO_EXT | AUDIO_EXT


from .errors import (CorruptMedia, DownloadError, DrivePermissionError, DriveQuotaError, MediaError, MediaTooLarge,
                     MediaTooLong, MediaTooShort, NoAudioTrack, SilentAudio, UnsupportedFormat)

MAX_BYTES = int(os.environ.get("CLASSQA_MAX_BYTES", str(2 * 1024 ** 3)))          # 2 GB
MAX_DURATION_S = float(os.environ.get("CLASSQA_MAX_DURATION_S", str(3 * 3600)))   # 3 h
MIN_DURATION_S = 60.0
__all__ = ["MediaError"]


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as e:  # ffmpeg missing
        raise MediaError("ffmpeg/ffprobe not found - install ffmpeg (packages.txt on Streamlit Cloud).") from e
    except subprocess.TimeoutExpired as e:
        raise CorruptMedia(f"Media processing timed out ({cmd[0]}); the file may be damaged.") from e


# ---------------------------------------------------------------- download
DRIVE_URL = "https://drive.usercontent.google.com/download"


def _drive_id(url: str) -> str | None:
    m = re.search(r"/d/([\w-]{20,})", url) or re.search(r"[?&]id=([\w-]{20,})", url)
    return m.group(1) if m else None


def _filename(resp, fallback: str) -> str:
    cd = resp.headers.get("content-disposition", "")
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd) or re.search(r'filename="?([^";]+)"?', cd)
    name = Path(requests_unquote(m.group(1)) if m else fallback).name
    if Path(name).suffix.lower() not in SUPPORTED_EXT:
        name += ".mp4"
    return re.sub(r"[^\w.\- ]", "_", name)


def requests_unquote(s: str) -> str:
    from urllib.parse import unquote
    return unquote(s)


def stream_download(url: str, dest_dir: Path, *, params: dict | None = None, fallback_name: str = "recording.mp4",
                    progress=None, max_retries: int = 3, chunk: int = 1 << 20) -> Path:
    """Stream a URL to disk in 1 MB chunks (never held in memory), enforce MAX_BYTES while
    downloading, report progress, and resume with HTTP Range after a dropped connection."""
    import requests

    out: Path | None = None
    done, total, attempt = 0, None, 0
    while True:
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, params=params, stream=True, timeout=(15, 120), headers=headers,
                              allow_redirects=True) as r:
                r.raise_for_status()
                ctype = r.headers.get("content-type", "")
                if "text/html" in ctype:
                    raise UnsupportedFormat("That link returned a web page, not a media file. Use a direct file "
                                            "link or a Google Drive *file* link shared with 'Anyone with the link'.")
                if out is None:
                    out = dest_dir / _filename(r, fallback_name)
                if done and r.status_code != 206:      # server ignored Range -> restart cleanly
                    done = 0
                length = r.headers.get("content-length")
                if total is None and length:
                    total = int(length) + done
                    if total > MAX_BYTES:
                        raise MediaTooLarge(f"The linked file is {total / 1024 ** 2:.0f} MB; the limit is "
                                            f"{MAX_BYTES // 1024 ** 2} MB.")
                with open(out, "ab" if done else "wb") as f:
                    for block in r.iter_content(chunk_size=chunk):
                        f.write(block)
                        done += len(block)
                        if done > MAX_BYTES:
                            raise MediaTooLarge(f"Download exceeded the {MAX_BYTES // 1024 ** 2} MB limit.")
                        if progress:
                            progress(done, total)
            return out
        except MediaError:
            raise
        except Exception as e:  # noqa: BLE001
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (401, 403, 404):
                raise DrivePermissionError(f"The link is not publicly accessible (HTTP {code}). Share it as "
                                           "'Anyone with the link can view'.") from e
            if code == 429:
                raise DriveQuotaError("The file host is rate-limiting downloads (HTTP 429). Try again later.") from e
            attempt += 1
            if attempt > max_retries:
                raise DownloadError(f"Download failed after {max_retries} retries: {str(e)[:150]}") from e
            time.sleep(min(2 ** attempt, 10))


def download_from_link(url: str, dest_dir: str | Path, progress=None) -> Path:
    """Download a recording from a Google Drive file link or a direct URL (streamed to disk)."""
    url = url.strip()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not re.match(r"^https?://", url):
        raise DownloadError("Please paste a full http(s) link.")
    if "drive.google.com" in url or "docs.google.com" in url:
        if "/folders/" in url:
            raise UnsupportedFormat("That is a Google Drive *folder* link. Share the individual recording file instead.")
        fid = _drive_id(url)
        if not fid:
            raise DownloadError("Could not find a file id in that Google Drive link.", detail=url[:120])
        try:
            # usercontent endpoint with confirm=t skips the large-file virus-scan page
            return stream_download(DRIVE_URL, dest_dir, params={"id": fid, "export": "download", "confirm": "t"},
                                   fallback_name=f"{fid}.mp4", progress=progress)
        except (DrivePermissionError, MediaTooLarge):
            raise
        except MediaError as first:
            # fallback: gdown handles older confirm-token flows
            try:
                import gdown
                out = gdown.download(id=fid, output=str(dest_dir) + os.sep, quiet=True, fuzzy=True)
            except Exception as e:  # noqa: BLE001
                raise classify_drive_error(str(e)) from e
            if not out or not Path(out).exists():
                raise first
            return Path(out)
    return stream_download(url, dest_dir, progress=progress)


def classify_drive_error(msg: str) -> MediaError:
    low = msg.lower()
    if "too many users" in low or "quota" in low or "rate limit" in low:
        return DriveQuotaError("Google Drive's download quota for this file is exhausted (too many recent downloads). "
                               "Try again later or upload the file directly.")
    if "permission" in low or "cannot retrieve the public link" in low or "access denied" in low or "403" in low:
        return DrivePermissionError("Google Drive refused the download. Share the file as "
                                    "'Anyone with the link can view' and try again.")
    return DownloadError(f"Google Drive download failed: {msg[:150]}")


# ---------------------------------------------------------------- probe
@dataclass
class MediaInfo:
    duration_s: float
    has_video: bool
    has_audio: bool
    width: int | None = None
    height: int | None = None


def probe(path: str | Path) -> MediaInfo:
    """Validate a recording before any API spend. Raises a specific MediaError subclass."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise CorruptMedia("The file is empty or missing.")
    if path.suffix.lower() not in SUPPORTED_EXT:
        raise UnsupportedFormat(f"Unsupported file type '{path.suffix}'. Use mp4, mov, mkv, webm or common audio formats.")
    if path.stat().st_size > MAX_BYTES:
        raise MediaTooLarge(f"File is larger than the {MAX_BYTES // 1024 ** 2} MB limit.")
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
              "-of", "json", str(path)], timeout=120)
    if r.returncode != 0:
        raise CorruptMedia("This file could not be read - it looks corrupt or is not a real audio/video file.",
                           detail=r.stderr.decode(errors="ignore")[-300:])
    try:
        data = json.loads(r.stdout or b"{}")
    except json.JSONDecodeError as e:
        raise CorruptMedia("This file could not be read (unparseable metadata).") from e
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    dur = float(data.get("format", {}).get("duration") or 0)
    if not a:
        raise NoAudioTrack("This file has no audio track - a class recording needs audio to be assessed.")
    if dur <= 0:
        raise CorruptMedia("The recording has no readable duration - it may be truncated.")
    if dur < MIN_DURATION_S:
        raise MediaTooShort(f"Recording is only {dur:.0f}s long - too short to assess a class.")
    if dur > MAX_DURATION_S:
        raise MediaTooLong(f"Recording is {dur / 3600:.1f} h long; the limit is {MAX_DURATION_S / 3600:.0f} h.")
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
        raise CorruptMedia("Audio could not be extracted from this file.", detail=r.stderr.decode(errors="ignore")[-300:])
    return Path(out_path)


def check_not_silent(sig: dict) -> None:
    """Reject recordings whose audio is effectively silent (e.g. muted capture)."""
    mx, mean = sig.get("max_volume_db"), sig.get("mean_volume_db")
    if (mx is not None and mx < -45) or (mean is not None and mean < -65):
        raise SilentAudio("The audio track is silent or almost silent (max level "
                          f"{mx} dB). Check that the recording captured meeting audio.")


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
