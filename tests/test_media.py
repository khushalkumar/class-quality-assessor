"""Input validation and download failure paths."""
import pytest

from classqa import media
from classqa.errors import (CorruptMedia, DownloadError, DrivePermissionError, DriveQuotaError, MediaTooLarge,
                            MediaTooLong, MediaTooShort, NoAudioTrack, SilentAudio, UnsupportedFormat)


def test_good_file_probes(media_dir):
    info = media.probe(media_dir / "good.mp4")
    assert info.has_audio and info.has_video and 85 < info.duration_s < 95


def test_audio_only_is_accepted_without_video(media_dir):
    info = media.probe(media_dir / "audio_only.mp3")
    assert info.has_audio and not info.has_video


@pytest.mark.parametrize("name,exc", [("corrupt.mp4", CorruptMedia), ("noaudio.mp4", NoAudioTrack),
                                      ("short.mp4", MediaTooShort), ("notes.txt", UnsupportedFormat)])
def test_invalid_media_rejected(media_dir, name, exc):
    with pytest.raises(exc) as e:
        media.probe(media_dir / name)
    assert e.value.user_message  # always a human-readable message


def test_too_long_and_too_large(media_dir, monkeypatch):
    monkeypatch.setattr(media, "MAX_DURATION_S", 30)
    with pytest.raises(MediaTooLong):
        media.probe(media_dir / "good.mp4")
    monkeypatch.setattr(media, "MAX_BYTES", 10)
    with pytest.raises(MediaTooLarge):
        media.probe(media_dir / "good.mp4")


def test_silent_audio_detected(media_dir):
    sig = media.audio_signal_metrics(media_dir / "silent.mp4")
    with pytest.raises(SilentAudio):
        media.check_not_silent(sig)
    media.check_not_silent(media.audio_signal_metrics(media_dir / "good.mp4"))  # loud tone passes


def test_drive_folder_link_rejected(tmp_path):
    with pytest.raises(UnsupportedFormat):
        media.download_from_link("https://drive.google.com/drive/folders/abcdefghijklmnopqrstuvwxyz", tmp_path)


@pytest.mark.parametrize("msg,exc", [
    ("Too many users have viewed or downloaded this file recently", DriveQuotaError),
    ("Cannot retrieve the public link of the file. You may need to change the permission", DrivePermissionError),
    ("Connection reset by peer", DownloadError)])
def test_drive_errors_classified(msg, exc):
    e = media.classify_drive_error(msg)
    assert type(e) is exc and e.user_message


def test_http_403_is_permission_error(tmp_path, monkeypatch):
    import requests

    class Resp:
        status_code = 403
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            err = requests.HTTPError("403 Forbidden")
            err.response = self
            raise err

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    with pytest.raises(DrivePermissionError):
        media.download_from_link("https://example.com/class.mp4", tmp_path)


def test_html_page_link_rejected(tmp_path, monkeypatch):
    import requests

    class Resp:
        headers = {"content-type": "text/html; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    with pytest.raises(UnsupportedFormat):
        media.download_from_link("https://example.com/page", tmp_path)


class _Stream:
    """Fake streaming response; can drop the connection after `fail_after` bytes once."""

    def __init__(self, data, status=200, ctype="video/mp4", headers=None, fail_after=None):
        self.data, self.status_code, self.fail_after = data, status, fail_after
        self.headers = {"content-type": ctype, "content-length": str(len(data)), **(headers or {})}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self.data), 4):
            if self.fail_after is not None and i >= self.fail_after:
                raise ConnectionError("connection reset")
            yield self.data[i:i + 4]


def test_stream_download_resumes_with_range(tmp_path, monkeypatch):
    import requests
    data = b"0123456789abcdefghij"
    calls = []

    def get(url, params=None, stream=True, timeout=None, headers=None, allow_redirects=True):
        calls.append(headers or {})
        if not headers:
            return _Stream(data, fail_after=8, headers={"content-disposition": 'attachment; filename="class.mp4"'})
        start = int(headers["Range"].split("=")[1].rstrip("-"))
        return _Stream(data[start:], status=206)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(media.time, "sleep", lambda s: None)
    seen = []
    out = media.stream_download("https://x/y", tmp_path, progress=lambda d, t: seen.append((d, t)))
    assert out.read_bytes() == data and out.name == "class.mp4"
    assert calls[1]["Range"] == "bytes=8-" and seen[-1][0] == len(data)


def test_stream_download_enforces_size_limit(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(media, "MAX_BYTES", 10)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Stream(b"x" * 100))
    with pytest.raises(MediaTooLarge):
        media.stream_download("https://x/y.mp4", tmp_path)


def test_private_drive_file_is_permission_error(tmp_path, monkeypatch):
    import gdown
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Stream(b"<html>sign in</html>", ctype="text/html"))

    def fake_gdown(**k):
        raise RuntimeError("Cannot retrieve the public link of the file. You may need to change the permission")
    monkeypatch.setattr(gdown, "download", fake_gdown)
    with pytest.raises(DrivePermissionError):
        media.download_from_link("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz012345/view", tmp_path)


def test_link_must_be_http(tmp_path):
    with pytest.raises(DownloadError):
        media.download_from_link("ftp://nope", tmp_path)
