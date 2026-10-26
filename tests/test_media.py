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
