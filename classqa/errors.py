"""Typed, user-facing failures. Every failure a job can end in has a stable
`code` (for metrics/ops dashboards) and a plain-language `user_message`."""
from __future__ import annotations


class ClassQAError(RuntimeError):
    code = "internal_error"
    retryable = False

    def __init__(self, user_message: str, *, detail: str | None = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail or user_message


# ---- input / media -------------------------------------------------------------
class MediaError(ClassQAError):
    code = "media_invalid"


class UnsupportedFormat(MediaError):
    code = "unsupported_format"


class CorruptMedia(MediaError):
    code = "corrupt_media"


class MediaTooLarge(MediaError):
    code = "media_too_large"


class MediaTooLong(MediaError):
    code = "media_too_long"


class MediaTooShort(MediaError):
    code = "media_too_short"


class NoAudioTrack(MediaError):
    code = "no_audio"


class SilentAudio(MediaError):
    code = "silent_audio"


class DownloadError(MediaError):
    code = "download_failed"
    retryable = True


class DrivePermissionError(DownloadError):
    code = "drive_permission"
    retryable = False


class DriveQuotaError(DownloadError):
    code = "drive_quota"


# ---- model ---------------------------------------------------------------------
class LLMError(ClassQAError):
    code = "llm_error"
    retryable = True


class LLMUnavailable(LLMError):
    code = "llm_unavailable"


class CircuitOpen(LLMError):
    code = "circuit_open"


class SchemaError(LLMError):
    code = "malformed_output"


class AuthError(LLMError):
    code = "llm_auth"
    retryable = False


# ---- guards --------------------------------------------------------------------
class BudgetExceeded(ClassQAError):
    code = "budget_exceeded"


class DailyQuotaExceeded(ClassQAError):
    code = "daily_quota"
    retryable = True
