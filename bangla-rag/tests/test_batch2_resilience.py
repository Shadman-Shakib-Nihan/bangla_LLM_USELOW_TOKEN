from __future__ import annotations

import pytest

from src.batch2_gemini import (
    GeminiAPIError,
    classify_gemini_error,
    compute_backoff,
)

DAILY_QUOTA_BODY = """
{
  "error": {
    "code": 429,
    "message": "You exceeded your current quota ... Generate content requests per day limit exceeded on Free Tier. Please retry in 18h35m9.672718829s.",
    "status": "RESOURCE_EXHAUSTED"
  }
}
"""

RPM_BODY = """
{
  "error": {
    "code": 429,
    "message": "Generate content requests per minute per project per model limit exceeded. Please retry in 12s.",
    "status": "RESOURCE_EXHAUSTED"
  }
}
"""


def test_daily_quota_classified_and_not_rate_limit() -> None:
    err = GeminiAPIError(429, DAILY_QUOTA_BODY)
    assert err.is_daily_quota_exhausted is True
    assert err.is_rate_limit is False
    assert classify_gemini_error(err) == "daily_quota_exhausted"
    assert err.retry_after_seconds == pytest.approx(18 * 3600 + 35 * 60 + 9.672718829)


def test_per_minute_429_is_rate_limit_not_daily() -> None:
    err = GeminiAPIError(429, RPM_BODY)
    assert err.is_daily_quota_exhausted is False
    assert err.is_rate_limit is True
    assert classify_gemini_error(err) == "rate_limit"
    assert err.retry_after_seconds == pytest.approx(12.0)


def test_transient_status_classification() -> None:
    assert classify_gemini_error(GeminiAPIError(503, "{}")) == "transient"
    assert classify_gemini_error(GeminiAPIError(500, "{}")) == "transient"


def test_timeout_classification() -> None:
    assert classify_gemini_error(TimeoutError("read timed out")) == "timeout"


def test_backoff_is_bounded_exponential() -> None:
    assert compute_backoff(1, None, base=2.0, factor=2.0, cap=60.0) == 2.0
    assert compute_backoff(2, None, base=2.0, factor=2.0, cap=60.0) == 4.0
    assert compute_backoff(3, None, base=2.0, factor=2.0, cap=60.0) == 8.0
    assert compute_backoff(10, None, base=2.0, factor=2.0, cap=60.0) == 60.0


def test_backoff_honors_retry_after_within_cap() -> None:
    assert compute_backoff(1, 15.0, base=2.0, factor=2.0, cap=60.0) == 15.0
    assert compute_backoff(1, 3600.0, base=2.0, factor=2.0, cap=60.0) == 60.0
