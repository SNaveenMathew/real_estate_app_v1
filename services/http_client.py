"""One HTTP client for the public geocoding services.

Three modules called the Census endpoints three ways: ``geo_utils`` used a bare
``requests.get`` with a 10 s timeout and no retry, the single-address geocoder the same with
15 s, and the batch geocoder its own 180 s call with a hand-rolled sleep loop. A Census
outage therefore behaved differently depending on which path a row happened to take - one
retried for a minute, two silently dropped the row. This module owns the session, the default
timeout and the retry rule, so every caller degrades the same way.

Retries cover what is actually transient: connection errors, timeouts and 5xx. A 4xx is a
malformed request and is raised immediately. Callers that own a longer backoff (the batch
geocoder, which prints progress between attempts) pass ``retries=0``.
"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15.0
DEFAULT_RETRIES = 2
_BACKOFF_SECONDS = 1.0

_session: requests.Session | None = None


def session() -> requests.Session:
    """The shared session (connection pooling, one place to configure headers)."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": "real-estate-app/1.0 (+local analysis)"})
    return _session


def reset_session() -> None:
    """Close and forget the session (tests, and after a connection-level failure)."""
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
    _session = None


def request(
    method: str, url: str, *, timeout: float | None = None, retries: int = DEFAULT_RETRIES, **kwargs
) -> requests.Response:
    """Perform one request, retrying transient failures with exponential backoff."""
    attempts = max(int(retries), 0) + 1
    last_exc: Exception | None = None
    last_response: requests.Response | None = None
    for attempt in range(attempts):
        try:
            response = session().request(method, url, timeout=timeout or DEFAULT_TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            last_exc, last_response = exc, None
            logger.warning("%s %s failed (attempt %s/%s): %s", method, url, attempt + 1, attempts, exc)
        else:
            if response.status_code < 500:
                response.raise_for_status()  # 4xx is not transient
                return response
            last_exc, last_response = None, response
            logger.warning(
                "%s %s returned %s (attempt %s/%s)", method, url, response.status_code, attempt + 1, attempts
            )
        if attempt + 1 < attempts:
            time.sleep(_BACKOFF_SECONDS * (2**attempt))
    if last_response is not None:
        last_response.raise_for_status()
    assert last_exc is not None
    raise last_exc


def get(url: str, **kwargs) -> requests.Response:
    return request("GET", url, **kwargs)


def post(url: str, **kwargs) -> requests.Response:
    return request("POST", url, **kwargs)
