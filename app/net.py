"""Shared network retry helpers.

Distinguishes transient "the server didn't answer at all" failures (DNS,
connect timeout, refused connection) from real application errors (bad
auth, 4xx/5xx, malformed payload). The former get retried and, if still
failing after that, surfaced as UpstreamUnavailable so the caller can
treat "the grading API is down" differently from "our code is broken" -
the latter should keep failing fast, exactly as before.
"""
import time
import urllib.error

import requests

RETRIES = 3
BACKOFF_SECONDS = 5  # doubles each attempt: 5s, 10s, 20s


class UpstreamUnavailable(RuntimeError):
    """Raised when the upstream API couldn't be reached after retries."""


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return False  # got a real response - a 4xx/5xx is not "server down"
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return True
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    return False


def with_retries(func, *args, **kwargs):
    """Call func(*args, **kwargs), retrying only transient connection failures.

    Non-transient exceptions (HTTP errors, bad JSON, etc.) propagate
    immediately, unretried - those are bugs, not flakiness.
    """
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if not _is_transient(exc):
                raise
            last_error = exc
            if attempt < RETRIES:
                wait = BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"Fallo transitorio (intento {attempt}/{RETRIES}): {exc}. Reintentando en {wait}s.", flush=True)
                time.sleep(wait)
    raise UpstreamUnavailable(f"Servidor no disponible tras {RETRIES} intentos: {last_error}") from last_error
