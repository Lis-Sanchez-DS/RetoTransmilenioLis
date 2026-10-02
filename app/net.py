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


class GatewayError(requests.exceptions.ConnectionError):
    """A gateway/overload status (429/502/503/504) - the server (or the proxy
    in front of it) is momentarily unavailable. Treated as transient."""


GATEWAY_STATUSES = (429, 502, 503, 504)


def raise_for_gateway_error(response) -> None:
    """Opt-in for callers talking to a gateway-fronted service (Supabase
    Storage): turns 429/502/503/504 into a retryable GatewayError. Not applied
    to the Pulso API on purpose - there a 5xx is a real application error."""
    status = getattr(response, "status_code", None)
    if status in GATEWAY_STATUSES:
        raise GatewayError(f"HTTP {status} de la pasarela")


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return False  # got a real response - a 4xx/5xx is not "server down"
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return True
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    return False


def with_retries(func, *args, retries: int | None = None, backoff: float | None = None, **kwargs):
    """Call func(*args, **kwargs), retrying only transient connection failures.

    Non-transient exceptions (HTTP errors, bad JSON, etc.) propagate
    immediately, unretried - those are bugs, not flakiness.
    """
    retries = RETRIES if retries is None else retries
    backoff = BACKOFF_SECONDS if backoff is None else backoff
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if not _is_transient(exc):
                raise
            last_error = exc
            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1))
                print(f"Fallo transitorio (intento {attempt}/{retries}): {exc}. Reintentando en {wait}s.", flush=True)
                time.sleep(wait)
    raise UpstreamUnavailable(f"Servidor no disponible tras {retries} intentos: {last_error}") from last_error
