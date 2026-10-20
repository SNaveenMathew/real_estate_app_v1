"""Model-call resilience: tell an outage from a bad request, remember outages, and let callers degrade.

Every model call in the app used to be a bare ``client.invoke(...)``.  With the model server down, the first call
raised, the exception escaped the orchestrator, and the user got HTTP 500 - even for questions the app can answer with
no model at all (the planner, the deterministic SQL compiler and the derived-measure providers need none).  With the
server slow, each call could wait out the full HTTP timeout, up to three times (the client library's default retries).

``invoke()`` is the one place a model is called:

* **classify** - connection errors, timeouts, HTTP 5xx / 429 are *transient*; a 400 (bad request, context overflow) is
  not the service's fault and is re-raised untouched, so existing handling of those is unchanged;
* **circuit breaker** - after ``llm_breaker_threshold`` consecutive transient failures the model is declared down and
  calls fail INSTANTLY for ``llm_breaker_cooldown_s``; then exactly one probe call is allowed, and its outcome closes or
  re-opens the breaker.  A dead server therefore costs one timeout, not one per call;
* **ModelUnavailable** - what callers see.  Each caller has a deterministic fallback (a fixed program, the compiled
  SQL, the raw evidence) and uses it: the app keeps working, in a degraded mode that says so.

Turns also carry a deadline (``turn()`` / ``use_turn()``).  Cancellation is cooperative: ``check_cancelled()`` runs before
every model call, so a turn past its deadline starts no further work, and a worker thread never has to be killed.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from config import settings


class ModelUnavailable(RuntimeError):
    """The language-model service is not responding (or is known not to be); callers degrade instead of waiting."""


class TurnCancelled(RuntimeError):
    """The turn's deadline passed or it was cancelled; no further model calls are started."""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
_TRANSIENT_HINTS = ("connect", "timeout", "timed out", "unavailable", "overloaded", "ratelimit", "rate limit", "remoteprotocol",
                    "readerror", "reset by peer", "broken pipe", "badgateway", "bad gateway", "gateway", "internalservererror",
                    "server disconnected")


def is_transient(exc: BaseException) -> bool:
    """True when the failure is the SERVICE's (retrying later may work); False for a bad request, context overflow, ..."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(code, int):
        return code >= 500 or code in (408, 425, 429)
    text = f"{type(exc).__name__} {exc}".lower()
    return any(hint in text for hint in _TRANSIENT_HINTS)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
class CircuitBreaker:
    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, threshold: int, cooldown_s: float, clock=time.monotonic):
        self.threshold, self.cooldown_s, self._clock = max(1, int(threshold)), float(cooldown_s), clock
        self._lock = threading.Lock()
        self.state, self.failures, self._opened_at, self._probing = self.CLOSED, 0, 0.0, False

    def before_call(self) -> None:
        with self._lock:
            if self.state == self.CLOSED:
                return
            if self.state == self.OPEN:
                if self._clock() - self._opened_at < self.cooldown_s:
                    raise ModelUnavailable(self._message())
                self.state, self._probing = self.HALF_OPEN, False
            if self._probing:                                    # one probe at a time; everyone else fails fast
                raise ModelUnavailable(self._message())
            self._probing = True

    def record_success(self) -> None:
        with self._lock:
            self.state, self.failures, self._probing = self.CLOSED, 0, False

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state == self.HALF_OPEN or self.failures >= self.threshold:
                self.state, self._opened_at, self._probing = self.OPEN, self._clock(), False

    def _message(self) -> str:
        wait = max(0.0, self.cooldown_s - (self._clock() - self._opened_at))
        return f"The language model service is not responding; not trying again for about {wait:.0f} seconds."

    def status(self) -> dict:
        with self._lock:
            retry_in = max(0.0, self.cooldown_s - (self._clock() - self._opened_at)) if self.state == self.OPEN else 0.0
            return {"state": self.state, "consecutive_failures": self.failures, "retry_in_s": round(retry_in, 1)}


_BREAKER: CircuitBreaker | None = None
_BREAKER_LOCK = threading.Lock()


def breaker() -> CircuitBreaker:
    global _BREAKER
    if _BREAKER is None:
        with _BREAKER_LOCK:
            if _BREAKER is None:
                _BREAKER = CircuitBreaker(settings.llm_breaker_threshold, settings.llm_breaker_cooldown_s)
    return _BREAKER


def configure_breaker(threshold: int, cooldown_s: float, clock=time.monotonic) -> CircuitBreaker:
    """Replace the breaker (tests, or reconfiguration at start-up)."""
    global _BREAKER
    with _BREAKER_LOCK:
        _BREAKER = CircuitBreaker(threshold, cooldown_s, clock)
    return _BREAKER


def reset() -> None:
    global _BREAKER
    with _BREAKER_LOCK:
        _BREAKER = None


# ---------------------------------------------------------------------------
# Turn deadline / cooperative cancellation
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    deadline: float | None = None                       # time.monotonic() value
    cancel: threading.Event = field(default_factory=threading.Event)


_TURN: ContextVar = ContextVar("agent_turn", default=None)


@contextmanager
def turn(deadline_s: float | None = None):
    t = Turn(time.monotonic() + deadline_s if deadline_s else None)
    token = _TURN.set(t)
    try:
        yield t
    finally:
        _TURN.reset(token)


@contextmanager
def use_turn(t: Turn):
    """Adopt a Turn created elsewhere (the HTTP layer keeps the handle so it can cancel)."""
    token = _TURN.set(t)
    try:
        yield t
    finally:
        _TURN.reset(token)


def check_cancelled() -> None:
    t = _TURN.get()
    if t is not None and (t.cancel.is_set() or (t.deadline is not None and time.monotonic() > t.deadline)):
        raise TurnCancelled("This request took longer than allowed and was stopped before it could finish.")


# ---------------------------------------------------------------------------
# The one place a model is called
# ---------------------------------------------------------------------------
def invoke(client, messages, *, role: str = "model"):
    """``client.invoke(messages)`` with classification, the breaker and cancellation.

    Raises ``ModelUnavailable`` for a transient service failure (or an open breaker), ``TurnCancelled`` past the deadline,
    and re-raises anything else (a bad request is not an outage) exactly as before.
    """
    check_cancelled()
    gate = breaker()
    gate.before_call()
    try:
        result = client.invoke(messages)
    except Exception as exc:
        if is_transient(exc):
            gate.record_failure()
            raise ModelUnavailable(f"The language model service is not responding ({role}: {type(exc).__name__}).") from exc
        gate.record_success()                           # the service answered (with an error about the request): it is up
        raise
    gate.record_success()
    return result
