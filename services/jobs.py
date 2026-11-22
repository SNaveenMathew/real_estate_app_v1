"""Background jobs for work that must outlive one HTTP request.

The onboarding ingest, an enrichment pass, an analysis sweep and a data-source
reload all take seconds to minutes. Running them inside the request held a
connection open for the whole time (and, before the threading change, froze the
event loop); a client that navigated away left the work running with nobody
watching it, and the browser had no way to say how far along it was.

This module is the smallest thing that fixes that, modelled on the commute job
the app already had: ``start()`` claims a slot and returns immediately, a worker
thread runs the callable, and ``GET /api/jobs/{id}`` returns a snapshot the page
polls. Jobs are in memory on purpose - they describe work in this process, not
durable state; the durable record is the catalog transaction the work itself
commits.

One job per kind at a time. The catalog is a single shared database and its
mutations serialize in the engine; letting two ingest jobs interleave their
profiling and DDL would only produce confusing half-states, so a second request
for a busy kind is answered with ``JobBusy`` (HTTP 409) instead.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

_MAX_JOBS = 50          # snapshots kept for polling; older finished ones are dropped


class JobBusy(RuntimeError):
    """A job of the same kind is already running."""

    def __init__(self, job: "Job"):
        self.job = job
        super().__init__(f"{job.kind} is already running ({job.label})")


@dataclass
class Job:
    id: str
    kind: str
    label: str
    state: str = "queued"                  # queued | running | done | error
    message: str = ""
    done: int | None = None
    total: int | None = None
    error: str = ""
    result: Any = None
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ── updates, called from the worker thread ──────────────────────────────
    def progress(self, message: str | None = None, *, done: int | None = None,
                 total: int | None = None) -> None:
        with self._lock:
            if message is not None:
                self.message = message
            if done is not None:
                self.done = done
            if total is not None:
                self.total = total

    def finish(self, result: Any = None, message: str = "") -> None:
        with self._lock:
            self.state, self.result, self.finished_at = "done", result, time.time()
            self.message = message or "Done."

    def fail(self, error: str) -> None:
        with self._lock:
            self.state, self.error, self.finished_at = "error", error, time.time()
            self.message = error

    # ── read model ──────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            elapsed = (self.finished_at or time.time()) - (self.started_at or self.created_at)
            return {
                "job_id": self.id,
                "kind": self.kind,
                "label": self.label,
                "state": self.state,
                "message": self.message,
                "done": self.done,
                "total": self.total,
                "error": self.error,
                "result": self.result,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_s": round(elapsed, 3),
                "running": self.state in ("queued", "running"),
            }


_JOBS: dict[str, Job] = {}
_ORDER: list[str] = []
_ACTIVE: dict[str, str] = {}         # kind -> job id currently running
_LOCK = threading.RLock()


def start(kind: str, label: str, fn: Callable[[Job], Any]) -> Job:
    """Claim ``kind`` and run ``fn(job)`` on a worker thread.

    ``fn`` receives the job so it can report progress; whatever it returns
    becomes ``snapshot()["result"]``. Any exception becomes ``state == "error"``
    with the message - a background failure must not vanish into a log line.
    """
    with _LOCK:
        running_id = _ACTIVE.get(kind)
        running = _JOBS.get(running_id) if running_id else None
        if running is not None and running.state in ("queued", "running"):
            raise JobBusy(running)
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label)
        _JOBS[job.id] = job
        _ORDER.append(job.id)
        _ACTIVE[kind] = job.id
        while len(_ORDER) > _MAX_JOBS:
            stale = _ORDER.pop(0)
            stale_job = _JOBS.get(stale)
            if stale_job is not None and stale_job.state not in ("queued", "running"):
                _JOBS.pop(stale, None)
            else:
                _ORDER.append(stale)
                break

    def runner() -> None:
        job.started_at = time.time()
        job.state = "running"
        try:
            result = fn(job)
        except Exception as exc:                       # noqa: BLE001 - reported to the caller
            job.fail(_describe(exc))
        else:
            job.finish(result)
        finally:
            with _LOCK:
                if _ACTIVE.get(kind) == job.id:
                    _ACTIVE.pop(kind, None)

    threading.Thread(target=runner, name=f"job-{kind}-{job.id}", daemon=True).start()
    return job


def get(job_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(job_id)


def recent(limit: int = 20) -> list[dict]:
    with _LOCK:
        ids = list(reversed(_ORDER))[:max(1, limit)]
        return [_JOBS[i].snapshot() for i in ids if i in _JOBS]


def wait(job_id: str, timeout: float = 300.0, poll: float = 0.02) -> Job:
    """Block until the job leaves the running states. For scripts and tests only."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.state not in ("queued", "running"):
            return job
        time.sleep(poll)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def _describe(exc: BaseException) -> str:
    from fastapi import HTTPException
    if isinstance(exc, HTTPException):
        detail = exc.detail
        return detail if isinstance(detail, str) else str(detail)
    return f"{type(exc).__name__}: {exc}"
