"""Run one agent turn WITHOUT freezing the server - bounded, cancellable, and honest about why it failed.

Before: ``/api/chat`` and ``/api/house/{id}/chat`` were ``async def`` handlers that called the blocking agent directly,
so for the whole turn (several model calls; tens of seconds on a local model) the event loop was blocked: the houses
list, the map, commute polling, the Data page and ``/metrics`` all waited.  There was no deadline, no limit on how many
turns could pile up, and a turn could not be stopped.

Now a turn runs on a worker thread:

* **bounded concurrency** - ``agent_max_concurrent_turns`` turns run at once (a local model server cannot parallelize much);
  others wait up to ``agent_queue_timeout_s`` for a slot, then get "busy" (HTTP 503) instead of piling up forever;
* **deadline** - past ``agent_turn_deadline_s`` the caller gets HTTP 504 and the turn is *cancelled cooperatively*: it
  starts no further model call (``agents/llm_resilience.check_cancelled``).  A thread cannot be killed, so the slot is
  released only when the thread really finishes - a stuck model call cannot silently exceed the concurrency limit;
* **client disconnect** - a cancelled request cancels its turn the same way;
* **context** - ContextVars (the query scope, the artifact collector) follow the work onto the worker thread;
* **meaningful failures** - ``describe_failure`` maps busy / deadline / model-unavailable to 503 / 504 / 503 with a message
  a user can act on, instead of ``HTTP 500: Agent error: <raw exception text>``.

Data access from the worker is safe because ``db/duckdb_store.get_conn()`` is per-thread.
"""
from __future__ import annotations

import asyncio
import contextvars
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

from agents import llm_resilience
from config import settings


class Busy(RuntimeError):
    """No turn slot became free within ``agent_queue_timeout_s``."""


class DeadlineExceeded(RuntimeError):
    """The turn ran longer than ``agent_turn_deadline_s`` and was cancelled."""


_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()
_SEMAPHORES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()      # one per event loop (tests create several)
_state = {"running": 0, "waiting": 0}


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _EXECUTOR is None:                       # dedicated: a hung turn can never starve other to_thread users
                _EXECUTOR = ThreadPoolExecutor(max_workers=max(4, int(settings.agent_max_concurrent_turns) * 2),
                                               thread_name_prefix="agent-turn")
    return _EXECUTOR


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _SEMAPHORES.get(loop)
    if sem is None:
        sem = _SEMAPHORES[loop] = asyncio.Semaphore(max(1, int(settings.agent_max_concurrent_turns)))
    return sem


def status() -> dict:
    return {"running": _state["running"], "waiting": _state["waiting"], "max_concurrent": int(settings.agent_max_concurrent_turns)}


async def run_turn(fn, *args, **kwargs):
    """``fn(*args, **kwargs)`` on a worker thread, under the limits above.  Returns its result or raises its exception."""
    sem = _semaphore()
    _state["waiting"] += 1
    try:
        await asyncio.wait_for(sem.acquire(), timeout=float(settings.agent_queue_timeout_s))
    except asyncio.TimeoutError:
        raise Busy("The assistant is busy with other requests.") from None
    finally:
        _state["waiting"] -= 1

    deadline_s = float(settings.agent_turn_deadline_s)
    handle = llm_resilience.Turn(deadline=time.monotonic() + deadline_s if deadline_s > 0 else None)
    ctx = contextvars.copy_context()                    # scope / artifacts / tracing follow the work onto the thread

    def work():
        with llm_resilience.use_turn(handle):
            return fn(*args, **kwargs)

    loop = asyncio.get_running_loop()
    _state["running"] += 1
    try:
        future = loop.run_in_executor(_executor(), lambda: ctx.run(work))
    except BaseException:
        _state["running"] -= 1
        sem.release()
        raise

    def finished(_f):                                   # runs on the event loop when the THREAD ends
        _state["running"] -= 1
        sem.release()

    future.add_done_callback(finished)
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout=deadline_s if deadline_s > 0 else None)
    except asyncio.TimeoutError:
        handle.cancel.set()                             # the worker starts no further model call
        raise DeadlineExceeded(f"The request took longer than {deadline_s:g} seconds and was stopped.") from None
    except asyncio.CancelledError:                      # the client went away
        handle.cancel.set()
        raise


def describe_failure(exc: BaseException) -> tuple[int, str, dict]:
    """``(http status, user-facing message, extra headers)`` for a failure of a turn."""
    if isinstance(exc, Busy):
        return 503, "The assistant is busy with other requests. Please try again in a moment.", {"Retry-After": "5"}
    if isinstance(exc, (DeadlineExceeded, llm_resilience.TurnCancelled)):
        return 504, "The request took too long and was stopped. Try a narrower question.", {}
    if isinstance(exc, llm_resilience.ModelUnavailable):
        return 503, "The language model service is not responding. Please try again in a minute.", {"Retry-After": "30"}
    return 500, f"Agent error: {exc}", {}
