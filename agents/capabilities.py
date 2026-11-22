"""What an approved function is allowed to touch, and for how long.

The general agent never runs model-written SQL: it runs a small program over a
fixed vocabulary of application functions (``agents/general_agent.py``). That
vocabulary is safe by construction — no arbitrary I/O — but a few entries reach
well beyond the local database: two geocode and route over public servers, one
queries the local vector store, and ``query_database`` waits on a model server.
A model that has been injected, or merely confused, can name any of them, and
"wait for an unreachable routing server" is a real way to hold a turn open.

This module makes those properties explicit and enforceable:

* every approved function is registered with **tags** describing what it can
  touch (``database``, ``network``, ``routing``, ``model``, ``local``) and a
  per-call **deadline**;
* a turn opens a **budget** (``start_turn_budget``) and network-touching calls
  are given ``min(their deadline, what is left)``;
* a call with no budget left raises ``CapabilityTimeout`` instead of starting,
  and a call that overruns is cut off by the transport layer
  (``agents/tools.py::_run_async_safely`` uses ``asyncio.wait_for``), so the
  deadline is a bound on the wait, not a note in a docstring.

The registry is also the source of the capability table in the docs
(``scripts/gen_docs_tables.py``), so the documentation cannot drift from the
functions the agent can actually call.
"""

from __future__ import annotations

import contextvars
import functools
import time
from dataclasses import dataclass
from typing import Callable

from config import settings

#: Tags used by the registry. ``network`` is the one that costs money and time.
NETWORK = "network"
DATABASE = "database"
ROUTING = "routing"
MODEL = "model"
LOCAL = "local"
METADATA = "metadata"

#: Turn-wide budget for approved function calls. A single call may not exceed
#: its own deadline; the turn budget stops five "cheap" calls adding up.
DEFAULT_TURN_BUDGET_S = 180.0


@dataclass(frozen=True)
class Capability:
    tags: tuple[str, ...]
    deadline_s: float
    description: str = ""

    @property
    def touches_network(self) -> bool:
        return NETWORK in self.tags


class CapabilityTimeout(RuntimeError):
    """An approved function was refused or cut off because its deadline passed."""

    def __init__(self, name: str, deadline_s: float):
        super().__init__(f"Approved function '{name}' exceeded its {deadline_s:g}s deadline.")
        self.name = name
        self.deadline_s = deadline_s


CAPABILITIES: dict[str, Capability] = {
    "query_database": Capability(
        (DATABASE, MODEL, "analytics"),
        180.0,
        "Plan, generate and run one read-only SQL statement.",
    ),
    "retrieve_data_model_context": Capability(
        (DATABASE, METADATA, LOCAL),
        30.0,
        "Plan plus targeted schema and vector metadata for one question.",
    ),
    "check_data_availability": Capability(
        (DATABASE,),
        15.0,
        "Row counts for every agent-visible table.",
    ),
    "get_database_schema": Capability(
        (DATABASE, METADATA),
        15.0,
        "Human-readable schema summary.",
    ),
    "find_bike_route": Capability(
        (NETWORK, ROUTING),
        60.0,
        "Bicycle route between two places (geocoding + local graph).",
    ),
    "find_transit_route": Capability(
        (NETWORK, ROUTING),
        60.0,
        "Transit route between two places (geocoding + transit graph).",
    ),
    "search_all_house_descriptions": Capability(
        (LOCAL, "retrieval"),
        30.0,
        "Vector search over saved house descriptions.",
    ),
}


def spec(name: str) -> Capability | None:
    return CAPABILITIES.get(name)


def capability_names() -> list[str]:
    return sorted(CAPABILITIES)


def describe() -> list[dict]:
    """Registry as rows for the generated capability table in the docs."""
    return [
        {
            "name": name,
            "tags": ", ".join(cap.tags),
            "deadline": f"{cap.deadline_s:g}s",
            "description": cap.description,
        }
        for name, cap in sorted(CAPABILITIES.items())
    ]


# ── Per-turn budget ──────────────────────────────────────────────────────────

_TURN_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "approved_function_turn_deadline", default=None
)

#: Last observed call durations, newest last. Bounded so a long-lived process
#: cannot grow it; used by tests and by /api/stats-adjacent diagnostics.
LAST_DURATIONS: list[tuple[str, float]] = []
_MAX_DURATIONS = 100


def start_turn_budget(seconds: float | None = None) -> contextvars.Token:
    """Open a budget for the current turn; returns a token for ``end_turn_budget``."""
    budget = DEFAULT_TURN_BUDGET_S if seconds is None else float(seconds)
    return _TURN_DEADLINE.set(time.monotonic() + max(budget, 0.0))


def end_turn_budget(token: contextvars.Token) -> None:
    try:
        _TURN_DEADLINE.reset(token)
    except ValueError:  # token from another context; nothing to restore
        _TURN_DEADLINE.set(None)


def remaining_budget() -> float | None:
    """Seconds left in this turn's budget, or ``None`` when no budget is open."""
    deadline = _TURN_DEADLINE.get()
    if deadline is None:
        return None
    return deadline - time.monotonic()


def effective_deadline(name: str, default: float | None = None) -> float:
    """How long this call may take: its own deadline, capped by the turn budget."""
    cap = spec(name)
    own = cap.deadline_s if cap else (default if default is not None else DEFAULT_TURN_BUDGET_S)
    left = remaining_budget()
    if left is None:
        return float(own)
    return max(min(float(own), left), 0.0)


def record(name: str, seconds: float) -> None:
    LAST_DURATIONS.append((name, round(seconds, 4)))
    del LAST_DURATIONS[:-_MAX_DURATIONS]


def capability(name: str) -> Callable:
    """Tag and bound one approved function.

    The wrapper refuses to start when the turn's budget is already exhausted and
    records every duration. It does not swallow exceptions: a tool that fails
    still reports its own structured failure.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            deadline = effective_deadline(name)
            if deadline <= 0:
                raise CapabilityTimeout(name, 0.0)
            started = time.monotonic()
            try:
                return fn(*args, **kwargs)
            finally:
                record(name, time.monotonic() - started)

        wrapper.capability = name  # type: ignore[attr-defined]
        return wrapper

    return decorator


def settings_budget() -> float:
    """The configured turn budget, if the app exposes one."""
    return float(getattr(settings, "approved_function_budget_s", DEFAULT_TURN_BUDGET_S))
