"""Mandatory query scope: structural, not prose.

House Chat used to scope its SQL by appending a sentence to the request text ("Restrict house-specific results to this
house").  A model is free to ignore a sentence, and nothing verified that it did not.  A scope is now a small immutable
value held in a ``ContextVar``:

* it is set by CODE around the call (never by the model), so the model can neither see nor alter nor drop it;
* the planner turns it into a ``scope_filters`` line of the plan and forces the scope table into the planned tables;
* the deterministic SQL compiler adds it to the WHERE clause;
* ``agents/plan_contract.py`` verifies the statement contains it as a POSITIVE filter (not negated, not commented out,
  not voided by an OR) - so a statement that drops the scope is rejected and repaired, not executed.

A ContextVar (rather than a global) means concurrent turns on different threads never see each other's scope, and a
scope set before ``asyncio.to_thread`` follows the work onto the worker thread.

A scope can be waived only explicitly, by the caller choosing not to set one (House Chat: ``scope="all_houses"``).
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ScopeFilter:
    table: str
    column: str
    value: str

    def __post_init__(self):
        if not (_IDENT.match(self.table) and _IDENT.match(self.column)):
            raise ValueError(f"invalid scope target {self.table!r}.{self.column!r}")
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("a scope value must be a non-empty string")

    def render(self) -> str:
        return f"{self.table.lower()}.{self.column.lower()} = '{self.value.replace(chr(39), chr(39) * 2)}'"


_CURRENT: ContextVar[tuple] = ContextVar("query_scope", default=())


def current() -> tuple[ScopeFilter, ...]:
    return _CURRENT.get()


@contextmanager
def scoped(*filters: ScopeFilter):
    """Everything planned or validated inside the block must honour ``filters``."""
    token = _CURRENT.set(tuple(filters))
    try:
        yield
    finally:
        _CURRENT.reset(token)
