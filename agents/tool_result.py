"""Typed envelope for what an approved function returns to the model.

Why this exists
---------------
``query_database`` and the route tools render a string for the model to read,
and everything downstream used to guess what that string *meant*: the response
validator looked for digit-bearing lines to decide "that was a real result",
error handling looked for ``SQL Error`` substrings, and a 0-row answer had to be
recognised by its prefix. Those guesses are exactly the kind of thing that
breaks when a message is reworded.

``ToolResult`` keeps the rendered text (it is a ``str`` subclass, so prompts,
``str(result)``, f-strings, JSON dumps and every existing caller keep working)
and carries the facts the application already knows:

    text        what the model reads
    kind        data | route | schema | status | not_answered | error
    ok          did the function do what it was asked
    status      the NOT_ANSWERED code, when kind == "not_answered"
    row_count   rows returned, when the caller knows them
    columns     column names, when the caller knows them

The validator prefers these facts over its text heuristics, and falls back to
the heuristics only for results that arrive as a plain string (e.g. a tool the
application does not own).
"""

from __future__ import annotations

from typing import Any, Iterable

KINDS = ("data", "route", "schema", "status", "not_answered", "error")

#: Kinds that are never evidence of query results, whatever digits they hold.
NON_EVIDENCE_KINDS = ("schema", "status", "not_answered", "error")


class ToolResult(str):
    """A tool's rendered text plus the facts about it (see module docstring)."""

    __slots__ = ("ok", "kind", "status", "row_count", "columns")

    # Declared for type checkers: __slots__ creates the storage, not the annotations.
    ok: bool
    kind: str
    status: str | None
    row_count: int | None
    columns: tuple[str, ...]

    def __new__(
        cls,
        text: str,
        *,
        ok: bool = True,
        kind: str = "data",
        status: str | None = None,
        row_count: int | None = None,
        columns: Iterable[str] = (),
    ) -> "ToolResult":
        obj = super().__new__(cls, text)
        obj.ok = bool(ok)
        obj.kind = kind if kind in KINDS else "data"
        obj.status = status
        obj.row_count = row_count
        obj.columns = tuple(columns)
        return obj

    def facts(self) -> dict[str, Any]:
        """The metadata as plain data (for trace payloads and tests)."""
        return {
            "ok": self.ok,
            "kind": self.kind,
            "status": self.status,
            "row_count": self.row_count,
            "columns": list(self.columns),
        }

    def is_evidence(self) -> bool:
        """True when this result is something a grounded answer may cite."""
        return self.ok and self.kind not in NON_EVIDENCE_KINDS

    def __repr__(self) -> str:  # keep reprs readable in logs
        head = str(self)[:40].replace("\n", " ")
        return f"ToolResult({head!r}, kind={self.kind!r}, ok={self.ok})"


def facts(value: Any) -> dict[str, Any] | None:
    """Metadata for ``value`` when it is a ToolResult, else ``None``."""
    return value.facts() if isinstance(value, ToolResult) else None


def as_result(value: Any, *, kind: str = "data") -> ToolResult:
    """Coerce a legacy string return into the envelope without changing its text."""
    if isinstance(value, ToolResult):
        return value
    return ToolResult(str(value), kind=kind)


def not_answered(text: str, status: str) -> ToolResult:
    return ToolResult(text, ok=False, kind="not_answered", status=status)


def failure(text: str, status: str | None = None) -> ToolResult:
    """A call that did not do what it was asked, named by ``status`` when the caller knows the code."""
    return ToolResult(text, ok=False, kind="error", status=status)


def status_result(text: str, status: str, *, ok: bool = True, **kwargs) -> ToolResult:
    """A non-evidence report (a zero-row answer, a refusal, a diagnostic) carrying its code."""
    return ToolResult(text, ok=ok, kind="status", status=status, **kwargs)


def result_failed(value: Any) -> bool:
    """Did this tool result fail in a way a second step could plausibly fix?

    The single retry predicate shared by both orchestrators (``general_agent`` and ``house_agent``).  Typed results
    answer it from their own facts; plain strings fall back to the shared vocabulary in ``agents/answer_status.py``,
    so neither agent keeps its own list of substrings to look for.
    """
    from agents import answer_status

    data = facts(value)
    if data is not None:
        if data["ok"] and data["kind"] not in NON_EVIDENCE_KINDS:
            return False
        code = data.get("status")
        return bool(code) and code in answer_status.RETRYABLE
    code = answer_status.classify_text(value)
    return bool(code) and code in answer_status.RETRYABLE
