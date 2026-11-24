"""The ONE finite vocabulary for "what happened to this step".

Why: when a request could not be answered, the evidence the answer model reads used to be a free-text sentence such as
``Code Agent error: Cannot compile SQL: plan selected no tables.``  The model summarised that as "the database query
encountered an error" - wrong, because nothing was ever sent to the database.  Every layer that has to react to a
non-answer (the tool that produces it, the orchestrator deciding whether to retry, the validator guarding against
invented numbers, the eval harness, telemetry) needs the same reliable signal, so it is one tiny shared format instead
of ad-hoc substring checks:

    NOT_ANSWERED[<code>]: <message that is safe to show the user>

The set below is CLOSED and EXHAUSTIVE: every failure path in the application maps onto one of these codes, and
``INTERNAL_ERROR`` exists precisely so that an unanticipated exception is still classifiable rather than unnamed.
``agents/plan_status.py`` uses the planning codes; ``agents/tools.py`` uses the SQL and routing codes; the agent
orchestrators use the safety, privacy, clarification and limit codes.  ``CATEGORY`` groups them for telemetry and for
dashboards that should not enumerate codes.

Codes
-----
Success is not a code: a typed ``agents/tool_result.ToolResult`` says ``ok=True`` and carries ``OK`` in ``status``.

planning
  unplannable         No table/measure matched the request, so no SQL was generated.                 (retryable)
  plan_incomplete     A table matched but the plan has no operation, so nothing can be selected.
  plan_inconsistent   The plan projects a table it does not include (the compiler would emit invalid SQL).
  plan_unconnected    The planned tables are not connected by catalog relationships (a cross product).
place
  place_missing       A derived measure needs a named place and the request has none.                (retryable)
  place_ambiguous     A place name fits several places; the user must choose.
data
  topic_unavailable   The topic is known NOT to be loaded (a ``gap`` concept); nothing may be estimated.
  data_unavailable    The data needed exists in the model but is missing/incomplete in the database.
  result_empty        A valid query executed and matched no rows (a factual zero, not an error).
sql
  sql_invalid         The statement was rejected by the contract/validator, so it never ran.          (retryable)
  sql_execution_error The database rejected or failed the statement (binder/catalog/runtime).         (retryable)
routing
  route_not_found     No continuous route exists on the loaded network.
  route_no_data       The routing dataset itself is not loaded.
  route_input         A route endpoint/parameter could not be resolved.
  route_unavailable   The routing/geocoding service failed (transient, external).
limits
  limit_exceeded      The request is larger than one call supports (too many items, too broad, timed out).
  tool_timeout        One approved call exceeded its own deadline.
  invalid_argument    The model called an approved function with an argument it does not accept.
model
  model_unavailable   The language model service is not responding and this step needs it.
safety / privacy
  refused_unsafe      The request asked for something outside the read-only, non-overridable contract.
  refused_pii         The request asked for personal fields the catalogue does not hold.
clarification
  needs_clarification The assistant asked the user a question instead of guessing.
internal
  internal_error      An unanticipated failure. Always classifiable, never silently renamed.

*Retryable* means a differently-worded request - or a repaired program/SQL statement - could succeed inside the same
turn; the others need the user (or an operator) and retrying would only repeat the same answer.

Messages must contain no digits: the validator's "does this output contain real data?" heuristic is digit-based, and a
guidance sentence must never be mistaken for data (tests enforce this for every template).
"""

from __future__ import annotations

import re

# ── Success ──────────────────────────────────────────────────────────────────
OK = "ok"

# ── Planning ─────────────────────────────────────────────────────────────────
UNPLANNABLE = "unplannable"
PLAN_INCOMPLETE = "plan_incomplete"
PLAN_INCONSISTENT = "plan_inconsistent"
PLAN_UNCONNECTED = "plan_unconnected"

# ── Places ───────────────────────────────────────────────────────────────────
PLACE_MISSING = "place_missing"
PLACE_AMBIGUOUS = "place_ambiguous"

# ── Data ─────────────────────────────────────────────────────────────────────
TOPIC_UNAVAILABLE = "topic_unavailable"
DATA_UNAVAILABLE = "data_unavailable"
RESULT_EMPTY = "result_empty"

# ── SQL ──────────────────────────────────────────────────────────────────────
SQL_INVALID = "sql_invalid"
SQL_EXECUTION_ERROR = "sql_execution_error"

# ── Routing ──────────────────────────────────────────────────────────────────
ROUTE_NOT_FOUND = "route_not_found"
ROUTE_NO_DATA = "route_no_data"
ROUTE_INPUT = "route_input"
ROUTE_UNAVAILABLE = "route_unavailable"

# ── Limits ───────────────────────────────────────────────────────────────────
LIMIT_EXCEEDED = "limit_exceeded"
TOOL_TIMEOUT = "tool_timeout"
INVALID_ARGUMENT = "invalid_argument"

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_UNAVAILABLE = "model_unavailable"

# ── Safety / privacy ─────────────────────────────────────────────────────────
REFUSED_UNSAFE = "refused_unsafe"
REFUSED_PII = "refused_pii"

# ── Clarification ────────────────────────────────────────────────────────────
NEEDS_CLARIFICATION = "needs_clarification"

# ── Internal ─────────────────────────────────────────────────────────────────
INTERNAL_ERROR = "internal_error"

CODES = frozenset(
    {
        UNPLANNABLE,
        PLAN_INCOMPLETE,
        PLAN_INCONSISTENT,
        PLAN_UNCONNECTED,
        PLACE_MISSING,
        PLACE_AMBIGUOUS,
        TOPIC_UNAVAILABLE,
        DATA_UNAVAILABLE,
        RESULT_EMPTY,
        SQL_INVALID,
        SQL_EXECUTION_ERROR,
        ROUTE_NOT_FOUND,
        ROUTE_NO_DATA,
        ROUTE_INPUT,
        ROUTE_UNAVAILABLE,
        LIMIT_EXCEEDED,
        TOOL_TIMEOUT,
        INVALID_ARGUMENT,
        MODEL_UNAVAILABLE,
        REFUSED_UNSAFE,
        REFUSED_PII,
        NEEDS_CLARIFICATION,
        INTERNAL_ERROR,
    }
)

#: Grouping for telemetry/dashboards: every code belongs to exactly one category (tests enforce the partition).
CATEGORY = {
    UNPLANNABLE: "planning",
    PLAN_INCOMPLETE: "planning",
    PLAN_INCONSISTENT: "planning",
    PLAN_UNCONNECTED: "planning",
    PLACE_MISSING: "place",
    PLACE_AMBIGUOUS: "place",
    TOPIC_UNAVAILABLE: "data",
    DATA_UNAVAILABLE: "data",
    RESULT_EMPTY: "data",
    SQL_INVALID: "sql",
    SQL_EXECUTION_ERROR: "sql",
    ROUTE_NOT_FOUND: "routing",
    ROUTE_NO_DATA: "routing",
    ROUTE_INPUT: "routing",
    ROUTE_UNAVAILABLE: "routing",
    LIMIT_EXCEEDED: "limits",
    TOOL_TIMEOUT: "limits",
    INVALID_ARGUMENT: "limits",
    MODEL_UNAVAILABLE: "model",
    REFUSED_UNSAFE: "safety",
    REFUSED_PII: "privacy",
    NEEDS_CLARIFICATION: "clarification",
    INTERNAL_ERROR: "internal",
}

#: Short labels for traces/logs, so a dashboard never has to render raw codes.
LABELS = {
    UNPLANNABLE: "request matched no measure",
    PLAN_INCOMPLETE: "plan has no operation",
    PLAN_INCONSISTENT: "plan projects an unplanned table",
    PLAN_UNCONNECTED: "planned tables are not joinable",
    PLACE_MISSING: "no place named",
    PLACE_AMBIGUOUS: "place is ambiguous",
    TOPIC_UNAVAILABLE: "topic not loaded",
    DATA_UNAVAILABLE: "data missing",
    RESULT_EMPTY: "no matching rows",
    SQL_INVALID: "statement rejected",
    SQL_EXECUTION_ERROR: "statement failed",
    ROUTE_NOT_FOUND: "no route",
    ROUTE_NO_DATA: "routing data not loaded",
    ROUTE_INPUT: "route endpoint unresolved",
    ROUTE_UNAVAILABLE: "routing service failed",
    LIMIT_EXCEEDED: "request too large",
    TOOL_TIMEOUT: "call timed out",
    INVALID_ARGUMENT: "invalid argument",
    MODEL_UNAVAILABLE: "model unavailable",
    REFUSED_UNSAFE: "refused: unsafe instruction",
    REFUSED_PII: "refused: personal fields",
    NEEDS_CLARIFICATION: "asked the user",
    INTERNAL_ERROR: "internal error",
}

#: Codes a retry inside the same turn can plausibly fix.  Everything else needs the user or an operator.
RETRYABLE = frozenset({UNPLANNABLE, PLACE_MISSING, SQL_INVALID, SQL_EXECUTION_ERROR, RESULT_EMPTY})

#: User-safe sentence for a code whose technical detail must never reach the user verbatim.  Digit-free by construction.
LIMITATION_TEXT = {
    PLAN_INCOMPLETE: (
        "This question does not match a way of querying the loaded data that this assistant supports, so no query was "
        "run and nothing was estimated."
    ),
    PLAN_INCONSISTENT: (
        "This question needs columns from two datasets that the query plan could not combine, so no query was run and "
        "nothing was estimated."
    ),
    PLAN_UNCONNECTED: (
        "This question needs datasets that the catalog does not relate to each other, so no query was run and nothing "
        "was estimated."
    ),
    SQL_INVALID: "The query was rejected by the read-only data contract before it ran. Nothing was estimated.",
    SQL_EXECUTION_ERROR: (
        "The database rejected the generated query, so no result was produced. Nothing was estimated."
    ),
    INTERNAL_ERROR: "Something went wrong while answering, so no result was produced. Nothing was estimated.",
}

# Anchored to the start of the text or to the "[RESULT]" header that query_database puts in front of results, so a
# data value that happens to contain the words can never be read as a status.
_STATUS = re.compile(r"(?:\A|\[RESULT\][ \t]*\n)[ \t]*NOT_ANSWERED\[([a-z_]+)\]:[ \t]*(.*)", re.S)


def not_answered(code: str, message: str) -> str:
    if code not in CODES:
        raise ValueError(f"unknown answer status code: {code!r}")
    return f"NOT_ANSWERED[{code}]: {message.strip()}"


def limitation(code: str, detail: str = "") -> str:
    """A user-safe status line: the code's plain sentence plus (optionally) the technical detail in parentheses."""
    sentence = LIMITATION_TEXT.get(code) or LABELS.get(code) or "This request could not be answered."
    detail = (detail or "").strip()
    return not_answered(code, f"{sentence} ({detail})" if detail else sentence)


def status_code(text) -> str | None:
    if not isinstance(text, str) or "NOT_ANSWERED[" not in text:
        return None
    m = _STATUS.search(text)
    return m.group(1) if m else None


def status_message(text) -> str:
    """The user-safe message of a status line ('' when ``text`` is not one)."""
    if not isinstance(text, str):
        return ""
    m = _STATUS.search(text)
    return m.group(2).strip() if m else ""


def is_not_answered(text) -> bool:
    return status_code(text) is not None


def is_retryable(text) -> bool:
    return status_code(text) in RETRYABLE


def category(code: str | None) -> str:
    return CATEGORY.get(code or OK, "ok" if code in (None, OK) else "internal")


def label(code: str | None) -> str:
    return LABELS.get(code or OK, "ok" if code in (None, OK) else "internal error")


def outcome(code: str | None) -> dict:
    """The typed summary an orchestrator/API/eval harness returns for a step (``status`` is ``ok`` on success)."""
    code = code or OK
    return {"status": code, "category": category(code), "label": label(code), "retryable": code in RETRYABLE}


# ── Classification of everything that is not already a status line ───────────
#: Legacy/third-party text that predates the vocabulary (or arrives from a library) maps onto it here, once, so no
#: caller has to substring-match prose again.  Order matters: the first matching entry wins.
_TEXT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (SQL_EXECUTION_ERROR, ("sql error", "binder error", "binderexception", "catalog error", "parser error")),
    (RESULT_EMPTY, ("query returned 0 rows", "no matching records", "0 rows")),
    (MODEL_UNAVAILABLE, ("model is unavailable", "model unavailable", "not responding", "connection refused")),
    (TOOL_TIMEOUT, ("exceeded its", "timed out", "timeout")),
    (INVALID_ARGUMENT, ("scope must be", "unexpected keyword", "invalid argument", "too many positional")),
    (SQL_INVALID, ("only read-only select", "must begin with select", "unknown table", "not allowed")),
    (PLAN_INCONSISTENT, ("not connected by catalog relationships", "no catalog join path")),
    (PLAN_INCOMPLETE, ("plan has no operation", "plan selected no tables")),
)


def classify_text(text) -> str | None:
    """The code for a tool result: its own marker first, then a single table of legacy phrasings, else None.

    Returns ``None`` when the text is ordinary (evidence or a plain string from a tool that speaks no vocabulary).
    """
    code = status_code(text)
    if code:
        return code
    if not isinstance(text, str) or not text.strip():
        return None
    lowered = text.lower()
    if lowered.startswith("code agent error") or lowered.startswith("error:"):
        for candidate, markers in _TEXT_RULES:
            if any(m in lowered for m in markers):
                return candidate
        return INTERNAL_ERROR
    for candidate, markers in _TEXT_RULES:
        if any(m in lowered for m in markers):
            return candidate
    return None


def classify_error(exc: BaseException) -> str:
    """The code for an exception, so an unexpected failure is still named instead of becoming prose."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code in CODES:
        return code
    try:
        from agents import llm_resilience

        if isinstance(exc, llm_resilience.ModelUnavailable):
            return MODEL_UNAVAILABLE
        if isinstance(exc, llm_resilience.TurnCancelled):
            return LIMIT_EXCEEDED
    except Exception:
        pass
    name = type(exc).__name__
    text = f"{name}: {exc}".lower()
    if "timeout" in text or "timed out" in text:
        return TOOL_TIMEOUT
    for candidate, markers in _TEXT_RULES:
        if any(m in text for m in markers):
            return candidate
    if isinstance(exc, ValueError):
        return SQL_INVALID
    return INTERNAL_ERROR
