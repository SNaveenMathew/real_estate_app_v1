"""Machine-readable "this question was NOT answered" markers.

Why: when a request could not be answered, the evidence the answer model reads used to be a free-text sentence such as
``Code Agent error: Cannot compile SQL: plan selected no tables.``  The model summarised that as "the database query
encountered an error" - wrong, because nothing was ever sent to the database.  Every layer that has to react to a
non-answer (the tool that produces it, the orchestrator deciding whether to retry, the validator guarding against
invented numbers) needs the same reliable signal, so it is one tiny shared format instead of ad-hoc substring checks:

    NOT_ANSWERED[<code>]: <message that is safe to show the user>

Codes
-----
unplannable       The request matched no table or measure in the data model, so no SQL was generated.   (retryable)
place_missing     A derived measure needs a named place and the request has none.                        (retryable)
place_ambiguous   A place name fits several places; the user must choose.
topic_unavailable The topic is known NOT to be loaded (a ``gap`` concept); nothing may be estimated.
data_unavailable  The data needed exists in the model but is missing/incomplete in the database.
limit_exceeded    The request is larger than one call supports.

*Retryable* means a differently-worded request from the orchestrator could succeed; the others need the user (or an
operator) and retrying would only repeat the same answer.

Messages must contain no digits: the validator's "does this output contain real data?" heuristic is digit-based, and a
guidance sentence must never be mistaken for data (tests enforce this for every template).
"""
from __future__ import annotations

import re

UNPLANNABLE = "unplannable"
PLACE_MISSING = "place_missing"
PLACE_AMBIGUOUS = "place_ambiguous"
TOPIC_UNAVAILABLE = "topic_unavailable"
DATA_UNAVAILABLE = "data_unavailable"
LIMIT_EXCEEDED = "limit_exceeded"

CODES = frozenset({UNPLANNABLE, PLACE_MISSING, PLACE_AMBIGUOUS, TOPIC_UNAVAILABLE, DATA_UNAVAILABLE, LIMIT_EXCEEDED})
RETRYABLE = frozenset({UNPLANNABLE, PLACE_MISSING})

# Anchored to the start of the text or to the "[RESULT]" header that query_database puts in front of results, so a
# data value that happens to contain the words can never be read as a status.
_STATUS = re.compile(r"(?:\A|\[RESULT\][ \t]*\n)[ \t]*NOT_ANSWERED\[([a-z_]+)\]:[ \t]*(.*)", re.S)


def not_answered(code: str, message: str) -> str:
    if code not in CODES:
        raise ValueError(f"unknown answer status code: {code!r}")
    return f"NOT_ANSWERED[{code}]: {message.strip()}"


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
