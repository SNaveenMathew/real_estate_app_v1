"""Answer requests for personal fields from the catalogue: this database does not hold them.

The house and sold-home tables are property records - address, price, status, scores - never seller names, phone
numbers, email addresses or other personal identifiers.  A request for those fields cannot be planned: the code agent
returns empty SQL for it, and the answer model then narrated that internal failure as the cause ("the database query
encountered an error").  That reason is not something the data can support, so it reads as fabrication - and it is the
model's invention either way, because the real reason is simply that the columns do not exist.

Absence is a fact about the catalogue, so it is answered from the catalogue: name the fields that do not exist, say so
plainly, and offer what the same rows do hold.  The check is a column lookup, never a guess, and it fires before any
model call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: ``(label, request pattern, column-name hints)``.  A field counts as absent only when no loaded column matches any of
#: its hints, so a dataset that does carry the field is never told that it does not.
_FIELDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "seller-name",
        r"\b(?:seller|owner|buyer|listing agent|agent)(?:'s)?\s+(?:full\s+)?name\b|\bfull name\b",
        ("seller", "owner", "agent_name", "first_name", "last_name", "full_name"),
    ),
    ("phone", r"\bphone(?:\s+number)?s?\b|\bmobile\b", ("phone", "mobile")),
    ("email", r"\be-?mails?(?:\s+address(?:es)?)?\b", ("email",)),
    ("social-security", r"\bssn\b|\bsocial security\b", ("ssn", "social_security")),
    ("date-of-birth", r"\bdate of birth\b|\bdob\b|\bbirthdate\b", ("birth", "dob")),
)

_OFFER = (
    "I can list each house's address, city, list price, status, beds, baths and Walk, Bike and Transit scores instead."
)


@dataclass(frozen=True)
class Limitation:
    """A plain, code-owned answer for personal fields this database does not hold."""

    text: str
    fields: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"absent_fields": list(self.fields)}


def loaded_columns() -> set[str] | None:
    """Lower-cased names of every column actually present, or ``None`` when the database cannot be read."""
    try:
        from db import duckdb_store as store

        frame = store.query("SELECT column_name FROM information_schema.columns")
        return {str(value).lower() for value in frame.iloc[:, 0].tolist()}
    except Exception:  # a privacy answer must never be the thing that breaks a turn
        return None


def _join(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def personal_field_limitation(request: str) -> Limitation | None:
    """The answer for this request when it asks for personal fields that are not loaded, else ``None``."""
    asked = [(label, hints) for label, pattern, hints in _FIELDS if re.search(pattern, request, re.IGNORECASE)]
    if not asked:
        return None
    columns = loaded_columns()
    if columns is None:
        return None  # cannot prove absence, so the normal path answers
    absent = [label for label, hints in asked if not any(hint in column for hint in hints for column in columns)]
    if not absent:
        return None  # every requested field IS loaded: this is an ordinary question
    text = (
        f"The house and sold-home records in this database have no {_join(absent)} columns, "
        f"so I can't provide them. {_OFFER}"
    )
    return Limitation(text, tuple(absent))
