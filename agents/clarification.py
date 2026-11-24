"""Ask ONE deterministic clarifying question when a turn cannot be planned but a place is known.

A follow-up such as *"and how many do I have in Denver?"* names a place and no measure.  Inheriting the measure from
history is the tempting shortcut and it is wrong: concatenating the previous turn plans ``houses.city = 'Pittsburgh'``
AND ``houses.city = 'Denver'``, i.e. it answers a different question with a confident number.  A wrong number the user
cannot detect is strictly worse than a question, so the agent asks - which is what ``unplannable_result`` already
recommends in prose, now turned into an actual question offering the measures this catalog covers for that place.

Two rules keep it honest in production:

* the trigger is a STATUS CODE plus a recognized place, never a model's hunch.  A turn that produced real evidence can
  therefore never be interrogated, and an offline-reproducible fact decides whether to ask.
* the text carries no digits: the response validator's "does this contain data?" heuristic is digit-based, and a
  question must never be mistaken for an answer.

The conversational loop lives in the caller (the orchestrator, the HTTP layer, or the eval harness): this module only
decides *whether* to ask and *what* to ask.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import db.schema_catalog as schema
from agents import answer_status as status

#: The one status a user reply resolves.  ``unplannable`` without a place stays a plain guidance message (there is
#: nothing to anchor a question to); ``place_ambiguous`` already asks its own question through the answer model, so
#: templating it here would only replace a better sentence; every other status needs an operator, not the user.
CLARIFYABLE = frozenset({status.UNPLANNABLE})


@dataclass(frozen=True)
class Clarification:
    """A question to put to the user, plus the machine-readable reason it is being asked."""

    text: str
    code: str
    places: tuple[str, ...]
    candidates: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"code": self.code, "places": list(self.places), "candidates": list(self.candidates)}


def _recognized(request: str) -> list[dict]:
    try:
        found = schema.recognized_entities(request)
    except Exception:  # a clarification must never be the thing that breaks a turn
        return []
    return [
        e
        for e in found
        if e.get("entity_type") != "tract_fips" and not any(ch.isdigit() for ch in str(e.get("value", "")))
    ]


_TRAILING = {"of", "in", "for", "to", "by", "with", "and", "the", "a", "an"}


def _place_names(places: list[dict]) -> tuple[str, ...]:
    """'Denver', not 'Denver, Denver, CO Metro Area and denver': one name per place, plain names preferred.

    A metro value is the same place qualified with comma-separated components, and the catalog may hold several cases
    of a name, so the shortest spelling wins and its capitalization is normalized to the most proper one seen.
    """
    kept: list[str] = []
    for value in sorted({str(e["value"]) for e in places}, key=len):
        low = value.lower()
        same = next((i for i, k in enumerate(kept) if k.lower() == low), None)
        if same is not None:
            # same place, different casing: keep the capitalised spelling, whichever arrived first
            if value[:1].isupper() and not kept[same][:1].isupper():
                kept[same] = value
            continue
        if any(low.startswith(k.lower()) and not low[len(k) : len(k) + 1].isalnum() for k in kept):
            continue
        kept.append(value)
    return tuple(sorted(kept))


_COUNT_PHRASE = re.compile(r"\b(how many|number of|count|how much)\b", re.IGNORECASE)


def _speaks_for(alias: str) -> bool:
    """An offer a user can actually repeat back: multi-word, digit-free, not ending on a preposition."""
    words = alias.split()
    return (
        len(words) >= 2
        and not any(ch.isdigit() for ch in alias)
        and len(alias) <= 40
        and words[0].lower() not in _TRAILING
        and words[-1].lower() not in _TRAILING
    )


def _has_count(concept: dict) -> bool:
    return any(str(o.get("name") or o.get("op") or "") == "count" for o in (concept.get("operations") or ()))


def _candidates(places: list[dict], request: str, limit: int = 4) -> tuple[str, ...]:
    """One measure per table the place lives in - the offer must be answerable.

    Every candidate is an alias of a concept over a table that can carry the recognized place, so a user repeating it
    back lands on a plannable measure.  Which concept speaks for a table follows the question's shape: a counting
    question ("how many") is offered the table's count concept, anything else its richest measure.
    """
    tables = [str(e["table"]) for e in places if e.get("table")]
    counting = bool(_COUNT_PHRASE.search(request))
    picked: list[str] = []
    for table in dict.fromkeys(tables):
        options = [
            (key, concept)
            for key, concept in sorted(schema.SEMANTIC_GLOSSARY.items())
            if not concept.get("gap") and concept.get("aliases") and str((concept.get("tables") or [""])[0]) == table
        ]
        if not options:
            continue
        counts = [pair for pair in options if _has_count(pair[1])]
        if counting and counts:
            options = counts
        else:
            options.sort(key=lambda pair: (-len(pair[1].get("operations") or ()), pair[0]))
        concept = options[0][1]
        alias = next((a for a in concept["aliases"] if _speaks_for(a)), concept["aliases"][0])
        if alias.lower() in {p.lower() for p in picked}:
            continue
        picked.append(alias)
        if len(picked) >= limit:
            break
    return tuple(picked)


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def question_for(request: str, evidence: str) -> Clarification | None:
    """The question this turn should ask, or ``None`` when it answered (or refused) on its own."""
    code = status.status_code(evidence)
    if code not in CLARIFYABLE:
        return None
    places = _recognized(request)
    names = _place_names(places)
    if not names:
        return None  # nothing to anchor a question to: the status message already gives the guidance

    candidates = _candidates(places, request)
    where = _join(list(names))
    if candidates:
        text = (
            f"I recognized {where} but not which measure you want. "
            f"Which one do you mean - for example {_join(list(candidates))}?"
        )
    else:
        text = f"I recognized {where} but not which measure you want. Which measure should I use?"
    return Clarification(text, code, names, candidates)
