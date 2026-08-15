"""
eval/scoring.py

Deterministic "assert equal" scoring for structured golden examples.

Rather than parsing arbitrary natural-language replies into structured data
(fragile in general), this extracts evidence directly from the reply text
using each example's declared value_type:
  - "string":            substring search for each expected/distractor value
  - "number"/"currency": numbers pulled from the text (digits or small
                          number-words), compared with tolerance

Order handling, per GoldenExample.order_matters:
  - True:  every expected value must be found, AND their first-occurrence
           positions in the reply must be non-decreasing in the same
           sequence as `expected` — an ordered/sequence comparison.
  - False: every expected value must be found; position doesn't matter —
           a set comparison.

Either way, if any declared `distractor` is also found, that's an automatic
FAIL regardless of whether the correct values are present too. Distractors
encode "this specific wrong-but-plausible answer must not appear" (e.g. a
NULL averaged in as zero, a filter that should have been applied but wasn't)
rather than a random incorrect number.

Known limitation: the order check uses first-occurrence position, so a
reply that mentions a lower-ranked item in passing before the actual ranked
list (e.g. "Denver and Miami are both large. Ranked: 1. Miami 2. Denver...")
can produce a false FAIL. Golden questions are phrased to elicit a direct
ranked answer to minimize this, but it's a real tradeoff of text-based
grading rather than a full NL parse — noted here rather than silently
accepted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from eval.golden_set import GoldenExample


@dataclass
class ScoreResult:
    example_id: str
    verdict: str                 # "PASS" | "FAIL" | "ERROR"
    method: str                   # "assert_equal" | "llm_judge"
    reason: str
    reply: str = ""
    extra: dict = field(default_factory=dict)


_NUMBER_RE = re.compile(r'-?\$?\d[\d,]*\.?\d*%?')

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def _extract_numbers(text: str) -> list[tuple[float, int]]:
    """[(value, char_position), ...] for every number-looking token in text,
    including digit numbers and small spelled-out numbers (zero..twenty)."""
    out = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group().replace("$", "").replace(",", "").replace("%", "")
        try:
            out.append((float(raw), m.start()))
        except ValueError:
            continue
    for m in re.finditer(r"[A-Za-z]+", text):
        word = m.group().lower()
        if word in _WORD_NUMBERS:
            out.append((float(_WORD_NUMBERS[word]), m.start()))
    return out


def _find_number(text: str, target: float, tolerance: float) -> int | None:
    """First char position where a number ~= target appears, else None."""
    tol = abs(target) * tolerance if tolerance else 0
    for value, pos in _extract_numbers(text):
        if abs(value - target) <= tol:
            return pos
    return None


def _find_string(text: str, target: str) -> int | None:
    idx = text.lower().find(str(target).lower())
    return idx if idx >= 0 else None


def _find(text: str, target, value_type: str, tolerance: float) -> int | None:
    if value_type == "string":
        return _find_string(text, target)
    return _find_number(text, float(target), tolerance)


def score_structured(example: GoldenExample, reply: str) -> ScoreResult:
    if example.answer_type != "structured":
        raise ValueError(f"{example.id} is not a structured example")

    if not reply or not reply.strip():
        return ScoreResult(example.id, "FAIL", "assert_equal", "empty reply", reply)

    # Distractors first: a plausible-looking wrong answer fails the example
    # even if the right values also happen to be present.
    for d in example.distractors:
        pos = _find(reply, d, example.value_type, example.tolerance)
        if pos is not None:
            return ScoreResult(example.id, "FAIL", "assert_equal",
                                f"distractor value found in reply: {d!r}", reply,
                                extra={"distractor": d})

    positions = []
    for val in example.expected:
        pos = _find(reply, val, example.value_type, example.tolerance)
        if pos is None:
            return ScoreResult(example.id, "FAIL", "assert_equal",
                                f"expected value not found in reply: {val!r}", reply,
                                extra={"missing": val})
        positions.append(pos)

    if example.order_matters:
        if positions != sorted(positions):
            return ScoreResult(
                example.id, "FAIL", "assert_equal",
                f"values found out of order — expected sequence {example.expected}, "
                f"found at character positions {positions}", reply,
                extra={"positions": positions},
            )
        reason = f"all {len(example.expected)} expected values found, in the required order"
    else:
        reason = f"all {len(example.expected)} expected values found (order not required)"

    return ScoreResult(example.id, "PASS", "assert_equal", reason, reply)
