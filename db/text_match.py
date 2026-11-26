"""Token-level phrase matching for the planner (concept aliases and entity values).

The planner asks one question thousands of times per request: *does this normalized alias occur, as whole
words, in the normalized request?*  It used to build and run one regular expression per alias and per live
entity value.  Python's ``re`` keeps only a few hundred compiled patterns, so with a realistic catalog (hundreds
of aliases plus every metro and city name) nearly every call re-compiled thousands of patterns - the dominant
fixed cost of every chat turn, and one that grew with every dataset added.

A normalized text contains only ``[a-z0-9]`` and single spaces, so "alias occurs at word boundaries" is exactly
"the alias's tokens occur contiguously in the text's tokens".  This module answers that with a per-text
n-gram index, built lazily and cached, so matching costs O(request) however large the catalog or the data are.

Semantics are intentionally identical to the regex they replace, including the non-overlapping-match rule of
``re.finditer`` that the declared-precedence (``overrides``) logic relies on.  ``tests/test_text_match.py``
proves the equivalence differentially against the original regex implementation.
"""

from __future__ import annotations

import re
from functools import lru_cache

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@lru_cache(maxsize=65536)
def normalize(text: str) -> str:
    """Lowercase, collapse every run of non-alphanumerics to one space, strip.  Pure, so safely cached."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


class TextIndex:
    """Token positions and lazily-built n-gram tables for ONE normalized text."""

    __slots__ = ("tokens", "starts", "ends", "_grams")

    def __init__(self, normalized: str):
        self.tokens: list[str] = normalized.split(" ") if normalized else []
        self.starts: list[int] = []
        self.ends: list[int] = []
        pos = 0
        for tok in self.tokens:
            self.starts.append(pos)
            self.ends.append(pos + len(tok))
            pos += len(tok) + 1
        self._grams: dict[int, dict[str, list[int]]] = {}

    def grams(self, n: int) -> dict[str, list[int]]:
        """phrase -> ascending token start positions, for every n-token window of the text."""
        table = self._grams.get(n)
        if table is None:
            table = {}
            toks = self.tokens
            for i in range(len(toks) - n + 1):
                table.setdefault(" ".join(toks[i : i + n]), []).append(i)
            self._grams[n] = table
        return table


@lru_cache(maxsize=512)
def index_for(normalized: str) -> TextIndex:
    return TextIndex(normalized)


def phrase_spans(text: str, alias: str) -> list[tuple[int, int, int]]:
    """``(char_start, char_end, n_words)`` of each non-overlapping whole-word occurrence of ``alias`` in ``text``.

    Offsets refer to the *normalized* text, exactly as the regex-based implementation reported them.
    """
    a = normalize(alias)
    if not a:
        return []
    n = a.count(" ") + 1
    idx = index_for(normalize(text))
    occurrences = idx.grams(n).get(a)
    if not occurrences:
        return []
    out: list[tuple[int, int, int]] = []
    next_free = 0  # re.finditer never returns overlapping matches
    for p in occurrences:
        if p < next_free:
            continue
        out.append((idx.starts[p], idx.ends[p + n - 1], n))
        next_free = p + n
    return out


def has_phrase(text: str, alias: str) -> bool:
    a = normalize(alias)
    if not a:
        return False
    return a in index_for(normalize(text)).grams(a.count(" ") + 1)


def stem_token(token: str) -> str:
    """Conservative plural/inflection stemmer used only for fallback matching (never alters ``normalize``)."""
    t = token.lower()
    if len(t) <= 3 or t.isdigit():
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith(("ses", "xes", "zes", "ches", "shes")) and len(t) > 5:
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is", "as")):
        return t[:-1]
    return t


def damerau_distance(a: str, b: str, max_dist: int = 2) -> int:
    """Optimal string alignment (bounded Damerau-Levenshtein) distance, early-exiting past ``max_dist``."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if a == b:
        return 0
    prev2 = list(range(lb + 1))
    prev1 = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [max_dist + 1] * lb
        row_min = curr[0]
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            best = min(curr[j - 1] + 1, prev1[j] + 1, prev1[j - 1] + cost)
            if i > 1 and j > 1 and ai == b[j - 2] and a[i - 2] == b[j - 1]:
                best = min(best, prev2[j - 2] + 1)
            curr[j] = best
            if best < row_min:
                row_min = best
        if row_min > max_dist:
            return max_dist + 1
        prev2, prev1 = prev1, curr
    return prev1[lb]


def repair_typos(
    text: str,
    vocab: set[str] | frozenset[str],
    *,
    protected: set[str] | frozenset[str] = (),
) -> tuple[str, bool]:
    """Repair clear out-of-vocabulary spelling typos (e.g. transpositions like ``hosues`` -> ``houses``).

    Returns ``(normalized_text, changed)``. Common English words in ``protected`` and tokens already in
    ``vocab`` are never altered, guaranteeing zero drift on well-spelled queries.
    """
    normed = normalize(text)
    if not normed:
        return "", False
    tokens = normed.split()
    out: list[str] = []
    changed = False
    buckets: dict[int, list[str]] = {}
    for w in vocab:
        if len(w) >= 5:
            buckets.setdefault(len(w), []).append(w)

    for tok in tokens:
        # Never treat a regular English plural (e.g. "scores" when "score" is in vocab) as a spelling typo.
        if tok in vocab or stem_token(tok) in vocab or tok in protected or len(tok) < 5 or tok.isdigit():
            out.append(tok)
            continue
        limit = 1
        cands: list[str] = []
        for length in range(max(5, len(tok) - limit), len(tok) + limit + 1):
            cands.extend(buckets.get(length, ()))
        best_dist = limit + 1
        winners: list[str] = []
        for cand in cands:
            d = damerau_distance(tok, cand, max_dist=limit)
            if d > limit:
                continue
            if d < best_dist:
                best_dist = d
                winners = [cand]
            elif d == best_dist:
                winners.append(cand)
        if len(winners) == 1:
            out.append(winners[0])
            changed = True
        else:
            out.append(tok)
    return (" ".join(out), changed)
