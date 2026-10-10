"""The planner's token-index matcher must be indistinguishable from the regex it replaced.

``REFERENCE_*`` below are the original implementations, kept verbatim as the oracle.  If this test fails, planning
behaviour has changed for every request, not just the one being edited.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import text_match  # noqa: E402


def REFERENCE_NORMALIZE(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def REFERENCE_SPANS(text: str, alias: str):
    t = REFERENCE_NORMALIZE(text)
    a = REFERENCE_NORMALIZE(alias)
    if not a:
        return []
    pattern = r"(?<!\w)" + re.escape(a) + r"(?!\w)"
    return [(m.start(), m.end(), len(a.split())) for m in re.finditer(pattern, t)]


VOCAB = ["a", "b", "aa", "pa", "pittsburgh", "metro", "area", "of", "the", "s", "msa", "population", "42003", "x1",
         "st", "louis", "café", "ñandú", "O'Hare", "e-mail", "x_y", "A", "PA", "Pittsburgh", "square", "mile", "miles"]
SEPARATORS = [" ", " ", " ", ", ", "-", "'", "_", "  ", "\t", " & ", ". ", "/", "--"]


def _random_text(rng: random.Random) -> str:
    n = rng.randint(0, 14)
    parts = []
    for _ in range(n):
        parts.append(rng.choice(VOCAB))
        parts.append(rng.choice(SEPARATORS))
    return "".join(parts)


def test_normalize_matches_reference():
    rng = random.Random(1)
    for _ in range(5000):
        s = _random_text(rng) + rng.choice(["", "  ", "!!", "\u00a0x"])
        assert text_match.normalize(s) == REFERENCE_NORMALIZE(s)


def test_phrase_spans_match_reference_on_random_inputs():
    rng = random.Random(2)
    checked = hits = 0
    for _ in range(30000):
        text = _random_text(rng)
        toks = REFERENCE_NORMALIZE(text).split()
        if toks and rng.random() < 0.75:                 # mostly aliases that really occur (incl. repeats/overlaps)
            i = rng.randrange(len(toks))
            j = min(len(toks), i + rng.randint(1, 4))
            alias = " ".join(toks[i:j])
            alias = rng.choice([alias, alias.upper(), alias.replace(" ", ", "), f" {alias}- "])
        else:
            alias = "".join(rng.choice(VOCAB) + rng.choice(SEPARATORS) for _ in range(rng.randint(0, 3)))
        expected = REFERENCE_SPANS(text, alias)
        assert text_match.phrase_spans(text, alias) == expected, (text, alias)
        assert text_match.has_phrase(text, alias) == bool(expected), (text, alias)
        checked += 1
        hits += bool(expected)
    assert checked == 30000 and hits > 8000            # the oracle comparison must exercise real matches


@pytest.mark.parametrize("text,alias", [
    ("a a a", "a a"), ("a a a a", "a a"), ("aaa aa a", "a"), ("", "x"), ("x", ""), ("   ", "  "),
    ("Pittsburgh, PA Metro Area", "pittsburgh pa"), ("Pittsburgh's population", "s population"),
    ("x-ray x ray", "x ray"), ("42003010300", "42003010300"), ("sold-home records", "sold home records"),
])
def test_edge_cases_match_reference(text, alias):
    assert text_match.phrase_spans(text, alias) == REFERENCE_SPANS(text, alias)


def test_semantic_matches_identical_under_both_matchers(tmp_path_factory, monkeypatch):
    """Catalog-level equivalence: concept matching (incl. overrides) must not change for any phrasing."""
    import census_fixture
    fx = census_fixture.activate(tmp_path_factory.mktemp("tm"))
    try:
        import db.schema_catalog as schema
        rng = random.Random(3)
        aliases = [a for item in schema._glossary().values() for a in item.get("aliases", [])]
        queries = []
        for _ in range(400):
            picked = rng.sample(aliases, rng.randint(1, 3))
            filler = ["show", "the", "of", "in", "Pittsburgh", "and", "by", "for", "what", "is", "average"]
            words = picked + rng.sample(filler, 3)
            rng.shuffle(words)
            queries.append(" ".join(words))
        fast = [[(c["key"], c["match_score"]) for c in schema.semantic_matches(q)] for q in queries]
        monkeypatch.setattr(schema, "_alias_spans", REFERENCE_SPANS)
        monkeypatch.setattr(schema, "_alias_matches", lambda text, alias: bool(REFERENCE_SPANS(text, alias)))
        slow = [[(c["key"], c["match_score"]) for c in schema.semantic_matches(q)] for q in queries]
        assert fast == slow
        assert sum(1 for r in fast if r) > 300
    finally:
        census_fixture.deactivate(fx)
