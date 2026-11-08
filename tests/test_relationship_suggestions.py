"""services/relationship_suggestions.py: a model widens what is TESTED for joins, never what is ACCEPTED."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402
from agents import llm_resilience as lr  # noqa: E402
from config import settings  # noqa: E402
from services import relationship_discovery as disc  # noqa: E402
from services import relationship_suggestions as rs  # noqa: E402


class Suggester:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def invoke(self, messages, **kw):
        self.prompts.append("\n".join(str(m.content) for m in messages))
        out = self.replies.pop(0) if self.replies else '{"pairs": []}'
        if isinstance(out, BaseException):
            raise out
        return types.SimpleNamespace(content=out)

    @property
    def calls(self):
        return len(self.prompts)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("relsug"))
    import db.duckdb_store as store
    conn = store.get_conn()
    # An upload whose join column is a CITY NAME under a name that resembles nothing in the catalog ("locale"), plus a second
    # candidate column ("city") whose name DOES match houses.city, and a numeric measure.
    conn.execute("CREATE OR REPLACE TABLE up_stage AS SELECT city AS locale, city AS city, 12.5 AS score FROM (SELECT DISTINCT city FROM houses)")
    yield f
    census_fixture.deactivate(f)


DATASET = {"table_name": "up_final", "columns": [
    {"name": "locale", "role": "dimension", "include": True},
    {"name": "city", "role": "dimension", "include": True},
    {"name": "score", "role": "measure", "include": True}]}


@pytest.fixture()
def suggester(fx, monkeypatch):
    lr.reset()
    stub = Suggester()
    monkeypatch.setattr(rs, "_client", stub)
    monkeypatch.setattr(settings, "relationship_llm_suggestions_enabled", True)       # opt-in; these tests opt in, with a stub
    return stub


def discover(dataset=DATASET):
    import db.duckdb_store as store
    return disc.discover_relationships(store.get_conn(), dataset, "up_stage")


def by_key(proposals):
    return {p["title"]: p for p in proposals}


# --------------------------------------------------------------------------------------------------------------------
# The gap, and the baseline
# --------------------------------------------------------------------------------------------------------------------
def test_baseline_a_pair_with_unrelated_names_is_never_even_tested(fx, monkeypatch):
    monkeypatch.setattr(settings, "relationship_llm_suggestions_enabled", False)
    titles = " | ".join(by_key(discover()))
    assert "up_final.city joins houses.city" in titles            # the name-based pair is found, as always
    assert "up_final.locale" not in titles                        # the unrelated name is invisible to name-based candidates


def test_it_is_off_by_default_and_then_never_calls_a_model(fx, monkeypatch):
    stub = Suggester('{"pairs": [{"source": "locale", "target": "houses.city"}]}')
    monkeypatch.setattr(rs, "_client", stub)
    from config import Settings
    assert Settings().relationship_llm_suggestions_enabled is False
    monkeypatch.setattr(settings, "relationship_llm_suggestions_enabled", False)
    discover()
    assert stub.calls == 0


# --------------------------------------------------------------------------------------------------------------------
# The suggested path
# --------------------------------------------------------------------------------------------------------------------
def test_a_suggested_pair_is_tested_against_the_data_and_proposed_with_a_label(suggester):
    suggester.replies = ['{"pairs": [{"source": "locale", "target": "houses.city"}]}']
    proposals = by_key(discover())
    p = proposals["up_final.locale joins houses.city"]
    assert p["payload"]["origin"] == "llm-suggested" and p["payload"]["score"] >= 0.9
    assert p["evidence"]["stats"]["matched_distinct"] >= 1 and p["evidence"]["examples"]       # measured from the data
    assert any("assistant suggested" in w for w in p["evidence"]["warnings"])
    assert "origin" not in proposals["up_final.city joins houses.city"]["payload"]            # the name-based pair is not labelled


def test_a_suggested_proposal_has_exactly_the_same_shape_as_a_name_based_one(suggester):
    suggester.replies = ['{"pairs": [{"source": "locale", "target": "houses.city"}]}']
    proposals = by_key(discover())
    a, b = proposals["up_final.locale joins houses.city"], proposals["up_final.city joins houses.city"]
    assert set(a) == set(b) and set(a["evidence"]) == set(b["evidence"])
    assert set(a["payload"]) - {"origin"} == set(b["payload"])                               # downstream storage needs nothing new


def test_a_pair_whose_values_do_not_match_is_rejected_by_the_data_not_trusted(suggester):
    suggester.replies = ['{"pairs": [{"source": "locale", "target": "houses.house_id"}, {"source": "locale", "target": "houses.state"}]}']
    titles = " | ".join(by_key(discover()))
    assert "up_final.locale" not in titles                                                     # a confident guess, refuted by the values


def test_hallucinated_names_and_garbage_are_ignored(suggester):
    for reply in ('{"pairs": [{"source": "nope", "target": "houses.city"}, {"source": "locale", "target": "ghost.col"}]}',
                  '{"pairs": "locale"}', "no idea", "", "{broken", '{"pairs": [1, null, {"source": 5}]}'):
        suggester.replies = [reply]
        assert "up_final.locale" not in " | ".join(by_key(discover()))


def test_a_pair_that_name_matching_already_found_is_not_labelled_or_duplicated(suggester):
    suggester.replies = ['{"pairs": [{"source": "city", "target": "houses.city"}]}']
    proposals = discover()
    matches = [p for p in proposals if p["title"] == "up_final.city joins houses.city"]
    assert len(matches) == 1 and "origin" not in matches[0]["payload"]


def test_one_bounded_call_per_dataset_and_the_prompt_is_compact_and_data_light(suggester):
    suggester.replies = ['{"pairs": []}']
    discover()
    assert suggester.calls == 1
    prompt = suggester.prompts[0]
    assert "SOURCE COLUMNS" in prompt and "- locale (text" in prompt and "houses.city (text)" in prompt
    assert "score" not in prompt.split("TARGET COLUMNS")[0]                                  # the measure is not a join candidate
    assert len(prompt) < 20_000


def test_it_can_never_break_onboarding(suggester):
    for failure in (ConnectionError("refused"), RuntimeError("boom"), ValueError("x")):
        suggester.replies = [failure]
        titles = " | ".join(by_key(discover()))
        assert "up_final.city joins houses.city" in titles                                    # exactly the previous behaviour


def test_a_dead_model_is_remembered_by_its_own_breaker_and_cannot_affect_chat(suggester):
    suggester.replies = [ConnectionError("refused")] * 6
    for _ in range(4):
        discover()
    assert lr.breaker("relationships").state == "open" and lr.breaker("chat").state == "closed"
    calls = suggester.calls
    discover()
    assert suggester.calls == calls                                                           # open breaker: no further calls


@pytest.mark.parametrize("reply,expected_n", [
    ('{"pairs": [{"source": "locale", "target": "houses.city"}, {"source": "locale", "target": "houses.city"}]}', 1),     # de-duplicated
    ('{"pairs": [{"source": "locale", "target": "houses.city"}, {"source": "locale", "target": "houses.state"}, '
     '{"source": "locale", "target": "houses.zip_code"}, {"source": "locale", "target": "houses.address"}]}', 3),          # <= 3 per source
])
def test_pairs_are_deduplicated_and_capped_per_source(fx, reply, expected_n):
    sources = {"locale", "city"}
    import db.schema_catalog as schema
    targets = {f"houses.{c}": {"table": "houses", "column": c, "family": "text"} for c in ("city", "state", "zip_code", "address")}
    assert len(rs.parse_pairs(reply, sources, targets)) == expected_n
