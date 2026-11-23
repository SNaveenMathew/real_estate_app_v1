"""The result cache (services/cache.py) through its real integration points.

  * ``agents/llm_resilience.invoke`` — the one place every model call goes through, so the prompt
    cache covers both chats, the SQL Code Agent and the optional proposers at once;
  * ``db/duckdb_store.query_limited_cached`` — the shared SQL/CTE result cache;
  * ``cache.semantic_signature`` / ``lookup_semantic`` — reuse for a rephrased question;
  * the graph model and HTTP API behind ``/cache``.

The point of most of these tests is not that a hit happens, but that a hit CANNOT happen once the
data, the scope or the question has changed.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agents import llm_resilience
from config import settings
from db.duckdb_store import query_limited_cached
from services import cache


@pytest.fixture(autouse=True)
def _fresh_fingerprint(monkeypatch, _cache_off_by_default):
    """Cache ON (the suite's shared fixture turns it off), and recompute the data fingerprint on every
    access so a data change is visible to the very next call."""
    monkeypatch.setattr(settings, "cache_fingerprint_ttl_s", 0.0)
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "cache_prompt_enabled", True)
    monkeypatch.setattr(settings, "cache_semantic_enabled", True)
    monkeypatch.setattr(settings, "cache_sql_enabled", True)
    monkeypatch.setattr(settings, "cache_ttl_s", 86400.0)
    cache._VER.update(value="", at=0.0)
    llm_resilience.reset()
    yield
    cache._VER.update(value="", at=0.0)
    llm_resilience.reset()


class StubAgent:
    """A model client double: records every call, replays the scripted replies."""

    def __init__(self, replies, temperature=0.0, model="stub-model"):
        self.replies = list(replies)
        self.calls = []
        self.temperature = temperature
        self.model_name = model

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.replies.pop(0) if self.replies else "ok")


def human(text):
    return SimpleNamespace(type="human", content=text)


def system(text):
    return SimpleNamespace(type="system", content=text)


# ---------------------------------------------------------------------------
# Prompt / context cache
# ---------------------------------------------------------------------------


def test_an_unchanged_prompt_is_replayed_instead_of_regenerated(reference_data):
    agent = StubAgent(["SELECT 1", "SELECT 2"])
    messages = [system("You are a SQL Code Agent."), human("How many houses are there?")]

    first = llm_resilience.invoke(agent, messages, role="sql_agent")
    second = llm_resilience.invoke(agent, messages, role="sql_agent")

    assert first.content == "SELECT 1"
    assert second.content == "SELECT 1"  # replayed, not regenerated
    assert len(agent.calls) == 1
    assert getattr(second, "cache_hit", False) is True


def test_a_sampling_client_is_never_cached(reference_data):
    """Caching a temperature>0 call would pin one arbitrary sample as 'the' answer."""
    agent = StubAgent(["a", "b"], temperature=0.7)
    messages = [human("Write something.")]

    assert llm_resilience.invoke(agent, messages).content == "a"
    assert llm_resilience.invoke(agent, messages).content == "b"
    assert len(agent.calls) == 2


def test_the_same_prompt_under_two_roles_is_two_entries(reference_data):
    """A completion written for one role is not handed to another: the prompt templates differ."""
    agent = StubAgent(["as sql", "as answer"])
    messages = [system("shared preamble"), human("same text")]

    assert llm_resilience.invoke(agent, messages, role="sql_agent").content == "as sql"
    assert llm_resilience.invoke(agent, messages, role="answer_agent").content == "as answer"
    assert len(agent.calls) == 2


def test_a_reloaded_dataset_invalidates_prompt_entries(reference_data):
    """A cached completion may be stale the moment the data it summarises changes."""
    agent = StubAgent(["before the reload", "after the reload"])
    messages = [human("How many houses are there?")]

    assert llm_resilience.invoke(agent, messages).content == "before the reload"

    reference_data.execute(
        "INSERT INTO houses (house_id,address,city,state,zip) VALUES ('h9','1 New Row','Pittsburgh','PA','15213')"
    )

    assert llm_resilience.invoke(agent, messages).content == "after the reload"
    assert len(agent.calls) == 2


def test_turning_the_prompt_cache_off_bypasses_it(reference_data, monkeypatch):
    monkeypatch.setattr(settings, "cache_prompt_enabled", False)
    agent = StubAgent(["one", "two"])
    messages = [human("Same prompt.")]

    assert llm_resilience.invoke(agent, messages).content == "one"
    assert llm_resilience.invoke(agent, messages).content == "two"
    assert len(agent.calls) == 2


# ---------------------------------------------------------------------------
# SQL / CTE result cache
# ---------------------------------------------------------------------------


def test_an_identical_statement_reuses_its_rows(reference_data):
    sql = "WITH per_city AS (SELECT city, COUNT(*) AS n FROM houses GROUP BY city) SELECT * FROM per_city ORDER BY city"

    first, trunc1, hit1 = query_limited_cached(sql)
    second, trunc2, hit2 = query_limited_cached(sql)

    assert hit1 is False and trunc1 is False
    assert hit2 is True  # the second read never reached DuckDB
    assert list(second.columns) == list(first.columns)
    assert second.to_dict(orient="records") == first.to_dict(orient="records")


def test_a_changed_table_invalidates_the_sql_cache(reference_data):
    sql = "SELECT COUNT(*) AS n FROM houses"
    first, _, hit1 = query_limited_cached(sql)
    assert hit1 is False and int(first.iloc[0]["n"]) == 6

    reference_data.execute(
        "INSERT INTO houses (house_id,address,city,state,zip) VALUES ('h9','1 New Row','Pittsburgh','PA','15213')"
    )

    second, _, hit2 = query_limited_cached(sql)
    assert hit2 is False  # recomputed, not served from the cache
    assert int(second.iloc[0]["n"]) == 7


def test_an_empty_result_is_not_cached(reference_data):
    """A zero-row answer is usually a mistake being repaired, not a result worth remembering."""
    sql = "SELECT * FROM houses WHERE city = 'Nowhere'"
    _, _, hit1 = query_limited_cached(sql)
    _, _, hit2 = query_limited_cached(sql)
    assert hit1 is False and hit2 is False


def test_clearing_the_cache_stops_every_hit(reference_data):
    sql = "SELECT COUNT(*) AS n FROM houses"
    query_limited_cached(sql)
    assert query_limited_cached(sql)[2] is True

    assert cache.invalidate()["ok"] is True

    assert query_limited_cached(sql)[2] is False


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------


def test_a_rephrased_question_with_the_same_plan_reuses_the_answer(reference_data):
    from agents.query_planner import build_query_plan

    plan = build_query_plan("What is the average walk score?")
    signature = cache.semantic_signature(plan)
    assert signature, "the catalog should plan an average walk score question"

    answer = "[GENERATED SQL]\nSELECT AVG(walk_score) FROM houses\n[RESULT]\n62.5"
    cache.store_semantic(
        "query_database",
        "What is the average walk score?",
        signature,
        answer,
        meta={"concepts": plan.semantic_keys, "tables": plan.required_tables, "operation": plan.operation},
    )

    hit = cache.lookup_semantic("query_database", "What's the average walk score?", signature)
    assert hit is not None and hit["response"] == answer
    assert hit["similarity"] >= settings.cache_semantic_threshold


def test_a_differently_scoped_question_never_reuses_an_answer(reference_data):
    """Resolved entities are part of the signature: 'in Pittsburgh' is not 'everywhere'."""
    from agents.query_planner import build_query_plan

    everywhere = build_query_plan("How many houses are there?")
    pittsburgh = build_query_plan("How many houses are there in Pittsburgh?")
    assert cache.semantic_signature(everywhere) != cache.semantic_signature(pittsburgh)

    cache.store_semantic(
        "query_database", "How many houses are there?", cache.semantic_signature(everywhere), "all of them"
    )
    assert (
        cache.lookup_semantic("query_database", "How many houses are there?", cache.semantic_signature(pittsburgh))
        is None
    )


def test_a_question_the_planner_could_not_place_is_not_cached(reference_data):
    from agents.query_planner import build_query_plan

    plan = build_query_plan("What is the airspeed velocity of an unladen swallow?")
    assert cache.semantic_signature(plan) is None


# ---------------------------------------------------------------------------
# Cache page: graph model + HTTP API
# ---------------------------------------------------------------------------


def test_the_graph_shows_catalog_abstractions_and_their_connections(reference_data):
    import db.schema_catalog as schema
    from agents.query_planner import build_query_plan

    concept = schema.describe_catalog()["concepts"][0]
    plan = build_query_plan("What is the average walk score?")
    cache.store_semantic(
        "query_database",
        "What is the average walk score?",
        cache.semantic_signature(plan),
        "[RESULT]\n62.5",
        meta={"concepts": [concept["key"]], "tables": ["houses"], "operation": "avg"},
    )

    model = cache.graph()

    concepts = {c["key"]: c for c in model["abstractions"]}
    assert concept["key"] in concepts  # abstractions come from the live catalog ...
    assert concepts[concept["key"]]["questions"] == 1  # ... and show whether they have been reused
    assert any(q["request_text"] == "What is the average walk score?" for q in model["questions"])
    assert any(t["label"] == "houses" for t in model["tables"])
    assert {e["kind"] for e in model["edges"]} == {"asks", "reads"}


def test_the_cache_api_serves_stats_graph_entries_and_clear(reference_data):
    import main

    client = TestClient(main.app)
    from agents.query_planner import build_query_plan

    plan = build_query_plan("What is the average walk score?")
    cache.store_semantic(
        "query_database",
        "What is the average walk score?",
        cache.semantic_signature(plan),
        "[RESULT]\n62.5",
        meta={"concepts": plan.semantic_keys, "tables": plan.required_tables},
    )

    stats = client.get("/api/cache").json()
    assert stats["enabled"] is True and stats["semantic"]["entries"] == 1

    graph = client.get("/api/cache/graph").json()
    assert graph["abstractions"] and graph["questions"]

    listed = client.get("/api/cache/entries", params={"kind": "semantic"}).json()["entries"]
    assert listed and listed[0]["request_text"] == "What is the average walk score?"

    entry = client.get(f"/api/cache/entry/{listed[0]['cache_key']}").json()
    assert entry["response"] == "[RESULT]\n62.5"

    assert client.post("/api/cache/clear", json={}).json()["ok"] is True
    assert client.get("/api/cache").json()["total_entries"] == 0


def test_the_cache_page_url_is_served(reference_data):
    """Issue #16's page; it now draws the ontology (tests/test_ontology.py covers that binding), and
    ``/cache`` is kept as the historical URL so existing links keep working."""
    import main

    page = TestClient(main.app).get("/cache")
    assert page.status_code == 200
    assert "/static/cache.js" in page.text
