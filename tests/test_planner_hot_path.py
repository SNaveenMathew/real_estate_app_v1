"""The planner runs several times per chat turn.  Its cost must not grow with the catalog or the data, and requests
that have nothing to do with a feature must not pay for it.

Deterministic guards (counts and module checks), not wall-clock timings, so they never flake.
"""
from __future__ import annotations

import importlib
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import census_fixture  # noqa: E402
from plan_battery_queries import GROUPS  # noqa: E402


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("hot"))
    yield f
    census_fixture.deactivate(f)


@contextmanager
def realistic_scale(n_cities: int = 3000):
    """Enough distinct live entity values to exceed Python's 512-pattern regex cache, as real data does.

    On the tiny fixture alone the OLD regex matcher also reports zero compilations (everything fits in the cache), so a
    guard that ran only there could never fail.  Scale is what makes this test able to detect a regression.
    """
    import db.duckdb_store as store
    conn = store.get_conn()
    conn.execute("INSERT INTO houses (house_id, address, city, state, status) "
                 "SELECT 'scale' || i, 'a', 'Cityville' || i, 'PA', 'Active' FROM range(?) t(i)", [n_cities])
    try:
        yield
    finally:
        conn.execute("DELETE FROM houses WHERE house_id LIKE 'scale%'")


def _regex_compiler():
    """The function that really compiles a pattern (re._compile only reaches it on a cache miss)."""
    return getattr(re, "_compiler", None) or importlib.import_module("sre_compile")


def test_steady_state_planning_compiles_no_regular_expressions(fx, monkeypatch):
    """The old matcher compiled thousands of patterns per call (the cache holds a few hundred).  Any per-call dynamic
    pattern added to the planning path again would show up here immediately."""
    from agents.query_planner import build_query_plan
    questions = [q for qs in GROUPS.values() for q in qs]
    with realistic_scale():
        for q in questions:
            build_query_plan(q)                                # warm every cache once
        module, compiled = _regex_compiler(), []
        real = module.compile
        monkeypatch.setattr(module, "compile", lambda *a, **k: (compiled.append(a[0]), real(*a, **k))[1])
        for q in questions:
            build_query_plan(q)
    assert compiled == [], f"{len(compiled)} regex compilation(s) in steady-state planning, e.g. {compiled[:3]!r}"


def test_planning_work_does_not_depend_on_how_many_values_the_data_holds(fx):
    """Phrase matching is O(request): doubling the live entity values must not change the number of index lookups."""
    import db.duckdb_store as store
    from db import text_match
    from agents.query_planner import build_query_plan
    q = "How many houses are in Pittsburgh?"
    build_query_plan(q)
    before = text_match.index_for.cache_info()
    build_query_plan(q)
    small = text_match.index_for.cache_info().hits - before.hits
    conn = store.get_conn()
    conn.execute("INSERT INTO houses (house_id, address, city, state, status) "
                 "SELECT 'x' || i, 'a', 'City' || i, 'PA', 'Active' FROM range(3000) t(i)")
    try:
        build_query_plan(q)
        before = text_match.index_for.cache_info()
        build_query_plan(q)
        large = text_match.index_for.cache_info().hits - before.hits
    finally:
        conn.execute("DELETE FROM houses WHERE house_id LIKE 'x%'")
    # every extra value costs a dict lookup, never a new index/regex: the TEXT's index is built once and reused
    assert text_match.index_for.cache_info().currsize <= 512 and large >= small


def test_unrelated_requests_never_consult_the_place_lexicon(fx, monkeypatch):
    from agents.query_planner import build_query_plan
    from db import entity_lexicon
    calls = []
    monkeypatch.setattr(entity_lexicon, "get_lexicon", lambda *a, **k: calls.append(a[0]) or entity_lexicon.build(a[1]))
    for q in GROUPS["unrelated"]:
        build_query_plan(q)
    assert calls == [], f"unrelated questions triggered the lexicon: {calls}"


def test_house_and_tract_wording_is_excluded_before_any_lookup_happens(fx, monkeypatch):
    """excluded_terms are checked before requires_entity, so house/tract questions never pay for the entity lookup."""
    import db.schema_catalog as schema
    from agents.query_planner import build_query_plan
    calls = []
    real = schema.request_has_entity
    monkeypatch.setattr(schema, "request_has_entity", lambda *a, **k: (calls.append(a[0]), real(*a, **k))[1])
    for q in ["houses in Pittsburgh with population over 3000", "population of the tract containing this house",
              "which houses are in a high population neighborhood", "show houses within 5 square miles of downtown"]:
        build_query_plan(q)
    assert calls == []


def _fresh_import(statement: str) -> str:
    out = subprocess.run([sys.executable, "-c", statement], capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip().splitlines()[-1]


def test_importing_the_agents_does_not_load_the_geometry_provider():
    """The provider (and its geometry/numpy work) is imported lazily, only when a request needs a derived measure."""
    answer = _fresh_import("import sys, agents.tools, agents.general_agent; "
                           "print('services.census_metrics' in sys.modules, 'services.derived_measures' in sys.modules)")
    assert answer == "False True"
