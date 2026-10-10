"""A future dataset arrives through the catalog API (the same one the Data page uses).

Nothing in the planner, the tool layer, the answer policy, the validator or the orchestrator is edited in any of these
tests.  If one of them needs a code change to pass, the design has regressed from "metadata-driven" to "special-cased".
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402


@pytest.fixture()
def fx(tmp_path):
    f = census_fixture.activate(tmp_path)
    yield f
    from services import derived_measures
    derived_measures._PROVIDER_MODULES.pop("demo_provider", None)
    census_fixture.deactivate(f)


@pytest.fixture()
def sql_model(fx, monkeypatch):
    import agents.tools as tools

    class Stub:
        def __init__(self):
            self.replies, self.calls = [], 0

        def invoke(self, messages):
            self.calls += 1
            return types.SimpleNamespace(content=self.replies.pop(0) if self.replies else "")

    stub = Stub()
    monkeypatch.setattr(tools, "_agent", stub)
    return stub


def ask(request):
    import agents.tools as tools
    return tools.query_database.invoke({"request": request})


def register_concept(key, definition, dataset_id="future1"):
    import db.catalog_store as catalog_store
    import db.duckdb_store as store
    import db.schema_catalog as schema
    catalog_store.upsert_concept(store.get_conn(), key, definition, origin="upload", dataset_id=dataset_id)
    catalog_store.bump_version(store.get_conn())
    schema.reload()


def register_table(name, ddl, rows, dataset_id="future1", **meta):
    import db.catalog_store as catalog_store
    import db.duckdb_store as store
    import db.schema_catalog as schema
    from db.catalog_model import TableMeta
    conn = store.get_conn()
    conn.execute(ddl)
    for row in rows:
        conn.execute(f"INSERT INTO {name} VALUES ({','.join('?' for _ in row)})", list(row))
    catalog_store.upsert_table(conn, TableMeta(name=name, description=meta.get("description", name), grain=meta.get("grain", ""),
                                               domain=meta.get("domain", "other"), agent_visible=True, origin="upload",
                                               dataset_id=dataset_id), origin="upload", dataset_id=dataset_id)
    catalog_store.bump_version(conn)
    schema.reload()


# --------------------------------------------------------------------------------------------------------------------
def test_loading_a_dataset_silences_a_known_gap_with_no_code_change(fx, sql_model):
    """'median household income' is a declared gap; when income data is loaded, the gap yields on its own."""
    import db.catalog_store as catalog_store
    import db.duckdb_store as store
    import db.schema_catalog as schema
    from agents.query_planner import build_query_plan
    from db.catalog_model import AVG, _concept

    assert "NOT_ANSWERED[topic_unavailable]" in ask("median household income in Pittsburgh")

    register_table("acs_income", "CREATE TABLE acs_income (msa_name VARCHAR, median_household_income DOUBLE)",
                   [("Pittsburgh, PA Metro Area", 61000.0), ("Denver-Aurora-Lakewood, CO Metro Area", 85000.0)])
    register_concept("acs_median_income", _concept(
        "acs_median_income", ["acs_income"], ["median household income", "household income"],
        "Median household income by metro area.", columns=("acs_income.median_household_income",),
        operations=(AVG("acs_income.median_household_income"),), scope_guard=False))

    plan = build_query_plan("median household income in Pittsburgh")
    assert plan.semantic_keys == ["acs_median_income"], "the gap concept must yield to the real one"
    sql_model.replies = ["SELECT AVG(acs_income.median_household_income) FROM acs_income"]
    out = ask("median household income in Pittsburgh")
    assert "NOT_ANSWERED" not in out and "NOTE (not part" not in out
    assert "73000" in out and sql_model.calls == 1          # the new dataset's rows (avg of 61000 and 85000) were really queried

    # ... and retiring the dataset restores the honest "not loaded" answer
    catalog_store.retire_dataset(store.get_conn(), "future1")
    schema.reload()
    assert "NOT_ANSWERED[topic_unavailable]" in ask("median household income in Pittsburgh")


def test_a_new_derived_provider_plugs_in_through_metadata_alone(fx, sql_model):
    from db.catalog_model import _concept
    from services import derived_measures

    concept = _concept("demo_double_population", ["census_msa"], ["doubled population"], "Twice the MSA population.",
                       entity_types=("MSA",), scope_guard=False,
                       derived={"provider": "demo_provider", "measures": ["double_population"]})
    register_concept("demo_double_population", concept)
    assert "Code Agent error" in ask("doubled population of Pittsburgh")        # concept declared, provider not registered yet

    derived_measures.register_provider_module("demo_provider", "demo_provider")
    out = ask("doubled population of Pittsburgh")
    assert "double_population" in out and str(2 * fx.expected["Pittsburgh, PA Metro Area"]["population"]) in out
    assert "Doubled population is twice" in out and sql_model.calls == 0
    assert out.startswith("[GENERATED SQL]\nDerived measures: demo_provider\n[RESULT]\n")      # no evidence_label declared


def test_a_new_entity_domain_gets_place_matching_by_declaring_a_match_mode(fx, sql_model):
    """Any 'Name-Name, ST' label column gets city-component matching, state qualifiers and ambiguity detection."""
    import db.catalog_store as catalog_store
    import db.duckdb_store as store
    from agents.query_planner import build_query_plan
    from db.catalog_model import AVG, EntityDomain, _concept

    register_table("market_rents", "CREATE TABLE market_rents (market VARCHAR, rent DOUBLE)",
                   [("Shadyside-Squirrel Hill, PA", 1800.0), ("Springfield, MA", 1500.0), ("Springfield, MO", 900.0)])
    catalog_store.upsert_entity_domain(store.get_conn(), EntityDomain(
        "market_label", "market_rents", "market", "market", "Rental market labels", match_mode="components"),
        origin="upload", dataset_id="future1")
    catalog_store.bump_version(store.get_conn())
    register_concept("market_rent", _concept(
        "market_rent", ["market_rents"], ["average rent of", "rent in"], "Average rent by market.",
        columns=("market_rents.rent",), operations=(AVG("market_rents.rent"),), entity_types=("market",),
        groupings=("market_rents.market",), scope_guard=False))

    plan = build_query_plan("average rent of Squirrel Hill")                 # a non-principal part of the label
    assert {e["value"] for e in plan.resolved_entities} == {"Shadyside-Squirrel Hill, PA"}

    ambiguous = build_query_plan("average rent of Springfield")
    assert sum(1 for e in ambiguous.resolved_entities if e.get("ambiguous")) == 2
    out = ask("average rent of Springfield")
    assert "NOT_ANSWERED[place_ambiguous]" in out and "Springfield, MA" in out and "Springfield, MO" in out and sql_model.calls == 0
    assert {e["value"] for e in build_query_plan("average rent of Springfield, MO").resolved_entities} == {"Springfield, MO"}


def test_requires_entity_applies_to_any_entity_type(fx):
    from agents.query_planner import build_query_plan
    from db.catalog_model import AVG, _concept
    import db.catalog_store as catalog_store
    import db.duckdb_store as store
    from db.catalog_model import EntityDomain

    register_table("market_rents", "CREATE TABLE market_rents (market VARCHAR, rent DOUBLE)", [("Shadyside, PA", 1800.0)])
    catalog_store.upsert_entity_domain(store.get_conn(), EntityDomain(
        "market_label", "market_rents", "market", "market", "labels", match_mode="components"), origin="upload", dataset_id="future1")
    register_concept("market_rent_phrase", _concept(
        "market_rent_phrase", ["market_rents"], ["rent levels"], "Rent.", columns=("market_rents.rent",),
        operations=(AVG("market_rents.rent"),), entity_types=("market",), requires_entity=["market"], scope_guard=False))
    assert build_query_plan("rent levels in Shadyside").semantic_keys == ["market_rent_phrase"]
    assert build_query_plan("rent levels in Atlantis").semantic_keys == []          # no such market: concept stays out


def test_lint_catches_the_mistakes_that_would_otherwise_fail_silently(fx):
    from db import catalog_lint
    from db.catalog_model import _concept

    register_concept("typo_knob", _concept("typo_knob", ["census_msa"], ["typo phrase"], "x", require_entity=["MSA"]))
    register_concept("ghost_table", _concept("ghost_table", ["no_such_table"], ["ghost phrase"], "x"))
    register_concept("bad_provider", _concept("bad_provider", ["census_msa"], ["bad provider phrase"], "x", entity_types=("MSA",),
                                              derived={"provider": "nope", "measures": ["m"]}))
    register_concept("wrong_entity", _concept("wrong_entity", ["census_msa"], ["wrong entity phrase"], "x",
                                              derived={"provider": "msa_geometry", "measures": ["land_area"]}))
    register_concept("unknown_measure", _concept("unknown_measure", ["census_msa"], ["unknown measure phrase"], "x", entity_types=("MSA",),
                                                 derived={"provider": "msa_geometry", "measures": ["teleportation"]}))
    register_concept("needs_nothing", _concept("needs_nothing", ["census_msa"], ["needs nothing phrase"], "x", requires_entity=["galaxy"]))
    register_concept("gap_with_table", _concept("gap_with_table", ["census_msa"], ["gap table phrase"], "x", gap=True))

    found = {(i.code, i.subject) for i in catalog_lint.lint_catalog() if i.severity == "error"}
    assert ("unknown_key", "typo_knob") in found
    assert ("concept_dropped", "ghost_table") not in found                      # uploaded -> warning, not error
    assert any(i.code == "concept_dropped" and i.subject == "ghost_table" for i in catalog_lint.lint_catalog())
    assert ("unknown_provider", "bad_provider") in found
    assert ("derived_needs_entity", "wrong_entity") in found
    assert ("unknown_measure", "unknown_measure") in found
    assert ("unknown_entity_type", "needs_nothing") in found
    assert ("gap_with_tables", "gap_with_table") in found


def test_a_misspelled_knob_is_a_lint_error_even_though_the_planner_ignores_it(fx):
    """The planner silently ignores unknown keys (that is why the lint exists)."""
    from agents.query_planner import build_query_plan
    from db import catalog_lint
    from db.catalog_model import _concept
    register_concept("typo_knob", _concept("typo_knob", ["census_msa"], ["typo phrase"], "x", require_entity=["MSA"]))
    assert build_query_plan("typo phrase of Atlantis").semantic_keys == ["typo_knob"]     # the typo'd gate did nothing
    assert any(i.code == "unknown_key" for i in catalog_lint.lint_catalog())


def test_the_data_page_path_lets_a_future_dataset_claim_a_gap_topics_phrases(fx, sql_model):
    """Through the REAL onboarding function (services/dataset_onboarding.build_concepts), not a registry shortcut.

    Onboarding drops any alias a built-in concept already owns.  If a gap concept owned its phrases, the very dataset
    that finally covers the topic would get no usable alias and therefore no concept at all.
    """
    from agents.query_planner import build_query_plan
    from db import catalog_lint
    from services import dataset_onboarding as onboarding

    ds = {"table_name": "acs_income", "title": "ACS income by metro", "grain": "metro", "columns": [
        {"name": "msa_name", "role": "label", "dtype": "text", "include": True,
         "stats": {"distinct": 20, "min_len": 8, "unique_ratio": 1.0}},
        {"name": "median_household_income", "role": "measure", "include": True, "description": "Median household income",
         "unit": "usd", "synonyms": ["median household income", "household income"], "stats": {}}]}
    built = onboarding.build_concepts(ds)
    assert list(built["concepts"]) == ["acs_income_median_household_income"], built["warnings"]
    concept = built["concepts"]["acs_income_median_household_income"]
    assert concept["aliases"] == ["median household income", "household income"]
    assert not any("already used by" in w for w in built["warnings"]) and "overrides" not in concept

    register_table("acs_income", "CREATE TABLE acs_income (msa_name VARCHAR, median_household_income DOUBLE)",
                   [("Pittsburgh, PA Metro Area", 61000.0)])
    register_concept("acs_income_median_household_income", concept)
    assert build_query_plan("median household income in Pittsburgh").semantic_keys == ["acs_income_median_household_income"]
    # sharing a phrase with a gap concept is by design, so the lint must not call it a collision
    assert not [i for i in catalog_lint.lint_catalog() if i.code == "alias_collision" and "household income" in i.subject]


def test_built_in_aliases_are_still_protected_from_future_datasets(fx):
    """Only gap concepts are exempt: a real built-in concept still owns its phrases."""
    from services import dataset_onboarding as onboarding
    known = onboarding._known_aliases()
    assert known["land area"] == "census_land_area" and known["population density"] == "census_tract_population_density"
    assert "median household income" not in known and "poverty rate" not in known
