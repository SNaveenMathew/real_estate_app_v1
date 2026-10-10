"""query_database end to end (real planner, real policy, real provider, stubbed SQL model).

The most important assertions here are about what the SQL model is NOT asked to do: the original failure burned three
model calls on a request that could never succeed and then blamed the database.
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402

PIT = "Pittsburgh, PA Metro Area"
IND = "Indianapolis-Carmel-Anderson, IN Metro Area"


class StubSqlModel:
    """Counts calls; replies with queued SQL (empty completion when the queue is empty - the failure in the trace)."""

    def __init__(self, replies=()):
        self.replies, self.calls = list(replies), 0

    def invoke(self, messages):
        self.calls += 1
        return types.SimpleNamespace(content=self.replies.pop(0) if self.replies else "")


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("qd"))
    yield f
    census_fixture.deactivate(f)


@pytest.fixture()
def model(fx, monkeypatch):
    import agents.tools as tools
    stub = StubSqlModel()
    monkeypatch.setattr(tools, "_agent", stub)
    return stub


def ask(request, **kwargs):
    import agents.tools as tools
    return tools.query_database.invoke({"request": request, **kwargs})


def result_of(output: str) -> str:
    return output.split("[RESULT]\n", 1)[1]


# --------------------------------------------------------------------------------------------------------------------
# The reported request
# --------------------------------------------------------------------------------------------------------------------
def test_the_reported_request_is_answered_without_asking_the_sql_model(fx, model):
    out = ask("Verify the land areas of Pittsburgh and Indianapolis")
    assert model.calls == 0
    assert out.startswith("[GENERATED SQL]\nDerived tract-area aggregation\n[RESULT]\n")      # the provider's historical label
    body = result_of(out)
    assert not body.startswith("NOT_ANSWERED") and "Code Agent error" not in body
    assert PIT in body and IND in body
    assert f"{fx.expected[PIT]['area_sq_mi']:.2f}" in body and f"{fx.expected[IND]['area_sq_mi']:.2f}" in body
    # the user is told what the figure is (and is not) so a metro-wide area is not mistaken for city limits
    assert "not the city limits" in body and "computed, not read from a stored column" in body
    assert '"pittsburgh" -> ' + PIT in body and '"indianapolis" -> ' + IND in body


def test_derived_answer_is_emitted_as_an_artifact(fx, model):
    from agents.artifacts import collect_artifacts, reset_artifacts
    reset_artifacts()
    ask("Verify the land areas of Pittsburgh and Indianapolis")
    artifacts = collect_artifacts()
    assert len(artifacts) == 1 and artifacts[0]["type"] in {"chart", "table"}


def test_existing_density_questions_keep_their_numbers_and_wording(fx, model):
    body = result_of(ask("population density of Pittsburgh and Indianapolis"))
    assert body.startswith("Density is total tract population divided by total tract polygon area in square miles; "
                           "it is not the average of tract densities.")
    for name in (PIT, IND):
        assert f"{fx.expected[name]['density']:.2f}" in body and f"{fx.expected[name]['area_sq_mi']:.2f}" in body
    assert model.calls == 0


def test_units_follow_the_request(fx, model):
    body = result_of(ask("land area of Pittsburgh in square kilometers"))
    assert "land_area_sq_km" in body and "land_area_sq_mi" not in body


def test_measures_can_be_stated_in_requirements_not_just_the_request(fx, model):
    body = result_of(ask("Pittsburgh", requirements="Return the land area of each metro"))
    assert PIT in body and "land_area_sq_mi" in body and model.calls == 0


def test_combined_questions_run_the_provider_once_with_every_measure(fx, model):
    body = result_of(ask("What are the population, land area and density of Pittsburgh?"))
    assert "population_density" in body and "land_area_sq_mi" in body and "population" in body


def test_metro_without_county_membership_is_a_data_status_with_the_fix(fx, model):
    body = result_of(ask("land area of Denver"))
    assert body.startswith("NOT_ANSWERED[data_unavailable]") and "diagnose_msa" in body
    assert model.calls == 0


def test_a_derived_measure_with_no_place_asks_for_one(fx, model):
    for question in ("land area of the metro", "What is the population density?"):
        body = result_of(ask(question))
        assert body.startswith("NOT_ANSWERED[place_missing]"), question
        assert "not supported" in body and model.calls == 0


# --------------------------------------------------------------------------------------------------------------------
# Plain lookups still use SQL - and survive an empty model completion
# --------------------------------------------------------------------------------------------------------------------
def test_population_lookup_survives_an_empty_model_completion(fx, model):
    out = ask("What is the population of Pittsburgh?")
    assert model.calls == 1
    assert "SELECT census_msa.name, census_msa.population FROM census_msa WHERE census_msa.name = 'Pittsburgh, PA Metro Area'" in out
    assert str(fx.expected[PIT]["population"]) in result_of(out)


def test_population_lookup_uses_the_models_sql_when_it_gives_one(fx, model):
    model.replies = ["SELECT census_msa.name, census_msa.population FROM census_msa WHERE census_msa.name = 'Pittsburgh, PA Metro Area'"]
    out = ask("How many people live in Pittsburgh?")
    assert model.calls == 1 and str(fx.expected[PIT]["population"]) in result_of(out)


def test_state_qualified_names_resolve_the_ambiguity(fx, model):
    out = ask("population of Portland, ME")
    assert not result_of(out).startswith("NOT_ANSWERED")
    assert "Portland-South Portland, ME Metro Area" in out and "Portland-Vancouver" not in out


def test_county_listing_compiles_without_the_model(fx, model):
    out = ask("which counties make up the Pittsburgh metro")
    body = result_of(out)
    assert "Allegheny County" in body and "Beaver County" in body


# --------------------------------------------------------------------------------------------------------------------
# Not-answered paths: no model calls, honest messages
# --------------------------------------------------------------------------------------------------------------------
def test_ambiguous_places_are_never_guessed(fx, model):
    body = result_of(ask("population of Portland"))
    assert body.startswith("NOT_ANSWERED[place_ambiguous]")
    assert "Portland-South Portland, ME Metro Area" in body and "Portland-Vancouver-Hillsboro, OR-WA Metro Area" in body
    assert 'for example "Portland, ME"' in body and "not a database error" in body
    assert model.calls == 0


def test_a_known_unloaded_topic_is_stated_not_estimated(fx, model):
    body = result_of(ask("median household income in Pittsburgh"))
    assert body.startswith("NOT_ANSWERED[topic_unavailable]")
    assert "NOT part of the loaded data" in body and "Never estimate" in body and model.calls == 0


def test_unplannable_requests_do_not_spend_model_calls(fx, model):
    """The original bug: an empty plan can never pass validation, yet the model was asked three times."""
    out = ask("How is the weather on Mars")
    assert model.calls == 0
    body = result_of(out)
    assert body.startswith("NOT_ANSWERED[unplannable]")
    assert "not a database error" in body and "Code Agent error" not in body and "Cannot compile" not in body


def test_unplannable_message_recognises_a_place_and_suggests_measures(fx, model):
    body = result_of(ask("tell me about Pittsburgh"))
    assert body.startswith("NOT_ANSWERED[unplannable]") and "A place was recognized" in body and "Pittsburgh, PA Metro Area" in body
    assert "land area" in body or "population" in body.lower()


def test_unplannable_message_reports_empty_name_tables(fx, model):
    import db.duckdb_store as store
    conn = store.get_conn()
    saved = conn.execute("SELECT * FROM census_msa").df()
    conn.execute("DELETE FROM census_msa")
    try:
        body = result_of(ask("population of Pittsburgh"))
    finally:
        conn.register("_saved", saved)
        conn.execute("INSERT INTO census_msa SELECT * FROM _saved")
        conn.unregister("_saved")
    assert body.startswith("NOT_ANSWERED[unplannable]") and "census_msa" in body and "no rows" in body


def test_mixed_requests_report_what_the_derived_call_did_not_compute(fx, model):
    body = result_of(ask("land area of Pittsburgh and average NRI risk"))
    assert "land_area_sq_mi" in body and "Not computed by this call" in body and "overall NRI risk" in body


def test_partial_topics_are_flagged_after_a_real_sql_answer(fx, model):
    model.replies = ["SELECT AVG(houses.price) FROM houses WHERE houses.city = 'Pittsburgh'"]
    body = result_of(ask("average list price of houses in Pittsburgh by median household income"))
    assert "NOTE (not part of the query result)" in body and "NOT part of the loaded data" in body


def test_house_questions_are_untouched(fx, model):
    model.replies = ["SELECT COUNT(houses.house_id) FROM houses WHERE houses.city = 'Pittsburgh'"]
    out = ask("How many houses are in Pittsburgh?")
    assert model.calls == 1 and "NOT_ANSWERED" not in out and "NOTE (not part" not in out


# --------------------------------------------------------------------------------------------------------------------
# Contracts the rest of the app relies on
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("question", ["population of Portland", "median household income in Pittsburgh",
                                      "How is the weather on Mars", "land area of the metro", "tell me about Pittsburgh"])
def test_guidance_messages_contain_no_digits(fx, model, question):
    """A guidance sentence must never be mistaken for data by the digit-based validator heuristics."""
    from agents import answer_status
    body = result_of(ask(question))
    assert answer_status.is_not_answered(body)
    assert not re.search(r"\d", body)


def test_validator_never_treats_a_non_answer_as_data(fx, model):
    from agents.response_validator import _tool_returned_real_data
    for question in ("population of Portland", "land area of Denver", "How is the weather on Mars"):
        assert _tool_returned_real_data([("query_database", ask(question))]) is False, question
    assert _tool_returned_real_data([("query_database", ask("land area of Pittsburgh"))]) is True


def test_an_unexpected_provider_failure_keeps_the_legacy_error_contract(fx, model, monkeypatch):
    """Anything that is not a DerivedError must come back as 'Code Agent error: ...' (which the orchestrator retries),
    never as an exception that tears down the turn - exactly what the old density branch did."""
    from services import census_metrics

    def boom(*a, **k):
        raise RuntimeError("geometry exploded")

    monkeypatch.setattr(census_metrics.PROVIDER, "compute", boom)
    body = result_of(ask("land area of Pittsburgh"))
    assert body == "Code Agent error: geometry exploded"
    from agents import answer_status
    assert not answer_status.is_not_answered(body) and model.calls == 0


def test_a_supporting_sql_concept_is_not_reported_as_skipped(fx, model):
    """'MSA population' is supplied by the provider whenever a derived call is already running."""
    body = result_of(ask("MSA population and land area of Pittsburgh"))
    assert "Not computed by this call" not in body
    assert "population" in body and "land_area_sq_mi" in body
