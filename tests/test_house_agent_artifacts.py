"""
Tests for artifact emission from house-agent approved functions and from the
check_data_availability general tool.

Issue #5 requested that every tool that produces structured data emit it as
a visual artifact (map / chart / table) where appropriate.  The new tests
here cover the house-agent code paths (agents/house_agent.py::make_house_approved_functions)
and the check_data_availability general tool, which are not reached by
test_artifacts.py (which focuses on classify_dataframe logic and the two
general-agent-facing wrappers query_database / find_bike_route).

All tests use the `reference_data` fixture from conftest.py so they run
against a real DuckDB schema with the same dtypes the production code sees.
"""
import pandas as pd
import pytest

from agents import artifacts as A


# ── Helpers ──────────────────────────────────────────────────────────────────


def _nri_row(conn, tract_fips: str) -> bool:
    """Return True if nri_tracts has a row for the given tract."""
    result = conn.execute(
        "SELECT COUNT(*) FROM nri_tracts WHERE tract_fips = ?", [tract_fips]
    ).fetchone()
    return result[0] > 0


# ── get_nri_risk_data — artifact emission ─────────────────────────────────────


def test_get_nri_risk_data_emits_chart_artifact(reference_data):
    """get_nri_risk_data() must emit a bar-chart artifact for the top hazards."""
    from agents.house_agent import make_house_approved_functions

    assert _nri_row(reference_data, "42003040100"), "fixture missing NRI row"

    A.reset_artifacts()
    funcs = make_house_approved_functions("h1")
    result = funcs["get_nri_risk_data"]()

    produced = A.collect_artifacts()
    assert len(produced) == 1, f"expected 1 artifact, got {len(produced)}"
    art = produced[0]
    assert art["type"] == "chart"
    assert art["chart_kind"] == "bar"
    # The title should call out what the chart shows
    assert "hazard" in (art.get("title") or "").lower()


def test_get_nri_risk_data_chart_categories_are_hazard_names(reference_data):
    """The chart categories must be natural-language hazard names, not column codes."""
    from agents.house_agent import make_house_approved_functions

    A.reset_artifacts()
    funcs = make_house_approved_functions("h1")
    funcs["get_nri_risk_data"]()

    art = A.collect_artifacts()[0]
    assert art["categories"], "chart categories must not be empty"
    for cat in art["categories"]:
        # Category must be a natural-language string, not a raw column code
        assert cat and not cat.startswith("rfld_"), f"unexpected raw category: {cat!r}"


def test_get_nri_risk_data_no_artifact_when_no_nri_row(reference_data):
    """
    If the house has a tract_fips that has no NRI row, get_nri_risk_data must
    return an error string and NOT emit an artifact.
    """
    from agents.house_agent import make_house_approved_functions

    # Insert a house whose tract has no NRI row
    reference_data.execute(
        "INSERT INTO houses (house_id, address, city, state, zip, tract_fips, lat, lon, status, price) "
        "VALUES ('h_no_nri', '1 Ghost St', 'Nowhere', 'XX', '00000', '99999000000', 40.0, -80.0, 'Active', 100000)"
    )

    A.reset_artifacts()
    funcs = make_house_approved_functions("h_no_nri")
    result = funcs["get_nri_risk_data"]()

    assert "no nri" in result.lower() or "not found" in result.lower()
    assert A.collect_artifacts() == []


def test_get_nri_risk_data_no_artifact_when_house_missing(reference_data):
    """A house that doesn't exist at all must not crash and must not emit an artifact."""
    from agents.house_agent import make_house_approved_functions

    A.reset_artifacts()
    funcs = make_house_approved_functions("nonexistent-house-id")
    result = funcs["get_nri_risk_data"]()

    assert "not available" in result.lower() or "not found" in result.lower()
    assert A.collect_artifacts() == []


# ── estimate_price_with_code — artifact emission ─────────────────────────────


def test_estimate_price_emits_table_artifact_for_similar_listings(reference_data):
    """
    When there are other houses in the same city, estimate_price_with_code
    should emit a table artifact for the 'similar active listings' result.
    """
    from agents.house_agent import make_house_approved_functions

    # Seed a second Pittsburgh house so city-level comps exist
    reference_data.execute(
        "INSERT INTO houses (house_id, address, city, state, zip, tract_fips, lat, lon, status, price, sqft) "
        "VALUES ('h_dup', '200 Other St', 'Pittsburgh', 'PA', '15213', '42003040100', "
        "40.44, -80.0, 'Active', 260000, 1200)"
    )

    A.reset_artifacts()
    funcs = make_house_approved_functions("h1")
    result = funcs["estimate_price_with_code"]()

    produced = A.collect_artifacts()
    table_arts = [a for a in produced if a["type"] == "table"]
    assert table_arts, "expected at least one table artifact for similar listings"


def test_estimate_price_no_crash_when_no_comparables(reference_data):
    """
    If there are no sold comps and no city peers, estimate_price_with_code may
    return a string result and emit no artifacts — it must never crash.
    """
    from agents.house_agent import make_house_approved_functions

    # Cleveland house (h6) has no sold comps and only one listing in its city
    A.reset_artifacts()
    funcs = make_house_approved_functions("h6")
    result = funcs["estimate_price_with_code"]()

    # No crash — result must be a string
    assert isinstance(result, str)
    A.collect_artifacts()  # confirm collect_artifacts doesn't raise


# ── get_nearby_sold_homes — artifact emission ─────────────────────────────────


def test_get_nearby_sold_homes_emits_table_artifact(reference_data):
    """
    get_nearby_sold_homes() must emit a table artifact when sold comps are present.
    """
    from agents.house_agent import make_house_approved_functions

    # h1's tract (42003040100) has a sold home from the conftest fixture
    A.reset_artifacts()
    funcs = make_house_approved_functions("h1")
    result = funcs["get_nearby_sold_homes"]()

    produced = A.collect_artifacts()
    assert len(produced) == 1
    art = produced[0]
    assert art["type"] == "table"
    assert "sold" in (art.get("title") or "").lower()


def test_get_nearby_sold_homes_no_artifact_when_no_comps(reference_data):
    """
    When there are no sold homes in the tract, the function must not emit an artifact.
    """
    from agents.house_agent import make_house_approved_functions

    # h4's tract (42101000100) has no sold homes in the fixture
    A.reset_artifacts()
    funcs = make_house_approved_functions("h4")
    result = funcs["get_nearby_sold_homes"]()

    assert "no sold homes" in result.lower() or "not found" in result.lower()
    assert A.collect_artifacts() == []


# ── check_data_availability — artifact emission ───────────────────────────────


def test_check_data_availability_emits_table_artifact(reference_data):
    """
    check_data_availability must emit a 'table' artifact listing row counts.
    """
    import agents.tools as tools

    A.reset_artifacts()
    result = tools.check_data_availability.invoke("")
    produced = A.collect_artifacts()

    assert len(produced) == 1
    art = produced[0]
    assert art["type"] == "table"
    col_keys = [c["key"] for c in art["columns"]]
    assert "table" in col_keys
    assert "rows" in col_keys


def test_check_data_availability_table_includes_core_tables(reference_data):
    """The availability table must include all core tables (houses, nri_tracts, etc.)."""
    import agents.tools as tools

    A.reset_artifacts()
    tools.check_data_availability.invoke("")
    art = A.collect_artifacts()[0]

    table_names = {row["table"] for row in art["rows"]}
    assert "houses" in table_names
    assert "nri_tracts" in table_names
    assert "sold_homes" in table_names


# ── Artifact bus isolation ────────────────────────────────────────────────────


def test_artifacts_isolated_between_house_turns(reference_data):
    """
    Artifacts from two separate house turns must not bleed into each other —
    reset_artifacts() wipes the slate between turns.
    """
    from agents.house_agent import make_house_approved_functions

    funcs = make_house_approved_functions("h1")

    # Turn 1
    A.reset_artifacts()
    funcs["get_nri_risk_data"]()
    turn1_artifacts = A.collect_artifacts()
    assert len(turn1_artifacts) == 1

    # Turn 2 — reset_artifacts is called again as run_house_chat would do
    A.reset_artifacts()
    assert A.collect_artifacts() == [], "artifacts were not cleared between turns"


def test_multiple_house_functions_accumulate_artifacts(reference_data):
    """
    Within a single turn, multiple approved functions may each emit an artifact;
    collect_artifacts returns them all in emission order.
    """
    from agents.house_agent import make_house_approved_functions

    # Seed a city peer so estimate_price emits its listing table
    reference_data.execute(
        "INSERT INTO houses (house_id, address, city, state, zip, tract_fips, lat, lon, status, price) "
        "VALUES ('h_peer', '5 Elm Peer', 'Pittsburgh', 'PA', '15213', '42003040100', "
        "40.44, -80.0, 'Active', 280000)"
    )

    A.reset_artifacts()
    funcs = make_house_approved_functions("h1")

    funcs["get_nri_risk_data"]()      # emits 1 chart
    funcs["get_nearby_sold_homes"]()  # emits 1 table

    produced = A.collect_artifacts()
    assert len(produced) >= 2
    types = [a["type"] for a in produced]
    assert "chart" in types
    assert "table" in types


# ── run_house_chat public contract ────────────────────────────────────────────


def test_run_house_chat_returns_three_tuple(reference_data, monkeypatch):
    """
    run_house_chat must return (reply_str, history_list, artifacts_list).
    history must include both the user and assistant turns.
    artifacts must be a list (possibly empty).
    """
    from agents import house_agent

    # Stub LLM calls so no real server is required
    class _FakeCodeLLM:
        def invoke(self, messages):
            class _R:
                content = "final_result = get_house_details()"
            return _R()

    class _FakeResponseLLM:
        def invoke(self, messages):
            class _R:
                content = "This is a stub answer about the house."
            return _R()

    monkeypatch.setattr(house_agent, "_get_house_code_agent", lambda: _FakeCodeLLM())
    monkeypatch.setattr(house_agent, "_get_house_response_agent", lambda: _FakeResponseLLM())

    # Stub guardrails to always pass
    from services.guardrails import GuardrailManager, GuardrailResult
    monkeypatch.setattr(
        GuardrailManager, "inspect_turn_input",
        staticmethod(lambda msg: GuardrailResult(passed=True, sanitized_text=msg, reasons=[]))
    )
    monkeypatch.setattr(
        house_agent, "validate_response",
        lambda reply, *a, **kw: reply
    )

    reply, history, artifacts = house_agent.run_house_chat(
        "h1", "Tell me about this house."
    )

    assert isinstance(reply, str) and reply
    assert isinstance(history, list)
    assert any(m.get("role") == "user" for m in history)
    assert any(m.get("role") == "assistant" for m in history)
    assert isinstance(artifacts, list)
