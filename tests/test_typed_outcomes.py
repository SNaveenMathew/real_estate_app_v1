"""The typed outcome contract: one finite vocabulary, one plan-state set, one retry predicate.

These tests are about completeness, not about particular questions: every code must be categorised, labelled,
renderable, parseable, and correctly classified from legacy text, and every plan failure mode must have its own state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agents import answer_status as st
from agents import plan_status, tool_result


def test_vocabulary_is_a_partition():
    assert set(st.CATEGORY) == st.CODES
    assert set(st.LABELS) == st.CODES
    assert st.RETRYABLE <= st.CODES
    assert set(st.LIMITATION_TEXT) <= st.CODES
    # every code belongs to exactly one category (a dict cannot repeat a key, but the categories must be meaningful)
    assert set(st.CATEGORY.values()) == {
        "planning",
        "place",
        "data",
        "sql",
        "routing",
        "limits",
        "model",
        "safety",
        "privacy",
        "clarification",
        "internal",
    }


@pytest.mark.parametrize("code", sorted(st.CODES))
def test_every_code_round_trips(code):
    line = st.not_answered(code, "a plain message")
    assert st.status_code(line) == code
    assert st.status_message(line) == "a plain message"
    assert st.is_not_answered(line)
    outcome = st.outcome(code)
    assert outcome["status"] == code and outcome["category"] == st.CATEGORY[code]
    assert outcome["retryable"] is (code in st.RETRYABLE)


@pytest.mark.parametrize("code", sorted(st.LIMITATION_TEXT))
def test_limitation_text_is_user_safe(code):
    text = st.LIMITATION_TEXT[code]
    assert not any(ch.isdigit() for ch in text), text


def test_unknown_code_is_rejected():
    with pytest.raises(ValueError):
        st.not_answered("definitely_not_a_code", "x")


def test_success_is_not_a_marker():
    assert st.outcome(None)["status"] == st.OK
    assert st.outcome(None)["category"] == "ok"
    assert st.classify_text("") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("SQL Error: Binder Error: referenced table houses", st.SQL_EXECUTION_ERROR),
        ("Code Agent error: Cannot compile SQL: plan has no operation.", st.PLAN_INCOMPLETE),
        ("Code Agent error: boom", st.INTERNAL_ERROR),
        ("The query executed successfully and returned 0 rows", st.RESULT_EMPTY),
        ("Only read-only SELECT SQL is allowed.", st.SQL_INVALID),
        ("scope must be 'this_house' or 'all_houses'", st.INVALID_ARGUMENT),
        ("model is unavailable right now", st.MODEL_UNAVAILABLE),
    ],
)
def test_legacy_text_is_classified(text, expected):
    assert st.classify_text(text) == expected


def test_real_evidence_is_not_classified_as_failure():
    data = "address        price\n1 Main St     250000\n2 Oak Ave     310000"
    assert st.classify_text(data) is None


def test_classify_error_names_exceptions():
    assert st.classify_error(ValueError("Only read-only SELECT SQL is allowed.")) == st.SQL_INVALID

    class Coded(RuntimeError):
        code = st.PLAN_INCONSISTENT

    assert st.classify_error(Coded("x")) == st.PLAN_INCONSISTENT
    assert st.classify_error(RuntimeError("nothing recognisable")) == st.INTERNAL_ERROR


# ── Plan states ──────────────────────────────────────────────────────────────


@dataclass
class _Plan:
    required_tables: list = field(default_factory=list)
    operation: str | None = None
    select_expression: str | None = None
    ordering: str | None = None
    grouping: list = field(default_factory=list)
    filters: list = field(default_factory=list)
    entity_filters: list = field(default_factory=list)
    scope_filters: list = field(default_factory=list)
    rollup_spec: dict = field(default_factory=dict)


def test_no_table_matched_is_its_own_state():
    report = plan_status.report(_Plan())
    assert report.state is plan_status.PlanState.NO_MATCH
    assert report.code == st.UNPLANNABLE


def test_table_without_operation_is_its_own_state():
    report = plan_status.report(_Plan(required_tables=["houses"]))
    assert report.state is plan_status.PlanState.NO_OPERATION
    assert report.missing_operation and report.code == st.PLAN_INCOMPLETE
    assert "nothing was estimated" in report.message()


def test_projection_of_an_unplanned_table_is_its_own_state():
    plan = _Plan(required_tables=["house_commute"], operation="rank", select_expression="houses.address, x")
    report = plan_status.report(plan)
    assert report.state is plan_status.PlanState.INCONSISTENT
    assert report.unplanned_tables == ("houses",)
    assert report.code == st.PLAN_INCONSISTENT


def test_unjoinable_tables_are_their_own_state():
    plan = _Plan(required_tables=["a", "b"], operation="avg")
    assert plan_status.report(plan, relationship_path=lambda t: []).state is plan_status.PlanState.UNCONNECTED
    assert plan_status.report(plan, relationship_path=lambda t: [1]).state is plan_status.PlanState.OK


def test_ok_plan_has_no_message():
    plan = _Plan(required_tables=["a"], operation="count", select_expression="a.id")
    report = plan_status.report(plan)
    assert report.state is plan_status.PlanState.OK and report.message() == ""
    assert report.to_dict()["status"] == st.OK


# ── The invariant the planner repairs ────────────────────────────────────────


def test_planner_includes_tables_named_by_the_projection():
    from agents.query_planner import build_query_plan

    plan = build_query_plan("What is the shortest bike commute?")
    # the projection names houses.address, so houses must be in the plan (it used to be omitted -> binder error)
    assert "house_commute" in plan.required_tables
    assert "houses" in plan.required_tables
    assert not plan_status.projected_tables(plan) - set(plan.required_tables)


# ── Typed tool results ───────────────────────────────────────────────────────


def test_envelope_reads_the_shared_vocabulary():
    from agents import tools

    data = tools._envelope("address   price\n1 Main   250000")
    assert data.kind == "data" and data.ok and data.status == st.OK

    refusal = tools._envelope(st.not_answered(st.PLACE_MISSING, "name a place"))
    assert refusal.kind == "not_answered" and not refusal.ok and refusal.status == st.PLACE_MISSING

    broken = tools._envelope("SQL Error: Binder Error: nope")
    assert broken.kind == "error" and broken.status == st.SQL_EXECUTION_ERROR


def test_route_kinds_map_onto_the_vocabulary():
    from agents import tools

    assert set(tools._ROUTE_CODES) == {"no_route", "no_data", "input", "routing_service"}
    assert set(tools._ROUTE_CODES.values()) <= st.CODES


def test_shared_retry_predicate():
    assert tool_result.result_failed("The query executed successfully and returned 0 rows")
    assert tool_result.result_failed("SQL Error: Binder Error: nope")
    assert not tool_result.result_failed("address   price\n1 Main   250000")
    assert not tool_result.result_failed(tool_result.ToolResult("x", kind="data", status=st.OK))
    assert tool_result.result_failed(tool_result.ToolResult("x", ok=False, kind="error", status=st.SQL_INVALID))
    # a refusal the user must act on is not retried inside the turn
    assert not tool_result.result_failed(
        tool_result.ToolResult("x", ok=False, kind="not_answered", status=st.PLACE_AMBIGUOUS)
    )
