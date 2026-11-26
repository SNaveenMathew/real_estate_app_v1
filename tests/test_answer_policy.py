"""Pre-SQL answer policy: a dataset level that is not loaded must be reported, never guessed."""

from __future__ import annotations

from datetime import date

from agents import answer_policy, answer_status
from agents.query_planner import build_query_plan


def _insert_zhvi(conn, region_type="zip", region_name="09170", value=250000.0):
    conn.execute(
        "INSERT INTO zhvi (region_id, region_type, region_name, date, home_value) VALUES (?, ?, ?, ?, ?)",
        ["1", region_type, region_name, "2019-01-31", value],
    )


def test_unloaded_level_is_reported_before_sql(fresh_db):
    """zhvi holds zip rows only: a metro question must name the missing level, not return a NULL aggregate."""
    _insert_zhvi(fresh_db.get_conn())

    plan = build_query_plan("home value by metro")
    out = answer_policy.route_before_sql(plan, "home value by metro")

    assert out is not None
    assert answer_status.is_not_answered(out) and answer_status.DATA_UNAVAILABLE in out
    detail = out.split("\n", 1)[1]
    assert "zhvi" in detail and "region_type" in detail and "'metro'" in detail
    assert "zip" in detail, "the message must name the level that IS loaded"
    # A guidance message must never read as data to the validator's digit heuristic.
    assert not any(ch.isdigit() for ch in detail)


def test_loaded_level_is_planned_exactly_as_before(fresh_db):
    _insert_zhvi(fresh_db.get_conn())

    assert answer_policy.availability_gap(build_query_plan("typical home value by zip")) is None


def test_an_empty_dataset_is_reported_as_empty(fresh_db):
    detail = answer_policy.availability_gap(build_query_plan("home value by metro"))

    assert detail and "zhvi" in detail and "no rows" in detail
    assert not any(ch.isdigit() for ch in detail)


def test_query_database_never_reaches_sql_for_an_unloaded_level(fresh_db):
    """The user-visible path: a typed non-answer, so the validator no longer treats it as grounded data."""
    from agents import tools

    _insert_zhvi(fresh_db.get_conn())

    out = tools.query_database.invoke({"request": "home value by metro"})

    assert answer_status.is_not_answered(str(out))
    assert answer_status.DATA_UNAVAILABLE in str(out)
    assert "'metro'" in str(out)
    assert getattr(out, "kind", None) == "not_answered"

# ---------------------------------------------------------------------------
# Monthly datasets: which month is the question about?
# ---------------------------------------------------------------------------
def _insert_months(conn, months, region_type="zip"):
    for i, month in enumerate(months):
        conn.execute(
            "INSERT INTO zhvi (region_id, region_type, region_name, date, home_value) VALUES (?, ?, ?, ?, ?)",
            ["1", region_type, "15203", date(month[0], month[1], 28), 200000.0 + i * 1000],
        )


def test_a_monthly_question_without_a_period_asks_current_or_explicit(fresh_db):
    _insert_months(fresh_db.get_conn(), [(2020, 1), (2020, 2), (2020, 3)])

    plan = build_query_plan("typical home value by zip")
    out = answer_policy.route_before_sql(plan, "typical home value by zip")

    assert out and answer_status.NEEDS_CLARIFICATION in out
    assert "current" in out and "2020-03" in out and "2020-01" in out
    assert not [f for f in plan.filters if ".date" in f], "the plan must not be pinned before the user chooses"


def test_current_pins_the_latest_loaded_month(fresh_db):
    _insert_months(fresh_db.get_conn(), [(2020, 1), (2020, 2), (2020, 3)])

    plan = build_query_plan("current typical home value by zip")
    assert answer_policy.route_before_sql(plan, "current typical home value by zip") is None
    assert "zhvi.date = DATE '2020-03-28'" in plan.filters


def test_an_explicit_month_is_pinned_to_that_month(fresh_db):
    _insert_months(fresh_db.get_conn(), [(2020, 1), (2020, 2), (2020, 3)])

    plan = build_query_plan("typical home value by zip in February 2020")
    assert answer_policy.route_before_sql(plan, "typical home value by zip in February 2020") is None
    assert "zhvi.date >= DATE '2020-02-01'" in plan.filters
    assert "zhvi.date <= DATE '2020-02-29'" in plan.filters


def test_an_explicit_year_covers_that_year(fresh_db):
    _insert_months(fresh_db.get_conn(), [(2020, 1), (2020, 2), (2020, 3)])

    plan = build_query_plan("typical home value by zip in 2020")
    assert answer_policy.route_before_sql(plan, "typical home value by zip in 2020") is None
    assert "zhvi.date >= DATE '2020-01-01'" in plan.filters and "zhvi.date <= DATE '2020-12-31'" in plan.filters


def test_a_period_outside_the_loaded_months_is_a_typed_non_answer(fresh_db):
    _insert_months(fresh_db.get_conn(), [(2020, 1), (2020, 2)])

    plan = build_query_plan("typical home value by zip in 1999")
    out = answer_policy.route_before_sql(plan, "typical home value by zip in 1999")

    assert out and answer_status.DATE_OUT_OF_RANGE in out
    assert "1999" in out and "2020-01" in out and "2020-02" in out
    assert not [f for f in plan.filters if ".date" in f]


def test_a_trend_question_is_not_pinned_to_one_month(fresh_db):
    _insert_months(fresh_db.get_conn(), [(2020, 1), (2020, 2), (2020, 3)])

    plan = build_query_plan("home value trend by zip")
    assert answer_policy.route_before_sql(plan, "home value trend by zip") is None
    assert not [f for f in plan.filters if "zhvi.date" in f]
