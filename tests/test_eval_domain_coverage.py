"""Ground truth for the crime / bike / transit / commute examples (recommendation 10).

Two independent things are checked, neither of which needs a model:

1. Every constant the new golden examples assert is the value a real SQL aggregate
   over the fixture returns — so a wrong expectation cannot silently mark a correct
   answer as failing.
2. Each question reaches the intended table through the deterministic query planner,
   so the examples are answerable with the tools the agent actually has.
"""

from __future__ import annotations

import db.duckdb_store as store
import pytest
from config import settings
from eval import fixtures
from eval.fixtures import build_fixture
from eval.golden_set import get_example


@pytest.fixture(scope="module")
def fixture_conn(tmp_path_factory):
    previous = settings.duckdb_path
    path = tmp_path_factory.mktemp("eval_fixture") / "real_estate.duckdb"
    try:
        build_fixture(path)
        yield store.get_conn()
    finally:
        store.close()
        settings.duckdb_path = previous


def scalar(conn, sql: str):
    return conn.execute(sql).fetchone()[0]


def test_crime_constants_match_sql(fixture_conn):
    conn = fixture_conn
    assert (
        scalar(conn, "SELECT COUNT(*) FROM crime_incidents WHERE city = 'pittsburgh'")
        == fixtures.PITTSBURGH_CRIME_COUNT
    )
    assert scalar(conn, "SELECT COUNT(*) FROM crime_incidents") == fixtures.TOTAL_CRIME_COUNT
    assert scalar(conn, "SELECT COUNT(DISTINCT city) FROM crime_incidents") == fixtures.CRIME_CITY_COUNT
    assert scalar(conn, "SELECT AVG(severity_weight) FROM crime_incidents") == pytest.approx(
        fixtures.AVG_CRIME_SEVERITY_WEIGHT, abs=0.01
    )
    assert scalar(conn, "SELECT SUM(severity_weight) FROM crime_incidents WHERE city = 'miami'") == pytest.approx(
        fixtures.MIAMI_CRIME_SEVERITY_TOTAL, abs=0.01
    )


def test_crime_severity_comes_from_the_taxonomy(fixture_conn):
    """Weights in the fixture must equal what the loader would have assigned."""
    rows = fixture_conn.execute(
        "SELECT raw_type, category, severity_weight FROM crime_incidents ORDER BY incident_id"
    ).fetchall()
    from services.crime_taxonomy import classify_crime_text

    assert rows
    for raw_type, category, weight in rows:
        expected_category, _label, expected_weight = classify_crime_text(raw_type)
        assert (category, weight) == (expected_category, expected_weight)


def test_bike_and_transit_constants_match_sql(fixture_conn):
    conn = fixture_conn
    assert (
        scalar(conn, "SELECT COUNT(*) FROM bike_routes WHERE city = 'pittsburgh'")
        == fixtures.PITTSBURGH_BIKE_ROUTE_COUNT
    )
    assert scalar(conn, "SELECT COUNT(*) FROM bike_routes") == fixtures.TOTAL_BIKE_ROUTE_COUNT
    assert scalar(conn, "SELECT COUNT(DISTINCT ntd_id) FROM transit_agencies") == fixtures.TRANSIT_AGENCY_COUNT
    assert scalar(conn, "SELECT SUM(upt) FROM transit_ridership") == pytest.approx(
        fixtures.TOTAL_TRANSIT_RIDERSHIP, abs=0.01
    )
    top = conn.execute(
        "SELECT agency FROM transit_ridership GROUP BY agency ORDER BY SUM(upt) DESC LIMIT 1"
    ).fetchone()[0]
    assert top == fixtures.TOP_RIDERSHIP_AGENCY
    assert scalar(conn, "SELECT COUNT(*) FROM transit_routes") == fixtures.TRANSIT_ROUTE_COUNT


def test_commute_constants_match_sql(fixture_conn):
    conn = fixture_conn
    assert scalar(conn, "SELECT AVG(drive_min) FROM house_commute") == pytest.approx(
        fixtures.AVG_DRIVE_TIME_TO_WORK, abs=0.01
    )
    assert scalar(conn, "SELECT MIN(bike_min) FROM house_commute") == pytest.approx(
        fixtures.SHORTEST_BIKE_COMMUTE, abs=0.01
    )


@pytest.mark.parametrize(
    "example_id, table",
    [
        ("crime_incident_count_pittsburgh", "crime_incidents"),
        ("crime_city_count", "crime_incidents"),
        ("crime_avg_severity_weight", "crime_incidents"),
        ("crime_miami_severity_total", "crime_incidents"),
        ("bike_route_count_pittsburgh", "bike_routes"),
        ("bike_route_count_loaded", "bike_routes"),
        ("transit_agency_count", "transit_agencies"),
        ("transit_ridership_total", "transit_ridership"),
        ("transit_top_ridership_agency", "transit_ridership"),
        ("transit_route_count", "transit_routes"),
        ("commute_avg_drive_time", "house_commute"),
        ("commute_shortest_bike_ride", "house_commute"),
    ],
)
def test_question_routes_to_its_table(example_id, table):
    from agents.query_planner import build_query_plan

    example = get_example(example_id)
    plan = build_query_plan(example.question)
    assert table in plan.required_tables, f"{example_id!r} planned against {plan.required_tables}"


def test_new_examples_carry_domain_tags():
    tags = {
        "crime": {
            "crime_incident_count_pittsburgh",
            "crime_city_count",
            "crime_avg_severity_weight",
            "crime_miami_severity_total",
        },
        "bike": {"bike_route_count_pittsburgh", "bike_route_count_loaded"},
        "transit": {
            "transit_agency_count",
            "transit_ridership_total",
            "transit_top_ridership_agency",
            "transit_route_count",
        },
        "commute": {"commute_avg_drive_time", "commute_shortest_bike_ride"},
    }
    for tag, ids in tags.items():
        for example_id in ids:
            assert tag in get_example(example_id).tags, f"{example_id!r} is missing tag {tag!r}"
