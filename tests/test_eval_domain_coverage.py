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


def test_remaining_capability_constants_and_routes(fixture_conn):
    from agents.query_planner import build_query_plan

    conn = fixture_conn
    assert scalar(conn, "SELECT AVG(home_value) FROM zhvi WHERE region_type = 'zip'") == pytest.approx(
        fixtures.AVG_ZIP_ZHVI, abs=0.01
    )
    assert scalar(
        conn, "SELECT heat_index FROM market_heat_index WHERE region_type = 'metro' AND msa_code = '38300'"
    ) == pytest.approx(fixtures.PITTSBURGH_MARKET_HEAT_INDEX, abs=0.01)
    assert (
        scalar(conn, "SELECT SUM(incidents) FROM crime_city_year WHERE city = 'pittsburgh'")
        == fixtures.PITTSBURGH_CRIME_COUNT
    )
    assert (
        scalar(conn, "SELECT agency FROM transit_safety ORDER BY event_total DESC LIMIT 1")
        == fixtures.TOP_TRANSIT_SAFETY_AGENCY
    )
    assert (
        scalar(conn, "SELECT COUNT(DISTINCT route_uid) FROM transit_route_agencies WHERE ntd_id = '30001'")
        == fixtures.TRANSIT_ROUTES_BY_AGENCY_COUNT
    )
    assert scalar(conn, "SELECT land_area_sq_mi FROM zcta_boundaries WHERE zcta5 = '15213'") == pytest.approx(
        fixtures.ZCTA_15213_LAND_AREA, abs=0.01
    )
    assert scalar(
        conn, "SELECT land_area_sq_mi FROM place_boundaries WHERE short_name = 'Pittsburgh'"
    ) == pytest.approx(fixtures.PITTSBURGH_PLACE_LAND_AREA, abs=0.01)
    assert (
        scalar(conn, "SELECT COUNT(DISTINCT route_uid) FROM transit_route_zctas WHERE zcta5 = '15213'")
        == fixtures.TRANSIT_ROUTES_IN_15213
    )
    assert (
        scalar(conn, "SELECT COUNT(DISTINCT route_uid) FROM transit_route_places WHERE place_geoid = '4261000'")
        == fixtures.TRANSIT_ROUTES_IN_PITTSBURGH_PLACE
    )
    assert (
        scalar(conn, "SELECT COUNT(DISTINCT route_id) FROM bike_route_tracts WHERE tract_fips = '42003140300'")
        == fixtures.BIKE_ROUTES_IN_PGH_TRACT
    )
    assert (
        scalar(conn, "SELECT COUNT(DISTINCT route_id) FROM bike_route_zctas WHERE zcta5 = '15213'")
        == fixtures.BIKE_ROUTES_IN_15213
    )
    assert (
        scalar(conn, "SELECT COUNT(DISTINCT route_id) FROM bike_route_places WHERE place_geoid = '4261000'")
        == fixtures.BIKE_ROUTES_IN_PITTSBURGH_PLACE
    )

    for example_id, expected_table in [
        ("house_snapshot_history_pittsburgh", "house_snapshots"),
        ("zhvi_zip_avg_home_value", "zhvi"),
        ("market_heat_index_pittsburgh", "market_heat_index"),
        ("crime_city_summary_pittsburgh", "crime_city_year"),
        ("transit_top_safety_events_agency", "transit_safety"),
        ("transit_routes_by_agency_count", "transit_route_agencies"),
        ("zcta_15213_land_area", "zcta_boundaries"),
        ("place_pittsburgh_land_area", "place_boundaries"),
        ("transit_routes_in_zcta_15213", "transit_route_zctas"),
        ("transit_routes_in_place_pittsburgh", "transit_route_places"),
        ("bike_routes_in_pgh_tract", "bike_route_tracts"),
        ("bike_routes_in_zcta_15213", "bike_route_zctas"),
        ("bike_routes_in_place_pittsburgh", "bike_route_places"),
    ]:
        plan = build_query_plan(get_example(example_id).question)
        assert expected_table in plan.required_tables, (example_id, plan.required_tables)


def test_sold_comp_exclusion_count_and_provenance(fixture_conn):
    import db.duckdb_store as store
    import db.schema_catalog as schema

    stats = store.get_price_stats_in_tract("42003140300")
    assert int(stats["sold_sold_count"]) == 1
    assert int(stats["sold_non_arms_length_excluded"]) == 1

    prov = schema.table_provenance("nri_tracts")
    assert prov["source_key"] == "nri" and prov["vintage"] and prov["licence"]
    ctx = schema.build_query_context("What is the riverine flood risk for the Pittsburgh metro?")
    assert "DATA PROVENANCE" in ctx and "FEMA National Risk Index" in ctx
