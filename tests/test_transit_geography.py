"""FTA route-to-area bridges: line/boundary membership, fractions, and catalog exposure."""

from __future__ import annotations

import json

from shapely.geometry import LineString, box

from services.loaders import tiger, transit


def _place_gdf():
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {
            "STATEFP": ["42"],
            "PLACEFP": ["61000"],
            "GEOID": ["4261000"],
            "NAME": ["Pittsburgh city"],
            "NAMELSAD": ["Pittsburgh city"],
            "LSAD": ["25"],
            "CLASSFP": ["C1"],
            "FUNCSTAT": ["A"],
            "ALAND": [143_000_000],
            "AWATER": [2_000_000],
            "INTPTLAT": ["40.4398"],
            "INTPTLON": ["-79.9764"],
        },
        geometry=[box(-80.02, 40.40, -79.92, 40.48)],
        crs="EPSG:4269",
    )


def _insert_route(conn, route_uid: str, line: LineString) -> None:
    conn.execute(
        "INSERT INTO transit_routes (route_uid, route_id, geometry_json) VALUES (?, ?, ?)",
        [route_uid, route_uid, json.dumps(line.__geo_interface__)],
    )


def test_transit_route_place_bridge_counts_positive_length_and_fraction(fresh_db):
    import db.duckdb_store as store

    conn = fresh_db.get_conn()
    _insert_route(conn, "r1", LineString([(-80.05, 40.44), (-79.99, 40.44)]))
    boundaries = tiger.normalize_place_frame(_place_gdf(), "tl_2026_42_place.shp")
    tiger._replace_table("place_boundaries", boundaries)

    assert transit.load_transit_route_places() == 1
    row = store.query("SELECT route_uid, place_geoid, overlap_m, route_fraction FROM transit_route_places").to_dict(
        "records"
    )[0]
    assert row["route_uid"] == "r1"
    assert row["place_geoid"] == "4261000"
    assert row["overlap_m"] > 1
    assert 0.4 < row["route_fraction"] < 0.6


def test_transit_route_bridge_excludes_boundary_touches(fresh_db):
    import db.duckdb_store as store

    conn = fresh_db.get_conn()
    _insert_route(conn, "r1", LineString([(-80.03, 40.44), (-80.02, 40.44)]))
    boundaries = tiger.normalize_place_frame(_place_gdf(), "tl_2026_42_place.shp")
    tiger._replace_table("place_boundaries", boundaries)

    assert transit.load_transit_route_places() == 0
    assert store.query("SELECT COUNT(*) AS n FROM transit_route_places").iloc[0]["n"] == 0


def test_missing_boundaries_leave_the_existing_transit_bridge_untouched(fresh_db):
    import db.duckdb_store as store

    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO transit_route_places (route_uid, place_geoid, overlap_m, route_fraction) "
        "VALUES ('r1', '4261000', 123.0, 0.5)"
    )

    assert transit.load_transit_route_places() == 0
    assert store.query("SELECT COUNT(*) AS n FROM transit_route_places").iloc[0]["n"] == 1


def test_catalog_exposes_transit_route_bridges_and_domains(fresh_db):
    import db.schema_catalog as schema
    from db.catalog_seed import SEED_DOMAINS

    schema.reload()
    for table in ("transit_route_zctas", "transit_route_places"):
        assert table in schema.TABLES
        assert SEED_DOMAINS[table] == "mobility"
        assert schema.DOMAIN_LABELS[SEED_DOMAINS[table]] == "Mobility"
        roles = {n.column: n.role for n in schema.TABLES[table].column_notes}
        assert roles["route_uid"] == "key"
        assert roles["overlap_m"] == "measure"

    keys = {r.key() for r in schema.RELATIONSHIPS}
    assert "transit_route_zctas:route_uid=transit_routes:route_uid" in keys
    assert "transit_route_zctas:zcta5=zcta_boundaries:zcta5" in keys
    assert "transit_route_places:route_uid=transit_routes:route_uid" in keys
    assert "transit_route_places:place_geoid=place_boundaries:place_geoid" in keys


def test_route_agency_bridge_splits_multi_valued_ntd_ids(fresh_db):
    """Some route files list several agencies in one semicolon-delimited field."""
    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO transit_routes (route_uid, route_id, ntd_id) VALUES "
        "('r1', '1', '90032;90034;90131'), ('r2', '2', '90032'), ('r3', '3', NULL)"
    )

    assert transit.load_transit_route_agencies() == 4
    rows = conn.execute(
        "SELECT route_uid, ntd_id, raw_ntd_id FROM transit_route_agencies ORDER BY route_uid, ntd_id"
    ).fetchall()
    assert rows == [
        ("r1", "90032", "90032;90034;90131"),
        ("r1", "90034", "90032;90034;90131"),
        ("r1", "90131", "90032;90034;90131"),
        ("r2", "90032", "90032"),
    ]


def test_route_agency_catalog_uses_the_bridge_not_equality_on_ntd_id(fresh_db):
    import db.schema_catalog as schema

    schema.reload()
    keys = {r.key() for r in schema.RELATIONSHIPS}
    assert "transit_routes:ntd_id=transit_agencies:ntd_id" not in keys
    assert "transit_route_agencies:route_uid=transit_routes:route_uid" in keys
    assert "transit_route_agencies:ntd_id=transit_agencies:ntd_id" in keys


def test_zip_route_question_plans_against_the_transit_zcta_bridge(fresh_db):
    import db.schema_catalog as schema
    from agents.query_planner import build_query_plan

    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO zcta_boundaries (zcta5, geometry_json) VALUES (?, ?)",
        ["15213", json.dumps(box(-79.98, 40.43, -79.94, 40.45).__geo_interface__)],
    )
    schema.reload()

    plan = build_query_plan("How many transit routes serve ZIP 15213?")
    assert "transit_route_zctas" in plan.required_tables
    assert "zcta_boundaries" in plan.required_tables
    assert plan.operation == "count"
    assert plan.select_expression and "DISTINCT transit_route_zctas.route_uid" in plan.select_expression
    assert plan.entity_filters == ["zcta_boundaries.zcta5 = '15213'"]

    miles = build_query_plan("Transit route miles in ZIP 15213")
    assert "transit_route_zctas" in miles.required_tables
    assert miles.operation == "sum"
    assert miles.select_expression and "transit_route_zctas.overlap_m / 1609.344" in miles.select_expression
    assert miles.entity_filters == ["zcta_boundaries.zcta5 = '15213'"]


def test_zip_route_plan_compiles_to_executable_sql(fresh_db):
    import db.duckdb_store as store
    from agents import tools
    from agents.query_planner import build_query_plan

    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO zcta_boundaries (zcta5, geometry_json) VALUES (?, ?)",
        ["15213", json.dumps(box(-79.98, 40.43, -79.94, 40.45).__geo_interface__)],
    )
    conn.execute(
        "INSERT INTO transit_routes (route_uid, route_id, geometry_json) VALUES (?, ?, ?)",
        ["r1", "61C", json.dumps(LineString([(-79.96, 40.44), (-79.95, 40.44)]).__geo_interface__)],
    )
    conn.execute(
        "INSERT INTO transit_route_zctas (route_uid, zcta5, overlap_m, route_fraction) VALUES (?, ?, ?, ?)",
        ["r1", "15213", 500.0, 0.5],
    )

    sql = tools._compile_sql_from_plan(build_query_plan("How many transit routes serve ZIP 15213?"))
    assert "COUNT(DISTINCT transit_route_zctas.route_uid)" in sql
    assert "zcta_boundaries.zcta5 = '15213'" in sql
    assert int(store.query(sql).iloc[0, 0]) == 1


def test_place_name_stopwords_do_not_become_entities(fresh_db):
    import db.schema_catalog as schema
    from agents.query_planner import build_query_plan

    conn = fresh_db.get_conn()
    conn.execute("INSERT INTO place_boundaries (place_geoid, short_name) VALUES ('1', 'Many')")
    conn.execute("INSERT INTO place_boundaries (place_geoid, short_name) VALUES ('2', 'Pittsburgh')")
    schema.reload()

    plan = build_query_plan("How many transit routes serve Pittsburgh city limits?")
    assert "transit_routes_in_place" in plan.semantic_keys
    assert plan.entity_filters == ["place_boundaries.short_name = 'Pittsburgh'"]
