"""
Tests for agents/artifacts.py: the deterministic table/chart/map
classifier, the per-turn artifact bus, and the two places that emit into
it today (agents/tools.py::query_database and ::find_bike_route).

These use `reference_data` (tests/conftest.py) for real DuckDB-typed
DataFrames rather than hand-built ones, so the numeric/datetime dtype
checks in classify_dataframe are exercised against what DuckDB actually
hands back, not an idealized pandas frame.
"""
import pandas as pd
import pytest

from agents import artifacts as A


# ── classify_dataframe: map ─────────────────────────────────────────────────

def test_classify_dataframe_detects_map_from_lat_lon(reference_data):
    import db.duckdb_store as store
    df = store.query("SELECT house_id, city, lat, lon FROM houses ORDER BY house_id")
    artifact = A.classify_dataframe(df, request="show houses on a map")
    assert artifact["type"] == "map"
    assert artifact["map_kind"] == "points"
    assert len(artifact["points"]) == 6
    p = artifact["points"][0]
    assert p["lat"] == pytest.approx(40.44)
    assert p["lon"] == pytest.approx(-80.0)
    assert p["label"]  # picked a non-coordinate column as the label
    assert "house_id" in p["fields"] or "city" in p["fields"]


def test_classify_dataframe_map_ignores_rows_with_missing_coordinates():
    df = pd.DataFrame({
        "city": ["A", "B", "C"],
        "lat": [40.1, None, 40.3],
        "lon": [-80.1, -80.2, None],
    })
    artifact = A.classify_dataframe(df, request="map it")
    assert artifact["type"] == "map"
    assert len(artifact["points"]) == 1  # only the first row has both coordinates


def test_classify_dataframe_map_hint_falls_back_when_no_coordinates():
    df = pd.DataFrame({"city": ["Pittsburgh", "Philadelphia", "Cleveland"], "avg_price": [250000.0, 410000.0, 180000.0]})
    artifact = A.classify_dataframe(df, request="chart it", presentation="map")
    # No lat/lon columns exist anywhere in the result -- a "map" hint must
    # never be forced into existence; it degrades to whatever the shape
    # actually supports (here, a chart).
    assert artifact["type"] != "map"
    assert artifact["type"] == "chart"


# ── classify_dataframe: chart ────────────────────────────────────────────────

def test_classify_dataframe_detects_chart_from_one_dimension_one_measure(reference_data):
    import db.duckdb_store as store
    df = store.query("SELECT city, AVG(price) AS avg_price FROM houses GROUP BY city ORDER BY city")
    artifact = A.classify_dataframe(df, request="average price by city")
    assert artifact["type"] == "chart"
    assert artifact["chart_kind"] == "bar"
    assert sorted(artifact["categories"]) == ["Cleveland", "Philadelphia", "Pittsburgh"]
    assert artifact["series"][0]["name"] == "Avg Price"
    assert len(artifact["series"][0]["values"]) == 3


def test_classify_dataframe_year_like_integer_column_is_a_dimension_not_a_measure():
    # "year" is numeric-dtype but is something you plot rows *by*, not a
    # measure to sum/average -- without this, {city, year, avg_price}
    # would look like 2 measures + 0 dimensions and never chart at all.
    df = pd.DataFrame({
        "city": ["Pittsburgh", "Pittsburgh", "Philadelphia", "Philadelphia"],
        "year": [2023, 2024, 2023, 2024],
        "avg_price": [240000, 250000, 400000, 410000],
    })
    artifact = A.classify_dataframe(df, request="chart price by city and year", presentation="chart")
    assert artifact["type"] == "chart"
    assert artifact["categories"] == ["Pittsburgh / 2023", "Pittsburgh / 2024", "Philadelphia / 2023", "Philadelphia / 2024"]


def test_classify_dataframe_line_chart_for_temporal_dimension_sorts_ascending():
    df = pd.DataFrame({"year": [2024, 2022, 2023], "count": [30, 10, 20]})
    artifact = A.classify_dataframe(df, request="trend over time")
    assert artifact["type"] == "chart"
    assert artifact["chart_kind"] == "line"
    assert artifact["categories"] == ["2022", "2023", "2024"]
    assert artifact["series"][0]["values"] == [10.0, 20.0, 30.0]


def test_classify_dataframe_chart_handles_nan_and_inf_as_null_points():
    df = pd.DataFrame({"city": ["A", "B", "C"], "val": [1.0, float("nan"), float("inf")]})
    artifact = A.classify_dataframe(df, request="chart it")
    assert artifact["type"] == "chart"
    assert artifact["series"][0]["values"] == [1.0, None, None]


# ── classify_dataframe: table (fallback) ────────────────────────────────────

def test_classify_dataframe_falls_back_to_table_for_wide_shape(reference_data):
    import db.duckdb_store as store
    df = store.query("SELECT house_id, address, city, price FROM houses ORDER BY house_id LIMIT 3")
    artifact = A.classify_dataframe(df, request="list houses")
    assert artifact["type"] == "table"
    assert [c["key"] for c in artifact["columns"]] == ["house_id", "address", "city", "price"]
    assert len(artifact["rows"]) == 3
    assert artifact["total_rows"] == 3
    assert artifact["truncated"] is False


def test_classify_dataframe_table_marks_truncation(monkeypatch):
    monkeypatch.setattr(A.settings, "presentation_table_max_rows", 2)
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 3], "c": ["p", "q", "r"]})
    artifact = A.classify_dataframe(df, request="wide list", presentation="table")
    assert artifact["type"] == "table"
    assert artifact["total_rows"] == 3
    assert len(artifact["rows"]) == 2
    assert artifact["truncated"] is True


# ── classify_dataframe: skip / hint edge cases ───────────────────────────────

def test_classify_dataframe_returns_none_for_single_row():
    df = pd.DataFrame({"n": [6]})
    assert A.classify_dataframe(df, request="how many houses") is None


def test_classify_dataframe_table_hint_forces_table_even_for_single_row():
    df = pd.DataFrame({"n": [6]})
    artifact = A.classify_dataframe(df, request="how many houses", presentation="table")
    assert artifact["type"] == "table"
    assert artifact["rows"] == [{"n": 6}]


def test_classify_dataframe_returns_none_for_empty_dataframe():
    df = pd.DataFrame({"city": [], "price": []})
    assert A.classify_dataframe(df, request="nothing matched") is None


def test_classify_dataframe_returns_none_for_none_input():
    assert A.classify_dataframe(None, request="oops") is None


def test_classify_dataframe_unknown_presentation_falls_back_to_auto():
    df = pd.DataFrame({"city": ["A", "B", "C"], "n": [1, 2, 3]})
    artifact = A.classify_dataframe(df, request="x", presentation="bogus-value")
    assert artifact["type"] == "chart"  # same as "auto" would produce


def test_classify_dataframe_never_raises_on_bad_input():
    # A column whose values can't be coerced the way the builders expect
    # should degrade to "no artifact", never propagate an exception into
    # the calling tool.
    df = pd.DataFrame({"lat": ["not-a-number"], "lon": ["also-not-a-number"], "extra": [1]})
    assert A.classify_dataframe(df, request="map it") is None  # 1 row, no measures -> None, and no crash


# ── artifact bus ─────────────────────────────────────────────────────────────

def test_reset_emit_collect_roundtrip():
    A.reset_artifacts()
    A.emit_artifact({"type": "table", "rows": []})
    A.emit_artifact({"type": "chart"})
    assert A.collect_artifacts() == [{"type": "table", "rows": []}, {"type": "chart"}]
    A.reset_artifacts()
    assert A.collect_artifacts() == []


def test_emit_artifact_ignores_falsy_values():
    A.reset_artifacts()
    A.emit_artifact(None)
    A.emit_artifact({})
    assert A.collect_artifacts() == []


# ── wiring: query_database emits through the same path House Chat uses ─────

def test_query_database_emits_chart_artifact_for_qualifying_result(reference_data, monkeypatch):
    import agents.tools as tools

    monkeypatch.setattr(tools, "generate_sql", lambda *a, **kw: "SELECT city, AVG(price) AS avg_price FROM houses GROUP BY city")
    monkeypatch.setattr(tools, "_validate_sql_against_plan", lambda sql, plan: None)

    A.reset_artifacts()
    result = tools.query_database.invoke({
        "request": "average price by city",
        "presentation": "chart",
    })
    assert "avg_price" in result.lower() or "AVG" in result
    produced = A.collect_artifacts()
    assert len(produced) == 1
    assert produced[0]["type"] == "chart"


def test_query_database_defaults_to_auto_presentation(reference_data, monkeypatch):
    import agents.tools as tools

    monkeypatch.setattr(tools, "generate_sql", lambda *a, **kw: "SELECT house_id, city, lat, lon FROM houses")
    monkeypatch.setattr(tools, "_validate_sql_against_plan", lambda sql, plan: None)

    A.reset_artifacts()
    tools.query_database.invoke({"request": "where are the houses"})
    produced = A.collect_artifacts()
    assert len(produced) == 1
    assert produced[0]["type"] == "map"  # lat/lon present -> map, even with no hint


# ── wiring: find_bike_route normalizes into the same generic envelope ──────

def test_find_bike_route_emits_bike_route_map_artifact(monkeypatch):
    import agents.tools as tools

    fake_result = {
        "route": {
            "shape": [[40.44, -80.0], [40.45, -80.01]],
            "bbox": {"north": 40.46, "south": 40.43, "east": -79.99, "west": -80.02},
            "used_infrastructure": {"type": "FeatureCollection", "features": []},
        },
        "summary": {"length": 1.2, "time": 600},
        "instructions": ["Head north", "Arrive"],
        "facilities": {"facility_segments": []},
        "start": {"lat": 40.44, "lon": -80.0},
        "end": {"lat": 40.45, "lon": -80.01},
        "city": "Pittsburgh, PA",
        "provider": "BikePGH",
        "attribution": "BikePGH network data",
        "crime_avoidance": {"enabled": False, "applied": False},
        "analysis_visualization": None,
    }

    async def fake_route_bike(*args, **kwargs):
        return fake_result

    monkeypatch.setattr(tools.bike_routing, "route_bike", fake_route_bike)

    A.reset_artifacts()
    result = tools.find_bike_route.invoke({"start": "A", "end": "B", "city": "Pittsburgh, PA"})
    assert '"kind": "route_found"' in result or "route_found" in result
    produced = A.collect_artifacts()
    assert len(produced) == 1
    assert produced[0]["type"] == "map"
    assert produced[0]["map_kind"] == "bike_route"
    assert produced[0]["route_shape"] == fake_result["route"]["shape"]


def test_find_bike_route_emits_crime_analysis_map_when_present(monkeypatch):
    import agents.tools as tools

    analysis_payload = {"filtered_bike_network": {"type": "FeatureCollection", "features": []}, "start": {}, "end": {}}
    fake_result = {
        "route": {"shape": [[40.44, -80.0], [40.45, -80.01]], "bbox": {}, "used_infrastructure": {"type": "FeatureCollection", "features": []}},
        "summary": {"length": 1.0, "time": 500},
        "instructions": [],
        "facilities": {"facility_segments": []},
        "start": {"lat": 40.44, "lon": -80.0},
        "end": {"lat": 40.45, "lon": -80.01},
        "city": "Pittsburgh, PA",
        "provider": "BikePGH",
        "attribution": "BikePGH network data",
        "crime_avoidance": {"enabled": True, "applied": True, "percentile": 90.0},
        "analysis_visualization": analysis_payload,
    }

    async def fake_route_bike(*args, **kwargs):
        return fake_result

    monkeypatch.setattr(tools.bike_routing, "route_bike", fake_route_bike)

    A.reset_artifacts()
    tools.find_bike_route.invoke({
        "start": "A", "end": "B", "city": "Pittsburgh, PA", "avoid_crime_dense_areas": True,
    })
    produced = A.collect_artifacts()
    assert len(produced) == 2
    assert produced[0]["type"] == "map" and produced[0]["map_kind"] == "bike_crime_analysis"
    assert produced[0]["analysis"] == analysis_payload
    assert produced[1]["map_kind"] == "bike_route"
