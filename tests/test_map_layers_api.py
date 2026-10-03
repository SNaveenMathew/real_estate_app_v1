"""api/map_layers.py through the real FastAPI app: list/data endpoints, validation, and the two
special (non-generic) entries that are listed but not served here."""
import pytest
from fastapi.testclient import TestClient

from services import dataset_onboarding as ob

PGH = dict(west=-80.1, south=40.35, east=-79.9, north=40.55)


@pytest.fixture()
def client(reference_data):
    import main
    return TestClient(main.app)


def test_list_layers_shape_and_exclusive_groups(client):
    out = client.get("/api/layers").json()
    names = {l["name"] for l in out["layers"]}
    assert {"houses", "nri_tracts", "census_tracts", "commute"} <= names
    assert set(out["exclusive_groups"]["fill"]) == {"nri_tracts", "census_tracts"}
    houses = next(l for l in out["layers"] if l["name"] == "houses")
    assert houses["default_on"] is True and houses["special"] == "houses"


def test_houses_and_commute_are_listed_but_have_no_data_route(client):
    assert client.get("/api/layers/houses", params=PGH).status_code == 404
    assert client.get("/api/layers/commute", params=PGH).status_code == 404


def test_unknown_layer_and_bad_bbox_are_rejected_cleanly(client):
    r = client.get("/api/layers/not_a_real_table", params=PGH)
    assert r.status_code == 422 and "not a map layer" in r.json()["detail"]
    r = client.get("/api/layers/nri_tracts", params={"west": -80.1, "south": 40.35, "east": -79.9})  # north missing
    assert r.status_code == 422


def test_bike_routes_over_http(client):
    reference_data = client.app  # noqa: F841 - the fixture already seeded via reference_data below
    import db.duckdb_store as store
    import json as _json
    store.get_conn().execute(
        "INSERT INTO bike_routes (route_id, city, layer_type, layer_label, color, min_lon, min_lat, max_lon, max_lat, "
        "geometry_json) VALUES ('r1','pittsburgh','bike_lanes','Bike Lanes','#0a5',-80.0,40.44,-79.99,40.45,?)",
        [_json.dumps({"type": "LineString", "coordinates": [[-80.0, 40.44], [-79.99, 40.45]]})])
    import db.schema_catalog as schema
    schema.reload()
    data = client.get("/api/layers/bike_routes", params=PGH).json()
    assert data["feature_count"] == 1 and data["features"][0]["geometry"]["type"] == "LineString"


def test_choropleth_measure_selection_over_http(client, tracts_gdf):
    r = client.get("/api/layers/nri_tracts", params={**PGH, "measure": "rfld_risks"})
    assert r.status_code == 200 and r.json()["measure"] == "rfld_risks"
    bad = client.get("/api/layers/nri_tracts", params={**PGH, "measure": "not_a_column"})
    assert bad.status_code == 422


def test_population_choropleth_serializes_missing_values_as_null(client, tracts_gdf):
    import db.duckdb_store as store
    store.get_conn().execute(
        "UPDATE census_tracts SET population = NULL WHERE tract_fips = '42003040100'")

    response = client.get("/api/layers/census_tracts", params={**PGH, "measure": "population"})

    assert response.status_code == 200
    data = response.json()
    missing = next(f for f in data["features"] if f["properties"]["tract_fips"] == "42003040100")
    assert missing["properties"]["value"] is None
    assert data["min"] == data["max"] == 4000


def test_a_dataset_approved_through_the_data_page_is_immediately_servable_here(client, walk_dataset, tracts_gdf):
    """No restart, no code change: the same catalog change General/House Chat pick up immediately (see
    tests/test_agent_integration.py) also reaches the map layer API on the very next request."""
    out = client.get("/api/layers").json()
    assert "smart_location_database" in {l["name"] for l in out["layers"]}
    r = client.get("/api/layers/smart_location_database", params={**PGH, "measure": "natwalkind"})
    assert r.status_code == 200 and r.json()["measure"] == "natwalkind"


def test_retiring_a_dataset_removes_it_from_the_api_immediately(client, walk_dataset):
    ob.retire_dataset(walk_dataset["dataset_id"])
    out = client.get("/api/layers").json()
    assert "smart_location_database" not in {l["name"] for l in out["layers"]}
    assert client.get("/api/layers/smart_location_database", params=PGH).status_code == 422


# ---------------------------------------------------------------------------
# Year filter — API-level (new feature)
# ---------------------------------------------------------------------------

def _seed_crime_years(rows_by_year):
    """Insert crime rows via the live DuckDB connection used by the test client."""
    import db.duckdb_store as store
    import db.schema_catalog as schema
    conn = store.get_conn()
    idx = 0
    for year, pts in rows_by_year.items():
        for lat, lon, sev in pts:
            conn.execute(
                "INSERT INTO crime_incidents (incident_id, city, lat, lon, category, "
                "category_label, severity_weight, year, month) VALUES (?,?,?,?,?,?,?,?,1)",
                [f"api_cy{idx}", "pittsburgh", lat, lon, "theft", "Theft", sev, year],
            )
            idx += 1
    # pad past HEAT_ROW_THRESHOLD so the classifier picks "heat"
    from services import map_layers as ml
    filler = [
        (f"api_cf{j}", "nowhere", 89.0, 179.0, "filler", "Filler", 0.0, 2024, 1)
        for j in range(ml.HEAT_ROW_THRESHOLD + 1)
    ]
    conn.executemany(
        "INSERT INTO crime_incidents (incident_id, city, lat, lon, category, "
        "category_label, severity_weight, year, month) VALUES (?,?,?,?,?,?,?,?,?)",
        filler,
    )
    schema.reload()


def test_layer_list_includes_year_options_for_crime(client):
    """GET /api/layers must include a non-empty, sorted year_options list for crime_incidents."""
    _seed_crime_years({2019: [(40.44, -80.0, 1.0)], 2021: [(40.45, -80.01, 1.0)]})
    out = client.get("/api/layers").json()
    crime = next((l for l in out["layers"] if l["name"] == "crime_incidents"), None)
    assert crime is not None, "crime_incidents must appear in the layer list"
    years = crime.get("year_options", [])
    assert 2019 in years and 2021 in years
    assert years == sorted(years)


def test_crime_heat_with_valid_year_returns_filtered_data(client):
    """?year=2019 must scope the aggregated grid to 2019 incidents only."""
    _seed_crime_years({
        2019: [(40.44, -80.00, 3.0)],
        2021: [(40.45, -80.01, 9.0)],
    })
    r = client.get("/api/layers/crime_incidents", params={**PGH, "year": 2019, "grid_deg": 0.01})
    assert r.status_code == 200
    data = r.json()
    assert data["incident_count"] == 1
    assert data["year"] == 2019
    assert sum(p[2] for p in data["points"]) == pytest.approx(3.0)


def test_crime_heat_with_nonexistent_year_returns_empty(client):
    """?year=1900 must return an empty points list, not a 4xx/5xx error."""
    _seed_crime_years({2019: [(40.44, -80.0, 1.0)]})
    r = client.get("/api/layers/crime_incidents", params={**PGH, "year": 1900, "grid_deg": 0.01})
    assert r.status_code == 200
    data = r.json()
    assert data["points"] == [] and data["incident_count"] == 0


def test_crime_heat_without_year_param_returns_all_years(client):
    """Omitting ?year= must aggregate all years (existing default behaviour preserved)."""
    _seed_crime_years({
        2019: [(40.44, -80.00, 3.0)],
        2021: [(40.45, -80.01, 7.0)],
    })
    r = client.get("/api/layers/crime_incidents", params={**PGH, "grid_deg": 0.01})
    assert r.status_code == 200
    data = r.json()
    assert data["incident_count"] == 2
    assert data["year"] is None


def test_crime_heat_year_param_must_be_integer(client):
    """?year=notanumber must be rejected by FastAPI validation (422)."""
    r = client.get("/api/layers/crime_incidents", params={**PGH, "year": "notanumber"})
    assert r.status_code == 422


def test_crime_heat_returns_max_year_weight(client):
    """The API response for crime heat must include max_year_weight."""
    _seed_crime_years({
        2019: [(40.44, -80.00, 3.0)],
        2020: [(40.44, -80.00, 8.0)],
    })
    r = client.get("/api/layers/crime_incidents", params={**PGH, "year": 2019, "grid_deg": 0.01})
    assert r.status_code == 200
    data = r.json()
    assert "max_year_weight" in data
    assert data["max_year_weight"] == pytest.approx(8.0)


def test_heat_layer_without_year_column_rejects_year_query(client, reference_data):
    """Querying a heat layer that lacks a year column (sold_homes) with ?year= must return 422."""
    from tests.test_map_layers import _insert_sold
    from services import map_layers as ml
    import db.schema_catalog as schema
    _insert_sold(reference_data, ml.HEAT_ROW_THRESHOLD + 10)
    schema.reload()
    r = client.get("/api/layers/sold_homes", params={**PGH, "year": 2020})
    assert r.status_code == 422
    assert "does not have an integer year column" in r.json()["detail"]

