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
