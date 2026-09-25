"""services/map_layers.py: layer classification, generic data builders, and the bike/tract specializations.

Every classification test asserts against the REAL table schemas (created by db.duckdb_store's own
_ensure_schema, the same code the running app uses) — never a hand-trimmed test schema — so a passing
test here means the real app would classify the real table the same way.
"""
import json

import pandas as pd
import pytest

import db.schema_catalog as schema
from services import dataset_onboarding as ob
from services import map_layers as ml


def _insert_crime(conn, rows, start=0):
    """Inserts ``rows`` real crime rows, then pads with enough filler (uniform, off in a corner of the
    map no test's bbox reaches) to cross HEAT_ROW_THRESHOLD - crime_incidents is only ever meant to be
    seen as a heat layer, and the real table always has hundreds of thousands of rows, so tests that
    exercise its heat behaviour need the classifier to actually land on "heat", not "points"."""
    for i, (lat, lon, city, cat, sev) in enumerate(rows, start=start):
        conn.execute("INSERT INTO crime_incidents (incident_id, city, lat, lon, category, category_label, "
                     "severity_weight, year, month) VALUES (?,?,?,?,?,?,?,2024,1)",
                     [f"c{i}", city, lat, lon, cat, cat.title(), sev])
    filler = [(f"cf{start}_{j}", "nowhere", 89.0, 179.0, "filler", "Filler", 0.0, 2024, 1) for j in range(ml.HEAT_ROW_THRESHOLD + 1)]
    conn.executemany("INSERT INTO crime_incidents (incident_id, city, lat, lon, category, category_label, "
                     "severity_weight, year, month) VALUES (?,?,?,?,?,?,?,?,?)", filler)


def _insert_sold(conn, n, with_coords=True, start=0):
    for i in range(start, start + n):
        lat, lon = (40.44 + i * 0.0001, -80.0 + i * 0.0001) if with_coords else (None, None)
        conn.execute("INSERT INTO sold_homes (sale_id, address, city, lat, lon, sold_price, list_price) "
                     "VALUES (?,?,?,?,?,?,?)", [f"s{i}", f"{i} Sale St", "Pittsburgh", lat, lon, 200000 + i, 210000 + i])


def _insert_bike(conn, features):
    """features: list of (route_id, layer_type, color, [[lon,lat],...])"""
    for route_id, layer_type, color, coords in features:
        geom = {"type": "LineString", "coordinates": coords}
        lons, lats = [c[0] for c in coords], [c[1] for c in coords]
        conn.execute("INSERT INTO bike_routes (route_id, city, layer_type, layer_label, color, min_lon, min_lat, "
                     "max_lon, max_lat, geometry_json, properties_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     [route_id, "pittsburgh", layer_type, layer_type.replace("_", " ").title(), color,
                      min(lons), min(lats), max(lons), max(lats), json.dumps(geom), json.dumps({"length_mi": 1.0})])


PGH_BBOX = (-80.1, 40.35, -79.9, 40.55)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_houses_is_special_and_never_duplicated_by_the_generic_deriver(reference_data):
    specs = ml.derive_layer_specs()
    assert "houses" not in [s["name"] for s in specs]           # would otherwise also qualify: it has lat/lon
    out = ml.describe_layers()
    names = [l["name"] for l in out["layers"]]
    assert names.count("houses") == 1
    houses = next(l for l in out["layers"] if l["name"] == "houses")
    assert houses["special"] == "houses" and houses["default_on"] is True and houses["kind"] == "points"
    others = [l for l in out["layers"] if l["name"] != "houses"]
    assert all(l["default_on"] is False for l in others)         # only Houses defaults on


def test_point_layer_becomes_markers_below_the_threshold_and_heat_above_it(reference_data):
    conn = reference_data
    _insert_sold(conn, 5)
    schema.reload()
    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "sold_homes")
    assert spec["kind"] == "points" and spec["group"] == "overlay"

    _insert_sold(conn, ml.HEAT_ROW_THRESHOLD, start=1000)   # now comfortably over the threshold
    schema.reload()
    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "sold_homes")
    assert spec["kind"] == "heat"
    assert spec["weight"] is None                       # no severity/weight-like column on sold_homes...
    weight_cols = {w["column"] for w in spec["weight_options"]}
    assert {"sold_price", "list_price"} <= weight_cols   # ...but price is offered as a selectable weight
    assert "lat" not in weight_cols and "sale_id" not in weight_cols   # identifiers/coords are never offered


def test_crime_gets_its_real_severity_column_as_the_default_weight(reference_data):
    _insert_crime(reference_data, [(40.44, -80.0, "pittsburgh", "theft", 2.0)])
    schema.reload()
    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "crime_incidents")
    assert spec["kind"] == "heat" and spec["weight"] == "severity_weight"
    cols = {w["column"] for w in spec["weight_options"]}
    assert "year" not in cols and "month" not in cols     # excluded: these look numeric but aren't a measure


def test_bike_routes_is_a_lines_layer_using_the_specialized_renderer(reference_data):
    _insert_bike(reference_data, [("r1", "bike_lanes", "#0a5", [[-80.0, 40.44], [-79.99, 40.45]])])
    schema.reload()
    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "bike_routes")
    assert spec["kind"] == "lines" and spec["group"] == "overlay" and spec["renderer"] == "bike_routes"


def test_nri_and_census_tracts_are_the_only_fill_layers_and_risk_score_ranks_first(reference_data):
    out = ml.describe_layers()
    assert set(out["exclusive_groups"]["fill"]) == {"nri_tracts", "census_tracts"}
    nri = next(l for l in out["layers"] if l["name"] == "nri_tracts")
    assert nri["kind"] == "choropleth" and nri["default_measure"] == "risk_score"
    cols = {m["column"] for m in nri["measures"]}
    assert {"tract_fips", "county_fips", "state_fips"} & cols == set()    # keys are never offered as a measure
    assert nri["measures"][0]["label"].lower().startswith("composite")   # the curated note, not a prettified name
    census = next(l for l in out["layers"] if l["name"] == "census_tracts")
    assert census["measures"] == [{"column": "population", "label": "Census total population.", "unit": ""}]


def test_choropleth_reports_unavailable_without_tract_geometry(reference_data):
    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "nri_tracts")
    assert spec["available"] is False and "geometry" in spec["reason"].lower()


def test_choropleth_reports_available_once_tract_geometry_is_loaded(reference_data, tracts_gdf):
    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "nri_tracts")
    assert spec["available"] is True and spec["reason"] is None


def test_tables_with_no_geometry_are_not_map_layers(reference_data):
    names = {s["name"] for s in ml.derive_layer_specs()}
    assert names.isdisjoint({"house_snapshots", "census_msa", "cbsa_counties", "geocode_cache"})


def test_a_dataset_linked_to_nri_tracts_becomes_a_choropleth_with_no_code_change(walk_dataset):
    """The exact scenario from the Data page tests: an uploaded block-group CSV linked to nri_tracts by
    tract_fips. No line of map_layers.py names 'smart_location_database'; it is found through the same
    catalog relationship the SQL planner uses."""
    spec = next((s for s in ml.derive_layer_specs() if s["name"] == "smart_location_database"), None)
    assert spec is not None, "an approved, tract-linked dataset must appear as a layer automatically"
    assert spec["kind"] == "choropleth" and spec["group"] == "fill"
    assert spec["anchor"] == "nri_tracts"
    assert "natwalkind" in {m["column"] for m in spec["measures"]}
    out = ml.describe_layers()
    assert set(out["exclusive_groups"]["fill"]) == {"nri_tracts", "census_tracts", "smart_location_database"}


def test_retiring_that_dataset_removes_it_from_the_layer_panel(walk_dataset):
    ob.retire_dataset(walk_dataset["dataset_id"])
    names = {s["name"] for s in ml.derive_layer_specs()}
    assert "smart_location_database" not in names


def test_an_uploaded_polygon_dataset_becomes_a_fill_layer_with_no_code_change(reference_data, tmp_path):
    """A second, independent auto-expansion path: a shapefile/GeoJSON upload with its OWN polygon
    geometry (not joined to anything) - e.g. neighborhood boundaries - rather than a join to nri_tracts."""
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "North Side", "score": 71.0},
         "geometry": {"type": "Polygon", "coordinates": [[[-80.02, 40.45], [-79.98, 40.45], [-79.98, 40.48], [-80.02, 40.48], [-80.02, 40.45]]]}},
        {"type": "Feature", "properties": {"name": "South Side", "score": 54.0},
         "geometry": {"type": "Polygon", "coordinates": [[[-80.02, 40.40], [-79.98, 40.40], [-79.98, 40.43], [-80.02, 40.43], [-80.02, 40.40]]]}},
    ]}
    p = tmp_path / "neighborhoods.geojson"
    p.write_text(json.dumps(fc))
    ds = ob.create_dataset(p, "neighborhoods.geojson")
    ds = ob.analyze(ds["dataset_id"])
    ob.decide(next(x for x in ds["proposals"] if x["kind"] == "table")["proposal_id"], "approve")

    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "neighborhoods")
    assert spec["kind"] == "polygons" and spec["group"] == "fill" and spec["available"] is True
    assert "score" in {m["column"] for m in spec["measures"]}

    data = ml.get_layer_data("neighborhoods", west=-80.1, south=40.35, east=-79.9, north=40.55, measure="score")
    assert data["feature_count"] == 2 and data["measure"] == "score" and (data["min"], data["max"]) == (54.0, 71.0)
    names = sorted(f["properties"]["name"] for f in data["features"])
    assert names == ["North Side", "South Side"]
    assert data["features"][0]["geometry"]["type"] == "Polygon"


# ---------------------------------------------------------------------------
# Generic data builders
# ---------------------------------------------------------------------------

def test_get_points_returns_geojson_within_bbox_and_drops_blob_columns(reference_data):
    _insert_sold(reference_data, 3)
    reference_data.execute("INSERT INTO sold_homes (sale_id, address, lat, lon, sold_price) VALUES "
                           "('far', 'Far away', 10.0, 10.0, 999999)")     # outside the bbox
    schema.reload()
    data = ml.get_points("sold_homes", *PGH_BBOX)
    assert data["feature_count"] == 3 and not data["truncated"]
    assert all(-80.1 <= f["geometry"]["coordinates"][0] <= -79.9 for f in data["features"])
    props = data["features"][0]["properties"]
    assert "sale_id" in props and "sold_price" in props and "lat" not in props and "lon" not in props


def test_get_points_limit_and_truncation(reference_data):
    _insert_sold(reference_data, 10)
    schema.reload()
    data = ml.get_points("sold_homes", *PGH_BBOX, limit=4)
    assert data["feature_count"] == 4 and data["truncated"] is True


def test_get_heat_aggregates_into_a_grid_and_weights_by_severity(reference_data):
    _insert_crime(reference_data, [
        (40.4401, -80.0001, "pittsburgh", "theft", 2.0), (40.4402, -80.0002, "pittsburgh", "theft", 2.0),   # same cell
        (40.50, -80.05, "pittsburgh", "assault", 8.0),                                                       # different cell
    ])
    schema.reload()
    data = ml.get_heat("crime_incidents", *PGH_BBOX, grid_deg=0.01)
    assert data["weight"] == "severity_weight" and data["incident_count"] == 3
    assert data["cell_count"] == 2
    assert data["max_weight"] == pytest.approx(8.0)           # the lone assault outweighs the two summed thefts (4.0)
    total = sum(p[2] for p in data["points"])
    assert total == pytest.approx(12.0)                        # 2 + 2 + 8, conserved through the grouping


def test_get_heat_omitting_weight_falls_back_to_the_layers_own_default(reference_data):
    _insert_crime(reference_data, [(40.44, -80.0, "pittsburgh", "theft", 5.0)])
    schema.reload()
    default = ml.get_heat("crime_incidents", *PGH_BBOX)
    explicit = ml.get_heat("crime_incidents", *PGH_BBOX, weight="severity_weight")
    assert default["points"] == explicit["points"]
    (glat, glon, gweight), = default["points"]
    assert (glat, gweight) == pytest.approx((40.44, 5.0)) and glon == pytest.approx(-80.0, abs=2e-3)
    with pytest.raises(ml.LayerError):
        ml.get_heat("crime_incidents", *PGH_BBOX, weight="not_a_real_column")


def test_get_heat_city_filter(reference_data):
    _insert_crime(reference_data, [(40.44, -80.00, "pittsburgh", "theft", 1.0), (40.45, -80.01, "cleveland", "theft", 1.0)])
    schema.reload()
    data = ml.get_heat("crime_incidents", -81.0, 40.0, -79.0, 41.5, city="pittsburgh")
    assert data["incident_count"] == 1


def test_get_lines_bike_routes_canonicalizes_overlapping_classifications(reference_data):
    """The real BikePGH data has the same street tagged as both a generic route and a bike lane; the
    higher-priority classification should own the overlap, matching the pre-existing behaviour."""
    _insert_bike(reference_data, [
        ("shared", "on_street_bike_route", "#999", [[-80.00, 40.44], [-79.98, 40.44]]),
        ("shared_lane", "bike_lanes", "#0a5", [[-80.00, 40.44], [-79.98, 40.44]]),   # identical geometry, higher priority
        ("solo", "trails", "#333", [[-80.05, 40.50], [-80.03, 40.50]]),   # inside PGH_BBOX but far from "shared"
    ])
    schema.reload()
    data = ml.get_lines("bike_routes", *PGH_BBOX)
    assert data["exclusive_display"] is True
    by_route = {f["properties"]["route_id"]: f for f in data["features"]}
    assert "solo" in by_route and by_route["solo"]["properties"]["display_priority"] == ml.BIKE_DISPLAY_PRIORITY["trails"]
    # the lower-priority "shared" route's identical geometry was fully consumed by "shared_lane"'s higher
    # priority, so nothing is left of it to draw
    assert "shared" not in by_route and "shared_lane" in by_route


def test_get_lines_rejects_a_non_line_layer(reference_data):
    with pytest.raises(ml.LayerError):
        ml.get_lines("crime_incidents", *PGH_BBOX)


def test_get_bike_routes_is_json_safe_when_every_row_in_view_has_a_null_color(reference_data):
    """Regression test: a bbox slice where every returned row's color is NULL makes pandas infer that
    whole column as float64 NaN rather than None/object dtype - and Starlette's JSONResponse rejects NaN
    outright (allow_nan=False), so this used to raise ValueError deep in FastAPI's response rendering."""
    _insert_bike(reference_data, [("only_null_color", "trails", None, [[-80.00, 40.44], [-79.98, 40.44]])])
    schema.reload()
    import json as _json
    data = ml.get_bike_routes("bike_routes", *PGH_BBOX)
    _json.dumps(data, allow_nan=False)             # would raise if any NaN survived
    assert data["features"][0]["properties"]["color"] is None


def test_get_tract_choropleth_direct_and_joined(walk_dataset, tracts_gdf):
    direct = ml.get_tract_choropleth("nri_tracts", "risk_score", *PGH_BBOX)
    assert direct["tract_count"] == 2 and direct["measure"] == "risk_score"       # the two Pittsburgh-area tracts
    assert all(f["properties"]["value"] == 50.0 for f in direct["features"])       # reference_data seeds a flat 50.0

    joined = ml.get_tract_choropleth("smart_location_database", "natwalkind", *PGH_BBOX)
    assert joined["tract_count"] == 2
    import duckdb as _d   # sanity: recompute the expected per-tract average independently of the implementation
    for f in joined["features"]:
        tract = f["properties"]["tract_fips"]
        expected = ml.store.query(
            "SELECT AVG(natwalkind) AS v FROM smart_location_database WHERE tract_fips = ?", [tract]).iloc[0]["v"]
        assert f["properties"]["value"] == pytest.approx(expected)


def test_get_tract_choropleth_measure_validation_and_missing_geometry(reference_data, tracts_gdf):
    with pytest.raises(ml.LayerError):
        ml.get_tract_choropleth("nri_tracts", "not_a_column", *PGH_BBOX)
    with pytest.raises(ml.LayerError):
        ml.get_tract_choropleth("houses", "price", *PGH_BBOX)     # houses is not a choropleth-kind layer at all


def test_get_tract_choropleth_without_geometry_returns_an_explained_empty_result(reference_data):
    out = ml.get_tract_choropleth("nri_tracts", "risk_score", *PGH_BBOX)
    assert out["features"] == [] and "geometry" in out["warning"].lower()


def test_get_layer_data_dispatches_by_kind_and_rejects_unknown_layers(reference_data, tracts_gdf):
    assert ml.get_layer_data("nri_tracts", west=-80.1, south=40.35, east=-79.9, north=40.55)["type"] == "FeatureCollection"
    with pytest.raises(ml.LayerError):
        ml.get_layer_data("not_a_real_table", west=0, south=0, east=1, north=1)
    with pytest.raises(ml.LayerError):
        ml.get_layer_data("houses", west=0, south=0, east=1, north=1)   # listed, but not served generically


# ---------------------------------------------------------------------------
# Grounded in the real, uploaded database: exactly what combinations exist today
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_db_copy(tmp_path_factory):
    import shutil
    src = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "real_estate.duckdb"
    if not src.exists():
        pytest.skip("data/real_estate.duckdb was not included in this build")
    dst = tmp_path_factory.mktemp("real_db") / "real_estate.duckdb"
    shutil.copy(src, dst)
    return dst


@pytest.fixture()
def against_real_db(real_db_copy, monkeypatch):
    from config import settings
    import db.duckdb_store as store
    store.close()
    monkeypatch.setattr(settings, "duckdb_path", real_db_copy)
    store.get_conn()
    schema.reload()
    yield store
    store.close()


def test_real_database_layer_inventory_and_fill_exclusivity(against_real_db):
    """This is the grounded answer to 'what combinations are allowed, given the current database':
    exactly two layers compete for the map's single fill slot, and five overlays combine freely with
    each other and with whichever fill (if any) is showing."""
    out = ml.describe_layers()
    by_name = {l["name"]: l for l in out["layers"]}
    assert set(by_name) == {"houses", "bike_routes", "census_tracts", "crime_incidents", "nri_tracts", "sold_homes", "commute"}
    assert set(out["exclusive_groups"]["fill"]) == {"nri_tracts", "census_tracts"}
    assert {n for n, l in by_name.items() if l["group"] == "overlay"} == \
           {"houses", "bike_routes", "crime_incidents", "sold_homes", "commute"}
    assert by_name["crime_incidents"]["kind"] == "heat" and by_name["crime_incidents"]["weight"] == "severity_weight"
    assert by_name["sold_homes"]["kind"] == "heat" and by_name["sold_homes"]["weight"] is None
    assert by_name["bike_routes"]["kind"] == "lines" and by_name["bike_routes"]["renderer"] == "bike_routes"
    assert by_name["nri_tracts"]["default_measure"] == "risk_score" and len(by_name["nri_tracts"]["measures"]) >= 20
    assert by_name["census_tracts"]["default_measure"] == "population"
    # Orphaned pre-catalog table: real, has a matching lat/lon-free geometry column, but was never
    # registered through the Data page, so it correctly never appears as a layer.
    assert "bike_lanes" not in by_name


def test_real_database_bike_routes_render_without_error(against_real_db):
    data = ml.get_bike_routes("bike_routes", -80.2, 40.35, -79.8, 40.55, city="pittsburgh")
    assert data["feature_count"] > 0
    assert all(f["geometry"]["type"] in ("LineString", "MultiLineString") for f in data["features"])


def test_real_database_crime_heat_renders_without_error(against_real_db):
    data = ml.get_heat("crime_incidents", -80.2, 40.35, -79.8, 40.55, grid_deg=0.01, city="pittsburgh")
    assert data["cell_count"] > 0 and data["incident_count"] > 0
