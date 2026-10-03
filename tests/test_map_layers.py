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
    assert census["measures"] == [
        {"column": "population", "label": "Total", "unit": "people"},
        {"column": "population_density", "label": "Density", "unit": "people/sq mi"},
    ]


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


def test_a_plain_latlon_upload_becomes_a_points_layer_with_no_code_change(reference_data, tmp_path):
    """Regression test: the simplest possible geo upload - a CSV with human-written coordinate headers
    and whole-number values - used to disappear from the layer list entirely. Two independent causes,
    both fixed: (1) whole-degree coordinates infer as an integer SQL type, which the classifier required
    to be exactly "float"; (2) "Latitude"/"Longitude" sanitize to latitude/longitude, not the literal
    "lat"/"lon" this app's built-in tables happen to use, so the column-name check never matched."""
    p = tmp_path / "stations.csv"
    pd.DataFrame({"Station": ["A", "B", "C"], "Latitude": [40, 41, 42], "Longitude": [-80, -79, -78],
                 "Ridership": [1200, 800, 50]}).to_csv(p, index=False)
    ds = ob.create_dataset(p, "stations.csv")
    cols = {c["name"]: c["dtype"] for c in ds["columns"]}
    assert cols["latitude"] == "integer" and cols["longitude"] == "integer"   # confirms this exercises the int path
    ds = ob.analyze(ds["dataset_id"])
    ob.decide(next(x for x in ds["proposals"] if x["kind"] == "table")["proposal_id"], "approve")

    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "stations")
    assert spec["kind"] == "points" and spec["group"] == "overlay"
    assert (spec["lat_col"], spec["lon_col"]) == ("latitude", "longitude")

    data = ml.get_points("stations", -81, 39, -77, 43)
    assert data["feature_count"] == 3
    coords = sorted(f["geometry"]["coordinates"] for f in data["features"])
    assert coords == [[-80, 40], [-79, 41], [-78, 42]]
    props = data["features"][0]["properties"]
    assert props == {"station": "A", "ridership": 1200}    # latitude/longitude themselves are not "properties"


def test_latlon_alias_names_are_accepted_and_ranked_by_geographic_plausibility(reference_data, tmp_path):
    p = tmp_path / "sensors.csv"
    pd.DataFrame({"Sensor": ["s1", "s2"], "Lat": [40.4, 40.5], "Lng": [-80.1, -80.2],
                 "Reading": [5.5, 6.5]}).to_csv(p, index=False)
    ds = ob.create_dataset(p, "sensors.csv")
    ds = ob.analyze(ds["dataset_id"])
    ob.decide(next(x for x in ds["proposals"] if x["kind"] == "table")["proposal_id"], "approve")
    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "sensors")
    assert (spec["lat_col"], spec["lon_col"]) == ("lat", "lng")
    data = ml.get_points("sensors", -81, 40, -80, 41)
    assert data["feature_count"] == 2


def test_a_same_named_but_non_geographic_column_is_not_mistaken_for_coordinates(reference_data, tmp_path):
    """The safety net for the broader alias set: a column named like a coordinate but holding values well
    outside any real lat/lon range must not turn an ordinary table into a bogus map layer."""
    p = tmp_path / "readings.csv"
    pd.DataFrame({"Sensor": ["s1", "s2", "s3"], "y": [42000, 43000, 44000], "x": [91000, 92000, 93000],
                 "Value": [1.0, 2.0, 3.0]}).to_csv(p, index=False)
    ds = ob.create_dataset(p, "readings.csv")
    ds = ob.analyze(ds["dataset_id"])
    ob.decide(next(x for x in ds["proposals"] if x["kind"] == "table")["proposal_id"], "approve")
    assert not any(s["name"] == "readings" for s in ml.derive_layer_specs())


def test_builtin_tables_are_unaffected_even_when_currently_empty_or_sparse(reference_data):
    """sold_homes and crime_incidents use the literal "lat"/"lon" convention; that must be trusted
    unconditionally; the fixture used here only ever inserts a sold_homes row with no coordinates at
    all, which the broader alias set's range safety net would (correctly) reject - the exact-name path
    must not be subject to that same check, or an otherwise-fine, simply-quiet table would fall through
    to being misread as a tract choropleth instead of a heat/points layer."""
    row = reference_data.execute("SELECT lat, lon FROM sold_homes").fetchone()
    assert row == (None, None)          # confirms this test is exercising the all-NULL case, not a fluke
    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "sold_homes")
    assert spec["kind"] in ("points", "heat") and (spec["lat_col"], spec["lon_col"]) == ("lat", "lon")


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


def test_msa_population_density_uses_total_population_over_total_land_area(reference_data, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box
    from services import geo_utils

    tracts = gpd.GeoDataFrame(
        {"tract_fips": ["42003040100", "42003050100", "18097010100"]},
        geometry=[
            box(-80.01, 40.43, -79.99, 40.45),
            box(-79.99, 40.43, -79.95, 40.45),
            box(-86.20, 39.70, -86.17, 39.72),
        ],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(geo_utils, "_load_tracts_gdf", lambda: tracts)
    reference_data.execute("INSERT INTO census_tracts VALUES ('18097010100', '1400000US18097010100', 'Marion', 2000)")
    reference_data.executemany(
        "INSERT INTO census_msa VALUES (?, NULL, ?, NULL)",
        [("38300", "Pittsburgh, PA Metro Area")],
    )
    reference_data.executemany(
        "INSERT INTO cbsa_counties VALUES (?, ?, 'Metro', ?, ?, ?, ?)",
        [
            ("38300", "Pittsburgh, PA Metro Area", "42", "003", "Allegheny", "Pennsylvania"),
            ("26900", "Indianapolis, IN Metro Area", "18", "097", "Marion", "Indiana"),
        ],
    )

    names = ["Pittsburgh, PA Metro Area"]
    request = "Compare the total population density (sum(population)/sum(land area)) of Indianapolis vs Pittsburgh"
    result = ml.get_msa_population_density(names, request=request)
    pittsburgh_area_sq_mi = tracts.iloc[:2].to_crs("EPSG:5070").geometry.area.sum() / ml.SQUARE_METER_PER_SQUARE_MILE
    indianapolis_area_sq_mi = tracts.iloc[2:].to_crs("EPSG:5070").geometry.area.sum() / ml.SQUARE_METER_PER_SQUARE_MILE

    assert result["msa_name"].tolist() == ["Indianapolis, IN Metro Area", "Pittsburgh, PA Metro Area"]
    assert result.iloc[0]["population"] == 2000
    assert result.iloc[0]["land_area_sq_mi"] == pytest.approx(indianapolis_area_sq_mi)
    assert result.iloc[0]["population_density"] == pytest.approx(2000 / indianapolis_area_sq_mi)
    assert result.iloc[1]["population"] == 8000
    assert result.iloc[1]["land_area_sq_mi"] == pytest.approx(pittsburgh_area_sq_mi)
    assert result.iloc[1]["population_density"] == pytest.approx(8000 / pittsburgh_area_sq_mi)

    from agents.tools import query_database
    response = query_database.invoke({
        "request": request
    })
    assert "Pittsburgh, PA Metro Area" in response
    assert "Indianapolis, IN Metro Area" in response
    assert "population_density" in response
    assert "Code Agent error:" not in response


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


# ---------------------------------------------------------------------------
# Year filter and year_options (new feature)
# ---------------------------------------------------------------------------

def _insert_crime_years(conn, rows_by_year, start=0):
    """Insert crime rows with explicit years; still pads to cross HEAT_ROW_THRESHOLD so the
    classifier lands on 'heat'.  rows_by_year is {year: [(lat, lon, sev), ...]}."""
    idx = start
    for year, pts in rows_by_year.items():
        for lat, lon, sev in pts:
            conn.execute(
                "INSERT INTO crime_incidents (incident_id, city, lat, lon, category, "
                "category_label, severity_weight, year, month) VALUES (?,?,?,?,?,?,?,?,1)",
                [f"cy{idx}", "pittsburgh", lat, lon, "theft", "Theft", sev, year],
            )
            idx += 1
    # pad to guarantee heat classification
    filler = [
        (f"cf{start}_{j}", "nowhere", 89.0, 179.0, "filler", "Filler", 0.0, 2024, 1)
        for j in range(ml.HEAT_ROW_THRESHOLD + 1)
    ]
    conn.executemany(
        "INSERT INTO crime_incidents (incident_id, city, lat, lon, category, "
        "category_label, severity_weight, year, month) VALUES (?,?,?,?,?,?,?,?,?)",
        filler,
    )


def test_crime_spec_exposes_year_options(reference_data):
    """The crime_incidents layer spec must include a sorted year_options list when
    the table has a year column, so the frontend can build the year dropdown."""
    _insert_crime_years(reference_data, {2019: [(40.44, -80.0, 1.0)], 2021: [(40.45, -80.01, 1.0)]})
    schema.reload()
    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "crime_incidents")
    assert spec["kind"] == "heat"
    assert "year_options" in spec
    years = spec["year_options"]
    assert 2019 in years and 2021 in years
    assert years == sorted(years)                          # must be ascending


def test_crime_spec_year_options_excludes_none(reference_data):
    """year_options must only contain actual integer years, never None."""
    _insert_crime_years(reference_data, {2020: [(40.44, -80.0, 1.0)]})
    schema.reload()
    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "crime_incidents")
    assert all(y is not None and isinstance(y, int) for y in spec["year_options"])


def test_get_heat_year_filter_scopes_to_selected_year(reference_data):
    """With year=2019 only the 2019 incident falls in the result; the 2021 row is excluded."""
    _insert_crime_years(reference_data, {
        2019: [(40.44, -80.00, 3.0)],   # in bbox
        2021: [(40.45, -80.01, 7.0)],   # in bbox, different year
    })
    schema.reload()
    data_2019 = ml.get_heat("crime_incidents", *PGH_BBOX, grid_deg=0.01, year=2019)
    assert data_2019["incident_count"] == 1
    total = sum(p[2] for p in data_2019["points"])
    assert total == pytest.approx(3.0)                     # only the severity=3 row

    data_2021 = ml.get_heat("crime_incidents", *PGH_BBOX, grid_deg=0.01, year=2021)
    assert data_2021["incident_count"] == 1
    assert sum(p[2] for p in data_2021["points"]) == pytest.approx(7.0)


def test_get_heat_year_none_returns_all_years(reference_data):
    """year=None (default) must aggregate across every year — the sum of both years."""
    _insert_crime_years(reference_data, {
        2019: [(40.44, -80.00, 3.0)],
        2021: [(40.45, -80.01, 7.0)],
    })
    schema.reload()
    data_all = ml.get_heat("crime_incidents", *PGH_BBOX, grid_deg=0.01, year=None)
    assert data_all["incident_count"] == 2
    assert sum(p[2] for p in data_all["points"]) == pytest.approx(10.0)


def test_get_heat_year_with_no_matching_rows_returns_empty(reference_data):
    """Requesting a year that has no incidents in the viewport returns an empty points list."""
    _insert_crime_years(reference_data, {2019: [(40.44, -80.00, 1.0)]})
    schema.reload()
    data = ml.get_heat("crime_incidents", *PGH_BBOX, grid_deg=0.01, year=2022)
    assert data["points"] == [] and data["incident_count"] == 0 and data["max_weight"] == 0.0


def test_get_heat_year_filter_combined_with_weight(reference_data):
    """year and weight filters must compose: only 2020 rows, weighted by severity_weight."""
    _insert_crime_years(reference_data, {
        2020: [(40.44, -80.00, 4.0), (40.44, -80.00, 6.0)],   # same grid cell
        2021: [(40.44, -80.00, 99.0)],                          # different year — must be excluded
    })
    schema.reload()
    data = ml.get_heat("crime_incidents", *PGH_BBOX, grid_deg=0.01, weight="severity_weight", year=2020)
    assert data["incident_count"] == 2
    assert data["max_weight"] == pytest.approx(10.0)            # 4 + 6, only 2020 rows


def test_get_heat_response_includes_year_field(reference_data):
    """The response dict must echo back the year that was requested (mirrors weight echoing)."""
    _insert_crime_years(reference_data, {2019: [(40.44, -80.0, 1.0)]})
    schema.reload()
    data_year = ml.get_heat("crime_incidents", *PGH_BBOX, year=2019)
    assert data_year["year"] == 2019
    data_none = ml.get_heat("crime_incidents", *PGH_BBOX, year=None)
    assert data_none["year"] is None


def test_get_layer_data_passes_year_through_to_get_heat(reference_data):
    """get_layer_data is the dispatcher used by the API route; confirm year reaches get_heat."""
    _insert_crime_years(reference_data, {
        2018: [(40.44, -80.00, 2.0)],
        2022: [(40.45, -80.01, 5.0)],
    })
    schema.reload()
    data = ml.get_layer_data(
        "crime_incidents", west=-80.1, south=40.35, east=-79.9, north=40.55,
        grid_deg=0.01, year=2018,
    )
    assert data["incident_count"] == 1
    assert sum(p[2] for p in data["points"]) == pytest.approx(2.0)


def test_heat_layers_without_year_column_have_no_year_options(reference_data):
    """sold_homes has no year column; its spec must not expose year_options at all, so the
    frontend never renders a year dropdown for it."""
    _insert_sold(reference_data, ml.HEAT_ROW_THRESHOLD + 10)
    schema.reload()
    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "sold_homes")
    assert spec["kind"] == "heat"
    assert "year_options" not in spec or spec.get("year_options") is None or spec.get("year_options") == []


# ---------------------------------------------------------------------------
# Real-database grounded year filter tests (run only when real DB is present)
# ---------------------------------------------------------------------------

def test_real_database_crime_year_options_match_actual_data(against_real_db):
    """year_options in the real spec must be a non-empty sorted list covering the years
    that actually exist in crime_incidents (grounded from the live database)."""
    spec = next(s for s in ml.derive_layer_specs() if s["name"] == "crime_incidents")
    years = spec.get("year_options", [])
    assert len(years) > 0, "crime_incidents must expose at least one year"
    assert years == sorted(years)
    # The real database spans 2005–2023 (confirmed at implementation time)
    assert 2005 in years
    assert 2023 in years
    assert all(isinstance(y, int) for y in years)


def test_real_database_crime_heat_year_filter_reduces_count(against_real_db):
    """A year-filtered heat call must return fewer incidents than the unfiltered call
    (any year from the real data set will achieve this since each year is a strict subset)."""
    _all = ml.get_heat("crime_incidents", -80.2, 40.35, -79.8, 40.55, grid_deg=0.01)
    _2019 = ml.get_heat("crime_incidents", -80.2, 40.35, -79.8, 40.55, grid_deg=0.01, year=2019)
    assert _2019["incident_count"] > 0, "2019 must have incidents in the Pittsburgh bbox"
    assert _2019["incident_count"] < _all["incident_count"]
    assert _2019["year"] == 2019


def test_real_database_crime_heat_unknown_year_returns_empty(against_real_db):
    """A year that doesn't exist in the database (e.g. 1900) must return an empty result,
    not an error."""
    data = ml.get_heat("crime_incidents", -80.2, 40.35, -79.8, 40.55, grid_deg=0.01, year=1900)
    assert data["points"] == [] and data["incident_count"] == 0


def test_get_heat_rejects_year_for_table_without_year_column(reference_data):
    """sold_homes has no year column; passing year must raise LayerError."""
    _insert_sold(reference_data, ml.HEAT_ROW_THRESHOLD + 10)
    schema.reload()
    with pytest.raises(ml.LayerError, match="does not have an integer year column"):
        ml.get_heat("sold_homes", *PGH_BBOX, year=2020)


def test_get_heat_includes_max_year_weight_when_year_column_present(reference_data):
    """When a heat table has a year column, get_heat must return max_year_weight,
    which represents the maximum single-year cell density across all years."""
    _insert_crime_years(reference_data, {
        2019: [(40.44, -80.00, 3.0)],
        2020: [(40.44, -80.00, 7.0)],
    })
    schema.reload()
    d19 = ml.get_heat("crime_incidents", *PGH_BBOX, grid_deg=0.01, year=2019)
    assert d19["max_weight"] == pytest.approx(3.0)
    assert d19["max_year_weight"] == pytest.approx(7.0)


def test_real_database_crime_heat_max_year_weight_bounds(against_real_db):
    """In the real crime database, max_year_weight must be strictly positive and
    greater than or equal to any individual year's max_weight."""
    d = ml.get_heat("crime_incidents", -80.2, 40.35, -79.8, 40.55, grid_deg=0.01, year=2019)
    assert d["max_year_weight"] is not None
    assert d["max_year_weight"] >= d["max_weight"]
    assert d["max_year_weight"] > 0
