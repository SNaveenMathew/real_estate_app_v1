"""TIGER/Line ZCTA5 and Place onboarding: normalization, upload routing, and bridges."""

from __future__ import annotations

import json
import os

import geopandas as gpd
from shapely.geometry import LineString, box

from services import data_sources as ds
from services.loaders import tiger


def _zcta_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "ZCTA5CE20": ["00601", "15213"],
            "GEOID20": ["00601", "15213"],
            "CLASSFP20": ["B5", "B5"],
            "MTFCC20": ["G6350", "G6350"],
            "FUNCSTAT20": ["S", "S"],
            "ALAND20": [100_000_000, 6_318_000],
            "AWATER20": [0, 20_000],
            "INTPTLAT20": ["18.1805", "40.4356"],
            "INTPTLON20": ["-66.7499", "-79.9641"],
        },
        geometry=[box(-66.8, 18.1, -66.7, 18.2), box(-79.98, 40.43, -79.94, 40.45)],
        crs="EPSG:4269",
    )


def _place_gdf() -> gpd.GeoDataFrame:
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


def test_zcta_normalization_keeps_leading_zero_and_derives_map_fields():
    out = tiger.normalize_zcta_frame(_zcta_gdf(), "tl_2026_us_zcta520.shp")
    assert out["zcta5"].tolist() == ["00601", "15213"]
    row = out.set_index("zcta5").loc["15213"]
    assert row["land_area_sq_mi"] > 0
    assert row["water_area_sq_mi"] > 0
    assert row["lat"] == 40.4356 and row["lon"] == -79.9641
    assert row["min_lon"] < row["max_lon"] and row["min_lat"] < row["max_lat"]
    assert json.loads(row["geometry_json"])["type"] == "Polygon"
    assert json.loads(row["properties_json"])["ZCTA5CE20"] == "15213"


def test_place_normalization_preserves_official_name_and_adds_short_name():
    out = tiger.normalize_place_frame(_place_gdf(), "tl_2026_42_place.shp")
    row = out.iloc[0]
    assert row["place_geoid"] == "4261000"
    assert row["place_fips"] == "61000"
    assert row["state_fips"] == "42"
    assert row["name"] == "Pittsburgh city"
    assert row["short_name"] == "Pittsburgh"
    assert row["land_area_sq_mi"] > 0


def test_upload_header_detection_routes_tiger_columns_to_the_builtin_sources():
    assert ds._detect_tiger_zcta(list(_zcta_gdf().columns)) is True
    assert ds._detect_tiger_place(list(_place_gdf().columns)) is True
    assert ds._detect_tiger_zcta(list(_place_gdf().columns)) is False
    assert ds._detect_tiger_place(list(_zcta_gdf().columns)) is False
    assert ds.detect_source_for_columns(list(_zcta_gdf().columns)) == "tiger_zcta"
    assert ds.detect_source_for_columns(list(_place_gdf().columns)) == "tiger_places"


def test_newer_zip_wins_over_a_stale_extracted_shapefile(tmp_path, monkeypatch):
    """A zip uploaded through the Data page must replace extracted files that are still on disk.

    The loader deduplicates by GEOID with keep="last"; discovery has to order the
    newest source last for that to actually upsert rather than silently keep the
    old extracted copy.
    """
    from config import settings

    monkeypatch.setattr(settings, "tiger_dir", tmp_path)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data-root")
    extracted = tmp_path / "tl_2026_42_place"
    extracted.mkdir()
    old_shp = extracted / "tl_2026_42_place.shp"
    old_shp.write_bytes(b"old")
    new_zip = tmp_path / "tl_2026_42_place.zip"
    new_zip.write_bytes(b"new")
    os.utime(old_shp, (1_000_000, 1_000_000))
    os.utime(new_zip, (2_000_000, 2_000_000))

    assert tiger.tiger_boundary_paths("place")[-1] == new_zip


def test_tiger_precheck_reads_a_zipped_shapefile(tmp_path):
    import zipfile

    source = tmp_path / "tl_2026_42_place"
    source.mkdir()
    _place_gdf().to_file(source / "tl_2026_42_place.shp")
    archive = tmp_path / "tl_2026_42_place.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in source.iterdir():
            zf.write(path, path.name)

    assert tiger.boundary_columns(archive)  # reads the inner .shp's schema
    assert ds._make_tiger_precheck("place")(archive) == (True, "")
    assert ds._make_tiger_precheck("zcta")(archive)[0] is False
    assert ds.peek_header_candidates(archive, ".zip")


def test_catalog_and_ontology_expose_both_boundary_layers(fresh_db):
    import db.schema_catalog as schema
    from db import ontology

    schema.reload()
    for table in ("zcta_boundaries", "place_boundaries", "bike_route_zctas", "bike_route_places"):
        assert table in schema.TABLES
    zcta_roles = {n.column: n.role for n in schema.TABLES["zcta_boundaries"].column_notes}
    assert zcta_roles["geo_id"] == "key" and zcta_roles["aland"] == "measure"
    place_roles = {n.column: n.role for n in schema.TABLES["place_boundaries"].column_notes}
    assert place_roles["aland"] == "measure" and place_roles["classfp"] == "dimension"
    keys = {r.key() for r in schema.RELATIONSHIPS}
    assert "bike_route_places:place_geoid=place_boundaries:place_geoid" in keys
    assert "bike_route_zctas:zcta5=zcta_boundaries:zcta5" in keys
    graph = ontology.build_ontology()
    classes = {c["id"] for c in graph["classes"]}
    assert {"table:zcta_boundaries", "table:place_boundaries"} <= classes


def test_route_to_place_bridge_counts_only_positive_lengths(fresh_db):
    import db.duckdb_store as store

    conn = fresh_db.get_conn()
    conn.execute(
        "INSERT INTO bike_routes (route_id, city, layer_type, geometry_json) VALUES (?,?,?,?)",
        [
            "r1",
            "pittsburgh",
            "bike_lanes",
            json.dumps(LineString([(-80.01, 40.44), (-79.93, 40.44)]).__geo_interface__),
        ],
    )
    boundaries = tiger.normalize_place_frame(_place_gdf(), "tl_2026_42_place.shp")
    tiger._replace_table("place_boundaries", boundaries)
    assert tiger.load_bike_route_places() == 1
    rows = store.query("SELECT route_id, place_geoid, overlap_m FROM bike_route_places").to_dict("records")
    assert rows and rows[0]["place_geoid"] == "4261000"
    assert rows[0]["overlap_m"] > 1

    # A route that only touches the boundary (zero-length intersection) is not a membership.
    conn.execute("DELETE FROM bike_route_places")
    conn.execute(
        "UPDATE bike_routes SET geometry_json = ? WHERE route_id = 'r1'",
        [json.dumps(LineString([(-80.03, 40.44), (-80.02, 40.44)]).__geo_interface__)],
    )
    assert tiger.load_bike_route_places() == 0
    assert store.query("SELECT COUNT(*) AS n FROM bike_route_places").iloc[0]["n"] == 0


def test_extracted_national_folder_under_data_root_is_discovered(tmp_path, monkeypatch):
    from config import settings

    data_root = tmp_path / "data"
    tiger_dir = data_root / "tiger"
    national = data_root / "cb_2025_us_place_500k"
    state = tiger_dir / "tl_2026_42_place"
    for folder in (national, state):
        folder.mkdir(parents=True)
    national_shp = national / "cb_2025_us_place_500k.shp"
    state_shp = state / "tl_2026_42_place.shp"
    national_shp.write_bytes(b"national")
    state_shp.write_bytes(b"state")
    monkeypatch.setattr(settings, "data_dir", data_root)
    monkeypatch.setattr(settings, "tiger_dir", tiger_dir)

    assert tiger.tiger_boundary_paths("place") == [national_shp, state_shp]


def test_newer_vintage_beats_extraction_mtime(tmp_path, monkeypatch):
    """An extracted 2025 file must not override a 2026 file just because extraction reset its mtime."""
    from config import settings

    national = tmp_path / "cb_2025_us_place_500k"
    state = tmp_path / "tl_2026_42_place"
    for folder in (national, state):
        folder.mkdir()
    national_shp = national / "cb_2025_us_place_500k.shp"
    state_shp = state / "tl_2026_42_place.shp"
    national_shp.write_bytes(b"national")
    state_shp.write_bytes(b"state")
    os.utime(national_shp, (2_000_000, 2_000_000))
    os.utime(state_shp, (1_000_000, 1_000_000))
    monkeypatch.setattr(settings, "tiger_dir", tmp_path)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "unused")

    assert tiger.tiger_boundary_paths("place")[-1] == state_shp


def test_same_vintage_prefers_full_tiger_line(tmp_path, monkeypatch):
    from config import settings

    cartographic = tmp_path / "cb_2026_us_place_500k"
    tiger_line = tmp_path / "tl_2026_42_place"
    for folder in (cartographic, tiger_line):
        folder.mkdir()
    cartographic_shp = cartographic / "cb_2026_us_place_500k.shp"
    tiger_shp = tiger_line / "tl_2026_42_place.shp"
    cartographic_shp.write_bytes(b"cartographic")
    tiger_shp.write_bytes(b"tiger")
    os.utime(cartographic_shp, (2_000_000, 2_000_000))
    os.utime(tiger_shp, (1_000_000, 1_000_000))
    monkeypatch.setattr(settings, "tiger_dir", tmp_path)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "unused")

    assert tiger.tiger_boundary_paths("place")[-1] == tiger_shp


def test_zip_and_extracted_copy_of_one_dataset_are_read_once(tmp_path, monkeypatch):
    from config import settings

    extracted = tmp_path / "tl_2026_42_place"
    extracted.mkdir()
    shp = extracted / "tl_2026_42_place.shp"
    shp.write_bytes(b"extracted")
    archive = tmp_path / "tl_2026_42_place.zip"
    archive.write_bytes(b"zip")
    os.utime(shp, (1_000_000, 1_000_000))
    os.utime(archive, (2_000_000, 2_000_000))
    monkeypatch.setattr(settings, "tiger_dir", tmp_path)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "unused")

    assert tiger.tiger_boundary_paths("place") == [archive]


def test_source_file_label_is_relative_to_the_data_tree(tmp_path, monkeypatch):
    from config import settings

    data_root = tmp_path / "data"
    path = data_root / "cb_2025_us_place_500k" / "cb_2025_us_place_500k.shp"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    monkeypatch.setattr(settings, "data_dir", data_root)
    monkeypatch.setattr(settings, "tiger_dir", data_root / "tiger")

    label = tiger._source_label(path)
    assert "cb_2025_us_place_500k" in label and label.endswith(".shp")


def test_national_cartographic_columns_route_to_the_place_source():
    """The national CBF omits CLASSFP/FUNCSTAT/INTPTLAT; upload detection must still accept it."""
    columns = list(_place_gdf().drop(columns=["CLASSFP", "FUNCSTAT", "INTPTLAT", "INTPTLON"]).columns)
    assert ds._detect_tiger_place(columns) is True
    assert ds.detect_source_for_columns(columns) == "tiger_places"
