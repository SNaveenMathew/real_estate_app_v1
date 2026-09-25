"""Shared fixtures: every test runs against its own throw-away DuckDB (and therefore its own catalog)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOUSES = [
    ("h1", "123 Main Street", "Pittsburgh", "PA", "15213", "42003040100"),
    ("h2", "45 Oak Ave", "Pittsburgh", "PA", "15213", "42003040100"),
    ("h3", "9 Elm St", "Pittsburgh", "PA", "15217", "42003050100"),
    ("h4", "77 Pine Rd", "Philadelphia", "PA", "19104", "42101000100"),
    ("h5", "88 Cedar Ln", "Philadelphia", "PA", "19104", "42101000200"),
    ("h6", "1 River Dr", "Cleveland", "OH", "44101", "39035100100"),
]
TRACTS = [("42003040100", "Allegheny", "PA"), ("42003050100", "Allegheny", "PA"), ("42101000100", "Philadelphia", "PA"),
          ("42101000200", "Philadelphia", "PA"), ("39035100100", "Cuyahoga", "OH"), ("06037000100", "Los Angeles", "CA")]


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    """Point the app at an empty DuckDB file; the catalog reseeds itself on connect."""
    from config import settings
    import db.duckdb_store as store
    store.close()
    monkeypatch.setattr(settings, "duckdb_path", tmp_path / "test.duckdb")
    store.get_conn()
    import db.schema_catalog as schema
    schema.reload()
    yield store
    store.close()


@pytest.fixture()
def reference_data(fresh_db):
    """A few houses plus the tract tables they join to (stand-ins for the real redfin/NRI/census data)."""
    conn = fresh_db.get_conn()
    for h in HOUSES:
        conn.execute("INSERT INTO houses (house_id,address,city,state,zip,tract_fips,lat,lon,status,price) "
                     "VALUES (?,?,?,?,?,?,40.44,-80.0,'Active',250000)", list(h))
    for t, county, st in TRACTS:
        conn.execute("INSERT INTO nri_tracts (tract_fips,county_fips,state_fips,county_name,state_name,risk_score,rfld_risks) "
                     "VALUES (?,?,?,?,?,50.0,10.0)", [t, t[:5], t[:2], county, st])
        conn.execute("INSERT INTO census_tracts (tract_fips,geo_id,name,population) VALUES (?,?,?,4000)", [t, "1400000US" + t, "Tract"])
    conn.execute("INSERT INTO sold_homes (sale_id,address,city,tract_fips,sold_price) VALUES ('S1','123 Main Street','PITTSBURGH','42003040100',300000)")
    import db.schema_catalog as schema
    schema.reload()
    return conn


def write_blockgroups(tmp_path) -> Path:
    """EPA-Smart-Location-like file: 12-digit block-group GEOIDs (tract_fips is their first 11 digits)."""
    tracts = ["42003040100", "42003050100", "42101000100", "42101000200", "39035100100"] + [f"48{i:03d}0{i:05d}"[:11] for i in range(20)]
    rows = []
    for i, tr in enumerate(tracts):
        for bg in (1, 2):
            rows.append({"GEOID20": f"{tr}{bg}", "STATEFP": tr[:2], "COUNTYFP": tr[2:5], "TRACTCE": tr[5:], "CBSA_Name": "Test Metro",
                         "NatWalkInd": 5 + (i * 0.7 + bg) % 15, "D3B": 100 + i, "Lead Service Line %": round(3 + (i * 1.3) % 20, 1)})
    path = tmp_path / "Smart Location Database.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture()
def blockgroup_csv(tmp_path):
    return write_blockgroups(tmp_path)


def tract_polygons(tracts: list[tuple[str, tuple]]):
    """A small GeoDataFrame of box polygons, one per (tract_fips, (minx, miny, maxx, maxy)) - a stand-in
    for services/geo_utils.py's real tract-shapefile cache, for tests that need actual polygon geometry
    (choropleth rendering) rather than just the tract_fips join logic."""
    import geopandas as gpd
    from shapely.geometry import box
    return gpd.GeoDataFrame({"tract_fips": [t for t, _ in tracts]},
                            geometry=[box(*bounds) for _, bounds in tracts], crs="EPSG:4326")


@pytest.fixture()
def tracts_gdf(monkeypatch):
    """Patches geo_utils to serve the six reference_data/HOUSES tracts as real polygons on request."""
    from services import geo_utils
    gdf = tract_polygons([
        ("42003040100", (-80.01, 40.43, -79.99, 40.45)), ("42003050100", (-79.99, 40.43, -79.97, 40.45)),
        ("42101000100", (-75.20, 39.95, -75.18, 39.97)), ("42101000200", (-75.18, 39.95, -75.16, 39.97)),
        ("39035100100", (-81.70, 41.48, -81.68, 41.50)), ("06037000100", (-118.30, 34.05, -118.28, 34.07)),
    ])

    def _patch():
        monkeypatch.setattr(geo_utils, "_load_tracts_gdf", lambda: gdf)
        monkeypatch.setattr(geo_utils, "geometry_source", lambda: "test tract polygons")
    _patch()
    return gdf


@pytest.fixture()
def walk_dataset(reference_data, blockgroup_csv):
    """The block-group dataset, described, analysed and approved (table + links to houses and nri_tracts)."""
    from services import dataset_onboarding as ob
    ds = ob.create_dataset(blockgroup_csv, "Smart Location Database.csv")
    ds = ob.update_description(ds["dataset_id"], {
        "description": "EPA walkability by block group", "grain": "one row per census block group", "domain": "mobility",
        "columns": [{"name": "natwalkind", "description": "National Walkability Index",
                     "synonyms": ["walkability index", "national walkability index"], "unit": "index"}]})
    ds = ob.analyze(ds["dataset_id"])
    ob.decide(next(p for p in ds["proposals"] if p["kind"] == "table")["proposal_id"], "approve")
    for p in ds["proposals"]:
        if p["kind"] == "relationship" and p["payload"]["right_table"] in ("houses", "nri_tracts"):
            ob.decide(p["proposal_id"], "approve")
    return ob.detail(ds["dataset_id"])
