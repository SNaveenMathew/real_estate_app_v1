"""Synthetic census fixture shared by the census / planner-hardening tests.

Builds a throw-away DuckDB (schema + catalog seeded by the app itself) holding a few metros with counties, tracts and
tract populations, a tract-geometry cache whose polygons have *known* areas (so tests assert exact expected values
instead of snapshotting whatever the code returns), plus name-only metros for place-resolution tests.

Nothing touches the real ``data/`` directory: ``activate()`` points ``settings.duckdb_path`` / ``settings.nri_shp`` /
``settings.shapefile_dir`` at a temp dir and ``deactivate()`` restores them, so other test modules are unaffected.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd          # noqa: E402
import pandas as pd              # noqa: E402
from pyproj import Transformer   # noqa: E402
from shapely.geometry import box # noqa: E402

SQ_KM_PER_SQ_MI = 2.589988110336

# (cbsa_code, cbsa_title, msa display name, state_fips, anchor lon/lat, [(county_fips, county_name)], scale)
# ``scale`` stretches tract sides and populations so each metro has a distinct, easily asserted area and population.
METROS = [
    ("38300", "Pittsburgh, PA", "Pittsburgh, PA Metro Area", "42", (-80.00, 40.44),
     [("003", "Allegheny County"), ("007", "Beaver County")], 1.0),
    ("26900", "Indianapolis-Carmel-Anderson, IN", "Indianapolis-Carmel-Anderson, IN Metro Area", "18", (-86.16, 39.77),
     [("097", "Marion County"), ("057", "Hamilton County")], 1.5),
    # Deliberately ambiguous principal-city names, to exercise the "which one did you mean" path.
    ("38900", "Portland-Vancouver-Hillsboro, OR-WA", "Portland-Vancouver-Hillsboro, OR-WA Metro Area", "41",
     (-122.67, 45.52), [("051", "Multnomah County")], 0.8),
    ("38860", "Portland-South Portland, ME", "Portland-South Portland, ME Metro Area", "23",
     (-70.26, 43.66), [("005", "Cumberland County")], 0.5),
]

# Present in census_msa only (no counties/tracts): name-resolution tests, and "metro with no county membership".
NAME_ONLY_MSAS = [
    "Denver-Aurora-Lakewood, CO Metro Area", "Miami-Fort Lauderdale-West Palm Beach, FL Metro Area",
    "Austin-Round Rock-Georgetown, TX Metro Area", "Nashville-Davidson--Murfreesboro--Franklin, TN Metro Area",
    "Winston-Salem, NC Metro Area", "Salem, OR Metro Area", "Mobile, AL Metro Area",
    "Washington-Arlington-Alexandria, DC-VA-MD-WV Metro Area", "Minneapolis-St. Paul-Bloomington, MN-WI Metro Area",
    "Dallas-Fort Worth-Arlington, TX Metro Area", "Columbus, OH Metro Area", "Columbus, GA-AL Metro Area",
    "Columbus, IN Micro Area", "Reading, PA Metro Area", "Springfield, MA Metro Area", "Springfield, MO Metro Area",
    "Louisville/Jefferson County, KY-IN Metro Area", "Kansas City, MO-KS Metro Area",
]

# An unresolved ("X"-coded) metro whose title only matches cbsa_counties by name: exercises the title fallback.
X_CODED = ("Xabc1234", "Beaver Valley Test, PA Metro Area", "99999", "Beaver Valley Test, PA", "42", "007")

TRACT_SIDES_KM = (8.0, 10.0, 12.0)
TRACT_POPULATIONS = (3000, 4000, 5000)


@dataclass
class CensusFixture:
    tmp_path: Path
    expected: dict = field(default_factory=dict)   # msa display name -> expected population / area / density
    saved: dict = field(default_factory=dict)


def _build_geometry(rows: list[dict]) -> gpd.GeoDataFrame:
    to_albers = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    polys, fips = [], []
    for r in rows:
        x0, y0 = to_albers.transform(*r["anchor"])
        x = x0 + r["slot"] * 40_000.0           # keep tracts apart so they never overlap
        half = r["side_km"] * 1000.0 / 2.0
        polys.append(box(x - half, y0 - half, x + half, y0 + half))
        fips.append(r["tract_fips"])
    return gpd.GeoDataFrame({"tract_fips": fips}, geometry=polys, crs="EPSG:5070").to_crs("EPSG:4326")


def _clear_geometry_caches():
    from services import geo_utils
    for fn in (geo_utils._load_nri_geometry, geo_utils._load_tiger_shapefiles, geo_utils._load_tracts_gdf):
        fn.cache_clear()
    try:                                         # the provider keeps its own per-geometry area cache
        from services import census_metrics
        census_metrics.reset_caches()
    except Exception:
        pass


def activate(tmp_path: Path) -> CensusFixture:
    """Create the fixture database + geometry cache and point the app's settings at them."""
    import db.duckdb_store as store
    import db.schema_catalog as schema
    from config import settings

    fx = CensusFixture(tmp_path=Path(tmp_path))
    fx.saved = {k: getattr(settings, k) for k in ("duckdb_path", "nri_shp", "shapefile_dir")}
    store.close()
    settings.duckdb_path = fx.tmp_path / "fixture.duckdb"
    settings.nri_shp = fx.tmp_path / "nri" / "NRI_fixture.shp"
    settings.shapefile_dir = fx.tmp_path / "shapefiles"          # empty: forces the NRI-cache path
    _clear_geometry_caches()

    conn = store.get_conn()                                       # creates the schema and seeds the catalog
    cbsa_rows, msa_rows, tract_rows, geo_rows, nri_rows = [], [], [], [], []
    for cbsa, title, display, st, anchor, counties, scale in METROS:
        msa_pop, msa_area_km2, county_expect, slot = 0, 0.0, {}, 0
        for cfips, cname in counties:
            cbsa_rows.append({"cbsa_code": cbsa, "cbsa_title": title, "msa_type": "Metropolitan Statistical Area",
                              "state_fips": st, "county_fips": cfips, "county_name": cname, "state_name": "Fixture"})
            c_pop, c_area = 0, 0.0
            for i, (base_side, base_pop) in enumerate(zip(TRACT_SIDES_KM, TRACT_POPULATIONS)):
                side, pop = base_side * scale, int(base_pop * scale)
                tract = f"{st}{cfips}{(i + 1) * 100:06d}"
                tract_rows.append({"tract_fips": tract, "geo_id": f"1400000US{tract}", "name": f"Tract {tract}",
                                   "population": pop})
                nri_rows.append({"tract_fips": tract, "county_fips": st + cfips, "state_fips": st,
                                 "county_name": cname, "risk_score": 40.0 + i, "rfld_risks": 10.0 + i})
                geo_rows.append({"tract_fips": tract, "anchor": anchor, "slot": slot, "side_km": side})
                slot += 1
                c_pop += pop
                c_area += side * side
            county_expect[cname] = {"population": c_pop, "area_sq_mi": c_area / SQ_KM_PER_SQ_MI}
            msa_pop += c_pop
            msa_area_km2 += c_area
        msa_rows.append({"msa_code": cbsa, "geo_id": f"310M500US{cbsa}", "name": display, "population": msa_pop})
        fx.expected[display] = {"population": msa_pop, "area_sq_mi": msa_area_km2 / SQ_KM_PER_SQ_MI,
                                "density": msa_pop / (msa_area_km2 / SQ_KM_PER_SQ_MI), "counties": county_expect,
                                "cbsa_title": title}

    for i, name in enumerate(NAME_ONLY_MSAS):
        msa_rows.append({"msa_code": f"7{i:04d}", "geo_id": f"310M500US7{i:04d}", "name": name, "population": 100_000 + i})
    xcode, xname, xcbsa, xtitle, xst, xcounty = X_CODED
    msa_rows.append({"msa_code": xcode, "geo_id": "", "name": xname, "population": 0})
    cbsa_rows.append({"cbsa_code": xcbsa, "cbsa_title": xtitle, "msa_type": "Metropolitan Statistical Area",
                      "state_fips": xst, "county_fips": xcounty, "county_name": "Beaver County", "state_name": "Fixture"})

    store.upsert_df("cbsa_counties", pd.DataFrame(cbsa_rows))
    store.upsert_df("census_msa", pd.DataFrame(msa_rows))
    store.upsert_df("census_tracts", pd.DataFrame(tract_rows))
    nri = pd.DataFrame(nri_rows)
    cols = [r[1] for r in conn.execute("PRAGMA table_info('nri_tracts')").fetchall()]
    for c in cols:
        if c not in nri.columns:
            nri[c] = None
    store.upsert_df("nri_tracts", nri[cols])

    for hid, city, state, tract in (("h1", "Pittsburgh", "PA", "42003000100"), ("h2", "Pittsburgh", "PA", "42003000200"),
                                    ("h3", "Indianapolis", "IN", "18097000100"), ("h4", "Denver", "CO", None),
                                    ("h5", "Austin", "TX", None), ("h6", "Miami", "FL", None)):
        conn.execute("INSERT INTO houses (house_id, address, city, state, status, price, tract_fips) "
                     "VALUES (?, ?, ?, ?, 'Active', 250000, ?)", [hid, f"{hid} Main St", city, state, tract])

    (fx.tmp_path / "nri").mkdir(parents=True, exist_ok=True)
    _build_geometry(geo_rows).to_parquet(fx.tmp_path / "nri" / "nri_geometry_cache.parquet")
    _clear_geometry_caches()
    schema.reload()
    return fx


def deactivate(fx: CensusFixture) -> None:
    import db.duckdb_store as store
    from config import settings
    store.close()
    for k, v in fx.saved.items():
        setattr(settings, k, v)
    _clear_geometry_caches()
