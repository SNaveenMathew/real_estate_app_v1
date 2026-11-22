"""`NRI` loaders, moved out of services/data_loader.py.

Behaviour is unchanged: the facade in services/data_loader.py re-exports every name here, so
`from services.data_loader import ...` (and scripts that call it) keep working.
"""
from __future__ import annotations

"""
Data loader: reads raw files and populates DuckDB tables.
Run via setup_data.py — only needed when data files change.
"""

import pandas as pd
import numpy as np


from config import settings
import db.duckdb_store as store




__all__ = ['_NRI_HAZARDS', '_RFLD_ALIAS', '_normalise_columns', '_build_nri_out', 'load_nri', '_cache_nri_geometry']

# ── NRI ─────────────────────────────────────────────────────────────────────

_NRI_HAZARDS = ["avln", "cfld", "cwav", "drgt", "erqk", "hail", "hwav", "hrcn",
                "ifld", "istm", "lnds", "ltng", "swnd", "trnd", "tsun", "vlcn",
                "wfir", "wntw"]
# Note: RFLD (Riverine Flooding) was renamed to IFLD (Inland Flooding) in the
# 2023 NRI release. The database column is stored as rfld_risks for backward
# compatibility but is populated from IFLD_RISKS in the shapefile.
# _RFLD_ALIAS maps the old DB column name → the new shapefile column name.
_RFLD_ALIAS = "ifld"   # shapefile prefix for the inland flood hazard

# Shapefile DBF column names are capped at 10 characters, so some NRI column
# names are truncated vs the CSV version. This maps the truncated DBF name to
# the canonical name used everywhere else in the app.
# _SHP_COL_ALIASES = {
#     # Composite
#     "RISK_SCORE": "RISK_SCORE",   # exactly 10 — fine
#     "RISK_RATNG": "RISK_RATNG",
#     "RISK_NPCTL": "RISK_NPCTL",   # older CSV name
#     "RISK_SPCTL": "RISK_NPCTL",   # 2023 SHP rename (Spatial Percentile)
#     "EAL_SCORE":  "EAL_SCORE",
#     "EAL_RATNG":  "EAL_RATNG",
#     "EAL_VALT":   "EAL_VALT",
#     "SOVI_SCORE": "SOVI_SCORE",
#     "SOVI_RATNG": "SOVI_RATNG",
#     "RESL_SCORE": "RESL_SCORE",
#     "RESL_RATNG": "RESL_RATNG",
#     # State / county — occasionally differ between CSV and SHP
#     "STATEABBRV": "STATEABBRV",
#     "STATE":      "STATEABBRV",   # alternate name in some versions
#     "COUNTY":     "COUNTY",
#     "COUNTYNAME": "COUNTY",
#     "NRI_ID":     "NRI_ID",
#     "TRACTFIPS":  "TRACTFIPS",
#     "TRACT":      "TRACTFIPS",    # some SHP versions use TRACT
# }


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case all columns and apply alias map so downstream code is uniform."""
    df.columns = df.columns.str.upper().str.strip()
    # rename = {c: _SHP_COL_ALIASES[c] for c in df.columns if c in _SHP_COL_ALIASES}
    # return df.rename(columns=rename)
    return df


def _build_nri_out(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a normalised NRI DataFrame (from SHP or CSV) into the DB schema."""
    def _col(name: str):
        return df[name] if name in df.columns else None

    # Resolve TRACTFIPS — prefer explicit column, fall back to GEOID
    if "TRACTFIPS" in df.columns:
        fips_series = df["TRACTFIPS"].astype(str).str.zfill(11)
    elif "GEOID" in df.columns:
        fips_series = df["GEOID"].astype(str).str.zfill(11)
    else:
        raise ValueError("Cannot find TRACTFIPS or GEOID column in NRI data")

    out = pd.DataFrame({
        "tract_fips":  fips_series,
        "county_fips": fips_series.str[:5],
        "state_fips":  fips_series.str[:2],
        "state_name":  _col("STATEABBRV"),
        "county_name": _col("COUNTY"),
        "nri_id":      _col("NRI_ID"),
        "risk_score":  pd.to_numeric(_col("RISK_SCORE"), errors="coerce"),
        "risk_ratng":  _col("RISK_RATNG"),
        "risk_npctl":  pd.to_numeric(_col("RISK_NPCTL"), errors="coerce"),
        "eal_score":   pd.to_numeric(_col("EAL_SCORE"),  errors="coerce"),
        "eal_ratng":   _col("EAL_RATNG"),
        "eal_valt":    pd.to_numeric(_col("EAL_VALT"),   errors="coerce"),
        "sovi_score":  pd.to_numeric(_col("SOVI_SCORE"), errors="coerce"),
        "sovi_ratng":  _col("SOVI_RATNG"),
        "resl_score":  pd.to_numeric(_col("RESL_SCORE"), errors="coerce"),
        "resl_ratng":  _col("RESL_RATNG"),
    })

    # ── Hazard risk scores — flexible column matching ──────────────────
    # The 2023 NRI shapefile changed several things vs older CSV versions:
    #   • RFLD (Riverine Flooding) renamed to IFLD (Inland Flooding)
    #   • Each hazard now has three score columns: _RISKS (score), _RISKR (rating), _RISKV (value)
    #   • RISK_NPCTL renamed to RISK_SPCTL
    #
    # DB column names are kept stable (rfld_risks) for backward compatibility.
    # The mapping below handles both old CSV names and new SHP names.

    available_cols = set(df.columns)

    # DB column name → list of possible source column names in priority order
    # The first match found in available_cols wins.
    #
    # NOTE: this dict's key order must match nri_tracts' column order in
    # db/duckdb_store.py::_ensure_schema exactly — store.upsert_df() inserts
    # via `SELECT * FROM __tmp`, which is POSITIONAL, not name-based. (This
    # was previously out of sync — rfld_risks was first here but 12th in the
    # table — which silently wrote 8 of the 18 hazard scores into the wrong
    # columns for every tract ever loaded. Keep this alphabetical-by-hazard-
    # code order in sync with the DDL if you ever add a hazard there.)
    hazard_col_map = {
        # DB dest        source candidates (priority order)
        "avln_risks":  ["AVLN_RISKS", "AVLN_RISK"],
        "cfld_risks":  ["CFLD_RISKS", "CFLD_RISK"],
        "cwav_risks":  ["CWAV_RISKS", "CWAV_RISK"],
        "drgt_risks":  ["DRGT_RISKS", "DRGT_RISK"],
        "erqk_risks":  ["ERQK_RISKS", "ERQK_RISK"],
        "hail_risks":  ["HAIL_RISKS", "HAIL_RISK"],
        "hwav_risks":  ["HWAV_RISKS", "HWAV_RISK"],
        "hrcn_risks":  ["HRCN_RISKS", "HRCN_RISK"],
        "istm_risks":  ["ISTM_RISKS", "ISTM_RISK"],
        "lnds_risks":  ["LNDS_RISKS", "LNDS_RISK"],
        "ltng_risks":  ["LTNG_RISKS", "LTNG_RISK"],
        "rfld_risks":  ["RFLD_RISKS", "RFLD_RISK",  "IFLD_RISKS", "IFLD_RISK"],  # 2023: RFLD→IFLD
        "swnd_risks":  ["SWND_RISKS", "SWND_RISK"],
        "trnd_risks":  ["TRND_RISKS", "TRND_RISK"],
        "tsun_risks":  ["TSUN_RISKS", "TSUN_RISK"],
        "vlcn_risks":  ["VLCN_RISKS", "VLCN_RISK"],
        "wfir_risks":  ["WFIR_RISKS", "WFIR_RISK"],
        "wntw_risks":  ["WNTW_RISKS", "WNTW_RISK"],
    }

    matched, missing = [], []
    for dest_col, candidates in hazard_col_map.items():
        col_found = next((c for c in candidates if c in available_cols), None)
        if col_found:
            out[dest_col] = pd.to_numeric(df[col_found], errors="coerce")
            matched.append(f"{dest_col}←{col_found}")
        else:
            out[dest_col] = np.nan
            missing.append(dest_col)

    if missing:
        print(f"  ⚠ NRI: {len(missing)} hazard columns not found: {missing}")
        print(f"    Available sample: {sorted(available_cols)[:10]}")
    else:
        print(f"  ✓ NRI: all {len(hazard_col_map)} hazard columns mapped")

    out = out.replace(-9999, np.nan)
    out = out.dropna(subset=["tract_fips"])
    out = out.drop_duplicates(subset=["tract_fips"])
    return out


def load_nri() -> int:
    """
    Load NRI data — prefers the shapefile (NRI_Shapefile_CensusTracts.shp),
    falls back to the flat CSV (NRI_Table_CensusTracts.csv).

    The shapefile is strongly preferred because:
      1. It contains the tract polygons, so geo_utils can use it for spatial
         join without needing separate TIGER/Line shapefiles.
      2. It is the same data — all 18 hazard risk scores are present.

    Place files in data/nri/:
      Shapefile: extract NRI_Shapefile_CensusTracts.zip → .shp/.dbf/.prj/etc.
      CSV fallback: NRI_Table_CensusTracts.csv
    """
    import geopandas as gpd

    shp_path = settings.nri_shp
    # csv_path = settings.nri_csv

    # ── Try shapefile first ───────────────────────────────────────────────────
    if shp_path.exists():
        print(f"  Loading NRI shapefile: {shp_path.name} ...")
        print("  (This is a large file — first load may take 1-2 minutes)")
        gdf = gpd.read_file(shp_path)
        print(f"  Read {len(gdf)} features, CRS: {gdf.crs}")

        # Reproject to WGS-84 so spatial joins work with lat/lon houses
        if gdf.crs and str(gdf.crs).upper() != "EPSG:4326":
            print(f"  Reprojecting from {gdf.crs} → EPSG:4326 ...")
            gdf = gdf.to_crs("EPSG:4326")

        df = _normalise_columns(pd.DataFrame(gdf.drop(columns="geometry")))
        out = _build_nri_out(df)
        store.upsert_df("nri_tracts", out)
        print(f"  ✓ Loaded {len(out)} NRI tracts from shapefile")

        # Also cache the GeoDataFrame so geo_utils can use it for tract resolution
        # without loading anything twice (written to a temp parquet in data/nri/)
        _cache_nri_geometry(gdf, out["tract_fips"])
        return len(out)

    # ── Fall back to CSV ──────────────────────────────────────────────────────
    # if csv_path.exists():
    #     print(f"  NRI shapefile not found — loading CSV: {csv_path.name} ...")
    #     df = pd.read_csv(csv_path, low_memory=False)
    #     df = _normalise_columns(df)
    #     out = _build_nri_out(df)
    #     store.upsert_df("nri_tracts", out)
    #     print(f"  ✓ Loaded {len(out)} NRI tracts from CSV")
    #     print("  ⚠ Tip: use the shapefile instead — it also provides tract geometries")
    #     print("    for fast tract-FIPS resolution without separate TIGER/Line files.")
    #     print("    (The 'NRI' map layer also needs tract geometry — without the shapefile,")
    #     print("     add TIGER/Line shapefiles to data/shapefiles/ so it has polygons to draw.)")
    #     return len(out)

    print(f"  ✗ NRI data not found.")
    print(f"    Shapefile: {shp_path}")
    # print(f"    CSV:       {csv_path}")
    print("    Download from: https://www.fema.gov/about/openfema/data-sets/national-risk-index-data")
    return 0


def _cache_nri_geometry(gdf, tract_fips_series: pd.Series):
    """
    Save a minimal geometry parquet (tract_fips + polygon) so geo_utils can
    load it quickly on subsequent runs without re-reading the full NRI shapefile.
    """
    import geopandas as gpd
    cache_path = settings.nri_shp.parent / "nri_geometry_cache.parquet"
    try:
        slim = gdf[["geometry"]].copy()
        slim["tract_fips"] = tract_fips_series.values
        slim = gpd.GeoDataFrame(slim, geometry="geometry", crs="EPSG:4326")
        slim = slim[slim["tract_fips"].notna() & (slim["tract_fips"] != "")]
        slim.to_parquet(cache_path)
        print(f"  ✓ Geometry cache written → {cache_path.name} ({len(slim)} tracts)")
    except Exception as e:
        print(f"  ⚠ Could not write geometry cache: {e} (non-fatal)")
