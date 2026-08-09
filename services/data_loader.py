"""
Data loader: reads raw files and populates DuckDB tables.
Run via setup_data.py — only needed when data files change.
"""
import json
import hashlib
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from config import settings
import db.duckdb_store as store


# ── Redfin ──────────────────────────────────────────────────────────────────

# Map common Redfin column names → our schema
_REDFIN_COL_MAP = {
    "address": "address",
    "street address": "address",
    "city": "city",
    "state": "state",
    "zip": "zip",
    "zip/postal code": "zip",
    "date": "listing_date",
    "price": "price",
    "list price": "price",
    "property_type": "property_type",
    "nearest_big_city": "nearest_big_city",
    "crime_city": "crime_city",
    "beds": "beds",
    "baths": "baths",
    "square feet": "sqft",
    "sq_ft": "sqft",
    "sq.ft.": "sqft",
    "sqft": "sqft",
    "year built": "year_built",
    "hoa/month": "hoa_fee",
    "latitude": "lat",
    "longitude": "lon",
    "lat": "lat",
    "lon": "lon",
    "status": "status",
    "url (see http://www.redfin.com/buy-a-home/comparative-market-analysis for info on pricing)": "url",
    "walk score": "walk_score",
    "bike score": "bike_score",
    "transit score": "transit_score",
    "walkscore": "walk_score",
    "bikescore": "bike_score",
    "transitscore": "transit_score",
    "price_per_sq_ft": "price_per_sq_ft",
}


def _house_id(row: pd.Series) -> str:
    key = f"{row.get('address','')}-{row.get('city','')}-{row.get('zip','')}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _clean_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip()
        series = pd.to_numeric(series, errors="coerce")
    return series


def _extract_snapshot_date(filename: str) -> str | None:
    """
    Try to extract a date from a Redfin CSV filename.
    Patterns like: redfin_2024-01-15.csv, favorites_20240115.csv,
                   redfin_export_2024_01_15.csv
    Returns an ISO date string or None.
    """
    import re as _re
    from datetime import datetime
    patterns = [
        r'(\d{4})[_\-](\d{2})[_\-](\d{2})',  # 2024-01-15 or 2024_01_15
        r'(\d{4})(\d{2})(\d{2})',              # 20240115
    ]
    for pat in patterns:
        m = _re.search(pat, filename)
        if m:
            try:
                y, mo, d = m.group(1), m.group(2), m.group(3)
                dt = datetime(int(y), int(mo), int(d))
                if 2000 <= dt.year <= 2030:
                    return dt.date().isoformat()
            except Exception:
                continue
    return None


def load_redfin(geo_utils=None) -> int:
    """Load all Redfin CSVs from data/redfin/ directory."""
    redfin_dir = settings.redfin_dir
    csv_files = list(redfin_dir.glob("*.csv"))
    if not csv_files:
        print("  No Redfin CSV files found in data/redfin/")
        return 0

    all_rows = []
    for path in csv_files:
        print(f"  Loading Redfin: {path.name}")
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except Exception as e:
            print(f"    Error: {e}")
            continue

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        col_rename = {c: _REDFIN_COL_MAP[c] for c in df.columns if c in _REDFIN_COL_MAP}
        df = df.rename(columns=col_rename)

        # Known numeric columns
        for col in ["price", "beds", "baths", "sqft", "hoa_fee", "walk_score",
                    "bike_score", "transit_score"]:
            if col in df.columns:
                df[col] = _clean_numeric(df[col])

        # lat/lon required
        if "lat" not in df.columns or "lon" not in df.columns:
            print(f"    Warning: lat/lon columns not found in {path.name}, skipping")
            continue

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])

        df["source_file"] = path.name
        df["source_type"] = "redfin"

        # snapshot_date: prefer the listing/update date from the CSV itself.
        # Fall back to filename date, then file mtime.
        if "listing_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(
                df["listing_date"], errors="coerce"
            ).dt.date.astype(str)
            # Where the CSV date is missing, fill from filename
            file_date = _extract_snapshot_date(path.name)
            if not file_date:
                import os
                from datetime import datetime as _dt
                file_date = _dt.fromtimestamp(os.path.getmtime(path)).date().isoformat()
            df["snapshot_date"] = df["snapshot_date"].where(
                df["snapshot_date"].notna() & (df["snapshot_date"] != "NaT") & (df["snapshot_date"] != "nan"),
                file_date
            )
        else:
            snap_date = _extract_snapshot_date(path.name)
            if not snap_date:
                import os
                from datetime import datetime as _dt
                snap_date = _dt.fromtimestamp(os.path.getmtime(path)).date().isoformat()
            df["snapshot_date"] = snap_date

        # Store remaining columns as JSON
        known_cols = set(_REDFIN_COL_MAP.values()) | {"source_file", "source_type", "snapshot_date"}
        extra_cols = [c for c in df.columns if c not in known_cols]
        df["raw_json"] = df[extra_cols].apply(
            lambda r: json.dumps({k: v for k, v in r.items()
                                  if not (isinstance(v, float) and np.isnan(v))}),
            axis=1
        )

        all_rows.append(df)

    if not all_rows:
        return 0

    combined = pd.concat(all_rows, ignore_index=True)
    combined["house_id"] = combined.apply(_house_id, axis=1)

    # Ensure tract_fips exists (may not be in Redfin CSV)
    if "tract_fips" not in combined.columns:
        combined["tract_fips"] = None
    if "msa_code" not in combined.columns:
        combined["msa_code"] = None

    # For the houses table: keep only the LATEST row per house_id (most recent date)
    combined_sorted = combined.sort_values("snapshot_date", ascending=False, na_position="last")
    latest = combined_sorted.drop_duplicates(subset=["house_id"], keep="first").copy()

    # # Ensure schema columns
    # con = store.get_conn()
    # table_name = 'houses'
    # schema_result = con.sql(f"DESCRIBE {table_name}")
    # schema_cols = schema_result.to_df()['column_name'].tolist()
    
    # for col in schema_cols:
    #     if col not in combined.columns:
    #         combined[col] = None
    
    # combined = combined[schema_cols]

    # # Spatial join to get census tract FIPS (if geo_utils available)
    # if geo_utils and combined["tract_fips"].isna().any():
    #     print(f"  Resolving census tracts for {combined['tract_fips'].isna().sum()} houses...")
    #     combined = geo_utils.assign_tract_fips(combined)
    
    # Spatial join for tract FIPS on the deduplicated latest set
    if geo_utils and latest["tract_fips"].isna().any():
        print(f"  Resolving census tracts for {int(latest['tract_fips'].isna().sum())} houses...")
        latest = geo_utils.assign_tract_fips(latest)
        # Propagate resolved tract_fips back to combined for snapshots
        tract_map = latest.set_index("house_id")["tract_fips"].to_dict()
        combined["tract_fips"] = combined["house_id"].map(tract_map)

    # Ensure all required columns exist on both frames
    for col in store._HOUSES_COLS + ["snapshot_date", "source_type"]:
        if col not in combined.columns:
            combined[col] = None
        if col not in latest.columns:
            latest[col] = None

    # Write: all rows → snapshots, latest → houses table
    store.upsert_houses_with_snapshots(latest, combined)

    # if 'created_at' in combined.columns:
    #     combined['created_at'] = pd.to_datetime(combined['created_at']).dt.strftime('%Y-%m-%d')
    
    # store.upsert_houses(combined)
    # print(f"  Loaded {len(combined)} houses into DuckDB")
    # return len(combined)
    
    n_unique = latest["house_id"].nunique()
    n_total  = len(combined)
    print(f"  Loaded {n_unique} unique houses ({n_total} total rows → {n_total - n_unique} historical snapshots)")
    return n_unique


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
    hazard_col_map = {
        # DB dest        source candidates (priority order)
        "rfld_risks":  ["RFLD_RISKS", "RFLD_RISK",  "IFLD_RISKS", "IFLD_RISK"],  # 2023: RFLD→IFLD
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
        "swnd_risks":  ["SWND_RISKS", "SWND_RISK"],
        "trnd_risks":  ["TRND_RISKS", "TRND_RISK"],
        "tsun_risks":  ["TSUN_RISKS", "TSUN_RISK"],
        "vlcn_risks":  ["VLCN_RISKS", "VLCN_RISK"],
        "wfir_risks":  ["WFIR_RISKS", "WFIR_RISK"],
        "wntw_risks":  ["WNTW_RISKS", "WNTW_RISK"],
        "avln_risks":  ["AVLN_RISKS", "AVLN_RISK"],
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


# ── Census (tract) ───────────────────────────────────────────────────────────

def load_census_tracts() -> int:
    """
    Load census tract populations from DECENNIALPL2020.P1 data file.

    Accepts two formats from data.census.gov:

    Standard download (DECENNIALPL2020.P1-Data.csv):
        Row 0: GEO_ID, NAME, P1_001N, P1_002N, ...   ← column names
        Row 1: Geography, Geographic Area Name, !!Total:, ...  ← descriptions (skip)
        Row 2+: 1400000US01001020100, "Census Tract...", 1775, ...

    Legacy format:
        Row 0: GEO_ID, NAME, P1_001N  (or P1_001)
        Row 1+: data immediately

    Place file at: data/census/DECENNIALPL2020_P1_tract.csv
    Or any DECENNIALPL2020*.csv in data/census/ that contains tract GEO_IDs.
    """
    # Accept the configured path or any matching file in the census dir
    census_dir = settings.census_tract_csv.parent
    candidates = (
        [settings.census_tract_csv] if settings.census_tract_csv.exists() else []
    ) + sorted(census_dir.glob("DECENNIALPL2020*.csv"))
    # Exclude the MSA file (wide-format, no GEO_IDs starting with 1400000US)
    seen: set = set()
    candidates = [p for p in candidates
                  if not (str(p) in seen or seen.add(str(p)))]

    path = None
    for c in candidates:
        # Quick sniff: does this file look like a tract file?
        try:
            sniff = pd.read_csv(c, encoding="utf-8-sig", nrows=3, dtype=str)
            # Check any cell for a tract GEO_ID prefix
            flat = " ".join(sniff.values.flatten().astype(str))
            if "1400000US" in flat:
                path = c
                break
        except Exception:
            continue

    if path is None:
        print(f"  Census tract file not found in: {census_dir}")
        print("  Download DECENNIALPL2020.P1-Data.csv from data.census.gov")
        print("  Geography filter: Census Tracts → All United States")
        print("  Save as: data/census/DECENNIALPL2020_P1_tract.csv")
        return 0

    print(f"  Loading census tract populations: {path.name}")

    # Read with all-string dtype to avoid numeric parse issues in header rows
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, low_memory=False)

    # Detect and drop the description row (row 1 in the standard download).
    # It has values like "Geography", "Geographic Area Name", "!!Total:" — none
    # of which start with "1400000US".
    geo_col = next((c for c in raw.columns if "GEO_ID" in c.upper().replace(" ", "_")), None)
    if geo_col is None:
        # BOM sometimes fuses with the first column name
        geo_col = raw.columns[0]

    # Drop rows that are not actual tract rows
    data = raw[raw[geo_col].str.startswith("1400000US", na=False)].copy()

    if data.empty:
        print(f"  Error: no rows starting with '1400000US' found in {path.name}")
        print(f"  Columns: {list(raw.columns[:5])}")
        return 0

    # Population column — P1_001N (new) or P1_001 (legacy)
    pop_col = next(
        (c for c in data.columns if c.upper().replace(" ", "") in ("P1_001N", "P1_001")),
        None,
    )
    if pop_col is None:
        print(f"  Error: population column (P1_001N or P1_001) not found.")
        print(f"  Columns: {list(data.columns[:10])}")
        return 0

    name_col = next((c for c in data.columns if c.upper() == "NAME"), None)

    out = pd.DataFrame({
        "geo_id":     data[geo_col],
        "tract_fips": data[geo_col].str.replace("1400000US", "", regex=False).str.zfill(11),
        "name":       data[name_col] if name_col else None,
        "population": pd.to_numeric(data[pop_col], errors="coerce").astype("Int64"),
    })
    out = out.dropna(subset=["tract_fips"])
    out = out.drop_duplicates(subset=["tract_fips"])

    store.upsert_df("census_tracts", out)
    print(f"  ✓ Loaded {len(out):,} census tract populations")
    return len(out)


def _normalize_msa_name(name: str) -> str:
    """
    Normalize an MSA name to a canonical lowercase ASCII string for matching.

    Handles:
    - Trailing 'Metro Area' / 'Micro Area' suffix (census wide-format files only)
    - Double dashes: Nashville-Davidson--Murfreesboro → Nashville-Davidson-Murfreesboro
    - Accented characters: Cañon→Canon, Mayagüez→Mayaguez, Bayamón→Bayamon
    """
    import re as _re
    import unicodedata as _ud
    name = str(name).strip()
    # Strip trailing area type (wide-format files only)
    name = _re.sub(r'\s+(Metro|Micro)\s+Area$', '', name, flags=_re.IGNORECASE)
    # Collapse multiple consecutive dashes
    name = _re.sub(r'-{2,}', '-', name)
    # Normalize whitespace
    name = _re.sub(r'\s+', ' ', name).strip()
    # Decompose accented chars (NFD) and drop the combining marks (category Mn)
    # e.g. ñ → n + ̃ → n,  ü → u + ̈ → u,  é → e + ́ → e
    decomposed = _ud.normalize('NFD', name)
    return ''.join(c for c in decomposed if _ud.category(c) != 'Mn').lower()


def _first_city_state(normalized_name: str):
    """
    Extract (first_city, first_state) from a normalized MSA name.

    'bakersfield-delano, ca' → ('bakersfield', 'ca')
    'salt lake city-murray, ut' → ('salt lake city', 'ut')
    'new york-newark-jersey city, ny-nj-pa' → ('new york', 'ny')

    Used as a fallback key when MSA names differ between Census and CBSA vintages
    (e.g. 2020 census names vs 2023 CBSA delineation names).
    """
    import re as _re
    # Split on last ", STATE" pattern
    m = _re.match(r'^(.*),\s*([a-z]{2}(?:-[a-z]{2})*)$', normalized_name)
    if not m:
        return normalized_name, ''
    cities_part = m.group(1)
    states_part = m.group(2)
    first_city  = cities_part.split('-')[0].strip()
    first_state = states_part.split('-')[0]
    return first_city, first_state


def _build_cbsa_lookup(cbsa_df) -> dict:
    """
    Build a multi-key lookup dict from cbsa_counties data.
    Returns {key: cbsa_code} with four tiers of keys:
      1. Full normalized name          'bakersfield, ca'
      2. First-city + state tuple      ('bakersfield', 'ca')
      3. First-city only string        'bakersfield-ca'   (last resort, single-state)
      4. '_raw_titles' key             {norm_title: code} for fuzzy Jaccard matching
    """
    lookup: dict = {}
    raw_titles: dict = {}
    for _, row in cbsa_df.iterrows():
        code  = row["cbsa_code"]
        title = row["cbsa_title"]
        norm  = _normalize_msa_name(title)
        city, state = _first_city_state(norm)

        # Tier 1
        if norm not in lookup:
            lookup[norm] = code
        # Tier 2
        pair_key = (city, state)
        if pair_key not in lookup:
            lookup[pair_key] = code
        # Tier 3
        short_key = f"{city}-{state}"
        if short_key not in lookup:
            lookup[short_key] = code
        # Tier 4 raw index
        if norm not in raw_titles:
            raw_titles[norm] = code

    lookup["_raw_titles"] = raw_titles

    return lookup


def _all_words(norm: str) -> set:
    """Significant words (>2 chars) for fuzzy scoring."""
    import re as _re
    return {w for w in _re.split(r'[\s,\-]+', norm) if len(w) > 2}


# Known 2020→2023 OMB name changes where the first city also changed.
# These can't be matched by any string similarity strategy.
# Format: {normalize(2020_census_name): normalize(2023_cbsa_title)}
_KNOWN_RENAMES = {
    # Military base renamed from Fort Polk to Fort Johnson in 2023
    "fort polk south, la":              "fort johnson, la",
    # Poughkeepsie MSA expanded to include Port Jervis in 2023
    "poughkeepsie-newburgh-middletown, ny": "poughkeepsie-newburgh-middletown-port jervis, ny",
    # Dayton MSA renamed to include Kettering in 2023
    "dayton, oh":                       "dayton-kettering, oh",
    # Some California MSAs renamed
    "napa, ca":                         "napa, ca",
}


def _match_msa_to_cbsa(name: str, lookup: dict) -> str | None:
    """
    Try to match an MSA name to a CBSA code using a four-tier lookup.

    Tier 1 — full normalized name (exact after accent stripping + dash collapse)
    Tier 2 — (first_city, first_state) tuple  [handles added secondary cities]
    Tier 3 — city-state string (single-state MSAs only)
    Tier 4 — fuzzy Jaccard similarity ≥ 0.5 within same state  [vintage renames]
    Returns the CBSA code string or None if no match found.
    """
    norm = _normalize_msa_name(name)
    city, state = _first_city_state(norm)

    # Tiers 1-3: exact and near-exact
    if norm in lookup:
        return lookup[norm]
    # Tier 2: (first_city, first_state) tuple
    if (city, state) in lookup:
        return lookup[(city, state)]
    # Tier 3: city-state string (only when state is unambiguous single state)
    if '-' not in state:
        short_key = f"{city}-{state}"
        if short_key in lookup:
            return lookup[short_key]
    
    # Tier 4: fuzzy — find best Jaccard match within same state
    # lookup contains both string keys and tuple keys; filter to tuples (city, state)
    # and then find the closest cbsa code by comparing against a raw-title lookup
    # stored under the special key prefix "_raw"
    raw_key = "_raw_titles"
    if raw_key not in lookup:
        return None   # raw titles not available

    norm_words = _all_words(norm)
    best_code  = None
    best_score = 0.0

    for cbsa_norm, code in lookup[raw_key].items():
        c_city, c_state = _first_city_state(cbsa_norm)
        if c_state != state:           # must be same state(s)
            continue
        score = len(norm_words & _all_words(cbsa_norm)) / max(
            len(norm_words | _all_words(cbsa_norm)), 1
        )
        if score > best_score:
            best_score = score
            best_code  = code

    if best_score >= 0.5:
        return best_code

    # Tier 5: explicit known 2020→2023 OMB renames
    mapped_norm = _KNOWN_RENAMES.get(norm)
    if mapped_norm and mapped_norm in lookup:
        return lookup[mapped_norm]
    if mapped_norm:
        mapped_city, mapped_state = _first_city_state(mapped_norm)
        if (mapped_city, mapped_state) in lookup:
            return lookup[(mapped_city, mapped_state)]
    
    return None


def _detect_msa_format(df: pd.DataFrame) -> str:
    """
    Detect whether a census MSA CSV is in wide or long format.

    Wide  — MSA names are column headers; first column is 'Label (Grouping)'
    Long  — GEO_ID is a column; each row is one MSA
    """
    first_col = str(df.columns[0]).strip()
    if "label" in first_col.lower() or "grouping" in first_col.lower():
        return "wide"
    cols_upper = [c.upper() for c in df.columns]
    if any("GEO_ID" in c or "GEO ID" in c for c in cols_upper):
        return "long"
    # Fallback: if many columns look like city names they're MSA headers
    msa_like = sum(1 for c in df.columns if ", " in c and ("Area" in c or "Metro" in c))
    return "wide" if msa_like > 5 else "long"


def _load_msa_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse wide-format MSA census file.

    Row 0 header: ['Label (Grouping)', 'Aberdeen, SD Micro Area', ...]
    Row 1 data:   ['Total:', '42,287', ...]
    Rows 2+:      demographic sub-rows (ignored)
    """
    # Find the "Total:" row — it's always the first data row
    total_mask = df.iloc[:, 0].astype(str).str.strip().str.lower() == "total:"
    if not total_mask.any():
        # Try "Total" without colon
        total_mask = df.iloc[:, 0].astype(str).str.strip().str.lower().str.startswith("total")
    if not total_mask.any():
        raise ValueError("Could not find 'Total:' row in wide-format MSA file")

    total_row = df[total_mask].iloc[0]
    msa_names = list(df.columns[1:])          # skip 'Label (Grouping)' column
    pop_values = list(total_row.iloc[1:])     # skip the 'Total:' cell

    records = []
    for name, pop_str in zip(msa_names, pop_values):
        try:
            pop = int(str(pop_str).replace(",", "").strip())
        except (ValueError, AttributeError):
            pop = None
        records.append({"name": name.strip(), "population": pop})

    return pd.DataFrame(records)


def _load_msa_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse long-format MSA census file (original expected format).
    Each row is one MSA; columns include GEO_ID and a population column.
    """
    cols_upper = {c: c.upper() for c in df.columns}
    geo_col  = next((c for c, u in cols_upper.items() if "GEO_ID" in u or "GEO ID" in u), None)
    name_col = next((c for c, u in cols_upper.items() if u == "NAME"), None)
    pop_col  = next((c for c, u in cols_upper.items() if "P1_001" in u), None)

    if not geo_col or not pop_col:
        raise ValueError(
            f"Long-format MSA file missing GEO_ID or P1_001 column. "
            f"Found: {list(df.columns)}"
        )

    def _extract_code(geo_id: str) -> str:
        parts = str(geo_id).split("US")
        return parts[-1] if len(parts) > 1 else geo_id

    records = []
    for _, row in df.iterrows():
        geo_id = str(row[geo_col])
        if not (geo_id.startswith("31") or geo_id.startswith("33") or "US" in geo_id):
            continue
        records.append({
            "name":       str(row[name_col]).strip() if name_col else None,
            "msa_code":   _extract_code(geo_id).zfill(5),
            "geo_id":     geo_id,
            "population": pd.to_numeric(str(row[pop_col]).replace(",", ""), errors="coerce"),
        })
    return pd.DataFrame(records)


def load_census_msa() -> int:
    """
    Load MSA population data from a Census DECENNIALPL2020.P1 file.

    Accepts both file layouts produced by data.census.gov:

    WIDE FORMAT (MSAs as columns) — the typical download when you select
    all MSAs at once and export with geography as columns:
        Column headers: 'Label (Grouping)', 'Aberdeen, SD Micro Area', ...
        Row 1:          'Total:', '42,287', ...
        File:           DECENNIALPL2020_P1_msa.csv  (or any CSV in data/census/
                        that matches DECENNIALPL2020_P1*.csv)

    LONG FORMAT (MSAs as rows) — download via the advanced table builder:
        Columns:  GEO_ID, NAME, P1_001N (total population), ...
        Each row: one MSA

    After loading, msa_code is resolved by matching normalised MSA names
    against cbsa_title in the cbsa_counties table (if already loaded).
    The match handles:
        Census: "Nashville-Davidson--Murfreesboro--Franklin, TN Metro Area"
        CBSA:   "Nashville-Davidson--Murfreesboro--Franklin, TN"
    Both reduce to: "nashville-davidson-murfreesboro-franklin, tn"
    """
    # ── Find the file ─────────────────────────────────────────────────────
    census_dir = settings.census_msa_csv.parent
    candidates = (
        [settings.census_msa_csv] if settings.census_msa_csv.exists() else []
    ) + sorted(census_dir.glob("DECENNIALPL2020_P1*.csv"))

    seen = set()
    candidates = [p for p in candidates if not (str(p) in seen or seen.add(str(p)))]

    if not candidates:
        print(f"  Census MSA file not found in: {census_dir}")
        print("  Expected: DECENNIALPL2020_P1_msa.csv (or any DECENNIALPL2020_P1*.csv)")
        print("  Download table DECENNIALPL2020.P1 from data.census.gov")
        print("  → Geography: Metropolitan/Micropolitan Statistical Areas → All")
        return 0

    path = candidates[0]
    print(f"  Loading MSA populations: {path.name}")
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype=str)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return 0

    # ── Detect and parse ──────────────────────────────────────────────────
    fmt = _detect_msa_format(raw)
    print(f"  Detected format: {fmt}")

    try:
        if fmt == "wide":
            parsed = _load_msa_wide(raw)
        else:
            parsed = _load_msa_long(raw)
    except Exception as e:
        print(f"  Parse error: {e}")
        return 0

    if parsed.empty:
        print("  No records extracted from file")
        return 0

    # ── Ensure required columns ───────────────────────────────────────────
    for col in ["msa_code", "geo_id"]:
        if col not in parsed.columns:
            parsed[col] = None

    parsed["population"] = pd.to_numeric(
        parsed["population"].astype(str).str.replace(",", ""), errors="coerce"
    ).astype("Int64")

    parsed = parsed.dropna(subset=["name"])
    parsed = parsed[parsed["name"].str.strip() != ""]
    parsed = parsed.drop_duplicates(subset=["name"])

    print(f"  Parsed {len(parsed)} MSAs")

    # ── Match CBSA codes by name (multi-strategy) ─────────────────────────
    try:
        cbsa_df = store.query("""
            SELECT DISTINCT cbsa_code, cbsa_title
            FROM cbsa_counties
            WHERE cbsa_title IS NOT NULL
        """)
    except Exception:
        cbsa_df = pd.DataFrame(columns=["cbsa_code", "cbsa_title"])

    if not cbsa_df.empty:
        # Build normalised-name → code lookup
        lookup = _build_cbsa_lookup(cbsa_df)

        matched = 0
        codes = []
        match_tiers = {"exact": 0, "city_state": 0, "city_only": 0, "none": 0}
        for name in parsed["name"]:
            norm = _normalize_msa_name(name)
            city, state = _first_city_state(norm)
            # Try tier by tier and record which matched
            if norm in lookup:
                codes.append(lookup[norm])
                match_tiers["exact"] += 1
                matched += 1
            elif (city, state) in lookup:
                codes.append(lookup[(city, state)])
                match_tiers["city_state"] += 1
                matched += 1
            elif '-' not in state and f"{city}-{state}" in lookup:
                codes.append(lookup[f"{city}-{state}"])
                match_tiers["city_only"] += 1
                matched += 1
            else:
                codes.append(None)
                match_tiers["none"] += 1

        parsed["msa_code"] = codes
        print(f"  CBSA code matched: {matched}/{len(parsed)} MSAs")

        print(f"    Exact: {match_tiers['exact']}  "
              f"City+state fallback: {match_tiers['city_state']}  "
              f"City-only fallback: {match_tiers['city_only']}  "
              f"Unmatched: {match_tiers['none']}")
        if match_tiers["none"]:
            unmatched = parsed[parsed["msa_code"].isna()]["name"].tolist()
            print(f"  ⚠ Still unmatched (likely not in CBSA delineation file): "
                  f"{unmatched[:5]}")
    else:
        print("  ⚠ cbsa_counties not loaded yet — msa_code will be NULL")
        print("    Load cbsa_counties first, then re-run: "
              "python setup_data.py --only census")

    # Rows without a msa_code can still be useful for name-based queries.
    # Use a hash of the name as a fallback key so the table has no NULL PKs.
    import hashlib as _hl
    parsed["msa_code"] = parsed["msa_code"].where(
        parsed["msa_code"].notna(),
        parsed["name"].apply(lambda n: "X" + _hl.md5(n.encode()).hexdigest()[:7])
    )
    parsed["geo_id"] = parsed["geo_id"].fillna("")
    out = parsed[["msa_code", "geo_id", "name", "population"]]

    store.upsert_df("census_msa", out)
    print(f"  ✓ Loaded {len(out)} MSA populations into census_msa")
    return len(out)


def repair_msa_codes() -> int:
    """
    Fix X-prefixed fallback msa_codes in census_msa using multi-strategy matching.

    Strategy order:
      1. Full normalized name (handles accents, double-dashes)
      2. First-city + state (handles vintage renames like Bakersfield→Bakersfield-Delano)
      3. First-city only   (last resort for single-state MSAs)

    Safe to call on an existing database — only updates rows with X-codes.
    Returns the number of rows repaired.
    """
    # Load current X-coded rows
    x_rows = store.query(
        "SELECT msa_code, name FROM census_msa WHERE msa_code LIKE 'X%'"
    )
    if x_rows.empty:
        print("  No X-coded msa_codes found — nothing to repair.")
        return 0

    print(f"  Repairing {len(x_rows)} X-coded msa_codes using multi-strategy matching...")

    # Build CBSA lookup
    try:
        cbsa_df = store.query(
            "SELECT DISTINCT cbsa_code, cbsa_title FROM cbsa_counties WHERE cbsa_title IS NOT NULL"
        )
    except Exception as e:
        print(f"  Error reading cbsa_counties: {e}")
        return 0

    if cbsa_df.empty:
        print("  cbsa_counties is empty — cannot repair. Load it first.")
        return 0

    lookup = _build_cbsa_lookup(cbsa_df)

    repaired = 0
    still_unmatched = []
    conn = store.get_conn()
    for _, row in x_rows.iterrows():
        old_code = row["msa_code"]
        name     = row["name"]
        new_code = _match_msa_to_cbsa(name, lookup)
        if new_code:
            conn.execute(
                "UPDATE census_msa SET msa_code = ? WHERE msa_code = ?",
                [new_code, old_code]
            )
            repaired += 1
        else:
            still_unmatched.append(name)

    still_x = int(store.query(
        "SELECT COUNT(*) as n FROM census_msa WHERE msa_code LIKE 'X%'"
    ).iloc[0]["n"])
    print(f"  ✓ Repaired {repaired} rows. Remaining X-codes: {still_x}")
    if still_unmatched:
        print(f"  These {len(still_unmatched)} MSAs have no matching CBSA code "
              f"(likely genuinely absent from your delineation file vintage):")
        for name in still_unmatched[:10]:
            print(f"    • {name}")
        if len(still_unmatched) > 10:
            print(f"    ... and {len(still_unmatched) - 10} more")
        print()
        print("  These MSAs cannot be joined to NRI data, but they don't affect")
        print("  the flood risk query (they're typically small Micro areas).")
    
    return repaired


# ── CBSA crosswalk ───────────────────────────────────────────────────────────

def load_cbsa_crosswalk() -> int:
    """
    Load the MSA→county crosswalk from Census Bureau delineation file.
    
    Accepts any of these formats (auto-detected by extension):
      list1_2020.csv / list1_2023.csv   — CSV, skiprows=2
      list1_2020.xls / list1_2023.xlsx  — Excel, skiprows=2
    
    Download: https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
    Save to: data/census/   (any filename ending in .csv/.xls/.xlsx is accepted)
    """
        # Accept any delineation file in the census dir regardless of exact name
    census_dir = settings.cbsa_xlsx.parent
    candidates = (
        list(census_dir.glob("list*.xlsx")) +
        list(census_dir.glob("list*.xls")) +
        list(census_dir.glob("list*.csv")) +
        ([settings.cbsa_xlsx] if settings.cbsa_xlsx.exists() else [])
    )
    # Deduplicate while preserving order
    seen = set()
    candidates = [p for p in candidates if not (str(p) in seen or seen.add(str(p)))]

    if not candidates:
        print(f"  CBSA crosswalk not found in: {census_dir}")
        print("  Download delineation file from Census Bureau and save to data/census/")
        return 0
    
    path = candidates[0]
    print(f"  Loading CBSA crosswalk: {path.name}")
    
    suffix = path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            # Census delineation Excel files have 2 header rows before data
            df = pd.read_excel(path, skiprows=2, dtype=str)
        else:
            try:
                df = pd.read_csv(path, encoding="utf-8-sig", skiprows=2,
                                 dtype=str, low_memory=False)
            except Exception:
                df = pd.read_csv(path, encoding="latin-1", skiprows=2,
                                 dtype=str, low_memory=False)
    except Exception as e:
        print(f"  Error reading {path.name}: {e}")
        return 0

    df.columns = df.columns.str.strip()
    
    # Drop completely empty rows (common at end of Census Excel files)
    df = df.dropna(how="all")

    # Column detection — Census changes column names between vintages.
    # Print columns to help debug if mapping fails.
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "cbsa code":                               col_map[c] = "cbsa_code"
        elif cl == "cbsa title":                            col_map[c] = "cbsa_title"
        elif "metropolitan" in cl and "micro" in cl:        col_map[c] = "msa_type"
        elif cl in ("fips state code", "fips state"):       col_map[c] = "state_fips"
        elif cl in ("fips county code", "fips county"):     col_map[c] = "county_fips"
        elif "county/county" in cl or cl == "county name":  col_map[c] = "county_name"
        elif cl == "state name":                            col_map[c] = "state_name"
    df = df.rename(columns=col_map)

    needed = ["cbsa_code", "state_fips", "county_fips"]
    if not all(c in df.columns for c in needed):
        print(f"  Could not find required columns. Mapped columns: {list(df.columns)}")
        print("  Expected: 'CBSA Code', 'FIPS State Code', 'FIPS County Code'")
        print("  (Column names vary by Census vintage — check the file header)")
        return 0

    df = df.dropna(subset=needed)
    # Remove footnote rows where cbsa_code is not numeric
    df = df[df["cbsa_code"].str.strip().str.match(r"^\d+$", na=False)]
    
    df["cbsa_code"]   = df["cbsa_code"].astype(str).str.strip().str.zfill(5)
    df["state_fips"]  = df["state_fips"].astype(str).str.strip().str.zfill(2)
    df["county_fips"] = df["county_fips"].astype(str).str.strip().str.zfill(3)

    for col in ["cbsa_title", "msa_type", "county_name", "state_name"]:
        if col not in df.columns:
            df[col] = None

    out = df[["cbsa_code", "cbsa_title", "msa_type",
              "state_fips", "county_fips", "county_name", "state_name"]]
    store.upsert_df("cbsa_counties", out)
    print(f"  ✓ Loaded {len(out)} CBSA→county mappings")
    return len(out)


# ── Sold homes — county-parser architecture ───────────────────────────────────
#
# Each county website exports data in its own format.  A "parser" is a class
# with a single class method  parse(df) -> pd.DataFrame  that maps raw county
# columns to the standard sold_homes schema.
#
# To add a new county: subclass CountyParserBase, implement parse(), and add
# a signature tuple to _COUNTY_PARSERS so auto-detection can identify it.
#
# Standard schema output columns (all nullable except sale_id):
#   sale_id, source_county, source_file, parid, address, city, state, zip,
#   full_address, sold_price, sold_date, record_date, list_price,
#   sqft, beds, baths, year_built, sale_code, sale_desc, instr_type,
#   instr_type_desc, deed_book, deed_page, muni_code, muni_desc,
#   school_code, school_desc, is_arms_length, arms_length_flag,
#   lat, lon, geocode_status, geocode_source, geocode_accuracy,
#   tract_fips, county_fips
# ─────────────────────────────────────────────────────────────────────────────

# Sale codes that indicate non-arm's-length transactions.
# Each entry: (code_value_or_set, reason_string)
# Applies across county parsers; each parser maps its codes to this.
_NON_ARMS_LENGTH = {
    # Allegheny-specific codes (SALECODE column)
    "3":   "Love and affection / nominal consideration",
    "5":   "Quit-claim deed",
    "6":   "Foreclosure / sheriff's sale",
    "11":  "Related party transfer",
    "16":  "Government / tax sale",
    "99":  "Corrective deed or duplicate sale",
    "CO":  "Corrective instrument",         # INSTRTYP level
    # Multi-parcel (distorts per-unit price)
    "H":   "Multi-parcel sale",
    # Generic
    "LOVE AND AFFECTION": "Love and affection / nominal consideration",
    "CORRECTIVE":         "Corrective deed",
    "ESTATE":             "Estate sale",    # flagged but not always excluded
    "FORECLOSURE":        "Foreclosure",
    "SHERIFF":            "Sheriff's sale",
}

_ESTATE_CODES = {"37"}   # flagged as questionable but kept; analyst can filter


def _classify_arms_length(sale_code: str, sale_desc: str,
                          instr_type: str, price: float) -> tuple[bool, str]:
    """
    Return (is_arms_length, flag_reason).
    Anything clearly not at market value → False.
    Estate sales → True but with a flag noting they may be discounted.
    """
    sc   = str(sale_code  or "").strip().upper()
    sd   = str(sale_desc  or "").strip().upper()
    it   = str(instr_type or "").strip().upper()
    pr   = price or 0

    # Nominal / token consideration
    if pr <= 1:
        return False, "Nominal price ($1 or $0)"

    # Check sale code
    reason = _NON_ARMS_LENGTH.get(sc) or _NON_ARMS_LENGTH.get(it)
    if reason:
        return False, reason

    # Keyword scan of description
    for kw, r in _NON_ARMS_LENGTH.items():
        if len(kw) > 2 and kw in sd:
            return False, r

    # Estate sales: keep but flag
    if sc in _ESTATE_CODES or "ESTATE" in sd:
        return True, "Estate sale — may be below market"

    return True, ""


class CountyParserBase:
    county_name: str = "Unknown"
    state_abbr: str  = "XX"
    # Signature: set of uppercase column names that uniquely identify this county's export
    signature: frozenset = frozenset()

    @classmethod
    def matches(cls, columns: list[str]) -> bool:
        upper = {c.upper() for c in columns}
        return cls.signature.issubset(upper)

    @classmethod
    def parse(cls, df: pd.DataFrame, source_file: str) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _sale_id(cls, parid: str, sale_date: str) -> str:
        key = f"{cls.county_name}:{parid}:{sale_date}"
        return hashlib.md5(key.encode()).hexdigest()[:14]


class AlleghenyParser(CountyParserBase):
    """
    Parser for Allegheny County, PA property sales export.
    Source: https://apps.county.allegheny.pa.us/SaleSearch/SaleSearch
    Columns: _id, PARID, FULL_ADDRESS, PROPERTYHOUSENUM, PROPERTYFRACTION,
             PROPERTYADDRESSDIR, PROPERTYADDRESSSTREET, PROPERTYADDRESSSUF,
             PROPERTYADDRESSUNITDESC, PROPERTYUNITNO, PROPERTYCITY,
             PROPERTYSTATE, PROPERTYZIP, SCHOOLCODE, SCHOOLDESC,
             MUNICODE, MUNIDESC, RECORDDATE, SALEDATE, PRICE,
             DEEDBOOK, DEEDPAGE, SALECODE, SALEDESC, INSTRTYP, INSTRTYPDESC
    """
    county_name = "Allegheny, PA"
    state_abbr  = "PA"
    signature   = frozenset({"PARID", "MUNIDESC", "SCHOOLDESC",
                              "SALECODE", "SALEDESC", "INSTRTYP"})

    @classmethod
    def parse(cls, df: pd.DataFrame, source_file: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip().str.upper()

        def col(name):
            return df[name] if name in df.columns else pd.Series([None] * len(df))

        # ── Build address ────────────────────────────────────────────────
        import math as _math

        def _clean(val) -> str:
            """Return empty string for None, NaN, 'nan', 'none', 'null'."""
            if val is None:
                return ""
            if isinstance(val, float) and _math.isnan(val):
                return ""
            s = str(val).strip()
            return "" if s.lower() in ("nan", "none", "null", "<na>", "nat") else s

        def _build_street(row):
            parts = [
                _clean(row.get("PROPERTYHOUSENUM")),
                _clean(row.get("PROPERTYADDRESSDIR")),
                _clean(row.get("PROPERTYADDRESSSTREET")),
                _clean(row.get("PROPERTYADDRESSSUF")),
            ]
            unit   = _clean(row.get("PROPERTYUNITNO"))
            street = " ".join(p for p in parts if p)
            if unit:
                street += f" #{unit}"
            return street.strip()

        street = df.apply(_build_street, axis=1)
        city   = col("PROPERTYCITY").apply(_clean).str.title()
        state  = col("PROPERTYSTATE").apply(lambda v: _clean(v) or "PA").str.upper()
        zip_   = col("PROPERTYZIP").apply(_clean).str[:5]

        # Drop rows where the street came out entirely empty
        valid_street = street.str.strip().ne("")
        if not valid_street.all():
            n_blank = (~valid_street).sum()
            print(f"    Skipping {n_blank} rows with no street address")
        df    = df[valid_street].copy()
        street= street[valid_street]
        city  = city[valid_street]
        state = state[valid_street]
        zip_  = zip_[valid_street]

        full_address = (
            street + ", " + city + ", " + state + " " + zip_
        ).str.strip(", ")

        # ── Sale price ───────────────────────────────────────────────────
        sold_price = pd.to_numeric(
            col("PRICE").astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce"
        )

        # ── Dates ────────────────────────────────────────────────────────
        sold_date   = pd.to_datetime(col("SALEDATE"),   errors="coerce").dt.date
        record_date = pd.to_datetime(col("RECORDDATE"), errors="coerce").dt.date

        # ── Arm's-length classification ──────────────────────────────────
        sale_codes  = col("SALECODE").astype(str).str.strip()
        sale_descs  = col("SALEDESC").astype(str).str.strip()
        instr_types = col("INSTRTYP").astype(str).str.strip()

        arms_length_rows = [
            _classify_arms_length(sc, sd, it, pr)
            for sc, sd, it, pr in zip(
                sale_codes, sale_descs, instr_types, sold_price.fillna(0)
            )
        ]
        is_arms_length = [r[0] for r in arms_length_rows]
        arms_length_flag = [r[1] for r in arms_length_rows]

        # ── Parcel ID ─────────────────────────────────────────────────────
        parid = col("PARID").astype(str).str.strip()

        # ── Sale ID ──────────────────────────────────────────────────────
        sale_id = pd.Series([
            cls._sale_id(p, str(d))
            for p, d in zip(parid, sold_date)
        ])

        # ── County FIPS for Allegheny = 42003 ────────────────────────────
        county_fips = "42003"

        out = pd.DataFrame({
            "sale_id":          sale_id,
            "source_county":    cls.county_name,
            "source_file":      source_file,
            "parid":            parid,
            "address":          street,
            "city":             city,
            "state":            state,
            "zip":              zip_,
            "full_address":     full_address,
            "sold_price":       sold_price,
            "sold_date":        sold_date,
            "record_date":      record_date,
            "list_price":       None,
            "sqft":             None,
            "beds":             None,
            "baths":            None,
            "year_built":       None,
            "sale_code":        sale_codes,
            "sale_desc":        col("SALEDESC").astype(str).str.strip(),
            "instr_type":       instr_types,
            "instr_type_desc":  col("INSTRTYPDESC").astype(str).str.strip()
                                    if "INSTRTYPDESC" in df.columns else None,
            "deed_book":        col("DEEDBOOK").astype(str).str.strip(),
            "deed_page":        col("DEEDPAGE").astype(str).str.strip(),
            "muni_code":        col("MUNICODE").astype(str).str.strip(),
            "muni_desc":        col("MUNIDESC").astype(str).str.strip().str.strip(),
            "school_code":      col("SCHOOLCODE").astype(str).str.strip(),
            "school_desc":      col("SCHOOLDESC").astype(str).str.strip(),
            "is_arms_length":   is_arms_length,
            "arms_length_flag": arms_length_flag,
            "lat":              None,
            "lon":              None,
            "geocode_status":   "pending",
            "geocode_source":   None,
            "geocode_accuracy": None,
            "tract_fips":       None,
            "county_fips":      county_fips,
        })

        return out


class GenericSoldParser(CountyParserBase):
    """
    Fallback parser for CSV files that already have standard columns
    (e.g. the original sample_sold.csv with lat/lon included).
    Tries to map common column names to the standard schema.
    """
    county_name = "Unknown"
    signature   = frozenset()  # always matches as last resort

    @classmethod
    def matches(cls, columns):
        return True   # fallback — always try this last

    @classmethod
    def parse(cls, df: pd.DataFrame, source_file: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()
        col_rename = {c: _REDFIN_COL_MAP.get(c, c) for c in df.columns}
        df = df.rename(columns=col_rename)

        for c in ["sold_price", "list_price", "sqft", "beds", "baths"]:
            if c in df.columns:
                df[c] = _clean_numeric(df[c])

        if "sold_date" in df.columns:
            df["sold_date"] = pd.to_datetime(df["sold_date"], errors="coerce").dt.date

        if "address" not in df.columns:
            df["address"] = ""
        city  = df.get("city",  pd.Series([""] * len(df)))
        state = df.get("state", pd.Series([""] * len(df)))
        zip_  = df.get("zip",   pd.Series([""] * len(df)))

        df["full_address"] = (
            df["address"].astype(str) + ", " +
            city.astype(str) + ", " +
            state.astype(str) + " " +
            zip_.astype(str)
        ).str.strip(", ")

        # Generic files may already have lat/lon
        if "lat" in df.columns and "lon" in df.columns:
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            geocode_status = df["lat"].notna().map(
                lambda v: "manual" if v else "pending"
            )
        else:
            df["lat"] = None
            df["lon"] = None
            geocode_status = "pending"

        df["sale_id"] = df.apply(
            lambda r: hashlib.md5(
                f"generic:{r.get('address','')}{r.get('sold_date','')}".encode()
            ).hexdigest()[:14], axis=1
        )

        # Classify arm's-length (best effort with generic data)
        if "sale_code" in df.columns and "sale_desc" in df.columns:
            alc = [_classify_arms_length(sc, sd, "", pr)
                   for sc, sd, pr in zip(
                       df["sale_code"].fillna(""),
                       df["sale_desc"].fillna(""),
                       df.get("sold_price", pd.Series([0]*len(df))).fillna(0)
                   )]
        else:
            price = df.get("sold_price", pd.Series([0]*len(df))).fillna(0)
            alc = [(p > 1, "" if p > 1 else "Nominal price") for p in price]

        schema_cols = [
            "sale_id", "source_county", "source_file", "parid",
            "address", "city", "state", "zip", "full_address",
            "sold_price", "sold_date", "record_date", "list_price",
            "sqft", "beds", "baths", "year_built",
            "sale_code", "sale_desc", "instr_type", "instr_type_desc",
            "deed_book", "deed_page", "muni_code", "muni_desc",
            "school_code", "school_desc",
            "is_arms_length", "arms_length_flag",
            "lat", "lon", "geocode_status", "geocode_source", "geocode_accuracy",
            "tract_fips", "county_fips",
        ]
        for c in schema_cols:
            if c not in df.columns:
                df[c] = None

        df["source_county"]    = cls.county_name
        df["source_file"]      = source_file
        df["is_arms_length"]   = [r[0] for r in alc]
        df["arms_length_flag"] = [r[1] for r in alc]
        df["geocode_status"]   = geocode_status

        return df[schema_cols]


# Registry: listed in priority order; GenericSoldParser must be last
_COUNTY_PARSERS = [AlleghenyParser, GenericSoldParser]


def _detect_parser(columns: list[str]) -> type:
    """Return the first parser whose signature matches the file's columns."""
    upper = [c.upper() for c in columns]
    for parser in _COUNTY_PARSERS[:-1]:   # skip generic
        if parser.matches(upper):
            return parser
    return GenericSoldParser


# Full schema column list — must match sold_homes table exactly
_SOLD_SCHEMA_COLS = [
    "sale_id", "source_county", "source_file", "parid",
    "address", "city", "state", "zip", "full_address",
    "lat", "lon", "geocode_status", "geocode_source", "geocode_accuracy",
    "sold_price", "sold_date", "record_date", "list_price",
    "sqft", "beds", "baths", "year_built",
    "sale_code", "sale_desc", "instr_type", "instr_type_desc",
    "deed_book", "deed_page", "muni_code", "muni_desc",
    "school_code", "school_desc",
    "is_arms_length", "arms_length_flag",
    "tract_fips", "county_fips",
]


def load_sold_homes(run_geocoding: bool = True,
                    geo_utils=None) -> int:
    """
    Load all sold-homes CSV files from data/sold/.
    Auto-detects the county format and applies the correct parser.
    Runs geocoding (Census Batch → Census Single → Nominatim) for any rows
    without lat/lon, using a DuckDB cache to avoid duplicate API calls.

    Parameters
    ----------
    run_geocoding : If True (default), geocode addresses without coordinates.
    geo_utils     : If provided, also run spatial join for any rows where
                    geocoding returned lat/lon but no tract_fips.
    """
    from services.geocoder import geocode_dataframe

    sold_dir = settings.sold_dir
    csv_files = list(sold_dir.glob("*.csv"))
    if not csv_files:
        print("  No sold homes CSV files found in data/sold/")
        return 0

    all_rows = []
    for path in csv_files:
        print(f"  Loading sold: {path.name}")
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            raw = pd.read_csv(path, encoding="latin-1", low_memory=False)

        parser = _detect_parser(list(raw.columns))
        print(f"    Parser: {parser.__name__} ({parser.county_name})")

        try:
            parsed = parser.parse(raw, source_file=path.name)
        except Exception as e:
            print(f"    Parse error: {e}")
            import traceback; traceback.print_exc()
            continue

        # Ensure all schema columns exist
        for col in _SOLD_SCHEMA_COLS:
            if col not in parsed.columns:
                parsed[col] = None

        all_rows.append(parsed[_SOLD_SCHEMA_COLS])
        n_al  = int((parsed["is_arms_length"] == True).sum())
        n_nal = int((parsed["is_arms_length"] == False).sum())
        print(f"    {len(parsed)} records: "
              f"{n_al} arm's-length, {n_nal} non-arm's-length flagged")

    if not all_rows:
        return 0

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=["sale_id"])

    # ── Geocoding ─────────────────────────────────────────────────────────
    needs_geo = combined["lat"].isna() | combined["lon"].isna()
    if run_geocoding and needs_geo.any():
        print(f"  Geocoding {needs_geo.sum()} addresses …")
        combined = geocode_dataframe(
            combined,
            street_col="address",
            city_col="city",
            state_col="state",
            zip_col="zip",
            full_addr_col="full_address",
            verbose=True,
        )

    # ── Spatial join for any still missing tract_fips ─────────────────────
    needs_tract = combined["tract_fips"].isna() & combined["lat"].notna()
    if geo_utils and needs_tract.any():
        print(f"  Resolving {needs_tract.sum()} tract FIPS via spatial join …")
        combined = geo_utils.assign_tract_fips(combined)

    store.upsert_df("sold_homes", combined)

    n_geocoded = int((combined["geocode_status"] == "success").sum())
    n_pending  = int(combined["geocode_status"].isna().sum() +
                     (combined["geocode_status"] == "pending").sum())
    print(f"  ✓ {len(combined)} sold homes loaded "
          f"({n_geocoded} geocoded, {n_pending} pending)")
    return len(combined)
