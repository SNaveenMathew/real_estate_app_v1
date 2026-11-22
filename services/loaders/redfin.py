"""`Redfin` loaders, moved out of services/data_loader.py.

Behaviour is unchanged: the facade in services/data_loader.py re-exports every name here, so
`from services.data_loader import ...` (and scripts that call it) keep working.
"""
from __future__ import annotations

"""
Data loader: reads raw files and populates DuckDB tables.
Run via setup_data.py — only needed when data files change.
"""
import json
import hashlib
from typing import Optional

import pandas as pd
import numpy as np


from config import settings
import db.duckdb_store as store
from services.crime_sources import CRIME_CITY_KEYS
from services.zillow_sources import normalize_zip5




__all__ = ['_REDFIN_COL_MAP', '_house_id', '_normalize_crime_city', '_infer_crime_city', '_clean_numeric', '_extract_snapshot_date', 'load_redfin']

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


def _normalize_crime_city(value) -> Optional[str]:
    """Lowercase/trim a crime_city value from a Redfin CSV so it reliably
    matches crime_incidents.city (always a lowercase key, e.g. 'chicago')."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    return s or None


def _infer_crime_city(city_value) -> Optional[str]:
    """
    Best-effort fallback when a Redfin CSV doesn't already set crime_city:
    if the house's own `city` field is an exact (case-insensitive) match for
    one of the cities covered under data/crime/, use it.

    This only catches houses that are literally IN a covered city — a house
    in a suburb (e.g. Cambridge, for Boston's crime data) won't match and
    stays NULL. Set crime_city explicitly in the Redfin CSV for those; an
    explicit value always wins over this inference (see load_redfin).
    """
    if city_value is None or (isinstance(city_value, float) and pd.isna(city_value)):
        return None
    norm = str(city_value).strip().lower()
    return norm if norm in CRIME_CITY_KEYS else None


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
        # Peek the header so the zip column can be forced to text at parse
        # time. Without this, pandas infers an all-digit zip column as a
        # number and silently drops any leading zero (e.g. Cambridge, MA
        # "02139" -> 2139) — normalize_zip5() below can't recover a leading
        # zero that's already gone, so it has to not be dropped in the first
        # place.
        try:
            header_cols = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
        except Exception as e:
            print(f"    Error: {e}")
            continue
        zip_like_raw = {"zip", "zip/postal code"}   # raw headers _REDFIN_COL_MAP maps to "zip"
        dtype_overrides = {c: str for c in header_cols if c.strip().lower() in zip_like_raw}
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                              dtype=dtype_overrides or None)
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

        # Canonical 5-digit zip (see normalize_zip5 docstring) — this is what
        # ties a house to its zip-level ZHVI/Market Heat Index series.
        if "zip" in df.columns:
            df["zip"] = df["zip"].apply(normalize_zip5)

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

    # crime_city: the join key into crime_incidents.city (see
    # services/crime_sources.py and the houses<->crime_incidents Relationship
    # in db/schema_catalog.py). Normalize whatever the CSV provided; infer it
    # from `city` when the CSV didn't set it and the house is directly in one
    # of the covered cities.
    if "crime_city" not in combined.columns:
        combined["crime_city"] = None
    combined["crime_city"] = combined["crime_city"].apply(_normalize_crime_city)
    needs_inference = combined["crime_city"].isna()
    if needs_inference.any():
        combined.loc[needs_inference, "crime_city"] = (
            combined.loc[needs_inference, "city"].apply(_infer_crime_city)
        )
    if "nearest_big_city" not in combined.columns:
        combined["nearest_big_city"] = None

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
