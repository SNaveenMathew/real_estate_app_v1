"""
services/zillow_sources.py

Parsers for Zillow's standard wide-format research exports — currently used
for two data/<key>/ folders: data/zhvi/ (Zillow Home Value Index) and
data/market_heat_index/ (Zillow Market Heat Index), both downloaded from
https://www.zillow.com/research/data/.

Why one shared module instead of a per-city-style registry (contrast with
services/crime_sources.py)
-----------------------------------------------------------------------------
Every Zillow research file — regardless of metric or geography level — uses
the exact same shape: a handful of identifying columns followed by one column
per month, header = the month-end date (e.g. "2024-01-31"), one row per
region. Zillow publishes this same shape at every geography level (state,
metro, county, city, zip, neighborhood); which identifying columns are
present varies by level (e.g. a neighborhood or zip export adds State/City/
Metro/CountyName that a metro-level export omits), so parsing detects
whichever of the standard columns exist rather than assuming one fixed set.
Any number of files/geography levels can be dropped in the same folder —
they melt to one long table together, distinguished by `region_type`.

Standardized long-format schema returned by ZillowSourceBase.load_dir()
-----------------------------------------------------------------------------
    region_id     Zillow's RegionID
    region_type   'country' | 'state' | 'metro' | 'county' | 'city' | 'zip' | 'neighborhood'
    region_name   display name, e.g. 'New York, NY' or 'Maryvale'
    size_rank     Zillow's popularity/size ranking for the region (0 = largest)
    state_name    two-letter state, when the file includes it
    state         two-letter state (alternate column some exports use instead)
    city          containing city, when present (zip/neighborhood exports)
    metro         containing metro name, when present (zip/neighborhood/city exports)
    county_name   containing county, when present (zip/neighborhood/city exports)
    date          month-end date, parsed from the column header
    <value_col>   the metric itself — see each subclass's `value_column`
    source_file   originating filename, for traceability

services/data_loader.py::load_zhvi() / load_market_heat_index() add nothing
to this shape beyond deduplication before calling db.duckdb_store.upsert_df —
the DuckDB table columns match this schema 1:1 (see db/duckdb_store.py).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

# Matches Zillow's month-end column headers, e.g. "2024-01-31".
_DATE_COL_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_ZIP_DIGITS_RE = re.compile(r"[^0-9]")

# Zillow's raw RegionType value for a metro-level row is literally "msa" (see
# any real ZHVI/Market Heat Index export). Every other geography level's raw
# value already reads naturally (state/county/city/zip/neighborhood/country),
# so "msa" is normalized to "metro" here — the one place this parser's output
# schema is decided — rather than leaking Zillow's raw term into every
# downstream consumer (table docs, the geography-fallback tiers in
# db/duckdb_store.py, the catalog concepts, and any future one).
_REGION_TYPE_ALIASES = {"msa": "metro"}


def normalize_zip5(value) -> Optional[str]:
    """
    Normalize any raw zip-ish value to a clean 5-digit US zip string, or None
    if it doesn't look like one. Shared by this module (Zillow's own zip-level
    RegionName, which should already be clean, but this is a cheap defensive
    pass) and services/data_loader.py (houses.zip, which comes from arbitrary
    Redfin CSV exports and is NOT reliably clean — see load_redfin()).

    Handles the two real-world failure modes for US zips:
    - ZIP+4 ("15213-2622") -> keep only the first 5 digits ("15213")
    - Leading zeros dropped by numeric parsing (2139 -> "2139") -> zero-pad
      back to 5 ("02139"). This happens whenever a zip column gets read as a
      number instead of text somewhere upstream (spreadsheet edit, a CSV
      loaded without dtype=str, etc.) — as it can if not handled explicitly.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return None
    digits = _ZIP_DIGITS_RE.sub("", s)
    if not digits:
        return None
    digits = digits[:5]                # ZIP+4 or an accidentally-concatenated value: keep the first 5
    if len(digits) < 5:
        digits = digits.zfill(5)       # leading zero(s) previously stripped
    return digits

# Logical name -> candidate raw column names (matched case-insensitively).
# Only "region_id" and "region_name" are required; the rest are filled in
# with NULL when a given geography level's export doesn't include them.
_ID_COLUMNS: dict[str, list[str]] = {
    "region_id":   ["RegionID"],
    "region_type": ["RegionType"],
    "region_name": ["RegionName"],
    "size_rank":   ["SizeRank"],
    "state_name":  ["StateName"],
    "state":       ["State"],
    "city":        ["City"],
    "metro":       ["Metro"],
    "county_name": ["CountyName"],
}
_REQUIRED = ["region_id", "region_name"]


def _normalize_key(s: str) -> str:
    return str(s).strip().lower()


def _find_col(available_cols, candidates: list[str]) -> Optional[str]:
    norm_map = {_normalize_key(c): c for c in available_cols}
    for cand in candidates:
        key = _normalize_key(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _peek_header(path: Path) -> list[str]:
    """Read just the header row — cheap, and lets the full read below use the
    right dtype per column instead of loading everything as text."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return list(pd.read_csv(path, nrows=0, encoding=enc).columns)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("could not decode header as UTF-8 or Latin-1")


def _read_selected(path: Path, dtype: dict, usecols: list[str]) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, dtype=dtype, usecols=usecols, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("could not decode as UTF-8 or Latin-1")


def melt_wide_file(path: Path, value_name: str) -> Optional[pd.DataFrame]:
    """
    Read one Zillow wide-format export and melt it to the standardized long
    schema described in the module docstring. Returns None (after printing
    why) instead of raising, so one bad/unexpected file in a folder full of
    good ones doesn't abort the whole load.

    Reads only the columns actually needed, with the ~300+ month columns left
    to pandas' native numeric dtype instead of forced to text — a neighborhood-
    or zip-level export can carry millions of cells across those columns, and
    reading them all as strings (then converting after melting) roughly
    doubles peak memory and parse time for no benefit, since every one of them
    ends up numeric anyway.
    """
    try:
        header_cols = _peek_header(path)
    except Exception as e:
        print(f"    Skipping {path.name}: could not read file ({e})")
        return None

    resolved = {logical: _find_col(header_cols, cands) for logical, cands in _ID_COLUMNS.items()}
    missing_required = [f for f in _REQUIRED if not resolved.get(f)]
    if missing_required:
        preview = header_cols[:12] + (["..."] if len(header_cols) > 12 else [])
        print(f"    Skipping {path.name}: missing required column(s) {missing_required} "
              f"— doesn't look like a Zillow region export (columns found: {preview})")
        return None

    date_cols = [c for c in header_cols if _DATE_COL_RE.match(str(c).strip())]
    if not date_cols:
        print(f"    Skipping {path.name}: no date columns found "
              f"(expected headers like '2024-01-31')")
        return None

    id_cols_present = {logical: col for logical, col in resolved.items() if col}
    usecols = list(id_cols_present.values()) + date_cols
    dtype = {col: str for col in id_cols_present.values()}   # date columns: let pandas infer (numeric)

    try:
        raw = _read_selected(path, dtype=dtype, usecols=usecols)
    except Exception as e:
        print(f"    Skipping {path.name}: could not read file ({e})")
        return None

    df = raw.rename(columns={col: logical for logical, col in id_cols_present.items()})

    long = df.melt(
        id_vars=list(id_cols_present.keys()),
        value_vars=date_cols,
        var_name="date",
        value_name=value_name,
    )
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")

    before = len(long)
    long = long.dropna(subset=["date", value_name])
    dropped = before - len(long)
    if dropped:
        print(f"    Dropped {dropped:,} row(s) with a missing date or value")

    # Fill in any logical column this particular file didn't have, so every
    # file's output lines up under the same columns before concatenation.
    for logical in _ID_COLUMNS:
        if logical not in long.columns:
            long[logical] = None

    if "size_rank" in long.columns:
        long["size_rank"] = pd.to_numeric(long["size_rank"], errors="coerce").astype("Int64")
    if "region_type" in long.columns:
        long["region_type"] = long["region_type"].astype(object).where(long["region_type"].notna(), None)
        long["region_type"] = long["region_type"].apply(
            lambda v: _REGION_TYPE_ALIASES.get(str(v).strip().lower(), str(v).strip().lower()) if v is not None else None
        )

    # Defensive: Zillow's own zip-level exports are already clean 5-digit
    # zips, but normalize anyway in case a future export or a manually-edited
    # file isn't — this is the same normalization applied to houses.zip
    # (services/data_loader.py::load_redfin), so the two sides of a zip-level
    # join are guaranteed to compare equal regardless of either source's
    # original formatting.
    if "region_type" in long.columns and "region_name" in long.columns:
        is_zip = long["region_type"] == "zip"
        if is_zip.any():
            long.loc[is_zip, "region_name"] = long.loc[is_zip, "region_name"].apply(normalize_zip5)

    long["source_file"] = path.name
    return long


class ZillowSourceBase:
    """
    One subclass per Zillow research export type. Both currently defined
    sources share the exact same wide-file shape (see module docstring), so
    load_dir() needs no per-source override — only `value_column`, `table`
    and `label` differ. Add a new Zillow metric (e.g. ZORI, days-to-pending)
    by subclassing this with those three attributes set.
    """
    key: str = "unknown"            # matches the data/<key>/ folder name
    label: str = "Unknown"
    value_column: str = "value"     # name of the metric column in the long output
    table: str = "unknown"          # destination DuckDB table (db/duckdb_store.py)

    @classmethod
    def load_dir(cls, source_dir: Path) -> pd.DataFrame:
        files = sorted(
            p for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".csv"
        ) if source_dir.exists() else []

        frames = []
        for path in files:
            print(f"    Reading {path.name}...")
            long = melt_wide_file(path, cls.value_column)
            if long is None or long.empty:
                if long is not None:
                    print(f"    No usable rows in {path.name}")
                continue
            n_regions = long["region_id"].nunique()
            types = ", ".join(sorted(t for t in long["region_type"].dropna().unique().tolist()))
            date_lo, date_hi = long["date"].min(), long["date"].max()
            print(f"      \u2713 {n_regions:,} regions"
                  f"{' (' + types + ')' if types else ''}, "
                  f"{date_lo.date()} to {date_hi.date()}")
            frames.append(long)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


class ZhviParser(ZillowSourceBase):
    """Zillow Home Value Index — smoothed, seasonally-adjusted typical home value ($)."""
    key = "zhvi"
    label = "Zillow Home Value Index (ZHVI)"
    value_column = "home_value"
    table = "zhvi"


class MarketHeatIndexParser(ZillowSourceBase):
    """Zillow Market Heat Index — ~0-100+ buyer/seller market temperature (higher = hotter/more seller-favorable)."""
    key = "market_heat_index"
    label = "Zillow Market Heat Index"
    value_column = "heat_index"
    table = "market_heat_index"


ZILLOW_PARSERS: list[type[ZillowSourceBase]] = [ZhviParser, MarketHeatIndexParser]
