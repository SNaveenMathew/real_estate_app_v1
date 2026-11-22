"""`Zillow Research Data (ZHVI + Market Heat Index)` loaders, moved out of services/data_loader.py.

Behaviour is unchanged: the facade in services/data_loader.py re-exports every name here, so
`from services.data_loader import ...` (and scripts that call it) keep working.
"""
from __future__ import annotations

"""
Data loader: reads raw files and populates DuckDB tables.
Run via setup_data.py — only needed when data files change.
"""
import math

import pandas as pd


from config import settings
import db.duckdb_store as store
from services.zillow_sources import ZhviParser, MarketHeatIndexParser



from services.loaders.census import _build_cbsa_lookup, _match_msa_to_cbsa

__all__ = ['_ZILLOW_ID_COLS', '_ZILLOW_UPSERT_CHUNK_ROWS', '_compute_zillow_msa_codes', '_finish_zillow_load', 'load_zhvi', 'load_market_heat_index']

# ── Zillow Research Data (ZHVI + Market Heat Index) ──────────────────────────
# Source: https://www.zillow.com/research/data/ — see services/zillow_sources.py
# for the shared wide-to-long parser both of these build on.

_ZILLOW_ID_COLS = ["region_id", "region_type", "region_name", "size_rank",
                   "state_name", "state", "city", "metro", "county_name",
                   "msa_code", "date"]

# A neighborhood- or zip-level ZHVI export can melt out to several million rows.
# INSERT OR REPLACE against a same-sized existing table (re-running the loader
# after the first load) has to reconcile the full old + new key sets at once,
# so it's upserted in chunks to keep peak memory bounded regardless of file size.
_ZILLOW_UPSERT_CHUNK_ROWS = 500_000


def _compute_zillow_msa_codes(df: pd.DataFrame) -> pd.Series:
    """
    Resolve an `msa_code` (CBSA code) for every row of a melted ZHVI/Market
    Heat Index dataframe, so a house can be joined to the right metro-level
    row with a plain equality (houses.msa_code = zhvi.msa_code) instead of
    fuzzy-matching Zillow's metro names at query time.

    Reuses the exact tiered MSA-name matcher already proven for census_msa
    (_build_cbsa_lookup / _match_msa_to_cbsa, defined above) rather than a
    new, separately-maintained matching scheme — Zillow's metro names are
    the short "City, ST" form (e.g. "New York, NY"), which that matcher's
    tier 2 ((first_city, first_state) tuple) already handles correctly
    against the full CBSA titles.

    Which column feeds the matcher depends on the row's own geography:
      - region_type == 'metro'                        -> its own region_name
      - region_type in (zip, neighborhood, city, ...)  -> its `metro` column
      - region_type in (state, country), or no metro info -> NULL (no MSA)

    Only the *distinct* lookup keys get matched (a few hundred metro names at
    most, never one per row), then mapped back — this stays cheap even for a
    multi-million-row neighborhood- or zip-level file.
    """
    lookup_key = pd.Series(pd.NA, index=df.index, dtype="object")
    is_metro = df["region_type"] == "metro"
    lookup_key.loc[is_metro] = df.loc[is_metro, "region_name"]
    if "metro" in df.columns:
        other = (~is_metro) & df["metro"].notna() & (df["metro"].astype(str).str.strip() != "")
        lookup_key.loc[other] = df.loc[other, "metro"]

    distinct_keys = [k for k in lookup_key.dropna().unique().tolist() if str(k).strip()]
    if not distinct_keys:
        return pd.Series(None, index=df.index, dtype="object")

    try:
        cbsa_df = store.query("SELECT DISTINCT cbsa_code, cbsa_title FROM cbsa_counties")
    except Exception:
        cbsa_df = pd.DataFrame(columns=["cbsa_code", "cbsa_title"])

    if cbsa_df.empty:
        print("    Note: cbsa_counties is empty, so msa_code will be NULL for this load "
              "(metro-level fallback matching won't be available for these rows). "
              "Run `python setup_data.py --only census` first to enable it, then reload.")
        return pd.Series(None, index=df.index, dtype="object")

    cbsa_lookup = _build_cbsa_lookup(cbsa_df)
    resolved = {k: _match_msa_to_cbsa(k, cbsa_lookup) for k in distinct_keys}
    n_matched = sum(1 for v in resolved.values() if v)
    print(f"    Matched {n_matched}/{len(distinct_keys)} distinct metro name(s) to a CBSA code")

    return lookup_key.map(resolved)


def _finish_zillow_load(long_df: pd.DataFrame, value_col: str, table: str, label: str) -> int:
    """Shared tail end of load_zhvi()/load_market_heat_index(): resolve msa_code,
    align columns to the DuckDB schema, drop duplicate (region, date) rows, and
    upsert in chunks."""
    if long_df.empty:
        print(f"  No {label} data loaded.")
        return 0

    # msa_code isn't produced by the parser (services/zillow_sources.py) — it's
    # resolved here against the app's own cbsa_counties crosswalk (the same one
    # census_msa already depends on; see load_cbsa_crosswalk()).
    long_df["msa_code"] = _compute_zillow_msa_codes(long_df)

    cols = _ZILLOW_ID_COLS + [value_col, "source_file"]
    for c in cols:
        if c not in long_df.columns:
            long_df[c] = None
    out = long_df[cols].copy()
    del long_df

    out["region_id"] = out["region_id"].astype(str)
    out["region_type"] = out["region_type"].astype(object).where(out["region_type"].notna(), None)

    before = len(out)
    out = out.drop_duplicates(subset=["region_id", "region_type", "date"], keep="last")
    dropped = before - len(out)
    if dropped:
        print(f"  Deduplicated {dropped:,} row(s) with a repeated (region, region_type, date)")

    total = len(out)
    n_chunks = max(1, math.ceil(total / _ZILLOW_UPSERT_CHUNK_ROWS))
    for i in range(n_chunks):
        chunk = out.iloc[i * _ZILLOW_UPSERT_CHUNK_ROWS: (i + 1) * _ZILLOW_UPSERT_CHUNK_ROWS]
        store.upsert_df(table, chunk)
        if n_chunks > 1:
            print(f"    Upserted chunk {i + 1}/{n_chunks} ({len(chunk):,} rows)")

    n_regions = out["region_id"].nunique()
    print(f"  ✓ {total:,} {label} rows loaded across {n_regions:,} regions")
    return total


def load_zhvi() -> int:
    """
    Load Zillow Home Value Index exports from data/zhvi/ into the `zhvi` table.

    Accepts any of Zillow's standard geography-level exports (state, metro,
    county, city, zip, neighborhood) — drop any number of them into the
    folder; `region_type` distinguishes them in the resulting table. Safe to
    call when the directory doesn't exist yet.

    Download: https://www.zillow.com/research/data/ -> Data Type: "ZHVI" ->
    pick a geography + home type/tier (Smoothed, Seasonally Adjusted ($) is
    the typical choice) -> save the CSV into data/zhvi/.
    """
    long_df = ZhviParser.load_dir(settings.zhvi_dir)
    return _finish_zillow_load(long_df, ZhviParser.value_column, ZhviParser.table, ZhviParser.label)


def load_market_heat_index() -> int:
    """
    Load Zillow Market Heat Index exports from data/market_heat_index/ into
    the `market_heat_index` table. Same multi-geography handling as
    load_zhvi(). Safe to call when the directory doesn't exist yet.

    Download: https://www.zillow.com/research/data/ -> Data Type: "Market
    Heat Index" -> pick a geography -> save the CSV into data/market_heat_index/.
    """
    long_df = MarketHeatIndexParser.load_dir(settings.market_heat_dir)
    return _finish_zillow_load(long_df, MarketHeatIndexParser.value_column,
                                MarketHeatIndexParser.table, MarketHeatIndexParser.label)
