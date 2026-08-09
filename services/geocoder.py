"""
Address geocoder for sold-homes data (county records with no lat/lon).

Strategy
--------
1. DuckDB geocode_cache  — instant, no HTTP for already-seen addresses
2. Census Batch Geocoder — free, no key, up to 900 addresses per request,
                           returns lat/lon AND census tract FIPS in one shot
3. Single-address fallback — opt-in only, hard cap (default 0 for bulk loads).
                             Use `--only geocode` to retry stragglers later.

Key fixes vs the previous version
----------------------------------
- Batch response is parsed with Python's `csv` module, not str.split(",").
  The Census API wraps matched addresses in quotes because they contain commas.
  Naive splitting shifts every column right → everything scores No_Match →
  31k rows fall into the per-row single fallback → hours of blocking I/O.
- Results are written to cache AND to sold_homes after EVERY batch chunk,
  so Ctrl+C loses at most one chunk (~900 rows), not the whole run.
- Single-address fallback is capped at `single_fallback_limit` (default 0).
  Unresolved rows get geocode_status='pending', not 'failed', so that
  `python setup_data.py --only geocode` can retry them later.
"""

import csv
import io
import time
import hashlib
import requests
import pandas as pd
from typing import Optional

import db.duckdb_store as store


# ── Geocode cache ─────────────────────────────────────────────────────────────

def _ensure_cache_table():
    store.get_conn().execute("""
        CREATE TABLE IF NOT EXISTS geocode_cache (
            address_key     VARCHAR PRIMARY KEY,  -- MD5 of normalised address string
            full_address    VARCHAR,
            lat             DOUBLE,
            lon             DOUBLE,
            tract_fips      VARCHAR,
            geocode_source  VARCHAR,
            geocode_accuracy VARCHAR,
            geocoded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def _address_key(full_address: str) -> str:
    return hashlib.md5(full_address.strip().upper().encode()).hexdigest()


def _lookup_cache(keys: list[str]) -> dict[str, dict]:
    """Return {address_key: result_dict} for any addresses already cached."""
    _ensure_cache_table()
    if not keys:
        return {}
    placeholders = ", ".join("?" * len(keys))
    rows = store.query_json(
        f"SELECT * FROM geocode_cache WHERE address_key IN ({placeholders})", keys
    )
    return {r["address_key"]: r for r in rows}


def _save_cache(results: list[dict]):
    """Persist geocode results to cache."""
    _ensure_cache_table()
    if not results:
        return
    _ensure_cache_table()
    df = pd.DataFrame(results)
    # Explicit columns — must match geocode_cache schema exactly (no created_at)
    cache_cols = ["address_key", "full_address", "lat", "lon",
                  "tract_fips", "geocode_source", "geocode_accuracy"]
    for col in cache_cols:
        if col not in df.columns:
            df[col] = None
    df = df[cache_cols]
    conn = store.get_conn()
    # table_name = 'geocode_cache'
    # schema_result = conn.sql(f"DESCRIBE {table_name}")
    # schema_cols = schema_result.to_df()['column_name'].tolist()
    # for col in schema_cols:
    #     if col not in df.columns:
    #         df[col] = None
    
    # df = df[schema_cols]
    # conn.register("__geo_tmp", df)
    # conn.execute("INSERT OR REPLACE INTO geocode_cache SELECT * FROM __geo_tmp")
    # conn.unregister("__geo_tmp")
    conn.register("__geo_cache_tmp", df)
    conn.execute(
        f"INSERT OR REPLACE INTO geocode_cache "
        f"({', '.join(cache_cols)}) "
        f"SELECT {', '.join(cache_cols)} FROM __geo_cache_tmp"
    )
    conn.unregister("__geo_cache_tmp")


def _flush_to_sold_homes(updates: list[dict]):
    """Write geocode results into sold_homes immediately so progress survives Ctrl+C."""
    if not updates:
        return
    conn = store.get_conn()
    conn.executemany("""
        UPDATE sold_homes SET
            lat              = ?,
            lon              = ?,
            tract_fips       = ?,
            geocode_status   = ?,
            geocode_source   = ?,
            geocode_accuracy = ?
        WHERE sale_id = ?
    """, [
        (r["lat"], r["lon"], r["tract_fips"],
         r["geocode_status"], r["geocode_source"], r["geocode_accuracy"],
         r["sale_id"])
        for r in updates
    ])


# ── Census Batch Geocoder ─────────────────────────────────────────────────────
# Accepts a CSV with columns: id, street, city, state, zip
# Returns a CSV with matched coordinates and census geography

_CENSUS_BATCH_URL = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
_CENSUS_SINGLE_URL = (
    "https://geocoding.geo.census.gov/geocoder/geographies/address"
    "?street={street}&city={city}&state={state}&zip={zip}"
    "&benchmark=Public_AR_Current&vintage=Current_Current&format=json"
)
_BATCH_SIZE = 900   # stay under the 1,000-row hard limit


def _parse_census_batch_response(text: str) -> list[dict]:
    """
    Parse Census batch geocoder CSV response.

    The Census API returns TWO different column layouts depending on version:

    12-field (coordinates quoted as one field):
      0:ID  1:INPUT_ADDR  2:MATCH  3:TYPE  4:MATCHED_ADDR
      5:"lon,lat"  6:TIGER_ID  7:SIDE  8:STATE  9:COUNTY  10:TRACT  11:BLOCK

    13-field (coordinates as two separate unquoted fields):
      0:ID  1:INPUT_ADDR  2:MATCH  3:TYPE  4:MATCHED_ADDR
      5:lon  6:lat  7:TIGER_ID  8:SIDE  9:STATE  10:COUNTY  11:TRACT  12:BLOCK

    No_Match rows have only 3 fields (or 12 with empty values).
    """
    results = []
    reader = csv.reader(io.StringIO(text))
    for parts in reader:
        try:
            if len(parts) < 3:
                continue

            row_id     = parts[0].strip()
            match      = parts[2].strip()
            match_type = parts[3].strip() if len(parts) > 3 else ""

            lat = lon = tract_fips = None

            if match == "Match":
                # Detect layout: 12-field has quoted "lon,lat" in parts[5],
                # 13-field has unquoted lon in parts[5], lat in parts[6]
                if len(parts) >= 13:
                    # 13-field layout: coordinates are separate
                    coord_offset = 0   # extra field shifts everything right by 1
                    try:
                        lon = float(parts[5].strip())
                        lat = float(parts[6].strip())
                    except (ValueError, IndexError):
                        pass
                elif len(parts) >= 12:
                    # 12-field layout: coordinates are "lon,lat" in one field
                    coord_offset = -1  # one fewer field before FIPS
                    coords_str = parts[5].strip()
                    if coords_str and "," in coords_str:
                        try:
                            lon_s, lat_s = coords_str.split(",", 1)
                            lon = float(lon_s.strip())
                            lat = float(lat_s.strip())
                        except ValueError:
                            pass
                else:
                    coord_offset = -1

                # FIPS field positions shift by 1 depending on layout
                # 12-field: STATE=parts[8], COUNTY=parts[9], TRACT=parts[10]
                # 13-field: STATE=parts[9], COUNTY=parts[10], TRACT=parts[11]
                fips_start = 9 if len(parts) >= 13 else 8
                if len(parts) > fips_start + 2:
                    sf = parts[fips_start].strip()
                    cf = parts[fips_start + 1].strip()
                    tr = parts[fips_start + 2].strip()
                    if sf and cf and tr:
                        try:
                            tract_fips = (
                                sf.zfill(2) + cf.zfill(3) + tr.zfill(6)
                            )
                        except Exception:
                            pass

            results.append({
                "row_id":           row_id,
                "match":            match,
                "match_type":       match_type,
                "lat":              lat,
                "lon":              lon,
                "tract_fips":       tract_fips,
                "geocode_source":   "census_batch",
                "geocode_accuracy": match_type if match == "Match" else "No_Match",
            })
        except Exception:
            # Skip malformed rows — don't let one bad row kill the whole batch
            continue
    
    return results


def _geocode_batch_census(batch_df: pd.DataFrame, retries: int = 2,
                         debug_first: bool = False) -> list[dict]:
    """
    Send one batch (≤900 rows) to Census Batch Geocoder.
    batch_df must have columns: row_id, street, city, state, zip
    """
    csv_buf = io.StringIO()
    batch_df[["row_id", "street", "city", "state", "zip"]].to_csv(
        csv_buf, index=False, header=False
    )
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    if debug_first:
        print("\n  [DEBUG] First 3 rows sent to Census API:")
        for line in csv_buf.getvalue().splitlines()[:3]:
            print(f"    {line!r}")

    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                _CENSUS_BATCH_URL,
                files={"addressFile": ("addresses.csv", csv_bytes, "text/csv")},
                data={
                    "benchmark":  "Public_AR_Current",
                    "vintage":    "Current_Current",
                    "returntype": "geographies",
                },
                timeout=180,
            )
            resp.raise_for_status()

            if debug_first:
                print(f"  [DEBUG] HTTP {resp.status_code}, "
                      f"response length: {len(resp.text)} chars")
                print(f"  [DEBUG] First 3 response lines:")
                for line in resp.text.splitlines()[:3]:
                    print(f"    {line!r}")

            parsed = _parse_census_batch_response(resp.text)

            if debug_first:
                matched = sum(1 for r in parsed if r.get("match") == "Match")
                print(f"  [DEBUG] Parsed {len(parsed)} rows, "
                      f"{matched} matches")

            return parsed
        except requests.exceptions.Timeout:
            if attempt < retries:
                wait = 10 * (attempt + 1)
                print(f" [timeout, retry in {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                print(" [timed out — chunk stays pending]")
                return []
        except Exception as e:
            print(f" [error: {e}]")
            return []


# ── Single-address fallback ───────────────────────────────────────────────────

def _geocode_single_census(street: str, city: str,
                           state: str, zip_code: str) -> Optional[dict]:
    """One address at a time — only use for small targeted retries."""
    try:
        url = _CENSUS_SINGLE_URL.format(
            street=requests.utils.quote(street),
            city=requests.utils.quote(city),
            state=requests.utils.quote(state),
            zip=requests.utils.quote(zip_code or ""),
        )
        resp = requests.get(url, timeout=15)
        matches = resp.json().get("result", {}).get("addressMatches", [])
        if not matches:
            return None
        m = matches[0]
        coords = m.get("coordinates", {})
        tracts = m.get("geographies", {}).get("Census Tracts", [])
        tract_fips = None
        if tracts:
            t = tracts[0]
            tract_fips = (
                t.get("STATE", "").zfill(2)
                + t.get("COUNTY", "").zfill(3)
                + t.get("TRACT", "").zfill(6)
            )
        return {
            "lat":             coords.get("y"),
            "lon":             coords.get("x"),
            "tract_fips":      tract_fips,
            "geocode_source":  "census_single",
            "geocode_accuracy": "Exact",
        }
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def geocode_dataframe(
    df: pd.DataFrame,
    street_col:            str  = "address",
    city_col:              str  = "city",
    state_col:             str  = "state",
    zip_col:               str  = "zip",
    full_addr_col:         str  = "full_address",
    sale_id_col:           str  = "sale_id",
    single_fallback_limit: int  = 0,
    verbose:               bool = True,
) -> pd.DataFrame:
    """
    Geocode rows in df that are missing lat/lon.

    Parameters
    ----------
    df            : DataFrame with address columns and existing lat/lon (nullable)
    street_col    : column containing street address (e.g. "123 Main St")
    city_col      : city column
    state_col     : state abbreviation column
    zip_col       : ZIP code column
    full_addr_col : pre-built full address string (used for cache key + Nominatim)
    verbose       : print progress
    single_fallback_limit : Max rows to send one-by-one after batch No_Match.
                            Default 0 for bulk loads. Pass e.g. 500 for
                            `--only geocode` targeted retries.

    Returns
    -------
    df with lat, lon, tract_fips, geocode_status, geocode_source, geocode_accuracy
    populated for newly geocoded rows.

    Unresolved rows get geocode_status='pending' so they can be retried later.
    Results are persisted to sold_homes after EVERY batch chunk.
    """
    _ensure_cache_table()

    # Ensure output columns exist
    for col in ["lat", "lon", "tract_fips", "geocode_status",
                "geocode_source", "geocode_accuracy"]:
        if col not in df.columns:
            df[col] = None

    # Identify rows that need geocoding (no lat/lon yet)
    needs_geo = df["lat"].isna() | df["lon"].isna()
    if not needs_geo.any():
        if verbose:
            print("    All rows already have coordinates")
        return df

    todo = df[needs_geo].copy()
    if verbose:
        print(f"    {len(todo):,} addresses need geocoding")

    # Build full_address string
    def _build_full(row):
        parts = [
            str(row.get(street_col, "") or "").strip(),
            str(row.get(city_col,   "") or "").strip(),
            str(row.get(state_col,  "") or "").strip(),
            str(row.get(zip_col,    "") or "").strip(),
        ]
        return ", ".join(p for p in parts if p)

    if full_addr_col not in todo.columns or todo[full_addr_col].isna().all():
        todo[full_addr_col] = todo.apply(_build_full, axis=1)
        df.loc[todo.index, full_addr_col] = todo[full_addr_col]

    # ── Step 1: Cache lookup ──────────────────────────────────────────────
    addr_keys = [_address_key(str(a)) for a in todo[full_addr_col]]
    cache     = _lookup_cache(addr_keys)
    hits      = 0
    for idx, row in todo.iterrows():
        key = _address_key(str(row.get(full_addr_col, "")))
        if key in cache:
            c = cache[key]
            df.at[idx, "lat"]              = c["lat"]
            df.at[idx, "lon"]              = c["lon"]
            df.at[idx, "tract_fips"]       = c["tract_fips"]
            df.at[idx, "geocode_status"]   = "success" if c.get("lat") else "failed"
            df.at[idx, "geocode_source"]   = c["geocode_source"]
            df.at[idx, "geocode_accuracy"] = c["geocode_accuracy"]
            hits += 1

    if verbose and hits:
        remaining = len(todo) - hits
        print(f"    Cache: {hits:,} hits  |  {remaining:,} to fetch")

    # ── Step 2: Census Batch ──────────────────────────────────────────────
    still = df[needs_geo & df["lat"].isna()].copy()
    if still.empty:
        _print_summary(df, needs_geo, verbose)
        return df
    
    n_chunks = (len(still) + _BATCH_SIZE - 1) // _BATCH_SIZE
    if verbose:
        print(f"    Census Batch: {len(still):,} addresses → "
              f"{n_chunks} chunk(s) of ≤{_BATCH_SIZE}")

    still["row_id"] = still.index.astype(str)
    still["street"] = still[street_col].fillna("").astype(str).str.strip()
    still["city"]   = still[city_col  ].fillna("").astype(str).str.strip()
    still["state"]  = still[state_col ].fillna("").astype(str).str.strip()
    still["zip"]    = still[zip_col   ].fillna("").astype(str).str.strip()

    # ── Pre-flight: test one address before committing to 31k batches ─────
    if verbose:
        sample = still.iloc[0]
        print(f"    Pre-flight sample: street={sample['street']!r}  "
              f"city={sample['city']!r}  state={sample['state']!r}  "
              f"zip={sample['zip']!r}")
        # warn if fields look empty or malformed
        empty_street = still["street"].str.strip().eq("").sum()
        empty_city   = still["city"].str.strip().eq("").sum()
        if empty_street > len(still) * 0.5:
            print(f"    ⚠ WARNING: {empty_street:,}/{len(still):,} rows have "
                  f"empty street — check column mapping (street_col={street_col!r})")
        if empty_city > len(still) * 0.5:
            print(f"    ⚠ WARNING: {empty_city:,}/{len(still):,} rows have "
                  f"empty city — geocoding will likely fail")

    total_matched = 0
    for chunk_num, chunk_start in enumerate(range(0, len(still), _BATCH_SIZE), 1):
        chunk = still.iloc[chunk_start : chunk_start + _BATCH_SIZE].copy()
        print(f"    [{chunk_num:>3}/{n_chunks}] {len(chunk)} rows … ",
              end="", flush=True)

        # Pass debug_first=True only for the very first chunk
        results   = _geocode_batch_census(chunk, debug_first=(chunk_num == 1 and verbose))
        by_row_id = {r["row_id"]: r for r in results}

        cache_batch = []
        db_batch    = []

        for idx, row in chunk.iterrows():
            res       = by_row_id.get(str(idx))
            full_addr = str(row.get(full_addr_col, ""))
            sale_id   = row.get(sale_id_col)

            if res and res.get("lat"):
                df.at[idx, "lat"]              = res["lat"]
                df.at[idx, "lon"]              = res["lon"]
                df.at[idx, "tract_fips"]       = res["tract_fips"]
                df.at[idx, "geocode_status"]   = "success"
                df.at[idx, "geocode_source"]   = res["geocode_source"]
                df.at[idx, "geocode_accuracy"] = res["geocode_accuracy"]
                total_matched += 1

                cache_batch.append({
                    "address_key":     _address_key(full_addr),
                    "full_address":    full_addr,
                    "lat":             res["lat"],
                    "lon":             res["lon"],
                    "tract_fips":      res["tract_fips"],
                    "geocode_source":  res["geocode_source"],
                    "geocode_accuracy": res["geocode_accuracy"],
                })
                if sale_id:
                    db_batch.append({
                        "sale_id":         sale_id,
                        "lat":             res["lat"],
                        "lon":             res["lon"],
                        "tract_fips":      res["tract_fips"],
                        "geocode_status":  "success",
                        "geocode_source":  res["geocode_source"],
                        "geocode_accuracy": res["geocode_accuracy"],
                    })
            else:
                df.at[idx, "geocode_status"] = "pending"

        # Flush this chunk — safe to Ctrl+C after this point
        _save_cache(cache_batch)
        _flush_to_sold_homes(db_batch)

        pct = 100 * len(db_batch) // max(len(chunk), 1)
        print(f"{len(db_batch)}/{len(chunk)} matched ({pct}%)  "
              f"[total: {total_matched:,}]")

        # Early exit if 5 consecutive batches all return 0 — API is likely
        # rejecting the format, no point sending 35 more batches
        if chunk_num == 1 and len(db_batch) == 0:
            consecutive_zeros = getattr(_geocode_batch_census, '_consecutive_zeros', 0) + 1
        elif len(db_batch) == 0:
            consecutive_zeros = getattr(_geocode_batch_census, '_consecutive_zeros', 0) + 1
        else:
            consecutive_zeros = 0
        _geocode_batch_census._consecutive_zeros = consecutive_zeros

        if consecutive_zeros >= 5 and total_matched == 0:
            print(f"\n    ⚠ 5 consecutive batches returned 0 matches "
                  f"({consecutive_zeros * _BATCH_SIZE} rows tried, 0 matched).")
            print("    This usually means:")
            print("      1. The Census API is temporarily down — try again later")
            print("      2. Address fields are blank/malformed — check pre-flight output above")
            print("      3. ZIP codes are missing or wrong format (need 5 digits)")
            print("    Stopping early. Remaining rows marked 'pending'.")
            print("    → Retry later: python setup_data.py --only geocode")
            # Mark remaining rows as pending
            remaining_idx = still.iloc[chunk_start + _BATCH_SIZE:].index
            df.loc[remaining_idx, "geocode_status"] = "pending"
            break

        if chunk_start + _BATCH_SIZE < len(still):
            time.sleep(1)

    # ── Step 3: Optional single-address retry — hard-capped ───────────────
    pending_mask = needs_geo & (df["geocode_status"] == "pending")
    n_pending    = int(pending_mask.sum())

    if single_fallback_limit > 0 and n_pending > 0:
        cap      = min(single_fallback_limit, n_pending)
        to_retry = df[pending_mask].head(cap)
        if verbose:
            print(f"    Single-address retry: {cap} of {n_pending:,} pending "
                  f"(cap={single_fallback_limit}) …")

        cache_batch = []
        db_batch    = []
        resolved    = 0
        for i, (idx, row) in enumerate(to_retry.iterrows(), 1):
            res = _geocode_single_census(
                str(row.get(street_col, "") or ""),
                str(row.get(city_col,   "") or ""),
                str(row.get(state_col,  "") or ""),
                str(row.get(zip_col,    "") or ""),
            )
            full_addr = str(row.get(full_addr_col, ""))
            sale_id   = row.get(sale_id_col)

            if res and res.get("lat"):
                df.at[idx, "lat"]              = res["lat"]
                df.at[idx, "lon"]              = res["lon"]
                df.at[idx, "tract_fips"]       = res.get("tract_fips")
                df.at[idx, "geocode_status"]   = "success"
                df.at[idx, "geocode_source"]   = res["geocode_source"]
                df.at[idx, "geocode_accuracy"] = res["geocode_accuracy"]
                resolved += 1
                cache_batch.append({
                    "address_key":     _address_key(full_addr),
                    "full_address":    full_addr,
                    "lat":             res["lat"],
                    "lon":             res["lon"],
                    "tract_fips":      res.get("tract_fips"),
                    "geocode_source":  res["geocode_source"],
                    "geocode_accuracy": res["geocode_accuracy"],
                })
                if sale_id:
                    db_batch.append({
                        "sale_id":         sale_id,
                        "lat":             res["lat"],
                        "lon":             res["lon"],
                        "tract_fips":      res.get("tract_fips"),
                        "geocode_status":  "success",
                        "geocode_source":  res["geocode_source"],
                        "geocode_accuracy": res["geocode_accuracy"],
                    })
            else:
                df.at[idx, "geocode_status"] = "failed"

            # Flush every 50 rows in case of interruption
            if i % 50 == 0:
                _save_cache(cache_batch);  cache_batch = []
                _flush_to_sold_homes(db_batch); db_batch = []
                if verbose:
                    print(f"    … {i}/{cap} ({resolved} resolved)", flush=True)
            time.sleep(0.4)

        _save_cache(cache_batch)
        _flush_to_sold_homes(db_batch)

        if verbose:
            print(f"    Single retry complete: {resolved}/{cap} resolved")

    elif n_pending > 0 and verbose:
        print(f"    {n_pending:,} addresses remain 'pending'")
        print("    → Run:  python setup_data.py --only geocode  to retry later")

    _print_summary(df, needs_geo, verbose)
    return df


def _print_summary(df, needs_geo, verbose):
    if not verbose:
        return
    sub = df.loc[needs_geo, "geocode_status"]
    success = (sub == "success").sum()
    pending = (sub == "pending").sum()
    failed  = (sub == "failed").sum()
    na      = sub.isna().sum()
    print(f"    Geocoding — success: {success:,}  "
          f"pending: {pending + na:,}  failed: {failed:,}")


# ── Retry pending from DB ─────────────────────────────────────────────────────

def geocode_pending(single_fallback_limit: int = 500,
                    verbose: bool = True) -> int:
    """
    Re-attempt geocoding for sold_homes rows still marked pending.
    Runs another full batch pass first, then up to `single_fallback_limit`
    rows one-by-one. Safe to run multiple times.
    """
    pending_df = store.query("""
        SELECT * FROM sold_homes
        WHERE geocode_status IS NULL OR geocode_status = 'pending'
        LIMIT 50000
    """)
    if pending_df.empty:
        if verbose:
            print("  No pending geocodes found.")
        return 0

    if verbose:
        print(f"  Retrying {len(pending_df):,} pending rows …")
    
    before = (pending_df["geocode_status"] == "success").sum()
    updated = geocode_dataframe(
        pending_df,
        single_fallback_limit=single_fallback_limit,
        verbose=verbose,
    )
    after = (updated["geocode_status"] == "success").sum()
    return int(after - before)
