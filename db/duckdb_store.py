"""
DuckDB-backed structured data store.
All heavy analytics (SQL-like queries, joins) go through here.
"""
import json
import duckdb
import pandas as pd
from pathlib import Path
from typing import Any, Optional

from config import settings


_conn: Optional[duckdb.DuckDBPyConnection] = None


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        _conn = duckdb.connect(str(settings.duckdb_path))
        _ensure_schema(_conn)
    return _conn


def _ensure_schema(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS houses (
            house_id        VARCHAR PRIMARY KEY,
            address         VARCHAR,
            city            VARCHAR,
            state           VARCHAR,
            zip             VARCHAR,
            lat             DOUBLE,
            lon             DOUBLE,
            status          VARCHAR,
            price           DOUBLE,
            beds            DOUBLE,
            baths           DOUBLE,
            sqft            DOUBLE,
            year_built      INTEGER,
            hoa_fee         DOUBLE,
            price_per_sq_ft DOUBLE,
            walk_score      INTEGER,
            bike_score      INTEGER,
            transit_score   INTEGER,
            tract_fips      VARCHAR,   -- 11-digit census tract FIPS
            msa_code        VARCHAR,   -- CBSA code
            nearest_big_city VARCHAR,
            crime_city      VARCHAR,   -- normalized key into crime_incidents.city (see data_loader.load_redfin)
            source_file     VARCHAR,
            raw_json        VARCHAR,   -- everything else as JSON
            is_favorite     BOOLEAN DEFAULT FALSE,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Add is_favorite column if upgrading an existing database
    try:
        conn.execute("ALTER TABLE houses ADD COLUMN IF NOT EXISTS is_favorite BOOLEAN DEFAULT FALSE")
    except Exception:
        pass

    # house_snapshots — one row per (house × source_file × status × price).
    # Captures every distinct state observed across all Redfin CSV exports and
    # matched sold records. The `houses` table always holds the LATEST snapshot.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS house_snapshots (
            snapshot_id     VARCHAR PRIMARY KEY,  -- hash(house_id + source + status + price)
            house_id        VARCHAR NOT NULL,
            snapshot_date   DATE,                 -- extracted from filename or load date
            source_file     VARCHAR,
            source_type     VARCHAR,              -- 'redfin' | 'sold'
            status          VARCHAR,
            price           DOUBLE,               -- list price (Redfin) or sold price (sold)
            beds            DOUBLE,
            baths           DOUBLE,
            sqft            DOUBLE,
            hoa_fee         DOUBLE,
            year_built      INTEGER,
            walk_score      INTEGER,
            bike_score      INTEGER,
            transit_score   INTEGER,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS nri_tracts (
            tract_fips      VARCHAR PRIMARY KEY,
            county_fips     VARCHAR,
            state_fips      VARCHAR,
            state_name      VARCHAR,
            county_name     VARCHAR,
            nri_id          VARCHAR,
            -- Composite
            risk_score      DOUBLE,
            risk_ratng      VARCHAR,
            risk_npctl      DOUBLE,
            eal_score       DOUBLE,
            eal_ratng       VARCHAR,
            eal_valt        DOUBLE,    -- $ expected annual loss
            sovi_score      DOUBLE,
            sovi_ratng      VARCHAR,
            resl_score      DOUBLE,
            resl_ratng      VARCHAR,
            -- Per-hazard risk scores (18 hazards)
            avln_risks      DOUBLE,    -- Avalanche
            cfld_risks      DOUBLE,    -- Coastal Flooding
            cwav_risks      DOUBLE,    -- Cold Wave
            drgt_risks      DOUBLE,    -- Drought
            erqk_risks      DOUBLE,    -- Earthquake
            hail_risks      DOUBLE,    -- Hail
            hwav_risks      DOUBLE,    -- Heat Wave
            hrcn_risks      DOUBLE,    -- Hurricane
            istm_risks      DOUBLE,    -- Ice Storm
            lnds_risks      DOUBLE,    -- Landslide
            ltng_risks      DOUBLE,    -- Lightning
            rfld_risks      DOUBLE,    -- Riverine Flooding
            swnd_risks      DOUBLE,    -- Strong Wind
            trnd_risks      DOUBLE,    -- Tornado
            tsun_risks      DOUBLE,    -- Tsunami
            vlcn_risks      DOUBLE,    -- Volcanic Activity
            wfir_risks      DOUBLE,    -- Wildfire
            wntw_risks      DOUBLE     -- Winter Weather
        )
    """)

    # crime_incidents — standardized, severity-weighted crime records across
    # every city dropped into data/crime/<city>/. See services/crime_sources.py
    # (per-city parsers) and services/crime_taxonomy.py (category + weight
    # assignment). Populated by services/data_loader.py::load_crime().
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crime_incidents (
            incident_id     VARCHAR PRIMARY KEY,
            city            VARCHAR,    -- normalized key, e.g. 'pittsburgh' — matches houses.crime_city
            source_file     VARCHAR,
            occurred_at     TIMESTAMP,
            year            INTEGER,
            month           INTEGER,
            year_month      VARCHAR,    -- 'YYYY-MM', convenience for time-based grouping
            lat             DOUBLE,
            lon             DOUBLE,
            category        VARCHAR,    -- standardized key, e.g. 'aggravated_assault'
            category_label  VARCHAR,    -- human-readable, e.g. 'Aggravated Assault'
            severity_weight DOUBLE,     -- 1.0-10.0, see services/crime_taxonomy.py
            raw_type        VARCHAR,    -- original offense/type text, kept for reference
            location_text   VARCHAR     -- block address / cross-street, when the source has one
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS census_tracts (
            tract_fips  VARCHAR PRIMARY KEY,
            geo_id      VARCHAR,
            name        VARCHAR,
            population  INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS census_msa (
            msa_code    VARCHAR PRIMARY KEY,
            geo_id      VARCHAR,
            name        VARCHAR,
            population  INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS cbsa_counties (
            cbsa_code       VARCHAR,
            cbsa_title      VARCHAR,
            msa_type        VARCHAR,
            state_fips      VARCHAR,
            county_fips     VARCHAR,
            county_name     VARCHAR,
            state_name      VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sold_homes (
            -- Identity
            sale_id             VARCHAR PRIMARY KEY,  -- hash of parid+saledate or addr+saledate
            source_county       VARCHAR,              -- e.g. "Allegheny, PA"
            source_file         VARCHAR,

            -- Parcel
            parid               VARCHAR,              -- county assessor parcel ID

            -- Address (normalised)
            address             VARCHAR,              -- street only: "1006 Jacob St"
            city                VARCHAR,
            state               VARCHAR,
            zip                 VARCHAR,
            full_address        VARCHAR,              -- full string used for geocoding

            -- Geocoding (populated by services/geocoder.py)
            lat                 DOUBLE,               -- NULL until geocoded
            lon                 DOUBLE,
            geocode_status      VARCHAR,              -- pending | success | failed | manual
            geocode_source      VARCHAR,              -- census_batch | census_single | nominatim | manual
            geocode_accuracy    VARCHAR,              -- Exact | Non_Exact | No_Match | <nominatim type>

            -- Sale
            sold_price          DOUBLE,
            sold_date           DATE,
            record_date         DATE,                 -- date deed was recorded (may differ from sale)
            list_price          DOUBLE,               -- NULL for county data (no listing history)

            -- Property details (NULL for county-only data; populated if cross-referenced)
            sqft                DOUBLE,
            beds                DOUBLE,
            baths               DOUBLE,
            year_built          INTEGER,

            -- County metadata
            sale_code           VARCHAR,              -- county-specific code (e.g. "3", "H", "99")
            sale_desc           VARCHAR,              -- human-readable sale type
            instr_type          VARCHAR,              -- instrument type code (DE, CO, …)
            instr_type_desc     VARCHAR,              -- DEED, CORRECTIVE DEED, …
            deed_book           VARCHAR,
            deed_page           VARCHAR,
            muni_code           VARCHAR,
            muni_desc           VARCHAR,
            school_code         VARCHAR,
            school_desc         VARCHAR,

            -- Arm's-length classification
            is_arms_length      BOOLEAN,              -- FALSE → exclude from price comps
            arms_length_flag    VARCHAR,              -- reason if not arm's length

            -- Tract (from geocoding or spatial join)
            tract_fips          VARCHAR,              -- 11-digit census tract FIPS
            county_fips         VARCHAR               -- 5-digit (state+county)
        )
    """)

    # Geocode result cache — avoids re-calling the API on every setup_data run
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geocode_cache (
            address_key         VARCHAR PRIMARY KEY,  -- MD5 of normalised full_address
            full_address        VARCHAR,
            lat                 DOUBLE,
            lon                 DOUBLE,
            tract_fips          VARCHAR,
            geocode_source      VARCHAR,
            geocode_accuracy    VARCHAR,
            geocoded_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def query(sql: str, params=None) -> pd.DataFrame:
    """Run any SQL query and return a DataFrame."""
    conn = get_conn()
    if params:
        return conn.execute(sql, params).df()
    return conn.execute(sql).df()


def query_json(sql: str, params=None) -> list[dict]:
    """Run SQL query and return list of dicts safe for JSON serialization."""
    df = query(sql, params)
    # Replace NaN / Inf / -Inf with None — json.dumps rejects all three
    df = df.where(pd.notnull(df), other=None)
    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].apply(
            lambda v: None if v is not None and (v != v or v == float("inf") or v == float("-inf")) else v
        )
    # Dates/timestamps (e.g. created_at, snapshot_date) come back from DuckDB
    # as pandas Timestamp / NaT, neither of which json.dumps can handle
    # either — convert to ISO-8601 strings (None for NaT) so every caller of
    # this function actually gets what the docstring above promises.
    for col in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
        df[col] = df[col].apply(lambda v: None if pd.isna(v) else v.isoformat())
    return [_json_safe_record(r) for r in df.to_dict(orient="records")]


def _json_safe_record(record: dict) -> dict:
    """Defense in depth for query_json(): normalize any value type that
    to_dict() can hand back but json.dumps() rejects — a Timestamp/NaT that
    slipped past the column-dtype pass above (e.g. an object-dtype column
    holding a mixed value), or a numpy scalar (np.int64, np.bool_, ...)
    instead of a native Python type."""
    import numpy as np
    safe = {}
    for k, v in record.items():
        if isinstance(v, pd.Timestamp) or v is pd.NaT:
            safe[k] = None if pd.isna(v) else v.isoformat()
        elif isinstance(v, np.generic):
            safe[k] = v.item()
        else:
            safe[k] = v
    return safe


_SNAP_COLS = [
    "snapshot_id", "house_id", "snapshot_date", "source_file", "source_type",
    "status", "price", "beds", "baths", "sqft", "hoa_fee",
    "year_built", "walk_score", "bike_score", "transit_score",
]
_SNAP_INSERT = (
    f"INSERT OR IGNORE INTO house_snapshots ({', '.join(_SNAP_COLS)}) "
    f"SELECT {', '.join(_SNAP_COLS)} FROM __snap_tmp"
)

_HOUSES_COLS = [
    "house_id", "address", "city", "state", "zip", "lat", "lon",
    "status", "price", "beds", "baths", "sqft", "year_built",
    "hoa_fee", "walk_score", "bike_score", "transit_score",
    "tract_fips", "msa_code", "nearest_big_city", "crime_city",
    "source_file", "raw_json",
]
_HOUSES_INSERT = f"INSERT OR REPLACE INTO houses ({', '.join(_HOUSES_COLS)}) SELECT {', '.join(_HOUSES_COLS)} FROM __tmp_houses"


def _build_snap_df(df: pd.DataFrame, source_type: str = "redfin") -> pd.DataFrame:
    """Build a snapshot DataFrame from a houses DataFrame. Always returns a DataFrame."""
    import hashlib as _hl

    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError(f"_build_snap_df: expected DataFrame, got {type(df)}")
    
    snap_df = df.copy()
    if "source_type" not in snap_df.columns:
        snap_df["source_type"] = source_type
    if "snapshot_date" not in snap_df.columns:
        snap_df["snapshot_date"] = None

    # Ensure every required column exists
    for col in _SNAP_COLS:
        if col != "snapshot_id" and col not in snap_df.columns:
            snap_df[col] = None
    
    def _snap_id(row):
        key = (f"{row.get('house_id', '')}{row.get('source_file', '')}"
               f"{row.get('status', '')}{row.get('price', '')}")
        return _hl.md5(key.encode()).hexdigest()[:14]

    snap_df["snapshot_id"] = snap_df.apply(_snap_id, axis=1)

    result = snap_df[_SNAP_COLS].drop_duplicates(subset=["snapshot_id"])
    assert isinstance(result, pd.DataFrame), "_build_snap_df returned non-DataFrame"
    return result


def upsert_houses(df: pd.DataFrame):
    """
    Insert/replace house state AND record each row as a snapshot.
    Called by the old load_redfin path (backward compatible).
    """
    conn = get_conn()

    # ── Snapshots ─────────────────────────────────────────────────────────
    snap_df = _build_snap_df(df, source_type="redfin")
    conn.register("__snap_tmp", snap_df)
    conn.execute(_SNAP_INSERT)
    conn.unregister("__snap_tmp")

    # ── Latest state → houses ─────────────────────────────────────────────
    for col in _HOUSES_COLS:
        if col not in df.columns:
            df[col] = None
    houses_df = df[_HOUSES_COLS].copy()
    conn.register("__tmp_houses", houses_df)
    conn.execute(_HOUSES_INSERT)
    conn.unregister("__tmp_houses")


def add_sold_snapshot(house_id: str, sold_row: dict):
    """
    Record a matched sold transaction as a snapshot.
    Uses SALEDATE as both snapshot_date and created_at.
    Converts the date in Python (no TRY_CAST) to avoid DuckDB parameter issues.
    """
    import hashlib as _hl
    conn = get_conn()
    snap_id = _hl.md5(
        f"{house_id}{sold_row.get('sold_date','')}{sold_row.get('sold_price','')}".encode()
    ).hexdigest()[:14]

    sold_date = sold_row.get("sold_date")

    # Convert sold_date to a plain ISO string for created_at — done in Python,
    # not SQL, to avoid DuckDB counting ? inside TRY_CAST as an extra parameter.
    created_at = None
    if sold_date is not None:
        try:
            created_at = str(pd.to_datetime(sold_date).date())
        except Exception:
            pass
    
    conn.execute("""
        INSERT OR IGNORE INTO house_snapshots
            (snapshot_id, house_id, snapshot_date, source_file, source_type,
             status, price, beds, baths, sqft, year_built, created_at)
        VALUES (?, ?, ?, ?, 'sold', 'Sold', ?, ?, ?, ?, ?, ?)
    """, [
        snap_id,
        house_id,
        str(sold_date) if sold_date is not None else None,
        sold_row.get("source_file"),
        sold_row.get("sold_price"),
        sold_row.get("beds"),
        sold_row.get("baths"),
        sold_row.get("sqft"),
        sold_row.get("year_built"),
        created_at,
    ])


def get_house_history(house_id: str) -> list[dict]:
    """
    Return all snapshots for a house in reverse chronological order.
    snapshot_date is the authoritative date:
      - Redfin rows:  the 'date' field from the CSV (per-row listing date)
      - Sold rows:    the SALEDATE from the assessor record
    """
    return query_json("""
        SELECT
            snapshot_id,
            snapshot_date,
            source_type,
            source_file,
            status,
            price,
            beds,
            baths,
            sqft,
            hoa_fee,
            year_built,
            walk_score,
            bike_score,
            transit_score,
            created_at
        FROM house_snapshots
        WHERE house_id = ?
        ORDER BY snapshot_date DESC NULLS LAST, created_at DESC
    """, [house_id])


def match_sold_to_houses() -> int:
    """
    Match sold_homes records to Redfin houses and add them as 'sold' snapshots.

    Four matching tiers (applied in order, first match wins):
      1. Exact:   normalize(address) + zip5
      2. Number+street: house_number + first_street_word + zip5
         (handles suffix differences: "Ave" vs "Avenue" vs missing)
      3. Number-only + zip5 + city initial
         (handles directional differences: "N Highland" vs "Highland")
      4. House number + zip5 (last resort — only used for rare/short street names)

    Normalization aggressively strips suffixes, directionals, unit numbers and
    punctuation so Redfin "4002 Lorigan St" matches assessor "4002 LORIGAN ST".
    """
    import re as _re

    _SUFFIX_MAP = {
        "street": "st", "avenue": "av", "ave": "av", "boulevard": "blvd",
        "drive": "dr", "road": "rd", "court": "ct", "lane": "ln",
        "place": "pl", "way": "wy", "terrace": "ter", "circle": "cir",
        "trail": "trl", "highway": "hwy", "parkway": "pkwy", "pike": "pk",
    }
    _DIR = {"north": "n", "south": "s", "east": "e", "west": "w",
            "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw"}

    def _norm(addr: str) -> str:
        """Lowercase, remove punctuation, collapse whitespace."""
        if not addr:
            return ""
        a = str(addr).lower().strip()
        a = _re.sub(r'[.,#]', ' ', a)
        a = _re.sub(r'\s+', ' ', a).strip()
        # expand abbreviations then re-collapse
        tokens = a.split()
        tokens = [_DIR.get(t, _SUFFIX_MAP.get(t, t)) for t in tokens]
        return " ".join(tokens)

    def _house_num(norm_addr: str) -> str:
        """Return the leading house number token, or ''."""
        m = _re.match(r'^(\d+\w*)', norm_addr.strip())
        return m.group(1) if m else ""

    def _street_word(norm_addr: str) -> str:
        """Return first non-numeric, non-directional token (the street name)."""
        tokens = norm_addr.split()
        for i, t in enumerate(tokens):
            if i == 0 and _re.match(r'^\d', t):
                continue  # skip house number
            if t in _DIR.values():
                continue  # skip directionals
            return t
        return tokens[1] if len(tokens) > 1 else ""

    # ── Load houses ───────────────────────────────────────────────────────
    houses_df = query("""
        SELECT house_id, address, city, zip FROM houses WHERE address IS NOT NULL
    """)
    if houses_df.empty:
        return 0

    houses_df["norm"]    = houses_df["address"].apply(_norm)
    houses_df["zip5"]    = houses_df["zip"].astype(str).str[:5]
    houses_df["hnum"]    = houses_df["norm"].apply(_house_num)
    houses_df["sword"]   = houses_df["norm"].apply(_street_word)
    houses_df["city_init"] = houses_df["city"].astype(str).str[:3].str.lower()

    # Build lookup dicts for each tier
    t1 = {}   # (norm_addr, zip5) → house_id
    t2 = {}   # (hnum, sword, zip5) → house_id
    t3 = {}   # (hnum, zip5, city_init) → house_id

    for row in houses_df.itertuples(index=False):
        k1 = (row.norm, row.zip5)
        k2 = (row.hnum, row.sword, row.zip5)
        k3 = (row.hnum, row.zip5, row.city_init)
        if k1 not in t1: t1[k1] = row.house_id
        if k2 not in t2: t2[k2] = row.house_id
        if k3 not in t3: t3[k3] = row.house_id

    # ── Load sold homes ───────────────────────────────────────────────────
    sold_df = query("""
        SELECT s.*
        FROM sold_homes s
        WHERE s.address IS NOT NULL
          AND COALESCE(s.sold_price, 0) > 0
          AND NOT EXISTS (
              SELECT 1 FROM house_snapshots hs
              WHERE hs.source_type = 'sold'
                AND hs.snapshot_date::VARCHAR = s.sold_date::VARCHAR
                AND hs.price = s.sold_price
                AND hs.house_id IN (SELECT house_id FROM houses)
          )
        LIMIT 100000
    """)
    if sold_df.empty:
        print("  No unmatched sold records found")
        return 0

    sold_df["norm"]      = sold_df["address"].apply(_norm)
    sold_df["zip5"]      = sold_df["zip"].astype(str).str[:5]
    sold_df["hnum"]      = sold_df["norm"].apply(_house_num)
    sold_df["sword"]     = sold_df["norm"].apply(_street_word)
    sold_df["city_init"] = sold_df["city"].astype(str).str[:3].str.lower()

    matched = 0
    tiers   = {1: 0, 2: 0, 3: 0}
    for _, sold_row in sold_df.iterrows():
        nr   = sold_row["norm"]
        z5   = sold_row["zip5"]
        hn   = sold_row["hnum"]
        sw   = sold_row["sword"]
        ci   = sold_row["city_init"]

        house_id = (
            t1.get((nr, z5))
            or t2.get((hn, sw, z5))
            or t3.get((hn, z5, ci))
        )
        tier = (1 if t1.get((nr, z5)) else
                2 if t2.get((hn, sw, z5)) else
                3 if t3.get((hn, z5, ci)) else None)

        if house_id:
            add_sold_snapshot(house_id, sold_row.to_dict())
            matched += 1
            if tier:
                tiers[tier] += 1

    total = len(sold_df)
    print(f"  Matched {matched}/{total} sold records "
          f"(tier1={tiers[1]}, tier2={tiers[2]}, tier3={tiers[3]})")
    return matched


def upsert_houses_with_snapshots(latest_df: pd.DataFrame, all_rows_df: pd.DataFrame):
    """
    Write house data in two passes:
      1. latest_df → houses table (most-recent state per house_id)
      2. all_rows_df → house_snapshots (every observed state, INSERT OR IGNORE)
    """
    conn = get_conn()

    # ── Latest state → houses ─────────────────────────────────────────────
    for col in _HOUSES_COLS:
        if col not in latest_df.columns:
            latest_df[col] = None
    houses_df = latest_df[_HOUSES_COLS].copy()
    conn.register("__tmp_houses", houses_df)
    conn.execute(_HOUSES_INSERT)
    conn.unregister("__tmp_houses")
    # ── All rows → snapshots ──────────────────────────────────────────────
    snap_df = _build_snap_df(all_rows_df, source_type="redfin")
    if snap_df is None or snap_df.empty:
        print("  ⚠ No snapshot rows to write")
        return
    conn.register("__snap_tmp", snap_df)
    conn.execute(_SNAP_INSERT)
    conn.unregister("__snap_tmp")


def _table_has_pk(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    """Return True if the table has at least one PRIMARY KEY column."""
    try:
        result = conn.execute(f"PRAGMA table_info('{table}')").fetchdf()
        # 'pk' column is non-zero for primary key columns in DuckDB/SQLite pragma
        return bool((result["pk"] > 0).any())
    except Exception:
        # If pragma fails, fall back to safe truncate path
        return False


def upsert_df(table: str, df: pd.DataFrame):
    """
    Insert df into table.
    - Tables WITH a primary key: INSERT OR REPLACE (upsert semantics)
    - Tables WITHOUT a primary key (e.g. cbsa_counties): DELETE all + re-insert
    """
    conn = get_conn()
    
    # Check whether this table has a primary key
    has_pk = _table_has_pk(conn, table)
    
    conn.register("__tmp", df)
    if has_pk:
        conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM __tmp")
    else:
        # Safe truncate-and-reload: wrap in a transaction so a failure
        # doesn't leave the table empty
        conn.execute("BEGIN")
        try:
            conn.execute(f"DELETE FROM {table}")
            conn.execute(f"INSERT INTO {table} SELECT * FROM __tmp")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    
    conn.unregister("__tmp")


# def get_house(house_id: str) -> Optional[dict]:
#     rows = query_json("SELECT * FROM houses WHERE house_id = ?", [house_id])
#     return rows[0] if rows else None


def toggle_favorite(house_id: str) -> bool:
    """Toggle is_favorite for a house. Returns the new value."""
    conn = get_conn()
    conn.execute("""
        UPDATE houses SET is_favorite = NOT COALESCE(is_favorite, FALSE)
        WHERE house_id = ?
    """, [house_id])
    row = conn.execute(
        "SELECT is_favorite FROM houses WHERE house_id = ?", [house_id]
    ).fetchone()
    return bool(row[0]) if row else False


def get_all_houses() -> list[dict]:
    """
    Return all houses with status/price from the most recent snapshot
    (Redfin OR sold — whichever is newer). Redfin data alone never shows
    a house as Sold even if a matched sold record exists at a later date.
    """
    return query_json("""
        WITH latest_snap AS (
            SELECT DISTINCT ON (house_id)
                house_id,
                status,
                price,
                snapshot_date,
                source_type
            FROM house_snapshots
            WHERE snapshot_date IS NOT NULL
            ORDER BY house_id, snapshot_date DESC, created_at DESC
        )
        SELECT
            h.house_id,
            h.address,
            h.city,
            h.state,
            h.zip,
            h.lat,
            h.lon,
            COALESCE(ls.status, h.status)   AS status,
            COALESCE(ls.price,  h.price)    AS price,
            h.beds,
            h.baths,
            h.sqft,
            h.year_built,
            h.hoa_fee,
            h.walk_score,
            h.bike_score,
            h.transit_score,
            h.tract_fips,
            h.msa_code,
            h.source_file,
            h.is_favorite,
            h.created_at,
            ls.snapshot_date   AS latest_snapshot_date,
            ls.source_type     AS latest_source_type,
            n.risk_score,
            n.risk_ratng,
            n.eal_valt,
            n.rfld_risks,
            n.hrcn_risks,
            n.trnd_risks,
            n.wfir_risks,
            n.erqk_risks,
            n.swnd_risks,
            ct.population      AS tract_population
        FROM houses h
        LEFT JOIN latest_snap ls  ON h.house_id = ls.house_id
        LEFT JOIN nri_tracts n    ON h.tract_fips = n.tract_fips
        LEFT JOIN census_tracts ct ON h.tract_fips = ct.tract_fips
    """)


def get_house(house_id: str) -> Optional[dict]:
    """
    Return a single house with status/price from the most recent snapshot
    (Redfin OR sold).
    """
    rows = query_json("""
        WITH latest_snap AS (
            SELECT DISTINCT ON (house_id)
                house_id,
                status,
                price,
                snapshot_date,
                source_type
            FROM house_snapshots
            WHERE house_id = ?
              AND snapshot_date IS NOT NULL
            ORDER BY house_id, snapshot_date DESC, created_at DESC
        )
        SELECT
            h.* EXCLUDE (status, price),
            COALESCE(ls.status, h.status) AS status,
            COALESCE(ls.price,  h.price)  AS price,
            ls.snapshot_date AS latest_snapshot_date,
            ls.source_type   AS latest_source_type
        FROM houses h
        LEFT JOIN latest_snap ls ON h.house_id = ls.house_id
        WHERE h.house_id = ?
    """, [house_id, house_id])
    return rows[0] if rows else None


def get_nri_for_tract(tract_fips: str) -> Optional[dict]:
    rows = query_json("SELECT * FROM nri_tracts WHERE tract_fips = ?", [tract_fips])
    return rows[0] if rows else None


def get_sold_in_tract(tract_fips: str, arms_length_only: bool = True) -> list[dict]:
    """Return sold homes in a census tract, optionally filtering non-arm's-length."""
    filter_clause = "AND (is_arms_length IS NULL OR is_arms_length = TRUE)" \
                    if arms_length_only else ""
    return query_json(f"""
        SELECT sale_id, address, city, state, zip, sold_price, sold_date,
               sqft, beds, baths, year_built, sale_desc, is_arms_length,
               arms_length_flag, geocode_source, source_county
        FROM sold_homes
        WHERE tract_fips = ? {filter_clause}
        ORDER BY sold_date DESC
    """, [tract_fips])


def get_price_stats_in_tract(tract_fips: str) -> dict:
    """Price stats for comparable houses in the same census tract."""
    df = query("""
        SELECT
            COUNT(*) as count,
            MEDIAN(price) as median_price,
            AVG(price) as avg_price,
            MEDIAN(price / NULLIF(sqft, 0)) as median_price_per_sqft,
            AVG(price / NULLIF(sqft, 0)) as avg_price_per_sqft,
            MIN(price) as min_price,
            MAX(price) as max_price
        FROM houses
        WHERE tract_fips = ? AND price > 0
    """, [tract_fips])
    row = df.iloc[0].to_dict() if len(df) > 0 else {}

    # Only use arm's-length sales for sold comps
    sold_df = query("""
        SELECT
            COUNT(*) as sold_count,
            MEDIAN(sold_price) as median_sold,
            MEDIAN(sold_price / NULLIF(sqft, 0)) as median_sold_per_sqft,
            MIN(sold_date) as oldest_sale,
            MAX(sold_date) as newest_sale,
            COUNT(*) FILTER (WHERE is_arms_length = FALSE) as non_arms_length_excluded
        FROM sold_homes
        WHERE tract_fips = ?
          AND (is_arms_length IS NULL OR is_arms_length = TRUE)
          AND sold_price > 1000
    """, [tract_fips])
    if len(sold_df) > 0:
        row.update({f"sold_{k}": v for k, v in sold_df.iloc[0].to_dict().items()})
    return row


def get_geocode_stats() -> dict:
    """Summary of geocoding status across all sold homes."""
    rows = query_json("""
        SELECT
            geocode_status,
            geocode_source,
            COUNT(*) as count
        FROM sold_homes
        GROUP BY geocode_status, geocode_source
        ORDER BY count DESC
    """)
    total = query("SELECT COUNT(*) as n FROM sold_homes").iloc[0]["n"]
    return {"total": int(total), "breakdown": rows}


def get_top_msas_by_population(n: int = 50) -> list[dict]:
    return query_json(f"""
        SELECT m.*, 
               AVG(n.risk_score) as avg_risk_score,
               AVG(n.risk_npctl) as avg_risk_percentile,
               MEDIAN(n.eal_valt) as median_eal
        FROM census_msa m
        JOIN cbsa_counties cb ON m.msa_code = cb.cbsa_code
        JOIN nri_tracts n ON (cb.state_fips || cb.county_fips) = n.county_fips
        GROUP BY m.msa_code, m.name, m.population, m.geo_id
        ORDER BY m.population DESC
        LIMIT {n}
    """)


def close():
    global _conn
    if _conn:
        _conn.close()
        _conn = None
