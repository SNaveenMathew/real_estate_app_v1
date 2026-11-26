"""Physical schema, one function per domain.

``db.duckdb_store._ensure_schema`` used to be a 470-line function: every new table meant editing the
middle of it, and every test bootstrap ran the whole thing with no way to see which domain had grown.
The DDL now lives here, grouped the way the catalog groups it, and ``initialize(conn)`` runs the groups
in the same order as before. The statements themselves are unchanged - verified by comparing the built
schema (information_schema + PRAGMA table_info) against a snapshot taken before the split."""

from __future__ import annotations

import duckdb

APP_SETTINGS = """
        CREATE TABLE IF NOT EXISTS app_settings (
            key         VARCHAR PRIMARY KEY,
            value_json  VARCHAR NOT NULL,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """

BIKE_ROUTES = """
        CREATE TABLE IF NOT EXISTS bike_routes (
            route_id        VARCHAR PRIMARY KEY,
            city            VARCHAR,
            layer_type      VARCHAR,
            layer_label     VARCHAR,
            color           VARCHAR,
            source_file     VARCHAR,
            min_lon         DOUBLE,
            min_lat         DOUBLE,
            max_lon         DOUBLE,
            max_lat         DOUBLE,
            geometry_json   VARCHAR,
            properties_json VARCHAR
        )
    """

BIKE_ROUTE_TRACTS = """
        CREATE TABLE IF NOT EXISTS bike_route_tracts (
            route_id    VARCHAR,
            tract_fips  VARCHAR,
            overlap_m   DOUBLE,
            PRIMARY KEY (route_id, tract_fips)
        )
    """

BIKE_ROUTE_ZCTAS = """
        CREATE TABLE IF NOT EXISTS bike_route_zctas (
            route_id  VARCHAR,
            zcta5     VARCHAR,
            overlap_m DOUBLE,
            PRIMARY KEY (route_id, zcta5)
        )
    """

BIKE_ROUTE_PLACES = """
        CREATE TABLE IF NOT EXISTS bike_route_places (
            route_id    VARCHAR,
            place_geoid VARCHAR,
            overlap_m   DOUBLE,
            PRIMARY KEY (route_id, place_geoid)
        )
    """

TRANSIT_ROUTE_ZCTAS = """
        CREATE TABLE IF NOT EXISTS transit_route_zctas (
            route_uid      VARCHAR,
            zcta5          VARCHAR,
            overlap_m      DOUBLE,
            route_fraction DOUBLE,
            PRIMARY KEY (route_uid, zcta5)
        )
    """

TRANSIT_ROUTE_PLACES = """
        CREATE TABLE IF NOT EXISTS transit_route_places (
            route_uid      VARCHAR,
            place_geoid    VARCHAR,
            overlap_m      DOUBLE,
            route_fraction DOUBLE,
            PRIMARY KEY (route_uid, place_geoid)
        )
    """

TRANSIT_ROUTE_AGENCIES = """
        CREATE TABLE IF NOT EXISTS transit_route_agencies (
            route_uid   VARCHAR,
            ntd_id      VARCHAR,
            raw_ntd_id  VARCHAR,
            PRIMARY KEY (route_uid, ntd_id)
        )
    """

CRIME_CITY_YEAR = """
        CREATE OR REPLACE VIEW crime_city_year AS
        SELECT city,
               year,
               COUNT(*) AS incidents,
               SUM(COALESCE(severity_weight, 0)) AS severity_weighted
        FROM crime_incidents
        WHERE city IS NOT NULL AND year IS NOT NULL
        GROUP BY city, year
    """
CBSA_COUNTIES = """
        CREATE TABLE IF NOT EXISTS cbsa_counties (
            cbsa_code       VARCHAR,
            cbsa_title      VARCHAR,
            msa_type        VARCHAR,
            state_fips      VARCHAR,
            county_fips     VARCHAR,
            county_name     VARCHAR,
            state_name      VARCHAR
        )
    """

ZCTA_BOUNDARIES = """
        CREATE TABLE IF NOT EXISTS zcta_boundaries (
            zcta5           VARCHAR PRIMARY KEY,  -- 5-digit Census ZCTA (not an exact USPS ZIP boundary)
            geo_id          VARCHAR,
            classfp         VARCHAR,
            mtfcc           VARCHAR,
            funcstat        VARCHAR,
            aland           BIGINT,               -- land area, square metres (TIGER ALAND20)
            awater          BIGINT,               -- water area, square metres (TIGER AWATER20)
            land_area_sq_mi DOUBLE,
            water_area_sq_mi DOUBLE,
            lat             DOUBLE,               -- representative point latitude (EPSG:4326)
            lon             DOUBLE,
            min_lon         DOUBLE,
            min_lat         DOUBLE,
            max_lon         DOUBLE,
            max_lat         DOUBLE,
            geometry_json   VARCHAR,              -- GeoJSON Polygon / MultiPolygon
            properties_json VARCHAR,               -- every original TIGER field, for provenance
            source_file     VARCHAR
        )
    """

PLACE_BOUNDARIES = """
        CREATE TABLE IF NOT EXISTS place_boundaries (
            place_geoid     VARCHAR PRIMARY KEY,  -- 7-digit state+place GEOID
            place_fips      VARCHAR,              -- 5-digit place code within the state
            state_fips      VARCHAR,
            name            VARCHAR,              -- TIGER NAME, e.g. "Pittsburgh city"
            short_name      VARCHAR,              -- legal suffix removed, e.g. "Pittsburgh"
            namelsad        VARCHAR,
            lsad            VARCHAR,
            classfp         VARCHAR,
            funcstat        VARCHAR,
            aland           BIGINT,
            awater          BIGINT,
            land_area_sq_mi DOUBLE,
            water_area_sq_mi DOUBLE,
            lat             DOUBLE,
            lon             DOUBLE,
            min_lon         DOUBLE,
            min_lat         DOUBLE,
            max_lon         DOUBLE,
            max_lat         DOUBLE,
            geometry_json   VARCHAR,              -- GeoJSON Polygon / MultiPolygon
            properties_json VARCHAR,
            source_file     VARCHAR
        )
    """

CENSUS_MSA = """
        CREATE TABLE IF NOT EXISTS census_msa (
            msa_code    VARCHAR PRIMARY KEY,
            geo_id      VARCHAR,
            name        VARCHAR,
            population  INTEGER
        )
    """

CENSUS_TRACTS = """
        CREATE TABLE IF NOT EXISTS census_tracts (
            tract_fips  VARCHAR PRIMARY KEY,
            geo_id      VARCHAR,
            name        VARCHAR,
            population  INTEGER
        )
    """

CRIME_INCIDENTS = """
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
    """

DATA_SOURCE_LOG = """
        CREATE TABLE IF NOT EXISTS data_source_log (
            source_key      VARCHAR PRIMARY KEY,
            display_name    VARCHAR,
            last_loaded_at  TIMESTAMP,
            row_count       BIGINT,
            source_files    VARCHAR,   -- JSON list of filenames involved in the last refresh
            message         VARCHAR    -- short human-readable outcome, e.g. warnings/skips
        )
    """

GEOCODE_CACHE = """
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
    """

HOUSES = """
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
    """

HOUSE_COMMUTE = """
        CREATE TABLE IF NOT EXISTS house_commute (
            house_id             VARCHAR,
            work_key             VARCHAR,
            work_label           VARCHAR,
            drive_min            DOUBLE,
            drive_miles          DOUBLE,
            bike_min             DOUBLE,
            bike_miles            DOUBLE,
            walk_min             DOUBLE,
            walk_miles            DOUBLE,
            transit_min          DOUBLE,
            transit_transfers    INTEGER,
            straight_line_miles  DOUBLE,
            status                VARCHAR,
            source                VARCHAR,
            PRIMARY KEY (house_id, work_key)
        )
    """

HOUSE_SNAPSHOTS = """
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
    """

MARKET_HEAT_INDEX = """
        CREATE TABLE IF NOT EXISTS market_heat_index (
            region_id       VARCHAR,
            region_type     VARCHAR,
            region_name     VARCHAR,
            size_rank       INTEGER,
            state_name      VARCHAR,
            state           VARCHAR,
            city            VARCHAR,
            metro           VARCHAR,
            county_name     VARCHAR,
            msa_code        VARCHAR,    -- see zhvi.msa_code above
            date            DATE,
            heat_index      DOUBLE,     -- ~0-100+; higher = hotter / more seller-favorable market
            source_file     VARCHAR,
            PRIMARY KEY (region_id, region_type, date)
        )
    """

NRI_TRACTS = """
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
    """

SOLD_HOMES = """
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
    """

TRANSIT_AGENCIES = """
        CREATE TABLE IF NOT EXISTS transit_agencies (
            ntd_id                  VARCHAR,
            legacy_ntd_id           VARCHAR,
            agency                  VARCHAR,
            mode                    VARCHAR,
            tos                     VARCHAR,
            three_mode              VARCHAR,
            mode_status             VARCHAR,
            reporter_type           VARCHAR,
            organization_type       VARCHAR,
            hq_city                 VARCHAR,
            hq_state                VARCHAR,
            uace_cd                 VARCHAR,
            uza_name                VARCHAR,
            uza_sq_miles            DOUBLE,
            uza_population          DOUBLE,
            service_area_population DOUBLE,
            service_area_sq_miles   DOUBLE,
            operating_expenses      DOUBLE,
            fare_revenues           DOUBLE,
            avg_cost_per_trip       DOUBLE,
            avg_fares_per_trip      DOUBLE,
            msa_code                VARCHAR,
            PRIMARY KEY (ntd_id, mode, tos)
        )
    """

TRANSIT_RIDERSHIP = """
        CREATE TABLE IF NOT EXISTS transit_ridership (
            ntd_id      VARCHAR,
            agency      VARCHAR,
            mode        VARCHAR,
            tos         VARCHAR,
            msa_code    VARCHAR,
            date        DATE,
            year        INTEGER,
            month       INTEGER,
            upt         DOUBLE,
            PRIMARY KEY (ntd_id, mode, tos, date)
        )
    """

TRANSIT_ROUTES = """
        CREATE TABLE IF NOT EXISTS transit_routes (
            route_uid       VARCHAR PRIMARY KEY,
            route_id        VARCHAR,
            ntd_id          VARCHAR,
            feed_id         VARCHAR,
            shape_id        VARCHAR,
            agency_id       VARCHAR,
            route_short_name VARCHAR,
            route_long_name VARCHAR,
            route_desc      VARCHAR,
            route_type      VARCHAR,
            route_type_name VARCHAR,
            route_url       VARCHAR,
            route_color     VARCHAR,
            route_text_color VARCHAR,
            download_date   VARCHAR,
            min_lon         DOUBLE,
            min_lat         DOUBLE,
            max_lon         DOUBLE,
            max_lat         DOUBLE,
            geometry_json   VARCHAR,
            properties_json VARCHAR
        )
    """

TRANSIT_SAFETY = """
        CREATE TABLE IF NOT EXISTS transit_safety (
            year                            INTEGER,
            ntd_id                          VARCHAR,
            agency                          VARCHAR,
            mode                            VARCHAR,
            tos                             VARCHAR,
            city                            VARCHAR,
            state                           VARCHAR,
            uace_cd                         VARCHAR,
            uza_name                        VARCHAR,
            reporter_type                   VARCHAR,
            msa_code                        VARCHAR,
            event_total                     INTEGER,
            collision_total                 INTEGER,
            derailment_total                INTEGER,
            fire_total                      INTEGER,
            security_total                  INTEGER,
            transit_worker_assault_events   INTEGER,
            total_fatalities                INTEGER,
            fatality_passenger              INTEGER,
            fatality_employee               INTEGER,
            fatality_other                  INTEGER,
            total_injuries                  INTEGER,
            injury_passenger                INTEGER,
            injury_employee                 INTEGER,
            injury_other                    INTEGER,
            serious_injuries                INTEGER,
            upt                             DOUBLE,
            vrm                             DOUBLE,
            vrh                             DOUBLE,
            voms                            DOUBLE,
            PRIMARY KEY (year, ntd_id, mode, tos)
        )
    """

ZHVI = """
        CREATE TABLE IF NOT EXISTS zhvi (
            region_id       VARCHAR,
            region_type     VARCHAR,    -- country | state | metro | county | city | zip | neighborhood
            region_name     VARCHAR,
            size_rank       INTEGER,    -- Zillow popularity/size ranking; 0 = largest
            state_name      VARCHAR,
            state           VARCHAR,
            city            VARCHAR,
            metro           VARCHAR,
            county_name     VARCHAR,
            msa_code        VARCHAR,    -- CBSA code, resolved at load time from region_name (metro rows)
                                        -- or the metro column (zip/neighborhood/city rows) via the same
                                        -- CBSA-name matcher census_msa uses (services/data_loader.py).
                                        -- NULL for state/country rows or an unmatched metro name.
            date            DATE,
            home_value      DOUBLE,     -- ZHVI: smoothed, seasonally-adjusted typical home value ($)
            source_file     VARCHAR,
            PRIMARY KEY (region_id, region_type, date)
        )
    """


def drop_legacy_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Remove views from earlier builds; the architecture is physical-table-only."""


def create_core_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Houses, their snapshots and the commute rows derived from them."""
    conn.execute(HOUSES)
    conn.execute(HOUSE_COMMUTE)
    conn.execute(HOUSE_SNAPSHOTS)
    # Upgrading a database created before is_favorite existed: CREATE TABLE IF NOT EXISTS is a no-op there.
    try:
        conn.execute("ALTER TABLE houses ADD COLUMN IF NOT EXISTS is_favorite BOOLEAN DEFAULT FALSE")
    except Exception:
        pass


def create_geo_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Geography and hazard/valuation tables: NRI tracts, census tract/MSA populations, the CBSA crosswalk and Zillow series."""
    conn.execute(NRI_TRACTS)
    conn.execute(CENSUS_TRACTS)
    conn.execute(CENSUS_MSA)
    conn.execute(CBSA_COUNTIES)
    conn.execute(ZCTA_BOUNDARIES)
    conn.execute(PLACE_BOUNDARIES)
    conn.execute(ZHVI)
    conn.execute(MARKET_HEAT_INDEX)
    # Same upgrade path for msa_code, which earlier zhvi/market_heat_index tables lack.
    for _tbl in ("zhvi", "market_heat_index"):
        try:
            conn.execute(f"ALTER TABLE {_tbl} ADD COLUMN IF NOT EXISTS msa_code VARCHAR")
        except Exception:
            pass


def create_crime_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Standardized crime incidents (one row per offence, with the taxonomy's severity weight)."""
    conn.execute(CRIME_INCIDENTS)
    conn.execute(CRIME_CITY_YEAR)


def create_bike_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """BikePGH line geometry plus its route-to-tract spatial bridge."""
    conn.execute(BIKE_ROUTES)
    conn.execute(BIKE_ROUTE_TRACTS)
    conn.execute(BIKE_ROUTE_ZCTAS)
    conn.execute(BIKE_ROUTE_PLACES)


def create_transit_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """FTA National Transit Database tables plus the National Transit Map route geometry."""
    conn.execute(TRANSIT_AGENCIES)
    conn.execute(TRANSIT_RIDERSHIP)
    conn.execute(TRANSIT_SAFETY)
    conn.execute(TRANSIT_ROUTES)
    conn.execute(TRANSIT_ROUTE_ZCTAS)
    conn.execute(TRANSIT_ROUTE_PLACES)
    conn.execute(TRANSIT_ROUTE_AGENCIES)


def create_sales_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """County sold-home records and the geocode cache that avoids re-calling the Census geocoder."""
    conn.execute(SOLD_HOMES)
    conn.execute(GEOCODE_CACHE)


def create_app_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Bookkeeping tables: the Data page's refresh log, app settings, the catalog, the result cache and chat memory."""
    conn.execute(DATA_SOURCE_LOG)
    conn.execute(APP_SETTINGS)
    # app_settings gained value_json in an earlier build; keep the migration with the table it migrates.
    try:
        cols = [c[1].lower() for c in conn.execute("PRAGMA table_info(app_settings)").fetchall()]
        if "value_json" not in cols:
            conn.execute("ALTER TABLE app_settings ADD COLUMN value_json VARCHAR")
        if "value" in cols:
            conn.execute("UPDATE app_settings SET value_json = value WHERE value_json IS NULL AND value IS NOT NULL")
    except Exception:
        pass

    # The catalog, its seed, the result cache and chat memory own their own tables (and their own migrations).
    from db import catalog_store

    catalog_store.ensure_tables(conn)
    catalog_store.sync_seed(conn)

    from services import cache as result_cache

    result_cache.ensure_tables(conn)

    from services import session_memory

    session_memory.ensure_tables(conn)


# ── Join-key indexes ────────────────────────────────────────────────────────────
# The catalog declares 25 relationships, and every analytical query and map layer joins on
# them; without indexes DuckDB scans the whole table for each join — 740,679 crime rows, 85 k
# tracts and 300 k+ sales. Primary keys get theirs from the constraint; these are the foreign
# sides. Idempotent, so the cost is paid once per database file, not per connection.
_INDEXES = (
    ("houses", "city"),
    ("houses", "tract_fips"),
    ("houses", "msa_code"),
    ("houses", "crime_city"),
    ("house_commute", "house_id"),
    ("house_snapshots", "house_id"),
    ("nri_tracts", "county_fips"),
    ("cbsa_counties", "county_fips"),
    ("cbsa_counties", "cbsa_code"),
    ("place_boundaries", "state_fips"),
    ("place_boundaries", "short_name"),
    ("crime_incidents", "city"),
    ("crime_incidents", "year"),
    ("crime_incidents", "category"),
    ("sold_homes", "tract_fips"),
    ("sold_homes", "county_fips"),
    ("sold_homes", "city"),
    ("bike_routes", "city"),
    ("bike_route_zctas", "zcta5"),
    ("bike_route_places", "place_geoid"),
    ("transit_route_zctas", "zcta5"),
    ("transit_route_places", "place_geoid"),
    ("transit_route_agencies", "route_uid"),
    ("transit_route_agencies", "ntd_id"),
    ("transit_routes", "ntd_id"),
    ("transit_ridership", "ntd_id"),
    ("transit_ridership", "year"),
    ("transit_safety", "ntd_id"),
    ("zhvi", "region_name"),
    ("zhvi", "msa_code"),
    ("market_heat_index", "region_name"),
    ("market_heat_index", "msa_code"),
    ("chat_session_turns", "session_id"),
)


def create_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the join-key indexes the schema declares, ignoring ones that already exist."""
    for table, column in _INDEXES:
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table}({column})")


_ZIP_REPAIR_KEY = "zip_repair_v1"


def repair_zip_codes(conn: duckdb.DuckDBPyConnection, *, force: bool = False) -> int:
    """Zero-pad stored US ZIPs that lost a leading zero before they reached the database.

    A 4-digit ZIP is never valid in the US: ``1851`` is Cambridge's ``01851`` after a spreadsheet or a
    numeric CSV read dropped the zero.  Rows loaded before the loaders normalized on ingest keep the
    damaged form, and a re-created database restored from an older file inherits them, so the repair
    lives here rather than in a one-off script.

    Idempotent and conservative: it only rewrites values that are exactly four digits (leaving ZIP+4,
    already-clean values, NULLs and anything non-numeric untouched, because turning garbage into a
    plausible-looking ZIP is worse than leaving it visible).  A marker in ``app_settings`` stops the
    scan repeating on every connection; the storage boundary``_normalize_zip_frame`` is what keeps new
    writes clean, so the repair never has to run twice.
    """
    if not force:
        try:
            if conn.execute("SELECT 1 FROM app_settings WHERE key = ?", [_ZIP_REPAIR_KEY]).fetchone():
                return 0
        except Exception:
            pass  # app_settings not created yet: fall through and just repair
    changed = 0
    for table in ("houses", "sold_homes"):
        try:
            before = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE regexp_matches(CAST(zip AS VARCHAR), '^[0-9]{{4}}$')"
            ).fetchone()[0]
            if before:
                conn.execute(
                    f"UPDATE {table} SET zip = lpad(CAST(zip AS VARCHAR), 5, '0') "
                    "WHERE regexp_matches(CAST(zip AS VARCHAR), '^[0-9]{4}$')"
                )
            changed += int(before or 0)
        except Exception:
            continue  # table absent in this build: nothing to repair
    try:
        conn.execute(
            "INSERT OR REPLACE INTO app_settings (key, value_json) VALUES (?, ?)",
            [_ZIP_REPAIR_KEY, '"done"'],
        )
    except Exception:
        pass
    return changed


def repair_ct_tract_codes(conn: duckdb.DuckDBPyConnection, *, force: bool = False) -> int:
    """Re-key Connecticut census tracts onto the 2022 planning-region county codes.

    Connecticut replaced its eight counties with nine planning regions in 2022, and the tract GEOIDs moved
    with them: the tract number stayed, the five-digit county-equivalent prefix went from 090xx to 091xx
    (09009141600 -> 09170141600).  ``census_tracts`` is loaded from a 2020-vintage file and keeps the old
    prefix, while the house geocoder, ``nri_tracts`` and the ``cbsa_counties`` crosswalk all use the new
    one, so every Connecticut house silently dropped out of tract -> county -> CBSA -> MSA -> ZHVI
    (measured: 43 of 1,778 houses, all in CT).

    The prefix is not derivable from the tract number alone - a planning region spans several old counties
    (09140 covers 09001/09003/09005/09009) and two tract numbers repeat across counties - so the map comes
    from ``nri_tracts``, which is published on the current tract set and loaded before census in the full
    setup.  A number that is ambiguous there, or absent, is left exactly as it is and never guessed.
    Idempotent: only rows still on 090xx move, and only to a key that does not already exist.
    """
    try:
        if not conn.execute("SELECT 1 FROM nri_tracts WHERE state_fips = '09' LIMIT 1").fetchone():
            return 0  # no current-vintage source yet: retried on the next start rather than guessed
        moves = conn.execute(
            """
            SELECT ct.tract_fips AS old_fips, m.region || RIGHT(ct.tract_fips, 6) AS new_fips
            FROM census_tracts ct
            JOIN (
                SELECT RIGHT(tract_fips, 6) AS num, MIN(LEFT(tract_fips, 5)) AS region
                FROM nri_tracts
                WHERE state_fips = '09' AND LEFT(tract_fips, 5) LIKE '091%'
                GROUP BY 1
                HAVING COUNT(DISTINCT LEFT(tract_fips, 5)) = 1
            ) m ON m.num = RIGHT(ct.tract_fips, 6)
            WHERE ct.tract_fips LIKE '090%'
              AND NOT EXISTS (SELECT 1 FROM census_tracts t2 WHERE t2.tract_fips = m.region || RIGHT(ct.tract_fips, 6))
            """
        ).fetchall()
        if not moves:
            return 0
        conn.executemany("UPDATE census_tracts SET tract_fips = ? WHERE tract_fips = ?", [[new, old] for old, new in moves])
        return len(moves)
    except Exception:
        return 0  # a table this build does not have


def backfill_house_msa_codes(conn: duckdb.DuckDBPyConnection, *, force: bool = False) -> int:
    """Fill ``houses.msa_code`` from the census geography when a load could not.

    The Redfin export has no MSA column (see ``services/loaders/redfin.py``), so the loader derives the
    code from ``tract_fips`` -> county -> ``cbsa_counties``.  A load that ran before that crosswalk
    existed - or a house loaded without tract resolution - leaves the column NULL, and the metro ZHVI join
    then matches nothing.  This fills those rows from the same crosswalk.

    Deliberately unmarked: it runs only when there is something to fill *and* a crosswalk to fill it from,
    so loading census after redfin heals the column on the next start, and a later reload can never be
    swallowed by a "done" marker.  The two guards are one indexed probe each.
    """
    try:
        if not conn.execute("SELECT 1 FROM cbsa_counties LIMIT 1").fetchone():
            return 0
        pending = conn.execute(
            "SELECT COUNT(*) FROM houses WHERE msa_code IS NULL AND tract_fips IS NOT NULL"
        ).fetchone()[0]
        if not pending:
            return 0
        conn.execute(
            """
            UPDATE houses SET msa_code = c.cbsa_code
            FROM cbsa_counties c
            WHERE houses.msa_code IS NULL
              AND houses.tract_fips IS NOT NULL
              AND c.state_fips || c.county_fips = LEFT(houses.tract_fips, 5)
            """
        )
        left = conn.execute(
            "SELECT COUNT(*) FROM houses WHERE msa_code IS NULL AND tract_fips IS NOT NULL"
        ).fetchone()[0]
        return int(pending - left)
    except Exception:
        return 0


def initialize(conn: duckdb.DuckDBPyConnection) -> None:
    """Create every table this app owns, in dependency-free order."""
    drop_legacy_views(conn)
    create_core_tables(conn)
    create_geo_tables(conn)
    create_crime_tables(conn)
    create_bike_tables(conn)
    create_transit_tables(conn)
    create_sales_tables(conn)
    create_app_tables(conn)
    create_indexes(conn)  # last: chat_session_turns is created inside create_app_tables
    repair_zip_codes(conn)  # after app_settings exists: it records its marker there
    repair_ct_tract_codes(conn)  # CT moved to planning regions in 2022; heal 2020-vintage tract codes
    backfill_house_msa_codes(conn)  # tract -> county -> CBSA, once that crosswalk has been loaded
