"""
schema_catalog.py

Single source of truth for "what data exists and how it fits together."

This is the metadata layer: instead of telling the LLM *rules* in prose
("never GROUP BY msa_code", "always CAST population", "use tool X for
question type Y"), we attach short, structured notes to the actual
table / column / relationship they concern, and let the agent read them
through one tool (get_database_schema) and reason from there with plain
SQL (query_database). The agent decides what to query and how to join it;
this module just makes sure it isn't guessing at things we already know.

Two kinds of knowledge live here, deliberately kept separate:

  1. LIVE facts — column names, types, row counts. Pulled from the running
     DuckDB file itself (DESCRIBE / COUNT(*)) every time they're needed, so
     they're always correct even if a local .duckdb file predates a schema
     change in duckdb_store.py. This is what makes hardcoded warnings like
     "there is no msa_name column" unnecessary — the live column list
     simply doesn't offer it.

  2. CURATED facts — things no amount of schema introspection can tell you:
     which columns are reliably populated, which joins need a non-trivial
     expression (e.g. concatenating two zero-padded FIPS fragments), which
     tables need a default filter to be meaningful (arm's-length sales).
     This is genuine domain knowledge worth writing down ONCE.

Adding a new dataset:
  1. Add the table in db/duckdb_store.py::_ensure_schema, as before.
  2. Add one TableMeta entry below (a description, plus notes only for the
     non-obvious columns).
  3. If it joins to an existing table, add one Relationship entry below.
That's it. check_data_availability, get_database_schema, setup_data.py's
summary, and the response validator's fallback message all read from here,
so nothing else needs to change or drift out of sync.
"""
from __future__ import annotations

import re as _re
from dataclasses import dataclass, field

import db.duckdb_store as store


# ─────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class ColumnNote:
    column: str
    note: str


@dataclass
class Relationship:
    """A documented join path between two tables."""
    left_table: str
    left_on: str        # SQL expression, evaluated against left_table
    right_table: str
    right_on: str        # SQL expression, evaluated against right_table
    note: str = ""        # gotcha / explanation — format quirks, reliability, caveats

    def involves(self, tables: set[str]) -> bool:
        return self.left_table in tables and self.right_table in tables

    def render(self) -> str:
        line = f"{self.left_table}.{self.left_on} = {self.right_table}.{self.right_on}"
        return f"{line}\n    Note: {self.note}" if self.note else line


@dataclass
class TableMeta:
    name: str
    description: str
    setup_hint: str = ""                        # what to run if this table is empty
    column_notes: list[ColumnNote] = field(default_factory=list)
    hidden_columns: tuple[str, ...] = ()          # omit from LLM-facing schema (blobs, etc.)
    agent_visible: bool = True                    # include in get_database_schema output
    filter_hint: str = ""                          # default WHERE clause for "valid" rows


# ─────────────────────────────────────────────────────────────────────────
# NRI hazard columns — one place for the 18 FEMA hazard scores and their
# human labels. Reused by the schema renderer AND by the house agent's
# get_nri_risk_data tool, instead of two hand-maintained copies of the map.
# ─────────────────────────────────────────────────────────────────────────

NRI_HAZARD_COLUMNS: dict[str, str] = {
    "avln_risks": "Avalanche",
    "cfld_risks": "Coastal Flooding",
    "cwav_risks": "Cold Wave",
    "drgt_risks": "Drought",
    "erqk_risks": "Earthquake",
    "hail_risks": "Hail",
    "hwav_risks": "Heat Wave",
    "hrcn_risks": "Hurricane",
    "istm_risks": "Ice Storm",
    "lnds_risks": "Landslide",
    "ltng_risks": "Lightning",
    "rfld_risks": "Riverine Flooding",
    "swnd_risks": "Strong Wind",
    "trnd_risks": "Tornado",
    "tsun_risks": "Tsunami",
    "vlcn_risks": "Volcanic Activity",
    "wfir_risks": "Wildfire",
    "wntw_risks": "Winter Weather",
}


# ─────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────

TABLES: dict[str, TableMeta] = {

    "houses": TableMeta(
        name="houses",
        description=(
            "Your saved/favorited Redfin listings — the CURRENT (latest) state of "
            "each property. One row per house_id."
        ),
        setup_hint="Drop Redfin CSV export(s) in data/redfin/, then run: python setup_data.py --only redfin",
        hidden_columns=("raw_json",),
        column_notes=[
            ColumnNote("msa_code",
                       "Usually NULL. Redfin exports don't include CBSA codes, so most houses "
                       "have no metro-area code. Group/filter houses by city or state instead — "
                       "if you need metro-area rollups, check "
                       "COUNT(*) FILTER (WHERE msa_code IS NOT NULL) first so you know the coverage."),
            ColumnNote("tract_fips",
                       "11-digit census tract FIPS, resolved from lat/lon at load time. "
                       "This is the reliable key for joining to nri_tracts / census_tracts."),
            ColumnNote("status",
                       "Free-text from Redfin (Active, Pending, Contingent, Sold, Pre-Market, ...). "
                       "For the fullest history use house_snapshots instead."),
            ColumnNote("price",
                       "List price as of the last Redfin export. For a sold price use "
                       "house_snapshots (source_type='sold') or sold_homes."),
        ],
    ),

    "house_snapshots": TableMeta(
        name="house_snapshots",
        description=(
            "Full history of every observed state per house — one row per "
            "(house, source file, status, price) ever seen, across every Redfin "
            "re-export plus any matched county sale record. Use this table for "
            "price-history / days-on-market / 'how many price cuts' questions; "
            "`houses` only holds the latest snapshot."
        ),
        column_notes=[
            ColumnNote("source_type", "'redfin' (a listing snapshot) or 'sold' (a matched county sale record)."),
            ColumnNote("snapshot_date",
                       "Can be NULL for older rows where no date could be parsed from the source "
                       "filename — use ORDER BY snapshot_date DESC NULLS LAST."),
        ],
    ),

    "nri_tracts": TableMeta(
        name="nri_tracts",
        description=(
            "FEMA National Risk Index — one row per census tract, with a composite "
            f"risk score plus {len(NRI_HAZARD_COLUMNS)} per-hazard risk scores (columns "
            "ending in _risks; higher = more risk). See 'Per-hazard risk columns' below."
        ),
        setup_hint="Place NRI_Table_CensusTracts.csv (or the NRI shapefile) in data/nri/, then run: python setup_data.py --only nri",
        column_notes=[
            ColumnNote("tract_fips", "11-digit census tract FIPS — primary key; joins to houses/sold_homes."),
            ColumnNote("county_fips",
                       "5-digit (state+county) FIPS, e.g. '42003'. Use THIS — not tract_fips — "
                       "to reach cbsa_counties for metro-area rollups (see Relationships)."),
            ColumnNote("risk_score", "Composite score blending all hazards — use for 'overall risk' questions."),
            ColumnNote("eal_valt",
                       "Expected Annual Loss in dollars — use when the user wants a $ figure "
                       "rather than a 0-100 score."),
        ],
    ),

    "census_tracts": TableMeta(
        name="census_tracts",
        description="2020 Census total population, one row per census tract.",
        setup_hint=("Download table DECENNIALPL2020.P1 (Census Tracts geography) from "
                     "data.census.gov, save as data/census/DECENNIALPL2020_P1_tract.csv, "
                     "then run: python setup_data.py --only census"),
        column_notes=[
            ColumnNote("population",
                       "Cast defensively when sorting/comparing — CAST(population AS BIGINT) — "
                       "the live column type is shown above for the database you're actually "
                       "running against, but older database files may still have this as text."),
        ],
    ),

    "census_msa": TableMeta(
        name="census_msa",
        description="2020 Census total population per Metropolitan/Micropolitan Statistical Area (MSA).",
        setup_hint=("Download table DECENNIALPL2020.P1 (Metropolitan Statistical Area geography) "
                     "from data.census.gov, save as data/census/DECENNIALPL2020_P1_msa.csv, "
                     "then run: python setup_data.py --only census"),
        column_notes=[
            ColumnNote("msa_code",
                       "5-digit CBSA code, matched by MSA NAME against cbsa_counties.cbsa_title "
                       "when this table was loaded. Rows that couldn't be matched by name get a "
                       "placeholder starting with 'X' instead of a real CBSA code — filter those "
                       "out (WHERE msa_code NOT LIKE 'X%') before joining to cbsa_counties."),
            ColumnNote("population", "Cast defensively when sorting — CAST(population AS BIGINT)."),
        ],
    ),

    "cbsa_counties": TableMeta(
        name="cbsa_counties",
        description=(
            "Census Bureau CBSA delineation file — maps every US county to its "
            "Metropolitan/Micropolitan Statistical Area. This is the bridge table between "
            "MSA-level data (census_msa) and tract/county-level data (nri_tracts) — there is "
            "no direct msa_code<->tract_fips column, so route through this table (see Relationships)."
        ),
        setup_hint=("Download the CBSA delineation file (list1_*.xls/.xlsx/.csv) from census.gov "
                     "and save it under data/census/, then run: python setup_data.py --only census"),
        column_notes=[
            ColumnNote("state_fips", "2-digit, zero-padded."),
            ColumnNote("county_fips",
                       "3-digit, zero-padded — a COUNTY-only code, not a full FIPS. Concatenate "
                       "with state_fips to match nri_tracts.county_fips (see Relationships)."),
            ColumnNote("msa_type",
                       "'Metropolitan Statistical Area' or 'Micropolitan Statistical Area' — filter "
                       "to Metropolitan if the user specifically says 'metro' or 'MSA'."),
        ],
    ),

    "sold_homes": TableMeta(
        name="sold_homes",
        description=(
            "County assessor sale records — a broader, independent set of transactions beyond "
            "just your Redfin favorites (includes houses you never saved). NOT linked to `houses` "
            "by house_id; overlap with your favorites is captured separately in house_snapshots "
            "(source_type='sold'), matched by address."
        ),
        setup_hint="Drop county sale-record CSV(s) in data/sold/, then run: python setup_data.py --only sold",
        filter_hint="(is_arms_length IS NULL OR is_arms_length = TRUE) AND sold_price > 1000",
        column_notes=[
            ColumnNote("is_arms_length",
                       "FALSE means the sale was a gift, foreclosure, corrective deed, etc. — NOT a "
                       "market-rate transaction. For price comps or medians always apply the default "
                       "filter above, or your numbers will be skewed by non-market sales."),
            ColumnNote("tract_fips", "NULL until geocoded — check geocode_status='success' first."),
            ColumnNote("lat", "NULL until geocoded — check geocode_status='success' before relying on lat/lon."),
        ],
    ),

    "crime_incidents": TableMeta(
        name="crime_incidents",
        description=(
            "Standardized, severity-weighted crime incidents, combined from whatever per-city raw "
            "exports are dropped in data/crime/<city>/ (each city reports crime differently — see "
            "services/crime_sources.py for the per-city parsers and services/crime_taxonomy.py for "
            "how offense text is mapped to a common category + weight). Backs the 'Crime' map layer, "
            "but also queryable directly for questions like 'which of these tracts has the most "
            "severe crime' or 'burglaries in Pittsburgh in 2022'."
        ),
        setup_hint=(
            "Drop each city's raw crime export (.csv or .xlsx) in data/crime/<city>/ — one folder per "
            "city, e.g. data/crime/chicago/, data/crime/pittsburgh/ — then run: "
            "python setup_data.py --only crime"
        ),
        column_notes=[
            ColumnNote("city",
                       "Normalized key (e.g. 'chicago', 'pittsburgh') — matches the data/crime/<city>/ "
                       "folder name it was loaded from, and houses.crime_city (see Relationships)."),
            ColumnNote("severity_weight",
                       "1.0 (least severe) - 10.0 (most severe). Use SUM(severity_weight) rather than "
                       "COUNT(*) for anything about how BAD an area's crime is, not just how much of "
                       "it there is — that's the whole point of the weighting. COUNT(*) is still right "
                       "for plain incident-volume questions."),
            ColumnNote("category", "Standardized category key, e.g. 'aggravated_assault'. "
                                    "category_label is the human-readable form, e.g. 'Aggravated Assault'."),
            ColumnNote("raw_type",
                       "Original offense/description text from the source file, kept for reference — "
                       "not standardized, don't group by this across cities."),
            ColumnNote("year_month", "'YYYY-MM' string — convenient for GROUP BY on monthly trend questions."),
        ],
    ),

    "bike_routes": TableMeta(
        name="bike_routes",
        description=(
            "Standardized BikePGH-style bicycle network data across cities. Each row is one "
            "line feature from one of the seven recognized sublayers under data/bike/<city>/: "
            "Bike Lanes, Bikeable Sidewalks, Cautionary Bike Route, On Street Bike Route, "
            "Protected Bike Lanes, Sharrows, or Trails. Geometry is stored as GeoJSON in WGS-84 "
            "and each feature has a viewport-friendly bounding box. This table powers the "
            "'Bike Lanes' map layer and is also queryable for counts, coverage, cities, and "
            "layer-specific summaries. For spatial display/filtering, use the bbox columns; "
            "geometry_json is a serialized GeoJSON geometry and should not be parsed in SQL."
        ),
        setup_hint=(
            "Drop BikePGH-style shapefiles into data/bike/<city>/<layer folder>/ and run: "
            "python setup_data.py --only bike"
        ),
        hidden_columns=("geometry_json", "properties_json"),
        column_notes=[
            ColumnNote("city", "Normalized lower-case city folder name under data/bike/<city>/."),
            ColumnNote("layer_type", "Stable key: bike_lanes, bikeable_sidewalks, cautionary_bike_route, "
                                   "on_street_bike_route, protected_bike_lanes, sharrows, or trails."),
            ColumnNote("layer_label", "Human-readable sublayer label shown in the map."),
            ColumnNote("color", "Visualization color from the standard R implementation: steelblue, "
                               "lightblue, red, lightgreen, darkgreen, orange, or pink."),
            ColumnNote("min_lon/min_lat/max_lon/max_lat", "WGS-84 feature bounding box. Use these for SQL viewport/intersection filters."),
            ColumnNote("properties_json", "Original shapefile attributes serialized as JSON. Prefer standard columns above for cross-city analysis."),
        ],
    ),

    "geocode_cache": TableMeta(
        name="geocode_cache",
        description="Internal address→lat/lon cache so re-running setup_data.py doesn't re-hit the geocoding API.",
        agent_visible=False,   # bookkeeping table — not useful for analysis questions
    ),
}


RELATIONSHIPS: list[Relationship] = [
    Relationship("houses", "tract_fips", "nri_tracts", "tract_fips",
                 note="Always safe — tract_fips is well populated on houses."),
    Relationship("houses", "tract_fips", "census_tracts", "tract_fips",
                 note="Tract-level population for a house's neighborhood."),
    Relationship("sold_homes", "tract_fips", "nri_tracts", "tract_fips",
                 note="Only rows with sold_homes.geocode_status='success' have a tract_fips to join on."),
    Relationship("houses", "msa_code", "census_msa", "msa_code",
                 note="Weak join — houses.msa_code is usually NULL (Redfin doesn't export CBSA codes). "
                      "Prefer grouping houses by city/state; only use this for the minority of rows "
                      "that do have msa_code populated."),
    Relationship("census_msa", "msa_code", "cbsa_counties", "cbsa_code",
                 note="Exclude unmatched rows first: WHERE census_msa.msa_code NOT LIKE 'X%'."),
    Relationship("cbsa_counties", "state_fips || county_fips", "nri_tracts", "county_fips",
                 note="Concatenation join — cbsa_counties splits state and county into two "
                      "zero-padded fields; nri_tracts already stores the combined county code. "
                      "This is the only path from MSA-level rollups down to NRI risk scores."),
    Relationship("house_snapshots", "house_id", "houses", "house_id",
                 note="Full history per house — use for price-over-time / days-on-market questions."),
    Relationship("houses", "crime_city", "crime_incidents", "city",
                 note="Weak join — houses.crime_city is only populated when a Redfin CSV set it "
                      "explicitly, or when the house's own city exactly matches a covered crime-data "
                      "city (see services/data_loader._infer_crime_city). A house in a suburb (e.g. "
                      "Cambridge, MA) won't auto-match Boston's crime data unless crime_city was set "
                      "explicitly. This is a plain city-name join, not spatial — it does NOT restrict "
                      "to the house's own neighborhood, only its city."),
]


# ─────────────────────────────────────────────────────────────────────────
# Live introspection — always asks the running database, never hardcoded.
# ─────────────────────────────────────────────────────────────────────────

def _live_columns(table: str) -> list[tuple[str, str]]:
    """[(column_name, column_type), ...] straight from the running database."""
    try:
        df = store.query(f"DESCRIBE {table}")
        return list(zip(df["column_name"].tolist(), df["column_type"].tolist()))
    except Exception:
        return []


def _row_count(table: str) -> int | None:
    try:
        return int(store.query(f"SELECT COUNT(*) AS n FROM {table}").iloc[0]["n"])
    except Exception:
        return None


def list_table_names(agent_visible_only: bool = False) -> list[str]:
    names = list(TABLES.keys())
    if agent_visible_only:
        names = [n for n in names if TABLES[n].agent_visible]
    return names


# ─────────────────────────────────────────────────────────────────────────
# Rendering — what the agent (and the tools it calls) actually see.
# ─────────────────────────────────────────────────────────────────────────

def availability_report() -> tuple[str, dict[str, int]]:
    """
    Row counts for every registered table, plus setup hints for empty ones.
    Returns (rendered_text, {table: row_count}). Backs both the
    check_data_availability tool and the passive per-turn system context,
    so the two can never drift out of sync with each other again.
    """
    lines = ["Table row counts (0 = not loaded yet):"]
    counts: dict[str, int] = {}
    empty_hints = []
    for name, meta in TABLES.items():
        n = _row_count(name)
        counts[name] = n or 0
        if n is None:
            lines.append(f"  ?  {name:<20} (table missing or query failed)")
        else:
            status = "✓" if n > 0 else "✗ EMPTY"
            lines.append(f"  {status}  {name:<20} {n:>10,} rows")
            if n == 0 and meta.setup_hint:
                empty_hints.append(f"  {name}: {meta.setup_hint}")
    if empty_hints:
        lines.append("")
        lines.append("Setup instructions for empty tables:")
        lines.extend(empty_hints)
    return "\n".join(lines), counts


def render_schema_for_agent() -> str:
    """
    The LLM-facing schema: live columns/types for every agent-visible table,
    plus curated notes and the documented join graph. This is the one place
    that formats "what tables exist and how do I query them" — replacing
    what used to be a hardcoded docstring plus a chunk of the system prompt.
    """
    parts = []
    for name, meta in TABLES.items():
        if not meta.agent_visible:
            continue
        cols = [(c, t) for c, t in _live_columns(name) if c not in meta.hidden_columns]

        parts.append(f"### {name}")
        parts.append(meta.description)
        if cols:
            col_lines = "\n".join(f"    {c:<20} {t}" for c, t in cols)
            parts.append(f"  Columns (live, from the database you're running against):\n{col_lines}")
        if name == "nri_tracts":
            hz = "\n".join(f"    {c:<12} {label}" for c, label in NRI_HAZARD_COLUMNS.items())
            parts.append(f"  Per-hazard risk columns (0-100, higher = more risk):\n{hz}")
        if meta.filter_hint:
            parts.append(f"  Default filter for valid rows: WHERE {meta.filter_hint}")
        if meta.column_notes:
            notes = "\n".join(f"    - {n.column}: {n.note}" for n in meta.column_notes)
            parts.append(f"  Notes:\n{notes}")
        parts.append("")

    parts.append("### Relationships (joins)")
    parts.append(
        "There is no ORM/FK enforcement in DuckDB here — these are the join paths that "
        "actually return matching rows. Prefer them over guessing a natural-looking join."
    )
    for rel in RELATIONSHIPS:
        parts.append(f"  {rel.render()}")

    return "\n".join(parts)


def diagnose_empty_or_error(sql: str) -> str:
    """
    Best-effort diagnostics for a query that ran but returned 0 rows.

      1. If any referenced table is actually empty, say so (and how to load it) —
         this is the common case and needs no join-level reasoning at all.
      2. Otherwise, if two referenced tables share a documented Relationship,
         surface the documented join expression — the query likely used a
         different (wrong) one.

    Returns '' if there's nothing more useful to say than "0 rows".
    Generic by construction: works for any table pair with a declared
    Relationship, not just the one question this used to be hand-written for.
    """
    sql_upper = sql.upper()
    tables_in_query = {
        t.lower() for t in
        _re.findall(r'\bFROM\s+([A-Za-z_]\w*)', sql_upper) +
        _re.findall(r'\bJOIN\s+([A-Za-z_]\w*)', sql_upper)
    }
    known_tables = tables_in_query & set(TABLES.keys())

    empty = sorted(t for t in known_tables if (_row_count(t) or 0) == 0)
    if empty:
        hints = "\n".join(f"  - {t}: {TABLES[t].setup_hint}" for t in empty if TABLES[t].setup_hint)
        extra = f"\n{hints}" if hints else ""
        return (f"EMPTY TABLES DETECTED: {', '.join(empty)} have 0 rows.{extra}\n"
                f"These files have not been loaded yet. DO NOT fabricate results.")

    tips = [rel.render() for rel in RELATIONSHIPS if rel.involves(known_tables)]
    if tips:
        return ("Query returned 0 rows. All referenced tables have data, so this is most "
                "likely a join or filter issue rather than missing data. Documented join "
                "path(s) for these tables:\n" + "\n".join(f"  {t}" for t in tips))

    return ""
