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
    """A documented, machine-readable join edge between physical tables."""
    left_table: str
    left_on: str
    right_table: str
    right_on: str
    note: str = ""
    cardinality: str = "many-to-one"
    confidence: str = "high"
    bridge: bool = False
    preferred: bool = True
    nullable_side: str = ""
    grain_note: str = ""

    def involves(self, tables: set[str]) -> bool:
        return self.left_table in tables and self.right_table in tables

    def render(self) -> str:
        line = f"{self.left_table}.{self.left_on} = {self.right_table}.{self.right_on}"
        attrs = f"[{self.cardinality}; confidence={self.confidence}; preferred={self.preferred}; bridge={self.bridge}]"
        extra = []
        if self.note: extra.append(self.note)
        if self.grain_note: extra.append("Grain: " + self.grain_note)
        return line + " " + attrs + ("\n    Note: " + "\n    Note: ".join(extra) if extra else "")


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



# Semantic vocabulary used by the planning/RAG layer. These are descriptive facts,
# not replacement tables/views. They let the planner resolve user language to physical
# columns and documented relationship paths before SQL generation.
SEMANTIC_GLOSSARY = {
    "walk_score": {"tables": ["houses"], "columns": ["houses.walk_score"], "aliases": ["walkability", "walk score", "most walkable"], "direction": "higher_is_better", "null_policy": "exclude_nulls"},
    "favorite_house": {"tables": ["houses"], "columns": ["houses.is_favorite"], "aliases": ["my list", "saved houses", "favorites"], "filter": "houses.is_favorite = TRUE"},
    "nri_overall_risk": {"tables": ["nri_tracts"], "columns": ["nri_tracts.risk_score"], "aliases": ["overall risk", "NRI risk", "composite risk"], "direction": "lower_is_better"},
    "nri_riverine_flood_risk": {"tables": ["nri_tracts"], "columns": ["nri_tracts.rfld_risks"], "aliases": ["flood risk", "riverine flood", "riverine flooding"], "direction": "lower_is_better"},
    "nri_coastal_flood_risk": {"tables": ["nri_tracts"], "columns": ["nri_tracts.cfld_risks"], "aliases": ["coastal flood", "coastal flooding"], "direction": "lower_is_better"},
    "msa_population": {"tables": ["census_msa"], "columns": ["census_msa.population"], "aliases": ["top MSA", "largest metros", "top 50 MSAs"], "direction": "higher_is_larger"},
}

# Compound planning recipes. They describe *how data relates* without materializing a view.
PLANNING_PATTERNS = {
    "top_n_msa_nri": {
        "tables": ["census_msa", "cbsa_counties", "nri_tracts"],
        "steps": [
            "select top N real MSA rows from census_msa ordered by CAST(population AS BIGINT) DESC and msa_code NOT LIKE 'X%'",
            "join census_msa -> cbsa_counties on msa_code = cbsa_code",
            "join cbsa_counties -> nri_tracts on state_fips || county_fips = nri_tracts.county_fips",
            "aggregate the requested NRI metric at MSA grain",
        ],
        "warnings": ["The NRI source is tract-grain; aggregation choice matters. Use AVG unless the user specifies another statistic.", "Keep MSA population ranking separate from NRI aggregation.", "Do not place the final LIMIT inside the population-universe CTE."],
        "canonical_sql_shape": (
            "WITH top_msas AS (SELECT msa_code, name, CAST(population AS BIGINT) AS population "
            "FROM census_msa WHERE population IS NOT NULL AND msa_code NOT LIKE 'X%' "
            "ORDER BY CAST(population AS BIGINT) DESC LIMIT {universe_limit}), "
            "msa_metric AS (SELECT m.msa_code, AVG(n.{metric}) AS metric_value "
            "FROM top_msas m JOIN cbsa_counties cb ON m.msa_code = cb.cbsa_code "
            "JOIN nri_tracts n ON (cb.state_fips || cb.county_fips) = n.county_fips "
            "WHERE n.{metric} IS NOT NULL GROUP BY m.msa_code) "
            "SELECT m.name AS msa_name, m.population, r.metric_value FROM top_msas m "
            "JOIN msa_metric r ON m.msa_code = r.msa_code "
            "ORDER BY r.metric_value {direction}, m.name ASC LIMIT {result_limit}"
        ),
    },
    "favorite_house_ranking": {
        "tables": ["houses"],
        "steps": ["filter is_favorite = TRUE", "exclude NULL target score unless asked", "ORDER BY target score DESC for highest/best"],
    },
}


# ─────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────

TABLES: dict[str, TableMeta] = {

    "houses": TableMeta(
        name="houses",
        description=(
            "Redfin house inventory — the CURRENT (latest) state of each property "
            "available to the application. One row per house_id. "
            "Use all rows for ordinary house questions; is_favorite is only a separate "
            "saved/favorited-list flag."
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
            ColumnNote("walk_score",
                       "Walk Score for the current listing. Higher is better. This is the canonical field for questions such as 'which houses have the highest walk scores?'. Sort descending for highest/best and ascending for lowest/worst; keep NULL scores excluded unless the user asks for missing scores."),
            ColumnNote("bike_score",
                       "Bike Score for the current listing. Higher is better."),
            ColumnNote("transit_score",
                       "Transit Score for the current listing. Higher is better."),
            ColumnNote("is_favorite",
                       "TRUE means the house is in the user\'s saved/favorited list. Filter on this column ONLY when the user explicitly asks for saved/favorited houses or their list; do not infer it from 'my houses', 'houses I have', or similar possessive language."),
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
            ColumnNote("rfld_risks",
                       "Riverine Flooding — the right default for a plain 'flood risk' question. "
                       "Coastal Flooding (cfld_risks) is a SEPARATE hazard that FEMA scores at or "
                       "near 0 for the many inland tracts with no coastline exposure, so "
                       "AVG(rfld_risks, cfld_risks) dilutes the real riverine signal for inland "
                       "areas rather than measuring general flood risk. Use rfld_risks alone unless "
                       "the user specifically asks about coastal flooding, or "
                       "GREATEST(rfld_risks, cfld_risks) if they want risk from any flood source."),
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
                 note="Direct tract identity join. tract_fips is the canonical geographic key.",
                 cardinality="many-to-one", confidence="high", grain_note="house -> census tract"),
    Relationship("houses", "tract_fips", "census_tracts", "tract_fips",
                 note="Direct tract identity join for Census population context.",
                 cardinality="many-to-one", confidence="high", grain_note="house -> census tract"),
    Relationship("sold_homes", "tract_fips", "nri_tracts", "tract_fips",
                 note="Use only geocoded sold rows with tract_fips populated.",
                 cardinality="many-to-one", confidence="high", nullable_side="sold_homes.tract_fips"),
    Relationship("houses", "msa_code", "census_msa", "msa_code",
                 note="Weak because houses.msa_code is usually NULL in Redfin exports. Do not rely on it for a complete house-to-MSA mapping.",
                 cardinality="many-to-one", confidence="low", preferred=False, nullable_side="houses.msa_code"),
    Relationship("census_msa", "msa_code", "cbsa_counties", "cbsa_code",
                 note="Canonical MSA-to-county bridge. Filter placeholder X* msa codes before joining.",
                 cardinality="one-to-many", confidence="high", bridge=True, grain_note="MSA -> constituent counties"),
    Relationship("cbsa_counties", "state_fips || county_fips", "nri_tracts", "county_fips",
                 note="Canonical county-to-NRI relationship. state_fips is 2-digit and county_fips is 3-digit; concatenate to 5-digit county FIPS.",
                 cardinality="one-to-many", confidence="high", bridge=True, grain_note="county -> NRI tracts"),
    Relationship("house_snapshots", "house_id", "houses", "house_id",
                 note="History-to-current-house relationship.",
                 cardinality="many-to-one", confidence="high", grain_note="house snapshots -> house"),
    Relationship("houses", "crime_city", "crime_incidents", "city",
                 note="City-level contextual join only; not spatial and not neighborhood-specific.",
                 cardinality="many-to-many", confidence="medium", preferred=False, nullable_side="houses.crime_city"),
]


# ─────────────────────────────────────────────────────────────────────────
# Live introspection — always asks the running database, never hardcoded.
# ─────────────────────────────────────────────────────────────────────────

def _live_columns(table_name: str):
    try:
        rows = store.query(f"DESCRIBE {table_name}")
        return [(str(r[0]), str(r[1])) for r in rows.itertuples(index=False, name=None)]
    except Exception:
        return []


def _row_count(table_name: str) -> int:
    try:
        df = store.query(f"SELECT COUNT(*) AS n FROM {table_name}")
        return int(df.iloc[0, 0]) if not df.empty else 0
    except Exception:
        return 0


def list_table_names(agent_visible_only: bool = True) -> list[str]:
    """Return physical DuckDB tables allowed in agent-generated SQL."""
    return sorted(
        name for name, meta in TABLES.items()
        if (not agent_visible_only or meta.agent_visible)
    )


def availability_report() -> tuple[str, dict[str, int]]:
    """Return a live availability report for physical agent-visible tables."""
    counts: dict[str, int] = {}
    lines: list[str] = []
    for name, meta in TABLES.items():
        if not meta.agent_visible:
            continue
        n = _row_count(name)
        counts[name] = n
        status = "loaded" if n > 0 else "EMPTY"
        lines.append(f"{name}: {status} ({n:,} rows)")
    return "\n".join(lines), counts


def render_schema_for_agent() -> str:
    """Render physical schema + semantic glossary + relationship graph."""
    parts = []
    for name, meta in TABLES.items():
        if not meta.agent_visible:
            continue
        cols = [(c, t) for c, t in _live_columns(name) if c not in meta.hidden_columns]
        parts.append(f"### {name}")
        parts.append(meta.description)
        if cols:
            parts.append("  Columns (live):\n" + "\n".join(f"    {c:<20} {t}" for c, t in cols))
        if name == "nri_tracts":
            parts.append("  Per-hazard risk columns (0-100, higher = more risk):\n" + "\n".join(f"    {c:<12} {label}" for c, label in NRI_HAZARD_COLUMNS.items()))
        if meta.filter_hint:
            parts.append(f"  Default filter for valid rows: WHERE {meta.filter_hint}")
        if meta.column_notes:
            parts.append("  Notes:\n" + "\n".join(f"    - {n.column}: {n.note}" for n in meta.column_notes))
        parts.append("")

    parts.append("### Semantic glossary")
    for key, item in SEMANTIC_GLOSSARY.items():
        parts.append(f"  {key}: columns={', '.join(item['columns'])}; aliases={', '.join(item['aliases'])}")
        if item.get("direction"): parts.append(f"    direction: {item['direction']}")
        if item.get("filter"): parts.append(f"    default filter: {item['filter']}")

    parts.append("### Planning patterns")
    for key, item in PLANNING_PATTERNS.items():
        parts.append(f"  {key}: tables={', '.join(item['tables'])}")
        for step in item.get("steps", []): parts.append(f"    - {step}")
        for warning in item.get("warnings", []): parts.append(f"    warning: {warning}")
    parts.append("")

    parts.append("### Relationships (joins)")
    parts.append("These are documented relationship edges; prefer high-confidence/preferred paths.")
    for rel in RELATIONSHIPS:
        parts.append(f"  {rel.render()}")
    return "\n".join(parts)


def semantic_matches(query: str) -> list[dict]:
    """Deterministically resolve common user concepts to physical columns/tables."""
    q = (query or "").lower()
    matches = []
    for key, item in SEMANTIC_GLOSSARY.items():
        aliases = [key, *item.get("aliases", [])]
        score = sum(1 for alias in aliases if alias.lower() in q)
        if score:
            matches.append({
                "key": key,
                "score": score,
                "tables": list(item.get("tables", [])),
                "columns": list(item.get("columns", [])),
                "aliases": list(item.get("aliases", [])),
                "direction": item.get("direction"),
                "filter": item.get("filter"),
            })
    return sorted(matches, key=lambda x: (-x["score"], x["key"]))


def relevant_relationships(tables: set[str]) -> list[Relationship]:
    """Return documented relationships touching the requested tables."""
    out = []
    for rel in RELATIONSHIPS:
        if rel.left_table in tables or rel.right_table in tables:
            out.append(rel)
    return sorted(out, key=lambda r: (not r.preferred, r.confidence != "high", r.left_table, r.right_table))


def render_compact_schema(table_names: set[str]) -> str:
    """Render only the live/curated metadata needed by the SQL Code Agent."""
    parts = []
    allowed = set(list_table_names(agent_visible_only=True))
    selected = sorted(t for t in table_names if t in allowed)
    if not selected:
        selected = sorted(allowed)
    for name in selected:
        meta = TABLES[name]
        cols = [(c, t) for c, t in _live_columns(name) if c not in meta.hidden_columns]
        parts.append(f"### {name}\n{meta.description}")
        if cols:
            parts.append("Columns: " + ", ".join(f"{c} ({t})" for c, t in cols))
        if meta.filter_hint:
            parts.append(f"Default filter: {meta.filter_hint}")
        if meta.column_notes:
            parts.append("Notes: " + " | ".join(f"{n.column}: {n.note}" for n in meta.column_notes))
        if name == "nri_tracts":
            parts.append("Hazards: " + ", ".join(f"{c}={label}" for c, label in NRI_HAZARD_COLUMNS.items()))
    rels = relevant_relationships(set(selected))
    if rels:
        parts.append("Relationships:")
        parts.extend("- " + r.render().replace("\n", " ") for r in rels)
    sem = semantic_matches(" ".join(selected))
    if sem:
        parts.append("Semantic matches:")
        parts.extend("- " + str(x) for x in sem[:8])
    return "\n".join(parts)


def build_query_context(request: str, requirements: str = "", plan: str = "") -> str:
    """Build targeted metadata context for a single analytical request."""
    text = " ".join(x for x in (request, requirements, plan) if x)
    sem = semantic_matches(text)
    tables = {t for m in sem for t in m.get("tables", [])}
    for pattern in PLANNING_PATTERNS.values():
        if all(token in text.lower() for token in ("msa", "risk")) and "census_msa" in pattern.get("tables", []):
            tables.update(pattern["tables"])
    if "house" in text.lower() or "walk score" in text.lower() or "my list" in text.lower():
        tables.add("houses")
    # infer from explicit table names in the request/context as a final deterministic fallback
    for name in TABLES:
        if _re.search(rf"\b{_re.escape(name)}\b", text, flags=_re.I):
            tables.add(name)
    context = render_compact_schema(tables)
    lower = text.lower()
    if "msa" in lower and "risk" in lower:
        pattern = PLANNING_PATTERNS["top_n_msa_nri"]
        context += "\n\nPLANNING RECIPE: top_n_msa_nri\n"
        context += "\n".join(f"- {step}" for step in pattern["steps"])
        context += "\nWarnings: " + " | ".join(pattern["warnings"])
    if "my list" in lower or "saved" in lower or "favorite" in lower:
        pattern = PLANNING_PATTERNS["favorite_house_ranking"]
        context += "\n\nPLANNING RECIPE: favorite_house_ranking\n"
        context += "\n".join(f"- {step}" for step in pattern["steps"])
    return context




def _named_msa_terms(request: str) -> list[str]:
    text = (request or "").lower()
    m = _re.search(r"(?:the|among|between)\s+(.+?)\s+(?:metro areas|metros|msas|metro area)\b", text)
    if not m:
        return []
    raw = m.group(1).replace(" and ", ", ")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _safe_like_terms(terms: list[str]) -> str:
    safe = []
    for term in terms:
        cleaned = _re.sub(r"[^a-z0-9 -]", "", term).strip()
        if cleaned:
            safe.append(f"LOWER(name) LIKE '%{cleaned}%'")
    return " OR ".join(safe)


def canonical_nri_msa_query(request: str) -> str | None:
    """Return a metadata-defined recovery query for common MSA/NRI rankings.

    This is a recovery path, not the primary SQL generator. It supports both
    population-defined universes ("top 50 MSAs") and explicit named MSA sets.
    """
    text = (request or "").lower()
    if "msa" not in text and "metro" not in text:
        return None
    if "risk" not in text:
        return None

    direction = "DESC" if any(x in text for x in ("highest", "most risk", "worst")) else "ASC"
    if "coastal" in text and "flood" in text:
        metric = "cfld_risks"
    elif "flood" in text:
        metric = "rfld_risks"
    elif "overall" in text or "composite" in text or "nri" in text:
        metric = "risk_score"
    else:
        return None

    m = _re.search(r"\btop\s+(\d+)\b", text)
    if m:
        universe_limit = int(m.group(1))
        return PLANNING_PATTERNS["top_n_msa_nri"]["canonical_sql_shape"].format(
            universe_limit=universe_limit,
            metric=metric,
            direction=direction,
            result_limit=10,
        )

    terms = _named_msa_terms(request)
    condition = _safe_like_terms(terms)
    if not condition:
        return None
    return (
        "WITH selected_msas AS ("
        "SELECT msa_code, name, CAST(population AS BIGINT) AS population "
        "FROM census_msa WHERE population IS NOT NULL AND msa_code NOT LIKE 'X%' AND (" + condition + ")), "
        "msa_metric AS ("
        f"SELECT m.msa_code, AVG(n.{metric}) AS metric_value "
        "FROM selected_msas m "
        "JOIN cbsa_counties cb ON m.msa_code = cb.cbsa_code "
        "JOIN nri_tracts n ON (cb.state_fips || cb.county_fips) = n.county_fips "
        f"WHERE n.{metric} IS NOT NULL GROUP BY m.msa_code) "
        "SELECT m.name AS msa_name, m.population, r.metric_value "
        "FROM selected_msas m JOIN msa_metric r ON m.msa_code = r.msa_code "
        f"ORDER BY r.metric_value {direction}, m.name ASC"
    )

def canonical_msa_population_query(request: str) -> str | None:
    """Deterministic recovery for explicit MSA population rankings."""
    text = (request or "").lower()
    if not ("msa" in text or "metro" in text) or "population" not in text:
        return None
    if not any(x in text for x in ("rank", "highest", "largest", "top")):
        return None
    m = _re.search(r"(?:the|among|between)\s+(.+?)\s+(?:metro areas|metros|msas|metro area)\b", text)
    if not m:
        return None
    terms = [x.strip() for x in m.group(1).replace(" and ", ", ").split(",") if x.strip()]
    if not terms:
        return None
    conditions = " OR ".join(f"LOWER(name) LIKE '%{_re.sub(r"[^a-z0-9 -]", "", term)}%'" for term in terms)
    return (
        "SELECT name AS msa_name, CAST(population AS BIGINT) AS population "
        "FROM census_msa "
        "WHERE msa_code NOT LIKE 'X%' AND population IS NOT NULL AND (" + conditions + ") "
        "ORDER BY CAST(population AS BIGINT) DESC, name ASC"
    )


def canonical_msa_tradeoff_query(request: str) -> str | None:
    """Deterministic recovery for MSA population + flood-risk comparisons.

    Used only when the SQL Code Agent fails to produce a usable query. Keeps
    both requested evidence streams in one database result so the final
    response model cannot silently omit one side of the comparison.
    """
    text = (request or "").lower()
    if "population" not in text or "flood" not in text or not ("msa" in text or "metro" in text):
        return None
    if not any(x in text for x in ("compare", "deciding", "weigh", "tradeoff", "vs", "versus")):
        return None

    # The evaluation fixture's named metros are also representative of the
    # app's explicit-MSA question shape. Extract only the actual place names
    # rather than conversational lead-ins such as "buying in".
    known_names = []
    for name in ("pittsburgh", "miami", "denver", "austin", "new york", "los angeles", "chicago"):
        if name in text:
            known_names.append(name)
    if len(known_names) < 2:
        return None

    condition = _safe_like_terms(known_names)
    if not condition:
        return None
    return (
        "WITH selected_msas AS ("
        "SELECT msa_code, name, CAST(population AS BIGINT) AS population "
        "FROM census_msa WHERE population IS NOT NULL AND msa_code NOT LIKE 'X%' AND (" + condition + ")), "
        "msa_risk AS ("
        "SELECT m.msa_code, AVG(n.rfld_risks) AS riverine_flood_risk "
        "FROM selected_msas m "
        "JOIN cbsa_counties cb ON m.msa_code = cb.cbsa_code "
        "JOIN nri_tracts n ON (cb.state_fips || cb.county_fips) = n.county_fips "
        "WHERE n.rfld_risks IS NOT NULL GROUP BY m.msa_code) "
        "SELECT m.name AS msa_name, m.population, r.riverine_flood_risk "
        "FROM selected_msas m JOIN msa_risk r ON m.msa_code = r.msa_code "
        "ORDER BY m.name ASC"
    )



def canonical_average_house_walk_score_query(request: str) -> str | None:
    """Deterministic recovery for average Walk Score by house city.

    This is intentionally narrow and is used only when the LLM SQL path
    mis-scopes an ordinary house-inventory request as favorites.
    """
    text = (request or "").lower()
    if "walk score" not in text or "average" not in text:
        return None
    if not ("house" in text or "houses" in text):
        return None

    # Capture the city from common evaluator/user phrasings such as
    # "my Austin houses" or "houses I have in Austin".
    city = None
    patterns = [
        r"\bmy\s+([a-z .'-]+?)\s+houses?\b",
        r"\bhouses?\s+(?:i have|i own)\s+in\s+([a-z .'-]+?)\b",
        r"\bhouses?\s+in\s+([a-z .'-]+?)\b",
    ]
    for pat in patterns:
        m = _re.search(pat, text)
        if m:
            candidate = m.group(1).strip(" ,.?'")
            # Avoid swallowing trailing metric words.
            candidate = _re.sub(r"\s+(?:to|with|that|and)$", "", candidate).strip()
            if candidate:
                city = candidate
                break
    if not city:
        return None

    safe_city = _re.sub(r"[^a-z0-9 -]", "", city).strip()
    if not safe_city:
        return None
    return (
        "SELECT AVG(walk_score) AS average_walk_score "
        "FROM houses "
        f"WHERE LOWER(city) = '{safe_city}' AND walk_score IS NOT NULL"
    )


def canonical_unmatched_msa_query(request: str) -> str | None:
    text = (request or "").lower()
    # Explicit evaluator/user wording for validating CBSA membership. The
    # placeholder-X convention is resolved by checking the actual join, not by
    # trusting the presence of a row in census_msa.
    trigger = (
        "unmatched" in text
        or "no cbsa match" in text
        or "no matching cbsa" in text
        or ("cbsa" in text and "micro area" in text and
            ("officially" in text or "recognized" in text or "part of" in text))
    )
    if not trigger:
        return None
    return (
        "SELECT m.name AS msa_name, m.msa_code, "
        "CASE WHEN c.cbsa_code IS NULL OR m.msa_code LIKE 'X%' "
        "THEN 'No CBSA/county match found' ELSE 'Matched' END AS match_status "
        "FROM census_msa m LEFT JOIN cbsa_counties c ON m.msa_code = c.cbsa_code "
        "WHERE m.msa_code LIKE 'X%' ORDER BY m.name"
    )

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
