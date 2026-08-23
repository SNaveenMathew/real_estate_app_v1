"""Declarative data-model contract for analytical planning.

The catalog is deliberately *not* a routing table.  It describes facts that are
true of the repository's physical model: tables, fields, grain, aliases,
operations, entity domains and join relationships.  The planner composes these
facts into a query plan; it does not contain domain-specific question branches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from collections import deque

import db.duckdb_store as store


@dataclass(frozen=True)
class ColumnNote:
    column: str
    note: str


@dataclass(frozen=True)
class TableMeta:
    name: str
    description: str
    setup_hint: str = ""
    filter_hint: str = ""
    column_notes: tuple[ColumnNote, ...] = ()
    hidden_columns: tuple[str, ...] = ()
    agent_visible: bool = True
    grain: str = ""
    default_filter: str = ""


@dataclass(frozen=True)
class Relationship:
    left_table: str
    left_expr: str
    right_table: str
    right_expr: str
    note: str = ""
    cardinality: str = ""
    confidence: str = "high"
    bridge: bool = False
    preferred: bool = True
    grain_effect: str = ""

    def involves(self, tables: set[str]) -> bool:
        return self.left_table in tables and self.right_table in tables

    def key(self) -> str:
        return f"{self.left_table}:{self.left_expr}={self.right_table}:{self.right_expr}"

    def render(self) -> str:
        attrs = [self.cardinality, f"confidence={self.confidence}", f"preferred={self.preferred}", f"bridge={self.bridge}"]
        extra = []
        if self.note:
            extra.append(self.note)
        if self.grain_effect:
            extra.append("Grain: " + self.grain_effect)
        return f"{self.left_table}.{self.left_expr} = {self.right_table}.{self.right_expr} [{' ; '.join(attrs)}]" + (" Note: " + " ".join(extra) if extra else "")


@dataclass(frozen=True)
class EntityDomain:
    name: str
    table: str
    column: str
    entity_type: str
    description: str
    display_column: str | None = None
    match_mode: str = "exact_or_prefix"
    preferred_for: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Physical model
# ---------------------------------------------------------------------------

NRI_HAZARD_COLUMNS = {
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

TABLES = {
    "houses": TableMeta(
        "houses", "Current Redfin inventory; one row per current house_id.",
        grain="one current row per house",
        hidden_columns=("raw_json",),
        column_notes=(
            ColumnNote("city", "Current listing city."),
            ColumnNote("state", "Current listing state."),
            ColumnNote("status", "Current Redfin listing status such as Active, Pending or Sold."),
            ColumnNote("price", "Current list price, not historical sold price."),
            ColumnNote("walk_score", "Current Walk Score; NULL means missing. Numeric summaries exclude NULL unless missingness is requested."),
            ColumnNote("bike_score", "Current Bike Score; NULL means missing."),
            ColumnNote("transit_score", "Current Transit Score; NULL means missing."),
            ColumnNote("tract_fips", "11-digit tract FIPS; reliable tract-level join key."),
            ColumnNote("msa_code", "Usually NULL in Redfin exports; not a complete house-to-MSA key."),
            ColumnNote("is_favorite", "Explicit saved/favorite flag. Do not infer it from ordinary 'my houses' wording."),
        ),
    ),
    "house_snapshots": TableMeta(
        "house_snapshots", "Historical observed listing/sale states for houses.",
        grain="one observed state per house/source/status/price",
        column_notes=(
            ColumnNote("house_id", "Links to houses.house_id."),
            ColumnNote("snapshot_date", "Historical observation date; may be NULL."),
            ColumnNote("source_type", "redfin or sold."),
            ColumnNote("price", "Historical list price or matched sold price depending on source_type."),
        ),
    ),
    "nri_tracts": TableMeta(
        "nri_tracts", "FEMA National Risk Index metrics at census-tract grain.",
        grain="one row per census tract",
        column_notes=(
            ColumnNote("tract_fips", "11-digit tract FIPS primary key."),
            ColumnNote("county_fips", "5-digit state+county FIPS used for CBSA county bridging."),
            ColumnNote("risk_score", "Composite NRI risk score; higher means more risk."),
            ColumnNote("rfld_risks", "Riverine Flooding risk score; separate from coastal flooding."),
            ColumnNote("cfld_risks", "Coastal Flooding risk score."),
            ColumnNote("hrcn_risks", "Hurricane risk score."),
            ColumnNote("wfir_risks", "Wildfire risk score."),
        ),
    ),
    "census_tracts": TableMeta(
        "census_tracts", "2020 Census tract population records.",
        grain="one row per census tract",
        column_notes=(ColumnNote("tract_fips", "11-digit tract FIPS primary key."), ColumnNote("population", "Census total population.")),
    ),
    "census_msa": TableMeta(
        "census_msa", "2020 Census population records for MSA/micropolitan statistical areas.",
        grain="one row per MSA/micropolitan area",
        column_notes=(
            ColumnNote("msa_code", "Real CBSA code when matched; X-prefixed placeholder when unresolved."),
            ColumnNote("name", "Human-readable MSA display name, typically '<city>, <state> Metro Area'."),
            ColumnNote("population", "Census population for the MSA/micropolitan area."),
        ),
    ),
    "cbsa_counties": TableMeta(
        "cbsa_counties", "CBSA delineation bridge from MSA/CBSA code to constituent counties.",
        grain="one row per CBSA-county membership",
        column_notes=(
            ColumnNote("cbsa_code", "Joins census_msa.msa_code."),
            ColumnNote("state_fips", "2-digit zero-padded state FIPS."),
            ColumnNote("county_fips", "3-digit zero-padded county FIPS; concatenate state_fips || county_fips to reach 5-digit county FIPS."),
        ),
    ),
    "sold_homes": TableMeta(
        "sold_homes", "County assessor sale records, including geocoded and ungeocoded transactions.",
        grain="one row per recorded sale",
        column_notes=(
            ColumnNote("city", "Sale city label."),
            ColumnNote("state", "Sale state."),
            ColumnNote("sold_price", "Recorded transaction amount."),
            ColumnNote("is_arms_length", "TRUE/NULL are eligible for the documented market-sale filter; FALSE is non-arm's-length."),
            ColumnNote("tract_fips", "11-digit tract FIPS when geocoded; otherwise NULL."),
            ColumnNote("geocode_status", "Geocoding state such as pending or success."),
        ),
    ),
    "crime_incidents": TableMeta("crime_incidents", "Standardized crime incidents.", grain="one row per incident"),
    "bike_routes": TableMeta("bike_routes", "BikePGH-style line features.", grain="one row per line feature"),
    "geocode_cache": TableMeta("geocode_cache", "Internal geocoding cache.", agent_visible=False),
}

RELATIONSHIPS = [
    Relationship("houses", "tract_fips", "nri_tracts", "tract_fips", "Direct house-to-NRI tract identity join.", "many-to-one", grain_effect="house -> tract"),
    Relationship("houses", "tract_fips", "census_tracts", "tract_fips", "Direct house-to-Census tract identity join.", "many-to-one", grain_effect="house -> tract"),
    Relationship("sold_homes", "tract_fips", "nri_tracts", "tract_fips", "Only sold rows with tract_fips populated can use this join.", "many-to-one", grain_effect="sale -> tract"),
    Relationship("census_msa", "msa_code", "cbsa_counties", "cbsa_code", "Canonical MSA-to-county bridge.", "one-to-many", bridge=True, grain_effect="MSA -> counties"),
    Relationship("cbsa_counties", "state_fips || county_fips", "nri_tracts", "county_fips", "County bridge into NRI tracts.", "one-to-many", bridge=True, grain_effect="county -> tracts"),
    Relationship("census_tracts", "tract_fips", "nri_tracts", "tract_fips", "Shared tract identity; permits pairing Census tract population with NRI tract metrics.", "one-to-one", grain_effect="tract <-> tract"),
    Relationship("cbsa_counties", "state_fips || county_fips", "census_tracts", "LEFT(tract_fips, 5)", "County-to-Census-tract geography bridge; tract FIPS begins with the 5-digit state+county FIPS.", "one-to-many", bridge=True, grain_effect="county -> Census tracts"),
    Relationship("house_snapshots", "house_id", "houses", "house_id", "Historical observation to current house.", "many-to-one", grain_effect="snapshot -> house"),
    Relationship("houses", "crime_city", "crime_incidents", "city", "City-level contextual relationship; not spatial.", "many-to-many", confidence="medium", preferred=False),
]

# ---------------------------------------------------------------------------
# Semantic contract.  Every concept is metadata; none is a routing branch.
# ---------------------------------------------------------------------------

def _concept(key, tables, aliases, description, *, columns=(), operations=(), filters=(), null_policy="", orderings=(), groupings=(), grain="", entity_types=(), rollup=False, rollup_spec=None, required_terms=(), excluded_terms=(), default_operation=None):
    return {
        "key": key, "tables": list(tables), "columns": list(columns), "aliases": list(aliases),
        "description": description, "operations": list(operations), "filters": list(filters),
        "null_policy": null_policy, "orderings": list(orderings), "groupings": list(groupings),
        "grain": grain, "entity_types": list(entity_types), "rollup": rollup, "rollup_spec": rollup_spec or {}, "required_terms": list(required_terms), "excluded_terms": list(excluded_terms), "default_operation": default_operation,
    }

def _op(op, aliases, expr, *, direction=None, group_by=None):
    d = {"op": op, "aliases": aliases, "expr": expr}
    if direction: d["direction"] = direction
    if group_by: d["group_by"] = group_by
    return d

COUNT = lambda expr, **kw: _op("count", ["how many", "number of", "count"], expr, **kw)
AVG = lambda expr, **kw: _op("avg", ["average", "avg", "mean"], expr, **kw)
SUM = lambda expr, **kw: _op("sum", ["total", "sum", "combined"], expr, **kw)
MEDIAN = lambda expr, **kw: _op("median", ["median"], expr, **kw)
MIN = lambda expr, **kw: _op("min", ["lowest", "minimum", "min", "worst"], expr, **kw)
MAX = lambda expr, **kw: _op("max", ["highest", "maximum", "max", "best"], expr, **kw)
RANK_DESC = lambda expr, **kw: _op("rank", ["rank", "ranking", "highest to lowest", "from highest to lowest", "largest", "highest"], expr, direction="DESC", **kw)
RANK_ASC = lambda expr, **kw: _op("rank", ["lowest to highest", "from lowest to highest", "smallest", "lowest"], expr, direction="ASC", **kw)

SEMANTIC_GLOSSARY = {
    "house_inventory": _concept("house_inventory", ["houses"], ["house", "houses", "home", "homes", "property", "properties", "house inventory", "home inventory"], "Current Redfin inventory.", columns=("houses.house_id", "houses.city"), operations=(COUNT("houses.house_id"), {"op":"distinct","aliases":["which cities","cities","list of cities"],"expr":"houses.city"}), grain="house", groupings=("houses.city",)),
    "house_list_price": _concept("house_list_price", ["houses"], ["list price", "listing price", "asking price", "current list price"], "Current listing price.", columns=("houses.price",), operations=(AVG("houses.price"), SUM("houses.price"), MEDIAN("houses.price"), RANK_DESC("AVG(houses.price)", group_by="houses.city")), grain="house"),
    "house_status_active": _concept("house_status_active", ["houses"], ["active listings", "active listing", "active houses", "currently active", "status active"], "Houses whose current Redfin status is Active.", columns=("houses.status",), operations=(COUNT("houses.house_id"),), filters=("houses.status = 'Active'",), grain="house"),
    "house_status_pending": _concept("house_status_pending", ["houses"], ["pending", "are pending", "is pending", "pending listings", "pending listing", "pending houses", "currently pending", "status pending"], "Houses whose current Redfin status is Pending.", columns=("houses.status",), operations=(COUNT("houses.house_id"),), filters=("houses.status = 'Pending'",), grain="house"),
    "house_status_sold": _concept("house_status_sold", ["houses"], ["sold listings", "sold listing", "sold houses", "status sold"], "Houses whose current Redfin status is Sold.", columns=("houses.status",), operations=(COUNT("houses.house_id"),), filters=("houses.status = 'Sold'",), grain="house"),
    "house_walk_score": _concept("house_walk_score", ["houses"], ["walk score", "walkability", "walkable", "non-missing walk score"], "Current house Walk Score.", columns=("houses.walk_score",), operations=(AVG("houses.walk_score"), MAX("houses.walk_score"), MIN("houses.walk_score"),), null_policy="exclude NULL", orderings=("houses.walk_score DESC", "houses.walk_score ASC"), grain="house"),
    "house_missing_walk": _concept("house_missing_walk", ["houses"], ["missing walk score", "missing Walk Score", "walk score missing", "missing a walk score"], "Count of houses whose Walk Score is NULL.", excluded_terms=("non-missing", "not missing", "available walk score"), columns=("houses.walk_score",), operations=(COUNT("houses.house_id"),), filters=("houses.walk_score IS NULL",), grain="house"),
    "house_bike_score": _concept("house_bike_score", ["houses"], ["bike score", "bikeability", "bikeable"], "Current house Bike Score.", columns=("houses.bike_score",), operations=(AVG("houses.bike_score"),), null_policy="exclude NULL", grain="house"),
    "house_transit_score": _concept("house_transit_score", ["houses"], ["transit score", "transit accessibility", "transit access"], "Current house Transit Score.", columns=("houses.transit_score",), operations=(AVG("houses.transit_score"),), null_policy="exclude NULL", grain="house"),
    "house_favorite": _concept("house_favorite", ["houses"], ["saved house", "saved houses", "favorite house", "favorite houses", "favorites", "my favorites", "saved list", "my saved list"], "Explicit saved/favorited-house scope. Ordinary 'my houses' is not this concept.", columns=("houses.is_favorite",), filters=("houses.is_favorite = TRUE",), grain="house"),
    "census_tract_population": _concept("census_tract_population", ["census_tracts"], ["tract population", "census tract population", "population of the tract", "census population"], "Census population at tract grain. A named city/metro can identify the corresponding tract only through the documented geography bridge.", columns=("census_tracts.population",), operations=(RANK_DESC("census_tracts.population", group_by="census_msa.name"),), grain="tract", entity_types=("tract_fips", "MSA"), groupings=("census_msa.name",), rollup=False),
    "msa_population": _concept("msa_population", ["census_msa"], ["MSA population", "metro population", "metro area population", "metro areas", "combined population", "combined MSA population", "largest metro", "largest MSA", "smallest metro", "smallest MSA", "population ranking"], "Census MSA population.", columns=("census_msa.population",), operations=(SUM("CAST(census_msa.population AS BIGINT)"), RANK_DESC("CAST(census_msa.population AS BIGINT)", group_by="census_msa.name")), grain="MSA", entity_types=("MSA",), groupings=("census_msa.name",), required_terms=("population",)),
    "msa_cbsa_membership": _concept(
        "msa_cbsa_membership",
        ["census_msa", "cbsa_counties"],
        [
            "recognized CBSA", "part of a recognized CBSA",
            "recognized core based statistical area", "core based statistical area",
            "officially part of a CBSA", "CBSA match", "CBSA affiliation", "CBSA membership"
        ],
        "Whether a census_msa row has a matching cbsa_counties row by msa_code = cbsa_code. "
        "An X-prefixed census_msa.msa_code is documented as unresolved.",
        columns=("census_msa.msa_code", "census_msa.name", "cbsa_counties.cbsa_code"),
        operations=({"op": "membership", "aliases": [], "expr": "census_msa.msa_code, cbsa_counties.cbsa_code"},),
        default_operation="membership",
        grain="MSA", entity_types=("MSA",), groupings=("census_msa.name",)
    ),
    "nri_overall_risk": _concept("nri_overall_risk", ["nri_tracts"], ["overall NRI risk", "overall risk", "NRI risk", "composite NRI risk", "composite risk"], "Composite FEMA NRI risk score at tract grain.", columns=("nri_tracts.risk_score",), operations=(AVG("nri_tracts.risk_score"), RANK_DESC("AVG(nri_tracts.risk_score)", group_by="census_msa.name")), null_policy="exclude NULL", grain="tract", entity_types=("MSA", "tract_fips"), rollup=True, rollup_spec={"source_grain":"tract","target_grain":"MSA","within_group":"AVG","across_groups":"AVG","group_key":"census_msa.name"}, groupings=("census_msa.name",), default_operation="avg"),
    "nri_riverine_flood": _concept("nri_riverine_flood", ["nri_tracts"], ["riverine flood risk", "riverine flooding", "flood risk", "flooding risk", "river flood", "riverine flood"], "FEMA NRI riverine flooding risk score.", columns=("nri_tracts.rfld_risks",), operations=(AVG("nri_tracts.rfld_risks"), RANK_DESC("AVG(nri_tracts.rfld_risks)", group_by="census_msa.name")), null_policy="exclude NULL", grain="tract", entity_types=("MSA", "tract_fips"), rollup=True, rollup_spec={"source_grain":"tract","target_grain":"MSA","within_group":"AVG","across_groups":"AVG","group_key":"census_msa.name"}, groupings=("census_msa.name",), default_operation="avg"),
    "nri_coastal_flood": _concept("nri_coastal_flood", ["nri_tracts"], ["coastal flood risk", "coastal flooding", "coastal flood"], "FEMA NRI coastal flooding risk score.", columns=("nri_tracts.cfld_risks",), operations=(AVG("nri_tracts.cfld_risks"), RANK_DESC("AVG(nri_tracts.cfld_risks)")), null_policy="exclude NULL", grain="tract", entity_types=("MSA", "tract_fips"), rollup=True),
}

for col, label in NRI_HAZARD_COLUMNS.items():
    if col in {"rfld_risks", "cfld_risks"}:
        continue
    aliases = [label.lower(), f"{label.lower()} risk"]
    if col == "hrcn_risks": aliases += ["hurricane", "hurricanes"]
    if col == "wfir_risks": aliases += ["wildfire", "wildfires"]
    key = "nri_" + col[:-6] + "_risk"
    SEMANTIC_GLOSSARY[key] = _concept(key, ["nri_tracts"], aliases, f"FEMA NRI {label} risk score.", columns=(f"nri_tracts.{col}",), operations=(AVG(f"nri_tracts.{col}"), RANK_DESC(f"AVG(nri_tracts.{col})", group_by="census_msa.name")), null_policy="exclude NULL", grain="tract", entity_types=("MSA", "tract_fips"), rollup=True, rollup_spec={"source_grain":"tract","target_grain":"MSA","within_group":"AVG","across_groups":"AVG","group_key":"census_msa.name"}, groupings=("census_msa.name",), default_operation="avg")

SEMANTIC_GLOSSARY.update({
    "sold_price": _concept("sold_price", ["sold_homes"], ["sold price", "sale price", "sales price", "sold-home records", "sold home records"], "Recorded sold-home transaction price.", columns=("sold_homes.sold_price",), operations=(AVG("sold_homes.sold_price"), MAX("sold_homes.sold_price"), RANK_DESC("sold_homes.sold_price", group_by="sold_homes.city")), filters=("(sold_homes.is_arms_length IS NULL OR sold_homes.is_arms_length = TRUE)", "sold_homes.sold_price > 1000"), grain="sale", groupings=("sold_homes.city",)),
    "arms_length_sale": _concept("arms_length_sale", ["sold_homes"], ["arm's length", "arms length", "arms-length", "market sale", "market-rate sale"], "Market-comparable sale scope.", columns=("sold_homes.is_arms_length",), filters=("(sold_homes.is_arms_length IS NULL OR sold_homes.is_arms_length = TRUE)", "sold_homes.sold_price > 1000"), grain="sale"),
    "history": _concept("history", ["house_snapshots"], ["price history", "listing history", "historical price", "price changes", "price cuts"], "Historical listing/sale observations.", columns=("house_snapshots.snapshot_date", "house_snapshots.price"), grain="snapshot"),
})

ENTITY_DOMAINS = (
    EntityDomain("house_city", "houses", "city", "city", "Current house city labels", match_mode="exact_or_prefix", preferred_for=("house_inventory", "house_list_price", "house_walk_score", "house_bike_score", "house_transit_score")),
    EntityDomain("sold_city", "sold_homes", "city", "city", "Sold-home city labels", match_mode="exact_or_prefix", preferred_for=("sold_price",)),
    EntityDomain("msa_name", "census_msa", "name", "MSA", "MSA display names", match_mode="prefix", preferred_for=("msa_population", "nri_overall_risk", "nri_riverine_flood")),
    EntityDomain("tract_fips", "census_tracts", "tract_fips", "tract_fips", "Census tract identifiers", match_mode="exact", preferred_for=("census_tract_population",)),
    EntityDomain("nri_tract_fips", "nri_tracts", "tract_fips", "tract_fips", "NRI tract identifiers", match_mode="exact", preferred_for=("nri_overall_risk", "nri_riverine_flood")),
    EntityDomain("sold_tract_fips", "sold_homes", "tract_fips", "tract_fips", "Sold-home tract identifiers when geocoded", match_mode="exact", preferred_for=("sold_price", "arms_length_sale")),
)

REQUEST_INTENTS = [
    {"name": "comparison", "aliases": ["compare", "comparison", "versus", "vs", "tradeoff"]},
    {"name": "ranking", "aliases": ["rank", "ranking", "top", "highest", "lowest", "largest", "smallest", "best", "worst"]},
    {"name": "aggregation", "aliases": ["how many", "count", "number of", "average", "avg", "mean", "median", "total", "sum"]},
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _alias_spans(text: str, alias: str):
    t = _normalize(text)
    a = _normalize(alias)
    if not a:
        return []
    pattern = r"(?<!\w)" + re.escape(a) + r"(?!\w)"
    return [(m.start(), m.end(), len(a.split())) for m in re.finditer(pattern, t)]


def _alias_matches(text: str, alias: str) -> bool:
    return bool(_alias_spans(text, alias))


def semantic_matches(query: str) -> list[dict]:
    text = _normalize(query)
    hits = []
    for key, item in SEMANTIC_GLOSSARY.items():
        matches = []
        for alias in item.get("aliases", []):
            matches.extend((len(alias.split()), start, end, alias) for start, end, _ in _alias_spans(text, alias))
        if not matches:
            continue
        required_terms = item.get("required_terms", [])
        if required_terms and not all(_alias_matches(text, term) for term in required_terms):
            continue
        excluded_terms = item.get("excluded_terms", [])
        if any(_alias_matches(text, term) for term in excluded_terms):
            continue
        specificity = max(m[0] for m in matches)
        hits.append((specificity, len(matches), key, item))
    hits.sort(key=lambda x: (-x[0], -x[1], x[2]))
    # Suppress generic inventory if a more-specific sale/history concept is matched.
    specific_keys = {h[2] for h in hits if h[0] >= 2 or h[2] in {"sold_price", "arms_length_sale", "history", "nri_overall_risk", "nri_riverine_flood"}}
    out = []
    for spec, count, key, item in hits:
        if key == "house_inventory" and any(k in specific_keys for k in {"sold_price", "arms_length_sale", "history"}):
            continue
        row = dict(item)
        row["key"] = key
        row["match_score"] = spec * 10 + count
        out.append(row)
    return out


def match_request_intents(query: str) -> list[dict]:
    t = _normalize(query)
    hits = []
    for item in REQUEST_INTENTS:
        score = max((len(a.split()) for a in item["aliases"] if _alias_matches(t, a)), default=0)
        if score:
            hits.append((score, item))
    hits.sort(key=lambda x: -x[0])
    return [i for _, i in hits]


def tables_mentioned_in_text(query: str) -> list[str]:
    t = _normalize(query)
    return [name for name in TABLES if _alias_matches(t, name.replace("_", " "))]


def list_table_names(agent_visible_only: bool = True) -> list[str]:
    return sorted(n for n, m in TABLES.items() if not agent_visible_only or m.agent_visible)


def _live_columns(table_name: str):
    try:
        df = store.query(f"DESCRIBE {table_name}")
        return [(str(r[0]), str(r[1])) for r in df.itertuples(index=False, name=None)]
    except Exception:
        return []


def _row_count(table_name: str) -> int:
    try:
        df = store.query(f"SELECT COUNT(*) AS n FROM {table_name}")
        return int(df.iloc[0, 0]) if not df.empty else 0
    except Exception:
        return 0


def availability_report():
    counts = {t: _row_count(t) for t in list_table_names()}
    lines = ["Live data availability:"] + [f"- {t}: {n} rows" for t, n in counts.items()]
    return "\n".join(lines), counts


def _fetch_values(table: str, column: str, limit: int = 5000):
    try:
        df = store.query(f"SELECT DISTINCT {column} AS value FROM {table} WHERE {column} IS NOT NULL LIMIT {int(limit)}")
        return [str(v) for v in df["value"].tolist()]
    except Exception:
        return []


def resolve_request_entities(query: str, candidate_tables: set[str] | None = None) -> list[dict]:
    t = _normalize(query)
    domains = [d for d in ENTITY_DOMAINS if not candidate_tables or d.table in candidate_tables]
    results = []
    # Tract FIPS are unambiguous literals and should be resolved first.
    for d in domains:
        if d.entity_type == "tract_fips":
            for fips in re.findall(r"\b\d{11}\b", query):
                values = _fetch_values(d.table, d.column)
                if fips in values:
                    results.append({"entity_type": d.entity_type, "table": d.table, "column": d.column, "value": fips, "domain": d.name, "score": 100})
    # Prefer MSA domain when the request contains MSA/metro language or an MSA concept matched.
    msa_context = any(_alias_matches(t, a) for a in ["msa", "msas", "metro", "metro area", "metropolitan area", "metro areas"])
    matched_concepts = semantic_matches(query)
    if any("MSA" in c.get("entity_types", []) for c in matched_concepts):
        msa_context = True
    for d in domains:
        values = _fetch_values(d.table, d.column)
        for value in values:
            nv = _normalize(value)
            if d.match_mode == "exact_or_prefix":
                if _alias_matches(t, nv):
                    results.append({"entity_type": d.entity_type, "table": d.table, "column": d.column, "value": value, "domain": d.name, "score": 80})
            elif d.match_mode == "prefix":
                # Stored display value may be 'Pittsburgh, PA Metro Area'; a query token 'Pittsburgh' matches the leading label.
                raw_prefix = value.split(",", 1)[0].strip()
                prefix = _normalize(raw_prefix)
                if prefix and (_alias_matches(t, prefix) or _alias_matches(t, nv)):
                    score = 95 if msa_context else 55
                    results.append({"entity_type": d.entity_type, "table": d.table, "column": d.column, "value": value, "domain": d.name, "score": score})
    # Keep the best domain for each normalized value/entity type.
    best = {}
    for e in results:
        k = (e["entity_type"], e["table"], _normalize(e["value"]))
        if k not in best or e["score"] > best[k]["score"]:
            best[k] = e
    return sorted(best.values(), key=lambda e: (-e["score"], e["entity_type"], e["value"]))


def _entity_predicate(entity: dict) -> str:
    value = entity["value"].replace("'", "''")
    return f"{entity['table']}.{entity['column']} = '{value}'"


def relationship_path(required_tables: set[str]) -> list[Relationship]:
    """Return the minimal high-quality relationship forest connecting required tables.

    Intermediate bridge tables are allowed; they come from the declarative
    RELATIONSHIPS graph and are never selected from user-language rules.
    """
    required = set(required_tables)
    if len(required) <= 1:
        return []

    graph = {}
    for rel in RELATIONSHIPS:
        weight = (0 if rel.confidence == "high" else 10) + (0 if rel.preferred else 5) + (0 if rel.bridge else 1)
        graph.setdefault(rel.left_table, []).append((rel.right_table, rel, weight))
        graph.setdefault(rel.right_table, []).append((rel.left_table, rel, weight))

    def shortest_to_connected(source, connected):
        import heapq
        heap = [(0, source, [])]
        seen = {source: 0}
        while heap:
            cost, node, path = heapq.heappop(heap)
            if node in connected:
                return path, cost
            for nxt, rel, weight in graph.get(node, []):
                nc = cost + weight
                if nc < seen.get(nxt, float("inf")):
                    seen[nxt] = nc
                    heapq.heappush(heap, (nc, nxt, path + [rel]))
        return None, float("inf")

    seed = sorted(required)[0]
    connected = {seed}
    chosen = []
    while not required.issubset(connected):
        candidates = []
        for source in sorted(required - connected):
            path, cost = shortest_to_connected(source, connected)
            if path:
                candidates.append((cost, source, path))
        if not candidates:
            break
        _, _, path = min(candidates, key=lambda x: (x[0], x[1]))
        for rel in path:
            chosen.append(rel)
            connected.add(rel.left_table)
            connected.add(rel.right_table)

    # Preserve order but remove duplicate relationship edges.
    out = []
    seen = set()
    for rel in chosen:
        k = rel.key()
        if k not in seen:
            seen.add(k)
            out.append(rel)
    return out


def expand_required_tables(tables: set[str]) -> list[str]:
    # Connect the selected tables using the minimal relationship forest.
    path = relationship_path(tables)
    out = set(tables)
    for rel in path:
        out.add(rel.left_table); out.add(rel.right_table)
    return sorted(out)


def relationships_for_tables(tables: set[str]) -> list[Relationship]:
    return relationship_path(tables)


def _concept_requires_rollup(concepts: list[dict], entity_tables: set[str]) -> bool:
    return bool(entity_tables) and any(c.get("rollup") for c in concepts)


def build_query_context(request: str, requirements: str = "", plan: str = "", focused: bool = False) -> str:
    concepts = semantic_matches(request)
    if focused and plan:
        selected = set()
        marker = re.search(r"semantic_keys:\s*([^\n]+)", plan)
        if marker:
            selected.update(x.strip() for x in marker.group(1).split(",") if x.strip() not in {"none", "null"})
        if selected:
            concepts = [c for c in concepts if c.get("key") in selected or c.get("key") == "msa_cbsa_membership"]
    target_tables = {t for c in concepts for t in c.get("tables", [])}
    target_tables.update(tables_mentioned_in_text(request))
    resolved = resolve_request_entities(request, None)
    msa_entities = [e for e in resolved if e["entity_type"] == "MSA"]
    tract_entities = [e for e in resolved if e["entity_type"] == "tract_fips"]
    # Entity domain + semantic target determines the geography without hardcoded query branches.
    if msa_entities and _concept_requires_rollup(concepts, {e["table"] for e in msa_entities}):
        target_tables.add("census_msa")
        if any(c.get("rollup") for c in concepts):
            target_tables.add("cbsa_counties")
    if tract_entities:
        target_tables.add("census_tracts")
        if any(c.get("rollup") for c in concepts):
            target_tables.add("nri_tracts")
    target_tables = set(expand_required_tables(target_tables))

    parts = ["TARGETED DATA MODEL"]
    for table in sorted(target_tables):
        meta = TABLES[table]
        live = _live_columns(table)
        cols = [c for c, _ in live if c not in meta.hidden_columns]
        parts.append(f"TABLE {table}: {meta.description}; grain={meta.grain}; columns={', '.join(cols)}")
        for note in meta.column_notes:
            if note.column in cols:
                parts.append(f"  {table}.{note.column}: {note.note}")
    rels = relationships_for_tables(target_tables)
    if rels:
        parts.append("RELATIONSHIPS")
        parts.extend(f"- {r.render()}" for r in rels)
    if concepts:
        parts.append("SEMANTIC CONCEPTS")
        for c in concepts:
            parts.append(f"- {c['key']}: {c['description']}; columns={c['columns']}; operations={c['operations']}; filters={c['filters']}; grain={c['grain']}; entity_types={c['entity_types']}; rollup={c['rollup']}; rollup_spec={c.get('rollup_spec', {})}")
    if resolved:
        parts.append("RESOLVED LIVE ENTITY VALUES")
        for e in resolved[:20]:
            parts.append(f"- {e['entity_type']}: {e['table']}.{e['column']} = {e['value']}")
    return "\n".join(parts)


def render_schema_for_agent() -> str:
    lines = ["DATABASE MODEL"]
    for table in list_table_names():
        m = TABLES[table]
        cols = ", ".join(c for c, _ in _live_columns(table) if c not in m.hidden_columns)
        lines.append(f"TABLE {table}: {m.description}; grain={m.grain}; columns={cols}")
    lines.append("RELATIONSHIPS")
    lines.extend(f"- {r.render()}" for r in RELATIONSHIPS if r.preferred)
    return "\n".join(lines)


def diagnose_empty_or_error(sql: str) -> str:
    refs = set(re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)", sql, re.I))
    empty = [t for t in refs if t in TABLES and _row_count(t) == 0]
    if empty:
        return "EMPTY TABLES: " + ", ".join(sorted(empty))
    rels = [r.render() for r in RELATIONSHIPS if r.left_table in refs or r.right_table in refs]
    return "Potential relationship context:\n" + "\n".join(rels[:10]) if rels else ""
