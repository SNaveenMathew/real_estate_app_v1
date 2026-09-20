"""Built-in catalog definitions: the SEED for the unified catalog store.

This file is *not* read by the agents.  On first run (and on upgrade) ``db/catalog_store.sync_seed``
copies these definitions into the ``catalog_*`` tables of the DuckDB database; from then on the
store is the source of truth, and data sources added later (through the Data page) live in the
same tables.  You only need to edit this file to change what a fresh install starts with.
"""
from __future__ import annotations

from db.catalog_model import (
    ColumnNote, TableMeta, Relationship, EntityDomain,
    _concept, _op, COUNT, AVG, SUM, MEDIAN, MIN, MAX, RANK_DESC, RANK_ASC,
)


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

# Display grouping for the Data page's schema map (metadata, not routing).
SEED_DOMAINS = {
    "houses": "housing",
    "house_snapshots": "housing",
    "nri_tracts": "risk",
    "census_tracts": "geography",
    "census_msa": "geography",
    "cbsa_counties": "geography",
    "sold_homes": "sales",
    "crime_incidents": "safety",
    "bike_routes": "mobility",
    "geocode_cache": "system",
}
