"""Built-in catalog definitions: the SEED for the unified catalog store.

This file is *not* read by the agents.  On first run (and on upgrade) ``db/catalog_store.sync_seed``
copies these definitions into the ``catalog_*`` tables of the DuckDB database; from then on the
store is the source of truth, and data sources added later (through the Data page) live in the
same tables.  You only need to edit this file to change what a fresh install starts with.
"""
from __future__ import annotations

from db.catalog_model import (
    ColumnNote, TableMeta, Relationship, EntityDomain,
    _concept, _op, COUNT, AVG, SUM, MEDIAN, MIN, MAX, RANK_DESC, RANK_ASC, LOOKUP,
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
    "zhvi": TableMeta(
        "zhvi", "Zillow Home Value Index (ZHVI) — smoothed, seasonally-adjusted typical home value, by region and month.",
        grain="one row per region per month",
        column_notes=(
            ColumnNote("region_id", "Zillow's own region identifier; not documented as unique across region_type values, so treat (region_id, region_type) as the region key."),
            ColumnNote("region_type", "Geography level of this row: country, state, metro, county, city, zip, or neighborhood — a single query may mix levels unless filtered."),
            ColumnNote("region_name", "Display name. For region_type='zip' this is a clean 5-digit zip (normalized at load time — see services/zillow_sources.py::normalize_zip5 — so it always equality-joins cleanly against houses.zip, which is normalized the same way). For metro rows this is Zillow's short 'City, ST' form, e.g. 'New York, NY'."),
            ColumnNote("home_value", "ZHVI dollar value for that region and month; NULL months are dropped, not zero."),
            ColumnNote("metro", "Containing metro area name; only populated for zip/neighborhood/city-level rows."),
            ColumnNote("msa_code", "CBSA code, resolved at load time (services/data_loader.py::_compute_zillow_msa_codes) from region_name (metro rows) or the metro column (zip/neighborhood/city rows), reusing the same CBSA-name matcher census_msa is built from. NULL for state/country rows or an unmatched metro name. Join houses.msa_code = zhvi.msa_code (with region_type='metro') for a metro-level fallback when a house's own zip has no zip-level row."),
        ),
    ),
    "market_heat_index": TableMeta(
        "market_heat_index", "Zillow Market Heat Index — buyer/seller market temperature (~0-100+), by region and month.",
        grain="one row per region per month",
        column_notes=(
            ColumnNote("region_id", "Zillow's own region identifier; not documented as unique across region_type values, so treat (region_id, region_type) as the region key."),
            ColumnNote("region_type", "Geography level of this row: country, state, metro, county, city, zip, or neighborhood."),
            ColumnNote("region_name", "For region_type='zip' this is a clean 5-digit zip, normalized the same way as houses.zip (see zhvi.region_name note)."),
            ColumnNote("heat_index", "Higher values indicate a hotter, more seller-favorable market; lower values a more buyer-favorable one. Not strictly capped at 100."),
            ColumnNote("msa_code", "CBSA code; same resolution and join pattern as zhvi.msa_code (see that note)."),
        ),
    ),
    "geocode_cache": TableMeta("geocode_cache", "Internal geocoding cache.", agent_visible=False),
    "data_source_log": TableMeta(
        "data_source_log", "Tracks when each built-in data source was last refreshed from the Data page.",
        agent_visible=False, grain="one row per data source",
    ),
    "house_commute": TableMeta(
        "house_commute", "Commute time and distance from each house to the current work location.",
        setup_hint="Set a work location on the Commute tab to populate commute estimates for every house.",
        hidden_columns=("work_key",), grain="one row per house, for the current work location",
    ),
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
    Relationship("house_commute", "house_id", "houses", "house_id", "Commute estimate for the current work location, one row per house.", "many-to-one", grain_effect="house -> house (current work location)"),
    # ZHVI / Market Heat Index — see zhvi.msa_code / zhvi.region_name column notes above for how
    # each side is normalized so these are plain equality joins, not fuzzy string matching.
    # Two tiers per dataset (zip and metro); prefer zip when it has coverage, fall back to metro
    # (and further to a plain houses.state = zhvi.state_name equality, which needs no relationship
    # entry since both sides are already bare 2-letter codes). db.duckdb_store.get_zhvi_price_estimate
    # implements this same 3-tier fallback in code for House Chat's price-estimate tool; these
    # relationships are what let the general schema-driven agent do the equivalent in SQL.
    Relationship("houses", "zip", "zhvi", "region_name", "House ZIP to zip-level ZHVI (filter zhvi.region_type = 'zip'). Both sides are normalized to a clean 5-digit zip at load time.", "one-to-many", grain_effect="house -> zip-month series"),
    Relationship("houses", "msa_code", "zhvi", "msa_code", "House metro to metro-level ZHVI (filter zhvi.region_type = 'metro'), for when the house's own zip has no zip-level ZHVI row.", "one-to-many", grain_effect="house -> metro-month series"),
    Relationship("census_msa", "msa_code", "zhvi", "msa_code", "MSA to metro-level ZHVI (filter zhvi.region_type = 'metro').", "one-to-many", grain_effect="MSA -> metro-month series"),
    Relationship("houses", "zip", "market_heat_index", "region_name", "House ZIP to zip-level Market Heat Index (filter market_heat_index.region_type = 'zip').", "one-to-many", grain_effect="house -> zip-month series"),
    Relationship("houses", "msa_code", "market_heat_index", "msa_code", "House metro to metro-level Market Heat Index (filter market_heat_index.region_type = 'metro').", "one-to-many", grain_effect="house -> metro-month series"),
    Relationship("census_msa", "msa_code", "market_heat_index", "msa_code", "MSA to metro-level Market Heat Index (filter market_heat_index.region_type = 'metro').", "one-to-many", grain_effect="MSA -> metro-month series"),
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
    "msa_population": _concept("msa_population", ["census_msa"], ["MSA population", "metro population", "metro area population", "metro areas", "combined population", "combined MSA population", "largest metro", "largest MSA", "smallest metro", "smallest MSA", "population ranking", "rank MSAs", "top MSAs", "largest MSAs", "smallest MSAs", "MSAs by population", "all MSAs", "metros", "metropolitan areas", "metropolitan statistical areas", "micropolitan areas"], "Census MSA population.", columns=("census_msa.population",), operations=(SUM("CAST(census_msa.population AS BIGINT)"), RANK_DESC("CAST(census_msa.population AS BIGINT)", group_by="census_msa.name")), grain="MSA", entity_types=("MSA",), groupings=("census_msa.name",), required_terms=("population",),
        derived_support={"provider": "msa_geometry", "measures": ["population"]}),
    "census_tract_population_density": _concept(
        "census_tract_population_density",
        ["census_msa", "cbsa_counties", "census_tracts"],
        ["population density", "total population density", "people per square mile",
         "population per square mile", "population per sq mi", "population density by metro",
         "population density by MSA", "pop density", "density of population", "densely populated",
         "how dense", "how densely populated", "most densely populated", "least densely populated",
         "people per square kilometer", "persons per square mile", "residents per square mile",
         "inhabitants per square mile", "people per sq km", "population per square kilometer",
         "population per sq km"],
        "Population density for a named MSA is total Census tract population divided by the sum of its tract polygon areas in square miles. Use the documented MSA-to-county-to-tract relationships and do not average tract-level density values.",
        columns=("census_tracts.population",),
        operations=(SUM("census_tracts.population"),),
        groupings=("census_msa.name",),
        grain="tract",
        entity_types=("MSA",),
        default_operation="sum",
        overrides=["census_land_area", "census_place_population"],     # their phrases sit inside this one's
        derived={"provider": "msa_geometry", "measures": ["population", "land_area", "density"]},
    ),
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
    "zhvi": _concept("zhvi", ["zhvi"], ["zhvi", "home value index", "zillow home value index", "home value trend", "typical home value"], "Zillow Home Value Index (ZHVI): smoothed, seasonally-adjusted typical home value by region and month, at whichever geography level (zip/metro/state/etc.) is loaded — filter region_type for a single level.", columns=("zhvi.home_value",), operations=(AVG("zhvi.home_value"), MEDIAN("zhvi.home_value"), MIN("zhvi.home_value"), MAX("zhvi.home_value"), RANK_DESC("zhvi.home_value"), RANK_ASC("zhvi.home_value")), grain="region-month"),
    "market_heat_index": _concept("market_heat_index", ["market_heat_index"], ["market heat index", "zillow heat index", "buyer's market", "seller's market", "market temperature"], "Zillow Market Heat Index: buyer/seller market temperature (~0-100+) by region and month; higher is more seller-favorable. Mixed geography levels — filter region_type for a single level.", columns=("market_heat_index.heat_index",), operations=(AVG("market_heat_index.heat_index"), MEDIAN("market_heat_index.heat_index"), MIN("market_heat_index.heat_index"), MAX("market_heat_index.heat_index"), RANK_DESC("market_heat_index.heat_index"), RANK_ASC("market_heat_index.heat_index")), grain="region-month"),
})

# Commute concepts. Each mode phrase (e.g. "bike commute") textually contains the generic
# "commute" alias that house_commute_drive matches, so — per the `overrides` mechanism in
# db/schema_catalog.py's semantic_matches() — each of the other four declares drive as
# overridden: a query naming a specific mode is never *also* read as the generic (drive)
# commute concept. "shortest"/"longest" are given directly as rank-operation aliases since
# the built-in RANK_ASC/RANK_DESC alias lists don't include them.
SEMANTIC_GLOSSARY.update({
    "house_commute_drive": _concept(
        "house_commute_drive", ["house_commute"],
        ["commute", "commute time", "commute time to work", "driving commute", "drive to work",
         "driving to work", "drive time to work", "time to work"],
        "Free-flow driving commute time to the current work location.",
        columns=("house_commute.drive_min", "house_commute.drive_miles"),
        operations=(
            AVG("house_commute.drive_min"), MEDIAN("house_commute.drive_min"),
            _op("rank", ["shortest commute", "quickest commute", "fastest commute", "best commute"],
                "house_commute.drive_min", direction="ASC", group_by="houses.address"),
            _op("rank", ["longest commute", "slowest commute", "worst commute"],
                "house_commute.drive_min", direction="DESC", group_by="houses.address"),
        ),
        null_policy="exclude NULL", grain="house",
    ),
    "house_commute_bike": _concept(
        "house_commute_bike", ["house_commute"],
        ["bike commute", "biking commute", "bike to work", "biking to work",
         "cycling commute", "cycling to work", "bike time to work"],
        "Free-flow biking commute time to the current work location.",
        columns=("house_commute.bike_min", "house_commute.bike_miles"),
        operations=(
            AVG("house_commute.bike_min"), MEDIAN("house_commute.bike_min"),
            _op("rank", ["shortest bike commute", "quickest bike commute", "fastest bike commute"],
                "house_commute.bike_min", direction="ASC", group_by="houses.address"),
            _op("rank", ["longest bike commute", "slowest bike commute"],
                "house_commute.bike_min", direction="DESC", group_by="houses.address"),
        ),
        null_policy="exclude NULL", grain="house",
    ),
    "house_commute_walk": _concept(
        "house_commute_walk", ["house_commute"],
        ["walk commute", "walking commute", "walk to work", "walking to work", "walk time to work"],
        "Free-flow walking commute time to the current work location.",
        columns=("house_commute.walk_min", "house_commute.walk_miles"),
        operations=(
            AVG("house_commute.walk_min"), MEDIAN("house_commute.walk_min"),
            _op("rank", ["shortest walk commute", "quickest walk commute"],
                "house_commute.walk_min", direction="ASC", group_by="houses.address"),
            _op("rank", ["longest walk commute", "slowest walk commute"],
                "house_commute.walk_min", direction="DESC", group_by="houses.address"),
        ),
        null_policy="exclude NULL", grain="house",
    ),
    "house_commute_transit": _concept(
        "house_commute_transit", ["house_commute"],
        ["transit commute", "public transit commute", "transit to work", "public transit to work",
         "bus commute", "train commute"],
        "Public-transit commute time to the current work location (requires a self-hosted trip planner).",
        columns=("house_commute.transit_min", "house_commute.transit_transfers"),
        operations=(
            AVG("house_commute.transit_min"), MEDIAN("house_commute.transit_min"),
            _op("rank", ["shortest transit commute", "quickest transit commute"],
                "house_commute.transit_min", direction="ASC", group_by="houses.address"),
            _op("rank", ["longest transit commute", "slowest transit commute"],
                "house_commute.transit_min", direction="DESC", group_by="houses.address"),
        ),
        null_policy="exclude NULL", grain="house",
    ),
    "house_commute_distance": _concept(
        "house_commute_distance", ["house_commute"],
        ["commute distance", "distance to work", "how far to work", "miles to work",
         "driving distance to work"],
        "Commute distance in miles (driving distance to the current work location).",
        columns=("house_commute.drive_miles", "house_commute.straight_line_miles"),
        operations=(AVG("house_commute.drive_miles"), MAX("house_commute.drive_miles"),
                    MIN("house_commute.drive_miles")),
        null_policy="exclude NULL", grain="house",
    ),
})
for _mode_key in ("house_commute_bike", "house_commute_walk", "house_commute_transit", "house_commute_distance"):
    SEMANTIC_GLOSSARY[_mode_key]["overrides"] = ["house_commute_drive"]

# ---------------------------------------------------------------------------
# Census place questions ("population of Denver", "land area of Pittsburgh", "which counties make up ...").
#
# Design rules these concepts follow (see AGENT_ARCHITECTURE.md §11.3 "Concept knobs" and §11.8 "Adding a dataset"):
#   * Aliases are SPECIFIC phrases, never bare generic words.  onboarding (services/dataset_onboarding.py) refuses to
#     let an uploaded dataset claim an alias a built-in concept already owns, so a generic alias here would make that
#     word undiscoverable for every future dataset.
#   * ``requires_entity`` keeps natural phrasing ("population of ...") from capturing questions that merely contain the
#     words: the concept applies only when the request names a metro/place that really exists in the data.
#   * ``excluded_terms`` hands house/tract/listing questions back to the concepts that own them.
#   * ``derived`` / ``derived_support`` route polygon-derived measures to a registered provider, not to SQL.
#   * ``scope_guard=False``: these concepts add no new filterable column names to the unplanned-filter guard.
# ---------------------------------------------------------------------------
_HOUSE_WORDS = ("house", "houses", "home", "homes", "listing", "listings", "property", "properties", "sold")
_TRACT_WORDS = ("tract", "tracts", "census tract", "fips", "block group")
_DENSITY_WORDS = ("density", "per square mile", "per square kilometer", "per sq mi", "per sq km")
_METRO_GEOMETRY = {"provider": "msa_geometry"}

SEMANTIC_GLOSSARY.update({
    "census_place_population": _concept(
        "census_place_population", ["census_msa"],
        ["population of", "population in", "populations of", "s population", "how many people live in",
         "how many people are in", "how many people reside in", "how many residents", "number of residents",
         "number of people living in", "residents of", "inhabitants of", "people live in", "people living in",
         "how populous", "how many live in"],
        "Total Census population of a named metropolitan/micropolitan statistical area. A city name resolves to the metro "
        "area that contains it, so the figure describes the whole metro area, not the city limits.",
        columns=("census_msa.name", "census_msa.population"),
        operations=(LOOKUP("census_msa.name, census_msa.population"),),
        grain="MSA", entity_types=("MSA",), groupings=("census_msa.name",),
        excluded_terms=_HOUSE_WORDS + _TRACT_WORDS + _DENSITY_WORDS,
        requires_entity=["MSA"], scope_guard=False,
        derived_support=dict(_METRO_GEOMETRY, measures=["population"]),
    ),
    "census_land_area": _concept(
        "census_land_area", ["census_msa", "cbsa_counties", "census_tracts"],
        ["land area", "land areas", "total land area", "square miles", "square mile", "sq mi", "sq miles", "sq mile",
         "square kilometers", "square kilometres", "square kilometer", "square kilometre", "square km", "sq km",
         "bigger in area", "larger in area", "biggest in area", "largest in area", "smaller in area",
         "smallest in area", "area comparison", "geographic area"],
        "Land area of a named metropolitan/micropolitan statistical area: the summed areas of the census-tract boundary "
        "polygons of its member counties, in square miles (or square kilometers when asked). It is NOT a stored column; the "
        "application computes it. A city name resolves to the metro area that contains it, so the figure describes the whole "
        "metro area, not the city limits.",
        grain="tract", entity_types=("MSA",), groupings=("census_msa.name",),
        excluded_terms=_HOUSE_WORDS + ("lot", "lots", "square feet", "square foot", "sqft", "sq ft", "acre", "acres"),
        scope_guard=False, derived=dict(_METRO_GEOMETRY, measures=["land_area"]),
    ),
    "census_profile": _concept(
        "census_profile", ["census_msa", "cbsa_counties", "census_tracts"],
        ["census data", "census profile", "census statistics", "census facts", "census numbers", "census information",
         "quick facts", "key facts"],
        "Headline census figures for a named metro area: population, land area and population density.",
        grain="tract", entity_types=("MSA",), groupings=("census_msa.name",),
        excluded_terms=_HOUSE_WORDS + _TRACT_WORDS,
        requires_entity=["MSA"], scope_guard=False,
        derived=dict(_METRO_GEOMETRY, measures=["population", "land_area", "density"]),
    ),
    "msa_counties": _concept(
        "msa_counties", ["census_msa", "cbsa_counties"],
        ["counties in", "counties are in", "counties make up", "counties does", "which counties", "member counties",
         "constituent counties", "list of counties", "county list", "counties comprise", "counties belong"],
        "The counties that make up a named metropolitan/micropolitan statistical area (CBSA delineation).",
        columns=("cbsa_counties.county_name", "cbsa_counties.state_name"),
        operations=(LOOKUP("cbsa_counties.county_name, cbsa_counties.state_name"),),
        grain="county", entity_types=("MSA",), groupings=("census_msa.name",),
        excluded_terms=_HOUSE_WORDS + _TRACT_WORDS,
        requires_entity=["MSA"], scope_guard=False,
    ),
    "census_unloaded_topics": _concept(
        "census_unloaded_topics", [],
        ["median household income", "household income", "median income", "per capita income", "average income",
         "income level", "income levels", "income distribution", "poverty", "poverty rate", "unemployment",
         "unemployment rate", "median age", "average age", "age distribution", "age groups", "age breakdown",
         "demographics", "demographic", "demographic breakdown", "racial makeup", "racial composition", "ethnic makeup",
         "race and ethnicity", "population by race", "population by age", "foreign born", "household size",
         "average household size", "number of households", "homeownership rate", "owner occupied", "renter occupied",
         "vacancy rate", "housing units", "median rent", "median gross rent", "educational attainment",
         "education level", "college educated", "population growth", "population change", "population trend",
         "population over time", "population history", "growth in population", "population decline",
         "population increase"],
        "Census topics that are NOT part of the loaded data: household and per-capita income, poverty, unemployment, age, "
        "race and ethnicity, households and household size, housing units, vacancy and rent, education, and population change "
        "over time. What IS loaded: total population of metropolitan/micropolitan areas and of census tracts from the latest "
        "decennial Census, which counties make up each metro area, and land area and population density derived from tract "
        "boundaries. Never estimate values for the unloaded topics.",
        gap=True, scope_guard=False,
    ),
})

ENTITY_DOMAINS = (
    EntityDomain("house_city", "houses", "city", "city", "Current house city labels", match_mode="exact_or_prefix", preferred_for=("house_inventory", "house_list_price", "house_walk_score", "house_bike_score", "house_transit_score")),
    EntityDomain("sold_city", "sold_homes", "city", "city", "Sold-home city labels", match_mode="exact_or_prefix", preferred_for=("sold_price",)),
    EntityDomain("msa_name", "census_msa", "name", "MSA", "MSA display names", match_mode="components", preferred_for=("msa_population", "nri_overall_risk", "nri_riverine_flood", "census_place_population", "census_land_area")),
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
    "zhvi": "housing",
    "market_heat_index": "housing",
    "geocode_cache": "system",
    "data_source_log": "system",
    "house_commute": "housing",
}
