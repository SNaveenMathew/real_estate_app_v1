"""
eval/fixtures.py

Builds a small, deterministic DuckDB database for evaluation — separate from
whatever real data the app is loaded with. Every value here is hand-picked
so the golden set's expected answers can be verified by inspection, not
just trusted. golden_set.py imports the constants below instead of
duplicating magic numbers, so the two files can't drift apart.

Covers every data source the general agent draws on that can be faked
honestly:
  - Redfin homes   (houses)
  - Sold homes     (sold_homes)
  - NRI            (nri_tracts)
  - Census         (census_tracts, census_msa, cbsa_counties)
  - Crime          (crime_incidents; severity from services/crime_taxonomy.py)
  - BikePGH        (bike_routes)
  - FTA transit    (transit_agencies, transit_ridership, transit_routes)
  - Commute        (house_commute)

The last four exist so the golden set has a quality gate for crime, bike,
transit and commute at all (recommendation 10). Their expected values are
re-derived from this fixture by SQL in tests/test_eval_domain_coverage.py.

Design choices worth knowing about when reading the golden set:
  - houses.msa_code is NULL for every house, matching real Redfin exports
    (see db/schema_catalog.py) — this is what makes `austin_avg_walk_score`
    a genuine NULL-handling test rather than a trivial one.
  - MSA_POPULATION and MSA_FLOOD_RISK are deliberately in DIFFERENT rank
    orders (Miami/Denver/Pittsburgh/Austin vs Miami/Denver/Austin/Pittsburgh)
    so a rank-order test can't pass by accident.
  - One MSA (UNMATCHED_MSA) has a placeholder 'X...' code that intentionally
    does not join to cbsa_counties, exercising that documented gotcha.
  - Pittsburgh has one arm's-length sale and one $1 non-arms-length sale in
    the SAME tract, so `pittsburgh_arms_length_avg_sold_price` only passes
    if the agent actually applies the documented sold_homes filter.
"""
from __future__ import annotations

import json
from pathlib import Path

import db.duckdb_store as store


# ─────────────────────────────────────────────────────────────────────────
# Metro areas / counties / tracts — the shared geography every table hangs off
# ─────────────────────────────────────────────────────────────────────────

# name -> (state, county_fips[5-digit], cbsa_code, tract_fips, msa_population,
#          tract_population, nri_risk_score, nri_rfld_risks, nri_hrcn_risks, nri_wfir_risks)
METROS = {
    "Pittsburgh": dict(state="PA", county_fips="42003", cbsa_code="38300",
                        tract_fips="42003140300", msa_population=2370930,
                        tract_population=4200, risk_score=45.2, rfld_risks=12.5,
                        hrcn_risks=3.1, wfir_risks=8.0),
    "Denver": dict(state="CO", county_fips="08031", cbsa_code="19740",
                    tract_fips="08031002500", msa_population=2963821,
                    tract_population=5100, risk_score=38.0, rfld_risks=25.0,
                    hrcn_risks=1.0, wfir_risks=22.0),
    "Miami": dict(state="FL", county_fips="12086", cbsa_code="33100",
                   tract_fips="12086001100", msa_population=6091747,
                   tract_population=6300, risk_score=72.5, rfld_risks=68.0,
                   hrcn_risks=55.0, wfir_risks=5.0),
    "Austin": dict(state="TX", county_fips="48453", cbsa_code="12420",
                    tract_fips="48453001700", msa_population=2295303,
                    tract_population=3900, risk_score=30.1, rfld_risks=15.2,
                    hrcn_risks=2.0, wfir_risks=30.0),
}

# Ground-truth rankings, derived (not hand-guessed) from METROS above so
# they can never silently drift out of sync with the fixture data.
MSA_POPULATION_RANK_DESC = sorted(METROS, key=lambda m: -METROS[m]["msa_population"])
MSA_FLOOD_RISK_RANK_DESC = sorted(METROS, key=lambda m: -METROS[m]["rfld_risks"])

# An MSA whose name never matched a real CBSA at load time — gets a
# placeholder code (see db/schema_catalog.py's census_msa.msa_code note).
UNMATCHED_MSA_NAME = "Sample Micro Area"
UNMATCHED_MSA_CODE = "Xplaceholder1"
UNMATCHED_MSA_POPULATION = 15000

# ─────────────────────────────────────────────────────────────────────────
# Houses (Redfin) — msa_code intentionally NULL on every row, matching
# real Redfin exports (db/schema_catalog.py: houses.msa_code note).
# ─────────────────────────────────────────────────────────────────────────

HOUSES = [
    dict(house_id="h1", city="Pittsburgh", metro="Pittsburgh", walk_score=82, price=350000, status="Active", is_favorite=True),
    dict(house_id="h2", city="Pittsburgh", metro="Pittsburgh", walk_score=45, price=275000, status="Active", is_favorite=True),
    dict(house_id="h3", city="Denver", metro="Denver", walk_score=70, price=525000, status="Active", is_favorite=True),
    dict(house_id="h4", city="Miami", metro="Miami", walk_score=90, price=610000, status="Pending", is_favorite=True),
    dict(house_id="h5", city="Austin", metro="Austin", walk_score=55, price=430000, status="Active", is_favorite=True),
    dict(house_id="h6", city="Austin", metro="Austin", walk_score=None, price=399000, status="Sold", is_favorite=True),
]

HOUSE_COUNT_BY_CITY = {"Pittsburgh": 2, "Denver": 1, "Miami": 1, "Austin": 2}
TOTAL_HOUSE_COUNT = len(HOUSES)
# h6's walk_score is NULL — a correct AVG() ignores it (55.0), not treats it as 0 (27.5).
AUSTIN_AVG_WALK_SCORE = 55.0
AUSTIN_AVG_WALK_SCORE_WRONG_IF_NULL_TREATED_AS_ZERO = 27.5

# ─────────────────────────────────────────────────────────────────────────
# Sold homes — one arm's-length sale and one $1 non-arms-length sale share
# Pittsburgh's tract on purpose (tests the documented default filter).
# ─────────────────────────────────────────────────────────────────────────

SOLD_HOMES = [
    dict(sale_id="s1", metro="Pittsburgh", sold_price=340000, is_arms_length=True, geocode_status="success"),
    dict(sale_id="s2", metro="Pittsburgh", sold_price=1, is_arms_length=False, geocode_status="success"),
    dict(sale_id="s3", metro="Denver", sold_price=510000, is_arms_length=True, geocode_status="success"),
    dict(sale_id="s4", metro="Miami", sold_price=590000, is_arms_length=None, geocode_status="success"),
    dict(sale_id="s5", metro="Austin", sold_price=415000, is_arms_length=True, geocode_status="success"),
    # Ungeocoded — no tract_fips yet. Included for realism; no golden example
    # keys on it directly, but it should never silently contaminate a
    # tract-scoped aggregate.
    dict(sale_id="s6", metro=None, sold_price=250000, is_arms_length=True, geocode_status="pending"),
]

# Only s1 passes the documented filter for Pittsburgh's tract:
# (is_arms_length IS NULL OR is_arms_length = TRUE) AND sold_price > 1000
# s2 fails on both counts (FALSE, and $1). Average of {340000} = 340000.
PITTSBURGH_ARMS_LENGTH_AVG_SOLD_PRICE = 340000.0
# The wrong answer you'd get by naively averaging both Pittsburgh sales
# without the filter: (340000 + 1) / 2.
PITTSBURGH_AVG_SOLD_PRICE_WRONG_IF_FILTER_SKIPPED = 170000.5


# ─────────────────────────────────────────────────────────────────────────
# Crime, bike, transit and commute — the four capabilities that had no eval
# coverage at all (audit §12, recommendation 10). Deliberately small: every
# number below is checkable by eye, and tests/test_eval_domain_coverage.py
# asserts each derived constant against a real SQL aggregate over this fixture.
# ─────────────────────────────────────────────────────────────────────────

CRIME_INCIDENTS = [
    dict(incident_id="crime-pgh-assault", city="pittsburgh", raw_type="Aggravated Assault",
         occurred_at="2024-03-04 21:10:00", lat=40.4406, lon=-79.9959, location_text="400 Block Forbes Ave"),
    dict(incident_id="crime-pgh-burglary", city="pittsburgh", raw_type="Burglary",
         occurred_at="2024-03-11 02:40:00", lat=40.4440, lon=-79.9900, location_text="100 Block Atwood St"),
    dict(incident_id="crime-pgh-theft", city="pittsburgh", raw_type="Theft",
         occurred_at="2024-04-02 13:05:00", lat=40.4380, lon=-80.0010, location_text="500 Block Craig St"),
    dict(incident_id="crime-mia-homicide", city="miami", raw_type="Homicide",
         occurred_at="2024-02-18 23:30:00", lat=25.7743, lon=-80.1937, location_text="200 Block NE 2nd Ave"),
    dict(incident_id="crime-den-theft", city="denver", raw_type="Theft",
         occurred_at="2024-05-09 16:45:00", lat=39.7392, lon=-104.9903, location_text="1500 Block Blake St"),
]


def _crime_category(raw_type: str) -> tuple[str, str, float]:
    """Category, label and severity come from the app's own taxonomy, never from a number typed here."""
    from services.crime_taxonomy import classify_crime_text
    return classify_crime_text(raw_type)


def _crime_weight(raw_type: str) -> float:
    return _crime_category(raw_type)[2]


CRIME_SEVERITY_WEIGHTS = [_crime_weight(c["raw_type"]) for c in CRIME_INCIDENTS]
PITTSBURGH_CRIME_COUNT = sum(1 for c in CRIME_INCIDENTS if c["city"] == "pittsburgh")
TOTAL_CRIME_COUNT = len(CRIME_INCIDENTS)
CRIME_CITY_COUNT = len({c["city"] for c in CRIME_INCIDENTS})
AVG_CRIME_SEVERITY_WEIGHT = round(sum(CRIME_SEVERITY_WEIGHTS) / len(CRIME_SEVERITY_WEIGHTS), 4)
MIAMI_CRIME_SEVERITY_TOTAL = round(
    sum(w for c, w in zip(CRIME_INCIDENTS, CRIME_SEVERITY_WEIGHTS) if c["city"] == "miami"), 4
)

# BikePGH-style line geometry. Three segments in Pittsburgh prove the city filter,
# one in Denver proves the table is not Pittsburgh-only.
BIKE_ROUTES = [
    dict(route_id="bike-pgh-lane-1", city="pittsburgh", layer_type="bike_lane", layer_label="Bike Lanes",
         color="#2e7d32", geometry=[[-79.9959, 40.4406], [-79.9900, 40.4450]]),
    dict(route_id="bike-pgh-lane-2", city="pittsburgh", layer_type="bike_lane", layer_label="Bike Lanes",
         color="#2e7d32", geometry=[[-79.9800, 40.4380], [-79.9750, 40.4420]]),
    dict(route_id="bike-pgh-trail-1", city="pittsburgh", layer_type="trail", layer_label="Trails",
         color="#1565c0", geometry=[[-80.0100, 40.4300], [-80.0000, 40.4360]]),
    dict(route_id="bike-den-lane-1", city="denver", layer_type="bike_lane", layer_label="Bike Lanes",
         color="#2e7d32", geometry=[[-104.9903, 39.7392], [-104.9850, 39.7430]]),
]
TOTAL_BIKE_ROUTE_COUNT = len(BIKE_ROUTES)
PITTSBURGH_BIKE_ROUTE_COUNT = sum(1 for b in BIKE_ROUTES if b["city"] == "pittsburgh")

# FTA National Transit Database rows: two agencies, four months of ridership.
TRANSIT_AGENCIES = [
    dict(ntd_id="30001", agency="Pittsburgh Regional Transit", mode="MB", tos="DO",
         hq_city="Pittsburgh", hq_state="PA", uza_name="Pittsburgh, PA", msa_code="38300"),
    dict(ntd_id="30002", agency="Regional Transportation District", mode="MB", tos="DO",
         hq_city="Denver", hq_state="CO", uza_name="Denver--Aurora, CO", msa_code="19740"),
]
TRANSIT_AGENCY_COUNT = len(TRANSIT_AGENCIES)
TRANSIT_RIDERSHIP = [
    dict(ntd_id="30001", agency="Pittsburgh Regional Transit", mode="MB", tos="DO", msa_code="38300",
         date="2024-01-01", year=2024, month=1, upt=100000.0),
    dict(ntd_id="30001", agency="Pittsburgh Regional Transit", mode="MB", tos="DO", msa_code="38300",
         date="2024-02-01", year=2024, month=2, upt=120000.0),
    dict(ntd_id="30002", agency="Regional Transportation District", mode="MB", tos="DO", msa_code="19740",
         date="2024-01-01", year=2024, month=1, upt=50000.0),
    dict(ntd_id="30002", agency="Regional Transportation District", mode="MB", tos="DO", msa_code="19740",
         date="2024-02-01", year=2024, month=2, upt=60000.0),
]
TOTAL_TRANSIT_RIDERSHIP = sum(r["upt"] for r in TRANSIT_RIDERSHIP)          # 330,000
TRANSIT_RIDERSHIP_BY_AGENCY = {
    agency: sum(r["upt"] for r in TRANSIT_RIDERSHIP if r["agency"] == agency)
    for agency in {r["agency"] for r in TRANSIT_RIDERSHIP}
}
TOP_RIDERSHIP_AGENCY = max(TRANSIT_RIDERSHIP_BY_AGENCY, key=TRANSIT_RIDERSHIP_BY_AGENCY.get)

TRANSIT_ROUTES = [
    dict(route_uid="tr-pgh-1", route_id="61A", ntd_id="30001", route_short_name="61A",
         route_long_name="Braddock-Squirrel Hill", route_type="3", route_type_name="Bus",
         geometry=[[-79.9959, 40.4406], [-79.9400, 40.4380]]),
    dict(route_uid="tr-pgh-2", route_id="P1", ntd_id="30001", route_short_name="P1",
         route_long_name="East Busway", route_type="3", route_type_name="Bus",
         geometry=[[-79.9959, 40.4406], [-79.9200, 40.4420]]),
]
TRANSIT_ROUTE_COUNT = len(TRANSIT_ROUTES)

# Commute estimates for two houses to one work location ("ok" is the status
# services/commute.py writes for a complete row).
COMMUTE_ROWS = [
    dict(house_id="h1", work_key="fixture-office", work_label="Office", drive_min=22.0, drive_miles=5.0,
         bike_min=35.0, bike_miles=4.5, walk_min=60.0, walk_miles=4.0, transit_min=40.0,
         transit_transfers=1, straight_line_miles=4.0, status="ok", source="fixture"),
    dict(house_id="h3", work_key="fixture-office", work_label="Office", drive_min=18.0, drive_miles=4.2,
         bike_min=30.0, bike_miles=3.8, walk_min=55.0, walk_miles=3.5, transit_min=35.0,
         transit_transfers=0, straight_line_miles=3.4, status="ok", source="fixture"),
]
AVG_DRIVE_TIME_TO_WORK = round(sum(r["drive_min"] for r in COMMUTE_ROWS) / len(COMMUTE_ROWS), 4)   # 20.0
SHORTEST_BIKE_COMMUTE = min(r["bike_min"] for r in COMMUTE_ROWS)                                    # 30.0


def build_fixture(db_path: Path) -> None:
    """(Re)build the evaluation fixture database at db_path from scratch."""
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    from config import settings
    store.close()  # drop any cached connection to a different .duckdb file
    settings.duckdb_path = db_path

    conn = store.get_conn()  # runs the app's own _ensure_schema()

    # ── cbsa_counties + census_msa (skip the unmatched MSA's county — it
    #    doesn't have one, that's the point) ────────────────────────────
    for name, m in METROS.items():
        conn.execute(
            "INSERT INTO cbsa_counties (cbsa_code, cbsa_title, msa_type, state_fips, "
            "county_fips, county_name, state_name) VALUES (?, ?, 'Metropolitan Statistical Area', ?, ?, ?, ?)",
            [m["cbsa_code"], f"{name}, {m['state']}", m["county_fips"][:2], m["county_fips"][2:], name, m["state"]],
        )
        conn.execute(
            "INSERT INTO census_msa (msa_code, geo_id, name, population) VALUES (?, ?, ?, ?)",
            [m["cbsa_code"], f"geo-{name}", f"{name}, {m['state']} Metro Area", m["msa_population"]],
        )
    conn.execute(
        "INSERT INTO census_msa (msa_code, geo_id, name, population) VALUES (?, ?, ?, ?)",
        [UNMATCHED_MSA_CODE, "geo-unmatched", UNMATCHED_MSA_NAME, UNMATCHED_MSA_POPULATION],
    )

    # ── nri_tracts + census_tracts ───────────────────────────────────────
    for name, m in METROS.items():
        conn.execute(
            "INSERT INTO nri_tracts (tract_fips, county_fips, state_fips, state_name, county_name, "
            "risk_score, risk_ratng, rfld_risks, hrcn_risks, wfir_risks) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [m["tract_fips"], m["county_fips"], m["county_fips"][:2], m["state"], name,
             m["risk_score"], "Relatively Moderate", m["rfld_risks"], m["hrcn_risks"], m["wfir_risks"]],
        )
        conn.execute(
            "INSERT INTO census_tracts (tract_fips, geo_id, name, population) VALUES (?, ?, ?, ?)",
            [m["tract_fips"], f"geo-tract-{name}", f"Census Tract in {name}", m["tract_population"]],
        )

    # ── houses ────────────────────────────────────────────────────────────
    import pandas as pd
    if not HOUSES or any(not bool(h.get("is_favorite")) for h in HOUSES):
        raise RuntimeError(
            "Evaluation fixture invariant failed: every fixture house must be saved/favorite "
            "because the golden set exercises saved-house semantics."
        )
    houses_df = pd.DataFrame([
        {
            "house_id": h["house_id"], "address": f"{100 + i} Main St", "city": h["city"],
            "state": METROS[h["metro"]]["state"], "zip": "00000", "lat": 0.0, "lon": 0.0,
            "status": h["status"], "price": float(h["price"]), "beds": 3.0, "baths": 2.0,
            "sqft": 1500.0, "year_built": 1990, "hoa_fee": 0.0,
            "walk_score": h["walk_score"], "bike_score": 50, "transit_score": 50,
            "tract_fips": METROS[h["metro"]]["tract_fips"], "msa_code": None,
            "source_file": "eval_fixture.csv", "raw_json": "{}", "is_favorite": True,
        }
        for i, h in enumerate(HOUSES)
    ])
    store.upsert_houses(houses_df)

    # ── sold_homes ────────────────────────────────────────────────────────
    for s in SOLD_HOMES:
        tract_fips = METROS[s["metro"]]["tract_fips"] if s["metro"] else None
        county_fips = METROS[s["metro"]]["county_fips"] if s["metro"] else None
        conn.execute(
            "INSERT INTO sold_homes (sale_id, source_county, address, city, state, zip, "
            "sold_price, sold_date, sqft, is_arms_length, tract_fips, county_fips, geocode_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [s["sale_id"], s["metro"] or "unknown", f"{s['sale_id']} Sale St",
             s["metro"] or "Unknown", METROS.get(s["metro"], {}).get("state", "??"), "00000",
             float(s["sold_price"]), "2024-06-01", 1500.0, s["is_arms_length"],
             tract_fips, county_fips, s["geocode_status"]],
        )

    # ── crime_incidents ──────────────────────────────────────────────────
    for c in CRIME_INCIDENTS:
        category, label, weight = _crime_category(c["raw_type"])
        occurred = c["occurred_at"]
        conn.execute(
            "INSERT INTO crime_incidents (incident_id, city, source_file, occurred_at, year, month, year_month, "
            "lat, lon, category, category_label, severity_weight, raw_type, location_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [c["incident_id"], c["city"], "eval_fixture.csv", occurred, int(occurred[:4]), int(occurred[5:7]),
             occurred[:7], c["lat"], c["lon"], category, label, weight, c["raw_type"], c["location_text"]],
        )

    # ── bike_routes ──────────────────────────────────────────────────────
    for b in BIKE_ROUTES:
        lons = [point[0] for point in b["geometry"]]
        lats = [point[1] for point in b["geometry"]]
        conn.execute(
            "INSERT INTO bike_routes (route_id, city, layer_type, layer_label, color, source_file, "
            "min_lon, min_lat, max_lon, max_lat, geometry_json, properties_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [b["route_id"], b["city"], b["layer_type"], b["layer_label"], b.get("color"), "eval_fixture.geojson",
             min(lons), min(lats), max(lons), max(lats),
             json.dumps({"type": "LineString", "coordinates": b["geometry"]}),
             json.dumps({"layer": b["layer_type"]})],
        )

    # ── transit agencies / ridership / routes ─────────────────────────────
    for a in TRANSIT_AGENCIES:
        conn.execute(
            "INSERT INTO transit_agencies (ntd_id, agency, mode, tos, hq_city, hq_state, uza_name, msa_code) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [a["ntd_id"], a["agency"], a["mode"], a["tos"], a["hq_city"], a["hq_state"],
             a["uza_name"], a["msa_code"]],
        )
    for r in TRANSIT_RIDERSHIP:
        conn.execute(
            "INSERT INTO transit_ridership (ntd_id, agency, mode, tos, msa_code, date, year, month, upt) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [r["ntd_id"], r["agency"], r["mode"], r["tos"], r["msa_code"], r["date"],
             r["year"], r["month"], r["upt"]],
        )
    for t in TRANSIT_ROUTES:
        lons = [point[0] for point in t["geometry"]]
        lats = [point[1] for point in t["geometry"]]
        conn.execute(
            "INSERT INTO transit_routes (route_uid, route_id, ntd_id, route_short_name, route_long_name, "
            "route_type, route_type_name, min_lon, min_lat, max_lon, max_lat, geometry_json, properties_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [t["route_uid"], t["route_id"], t["ntd_id"], t["route_short_name"], t["route_long_name"],
             t["route_type"], t["route_type_name"], min(lons), min(lats), max(lons), max(lats),
             json.dumps({"type": "LineString", "coordinates": t["geometry"]}), "{}"],
        )

    # ── house_commute ─────────────────────────────────────────────────────
    for row in COMMUTE_ROWS:
        conn.execute(
            "INSERT INTO house_commute (house_id, work_key, work_label, drive_min, drive_miles, bike_min, "
            "bike_miles, walk_min, walk_miles, transit_min, transit_transfers, straight_line_miles, status, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [row["house_id"], row["work_key"], row["work_label"], row["drive_min"], row["drive_miles"],
             row["bike_min"], row["bike_miles"], row["walk_min"], row["walk_miles"], row["transit_min"],
             row["transit_transfers"], row["straight_line_miles"], row["status"], row["source"]],
        )

    store.close()


if __name__ == "__main__":
    from config import settings
    build_fixture(settings.eval_fixture_duckdb_path)
    print(f"Built evaluation fixture at {settings.eval_fixture_duckdb_path}")
