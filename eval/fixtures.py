"""
eval/fixtures.py

Builds a small, deterministic DuckDB database for evaluation — separate from
whatever real data the app is loaded with. Every value here is hand-picked
so the golden set's expected answers can be verified by inspection, not
just trusted. golden_set.py imports the constants below instead of
duplicating magic numbers, so the two files can't drift apart.

Covers all four data sources the general agent draws on:
  - Redfin homes   (houses)
  - Sold homes     (sold_homes)
  - NRI            (nri_tracts)
  - Census         (census_tracts, census_msa, cbsa_counties)

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

    store.close()


if __name__ == "__main__":
    from config import settings
    build_fixture(settings.eval_fixture_duckdb_path)
    print(f"Built evaluation fixture at {settings.eval_fixture_duckdb_path}")
