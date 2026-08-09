"""
LangChain tools shared by house agent and general agent.
"""
import json
import textwrap
import traceback
from typing import Any

import pandas as pd
from langchain.tools import tool

import db.duckdb_store as store
import db.vector_store as vs


# ── House-specific tools ─────────────────────────────────────────────────────

def make_house_tools(house_id: str):
    """Return tools bound to a specific house."""

    @tool
    def get_house_details(_: str = "") -> str:
        """Get the structured details of this house from the database."""
        house = store.get_house(house_id)
        if not house:
            return "House not found in database."
        # Remove internal fields
        house.pop("raw_json", None)
        return json.dumps({k: v for k, v in house.items() if v is not None}, indent=2)

    @tool
    def get_nri_risk_data(_: str = "") -> str:
        """Get FEMA National Risk Index data for this house's census tract."""
        house = store.get_house(house_id)
        if not house or not house.get("tract_fips"):
            return "Census tract not available for this house."
        nri = store.get_nri_for_tract(house["tract_fips"])
        if not nri:
            return f"No NRI data found for tract {house['tract_fips']}."
        # Return a human-readable summary
        hazards = {
            "Avalanche": nri.get("avln_risks"),
            "Coastal Flooding": nri.get("cfld_risks"),
            "Cold Wave": nri.get("cwav_risks"),
            "Drought": nri.get("drgt_risks"),
            "Earthquake": nri.get("erqk_risks"),
            "Hail": nri.get("hail_risks"),
            "Heat Wave": nri.get("hwav_risks"),
            "Hurricane": nri.get("hrcn_risks"),
            "Ice Storm": nri.get("istm_risks"),
            "Landslide": nri.get("lnds_risks"),
            "Lightning": nri.get("ltng_risks"),
            "Riverine Flooding": nri.get("rfld_risks"),
            "Strong Wind": nri.get("swnd_risks"),
            "Tornado": nri.get("trnd_risks"),
            "Tsunami": nri.get("tsun_risks"),
            "Volcanic Activity": nri.get("vlcn_risks"),
            "Wildfire": nri.get("wfir_risks"),
            "Winter Weather": nri.get("wntw_risks"),
        }
        top = sorted(
            [(k, v) for k, v in hazards.items() if v and v > 0],
            key=lambda x: x[1], reverse=True
        )[:5]
        return json.dumps({
            "tract_fips": nri.get("tract_fips"),
            "county": nri.get("county_name"),
            "state": nri.get("state_name"),
            "composite_risk_score": nri.get("risk_score"),
            "composite_risk_rating": nri.get("risk_ratng"),
            "risk_percentile": nri.get("risk_npctl"),
            "expected_annual_loss_usd": nri.get("eal_valt"),
            "social_vulnerability": nri.get("sovi_ratng"),
            "community_resilience": nri.get("resl_ratng"),
            "top_5_hazards_by_risk_score": top,
        }, indent=2)

    @tool
    def search_house_documents(query: str) -> str:
        """Search documents and descriptions stored for this house."""
        docs = vs.search_house(house_id, query, n_results=4)
        if not docs:
            return "No documents found for this house yet."
        return "\n\n".join(f"[{d['metadata'].get('doc_type','text')}]\n{d['text']}"
                           for d in docs)

    @tool
    def estimate_price_with_code(_: str = "") -> str:
        """
        Compute price estimates using comparable data from DuckDB.
        Uses only arm's-length sold transactions. Returns statistics and
        multiple estimation methods.
        """
        house = store.get_house(house_id)
        if not house:
            return "House not found."

        tract = house.get("tract_fips")
        sqft  = house.get("sqft")
        price = house.get("price")

        lines = []

        # --- Active listings in same tract ---
        if tract:
            stats = store.get_price_stats_in_tract(tract)
            if stats.get("count", 0) > 0:
                lines.append("### Comparable Active Listings (same census tract)")
                lines.append(f"  Count: {int(stats['count'])}")
                lines.append(f"  Median list price: ${stats.get('median_price', 0):,.0f}")
                lines.append(f"  Avg list price: ${stats.get('avg_price', 0):,.0f}")
                if stats.get("median_price_per_sqft"):
                    ppsf = stats["median_price_per_sqft"]
                    lines.append(f"  Median price/sqft: ${ppsf:,.2f}")
                    if sqft:
                        est = ppsf * sqft
                        lines.append(
                            f"  → Estimated value (median $/sqft × {sqft:.0f} sqft): ${est:,.0f}"
                        )

            if stats.get("sold_sold_count", 0) > 0:
                lines.append("\n### Recent Sales — arm's-length only (same census tract)")
                lines.append(f"  Sold count: {int(stats['sold_sold_count'])}")
                lines.append(f"  Median sold price: ${stats.get('sold_median_sold', 0):,.0f}")
                if stats.get("sold_median_sold_per_sqft") and sqft:
                    est2 = stats["sold_median_sold_per_sqft"] * sqft
                    lines.append(
                        f"  → Estimated value (sold $/sqft × {sqft:.0f} sqft): ${est2:,.0f}"
                    )
                lines.append(
                    f"  Sale date range: {stats.get('sold_oldest_sale')} → "
                    f"{stats.get('sold_newest_sale')}"
                )

        # --- This house's list price ---
        if price:
            lines.append(f"\n### Current List Price: ${price:,.0f}")
            if sqft:
                lines.append(f"  Price/sqft: ${price/sqft:,.2f}")

        # --- Nearby houses (same city) ---
        city = house.get("city")
        if city:
            nearby = store.query("""
                SELECT price, sqft, beds, baths, status,
                       ROUND(price / NULLIF(sqft, 0), 0) as ppsf
                FROM houses
                WHERE city = ? AND price > 0 AND house_id != ?
                ORDER BY ABS(price - COALESCE(?, price)) ASC
                LIMIT 8
            """, [city, house_id, price])
            if len(nearby) > 0:
                lines.append(f"\n### Similar Active Listings in {city}")
                lines.append(nearby.to_string(index=False))

        # --- County sold comps nearby (within same municipality if available) ---
        if tract:
            county_comps = store.query("""
                SELECT address, city, sold_price, sqft, sold_date,
                       muni_desc, sale_desc, arms_length_flag,
                       ROUND(sold_price / NULLIF(sqft, 0), 0) as ppsf
                FROM sold_homes
                WHERE tract_fips = ?
                  AND (is_arms_length IS NULL OR is_arms_length = TRUE)
                  AND sold_price > 1000
                ORDER BY sold_date DESC
                LIMIT 8
            """, [tract])
            if len(county_comps) > 0:
                lines.append("\n### County-recorded Arm's-Length Sales (same tract)")
                lines.append(county_comps.to_string(index=False))

        if not lines:
            return "Insufficient data for price estimation. More sales data needed."
        return "\n".join(lines)

    @tool
    def get_nearby_sold_homes(_: str = "") -> str:
        """Get recent sold homes in the same census tract for comps."""
        house = store.get_house(house_id)
        if not house or not house.get("tract_fips"):
            return "No tract data available."
        sold = store.get_sold_in_tract(house["tract_fips"])
        if not sold:
            return "No sold homes found in this census tract yet."
        df = pd.DataFrame(sold[:10])
        cols = ["address", "sold_price", "sqft", "beds", "baths", "sold_date"]
        cols = [c for c in cols if c in df.columns]
        return df[cols].to_string(index=False)

    return [get_house_details, get_nri_risk_data, search_house_documents,
            estimate_price_with_code, get_nearby_sold_homes]


# ── General tools ────────────────────────────────────────────────────────────

@tool
def check_data_availability(_: str = "") -> str:
    """
    ALWAYS call this first before answering any question that requires data.
    Returns the row count for every table so you know what is actually loaded.
    If a table has 0 rows, you CANNOT answer questions that depend on it —
    tell the user which files need to be loaded instead.
    """
    tables = [
        "houses", "nri_tracts", "census_tracts",
        "census_msa", "cbsa_counties", "sold_homes", "geocode_cache",
    ]
    lines = ["Table row counts (0 = not loaded yet):"]
    for t in tables:
        try:
            n = store.query(f"SELECT COUNT(*) as n FROM {t}").iloc[0]["n"]
            status = "✓" if n > 0 else "✗ EMPTY"
            lines.append(f"  {status}  {t:<20} {int(n):>10,} rows")
        except Exception as e:
            lines.append(f"  ?  {t:<20} error: {e}")

    lines.append("")
    lines.append("Setup instructions for empty tables:")
    lines.append("  census_msa:    download DECENNIALPL2020.P1 for MSAs from data.census.gov")
    lines.append("                 → save as data/census/DECENNIALPL2020_P1_msa.csv")
    lines.append("                 → run: python setup_data.py --only census")
    lines.append("  cbsa_counties: download list1_2020.csv from Census CBSA delineation files")
    lines.append("                 → save as data/census/list1_2020.csv")
    lines.append("                 → run: python setup_data.py --only census")
    return "\n".join(lines)


@tool
def query_database(sql: str) -> str:
    """
    Execute a SELECT query against the real estate DuckDB database.
    Call check_data_availability first to confirm the tables you need have data.
    Do NOT guess or fabricate results if the query returns empty — report it.
    """
    sql_upper = sql.upper().strip()
    if any(kw in sql_upper for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]):
        return "ERROR: Only SELECT queries are allowed."
    try:
        df = store.query(sql)
        if len(df) == 0:
            # Try to detect if the FROM table is itself empty
            import re
            tables_in_query = re.findall(r'\bFROM\s+(\w+)', sql_upper)
            tables_in_query += re.findall(r'\bJOIN\s+(\w+)', sql_upper)
            empty_tables = []
            for t in set(tables_in_query):
                try:
                    n = store.query(f"SELECT COUNT(*) as n FROM {t}").iloc[0]["n"]
                    if n == 0:
                        empty_tables.append(t)
                except Exception:
                    pass
            if empty_tables:
                return (
                    f"EMPTY TABLES DETECTED: {', '.join(empty_tables)} have 0 rows.\n"
                    f"These files have not been loaded yet. Run check_data_availability "
                    f"for setup instructions. DO NOT fabricate results."
                )
            return "Query returned 0 rows. The data may not match your filter criteria."
        if len(df) > 50:
            return df.head(50).to_string(index=False) + f"\n... ({len(df)} total rows, showing 50)"
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {e}\nCheck column names with get_database_schema."


@tool
def get_top_msas_by_flood_risk(
    top_n_by_population: int = 50,
    show_lowest_n: int = 10,
    hazard: str = "rfld_risks",
) -> str:
    """
    Rank the largest MSAs by a specific NRI hazard risk score (default: riverine flood).

    Parameters
    ----------
    top_n_by_population : how many MSAs to consider (ranked by population)
    show_lowest_n       : how many to show in output (lowest risk first)
    hazard              : NRI column — rfld_risks, cfld_risks, hrcn_risks, trnd_risks,
                          wfir_risks, erqk_risks, swnd_risks, hail_risks, hwav_risks,
                          drgt_risks, risk_score (composite), etc.
    
    """
    import logging
    log = logging.getLogger("flood_risk_tool")
    # Validate hazard column
    valid_hazards = [
        "rfld_risks", "cfld_risks", "hrcn_risks", "trnd_risks", "wfir_risks",
        "erqk_risks", "swnd_risks", "hail_risks", "hwav_risks", "drgt_risks",
        "ltng_risks", "wntw_risks", "istm_risks", "lnds_risks", "cwav_risks",
        "tsun_risks", "vlcn_risks", "avln_risks", "risk_score",
    ]
    if hazard not in valid_hazards:
        return f"Unknown hazard '{hazard}'. Valid options: {', '.join(valid_hazards)}"

    hazard_label = {
        "rfld_risks": "Inland / Riverine Flood", "cfld_risks": "Coastal Flood",
        "hrcn_risks": "Hurricane",      "trnd_risks": "Tornado",
        "wfir_risks": "Wildfire",       "erqk_risks": "Earthquake",
        "swnd_risks": "Strong Wind",    "hail_risks": "Hail",
        "hwav_risks": "Heat Wave",      "drgt_risks": "Drought",
        "risk_score": "Composite Risk",
    }.get(hazard, hazard)

    # ── Step 1: Count rows in each required table ─────────────────────────
    try:
        nri_count  = int(store.query("SELECT COUNT(*) as n FROM nri_tracts").iloc[0]["n"])
        cbsa_count = int(store.query("SELECT COUNT(*) as n FROM cbsa_counties").iloc[0]["n"])
        msa_count  = int(store.query("SELECT COUNT(*) as n FROM census_msa WHERE population IS NOT NULL").iloc[0]["n"])
    except Exception as e:
        return f"Database error reading table counts: {e}"

    log.warning(f"[flood] Step1 — nri={nri_count:,} cbsa={cbsa_count:,} msa={msa_count:,}")
    print(f"[flood] Step1 — nri_tracts={nri_count:,}  cbsa_counties={cbsa_count:,}  census_msa={msa_count:,}")

    if nri_count == 0:
        return "ERROR: nri_tracts is empty. Run: python setup_data.py --only nri"
    if cbsa_count == 0:
        return "ERROR: cbsa_counties is empty. Run: python setup_data.py --only census"

    # ── Step 2: Diagnose the cbsa↔nri county join ─────────────────────────
    try:
        sample_cbsa = store.query(
            "SELECT state_fips, county_fips, state_fips || county_fips AS joined_key FROM cbsa_counties LIMIT 3"
        )
        sample_nri  = store.query("SELECT county_fips FROM nri_tracts LIMIT 3")
        cbsa_key_sample = sample_cbsa["joined_key"].tolist()
        nri_key_sample  = sample_nri["county_fips"].tolist()

        join_count = int(store.query(f"""
            SELECT COUNT(DISTINCT cb.cbsa_code) as n
            FROM cbsa_counties cb
            JOIN nri_tracts n ON (cb.state_fips || cb.county_fips) = n.county_fips
            WHERE n.{hazard} IS NOT NULL
        """).iloc[0]["n"])
    except Exception as e:
        return f"Diagnostic query failed: {e}"

    print(f"[flood] Step2 — cbsa↔nri join: {join_count} matching CBSAs")
    print(f"[flood]   cbsa key samples: {cbsa_key_sample}")
    print(f"[flood]   nri  key samples: {nri_key_sample}")

    if join_count == 0:
        return (
            f"DIAGNOSTIC: cbsa_counties ↔ nri_tracts join returns 0 rows.\n"
            f"cbsa key (state_fips||county_fips): {cbsa_key_sample}\n"
            f"nri county_fips:                    {nri_key_sample}\n"
            "Both must be 5-digit strings like '42003'. "
            "Check NRI shapefile and CBSA file are compatible vintages."
        )

    # ── Step 3: Check msa_code ↔ cbsa_code linkage ───────────────────────
    total_msas   = top_n_by_population
    matched_cbsa = 0

    if msa_count > 0:
        try:
            linkage = store.query(f"""
                WITH top_msas AS (
                    SELECT msa_code, name, population
                    FROM census_msa
                    WHERE population IS NOT NULL
                    ORDER BY CAST(population AS BIGINT) DESC
                    LIMIT {top_n_by_population}
                )
                SELECT
                    COUNT(*) AS total_msas,
                    COUNT(cb.cbsa_code) AS matched_to_cbsa
                FROM top_msas m
                LEFT JOIN (SELECT DISTINCT cbsa_code FROM cbsa_counties) cb
                  ON m.msa_code = cb.cbsa_code
            """)
            total_msas   = int(linkage["total_msas"].iloc[0])
            matched_cbsa = int(linkage["matched_to_cbsa"].iloc[0])
        except Exception as e:
            matched_cbsa = -1

        # Also count X-coded rows in the top N
        try:
            x_in_top = int(store.query(f"""
                SELECT COUNT(*) as n FROM (
                    SELECT msa_code FROM census_msa
                    WHERE population IS NOT NULL
                    ORDER BY CAST(population AS BIGINT) DESC
                    LIMIT {top_n_by_population}
                ) t WHERE msa_code LIKE 'X%'
            """).iloc[0]["n"])
        except Exception:
            x_in_top = -1

        print(f"[flood] Step3 — top {total_msas} MSAs: {matched_cbsa} have valid cbsa_code, "
              f"{x_in_top} still have X-codes")

        if matched_cbsa == 0:
            sample_msa   = store.query("SELECT msa_code FROM census_msa LIMIT 3")["msa_code"].tolist()
            sample_cbsa2 = store.query("SELECT cbsa_code FROM cbsa_counties LIMIT 3")["cbsa_code"].tolist()
            return (
                f"DIAGNOSTIC: census_msa.msa_code doesn't match cbsa_counties.cbsa_code.\n"
                f"census_msa samples:    {sample_msa}\n"
                f"cbsa_counties samples: {sample_cbsa2}\n"
                "Run: python setup_data.py --only repair"
            )

    # ── Step 4: Run the actual query ──────────────────────────────────────
    print(f"[flood] Step4 — running final query (msa_count={msa_count})")

    if msa_count > 0:
        sql = f"""
            WITH top_msas AS (
                SELECT msa_code, name,
                       CAST(population AS BIGINT) AS population
                FROM census_msa
                WHERE population IS NOT NULL
                ORDER BY CAST(population AS BIGINT) DESC
                LIMIT {top_n_by_population}
            ),
            msa_risk AS (
                SELECT cb.cbsa_code,
                       AVG(n.{hazard})              AS avg_hazard_score,
                       COUNT(DISTINCT n.tract_fips) AS tract_count
                FROM cbsa_counties cb
                JOIN nri_tracts n
                  ON (cb.state_fips || cb.county_fips) = n.county_fips
                WHERE n.{hazard} IS NOT NULL
                GROUP BY cb.cbsa_code
            )
            SELECT m.name                         AS msa_name,
                   m.population,
                   ROUND(r.avg_hazard_score, 2)   AS avg_{hazard},
                   r.tract_count
            FROM top_msas m
            JOIN msa_risk r ON m.msa_code = r.cbsa_code
            ORDER BY r.avg_hazard_score ASC
            LIMIT {show_lowest_n}
        """
        source_note = (
            f"Source: Census 2020 MSA populations + FEMA NRI tracts.\n"
            f"Top {top_n_by_population} by population, showing {show_lowest_n} "
            f"with lowest {hazard_label} risk.\n"
            f"({join_count} CBSAs have NRI data; "
            f"{matched_cbsa}/{total_msas} top MSAs linked to CBSA codes.)"
        )
    else:
        sql = f"""
            WITH msa_risk AS (
                SELECT cb.cbsa_code,
                       cb.cbsa_title AS msa_name,
                       AVG(n.{hazard}) AS avg_hazard_score,
                       COUNT(DISTINCT n.tract_fips) AS tract_count,
                       COUNT(DISTINCT cb.state_fips || cb.county_fips) AS county_count
                FROM cbsa_counties cb
                JOIN nri_tracts n
                  ON (cb.state_fips || cb.county_fips) = n.county_fips
                WHERE n.{hazard} IS NOT NULL
                  AND cb.msa_type ILIKE '%Metropolitan%'
                GROUP BY cb.cbsa_code, cb.cbsa_title
            )
            SELECT msa_name, county_count,
                   ROUND(avg_hazard_score, 2) AS avg_{hazard},
                   tract_count
            FROM msa_risk
            ORDER BY avg_hazard_score ASC
            LIMIT {show_lowest_n}
        """
        source_note = (
            f"⚠ census_msa population data not loaded — cannot rank by population.\n"
            f"Showing {show_lowest_n} Metro MSAs with lowest {hazard_label} risk.\n"
            "Load census_msa for top-by-population ranking."
        )

    try:
        df = store.query(sql)
    except Exception as e:
        print(f"[flood] Step4 SQL ERROR: {e}")
        return f"SQL Error: {e}"

    print(f"[flood] Step4 — query returned {len(df)} rows")

    if df.empty:
        # Extra diagnostic: check each join individually
        top_n_check = store.query(f"""
            SELECT COUNT(*) as n FROM census_msa
            WHERE population IS NOT NULL
            ORDER BY CAST(population AS BIGINT) DESC
            LIMIT {top_n_by_population}
        """).iloc[0]["n"] if msa_count > 0 else 0

        risk_check = store.query(f"""
            SELECT COUNT(DISTINCT cbsa_code) as n FROM (
                SELECT cb.cbsa_code
                FROM cbsa_counties cb
                JOIN nri_tracts n ON (cb.state_fips || cb.county_fips) = n.county_fips
                WHERE n.{hazard} IS NOT NULL
                GROUP BY cb.cbsa_code
            )
        """).iloc[0]["n"]

        print(f"[flood] Step4 EMPTY — top_msas rows: {top_n_check}, msa_risk CBSAs: {risk_check}")
        print(f"[flood]   matched_cbsa={matched_cbsa}/{total_msas}")
        print("[flood]   Hint: run 'python debug_flood_query.py' for full details")
        return (
            f"Query ran but returned 0 rows.\n\n"
            f"Diagnostic:\n"
            f"  cbsa↔nri join: {join_count} CBSAs have NRI data ✓\n"
            f"  msa↔cbsa link: {matched_cbsa}/{total_msas} top MSAs have valid codes\n"
            f"  msa_risk CBSAs in db: {risk_check}\n\n"
            f"Run for full step-by-step details:\n"
            f"  python debug_flood_query.py"
        )
    # output  = f"### {hazard_label} Risk — Lowest Among Top MSAs\n\n"
    # output += df.to_string(index=False)
    # output += f"\n\n{source_note}"
    # return output
    valid_hazards = [
        "rfld_risks", "cfld_risks", "hrcn_risks", "trnd_risks", "wfir_risks",
        "erqk_risks", "swnd_risks", "hail_risks", "hwav_risks", "drgt_risks",
        "ltng_risks", "wntw_risks", "istm_risks", "lnds_risks", "cwav_risks",
        "tsun_risks", "vlcn_risks", "avln_risks", "risk_score",
    ]
    if hazard not in valid_hazards:
        return f"Unknown hazard '{hazard}'. Valid: {', '.join(valid_hazards)}"

    hazard_label = {
        "rfld_risks": "Inland / Riverine Flood", "cfld_risks": "Coastal Flood",
        "hrcn_risks": "Hurricane",      "trnd_risks": "Tornado",
        "wfir_risks": "Wildfire",       "erqk_risks": "Earthquake",
        "swnd_risks": "Strong Wind",    "hail_risks": "Hail",
        "hwav_risks": "Heat Wave",      "drgt_risks": "Drought",
        "risk_score": "Composite Risk",
    }.get(hazard, hazard)

    # ── Step 1: Count rows in each required table ─────────────────────────
    try:
        nri_count  = int(store.query("SELECT COUNT(*) as n FROM nri_tracts").iloc[0]["n"])
        cbsa_count = int(store.query("SELECT COUNT(*) as n FROM cbsa_counties").iloc[0]["n"])
        msa_count  = int(store.query("SELECT COUNT(*) as n FROM census_msa WHERE population IS NOT NULL").iloc[0]["n"])
    except Exception as e:
        return f"Database error reading table counts: {e}"

    if nri_count == 0:
        return "ERROR: nri_tracts is empty. Run: python setup_data.py --only nri"
    if cbsa_count == 0:
        return "ERROR: cbsa_counties is empty. Run: python setup_data.py --only census"

    # ── Step 2: Diagnose the cbsa↔nri county join ─────────────────────────
    # This is the most common failure point — check key formats on both sides.
    try:
        sample_cbsa = store.query("""
            SELECT state_fips, county_fips,
                   state_fips || county_fips AS joined_key
            FROM cbsa_counties LIMIT 3
        """)
        sample_nri = store.query("""
            SELECT county_fips FROM nri_tracts LIMIT 3
        """)
        cbsa_key_sample = sample_cbsa["joined_key"].tolist()
        nri_key_sample  = sample_nri["county_fips"].tolist()

        # Check how many counties actually match
        join_count = int(store.query(f"""
            SELECT COUNT(DISTINCT cb.cbsa_code) as n
            FROM cbsa_counties cb
            JOIN nri_tracts n
              ON (cb.state_fips || cb.county_fips) = n.county_fips
            WHERE n.{hazard} IS NOT NULL
        """).iloc[0]["n"])
    except Exception as e:
        return f"Diagnostic query failed: {e}"

    if join_count == 0:
        return (
            f"DIAGNOSTIC: The county FIPS join between cbsa_counties and nri_tracts "
            f"returns 0 matching counties.\n\n"
            f"cbsa_counties key format (state_fips || county_fips): {cbsa_key_sample}\n"
            f"nri_tracts county_fips format:                        {nri_key_sample}\n\n"
            "These must be the same format (both 5-digit strings like '42003').\n"
            "Check that the NRI shapefile and CBSA delineation file are from "
            "compatible vintages."
        )

    # ── Step 3: Check msa_code ↔ cbsa_code linkage ───────────────────────
    if msa_count > 0:
        try:
            # How many of the top-N MSAs by population have a matching cbsa_code?
            linkage = store.query(f"""
                WITH top_msas AS (
                    SELECT msa_code, name, population
                    FROM census_msa
                    WHERE population IS NOT NULL
                    ORDER BY CAST(population AS BIGINT) DESC
                    LIMIT {top_n_by_population}
                )
                SELECT
                    COUNT(*) AS total_msas,
                    COUNT(cb.cbsa_code) AS matched_to_cbsa
                FROM top_msas m
                LEFT JOIN (SELECT DISTINCT cbsa_code FROM cbsa_counties) cb
                  ON m.msa_code = cb.cbsa_code
            """)
            total_msas   = int(linkage["total_msas"].iloc[0])
            matched_cbsa = int(linkage["matched_to_cbsa"].iloc[0])
        except Exception as e:
            matched_cbsa = -1
            total_msas   = top_n_by_population

        if matched_cbsa == 0:
            # msa_code doesn't match any cbsa_code — name matching probably failed
            sample_msa_codes  = store.query("SELECT msa_code FROM census_msa LIMIT 3")["msa_code"].tolist()
            sample_cbsa_codes = store.query("SELECT cbsa_code FROM cbsa_counties LIMIT 3")["cbsa_code"].tolist()
            return (
                f"DIAGNOSTIC: census_msa.msa_code does not match cbsa_counties.cbsa_code.\n\n"
                f"census_msa.msa_code samples:   {sample_msa_codes}\n"
                f"cbsa_counties.cbsa_code samples: {sample_cbsa_codes}\n\n"
                "The MSA name→CBSA code matching failed during data loading.\n"
                "Fix: delete data/real_estate.duckdb and re-run setup_data.py\n"
                "(CBSA crosswalk loads before MSA populations, enabling name matching)."
            )

    # ── Step 4: Run the actual query ──────────────────────────────────────
    if msa_count > 0:
        sql = f"""
            WITH top_msas AS (
                SELECT msa_code, name, population
                FROM census_msa
                WHERE population IS NOT NULL
                ORDER BY CAST(population AS BIGINT) DESC
                LIMIT {top_n_by_population}
            ),
            msa_risk AS (
                SELECT
                    cb.cbsa_code,
                    AVG(n.{hazard})              AS avg_hazard_score,
                    COUNT(DISTINCT n.tract_fips) AS tract_count
                FROM cbsa_counties cb
                JOIN nri_tracts n
                  ON (cb.state_fips || cb.county_fips) = n.county_fips
                WHERE n.{hazard} IS NOT NULL
                GROUP BY cb.cbsa_code
            )
            SELECT
                m.name                              AS msa_name,
                CAST(m.population AS BIGINT)        AS population,
                ROUND(r.avg_hazard_score, 2)        AS avg_{hazard},
                r.tract_count
            FROM top_msas m
            JOIN msa_risk r ON m.msa_code = r.cbsa_code
            ORDER BY r.avg_hazard_score ASC
            LIMIT {show_lowest_n}
        """
        source_note = (
            f"Source: Census 2020 MSA populations + FEMA NRI tracts.\n"
            f"Scope: top {top_n_by_population} MSAs by population, "
            f"showing {show_lowest_n} with lowest {hazard_label} risk.\n"
            f"({join_count} CBSA areas have NRI data; "
            f"{matched_cbsa}/{total_msas} top MSAs linked to CBSA codes.)"
        )

    else:
        # Path B: no census_msa — rank all CBSAs by NRI directly
        sql = f"""
            WITH msa_risk AS (
                SELECT
                    cb.cbsa_code,
                    cb.cbsa_title                        AS msa_name,
                    AVG(n.{hazard})                      AS avg_hazard_score,
                    COUNT(DISTINCT n.tract_fips)         AS tract_count,
                    COUNT(DISTINCT cb.state_fips || cb.county_fips) AS county_count
                FROM cbsa_counties cb
                JOIN nri_tracts n
                  ON (cb.state_fips || cb.county_fips) = n.county_fips
                WHERE n.{hazard} IS NOT NULL
                  AND cb.msa_type ILIKE '%Metropolitan%'
                GROUP BY cb.cbsa_code, cb.cbsa_title
            )
            SELECT
                msa_name,
                county_count,
                ROUND(avg_hazard_score, 2) AS avg_{hazard},
                tract_count
            FROM msa_risk
            ORDER BY avg_hazard_score ASC
            LIMIT {show_lowest_n}
        """
        source_note = (
            f"⚠ census_msa population data not loaded — cannot rank by population.\n"
            f"Showing {show_lowest_n} Metro MSAs with lowest {hazard_label} risk overall.\n"
            f"Load census_msa for top-{top_n_by_population}-by-population ranking."
        )

    try:
        df = store.query(sql)
    except Exception as e:
        return f"SQL Error: {e}"
    
    if df.empty:
        return (
            f"Query ran but returned 0 rows (unexpected — {join_count} CBSA areas "
            f"have NRI data and {matched_cbsa}/{total_msas} top MSAs have CBSA codes).\n"
            f"Try: query_database with a simpler version to debug further."
        )

    output  = f"### {hazard_label} Risk — Lowest Among Top MSAs\n\n"
    output += df.to_string(index=False)
    output += f"\n\n{source_note}"
    return output


@tool
def search_all_house_descriptions(query: str) -> str:
    """Search all house descriptions stored in the vector knowledge base."""
    docs = vs.search_all(query, n_results=6)
    if not docs:
        return "No documents found in the knowledge base yet."
    return "\n\n".join(
        f"[House {d['metadata'].get('house_id','?')} | {d['metadata'].get('doc_type','text')}]\n{d['text'][:300]}"
        for d in docs
    )


@tool
def get_house_stats_by_area(
    metric: str = "walk_score",
    threshold: float = 70,
    group_by: str = "city",
    min_houses: int = 2,
) -> str:
    """
    Compute statistics for houses grouped by geographic area.
    Use this for questions like:
      - "What percentage of houses have walk_score >= 70 in each city/metro?"
      - "Which cities have the best average walk/bike/transit scores?"
      - "What is the median price per sqft by city?"

    Parameters
    ----------
    metric     : Column to analyse — walk_score, bike_score, transit_score,
                 price, sqft, hoa_fee, or price_per_sqft (computed)
    threshold  : For percentage queries — count rows where metric >= threshold.
                 Set to 0 to get averages/medians instead.
    group_by   : 'city'  — group by houses.city (always populated, most reliable)
                 'state' — group by houses.state
                 'msa'   — group by census_msa.name via msa_code join
                           NOTE: only works for houses where msa_code is populated.
                           Many Redfin exports don't include CBSA codes, so this
                           may return fewer rows than grouping by city.
    min_houses : Minimum houses in a group to include it (filters out singletons)
    """
    valid_metrics = ["walk_score", "bike_score", "transit_score",
                     "price", "sqft", "hoa_fee", "price_per_sqft"]
    if metric not in valid_metrics:
        return f"Unknown metric '{metric}'. Valid: {', '.join(valid_metrics)}"

    # Build the value expression
    if metric == "price_per_sqft":
        val_expr = "price / NULLIF(sqft, 0)"
        col_label = "price_per_sqft"
    else:
        val_expr = metric
        col_label = metric

    # Build the group expression
    if group_by == "msa":
        group_expr  = "m.name"
        group_label = "metro_area"
        from_clause = "FROM houses h LEFT JOIN census_msa m ON h.msa_code = m.msa_code"
        where_extra = "AND m.name IS NOT NULL"
    elif group_by == "state":
        group_expr  = "h.state"
        group_label = "state"
        from_clause = "FROM houses h"
        where_extra = "AND h.state IS NOT NULL"
    else:
        group_expr  = "h.city"
        group_label = "city"
        from_clause = "FROM houses h"
        where_extra = "AND h.city IS NOT NULL"

    if threshold > 0:
        # Percentage query
        sql = f"""
            SELECT
                {group_expr}                             AS {group_label},
                COUNT(*)                                 AS total_houses,
                SUM(CASE WHEN {val_expr} >= {threshold}
                         THEN 1 ELSE 0 END)              AS above_threshold,
                ROUND(100.0 *
                    SUM(CASE WHEN {val_expr} >= {threshold}
                             THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(*), 0), 1)             AS pct_above_{threshold:.0f}
            {from_clause}
            WHERE {val_expr} IS NOT NULL
              {where_extra}
            GROUP BY {group_expr}
            HAVING COUNT(*) >= {min_houses}
            ORDER BY pct_above_{threshold:.0f} DESC
        """
        header = f"% of houses with {col_label} ≥ {threshold:.0f}, by {group_label}"
    else:
        # Average / stats query
        sql = f"""
            SELECT
                {group_expr}                             AS {group_label},
                COUNT(*)                                 AS total_houses,
                ROUND(AVG({val_expr}), 1)                AS avg_{col_label},
                ROUND(MIN({val_expr}), 1)                AS min_{col_label},
                ROUND(MAX({val_expr}), 1)                AS max_{col_label}
            {from_clause}
            WHERE {val_expr} IS NOT NULL
              {where_extra}
            GROUP BY {group_expr}
            HAVING COUNT(*) >= {min_houses}
            ORDER BY avg_{col_label} DESC
        """
        header = f"Average {col_label} by {group_label}"

    try:
        df = store.query(sql)
    except Exception as e:
        return f"SQL Error: {e}"

    if df.empty:
        if group_by == "msa":
            # Try city as fallback and explain
            return (
                "No results from MSA grouping — `msa_code` is not populated for "
                "these houses (Redfin exports don't include CBSA codes).\n\n"
                "Try: get_house_stats_by_area with group_by='city' instead, "
                "which always works."
            )
        return f"No results. Check that {metric} is populated in the houses table."

    note = ""
    if group_by == "msa":
        n_houses_total = store.query("SELECT COUNT(*) as n FROM houses").iloc[0]["n"]
        n_houses_with_msa = store.query(
            "SELECT COUNT(*) as n FROM houses WHERE msa_code IS NOT NULL"
        ).iloc[0]["n"]
        if n_houses_with_msa < n_houses_total:
            note = (f"\nNote: only {int(n_houses_with_msa)}/{int(n_houses_total)} houses "
                    f"have msa_code populated. For full coverage use group_by='city'.")

    return f"### {header}\n\n{df.to_string(index=False)}{note}"


@tool
def get_database_schema(_: str = "") -> str:
    """Return the database schema so you can write correct SQL queries."""
    return textwrap.dedent("""
    Tables:
    - houses(house_id, address, city, state, zip, lat, lon, status, price,
             beds, baths, sqft, year_built, hoa_fee, walk_score, bike_score,
             transit_score, tract_fips, msa_code, source_file)

      IMPORTANT: houses.msa_code is often NULL — Redfin exports don't include
      CBSA codes. For grouping houses by area ALWAYS use 'city' or 'state':
        GROUP BY city       ← always works
        GROUP BY state      ← always works
        GROUP BY msa_code   ← often returns few/no rows because msa_code is NULL

      There is NO 'msa_name' column in houses. Never use it.

    - nri_tracts(tract_fips, county_fips, state_fips, state_name, county_name,
                 risk_score, risk_ratng, risk_npctl, eal_score, eal_ratng, eal_valt,
                 sovi_score, sovi_ratng, resl_score, resl_ratng,
                 avln_risks, cfld_risks, cwav_risks, drgt_risks, erqk_risks,
                 hail_risks, hwav_risks, hrcn_risks, istm_risks, lnds_risks,
                 ltng_risks, rfld_risks, swnd_risks, trnd_risks, tsun_risks,
                 vlcn_risks, wfir_risks, wntw_risks)

    - census_tracts(tract_fips, geo_id, name, population)
        - census_msa(msa_code, geo_id, name, population)
    - cbsa_counties(cbsa_code, cbsa_title, msa_type,
                    state_fips, county_fips, county_name, state_name)

    - sold_homes(sale_id, source_county, source_file, parid,
                 address, city, state, zip, full_address,
                 lat, lon, geocode_status, geocode_source, geocode_accuracy,
                 sold_price, sold_date, record_date, list_price,
                 sqft, beds, baths, year_built,
                 sale_code, sale_desc, instr_type, instr_type_desc,
                 deed_book, deed_page, muni_code, muni_desc, school_code, school_desc,
                 is_arms_length, arms_length_flag, tract_fips, county_fips)

    Key joins:
    - houses.tract_fips = nri_tracts.tract_fips
    - sold_homes.tract_fips = nri_tracts.tract_fips
    - census_msa.msa_code = cbsa_counties.cbsa_code
    - (cbsa_counties.state_fips || cbsa_counties.county_fips) = nri_tracts.county_fips

    SOLD HOMES: always filter with:
      WHERE (is_arms_length IS NULL OR is_arms_length = TRUE) AND sold_price > 1000
    """)


GENERAL_TOOLS = [
    check_data_availability,
    get_top_msas_by_flood_risk,
    get_house_stats_by_area,
    query_database,
    search_all_house_descriptions,
    get_database_schema,
]
