"""
LangChain tools shared by house agent and general agent.
"""
import json
import traceback
from typing import Any

import pandas as pd
from langchain.tools import tool

import db.duckdb_store as store
import db.vector_store as vs
import db.schema_catalog as schema


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
        # Keep every field, even when the value is None — a key that's
        # missing outright is easy to misread as "not asked for" rather
        # than "not available for this house." An explicit
        # `"walk_score": null` is unambiguous; a silently absent key isn't.
        return json.dumps(house, indent=2)

    @tool
    def get_nri_risk_data(_: str = "") -> str:
        """Get FEMA National Risk Index data for this house's census tract."""
        house = store.get_house(house_id)
        if not house or not house.get("tract_fips"):
            return "Census tract not available for this house."
        nri = store.get_nri_for_tract(house["tract_fips"])
        if not nri:
            return f"No NRI data found for tract {house['tract_fips']}."
        # Return a human-readable summary. Hazard labels come from the shared
        # catalog (db/schema_catalog.py) so this stays in sync with the schema
        # tool the general agent uses instead of keeping its own copy.
        hazards = {label: nri.get(col) for col, label in schema.NRI_HAZARD_COLUMNS.items()}
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
    Row count for every table, so you know what is actually loaded.
    A live snapshot of this is already included in your system context each
    turn — you don't need to call this proactively. It's here in case you
    want to double-check after data may have been reloaded mid-conversation.
    If a table has 0 rows, you CANNOT answer questions that depend on it —
    tell the user which files need to be loaded instead.
    """
    report, _ = schema.availability_report()
    return report


@tool
def query_database(sql: str) -> str:
    """
    Execute a SELECT query against the real estate DuckDB database.
    Call get_database_schema first if you haven't already this conversation —
    it documents live columns, which ones are unreliable/NULL-heavy, and the
    exact join expressions for tables that don't share an obvious key.
    Do NOT guess or fabricate results if the query returns empty — report it.
    """
    sql_upper = sql.upper().strip()
    if any(kw in sql_upper for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]):
        return "ERROR: Only SELECT queries are allowed."
    try:
        df = store.query(sql)
        if len(df) == 0:
            diagnosis = schema.diagnose_empty_or_error(sql)
            return diagnosis if diagnosis else "Query returned 0 rows. The data may not match your filter criteria."
        if len(df) > 50:
            return df.head(50).to_string(index=False) + f"\n... ({len(df)} total rows, showing 50)"
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {e}\nCheck column names and joins with get_database_schema."


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
def get_database_schema(_: str = "") -> str:
    """
    Return the live database schema: every table's actual columns (introspected
    from the running database, so this can't go stale), notes on data-quality
    quirks (which columns are unreliable/NULL-heavy, valid value ranges), and
    the documented join path for every pair of tables that don't share an
    obvious key. Call this before writing SQL against a table you haven't
    already queried successfully earlier in this conversation.
    """
    return schema.render_schema_for_agent()


GENERAL_TOOLS = [
    check_data_availability,
    get_database_schema,
    query_database,
    search_all_house_descriptions,
]
