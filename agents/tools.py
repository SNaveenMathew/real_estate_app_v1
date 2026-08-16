"""
LangChain tools shared by house agent and general agent.
"""
import json
import re
import traceback
from typing import Any, Optional

import pandas as pd
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
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


_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|COPY|EXPORT|ATTACH|DETACH|INSTALL|LOAD|CALL|PRAGMA)\b",
    re.I,
)
_TABLE_REF = re.compile(
    r"\b(?:FROM|JOIN|UPDATE|INTO|TABLE)\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)",
    re.I,
)

def _live_agent_schema() -> str:
    # Reuse the same live-schema renderer exposed to General Chat. This keeps
    # the sub-agent aligned with real column names and documented joins.
    return schema.render_schema_for_agent()


SYSTEM_PROMPT = """You are the shared SQL Code Agent in a real-estate analytics application.

Your job is to generate ONE safe DuckDB SELECT statement for General Chat.
General Chat is responsible for interpretation, multi-step planning, and the
final user-facing answer. You are responsible for translating its analytical
need into SQL against the live database schema below.

Rules:
1. Return SQL only. No markdown fences, no explanation, no prose.
2. Only SELECT statements, or WITH ... SELECT statements, are allowed.
3. Use only tables listed in the LIVE SCHEMA. Never invent tables or columns.
4. Prefer the documented relationships and column notes over guessed joins.
5. Honor table-specific default filters when the request is about valid/market data.
6. Use explicit aliases and deterministic ORDER BY clauses where rankings are involved.
7. Respect requested city/state/metro/time/price/risk filters exactly.
8. For questions asking for a count, count the requested unit (incidents, houses,
   routes, tracts, sales, etc.), not an unrelated join multiplication.
9. For comparisons/rankings, return enough fields for General Chat to explain the
   result, but keep the result set reasonably small with LIMIT when appropriate.
10. If the request combines multiple datasets, use documented joins or CTEs.
11. Do not parse serialized geometry/blobs in SQL when metadata says they are intended
   for map rendering only.
12. Never use SELECT * when a narrower projection can answer the request.
13. If the request is ambiguous, prefer the interpretation explicitly stated in the
   General Chat requirements/plan rather than inventing a new scope.

The query must answer the stated analytical request directly; do not write a generic
schema-inspection query unless General Chat explicitly asks for metadata.

LIVE SCHEMA
===========
""" + _live_agent_schema()


def _extract_content(resp) -> str:
    content = resp.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
            else:
                chunks.append(str(item))
        return "".join(chunks).strip()
    return str(content).strip()


def _clean_sql(raw: str) -> str:
    sql = raw.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.I)
    sql = re.sub(r"\s*```$", "", sql)
    if ";" in sql:
        # One statement is required; reject multiple statements instead of trying
        # to guess which one the model intended.
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        if len(statements) != 1:
            raise ValueError("Code agent returned multiple SQL statements.")
        sql = statements[0]
    return sql.strip()


def validate_sql(sql: str) -> str:
    if not sql:
        raise ValueError("Code agent returned empty SQL.")
    if _FORBIDDEN.search(sql):
        raise ValueError("Only read-only SELECT SQL is allowed.")
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, flags=re.I):
        raise ValueError("Generated SQL must begin with SELECT or WITH.")

    refs = {m.group(1).lower() for m in _TABLE_REF.finditer(sql)}
    allowed = set(schema.list_table_names(agent_visible_only=True))
    unknown = refs - allowed
    if unknown:
        raise ValueError(f"Code agent referenced unknown table(s): {', '.join(sorted(unknown))}")
    if not refs:
        raise ValueError("Generated SQL does not reference a known agent-visible table.")
    return sql


_agent: Optional[ChatOpenAI] = None


def get_code_agent() -> ChatOpenAI:
    global _agent
    if _agent is None:
        _agent = ChatOpenAI(
            base_url=settings.llama_server_base_url,
            api_key="not-needed",
            model=settings.llama_server_model,
            temperature=0.0,
        )
    return _agent


def generate_sql(request: str, requirements: str = "", plan: str = "") -> str:
    """Generate and validate a SELECT from a request and optional General Chat plan."""
    prompt_parts = [f"USER REQUEST:\n{request.strip()}"]
    if requirements.strip():
        prompt_parts.append(f"GENERAL CHAT REQUIREMENTS:\n{requirements.strip()}")
    if plan.strip():
        prompt_parts.append(f"GENERAL CHAT PLAN:\n{plan.strip()}")
    prompt_parts.append("Generate the single best SQL query now.")

    response = get_code_agent().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(prompt_parts)),
    ])
    sql = validate_sql(_clean_sql(_extract_content(response)))
    return sql


def run_code_query(request: str, requirements: str = "", plan: str = "") -> tuple[str, str]:
    """Generate SQL with the code agent, execute it, and return (sql, result_text)."""
    sql = generate_sql(request, requirements=requirements, plan=plan)
    try:
        df = store.query(sql)
    except Exception as exc:
        raise RuntimeError(f"Generated SQL failed: {exc}\nSQL: {sql}") from exc

    if len(df) == 0:
        diagnosis = schema.diagnose_empty_or_error(sql)
        return sql, diagnosis or "Query returned 0 rows."

    if len(df) > 50:
        return sql, df.head(50).to_string(index=False) + f"\n... ({len(df)} total rows, showing 50)"
    return sql, df.to_string(index=False)

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
def query_database(request: str, requirements: str = "", plan: str = "") -> str:
    """
    Use the shared SQL Code Agent to answer analytical questions from any
    agent-visible DuckDB dataset.

    Pass the user's analytical request plus any explicit General Chat
    requirements or multi-step plan. The Code Agent uses the live schema
    metadata and documented relationships to generate one read-only SELECT,
    executes it, and returns both the generated SQL and live result so General
    Chat can inspect the evidence and continue thinking/planning before it
    answers the user.
    """
    try:
        sql, result = run_code_query(
            request, requirements=requirements, plan=plan
        )
    except Exception as exc:
        return f"Code Agent error: {exc}"
    return f"[GENERATED SQL]\n{sql}\n[RESULT]\n{result}"


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
