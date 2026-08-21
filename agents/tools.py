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

from config import settings, LLM_STOP_SEQUENCES
import db.duckdb_store as store
import db.vector_store as vs
import db.schema_catalog as schema
import services.bike_routing as bike_routing
from agents.query_planner import build_query_plan
import asyncio
import threading


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


SYSTEM_PROMPT = """You are the SQL Code Agent for a real-estate analytics application.

Generate exactly ONE safe DuckDB SELECT statement (WITH ... SELECT allowed).
The query must directly answer the user's analytical request.

Rules:
1. Use ONLY physical tables and columns present in the TARGETED LIVE DATA MODEL below.
2. Do not invent tables, columns, joins, or metrics.
3. Prefer documented relationship paths and honor their cardinality / fanout warnings.
4. Respect semantic aliases and canonical columns supplied by the metadata.
5. For rankings, make the ranking universe explicit before applying the requested LIMIT.
6. When a request says 'top N MSAs', first determine the MSA universe by population, then aggregate the requested risk metric over those MSAs; do not limit tracts before the MSA population ranking.
7. For NRI tract-to-MSA analysis, use census_msa -> cbsa_counties -> nri_tracts. Do not invent a direct MSA-to-tract join.
8. Preserve the requested hazard exactly (e.g. riverine flooding vs coastal flooding vs overall risk).
9. Keep the output small and useful: return the identity/grouping fields plus the requested metric, with deterministic ORDER BY.
10. Do not parse geometry/blob columns in SQL.
11. Output ONLY SQL — no markdown fences or explanation.

TARGETED LIVE DATA MODEL
========================
"""



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
            # Tight ceiling on purpose: this agent's whole job is one bare SQL
            # statement (see SYSTEM_PROMPT above), so a well-formed reply is at
            # most a couple hundred tokens. Without a cap, a call that doesn't
            # hit a recognized stop token keeps decoding indefinitely instead
            # of returning — this is what actually happened (see config.py's
            # "LLM generation limits" comment): one such call ran to 24k+
            # tokens and ~5.5 minutes before being cut off at the server's
            # context limit.
            max_tokens=settings.code_agent_max_tokens,
            timeout=settings.llm_request_timeout,
            stop=LLM_STOP_SEQUENCES,
        )
    return _agent


def generate_sql(request: str, requirements: str = "", plan: str = "") -> str:
    """Generate and validate a SELECT using mandatory structured planning + metadata grounding."""
    query_plan = build_query_plan(request, requirements=requirements, plan=plan)
    targeted = schema.build_query_context(request, requirements=requirements, plan=plan)
    prompt_parts = [
        f"USER REQUEST:\n{request.strip()}",
        "STRUCTURED QUERY PLAN:\n" + query_plan.render(),
        f"TARGETED DATA MODEL RETRIEVAL:\n{targeted}",
    ]
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
    """Generate, execute, and (when needed) repair a SQL query.

    The repair remains inside the SQL Code Agent contract: the model receives
    the actual zero-row/join diagnosis and generates a corrected SELECT.
    This prevents General Chat from having to hard-code dataset-specific SQL.
    """
    last_sql = ""
    repair_note = ""

    for attempt in range(2):
        effective_plan = plan
        if repair_note:
            effective_plan = (
                (plan + "\n\n") if plan.strip() else ""
            ) + "SQL REPAIR CONTEXT FROM THE PREVIOUS EXECUTION:\n" + repair_note

        sql = generate_sql(
            request,
            requirements=requirements,
            plan=effective_plan,
        )
        last_sql = sql

        try:
            df = store.query(sql)
        except Exception as exc:
            if attempt == 0:
                repair_note = (
                    f"The previous generated SQL failed to execute with this error: {exc}\n"
                    f"Previous SQL:\n{sql}\n"
                    "Generate a corrected SELECT using only the live schema and documented joins."
                )
                continue
            raise RuntimeError(f"Generated SQL failed: {exc}\nSQL: {sql}") from exc

        if len(df) == 0:
            diagnosis = schema.diagnose_empty_or_error(sql) or "Query returned 0 rows."
            if attempt == 0:
                repair_note = (
                    f"The previous generated SQL returned no rows.\n"
                    f"Diagnostic evidence:\n{diagnosis}\n"
                    f"Previous SQL:\n{sql}\n"
                    "Generate a corrected SELECT that still answers the original user request."
                )
                continue
            return sql, diagnosis

        if len(df) > 50:
            return sql, df.head(50).to_string(index=False) + f"\n... ({len(df)} total rows, showing 50)"
        return sql, df.to_string(index=False)

    return last_sql, "Query could not be completed."

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
def retrieve_data_model_context(query: str) -> str:
    """Mandatory grounding retrieval: plan + targeted live schema + vector metadata."""
    structured_plan = build_query_plan(query)
    retrieval_query = query + "\n" + structured_plan.render()
    parts = [
        "[STRUCTURED QUERY PLAN]",
        structured_plan.render(),
        "[TARGETED LIVE DATA MODEL]",
        schema.build_query_context(query, plan=structured_plan.render()),
    ]
    try:
        docs = vs.search_data_model(retrieval_query, n_results=10)
        if docs:
            parts.append("[SEMANTIC METADATA RETRIEVAL]")
            parts.extend(d["text"] for d in docs)
    except Exception as exc:
        parts.append(f"[VECTOR METADATA FALLBACK] unavailable: {exc}")
    return "\n\n".join(p for p in parts if p)

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
    structured_plan = build_query_plan(request, requirements=requirements, plan=plan)
    effective_plan = ((plan + "\n\n") if plan.strip() else "") + "STRUCTURED QUERY PLAN:\n" + structured_plan.render()
    try:
        sql, result = run_code_query(
            request, requirements=requirements, plan=effective_plan
        )
    except Exception as exc:
        sql, result = "", f"Code Agent error: {exc}"

    # LLM-first architecture: only after generation/repair fails do we use the
    # metadata-defined canonical recovery for the well-known MSA->county->NRI
    # ranking shape. This does NOT create a view and does NOT bypass planning.
    recovery_sql = None
    lower_result = str(result).lower()
    if ("0 rows" in lower_result or "returned no rows" in lower_result or
            "code agent error" in lower_result):
        recovery_sql = schema.canonical_nri_msa_query(request)
    if recovery_sql:
        try:
            recovered = store.query(recovery_sql)
            if len(recovered) > 0:
                return (
                    f"[GENERATED SQL]\n{sql or '[LLM query failed/returned no rows]'}\n"
                    f"[CANONICAL RECOVERY SQL]\n{recovery_sql}\n[RESULT]\n"
                    f"{recovered.head(50).to_string(index=False)}"
                    + (f"\n... ({len(recovered)} total rows, showing 50)" if len(recovered) > 50 else "")
                )
            result = (str(result) + "\nCanonical recovery query also returned 0 rows.").strip()
        except Exception as rec_exc:
            result = (str(result) + f"\nCanonical recovery failed: {rec_exc}").strip()

    return f"[GENERATED SQL]\n{sql}\n[RESULT]\n{result}"


def _run_async_safely(async_fn, *args, **kwargs):
    """Run an async function from sync code or from an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_fn(*args, **kwargs))

    result = []
    error = []

    def runner():
        try:
            result.append(asyncio.run(async_fn(*args, **kwargs)))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if error:
        raise error[0]
    if not result:
        raise RuntimeError("Async worker returned no result.")
    return result[0]


@tool
def find_bike_route(
    start: str,
    end: str,
    city: str = "Pittsburgh, PA",
    avoid_crime_dense_areas: bool = False,
    crime_density_percentile: float = 90.0,
) -> str:
    """Find a bicycle route between two places.

    Places may be neighborhoods, landmarks, parks, addresses, or coordinates.
    For requests that explicitly ask to avoid crime-dense/high-crime areas, set
    ``avoid_crime_dense_areas=True``. The routing layer will filter graph edges
    that intersect the highest-density local crime cells BEFORE Dijkstra runs.
    ``crime_density_percentile`` controls how selective that exclusion is; the
    default 90 means only the top 10% of occupied crime-density cells are
    excluded. This is a route-avoidance heuristic, not a safety guarantee.
    """
    try:
        result = _run_async_safely(
            bike_routing.route_bike,
            start.strip(),
            end.strip(),
            city=(city or "Pittsburgh, PA").strip(),
            avoid_crime_dense_areas=bool(avoid_crime_dense_areas),
            crime_density_percentile=float(crime_density_percentile),
        )
        # A routing failure/no-route result intentionally uses route=None.
        # Never call .get() on that None value; normalize it to an empty dict
        # so the structured no-route response can be returned to the caller.
        if not isinstance(result, dict):
            raise TypeError(f"Bike routing returned an unexpected result type: {type(result).__name__}")
        route_data = result.get("route") or {}
        summary = route_data.get("summary") or {}
        facilities = route_data.get("local_bike_facilities") or {}
        instructions = []
        for maneuver in (route_data.get("maneuvers") or [])[:12]:
            instruction = maneuver.get("instruction") or maneuver.get("verbal_pre_transition_instruction")
            if instruction:
                instructions.append(instruction)
        if result.get("no_route"):
            note = result.get("note") or "No continuous path exists using the filtered BikePGH network."
            if result.get("crime_filter_error"):
                message = f"No — Not possible: {note}"
            else:
                message = f"No — Not possible: no continuous bike path exists using the filtered BikePGH network. {note}"
            return json.dumps({
                "status": "analysis",
                "kind": "no_route",
                "message": message,
                "start": result.get("start"),
                "end": result.get("end"),
                "city": result.get("city") or city or "Pittsburgh, PA",
                "crime_avoidance": result.get("crime_avoidance") or {"enabled": False},
                "analysis_visualization": result.get("analysis_visualization"),
            }, indent=2)

        crime_meta = result.get("crime_avoidance") or {"enabled": False, "applied": False}
        success_message = (
            "Yes — a continuous BikePGH route exists after filtering out the selected high-density crime areas."
            if bool(crime_meta.get("enabled")) and bool(crime_meta.get("applied"))
            else "Yes — a continuous BikePGH route was found."
        )
        return json.dumps({
            "status": "ok",
            "kind": "route_found",
            "message": success_message,
            "presentation": "route_map",
            "start": result.get("start"),
            "end": result.get("end"),
            "city": result.get("city") or city or "Pittsburgh, PA",
            "provider": result.get("provider"),
            "distance_miles": round(float(summary.get("length", 0) or 0), 2),
            "duration_minutes": round(float(summary.get("time", 0) or 0) / 60.0, 1),
            "bike_facility_overlap_percent": facilities.get("facility_overlap_pct", 0.0),
            "bike_infrastructure_near_route": facilities.get("facility_segments", []),
            "used_infrastructure": route_data.get("used_infrastructure") or {"type": "FeatureCollection", "features": []},
            "alternatives_considered": result.get("alternatives_considered", 1),
            "crime_avoidance": result.get("crime_avoidance") or {"enabled": False},
            "analysis_visualization": result.get("analysis_visualization"),
            "turn_by_turn": instructions,
            "route_shape": route_data.get("shape", []) or [],
            "bbox": route_data.get("bbox"),
            "attribution": result.get("attribution"),
            "note": result.get("note"),
        }, indent=2)
    except ValueError as exc:
        message = str(exc)
        lower = message.lower()
        if "no bikepgh infrastructure data is loaded" in lower or "no routable line geometry" in lower:
            kind = "no_data"
        elif "no continuous path exists" in lower or "unable to split" in lower:
            kind = "no_route"
        else:
            kind = "input"
        return json.dumps({"status": "error", "kind": kind, "message": message,
                           "start": start, "end": end, "city": city or "Pittsburgh, PA"})
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "kind": "routing_service",
            "message": str(exc),
            "start": start,
            "end": end,
            "city": city or "Pittsburgh, PA",
        })


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
    retrieve_data_model_context,
    find_bike_route,
    search_all_house_descriptions,
]
