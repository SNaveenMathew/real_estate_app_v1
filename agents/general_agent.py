"""
General agent — answers broad questions about metro areas, national risk,
portfolio-level analysis across all houses, etc.
"""
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import settings
from agents.tools import GENERAL_TOOLS
import db.duckdb_store as store
import re as _re


SYSTEM_PROMPT = """You are a real estate and risk analyst assistant backed by a local database.

━━━ ANTI-HALLUCINATION RULES — FOLLOW THESE WITHOUT EXCEPTION ━━━

1. ALWAYS call check_data_availability at the start of every conversation turn.
2. If a table is EMPTY (0 rows), tell the user — do NOT invent data.
3. If a query returns 0 rows, stop and report it. Never fabricate results.
4. All numbers and rankings must come from tool output only.

━━━ TOOL ROUTING — MANDATORY ━━━

For walk score / bike score / transit score / price stats BY AREA:
  → ALWAYS use get_house_stats_by_area
  → Parameters: metric='walk_score', threshold=70, group_by='city'
  → Use group_by='city' by default — msa_code is often NULL in houses

For MSA flood / risk / hazard ranking questions:
  → ALWAYS use get_top_msas_by_flood_risk

For custom SQL:
  → Use query_database, but call get_database_schema first
  → NEVER use 'msa_name' — it does not exist
  → NEVER GROUP BY msa_code — it is NULL for most houses
  → GROUP BY city or state instead

━━━ CRITICAL: houses.msa_code IS USUALLY NULL ━━━

Redfin exports do not include CBSA codes. houses.msa_code is NULL for most rows.
Any query that JOINs houses to census_msa or groups by msa_code will return
zero or very few results. ALWAYS use city or state for grouping houses.

━━━ CORRECT SQL PATTERNS ━━━

Group houses by city (CORRECT):
  SELECT city, COUNT(*), AVG(walk_score) FROM houses GROUP BY city

Group houses by metro area (WRONG — msa_code is NULL):
  SELECT msa_code, ... FROM houses GROUP BY msa_code  ← returns 0 rows

Join nri_tracts to cbsa_counties:
  FROM cbsa_counties cb
  JOIN nri_tracts n ON (cb.state_fips || cb.county_fips) = n.county_fips

Sort population correctly (stored as VARCHAR):
  ORDER BY CAST(population AS BIGINT) DESC

━━━ FORMATTING ━━━

- Dollar amounts: $1,234,567
- Scores and percentages: one decimal place
- Use markdown tables for comparisons
"""


def _get_data_availability_context() -> str:
    """Build a live data availability summary to prepend to every system message."""
    tables = ["houses", "nri_tracts", "census_tracts",
              "census_msa", "cbsa_counties", "sold_homes"]
    lines = ["[LIVE DATABASE STATUS]"]
    for t in tables:
        try:
            n = int(store.query(f"SELECT COUNT(*) as n FROM {t}").iloc[0]["n"])
            status = "LOADED" if n > 0 else "EMPTY — do not query"
            lines.append(f"  {t:<20} {status} ({n:,} rows)")
        except Exception:
            lines.append(f"  {t:<20} unknown")
    lines.append("[END STATUS]\n")
    return "\n".join(lines)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: list(x) + list(y)]


def build_general_agent():
    tool_node = ToolNode(GENERAL_TOOLS)

    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.0,
    ).bind_tools(GENERAL_TOOLS)

    def agent_node(state: AgentState):
        # Inject live data availability into every system message so the model
        # always knows what's loaded without needing to call the tool first
        availability = _get_data_availability_context()
        system = SystemMessage(content=f"{availability}\n{SYSTEM_PROMPT}")
        messages = [system] + list(state["messages"])
        response = llm.invoke(messages)
        return {"messages": [response]}

    def router(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# Singleton — rebuilt if you call invalidate_general_agent()
_general_agent = None


def get_general_agent():
    global _general_agent
    if _general_agent is None:
        _general_agent = build_general_agent()
    return _general_agent


def invalidate_general_agent():
    """Call after loading new data so the agent picks up updated availability."""
    global _general_agent
    _general_agent = None


def run_general_chat(message: str, history: list[dict] = None) -> tuple[str, list[dict]]:
    # ── Direct bypasses — call tools directly for well-defined question types ─
    for bypass_fn in [_try_direct_msa_answer, _try_direct_score_by_area]:
        direct_reply = bypass_fn(message)
        if direct_reply is not None:
            updated_history = (history or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": direct_reply},
            ]
            return direct_reply, updated_history

    # ── Normal agent path ─────────────────────────────────────────────────
    agent = get_general_agent()

    lc_messages: list[BaseMessage] = []
    for h in (history or []):
        if h["role"] == "user":
            lc_messages.append(HumanMessage(content=h["content"]))
        else:
            lc_messages.append(AIMessage(content=h["content"]))
    lc_messages.append(HumanMessage(content=message))

    result = agent.invoke({"messages": lc_messages})
    all_messages = list(result["messages"])

    ai_messages = [m for m in all_messages if isinstance(m, AIMessage)]
    raw_reply = ai_messages[-1].content if ai_messages else "I couldn't generate a response."

    # ── Validate: replace hallucinated data tables with honest errors ─────
    from agents.response_validator import validate_response
    reply = validate_response(raw_reply, all_messages, strict=True)

    if reply != raw_reply:
        import logging
        logging.getLogger(__name__).warning(
            "Hallucination detected and blocked in general agent response."
        )

    updated_history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return reply, updated_history


# ── Patterns that trigger the direct bypass ───────────────────────────────────

_MSA_FLOOD_PATTERNS = [
    r"top\s+\d+\s+ms[a]",
    r"ms[a]s?.*(flood|risk|hurricane|tornado|wildfire|earthquake)",
    r"(flood|risk|hazard|disaster).*(ms[a]|metro|city|cities)",
    r"lowest.*(flood|risk)",
    r"safest.*(metro|city|cities|ms[a])",
    r"which.*(metro|ms[a]).*(risk|flood|safe)",
]

# Patterns for score/metric by area questions
_SCORE_BY_AREA_PATTERNS = [
    r"(walk|bike|transit)\s*score.*(by|per|each|every).*(city|area|metro|zip|neighborhood)",
    r"(by|per|each|every).*(city|area|metro).*(walk|bike|transit)\s*score",
    r"percentage.*(walk|bike|transit|score)",
    r"(walk|bike|transit).*score.*>=?\s*\d+",
    r"which.*(city|cities|area).*(walk|walkable|bike|transit)",
    r"most\s+walkable",
    r"best\s+(walk|bike|transit)\s*score",
]

_SCORE_METRIC_MAP = {
    "walk":    "walk_score",
    "bike":    "bike_score",
    "transit": "transit_score",
}

_THRESHOLD_RE = _re.compile(r'>=?\s*(\d+)')

_HAZARD_KEYWORDS = {
    "flood":     "rfld_risks",
    "riverine":  "rfld_risks",
    "coastal":   "cfld_risks",
    "hurricane": "hrcn_risks",
    "tornado":   "trnd_risks",
    "wildfire":  "wfir_risks",
    "fire":      "wfir_risks",
    "earthquake":"erqk_risks",
    "wind":      "swnd_risks",
    "hail":      "hail_risks",
    "heat":      "hwav_risks",
    "drought":   "drgt_risks",
    "risk":      "risk_score",   # composite
    "overall":   "risk_score",
}


def _try_direct_msa_answer(message: str) -> str | None:
    """
    If the message matches an MSA risk pattern, call get_top_msas_by_flood_risk
    directly and return the result. Returns None if the message doesn't match,
    meaning the normal agent path should run instead.
    """
    msg_lower = message.lower()

    # Check if message is about MSA risk/flood rankings
    if not any(_re.search(p, msg_lower) for p in _MSA_FLOOD_PATTERNS):
        return None

    # Detect which hazard they're asking about
    hazard = "rfld_risks"   # default to riverine flood
    for keyword, col in _HAZARD_KEYWORDS.items():
        if keyword in msg_lower:
            hazard = col
            break

    # Detect how many MSAs they want
    top_n_match = _re.search(r"top\s+(\d+)", msg_lower)
    top_n = int(top_n_match.group(1)) if top_n_match else 50
    top_n = min(top_n, 100)   # cap for safety

    show_n_match = _re.search(r"show\s+(\d+)|bottom\s+(\d+)|lowest\s+(\d+)", msg_lower)
    show_n = int(next(g for g in show_n_match.groups() if g) ) if show_n_match else 10

    # Call the tool directly (it handles empty census_msa gracefully)
    from agents.tools import get_top_msas_by_flood_risk
    result = get_top_msas_by_flood_risk.invoke({
        "top_n_by_population": top_n,
        "show_lowest_n": show_n,
        "hazard": hazard,
    })

    # Wrap in a brief explanatory prefix
    hazard_label = {
        "rfld_risks": "riverine flood", "cfld_risks": "coastal flood",
        "hrcn_risks": "hurricane",      "trnd_risks": "tornado",
        "wfir_risks": "wildfire",       "erqk_risks": "earthquake",
        "swnd_risks": "strong wind",    "risk_score": "overall composite",
    }.get(hazard, hazard)

    prefix = (
        f"Here are the MSAs with the **lowest {hazard_label} risk** "
        f"among the top {top_n} by population:\n\n"
    )
    return prefix + result


def _try_direct_score_by_area(message: str) -> str | None:
    """
    Direct bypass for walk/bike/transit score questions grouped by city/area.
    Calls get_house_stats_by_area directly so the LLM never writes the SQL.
    Returns None if the message doesn't match.
    """
    import re as _re
    msg_lower = message.lower()

    if not any(_re.search(p, msg_lower) for p in _SCORE_BY_AREA_PATTERNS):
        return None

    # Detect metric
    metric = "walk_score"
    for kw, col in _SCORE_METRIC_MAP.items():
        if kw in msg_lower:
            metric = col
            break

    # Detect threshold
    threshold = 70.0
    m = _THRESHOLD_RE.search(message)
    if m:
        threshold = float(m.group(1))

    # Detect group_by — default city
    group_by = "city"
    if "state" in msg_lower:
        group_by = "state"
    elif any(w in msg_lower for w in ("metro", "metropolitan", "msa")):
        group_by = "msa"

    from agents.tools import get_house_stats_by_area
    result = get_house_stats_by_area.invoke({
        "metric": metric,
        "threshold": threshold,
        "group_by": group_by,
        "min_houses": 2,
    })

    # If MSA grouping returned nothing, automatically retry with city
    if "msa_code is not populated" in result or "zero or very few" in result:
        result_city = get_house_stats_by_area.invoke({
            "metric": metric,
            "threshold": threshold,
            "group_by": "city",
            "min_houses": 2,
        })
        return (f"*(Note: grouping by metro area isn't possible because "
                f"`msa_code` is not populated for these Redfin houses. "
                f"Showing by city instead.)*\n\n{result_city}")

    label = metric.replace("_", " ").title()
    prefix = f"Here is the **{label} ≥ {threshold:.0f}** breakdown by {group_by}:\n\n"
    return prefix + result
