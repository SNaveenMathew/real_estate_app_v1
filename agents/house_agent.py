"""
House Chat implemented as a Code Agent.

Architecture (mirrors agents/general_agent.py):
    user request
        -> InputGuardrail screening
        -> LLM generates a small, declarative Python program over house-specific functions
        -> AST validator (CodeAgentGuardrail) permits only approved function calls
        -> deterministic application functions execute the program
        -> execution evidence is returned to the LLM
        -> LLM produces the final grounded response
        -> OutputGroundingGuardrail enforces score bounds and hallucination guard

The LLM decides WHAT data is needed and WHICH approved house functions to call.
The application owns execution, sandboxing, and security -- no rule-based routing.
"""
from __future__ import annotations

import ast
import inspect
import json
import re
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from config import settings, LLM_STOP_SEQUENCES
from agents.response_validator import validate_response


# -- Constants ----------------------------------------------------------------

HOUSE_CODE_AGENT_MAX_STEPS = 3
HOUSE_CODE_AGENT_MAX_CHARS = 12000
HOUSE_FINAL_RESPONSE_MAX_CHARS = 16000


# -- Approved house functions (house_id bound via closure) --------------------

def make_house_approved_functions(house_id: str) -> dict:
    """
    Build the set of approved functions for the house code agent.
    All callables are deterministic database or vector-store lookups.
    """
    import db.duckdb_store as store
    import db.vector_store as vs
    import db.schema_catalog as schema
    from agents.tools import query_database as general_query_database

    def get_house_details() -> str:
        """Return all structured fields for this house."""
        house = store.get_house(house_id)
        if not house:
            return "House not found in database."
        house.pop("raw_json", None)
        return json.dumps(house, indent=2)

    def get_nri_risk_data() -> str:
        """Return FEMA NRI data for this house's census tract."""
        house = store.get_house(house_id)
        if not house or not house.get("tract_fips"):
            return "Census tract not available for this house."
        nri = store.get_nri_for_tract(house["tract_fips"])
        if not nri:
            return f"No NRI data found for tract {house['tract_fips']}."
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

    def estimate_price_with_code() -> str:
        """Compute price estimates using comparable data from DuckDB."""
        import pandas as pd
        house = store.get_house(house_id)
        if not house:
            return "House not found."
        tract = house.get("tract_fips")
        sqft  = house.get("sqft")
        price = house.get("price")
        city  = house.get("city")
        lines = []
        if tract:
            stats = store.get_price_stats_in_tract(tract)
            if stats.get("count", 0) > 0:
                lines.append("### Comparable Active Listings (same census tract)")
                lines.append(f"  Count: {int(stats['count'])}")
                lines.append(f"  Median list price: ${stats.get('median_price', 0):,.0f}")
                lines.append(f"  Avg list price: ${stats.get('avg_price', 0):,.0f}")
                if stats.get("median_price_per_sqft") and sqft:
                    ppsf = stats["median_price_per_sqft"]
                    lines.append(f"  Median price/sqft: ${ppsf:,.2f}")
                    lines.append(f"  Estimated value (median $/sqft x {sqft:.0f} sqft): ${ppsf * sqft:,.0f}")
            if stats.get("sold_sold_count", 0) > 0:
                lines.append("\n### Recent Sales (arm's-length, same census tract)")
                lines.append(f"  Sold count: {int(stats['sold_sold_count'])}")
                lines.append(f"  Median sold price: ${stats.get('sold_median_sold', 0):,.0f}")
                if stats.get("sold_median_sold_per_sqft") and sqft:
                    lines.append(f"  Estimated value (sold $/sqft x {sqft:.0f} sqft): ${stats['sold_median_sold_per_sqft'] * sqft:,.0f}")
                lines.append(f"  Sale date range: {stats.get('sold_oldest_sale')} to {stats.get('sold_newest_sale')}")
        if price:
            lines.append(f"\n### Current List Price: ${price:,.0f}")
            if sqft:
                lines.append(f"  Price/sqft: ${price/sqft:,.2f}")
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
        return "\n".join(lines) if lines else "Insufficient data for price estimation."

    def get_nearby_sold_homes() -> str:
        """Return recent sold comparables in the same census tract."""
        import pandas as pd
        house = store.get_house(house_id)
        if not house or not house.get("tract_fips"):
            return "No tract data available."
        sold = store.get_sold_in_tract(house["tract_fips"])
        if not sold:
            return "No sold homes found in this census tract yet."
        df = pd.DataFrame(sold[:10])
        cols = [c for c in ["address", "sold_price", "sqft", "beds", "baths", "sold_date", "arms_length_flag"] if c in df.columns]
        return df[cols].to_string(index=False)

    def search_house_documents(query: str) -> str:
        """Search stored descriptions and documents for this house."""
        docs = vs.search_house(house_id, query, n_results=4)
        if not docs:
            return "No documents found for this house yet."
        return "\n\n".join(
            f"[{d['metadata'].get('doc_type', 'text')}]\n{d['text']}"
            for d in docs
        )

    def query_database(request: str) -> str:
        """Run a read-only analytical query scoped to this house."""
        scoped_request = (
            f"{request}\n\nThis is House Chat for house_id={house_id!r}. "
            "Restrict house-specific results to this house."
        )
        return general_query_database.invoke({
            "request": scoped_request,
            "requirements": f"Keep the result scoped to house_id={house_id!r}.",
        })

    def get_commute_info() -> str:
        """Estimated commute from this house to the user's saved work location (free-flow; no traffic)."""
        from services import commute
        return commute.describe_for_chat(house_id)

    def get_linked_dataset_records(dataset: str = "") -> str:
        """Records from user-added datasets that link to this house through approved catalog relationships."""
        from services.house_links import linked_records
        return linked_records(house_id, dataset)

    return {
        "get_house_details": get_house_details,
        "get_nri_risk_data": get_nri_risk_data,
        "estimate_price_with_code": estimate_price_with_code,
        "get_nearby_sold_homes": get_nearby_sold_homes,
        "search_house_documents": search_house_documents,
        "query_database": query_database,
        "get_linked_dataset_records": get_linked_dataset_records,
        "get_commute_info": get_commute_info,
    }


# -- AST validation -----------------------------------------------------------

class HouseCodeAgentProgramError(ValueError):
    pass


def _invoke_approved(name: str, function_obj, args: tuple, kwargs: dict):
    """Invoke an approved house function, normalizing positional args."""
    if args:
        fields = list(inspect.signature(function_obj).parameters.keys())
        if len(args) > len(fields):
            raise HouseCodeAgentProgramError(
                f"Too many positional arguments for approved function '{name}'."
            )
        for field, value in zip(fields, args):
            if field in kwargs:
                raise HouseCodeAgentProgramError(
                    f"Argument '{field}' supplied both positionally and by keyword."
                )
            kwargs[field] = value
    return function_obj(**kwargs)


def _validate_house_program(source: str, approved_names: set) -> ast.Module:
    """AST validation: only approved function calls are permitted."""
    if not source:
        raise HouseCodeAgentProgramError("Code agent returned empty code.")
    if len(source) > HOUSE_CODE_AGENT_MAX_CHARS:
        raise HouseCodeAgentProgramError("Generated code exceeded the maximum size.")

    from services.guardrails import CodeAgentGuardrail
    guard_res = CodeAgentGuardrail.validate_code_program(
        source, approved_names, max_chars=HOUSE_CODE_AGENT_MAX_CHARS
    )
    if not guard_res.passed:
        raise HouseCodeAgentProgramError("; ".join(guard_res.reasons))

    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise HouseCodeAgentProgramError(f"Generated code is invalid Python: {exc}") from exc

    allowed_stmt = (ast.Assign, ast.Expr)
    assigned_names: set = set()
    call_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(node, allowed_stmt):
            raise HouseCodeAgentProgramError(
                f"Unsupported statement {type(node).__name__}; only assignments and function calls are allowed."
            )
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute, ast.Subscript, ast.Lambda)):
            raise HouseCodeAgentProgramError(f"Unsupported expression {type(node).__name__}.")
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise HouseCodeAgentProgramError("Assignments must target one simple variable name.")
            assigned_names.add(node.targets[0].id)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in assigned_names and node.id not in approved_names:
                raise HouseCodeAgentProgramError(f"Unknown variable '{node.id}' in generated code.")
        if isinstance(node, ast.Call):
            call_count += 1
            if not isinstance(node.func, ast.Name) or node.func.id not in approved_names:
                raise HouseCodeAgentProgramError("Generated code may call only approved house functions.")
            for kw in node.keywords:
                if kw.arg is None:
                    raise HouseCodeAgentProgramError("**kwargs are not allowed.")
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Call):
            raise HouseCodeAgentProgramError("Only bare function-call expressions are allowed.")
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.FunctionDef, ast.ClassDef)):
            raise HouseCodeAgentProgramError("Control flow and definitions are not allowed in generated code.")

    if call_count == 0:
        raise HouseCodeAgentProgramError("Generated code must call at least one approved house function.")
    return tree


def _execute_house_program(source: str, approved_functions: dict) -> tuple:
    """Validate and execute the generated program; return (final_result, calls)."""
    approved_names = set(approved_functions.keys())
    tree = _validate_house_program(source, approved_names)
    calls: list = []

    namespace: dict = {}
    for name, fn in approved_functions.items():
        def make_wrapper(function_name: str, function_obj):
            def wrapped(*args, **kwargs):
                result = _invoke_approved(function_name, function_obj, args, kwargs)
                calls.append((function_name, result))
                return result
            return wrapped
        namespace[name] = make_wrapper(name, fn)

    namespace["__builtins__"] = {}
    exec(compile(tree, "<house-code-agent>", "exec"), namespace, namespace)

    final_result = namespace.get("final_result")
    if final_result is None and calls:
        final_result = calls[-1][1]
    if final_result is None:
        raise HouseCodeAgentProgramError("Generated code did not produce a result.")
    return final_result, calls


# -- Prompts ------------------------------------------------------------------

_HOUSE_CODE_AGENT_PROMPT = """You are the House Chat Code Agent for a real-estate application.

Your job is to translate the user's question into a SMALL Python program that calls
approved house functions. The application executes the program and returns results.
You are NOT answering the user yet -- you are gathering data.

APPROVED FUNCTIONS
==================
1. get_house_details()
   -> All structured fields: price, beds, baths, sqft, status, walk_score,
     bike_score, transit_score, tract_fips, city, address, etc.

2. get_nri_risk_data()
   -> FEMA NRI composite risk score/rating, EAL, social vulnerability,
     community resilience, and top-5 individual hazard scores for the
     house's census tract.

3. estimate_price_with_code()
   -> Price estimates from arm's-length sold comps and active listings in
     the same census tract, plus nearby active listings in the same city.

4. get_nearby_sold_homes()
   -> Recent sold comparables in the same census tract (up to 10),
     including non-arm's-length flagging.

5. search_house_documents(query: str)
   -> Search stored descriptions, inspection notes, and Redfin/Zillow
     text for this house.

6. query_database(request: str)
    -> Run a read-only analytical query scoped to this house when the other
        functions do not provide the requested computation.

7. get_commute_info()
   -> Estimated commute from this house to the user\'s saved work location:
      drive, bike and walk minutes and miles (free-flow, no traffic) and
      transit minutes when available, plus the work location and how fresh
      the estimate is.

RULES
=====
1. Output ONLY executable Python code. No markdown fences and no explanation.
2. The code may contain only assignments and calls to the approved functions above.
3. Use keyword arguments only (except for no-argument functions).
4. NEVER invent numbers, scores, or facts. Call a function to retrieve data.
5. For walkability/scores: call get_house_details().
   Do NOT state Bike Score or Transit Score unless the tool returns non-NULL values.
6. For NRI/risk/hazard: call get_nri_risk_data().
7. For price estimation: call estimate_price_with_code().
8. For comp/sold home: call get_nearby_sold_homes().
9. For stored descriptions/documents: call search_house_documents(query=...).
10. For custom house-specific analysis: call query_database(request=...).
11. For commute, travel time or distance to work: call get_commute_info().
12. You may call multiple functions when the question needs multiple sources.
13. Set `final_result` to the most relevant result for the response model.

EXAMPLES
========
User: What is this house's Walk Score?
Code:
final_result = get_house_details()

User: How long is the commute to work?
Code:
final_result = get_commute_info()

User: What are the top flood and wildfire risks?
Code:
final_result = get_nri_risk_data()

User: Give me a price estimate.
Code:
details = get_house_details()
final_result = estimate_price_with_code()

User: What recent sales are available as comps?
Code:
final_result = get_nearby_sold_homes()

User: What does the stored description say about the kitchen?
Code:
final_result = search_house_documents(query="kitchen description features")
"""

_HOUSE_FINAL_RESPONSE_PROMPT = """You are the final response writer for a House Chat Code Agent.

Use ONLY the evidence produced by the executed house functions below. Do not
invent facts, numbers, scores, or conclusions. The tool output is authoritative.

KEY RULES
=========
- A NULL/missing score value means the value is unavailable for this house.
  Say so clearly. Do not invent or estimate a replacement number.
- Do not claim a Walk Score, Bike Score, or Transit Score unless the tool
  explicitly returned a non-NULL numeric value for that field.
- Format prices with commas and dollar signs (e.g. $450,000).
- When discussing NRI risk, explain what the scores mean in plain language.
- Be concise and direct. Use markdown tables when comparing multiple values.

Return only the user-facing answer.
"""


# -- LLM singletons -----------------------------------------------------------

_house_code_agent: ChatOpenAI | None = None
_house_response_agent: ChatOpenAI | None = None


def _get_house_code_agent() -> ChatOpenAI:
    global _house_code_agent
    if _house_code_agent is None:
        _house_code_agent = ChatOpenAI(
            base_url=settings.llama_server_base_url,
            api_key="not-needed",
            model=settings.llama_server_model,
            temperature=0.0,
            max_tokens=min(getattr(settings, "agent_max_tokens", 2000), 1200),
            timeout=settings.llm_request_timeout,
            stop=LLM_STOP_SEQUENCES,
        )
    return _house_code_agent


def _get_house_response_agent() -> ChatOpenAI:
    global _house_response_agent
    if _house_response_agent is None:
        _house_response_agent = ChatOpenAI(
            base_url=settings.llama_server_base_url,
            api_key="not-needed",
            model=settings.llama_server_model,
            temperature=0.0,
            max_tokens=min(getattr(settings, "agent_max_tokens", 2000), 1400),
            timeout=settings.llm_request_timeout,
            stop=LLM_STOP_SEQUENCES,
        )
    return _house_response_agent


# -- Shared helpers -----------------------------------------------------------

def _extract_text(resp: Any) -> str:
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def _clean_code(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _bounded_history(history: list | None, max_chars: int = 8000) -> list:
    items = list(history or [])
    kept: list = []
    total = 0
    for item in reversed(items):
        content = str(item.get("content") or "")
        if kept and total + len(content) > max_chars:
            break
        kept.append({"role": item.get("role", "user"), "content": content})
        total += len(content)
    kept.reverse()
    return kept


def _messages_to_text(history: list | None, current_message: str) -> str:
    lines = []
    for item in _bounded_history(history):
        role = item.get("role", "user").upper()
        lines.append(f"{role}: {item.get('content', '')}")
    lines.append(f"USER: {current_message}")
    return "\n\n".join(lines)


def _render_evidence(calls: list) -> str:
    chunks = []
    for name, result in calls:
        text = str(result)
        if len(text) > 8000:
            text = text[:8000] + "\n...[truncated]"
        chunks.append(f"[{name}]\n{text}")
    return "\n\n".join(chunks)


def _tool_messages_from_calls(calls: list) -> list:
    return [
        ToolMessage(content=str(result), tool_call_id=f"house-agent-{i}", name=name)
        for i, (name, result) in enumerate(calls)
    ]


def _get_stored_description(house_id: str) -> str | None:
    try:
        import db.vector_store as vs
        doc = vs.get_description(house_id)
        return doc["text"] if doc else None
    except Exception:
        return None


# -- Program generation with recovery ----------------------------------------

def _linked_datasets_prompt() -> str:
    """Prompt section for user-added datasets that link to houses (empty when there are none).

    Built from the unified catalog on every request, so a dataset approved on the Data page is
    available to the very next House Chat turn: no restart and no prompt edit.
    """
    import db.schema_catalog as schema
    try:
        linked = schema.house_linked_datasets()
    except Exception:
        return ""
    if not linked:
        return ""
    clean = lambda t: " ".join(str(t or "").split())
    lines = [
        "ADDITIONAL LINKED DATASETS (added by the user on the Data page; joined to houses by approved relationships)",
        "=" * 78,
        '8. get_linked_dataset_records(dataset: str = "")',
        '   -> Rows from the datasets below that link to THIS house. Pass dataset="<name>" to narrow to one; omit it for all.',
        "   Datasets:",
    ]
    for d in linked:
        measures = f" Measures: {', '.join(d['measures'])}." if d["measures"] else ""
        lines.append(f"   - {d['name']}: {clean(d['description'])} Grain: {clean(d['grain']) or 'unspecified'}.{measures} Join: {d['join']}")
    first = linked[0]["name"]
    lines += [
        "   For comparisons across houses, or aggregates over these datasets, use query_database(request=...).",
        f"   Example: User: What does {first} say about this house?",
        f'   Code: final_result = get_linked_dataset_records(dataset="{first}")',
    ]
    return "\n".join(lines)


def _generate_house_program(
    user_message: str,
    house_id: str,
    history: list | None,
    approved_names: set,
    stored_description: str | None,
    prior_evidence: str = "",
) -> str:
    desc_section = (
        f"\nSTORED PROPERTY DESCRIPTION (already loaded -- do NOT call "
        f"search_house_documents to re-fetch it):\n{stored_description[:3000]}\n"
        if stored_description else ""
    )
    conversation = _messages_to_text(history, user_message)
    prompt_parts = [
        _HOUSE_CODE_AGENT_PROMPT,
        _linked_datasets_prompt(),
        f"HOUSE ID: {house_id}",
        desc_section,
        f"CONVERSATION CONTEXT:\n{conversation}",
    ]
    if prior_evidence:
        prompt_parts.append("EXECUTED EVIDENCE FROM A PREVIOUS STEP:\n" + prior_evidence[:6000])
    prompt_parts.append("Generate the next small Python program now.")

    response = _get_house_code_agent().invoke([
        SystemMessage(content="\n\n".join(p for p in prompt_parts if p)),
        HumanMessage(content=user_message),
    ])
    code = _clean_code(_extract_text(response))

    if code:
        try:
            _validate_house_program(code, approved_names)
            return code
        except Exception as first_exc:
            first_error = str(first_exc)
    else:
        first_error = "Code agent returned empty code."

    # Recovery: narrow prompt
    retry_prompt = (
        "You are a code generator for a house real-estate agent. "
        "Return exactly one executable Python statement calling an approved function. "
        "No explanation, no markdown, no blank response.\n"
        f"USER REQUEST: {user_message}\n"
        f"FIRST GENERATION ERROR: {first_error}\n"
        "Approved functions: " + ", ".join(sorted(approved_names))
    )
    retry = _get_house_code_agent().invoke([
        SystemMessage(content=retry_prompt),
        HumanMessage(content="Generate the statement now."),
    ])
    retry_code = _clean_code(_extract_text(retry))
    if retry_code:
        try:
            _validate_house_program(retry_code, approved_names)
            return retry_code
        except Exception:
            pass

    # Deterministic last-resort: fetch all house details -- always safe
    return "final_result = get_house_details()"


# -- Final answer synthesis ---------------------------------------------------

def _write_house_final_answer(user_message: str, history: list | None, evidence: str) -> str:
    conversation = _messages_to_text(history, user_message)
    prompt = (
        _HOUSE_FINAL_RESPONSE_PROMPT
        + "\n\nCONVERSATION:\n" + conversation
        + "\n\nEXECUTED EVIDENCE:\n" + evidence[:HOUSE_FINAL_RESPONSE_MAX_CHARS]
    )
    response = _get_house_response_agent().invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Write the final answer to the user's request. Return non-empty text grounded in the evidence."),
    ])
    answer = _extract_text(response)
    if answer:
        return answer

    retry_prompt = (
        "Answer the user's request using ONLY the evidence below.\n"
        "Do not invent facts. Return concise, non-empty plain text or markdown.\n\n"
        f"USER REQUEST:\n{user_message}\n\nEVIDENCE:\n{evidence[:10000]}"
    )
    retry = _get_house_response_agent().invoke([
        SystemMessage(content=retry_prompt),
        HumanMessage(content="Answer now."),
    ])
    return _extract_text(retry) or evidence.strip() or "I could not produce a response."


# -- Public entry-point -------------------------------------------------------

def run_house_chat(
    house_id: str,
    message: str,
    history: list | None = None,
) -> tuple:
    """
    Run one turn of house-specific chat using the Code Agent architecture.
    Returns (response_text, updated_history).
    history is a list of {"role": "user"|"assistant", "content": "..."}.
    """
    from observability import (
        start_house_chat,
        trace_span,
        set_span_input,
        set_span_output,
        mark_span_error,
        end_house_chat,
        record_house_chat_success,
        record_house_chat_error,
    )
    from services.guardrails import GuardrailManager, OutputGroundingGuardrail

    started_at = perf_counter()
    _, root_span, trace_id, trace_url = start_house_chat(message, house_id, len(history or []))

    # 1. Input Guardrail
    input_guard = GuardrailManager.inspect_turn_input(message)
    if not input_guard.passed:
        blocked_reply = (
            "I'm sorry, but I cannot process this request because it violates "
            "safety guidelines or attempts to override system instructions."
        )
        updated_history = list(history or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": blocked_reply},
        ]
        end_house_chat(
            root_span, trace_id=trace_id, reply=blocked_reply,
            started_at=started_at, tool_call_count=0,
        )
        return blocked_reply, updated_history

    # 2. Build approved function set bound to this house_id
    approved_functions = make_house_approved_functions(house_id)
    approved_names = set(approved_functions.keys())

    # Inject stored description so description questions do not waste a step
    stored_description = _get_stored_description(house_id)

    all_calls: list = []
    generated_programs: list = []

    try:
        evidence = ""

        for step in range(HOUSE_CODE_AGENT_MAX_STEPS):
            with trace_span(
                f"house_chat.code_agent.step_{step + 1}",
                attributes={
                    "openinference.span.kind": "LLM",
                    "code_agent.step": step + 1,
                    "house_chat.house_id": house_id,
                },
            ) as code_span:
                set_span_input(code_span, {
                    "message": message,
                    "prior_evidence": evidence[:4000],
                }, mime_type="application/json")
                try:
                    program = _generate_house_program(
                        user_message=message,
                        house_id=house_id,
                        history=history,
                        approved_names=approved_names,
                        stored_description=stored_description,
                        prior_evidence=evidence,
                    )
                    generated_programs.append(program)
                    set_span_output(code_span, {"generated_code": program}, mime_type="application/json")
                except Exception as exc:
                    mark_span_error(code_span, exc)
                    raise

            with trace_span(
                f"house_chat.code_execution.step_{step + 1}",
                attributes={
                    "openinference.span.kind": "TOOL",
                    "code_agent.step": step + 1,
                },
            ) as exec_span:
                set_span_input(exec_span, {"code": program}, mime_type="application/json")
                try:
                    final_result, calls = _execute_house_program(program, approved_functions)
                    all_calls.extend(calls)
                    step_evidence = _render_evidence(calls)
                    evidence = (evidence + "\n\n" + step_evidence).strip()
                    set_span_output(exec_span, {
                        "function_calls": [name for name, _ in calls],
                        "result_preview": str(final_result)[:500],
                    }, mime_type="application/json")
                except Exception as exc:
                    mark_span_error(exec_span, exc)
                    all_calls.append(("code_execution_error", str(exc)))
                    evidence = (evidence + f"\n\n[code_execution_error]\n{exc}").strip()
                    continue

            # Only loop if the first step produced no substantive data
            if step == 0:
                step_failed = any(
                    isinstance(result, str) and (
                        "not found" in result.lower()
                        or "no documents" in result.lower()
                        or "no data" in result.lower()
                        or "code agent error" in result.lower()
                    )
                    for _, result in calls
                )
                if step_failed:
                    continue
            break

        # 3. Final answer synthesis
        evidence = _render_evidence(all_calls)
        with trace_span(
            "house_chat.response",
            attributes={"openinference.span.kind": "LLM"},
        ) as response_span:
            set_span_input(response_span, {
                "message": message, "evidence": evidence[:8000],
            }, mime_type="application/json")
            raw_reply = _write_house_final_answer(message, history, evidence)
            set_span_output(response_span, {"reply": raw_reply}, mime_type="application/json")

        # 4. Response validation (evidence-grounding)
        all_lc_messages: list = [
            HumanMessage(content=message),
            *_tool_messages_from_calls(all_calls),
            AIMessage(content=raw_reply),
        ]
        reply = validate_response(raw_reply, all_lc_messages, strict=True)

        # 5. Output grounding guardrails
        try:
            import db.duckdb_store as store
            house = store.get_house(house_id) or {}
            walk_score = house.get("walk_score")
            reply, _ = OutputGroundingGuardrail.enforce_missing_score_guard(
                message, reply, walk_score, score_name="Walk Score"
            )
            score_res = OutputGroundingGuardrail.validate_scores_in_text(reply)
            if not score_res.passed:
                reply += f"\n\n*(Note: {'; '.join(score_res.reasons)})*"
        except Exception:
            pass

        updated_history = list(history or []) + [
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": reply,
                "trace_id": trace_id,
                "trace_url": trace_url,
            },
        ]

        end_house_chat(
            root_span, trace_id=trace_id, reply=reply,
            started_at=started_at, tool_call_count=len(all_calls),
        )
        record_house_chat_success(
            latency_seconds=perf_counter() - started_at,
            llm_calls=len(generated_programs) + 1,
            tool_calls=len(all_calls),
            reply_chars=len(reply),
            validation_changed=(reply != raw_reply),
        )
        return reply, updated_history

    except Exception as exc:
        mark_span_error(root_span, exc)
        end_house_chat(
            root_span, trace_id=trace_id, reply=None,
            started_at=started_at, tool_call_count=len(all_calls), error=exc,
        )
        record_house_chat_error(perf_counter() - started_at)
        raise