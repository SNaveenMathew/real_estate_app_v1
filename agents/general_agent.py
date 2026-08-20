"""
General agent — answers broad questions about metro areas, national risk,
portfolio-level analysis across all houses, etc.
"""
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import settings, LLM_STOP_SEQUENCES
from agents.tools import GENERAL_TOOLS
import json

import db.schema_catalog as schema


SYSTEM_PROMPT = """You are a real estate and risk analyst assistant backed by a local DuckDB database.

A live snapshot of what's actually loaded (row counts per table) is included
below — you don't need to re-check it unless you suspect it changed mid-conversation.

Ground rules:
1. Every number, ranking, or comparison you state must come from a tool call
   made in this conversation — never estimate or recall a figure from memory.
2. If a table is empty or a query returns 0 rows, say so plainly. Do not fill
   the gap with a plausible-sounding invented answer.
3. For ANY question that requires analytical data from DuckDB, use the
   `query_database` tool instead of writing SQL yourself. This applies to
   houses, snapshots, sales, NRI, Census, crime, bike, and every other
   agent-visible dataset. `query_database` is backed by the shared SQL Code
   Agent, which generates the SQL from the live table metadata and documented
   relationships. Put the user's analytical request in `request`; when your
   reasoning has explicit requirements or a multi-step plan, pass those in
   `requirements` and `plan`.
4. Treat the `query_database` result as evidence, not as the final answer.
   Inspect the generated SQL and returned rows, reconcile them with the user's
   intent, and continue your reasoning/planning before responding. If the
   question needs another query after seeing the first result, call
   `query_database` again with the updated requirements or plan.
5. If `query_database` errors or returns 0 rows, read the message carefully.
   Revise the request/requirements/plan and retry when appropriate, or explain
   the data limitation. Never fabricate a result.

`get_database_schema` remains available when you need to inspect the live schema
directly, but do not bypass `query_database` by writing SQL yourself for an
analytical data question.

Formatting: dollar amounts as $1,234,567; scores/percentages to one decimal
place; use a markdown table when comparing more than a couple of rows.

BIKING / BIKE-ROUTING PRESENTATION RULES:
1. For a question such as "is there a safe way to bike from A to B?" determine
   whether the locally loaded BikePGH network documents a continuous bikeable
   path. Treat "safe" as "supported by documented BikePGH bicycle
   infrastructure"; do NOT claim a safety guarantee. The final answer MUST
   start with exactly one of: "Yes", "No", or "No data available for <city or MSA>
   to answer this question." If there is no BikePGH data, use the third form.
2. For a request to FIND/SHOW/ROUTE a bike path between two endpoints, call
   find_bike_route. Endpoints may be neighborhoods, landmarks, parks,
   addresses, or coordinates; exact map clicks are not required.
3. For FIND/SHOW/ROUTE requests that explicitly ask to avoid crime-dense,
   high-crime, dangerous, or crime-heavy areas, call find_bike_route with
   avoid_crime_dense_areas=true. The application execution layer will enforce
   this flag even if the model omits it. Let the tool apply the spatial crime
   filter before Dijkstra; do not invent a safety route yourself.
4. For every crime-aware route request, an intermediate map MUST be preserved and
   rendered showing the post-filter BikePGH network, crime density, source, and
   destination. This map is required whether or not Dijkstra finds a path.
5. For crime-aware requests, NEVER present a route unless the tool explicitly
   reports that the crime filter was applied. If the filter could not be applied,
   treat the route as unavailable and explain the limitation. Do not fall back to
   an ordinary or neighborhood route.
6. If find_bike_route returns a successful crime-filtered route, return a concise
   text summary and let the application render a separate final route map. If
   kind="no_route", return text explaining that no continuous path exists using
   the filtered BikePGH network. Do NOT request an external routing service or
   invent a road route.
6. If find_bike_route returns kind="no_data", return exactly:
   "No data available for <city or MSA> to answer this question."
"""


def _get_data_availability_context() -> str:
    """Live data-availability summary, prepended to every system message.
    Shares its table list and counting logic with the check_data_availability
    tool (both read db/schema_catalog.py) so the two can't drift apart."""
    report, _ = schema.availability_report()
    return f"[LIVE DATABASE STATUS]\n{report}\n[END STATUS]\n"


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: list(x) + list(y)]


def _message_requests_crime_avoidance(messages: Sequence[BaseMessage]) -> bool:
    """Deterministically detect crime-avoidance routing intent.

    This is an execution guard, not an LLM replacement: the local LLM still
    decides to use find_bike_route, but it cannot accidentally downgrade an
    explicitly crime-aware routing request into an ordinary route.
    """
    text_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                text_parts.append(content.lower())
            elif isinstance(content, list):
                text_parts.extend(str(x.get("text", "")) for x in content if isinstance(x, dict))
    text = "\n".join(text_parts)
    crime_terms = (
        "crime", "criminal", "dangerous", "unsafe", "high-crime",
        "high crime", "crime-prone", "crime prone", "avoid crime",
        "avoid high-crime", "avoid dangerous", "avoid unsafe",
    )
    return any(term in text for term in crime_terms)


def _extract_bike_route_endpoints(message: str) -> tuple[str, str] | None:
    """Extract endpoints from a direct bike-route request.

    Keeping this deterministic avoids sending route requests through the
    general LLM agent, which can otherwise accumulate a very large history
    and hit the local model context limit.
    """
    import re
    text = " ".join((message or "").split())
    if not re.search(r"\bbike\b", text, re.I) or not re.search(r"\broute\b", text, re.I):
        return None
    m = re.search(
        r"\bfrom\s+(.+?)\s+\bto\s+(.+?)(?=\s+(?:that\s+)?(?:avoids?|avoiding|without)\b|[.!?]?$)",
        text, re.I,
    )
    if not m:
        return None
    start_text = m.group(1).strip(" ,")
    end_text = m.group(2).strip(" ,")
    if not start_text or not end_text:
        return None
    return start_text, end_text


def _extract_crime_aware_bike_route_endpoints(message: str) -> tuple[str, str] | None:
    """Backward-compatible wrapper for crime-aware route detection."""
    text = " ".join((message or "").split())
    if not _message_requests_crime_avoidance([HumanMessage(content=text)]):
        return None
    return _extract_bike_route_endpoints(text)


def _bounded_agent_history(history: list[dict] | None, max_chars: int = 24000) -> list[dict]:
    """Keep general-agent prompts comfortably below the local model context limit."""
    items = list(history or [])
    if not items:
        return []
    kept = []
    total = 0
    for item in reversed(items):
        content = str(item.get("content") or "")
        size = len(content)
        if kept and total + size > max_chars:
            break
        kept.append({"role": item.get("role", "user"), "content": content})
        total += size
    kept.reverse()
    return kept


def build_general_agent():
    tool_map = {getattr(t, "name", ""): t for t in GENERAL_TOOLS}

    def tools_node(state: AgentState):
        """Execute tool calls while enforcing explicit crime-aware routing.

        We keep the existing tools and LangGraph architecture. The only added
        behavior is a deterministic guard at execution time: for an explicitly
        crime-aware bike request, any find_bike_route call is forced to enable
        the crime filter.
        """
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        enforce_crime = _message_requests_crime_avoidance(state["messages"])
        outputs = []
        for call in tool_calls:
            name = call.get("name", "")
            tool = tool_map.get(name)
            if tool is None:
                outputs.append(ToolMessage(
                    content=f"Unknown tool: {name}",
                    tool_call_id=call.get("id", ""),
                    name=name,
                ))
                continue
            args = dict(call.get("args") or {})
            if enforce_crime and name == "find_bike_route":
                args["avoid_crime_dense_areas"] = True
                # Preserve the caller's explicit percentile if supplied.
                args.setdefault("crime_density_percentile", 90.0)
            try:
                result = tool.invoke(args)
                if isinstance(result, ToolMessage):
                    outputs.append(result)
                else:
                    outputs.append(ToolMessage(
                        content=str(result),
                        tool_call_id=call.get("id", ""),
                        name=name,
                    ))
            except Exception as exc:
                outputs.append(ToolMessage(
                    content=f"Tool error in {name}: {exc}",
                    tool_call_id=call.get("id", ""),
                    name=name,
                    status="error",
                ))
        return {"messages": outputs}
    # For 'llama-server -hf DuoNeural/Gemma-4-26B-A4B-it-GGUF:Q3_K_M -ngl 999 -c 28672 -fa on --cache-type-k q8_0 --cache-type-v q8_0'
    llm = ChatOpenAI(
        base_url=settings.llama_server_base_url,
        api_key="not-needed",  # llama-server doesn't check the key, but LangChain requires a non-empty string
        model=settings.llama_server_model,
        temperature=0.0,
        max_tokens=settings.agent_max_tokens,  # hard backstop against runaway generation
        timeout=settings.llm_request_timeout,
        stop=LLM_STOP_SEQUENCES,               # text-level stop, robust to a broken EOG token list
    ).bind_tools(GENERAL_TOOLS)
    # For 'ollama run --model llama3.1:8b'
    # llm = ChatOllama(
    #     base_url=settings.ollama_base_url,
    #     model=settings.ollama_model,
    #     temperature=0.0,
    # ).bind_tools(GENERAL_TOOLS)

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
    graph.add_node("tools", tools_node)
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


def _extract_bike_visualization_from_payload(payload: dict, crime_request: bool = False):
    """Normalize one authoritative bike-route tool payload for the frontend.

    The current user request, not prior chat history, determines whether a
    final route must prove crime filtering was applied.
    """
    if not isinstance(payload, dict):
        return None
    analysis = payload.get("analysis_visualization")
    crime_meta = payload.get("crime_avoidance") or {}
    applied = bool(crime_meta.get("enabled")) and bool(crime_meta.get("applied"))

    final_route = None
    if payload.get("presentation") == "route_map" and payload.get("route_shape") and (not crime_request or applied):
        final_route = {
            "type": "bike_route",
            "city": payload.get("city") or "Pittsburgh, PA",
            "start": payload.get("start"),
            "end": payload.get("end"),
            "route_shape": payload.get("route_shape") or [],
            "bbox": payload.get("bbox"),
            "distance_miles": payload.get("distance_miles"),
            "duration_minutes": payload.get("duration_minutes"),
            "turn_by_turn": payload.get("turn_by_turn") or [],
            "bike_infrastructure_near_route": payload.get("bike_infrastructure_near_route") or [],
            "used_infrastructure": payload.get("used_infrastructure") or {"type": "FeatureCollection", "features": []},
            "provider": payload.get("provider"),
            "attribution": payload.get("attribution"),
        }

    if analysis:
        result = {"type": "bike_crime_analysis", "analysis": analysis}
        if final_route:
            result["final_route"] = final_route
        return result
    return final_route


def _extract_bike_visualization(messages: list[BaseMessage], crime_request: bool | None = None):
    """Extract bike visualizations using the current request intent, not stale history."""
    import json
    if crime_request is None:
        crime_request = _message_requests_crime_avoidance(messages[-1:])

    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        content = message.content
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        analysis = payload.get("analysis_visualization")
        final_route = None
        crime_meta = payload.get("crime_avoidance") or {}
        crime_filter_applied = bool(crime_meta.get("enabled")) and bool(
            crime_meta.get("applied", crime_meta.get("enabled"))
        )

        if (payload.get("presentation") == "route_map" or payload.get("status") == "ok") and (
            not crime_request or crime_filter_applied
        ):
            final_route = {
                "type": "bike_route",
                "city": payload.get("city") or "Pittsburgh, PA",
                "start": payload.get("start"),
                "end": payload.get("end"),
                "route_shape": payload.get("route_shape") or [],
                "bbox": payload.get("bbox"),
                "distance_miles": payload.get("distance_miles"),
                "duration_minutes": payload.get("duration_minutes"),
                "turn_by_turn": payload.get("turn_by_turn") or [],
                "bike_infrastructure_near_route": payload.get("bike_infrastructure_near_route") or [],
                "used_infrastructure": payload.get("used_infrastructure") or {"type": "FeatureCollection", "features": []},
                "provider": payload.get("provider"),
                "attribution": payload.get("attribution"),
            }

        if analysis:
            result = {"type": "bike_crime_analysis", "analysis": analysis}
            if final_route and final_route.get("route_shape"):
                result["final_route"] = final_route
            return result

        if final_route:
            return final_route

    return None


def run_general_chat(
    message: str,
    history: list[dict] = None,
    include_metadata: bool = False,
    session_id: str | None = None,
):
    """Run General Chat and emit one Phoenix trace per answer."""
    from time import perf_counter
    from observability import (
        start_general_chat,
        trace_span,
        set_span_input,
        set_span_output,
        mark_span_error,
        end_general_chat,
    )

    started_at = perf_counter()
    _, root_span, trace_id, trace_url = start_general_chat(
        message, session_id, len(history or [])
    )
    try:
        agent = get_general_agent()

        # Route requests are handled deterministically below, but other LLM
        # requests still need a bounded history to avoid exceeding the local
        # model context window after long sessions.
        bounded_history = _bounded_agent_history(history)
        lc_messages: list[BaseMessage] = []
        for h in bounded_history:
            if h["role"] == "user":
                lc_messages.append(HumanMessage(content=h["content"]))
            else:
                lc_messages.append(AIMessage(content=h["content"]))
        lc_messages.append(HumanMessage(content=message))

        # Initialize routing state before branching. Ordinary questions may not
        # take the deterministic bike-route path, but the downstream routing
        # reconciliation code still inspects this value.
        bike_tool_result = None

        deterministic_route = _extract_bike_route_endpoints(message)
        route_is_crime_aware = _message_requests_crime_avoidance([HumanMessage(content=message)])
        if deterministic_route:
            start_text, end_text = deterministic_route
            from agents.tools import find_bike_route
            with trace_span(
                "general_chat.bike_route",
                attributes={
                    "openinference.span.kind": "TOOL",
                    "bike_route.start": start_text,
                    "bike_route.end": end_text,
                    "bike_route.crime_avoidance": route_is_crime_aware,
                },
            ) as route_span:
                try:
                    tool_json = find_bike_route.invoke({
                        "start": start_text,
                        "end": end_text,
                        "city": "Pittsburgh, PA",
                        "avoid_crime_dense_areas": bool(route_is_crime_aware),
                        "crime_density_percentile": 90.0,
                    })
                    try:
                        deterministic_payload = json.loads(tool_json) if isinstance(tool_json, str) else None
                    except Exception:
                        deterministic_payload = None
                    set_span_output(route_span, {
                        "result": deterministic_payload if isinstance(deterministic_payload, dict) else str(tool_json),
                    }, mime_type="application/json" if isinstance(deterministic_payload, dict) else "text/plain")
                except Exception as exc:
                    mark_span_error(route_span, exc)
                    raise
            all_messages = [*lc_messages, ToolMessage(
                content=str(tool_json),
                tool_call_id="deterministic-bike-route",
                name="find_bike_route",
            )]
            ai_messages = []
            bike_tool_result = deterministic_payload if isinstance(deterministic_payload, dict) else None
            if bike_tool_result:
                raw_reply = bike_tool_result.get("message") or (
                    f"Yes — a BikePGH route was found from {start_text} to {end_text}."
                    if bike_tool_result.get("status") == "ok"
                    else "The bike-routing request could not be completed."
                )
            else:
                raw_reply = "The crime-aware bike-routing tool returned an unreadable result."
        else:
            with trace_span(
                "general_chat.agent",
                attributes={
                    "openinference.span.kind": "AGENT",
                    "general_chat.history_length": len(history or []),
                    "general_chat.session_id": session_id or "",
                },
            ) as agent_span:
                set_span_input(agent_span, {"message": message, "history": history or []}, mime_type="application/json")
                try:
                    result = agent.invoke({"messages": lc_messages})
                    all_messages = list(result["messages"])
                    set_span_output(agent_span, {"message_count": len(all_messages)}, mime_type="application/json")
                except Exception as exc:
                    mark_span_error(agent_span, exc)
                    raise

            ai_messages = [m for m in all_messages if isinstance(m, AIMessage)]
            raw_reply = ai_messages[-1].content if ai_messages else "I couldn't generate a response."

        # Routing tool results are authoritative. In particular, when a
        # crime-aware bike request returns kind="no_route", do not let the
        # local LLM invent a neighborhood/approximate fallback route. The
        # intermediate filtered-network/crime map remains attached to the
        # tool result and is rendered independently by the UI.
        if not isinstance(bike_tool_result, dict):
            bike_tool_result = None
        for message_item in reversed(all_messages):
            # Prefer an already-parsed deterministic routing result.
            if isinstance(bike_tool_result, dict):
                break
            if not isinstance(message_item, ToolMessage):
                continue
            try:
                payload = json.loads(message_item.content) if isinstance(message_item.content, str) else None
            except Exception:
                payload = None
            if isinstance(payload, dict) and (
                payload.get("kind") in {"no_route", "no_data"}
                or payload.get("analysis_visualization")
                or payload.get("presentation") == "route_map"
            ):
                bike_tool_result = payload
                break

        crime_request = _message_requests_crime_avoidance([HumanMessage(content=message)])
        if crime_request and bike_tool_result:
            crime_meta = bike_tool_result.get("crime_avoidance") or {}
            if bike_tool_result.get("kind") in {"routing_service", "input"}:
                raw_reply = (
                    bike_tool_result.get("message")
                    or "The crime-aware routing request could not be evaluated."
                )
            elif bike_tool_result.get("presentation") == "route_map" and not bool(crime_meta.get("enabled")):
                raw_reply = (
                    "The requested crime-aware route could not be produced because "
                    "the crime-avoidance filter was not applied. No unfiltered route "
                    "is being substituted."
                )
                bike_tool_result["kind"] = "crime_filter_error"
                bike_tool_result["status"] = "error"
                bike_tool_result["message"] = raw_reply
                # Keep the intermediate visualization, if present, but prevent a
                # final route visualization from being rendered.
                if "analysis_visualization" in bike_tool_result:
                    bike_tool_result["presentation"] = None

        if bike_tool_result and bike_tool_result.get("kind") in {"no_route", "crime_filter_error"}:
            raw_reply = bike_tool_result.get("message") or (
                "No continuous path exists between the two endpoints using the "
                "filtered BikePGH infrastructure."
            )

        # Crime-aware routing is deterministic: the routing tool result is
        # authoritative. Do not let the final LLM rewrite a no-route, filter
        # failure, or successfully filtered route into a different outcome.
        crime_request = _message_requests_crime_avoidance([HumanMessage(content=message)])
        direct_routing_reply = None
        if crime_request and bike_tool_result:
            crime_meta = bike_tool_result.get("crime_avoidance") or {}
            analysis_present = bool(bike_tool_result.get("analysis_visualization"))
            applied = bool(crime_meta.get("enabled")) and bool(
                crime_meta.get("applied", False)
            )

            if bike_tool_result.get("kind") in {"no_route", "no_data"}:
                direct_routing_reply = bike_tool_result.get("message") or (
                    "No continuous path exists between the two endpoints using "
                    "the filtered BikePGH infrastructure."
                )
            elif bike_tool_result.get("kind") in {"routing_service", "crime_filter_error", "input"}:
                direct_routing_reply = bike_tool_result.get("message") or (
                    "The crime-aware routing request could not be evaluated."
                )
            elif bike_tool_result.get("presentation") == "route_map" and applied:
                # Let the normal validator check the generated route prose, but
                # never allow it to disable crime filtering or substitute a
                # different route.
                direct_routing_reply = raw_reply
            elif analysis_present:
                # We have the required intermediate map, but the route itself
                # was not proven crime-aware. Never present the unfiltered route.
                direct_routing_reply = (
                    "The requested crime-aware route could not be produced because "
                    "the crime-avoidance filter was not successfully applied. "
                    "No unfiltered route is being substituted."
                )

        if direct_routing_reply is not None:
            raw_reply = direct_routing_reply

        from agents.response_validator import validate_response

        # The routing result is authoritative for crime-aware route requests.
        # For ordinary General Chat, keep the normal response validator.
        # Initialize the reply on every path so non-crime conversations cannot
        # reach updated_history with an undefined local variable.
        reply = validate_response(raw_reply, all_messages, strict=True)
        if crime_request and bike_tool_result:
            if direct_routing_reply is not None and not (
                bike_tool_result.get("presentation") == "route_map"
                and bool((bike_tool_result.get("crime_avoidance") or {}).get("applied", False))
            ):
                reply = raw_reply
            else:
                # A successfully crime-filtered route may still be checked for prose quality.
                reply = validate_response(raw_reply, all_messages, strict=True)

        updated_history = (history or []) + [
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": reply,
                "trace_id": trace_id,
                "trace_url": trace_url,
            },
        ]

        visualization = _extract_bike_visualization(all_messages, crime_request=crime_request)
        if isinstance(bike_tool_result, dict):
            direct_visualization = _extract_bike_visualization_from_payload(
                bike_tool_result, crime_request=crime_request
            )
            # The current request controls whether an ordinary route or a
            # crime-filtered route is allowed to render. Never let stale
            # conversation history suppress a valid ordinary route map.
            if direct_visualization is not None:
                if not crime_request or (
                    direct_visualization.get("type") == "bike_crime_analysis"
                    or direct_visualization.get("type") == "bike_route"
                ):
                    visualization = direct_visualization
        end_general_chat(
            root_span,
            trace_id=trace_id,
            reply=reply,
            started_at=started_at,
            tool_call_count=sum(len(getattr(m, "tool_calls", None) or []) for m in ai_messages),
        )

        if include_metadata:
            return reply, updated_history, visualization, {
                "trace_id": trace_id,
                "trace_url": trace_url,
            }
        return reply, updated_history
    except Exception as exc:
        mark_span_error(root_span, exc)
        end_general_chat(
            root_span,
            trace_id=trace_id,
            reply=None,
            started_at=started_at,
            error=exc,
        )
        raise
