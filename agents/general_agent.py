"""
General Chat implemented as a Code Agent.

Architecture:
    user request
        -> LLM generates a small, declarative Python program
        -> AST validator permits only approved function calls/literals
        -> deterministic application functions execute that program
        -> execution evidence is returned to the LLM
        -> LLM produces the final grounded response

The LLM therefore decides WHAT computation is needed and WHICH approved
function(s) to call by generating code.  The application owns execution and
security; it never regex-classifies the user's intent to choose a tool.
"""
from __future__ import annotations

import ast
import json
import re
import inspect
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from config import settings, LLM_STOP_SEQUENCES
from agents.tools import (
    check_data_availability,
    get_database_schema,
    query_database,
    retrieve_data_model_context,
    find_bike_route,
    search_all_house_descriptions,
)
from agents.response_validator import validate_response
import db.schema_catalog as schema
from agents.query_planner import build_query_plan


CODE_AGENT_MAX_STEPS = 3
CODE_AGENT_MAX_CHARS = 16000
FINAL_RESPONSE_MAX_CHARS = 18000

APPROVED_FUNCTIONS = {
    "check_data_availability": check_data_availability,
    "get_database_schema": get_database_schema,
    "query_database": query_database,
    "retrieve_data_model_context": retrieve_data_model_context,
    "find_bike_route": find_bike_route,
    "search_all_house_descriptions": search_all_house_descriptions,
}

def _invoke_approved(name: str, function_obj, args: tuple[Any, ...], kwargs: dict[str, Any]):
    """Invoke an approved application function robustly.

    The Code Agent is instructed to emit keyword arguments, but local models can
    occasionally emit positional arguments. Because the callable set is already
    allow-listed, it is safe to normalize positional arguments here rather than
    failing the entire chat turn.
    """
    if args:
        if hasattr(function_obj, "args_schema"):
            fields = list(getattr(function_obj.args_schema, "model_fields", {}).keys())
        elif hasattr(function_obj, "func"):
            fields = list(inspect.signature(function_obj.func).parameters.keys())
        else:
            fields = list(inspect.signature(function_obj).parameters.keys())

        if len(args) > len(fields):
            raise CodeAgentProgramError(
                f"Too many positional arguments for approved function '{name}'."
            )
        for field, value in zip(fields, args):
            if field in kwargs:
                raise CodeAgentProgramError(
                    f"Argument '{field}' was supplied both positionally and by keyword."
                )
            kwargs[field] = value

    if hasattr(function_obj, "invoke"):
        return function_obj.invoke(kwargs)
    return function_obj(**kwargs)


def _get_data_availability_context() -> str:
    # Keep the routing prompt small. The General Code Agent does not need table
    # schemas; the SQL Code Agent owns schema/relationship reasoning. It only needs
    # to know which analytical capabilities are available.
    report, _ = schema.availability_report()
    compact = "\n".join(
        line for line in report.splitlines()
        if line.strip().lower().startswith(("nri_tracts", "census_msa", "cbsa_counties", "houses", "crime_incidents", "bike_routes"))
    )
    return (
        "[SEMANTIC CAPABILITIES]\n"
        "- General property analytics: houses / walk scores / price / listings\n"
        "- NRI tract risk: nri_tracts\n"
        "- NRI MSA analytics: physical tables census_msa + cbsa_counties + nri_tracts\n"
        "- MSA population universe: census_msa\n"
        "- MSA-to-NRI path: census_msa -> cbsa_counties -> nri_tracts\n"
        "- Crime analytics: crime_incidents\n"
        "- Bike routing: BikePGH network via find_bike_route\n"
        "[LIVE AVAILABILITY SUMMARY]\n" + compact + "\n[END CAPABILITIES]\n"
    )


def _bounded_history(history: list[dict] | None, max_chars: int = 12000) -> list[dict]:
    """Keep only a compact recent conversation context."""
    items = list(history or [])
    kept: list[dict] = []
    total = 0
    for item in reversed(items):
        content = str(item.get("content") or "")
        if kept and total + len(content) > max_chars:
            break
        kept.append({"role": item.get("role", "user"), "content": content})
        total += len(content)
    kept.reverse()
    return kept


def _messages_to_text(history: list[dict] | None, current_message: str) -> str:
    lines = []
    for item in _bounded_history(history):
        role = item.get("role", "user").upper()
        lines.append(f"{role}: {item.get('content', '')}")
    lines.append(f"USER: {current_message}")
    return "\n\n".join(lines)


def _extract_text(resp: Any) -> str:
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
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


class CodeAgentProgramError(ValueError):
    pass


def _validate_program(source: str) -> ast.Module:
    """Validate generated code as a tiny safe DSL over approved functions."""
    if not source:
        raise CodeAgentProgramError("Code agent returned empty code.")
    if len(source) > CODE_AGENT_MAX_CHARS:
        raise CodeAgentProgramError("Generated code exceeded the maximum size.")

    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise CodeAgentProgramError(f"Generated code is invalid Python: {exc}") from exc

    allowed_stmt = (ast.Assign, ast.Expr)
    allowed_expr = (
        ast.Call, ast.Constant, ast.Name, ast.Dict, ast.List, ast.Tuple,
        ast.keyword,
    )

    assigned_names: set[str] = set()
    call_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(node, allowed_stmt):
            raise CodeAgentProgramError(
                f"Unsupported statement {type(node).__name__}; only assignments and function calls are allowed."
            )

        if isinstance(node, ast.Import | ast.ImportFrom | ast.Attribute | ast.Subscript | ast.Lambda):
            raise CodeAgentProgramError(f"Unsupported expression {type(node).__name__}.")

        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise CodeAgentProgramError("Assignments must target one simple variable name.")
            assigned_names.add(node.targets[0].id)

        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in assigned_names and node.id not in APPROVED_FUNCTIONS:
                raise CodeAgentProgramError(f"Unknown variable '{node.id}' in generated code.")

        if isinstance(node, ast.Call):
            call_count += 1
            if not isinstance(node.func, ast.Name) or node.func.id not in APPROVED_FUNCTIONS:
                raise CodeAgentProgramError("Generated code may call only approved application functions.")
            if node.args:
                raise CodeAgentProgramError("Use keyword arguments for application functions.")
            for kw in node.keywords:
                if kw.arg is None:
                    raise CodeAgentProgramError("**kwargs are not allowed.")

            # Prefer keyword arguments, but do not make a harmless local-model
            # formatting deviation fatal. Positional arguments are normalized to
            # the approved callable's declared parameter order during execution.

        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Call):
            raise CodeAgentProgramError("Only bare function-call expressions are allowed.")

        if isinstance(node, ast.If | ast.For | ast.While | ast.Try | ast.With | ast.FunctionDef | ast.ClassDef):
            raise CodeAgentProgramError("Control flow and definitions are not allowed in generated code.")

    if call_count == 0:
        raise CodeAgentProgramError("Generated code must call at least one approved application function.")
    return tree


def _execute_program(source: str) -> tuple[Any, list[tuple[str, Any]]]:
    tree = _validate_program(source)
    calls: list[tuple[str, Any]] = []

    namespace: dict[str, Any] = {}
    for name, fn in APPROVED_FUNCTIONS.items():
        def make_wrapper(function_name: str, function_obj):
            def wrapped(*args, **kwargs):
                result = _invoke_approved(function_name, function_obj, args, kwargs)
                calls.append((function_name, result))
                return result
            return wrapped
        namespace[name] = make_wrapper(name, fn)

    namespace["__builtins__"] = {}
    exec(compile(tree, "<code-agent>", "exec"), namespace, namespace)

    final_result = namespace.get("final_result")
    if final_result is None and calls:
        final_result = calls[-1][1]
    if final_result is None:
        raise CodeAgentProgramError("Generated code did not produce a result.")
    return final_result, calls


CODE_AGENT_PROMPT = """You are the General Chat Code Agent for a real-estate application.

Your job is to translate the user's request into a SMALL Python program that calls
approved application functions. The program is executed by the application.
You are NOT writing SQL directly and you are NOT answering the user yet.

APPROVED FUNCTIONS
==================
1. check_data_availability() -> str
2. get_database_schema() -> str
3. query_database(request: str, requirements: str = "", plan: str = "") -> str
4. find_bike_route(start: str, end: str, city: str = "Pittsburgh, PA",
                   avoid_crime_dense_areas: bool = False,
                   crime_density_percentile: float = 90.0) -> str
5. search_all_house_descriptions(query: str) -> str

RULES
=====
1. Output ONLY executable Python code. No markdown fences and no explanation.
2. The code may contain only assignments and calls to the approved functions.
3. Use keyword arguments only.
4. For ANY analytical/data question, you MUST call query_database exactly as the execution step. Do not answer from memory and do not write SQL.
5. The application has already performed mandatory data-model retrieval and supplies a structured query plan. Preserve that plan when calling query_database.
6. For bike-route requests, call find_bike_route.
7. For crime-avoidance bike requests, set avoid_crime_dense_areas=True.
8. For ordinary shortest bike-route requests, set avoid_crime_dense_areas=False.
9. Do not substitute another routing service.
10. Keep queries focused and results reasonably small.
11. For “top N MSAs” + NRI questions, preserve the population universe and the MSA->county->tract relationship described by the structured plan.
12. For NRI hazard questions, preserve the user's hazard exactly. The SQL Code Agent
    maps it to the canonical semantic column.
13. IMPORTANT HOUSE-SCOPE RULE: phrases such as "my houses", "houses I have",
    "my Austin houses", or "houses in Austin" refer to the full house inventory.
    Do NOT add is_favorite = TRUE unless the user explicitly says "my list",
    "saved houses", "favorites", or "favorited houses".
14. When a request needs multiple independent evidence sources, you may make
    multiple approved calls and assign each result to a variable.
14. Set `final_result` to the most useful result for the final-response model.

Examples
========

User: Find the shortest bike route from A to B
Code:
final_result = find_bike_route(start="A", end="B", city="Pittsburgh, PA", avoid_crime_dense_areas=False)

User: Find a bike route from A to B that avoids crime-prone areas
Code:
final_result = find_bike_route(start="A", end="B", city="Pittsburgh, PA", avoid_crime_dense_areas=True, crime_density_percentile=90.0)

User: Which houses in my list have the highest walk scores?
Code:
final_result = query_database(
    request="Which houses in the user's current house list have the highest walk scores?",
    requirements="Return house identity/address and walk_score; rank descending; keep the result concise.",
    plan="Use the house/list dataset available to General Chat and return the top-ranked houses."
)
"""


FINAL_RESPONSE_PROMPT = """You are the final response writer for a real-estate Code Agent.

Use ONLY the evidence produced by the executed application functions. Do not
invent facts, numbers, routes, or map conclusions.

For bike routes:
- If the executed find_bike_route result says a route exists, summarize it.
- If it says no_route, clearly say it is not possible using the applicable
  BikePGH network; never invent an alternative.
- For crime-aware routing, never describe an unfiltered route as crime-aware.
- The application renders maps separately from your prose.

For analytical questions:
- Answer directly from query_database evidence.
- Use markdown tables when comparing several rows.
- Format dollar amounts with commas and scores/percentages to one decimal where appropriate.

Return only the user-facing answer.
"""


_code_agent: ChatOpenAI | None = None
_response_agent: ChatOpenAI | None = None


def _get_code_agent() -> ChatOpenAI:
    global _code_agent
    if _code_agent is None:
        _code_agent = ChatOpenAI(
            base_url=settings.llama_server_base_url,
            api_key="not-needed",
            model=settings.llama_server_model,
            temperature=0.0,
            max_tokens=min(getattr(settings, "agent_max_tokens", 2000), 1800),
            timeout=settings.llm_request_timeout,
            stop=LLM_STOP_SEQUENCES,
        )
    return _code_agent


def _get_response_agent() -> ChatOpenAI:
    global _response_agent
    if _response_agent is None:
        _response_agent = ChatOpenAI(
            base_url=settings.llama_server_base_url,
            api_key="not-needed",
            model=settings.llama_server_model,
            temperature=0.0,
            max_tokens=min(getattr(settings, "agent_max_tokens", 2000), 1400),
            timeout=settings.llm_request_timeout,
            stop=LLM_STOP_SEQUENCES,
        )
    return _response_agent


def _tool_messages_from_calls(calls: list[tuple[str, Any]]) -> list[ToolMessage]:
    messages = []
    for index, (name, result) in enumerate(calls):
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=f"code-agent-{index}",
                name=name,
            )
        )
    return messages


def _parse_bike_payloads(calls: list[tuple[str, Any]]) -> list[dict]:
    payloads: list[dict] = []
    for name, result in calls:
        if name != "find_bike_route" or not isinstance(result, str):
            continue
        try:
            parsed = json.loads(result)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads


def _extract_bike_visualization(payloads: list[dict]):
    for payload in reversed(payloads):
        analysis = payload.get("analysis_visualization")
        crime = payload.get("crime_avoidance") or {}
        applied = bool(crime.get("enabled")) and bool(crime.get("applied"))
        final_route = None
        if payload.get("presentation") == "route_map" and payload.get("route_shape"):
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
        if final_route:
            return final_route
    return None


def _generate_program(user_message: str, history: list[dict] | None, prior_evidence: str = "", data_model_context: str = "") -> str:
    query_plan = build_query_plan(user_message)
    prompt = [
        _get_data_availability_context(),
        "MANDATORY STRUCTURED QUERY PLAN:\n" + query_plan.render(),
        "MANDATORY DATA MODEL RETRIEVAL:\n" + (data_model_context or "No targeted metadata retrieved."),
        CODE_AGENT_PROMPT,
        "CONVERSATION CONTEXT:\n" + _messages_to_text(history, user_message),
    ]
    if prior_evidence:
        prompt.append("EXECUTED EVIDENCE FROM A PREVIOUS STEP:\n" + prior_evidence[:12000])
    prompt.append("Generate the next small Python program now.")
    response = _get_code_agent().invoke([
        SystemMessage(content="\n\n".join(prompt)),
        HumanMessage(content=user_message),
    ])
    code = _clean_code(_extract_text(response))
    if code:
        try:
            _validate_program(code)
            return code
        except Exception as first_exc:
            first_error = str(first_exc)
        else:
            first_error = ""
    else:
        first_error = "Code agent returned empty code."

    # Local models can occasionally emit an empty completion or malformed
    # Python. Recover with a tiny, syntax-constrained prompt rather than
    # allowing the failure to become a user-facing "no data" answer.
    # re-classifying the user's intent: ask the same Code Agent for the minimal
    # program and include only the semantic capabilities it needs.
    retry_prompt = (
        "You are a code generator. Return exactly one executable Python statement "
        "calling an approved function. For this user request, the usual analytical "
        "choice is query_database(...). No explanation, no markdown, no blank response.\n"
        f"USER REQUEST: {user_message}\n"
        "If it is an MSA/NRI risk question, call query_database with the request verbatim "
        "and mention the documented census_msa -> cbsa_counties -> nri_tracts relationship in requirements."
    )
    retry = _get_code_agent().invoke([
        SystemMessage(content=retry_prompt + f"\nFIRST GENERATION ERROR: {first_error}"),
        HumanMessage(content="Generate the statement now."),
    ])
    retry_code = _clean_code(_extract_text(retry))
    if retry_code:
        try:
            _validate_program(retry_code)
            return retry_code
        except Exception:
            pass

    # Last deterministic escape hatch for the common analytical case. This
    # still goes through query_database, which preserves the LLM-first/data-
    # model-grounded architecture while making malformed local-model output
    # non-fatal.
    return (
        "final_result = query_database("
        f"request={user_message!r}, "
        "requirements='Answer the user request from the database; preserve the structured plan and documented relationships.', "
        f"plan={query_plan.render()!r})"
    )


def _render_evidence(calls: list[tuple[str, Any]]) -> str:
    chunks = []
    for name, result in calls:
        text = str(result)
        if len(text) > 10000:
            text = text[:10000] + "\n...[truncated]"
        chunks.append(f"[{name}]\n{text}")
    return "\n\n".join(chunks)


def _extract_query_result_fallback(evidence: str) -> str:
    """Return the most useful executed result when the response LLM is empty.

    Code Agent execution is authoritative; an empty response from the final
    response model must never become an empty chat message. For database
    results, strip the generated-SQL wrapper and expose the result table/text.
    """
    if not evidence:
        return "I could not produce a response from the executed application functions."

    blocks = evidence.split("[query_database]")
    if len(blocks) > 1:
        candidate = blocks[-1].strip()
        if "[RESULT]" in candidate:
            candidate = candidate.split("[RESULT]", 1)[1].strip()
        if candidate and not candidate.lower().startswith(("code agent error", "query returned 0 rows", "0 rows")):
            return candidate

    # Generic fallback: use the final non-empty executed tool result.
    for block in reversed(evidence.split("\n\n")):
        block = block.strip()
        if block and not block.startswith("[GENERATED SQL]"):
            return block
    return evidence.strip()


def _write_final_answer(user_message: str, history: list[dict] | None, evidence: str) -> str:
    context = _messages_to_text(history, user_message)
    prompt = (
        FINAL_RESPONSE_PROMPT
        + "\n\nCONVERSATION:\n" + context
        + "\n\nEXECUTED EVIDENCE:\n" + evidence[:FINAL_RESPONSE_MAX_CHARS]
    )
    response = _get_response_agent().invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Write the final answer to the user's current request. Return non-empty text grounded in the evidence."),
    ])
    answer = _extract_text(response)
    if answer:
        return answer

    # Retry with a deliberately tiny prompt. This handles local models that
    # occasionally emit an empty completion after a long/complex system prompt.
    retry_prompt = (
        "Answer the user's request using ONLY the evidence below.\n"
        "Do not invent facts. Return concise, non-empty plain text or markdown.\n\n"
        f"USER REQUEST:\n{user_message}\n\nEVIDENCE:\n{evidence[:12000]}"
    )
    retry = _get_response_agent().invoke([
        SystemMessage(content=retry_prompt),
        HumanMessage(content="Answer now."),
    ])
    answer = _extract_text(retry)
    return answer or _extract_query_result_fallback(evidence)


def run_general_chat(
    message: str,
    history: list[dict] | None = None,
    include_metadata: bool = False,
    session_id: str | None = None,
):
    """Run the General Chat Code Agent and emit one Phoenix trace per answer."""
    from time import perf_counter
    from observability import (
        start_general_chat,
        trace_span,
        set_span_input,
        set_span_output,
        mark_span_error,
        end_general_chat,
        record_general_chat_success,
        record_general_chat_error,
    )

    started_at = perf_counter()
    _, root_span, trace_id, trace_url = start_general_chat(message, session_id, len(history or []))

    all_tool_messages: list[ToolMessage] = []
    all_calls: list[tuple[str, Any]] = []
    generated_programs: list[str] = []

    try:
        evidence = ""
        with trace_span(
            "general_chat.query_planner",
            attributes={"openinference.span.kind": "CHAIN"},
        ) as plan_span:
            planned = build_query_plan(message)
            set_span_input(plan_span, {"request": message}, mime_type="application/json")
            set_span_output(plan_span, planned.to_dict(), mime_type="application/json")

        with trace_span(
            "general_chat.data_model_rag",
            attributes={"openinference.span.kind": "RETRIEVER"},
        ) as rag_span:
            set_span_input(rag_span, {"query": message}, mime_type="application/json")
            model_context = retrieve_data_model_context.invoke(message)
            set_span_output(rag_span, {"context": str(model_context)[:12000]}, mime_type="application/json")

        for step in range(CODE_AGENT_MAX_STEPS):
            with trace_span(
                f"general_chat.code_agent.step_{step + 1}",
                attributes={
                    "openinference.span.kind": "LLM",
                    "code_agent.step": step + 1,
                },
            ) as code_span:
                set_span_input(code_span, {
                    "message": message,
                    "prior_evidence": evidence[:6000],
                }, mime_type="application/json")
                try:
                    program = _generate_program(message, history, prior_evidence=evidence, data_model_context=str(model_context))
                    generated_programs.append(program)
                    set_span_output(code_span, {"generated_code": program}, mime_type="application/json")
                except Exception as exc:
                    mark_span_error(code_span, exc)
                    raise

            with trace_span(
                f"general_chat.code_execution.step_{step + 1}",
                attributes={
                    "openinference.span.kind": "TOOL",
                    "code_agent.step": step + 1,
                },
            ) as exec_span:
                set_span_input(exec_span, {"code": program}, mime_type="application/json")
                try:
                    final_result, calls = _execute_program(program)
                    all_calls.extend(calls)
                    step_evidence = _render_evidence(calls)
                    evidence = (evidence + "\n\n" + step_evidence).strip()
                    all_tool_messages.extend(_tool_messages_from_calls(calls))
                    set_span_output(exec_span, {
                        "function_calls": [name for name, _ in calls],
                        "result": final_result,
                    }, mime_type="application/json")
                except Exception as exc:
                    mark_span_error(exec_span, exc)
                    # A malformed recovery program from a local model should not
                    # erase useful evidence from the previous execution. Give the
                    # next step a chance to regenerate instead of aborting the
                    # entire user turn.
                    all_calls.append(("code_execution_error", str(exc)))
                    evidence = (evidence + f"\n\n[code_execution_error]\n{exc}").strip()
                    continue

            # The code agent is intentionally a small program generator. A
            # second step is only useful when the application has evidence and
            # the model needs another targeted query/call.
            if step == 0:
                # Normally one generated program is enough. If the first
                # application call produced no substantive result (especially a
                # zero-row SQL query or a diagnostic telling us the join was
                # likely wrong), give the Code Agent one targeted recovery step
                # with the real execution evidence. This keeps recovery inside
                # the Code Agent architecture instead of hard-coding intent
                # or SQL fixes in application code.
                names = [name for name, _ in calls]
                step_failed = any(
                    isinstance(result, str) and
                    ("Query returned 0 rows" in result
                     or "Code Agent error:" in result
                     or "does not match" in result.lower()
                     or "join" in result.lower() and "likely" in result.lower())
                    for _, result in calls
                )
                if not any(name in {"query_database", "find_bike_route", "search_all_house_descriptions"} for name in names) or step_failed:
                    continue
            break

        evidence = _render_evidence(all_calls)
        with trace_span(
            "general_chat.response",
            attributes={"openinference.span.kind": "LLM"},
        ) as response_span:
            set_span_input(response_span, {"message": message, "evidence": evidence[:12000]}, mime_type="application/json")
            raw_reply = _write_final_answer(message, history, evidence)
            set_span_output(response_span, {"reply": raw_reply}, mime_type="application/json")

        all_messages: list[BaseMessage] = [
            HumanMessage(content=message),
            *all_tool_messages,
            AIMessage(content=raw_reply),
        ]
        reply = validate_response(raw_reply, all_messages, strict=True)

        updated_history = list(history or []) + [
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": reply,
                "trace_id": trace_id,
                "trace_url": trace_url,
            },
        ]

        bike_payloads = _parse_bike_payloads(all_calls)
        visualization = _extract_bike_visualization(bike_payloads)

        end_general_chat(
            root_span,
            trace_id=trace_id,
            reply=reply,
            started_at=started_at,
            tool_call_count=len(all_calls),
        )
        record_general_chat_success(
            latency_seconds=perf_counter() - started_at,
            llm_calls=len(generated_programs) + 1,
            tool_calls=len(all_calls),
            reply_chars=len(reply),
            validation_changed=(reply != raw_reply),
        )

        if include_metadata:
            return reply, updated_history, visualization, {
                "trace_id": trace_id,
                "trace_url": trace_url,
                "generated_code": generated_programs,
            }
        return reply, updated_history
    except Exception as exc:
        mark_span_error(root_span, exc)
        end_general_chat(
            root_span,
            trace_id=trace_id,
            reply=None,
            started_at=started_at,
            tool_call_count=len(all_calls),
            error=exc,
        )
        record_general_chat_error(perf_counter() - started_at)
        raise
