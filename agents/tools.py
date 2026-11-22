"""
LangChain tools for the general agent (General Chat).

House Chat binds no tools here: it executes the approved functions in
agents/house_agent.py, whose query_database delegates to the shared query_database
tool below. A house-only capability belongs there, not in this module - the
second copy this module used to carry is exactly what let the ZHVI estimate go
missing from the path users actually reach.
"""

import functools
import json
import logging
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
import services.transit_routing as transit_routing
from agents.query_planner import build_query_plan
from agents import answer_policy, answer_status, capabilities, llm_resilience, semantic_fallback, tool_result
from agents.artifacts import classify_dataframe, emit_artifact
import asyncio
import threading

from observability import trace_span, mark_span_error, set_span_output

capability = capabilities.capability  # read as a decorator below: @capability("find_bike_route")


_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|COPY|EXPORT|ATTACH|DETACH|INSTALL|LOAD|CALL|PRAGMA)\b",
    re.I,
)
_TABLE_REF = re.compile(
    r"\b(?:FROM|JOIN|UPDATE|INTO|TABLE)\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)",
    re.I,
)
_AGGREGATE_CALL_RE = re.compile(r"\b(?:AVG|SUM|COUNT|MEDIAN|MIN|MAX)\s*\(", re.I)


# ── Typed results and capability bounds (agents/tool_result.py, agents/capabilities.py) ──


def _envelope(text: str) -> tool_result.ToolResult:
    """Wrap query_database's rendered evidence with the facts downstream code needs.

    The text is unchanged. The envelope stops the response validator (and anything else) from deciding what the
    output *means* by counting digits or looking for the word "Error" in it.
    """
    if answer_status.is_not_answered(text):
        return tool_result.not_answered(text, answer_status.status_code(text) or "not_answered")
    if text.startswith(("Code Agent error", "ERROR:")) or "SQL Error" in text:
        return tool_result.failure(text)
    return tool_result.ToolResult(text, kind="data")


def _typed_tool(kind: str):
    """Give a JSON-returning tool a ToolResult carrying its own status, and a clean deadline failure."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                out = fn(*args, **kwargs)
            except capabilities.CapabilityTimeout:
                return tool_result.not_answered(
                    answer_status.not_answered(
                        answer_status.LIMIT_EXCEEDED,
                        "This step took longer than the time allowed for it, so it was stopped. Nothing was estimated.",
                    ),
                    answer_status.LIMIT_EXCEEDED,
                )
            try:
                payload = json.loads(out)
            except (TypeError, ValueError):
                return tool_result.ToolResult(str(out), kind=kind)
            status = str(payload.get("status") or "").lower()
            ok = status not in ("error", "analysis") and payload.get("kind") != "no_route"
            return tool_result.ToolResult(
                out,
                kind=kind if ok else "not_answered",
                ok=ok,
                status=None if ok else (payload.get("kind") or status or None),
            )

        return wrapper

    return decorator


SYSTEM_PROMPT = """You are a SQL Code Agent. Return exactly one read-only DuckDB SELECT statement.

Use only the USER REQUEST and the STRUCTURED QUERY PLAN / TARGETED DATA MODEL provided below.
The plan comes from the application schema catalog and live entity resolution. It is authoritative.
Do not reinterpret scope, invent filters, invent joins, or substitute display labels for resolved values.
Use documented relationship paths only. Apply aggregation/null/default-filter semantics from the metadata.
Treat the structured plan as the complete semantic scope. Never add filters that are not represented there.
When metadata marks a default filter as required, apply it unless the structured plan explicitly indicates an override.
Output SQL only.
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
            max_retries=settings.llm_max_retries,
        )
    return _agent


def generate_sql(request: str, requirements: str = "", plan: str = "", focused: bool = False) -> str:
    """Generate one read-only SELECT from the immutable structured plan.

    The SQL-generation span deliberately records the model's raw and cleaned
    SQL *before* validation, so a rejected query is still visible in Phoenix
    and local evaluation traces.
    """
    query_plan = build_query_plan(request)
    effective_plan = (plan or "").strip() or query_plan.render()
    targeted = schema.build_query_context(request, plan=effective_plan, focused=focused)
    prompt = "\n\n".join(
        [
            f"USER REQUEST:\n{request.strip()}",
            "STRUCTURED QUERY PLAN (authoritative):\n" + effective_plan,
            "TARGETED LIVE DATA MODEL (authoritative):\n" + targeted,
            "Generate exactly one DuckDB SELECT/WITH query. Do not add semantic scope, filters, tables, or joins absent from the plan/model.",
        ]
    )

    attrs = {
        "openinference.span.kind": "LLM",
        "sql.request": request.strip(),
        "sql.focused_retry": focused,
    }
    with trace_span("sql_generation", attributes=attrs) as span:
        if span is not None:
            try:
                span.set_attribute("sql.plan", effective_plan)
                span.set_attribute("sql.targeted_data_model", targeted)
            except Exception:
                pass
        try:
            response = llm_resilience.invoke(
                get_code_agent(),
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ],
                role="sql_agent",
            )
            raw = _extract_content(response)
            cleaned = _clean_sql(raw)
            if span is not None:
                try:
                    span.set_attribute("sql.raw_model_output", raw)
                    span.set_attribute("sql.cleaned", cleaned)
                    span.set_attribute("sql.output", cleaned)
                    span.set_attribute("sql.output_empty", not bool(cleaned.strip()))
                except Exception:
                    pass
            try:
                validated = validate_sql(cleaned)
            except Exception as exc:
                if span is not None:
                    try:
                        span.set_attribute("sql.validation_status", "rejected")
                        span.set_attribute("sql.validation_error", str(exc))
                    except Exception:
                        pass
                raise
            if span is not None:
                try:
                    span.set_attribute("sql.validation_status", "accepted")
                except Exception:
                    pass
            return validated
        except Exception as exc:
            mark_span_error(span, exc)
            raise


def _where_clause_text(low_sql: str) -> str:
    """Return only the WHERE-clause portion(s) of a lowercased SQL string.

    Used to scope the "unplanned filter" scan in _validate_sql_against_plan
    to actual row-filtering predicates. A JOIN ... ON predicate — e.g. the
    required relationship census_msa.msa_code = cbsa_counties.cbsa_code —
    structurally connects tables per the catalog's relationship graph; it
    is not a semantic filter narrowing which rows qualify, and must not be
    mistaken for one just because its column name is followed by '=' in
    the FROM clause. Handles multiple WHERE clauses (e.g. inside CTEs) by
    capturing each occurrence up to the next clause boundary.
    """
    segments = re.findall(
        r"\bwhere\b(.*?)(?=\bgroup\s+by\b|\border\s+by\b|\blimit\b|\bwhere\b|$)",
        low_sql,
        re.S,
    )
    return " ".join(segments)


def _validate_sql_against_plan(sql: str, query_plan) -> None:
    """Validate generated SQL against the declarative catalog plan and SQL safety guardrail."""
    from services.guardrails import CodeAgentGuardrail

    guard_res = CodeAgentGuardrail.validate_sql(sql)
    if not guard_res.passed:
        raise ValueError(f"SQL Guardrail violation: {'; '.join(guard_res.reasons)}")

    refs = {m.group(1).lower() for m in _TABLE_REF.finditer(sql)}
    planned = {t.lower() for t in query_plan.required_tables}
    if not planned:
        raise ValueError("Query plan selected no tables.")
    missing = planned - refs
    extra = refs - planned
    if missing:
        raise ValueError(f"Generated SQL omitted planned table(s): {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"Generated SQL introduced unplanned table(s): {', '.join(sorted(extra))}")

    low = sql.lower()
    # Every resolved live entity is an exact database fact. The SQL must carry it.
    for filt in query_plan.entity_filters:
        for literal in re.findall(r"'([^']+)'", filt):
            if literal.lower() not in low:
                raise ValueError(f"Generated SQL omitted resolved entity value: {literal}")

    # Planned semantic filters must survive generation. We compare normalized
    # field/value atoms, not a brittle full-string representation.
    for filt in query_plan.filters:
        atoms = [
            a
            for a in re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*|true|false|[0-9]+(?:\.[0-9]+)?", filt.lower())
            if a not in {"and", "or", "is", "null", "where", "not"}
        ]
        if not all(a in low for a in atoms):
            raise ValueError(f"Generated SQL omitted planned semantic filter: {filt}")

    # Detect scope filters on known semantic columns that the plan did not
    # select. Scoped to the WHERE clause only (see _where_clause_text) so a
    # required JOIN ... ON bridge from query_plan.required_relationships is
    # never mistaken for an invented row filter.
    where_low = _where_clause_text(low)
    semantic_filter_columns = {}
    for key, item in schema.SEMANTIC_GLOSSARY.items():
        if item.get("scope_guard") is False:
            # Concepts generated for user-added datasets: their measure columns are legitimately
            # filterable (e.g. "walkability index above 15"), so they do not feed this guard.
            continue
        for col in item.get("columns", []):
            semantic_filter_columns.setdefault(col.split(".")[-1].lower(), set()).add(key)
    selected = set(query_plan.semantic_keys)
    entity_filter_columns = set()
    for filt in query_plan.entity_filters:
        for m in re.finditer(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)\s*=", filt):
            entity_filter_columns.add(m.group(1).lower())
    for col, keys in semantic_filter_columns.items():
        if re.search(rf"\b{re.escape(col)}\b\s*(?:=|is|in|like|ilike|>|<|>=|<=)", where_low):
            if col in entity_filter_columns:
                continue
            if not selected.intersection(keys):
                raise ValueError(f"Generated SQL introduced an unplanned filter on semantic field '{col}'.")

    _verify_plan_contract(sql, query_plan)


_inspector_warned = False


def _verify_plan_contract(sql: str, query_plan) -> None:
    """Structural verification (agents/plan_contract.py): safety, approved joins, entity/scope filters, row limit.

    Runs AFTER the legacy checks so every message they produce for a statement they already rejected is unchanged; this
    only ADDS rejections.  If DuckDB's parse-tree format is ever unreadable the legacy checks alone apply (the previous
    behaviour) and the problem is logged loudly once, instead of every statement being rejected.
    """
    global _inspector_warned
    from agents import plan_contract, sql_inspector

    try:
        violations = plan_contract.verify(sql, query_plan)
    except sql_inspector.SqlInspectorUnavailable as exc:
        if not _inspector_warned:
            _inspector_warned = True
            import logging

            logging.getLogger(__name__).warning(
                "SQL inspector unavailable (%s): falling back to the legacy SQL checks only. "
                "Run tests/test_sql_inspector.py against this DuckDB version.",
                exc,
            )
        return
    mode = str(getattr(settings, "sql_contract_mode", "enforce") or "enforce").lower()
    blockers = plan_contract.blocking(violations, mode)
    if blockers:
        raise ValueError("SQL contract violated: " + " | ".join(v.message for v in blockers))
    if violations:
        import logging

        logging.getLogger(__name__).info(
            "SQL contract (mode=%s) would reject: %s", mode, "; ".join(v.message for v in violations)
        )


def _qualify_join_expr(table: str, expr: str) -> str:
    """Table-qualify every bare column reference in a relationship-key expr.

    Most catalog relationships key on a single bare column, where a plain
    f"{table}.{expr}" prefix is correct. A few key on a compound expression
    instead — e.g. "state_fips || county_fips" (concatenation) or
    "LEFT(tract_fips, 5)" (a function call) — and naively prefixing the
    whole string only qualifies the first token. That leaves later bare
    columns ambiguous once another joined table happens to share the same
    column name (e.g. both cbsa_counties and nri_tracts have county_fips),
    and turns a function call into invalid syntax ("table.LEFT(...)" is not
    "LEFT(table.col, ...)"). Qualify each bare identifier individually
    instead, leaving SQL function names (identifier immediately followed by
    '(') and already-qualified references untouched.
    """
    return schema.qualify_join_expr(table, expr)  # single shared implementation (db/schema_catalog.py)


def _group_equality_predicates(filters: list[str]) -> list[str]:
    """Combine same-column resolved-entity equalities into one IN (...).

    query_plan.entity_filters holds one "table.col = 'value'" predicate per
    resolved named entity — e.g. four separate MSA-name equalities for "the
    Pittsburgh, Denver, Miami, and Austin metro areas". AND-joining
    different-valued equalities on the same column is a contradiction (a
    column cannot equal two different literals in the same row) and would
    make the query always return zero rows; they need to be OR-combined —
    via IN (...) — instead. Filters that already target different columns
    are unaffected and still get AND-ed together as before. Anything not
    matching the simple "col = 'value'" shape is passed through unchanged.
    """
    pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*'(.*)'$", re.S)
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    passthrough: list[str] = []
    for filt in filters:
        m = pattern.match(filt.strip())
        if not m:
            passthrough.append(filt)
            continue
        col, val = m.group(1), m.group(2)
        if col not in grouped:
            grouped[col] = []
            order.append(col)
        grouped[col].append(val)
    out = []
    for col in order:
        values = grouped[col]
        if len(values) == 1:
            out.append(f"{col} = '{values[0]}'")
        else:
            value_list = ", ".join(f"'{v}'" for v in values)
            out.append(f"{col} IN ({value_list})")
    out.extend(passthrough)
    return out


def _compile_sql_from_plan(query_plan) -> str:
    """Compile a minimal read-only SQL query from the declarative plan.

    This is a safety-net only: it runs when the LLM SQL generator returns no SQL.
    It never changes routing; it uses the tables, relationships, operation,
    projection, grouping, ordering, filters and live entities already selected
    by the metadata planner.
    """
    tables = set(query_plan.required_tables)
    if not tables:
        raise ValueError("Cannot compile SQL: plan selected no tables.")

    rels = schema.relationship_path(tables)
    if len(tables) > 1 and len(rels) < len(tables) - 1:
        raise ValueError("Cannot compile SQL: selected tables are not fully connected by catalog relationships.")

    def table_of(expr: str | None) -> str | None:
        if not expr:
            return None
        for table in sorted(tables):
            if re.search(rf"\b{re.escape(table)}\.", expr):
                return table
        return None

    start = None
    if query_plan.rollup_spec:
        group_key = str(query_plan.rollup_spec.get("group_key", ""))
        start = table_of(group_key)
    if not start and query_plan.grouping:
        start = table_of(query_plan.grouping[0])
    if not start:
        start = table_of(query_plan.select_expression)
    if not start:
        start = sorted(tables)[0]

    from_sql = start
    connected = {start}
    remaining = set(tables) - connected
    while remaining:
        picked = None
        for rel in rels:
            left, right = rel.left_table, rel.right_table
            if left in connected and right in remaining:
                picked = (rel, right)
                break
            if right in connected and left in remaining:
                picked = (rel, left)
                break
        if not picked:
            raise ValueError("Cannot compile SQL: no catalog join path from current tables.")
        rel, new_table = picked
        left_qualified = _qualify_join_expr(rel.left_table, rel.left_expr)
        right_qualified = _qualify_join_expr(rel.right_table, rel.right_expr)
        from_sql += f" JOIN {new_table} ON {left_qualified} = {right_qualified}"
        connected.add(new_table)
        remaining.remove(new_table)

    op = query_plan.operation
    expr = query_plan.select_expression
    if not op:
        raise ValueError("Cannot compile SQL: plan has no operation.")

    select_sql = expr or "*"
    group_sql = list(query_plan.grouping)
    if op == "count":
        select_sql = f"COUNT({expr or '*'})"
    elif op in {"avg", "sum", "median", "min", "max"}:
        select_sql = f"{op.upper()}({expr})"
    elif op == "rank":
        # rank plans either already contain an explicit grouped projection
        # (e.g. MSA/NRI: "name, AVG(x)") or a scalar per-row projection
        # (e.g. sold-home ranking: "city, sold_price") — the group_by on
        # the catalog operation is a display/context grouping, not
        # necessarily a real aggregation.
        select_sql = expr or query_plan.aggregation or "*"

    # GROUP BY is only valid SQL — and only what's actually intended — when
    # every non-grouped projected column is wrapped in an aggregate call.
    # Some catalog rank operations pair a grouping key with a bare column
    # rather than an aggregate (see comment above); emitting GROUP BY for
    # those produces an invalid DuckDB query ("column must appear in the
    # GROUP BY clause or be part of an aggregate function"), so only keep
    # the clause when the projection actually aggregates something.
    if group_sql and not _AGGREGATE_CALL_RE.search(select_sql):
        group_sql = []

    predicates = []
    predicates.extend(_group_equality_predicates(query_plan.entity_filters))
    predicates.extend(query_plan.scope_filters)  # mandatory scope (agents/query_scope.py)
    predicates.extend(query_plan.filters)
    if "exclude NULL" in query_plan.null_policy and expr:
        for field in re.findall(r"\b[a-zA-Z_][\w]*\.[a-zA-Z_][\w]*\b", expr):
            predicates.append(f"{field} IS NOT NULL")
    where_sql = f" WHERE {' AND '.join(predicates)}" if predicates else ""

    group_clause = f" GROUP BY {', '.join(group_sql)}" if group_sql else ""
    order_clause = f" ORDER BY {query_plan.ordering}" if query_plan.ordering else ""
    limit_clause = f" LIMIT {int(query_plan.result_limit)}" if query_plan.result_limit else ""  # an explicit "top N"

    return f"SELECT {select_sql} FROM {from_sql}{where_sql}{group_clause}{order_clause}{limit_clause}"


def run_code_query(
    request: str,
    requirements: str = "",
    plan: str = "",
    presentation: str = "auto",
) -> tuple[str, str]:
    """Generate, validate, execute and generically repair one analytical query.

    `presentation` is passed straight through to
    agents.artifacts.classify_dataframe as a hint only — see that
    function's docstring for what it does and does not control. The text
    returned here (the evidence the final-answer LLM reads) is completely
    unaffected by it; only the emitted artifact, if any, changes shape.
    """
    query_plan = build_query_plan(request)
    if not query_plan.required_tables:
        # The validator rejects every statement that does not use a planned table, so with an empty plan no SQL a model
        # could write can succeed: asking it - up to three times - only burns time and ends in a misleading error.
        return "", answer_policy.unplannable_result(request)
    base_plan = query_plan.render()
    repair_note = ""
    last_sql = ""
    for attempt in range(3):
        effective_plan = base_plan
        if repair_note:
            effective_plan += "\n\nREPAIR CONTEXT:\n" + repair_note
        try:
            sql = generate_sql(request, plan=effective_plan, focused=(attempt > 0))
            with trace_span(
                "sql_validation",
                attributes={
                    "sql.attempt": attempt + 1,
                    "sql.statement": sql,
                },
            ) as validation_span:
                try:
                    _validate_sql_against_plan(sql, query_plan)
                    if validation_span is not None:
                        validation_span.set_attribute("sql.validation_status", "accepted")
                except Exception as validation_exc:
                    if validation_span is not None:
                        validation_span.set_attribute("sql.validation_status", "rejected")
                        validation_span.set_attribute("sql.validation_error", str(validation_exc))
                    raise
            last_sql = sql
            if not sql.strip():
                sql = _compile_sql_from_plan(query_plan)
        except llm_resilience.TurnCancelled:
            raise  # the deadline passed: never "recover" by doing more work
        except Exception as exc:
            # Never swallow silently: a model writing bad SQL (ValueError from validation) is routine, but anything else
            # before the model was even reached (prompt building, client set-up) is a defect that the compiled-SQL
            # fallback would otherwise hide.
            _sql_log = logging.getLogger(__name__)
            (_sql_log.info if isinstance(exc, ValueError) else _sql_log.warning)(
                "SQL attempt %d failed (%s: %s)", attempt + 1, type(exc).__name__, str(exc)[:300]
            )
            try:
                # The fallback compiler is intentionally limited to the empty/invalid
                # SQL case; it never replaces valid LLM-generated SQL.
                if not last_sql:
                    sql = _compile_sql_from_plan(query_plan)
                    _validate_sql_against_plan(sql, query_plan)
                    last_sql = sql
                    repair_note = "LLM SQL unavailable; executed catalog-compiled SQL."
                else:
                    raise
            except Exception:
                repair_note = f"Generation/validation failed: {exc}. Regenerate from the same authoritative plan; do not change user scope."
                if isinstance(exc, llm_resilience.ModelUnavailable):
                    raise exc  # the REAL cause (not the compile error being handled right now);
                    # a model known to be down will not answer the next attempt either
                if attempt == 2:
                    raise
                continue
        try:
            with trace_span(
                "sql_execution",
                attributes={
                    "sql.attempt": attempt + 1,
                    "sql.statement": sql,
                },
            ) as execution_span:
                try:
                    # Time-limited and row-capped, and served from the shared SQL/CTE cache when the
                    # identical statement already ran against unchanged data (db/duckdb_store.py).
                    df, truncated, cache_hit = store.query_limited_cached(sql, request_text=request)
                    if execution_span is not None:
                        execution_span.set_attribute("sql.cache_hit", bool(cache_hit))
                        execution_span.set_attribute("sql.truncated", bool(truncated))
                        execution_span.set_attribute("sql.row_count", int(len(df)))
                        execution_span.set_attribute("sql.result_empty", bool(df.empty))
                        if not df.empty:
                            set_span_output(execution_span, df.head(50).to_string(index=False))
                except Exception as execution_exc:
                    if execution_span is not None:
                        execution_span.set_attribute("sql.execution_status", "error")
                        execution_span.set_attribute("sql.execution_error", str(execution_exc))
                    raise
        except store.SqlTimeout as exc:
            # Re-running a statement that is too slow with the same plan would only repeat the wait: stop, and say so.
            return sql, answer_status.not_answered(
                answer_status.LIMIT_EXCEEDED,
                f"{exc} The question is too broad to answer in one step; narrow it (a place, a time range, a smaller set) "
                "and ask again.",
            )
        except Exception as exc:
            repair_note = (
                f"Execution failed: {exc}. Previous SQL: {sql}. Correct only the SQL error using the same plan."
            )
            if attempt == 2:
                raise RuntimeError(f"Generated SQL failed: {exc}\nSQL: {sql}") from exc
            continue
        if df.empty:
            diagnosis = schema.diagnose_empty_or_error(sql) or "No relationship diagnostics available."
            if attempt == 2:
                # Every repair attempt is exhausted, and the last SQL was a
                # validated query that executed without error — it just
                # matched no rows. That is very often the correct, factual
                # answer (e.g. "no CBSA match for this MSA"), not a broken
                # query, so say so plainly instead of returning the raw
                # repair-hint diagnostics (written for the next
                # SQL-generation attempt) as if they were the answer.
                return sql, (
                    "The query executed successfully against the documented schema and "
                    f"returned 0 rows — no matching records were found. {diagnosis}"
                )
            repair_note = f"The query returned 0 rows. Diagnostic context: {diagnosis}. Correct join/entity/filter mistakes without changing scope."
            continue
        try:
            emit_artifact(classify_dataframe(df, request=request, presentation=presentation))
        except Exception:
            pass  # a presentation-layer problem must never break an otherwise-successful query
        if truncated:
            return sql, df.head(50).to_string(index=False) + (
                f"\n... (more than {len(df)} rows exist; only the first {len(df)} were loaded and 50 are shown - "
                "add a filter or a LIMIT for the full picture)"
            )
        if len(df) > 50:
            return sql, df.head(50).to_string(index=False) + f"\n... ({len(df)} total rows, showing 50)"
        return sql, df.to_string(index=False)
    return last_sql, "Query could not be completed."


# ── General tools ────────────────────────────────────────────────────────────


@tool
@capability("check_data_availability")
def check_data_availability(_: str = "") -> str:
    """
    Row count for every table, so you know what is actually loaded.
    A live snapshot of this is already included in your system context each
    turn — you don't need to call this proactively. It's here in case you
    want to double-check after data may have been reloaded mid-conversation.
    If a table has 0 rows, you CANNOT answer questions that depend on it —
    tell the user which files need to be loaded instead.
    """
    report, counts = schema.availability_report()
    try:
        counts_df = pd.DataFrame(sorted(counts.items()), columns=["table", "rows"])
        emit_artifact(classify_dataframe(counts_df, title="Data availability", presentation="table"))
    except Exception:
        pass
    return tool_result.ToolResult(report, kind="status")


@tool
@capability("retrieve_data_model_context")
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
    return tool_result.ToolResult("\n\n".join(p for p in parts if p), kind="schema")


@tool
@capability("query_database")
def query_database(
    request: str,
    requirements: str = "",
    plan: str = "",
    presentation: str = "auto",
) -> str:
    """
    Use the shared SQL Code Agent to answer analytical questions from any
    agent-visible DuckDB dataset.

    Pass the user's analytical request plus any explicit General Chat
    requirements or multi-step plan. The Code Agent uses the live schema
    metadata and documented relationships to generate one read-only SELECT,
    executes it, and returns both the generated SQL and live result so General
    Chat can inspect the evidence and continue thinking/planning before it
    answers the user.

    `presentation` is an OPTIONAL hint: "map" if the user asked to see this
    on a map or asked where something is; "chart"/"plot"/"graph" if they
    asked to see a trend or a comparison across categories; "table" if they
    asked for a table or a side-by-side list; "auto" (the default)
    otherwise. This never overrides what the executed data can actually
    support — a "map" hint with no place/coordinate data in the result
    still renders as a chart or table, never a fabricated map.
    """
    structured_plan = build_query_plan(request)
    if not structured_plan.required_tables and requirements and requirements.strip():
        # The orchestrator often states the measure only in `requirements` ("Return one row per metro with its land
        # area") while `request` just names the places.  Plan from both before giving up: an empty plan can never succeed.
        enriched = f"{request.strip()} {requirements.strip()}"
        retry_plan = build_query_plan(enriched)
        if retry_plan.required_tables or retry_plan.semantic_keys:
            request, structured_plan = enriched, retry_plan

    # Nothing matched by alias: let a model PROPOSE concept keys (never SQL).  Proposed keys enter the plan as if an alias
    # had matched; everything after - routing, SQL contract, execution - is unchanged (agents/semantic_fallback.py).
    proposed: list[str] = []
    if not structured_plan.required_tables and not structured_plan.semantic_keys:
        proposed = semantic_fallback.propose_concepts(request)
    if proposed:
        with schema.forcing_concepts(proposed):
            return _envelope(_annotate_interpretation(_answer(request, requirements, presentation), proposed))

    # Semantic cache (services/cache.py): an earlier question the deterministic planner routed to the
    # SAME concepts, tables, operation and scope, on unchanged data, is reused verbatim - so the
    # evidence the answer model reads is identical to a fresh run.  House Chat delegates here too,
    # which is why the scope (e.g. this house) is part of the signature.
    from services import cache as result_cache

    signature = result_cache.semantic_signature(structured_plan)
    cached = result_cache.lookup_semantic("query_database", request, signature)
    if cached is not None:
        return _envelope(cached["response"])

    out = _answer(request, requirements, presentation, structured_plan)
    if not answer_status.is_not_answered(out) and "Code Agent error" not in out:
        result_cache.store_semantic(
            "query_database",
            request,
            signature,
            out,
            meta={
                "concepts": structured_plan.semantic_keys,
                "tables": structured_plan.required_tables,
                "operation": structured_plan.operation or "",
                "kind": "semantic",
            },
        )
    return _envelope(out)


def _annotate_interpretation(out: str, keys: list[str]) -> str:
    """Tell the answer model (and so the user) how a request nothing matched was interpreted."""
    if "[RESULT]\n" not in out or answer_status.is_not_answered(out):
        return out
    return out.replace("[RESULT]\n", f"[RESULT]\n(Interpreted as: {semantic_fallback.describe(keys)})\n", 1)


def _answer(request: str, requirements: str, presentation: str, structured_plan=None) -> str:
    structured_plan = structured_plan or build_query_plan(request)
    # Requests that must not reach SQL generation (ambiguous place, polygon-derived measure, known-unloaded topic) are
    # decided from catalog metadata in agents/answer_policy.py - there is no question-specific branch here.
    routed = answer_policy.route_before_sql(structured_plan, request, presentation=presentation)
    if routed is not None:
        return routed

    try:
        sql, result = run_code_query(request, presentation=presentation)
    except llm_resilience.TurnCancelled:
        raise
    except llm_resilience.ModelUnavailable:
        sql, result = (
            "",
            answer_status.not_answered(
                answer_status.MODEL_UNAVAILABLE,
                "The language model service is not responding, and writing a query for this question needs it. Nothing was "
                "estimated; ask again in a minute.",
            ),
        )
    except Exception as exc:
        sql, result = "", f"Code Agent error: {exc}"
    result += answer_policy.caveats_after_sql(structured_plan)

    # LLM-first architecture: SQL generation and repair are driven entirely by
    # the live schema plus semantic metadata. There are no question-specific
    # canonical SQL recipes in the execution layer.

    return f"[GENERATED SQL]\n{sql}\n[RESULT]\n{result}"


def _run_async_safely(async_fn, *args, deadline_s: float | None = None, **kwargs):
    """Run an async function from sync code or from an active event loop.

    ``deadline_s`` is a real bound, not a note: without a running loop the coroutine is wrapped in
    ``asyncio.wait_for``; with one, the worker thread is joined with a timeout and abandoned when it overruns
    (a running thread cannot be killed, so the vertex stops waiting — the coroutine is still cancelled by
    ``wait_for`` at the same deadline).
    """
    coro = async_fn(*args, **kwargs)
    if deadline_s is not None:
        coro = asyncio.wait_for(coro, timeout=max(float(deadline_s), 0.001))
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = []
    error = []

    def runner():
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(deadline_s)

    if error:
        raise error[0]
    if not result:
        raise TimeoutError(f"Async worker did not finish within {deadline_s}s.")
    return result[0]


@tool
@_typed_tool("route")
@capability("find_bike_route")
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
            deadline_s=capabilities.effective_deadline("find_bike_route"),
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
                message = (
                    f"No — Not possible: no continuous bike path exists using the filtered BikePGH network. {note}"
                )
            if result.get("analysis_visualization"):
                emit_artifact(
                    {
                        "type": "map",
                        "map_kind": "bike_crime_analysis",
                        "title": "Crime-aware route filtering",
                        "analysis": result.get("analysis_visualization"),
                    }
                )
            return json.dumps(
                {
                    "status": "analysis",
                    "kind": "no_route",
                    "message": message,
                    "start": result.get("start"),
                    "end": result.get("end"),
                    "city": result.get("city") or city or "Pittsburgh, PA",
                    "crime_avoidance": result.get("crime_avoidance") or {"enabled": False},
                    "analysis_visualization": result.get("analysis_visualization"),
                },
                indent=2,
            )

        crime_meta = result.get("crime_avoidance") or {"enabled": False, "applied": False}
        success_message = (
            "Yes — a continuous BikePGH route exists after filtering out the selected high-density crime areas."
            if bool(crime_meta.get("enabled")) and bool(crime_meta.get("applied"))
            else "Yes — a continuous BikePGH route was found."
        )
        if result.get("analysis_visualization"):
            emit_artifact(
                {
                    "type": "map",
                    "map_kind": "bike_crime_analysis",
                    "title": "Crime-aware route filtering",
                    "analysis": result.get("analysis_visualization"),
                }
            )
        if route_data.get("shape"):
            emit_artifact(
                {
                    "type": "map",
                    "map_kind": "bike_route",
                    "title": f"Bike route: {result.get('start')} \u2192 {result.get('end')}",
                    "city": result.get("city") or city or "Pittsburgh, PA",
                    "start": result.get("start"),
                    "end": result.get("end"),
                    "route_shape": route_data.get("shape", []) or [],
                    "bbox": route_data.get("bbox"),
                    "distance_miles": round(float(summary.get("length", 0) or 0), 2),
                    "duration_minutes": round(float(summary.get("time", 0) or 0) / 60.0, 1),
                    "turn_by_turn": instructions,
                    "bike_infrastructure_near_route": facilities.get("facility_segments", []),
                    "used_infrastructure": route_data.get("used_infrastructure")
                    or {"type": "FeatureCollection", "features": []},
                    "provider": result.get("provider"),
                    "attribution": result.get("attribution"),
                }
            )
        return json.dumps(
            {
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
                "used_infrastructure": route_data.get("used_infrastructure")
                or {"type": "FeatureCollection", "features": []},
                "alternatives_considered": result.get("alternatives_considered", 1),
                "crime_avoidance": result.get("crime_avoidance") or {"enabled": False},
                "analysis_visualization": result.get("analysis_visualization"),
                "turn_by_turn": instructions,
                "route_shape": route_data.get("shape", []) or [],
                "bbox": route_data.get("bbox"),
                "attribution": result.get("attribution"),
                "note": result.get("note"),
            },
            indent=2,
        )
    except ValueError as exc:
        message = str(exc)
        lower = message.lower()
        if "no bikepgh infrastructure data is loaded" in lower or "no routable line geometry" in lower:
            kind = "no_data"
        elif "no continuous path exists" in lower or "unable to split" in lower:
            kind = "no_route"
        else:
            kind = "input"
        return json.dumps(
            {
                "status": "error",
                "kind": kind,
                "message": message,
                "start": start,
                "end": end,
                "city": city or "Pittsburgh, PA",
            }
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "kind": "routing_service",
                "message": str(exc),
                "start": start,
                "end": end,
                "city": city or "Pittsburgh, PA",
            }
        )


@capability("search_all_house_descriptions")
def search_all_house_descriptions(query: str) -> str:
    """Search all house descriptions stored in the vector knowledge base."""
    docs = vs.search_all(query, n_results=6)
    if not docs:
        return tool_result.ToolResult("No documents found in the knowledge base yet.", kind="status")
    try:
        matches_df = pd.DataFrame(
            [
                {
                    "house_id": d["metadata"].get("house_id", "?"),
                    "doc_type": d["metadata"].get("doc_type", "text"),
                    "excerpt": d["text"][:200],
                }
                for d in docs
            ]
        )
        emit_artifact(classify_dataframe(matches_df, title=f'Matches for "{query}"', presentation="table"))
    except Exception:
        pass
    return tool_result.ToolResult(
        "\n\n".join(
            f"[House {d['metadata'].get('house_id', '?')} | {d['metadata'].get('doc_type', 'text')}]\n{d['text'][:300]}"
            for d in docs
        ),
        kind="data",
        row_count=len(docs),
    )


@tool
@capability("get_database_schema")
def get_database_schema(_: str = "") -> str:
    """
    Return the live database schema: every table's actual columns (introspected
    from the running database, so this can't go stale), notes on data-quality
    quirks (which columns are unreliable/NULL-heavy, valid value ranges), and
    the documented join path for every pair of tables that don't share an
    obvious key. Call this before writing SQL against a table you haven't
    already queried successfully earlier in this conversation.
    """
    return tool_result.ToolResult(schema.render_schema_for_agent(), kind="schema")


@tool
@_typed_tool("route")
@capability("find_transit_route")
def find_transit_route(
    start: str,
    end: str,
    city: str = "Pittsburgh, PA",
) -> str:
    """Find a public transit route between two places using the National Transit Map.

    Places may be neighborhoods, landmarks, parks, addresses, or coordinates.
    Finds the shortest path using buses, light rail, trains, and pedestrian transfers.
    Returns the transit lines used, boarding instructions, transfers, distance,
    and estimated travel time.
    """
    try:
        result = _run_async_safely(
            transit_routing.route_transit,
            start.strip(),
            end.strip(),
            city=(city or "Pittsburgh, PA").strip(),
            deadline_s=capabilities.effective_deadline("find_transit_route"),
        )
        if not isinstance(result, dict):
            raise TypeError(f"Transit routing returned an unexpected result type: {type(result).__name__}")
        route_data = result.get("route") or {}
        summary = route_data.get("summary") or {}
        transit_lines = result.get("transit_lines") or []
        instructions = []
        for maneuver in (route_data.get("maneuvers") or [])[:15]:
            instruction = maneuver.get("instruction")
            if instruction:
                instructions.append(instruction)

        start_name = (
            result.get("start", {}).get("display_name", start) if isinstance(result.get("start"), dict) else start
        )
        end_name = result.get("end", {}).get("display_name", end) if isinstance(result.get("end"), dict) else end
        line_names = [
            f"{l.get('route_type_name', 'Line')} {l.get('route_short_name') or l.get('route_long_name')}"
            for l in transit_lines
        ]

        if route_data.get("shape"):
            emit_artifact(
                {
                    "type": "map",
                    "map_kind": "transit_route",
                    "title": f"Transit route: {start_name} \u2192 {end_name}",
                    "city": result.get("city") or city or "Pittsburgh, PA",
                    "start": start_name,
                    "end": end_name,
                    "route_shape": route_data.get("shape", []) or [],
                    "bbox": route_data.get("bbox"),
                    "distance_miles": round(float(summary.get("length", 0) or 0), 2),
                    "duration_minutes": round(float(summary.get("duration_minutes", 0) or 0), 1),
                    "transit_lines_used": transit_lines,
                    "turn_by_turn": instructions,
                    "transfers": result.get("transfers", 0),
                    "provider": result.get("provider"),
                    "attribution": result.get("attribution"),
                }
            )

        return json.dumps(
            {
                "status": "ok",
                "kind": "route_found",
                "message": f"Yes — a transit route was found using {', '.join(line_names) if line_names else 'public transit'}.",
                "presentation": "route_map",
                "start": start_name,
                "end": end_name,
                "city": result.get("city") or city or "Pittsburgh, PA",
                "provider": result.get("provider"),
                "distance_miles": round(float(summary.get("length", 0) or 0), 2),
                "duration_minutes": round(float(summary.get("duration_minutes", 0) or 0), 1),
                "transfers": result.get("transfers", 0),
                "transit_lines_used": transit_lines,
                "turn_by_turn": instructions,
                "route_shape": route_data.get("shape", []) or [],
                "bbox": route_data.get("bbox"),
                "attribution": result.get("attribution"),
                "note": result.get("note"),
            },
            indent=2,
        )
    except ValueError as exc:
        message = str(exc)
        lower = message.lower()
        kind = "no_data" if "no transit routes" in lower else "no_route" if "no continuous" in lower else "input"
        return json.dumps(
            {
                "status": "error",
                "kind": kind,
                "message": message,
                "start": start,
                "end": end,
                "city": city or "Pittsburgh, PA",
            }
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "kind": "routing_service",
                "message": str(exc),
                "start": start,
                "end": end,
                "city": city or "Pittsburgh, PA",
            }
        )


GENERAL_TOOLS = [
    check_data_availability,
    get_database_schema,
    query_database,
    retrieve_data_model_context,
    find_bike_route,
    find_transit_route,
    search_all_house_descriptions,
]
