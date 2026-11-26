"""
response_validator.py

Claim-by-claim verification of agent responses against structured tool evidence
and raw tool outputs from the same conversation turn (zero blind fast-pass).

Gates:
  1. Conflict / Non-answer / Empty-evidence check
  2. Table / Provenance / Artifact integrity check
  3. Quantitative number & entity claim grounding (structured ToolEvidence or text)
  4. Multi-entity direction comparison consistency + unconditional score bounds
  5. Deterministic grounded fallback when mixed fabrication or contradiction occurs
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agents import answer_status, tool_result


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_numbers(text: str) -> set[str]:
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", text)
    return set(re.findall(r"\b\d+\.?\d*\b", cleaned))


def _extract_tool_outputs(messages: list[BaseMessage]) -> list[tuple[str, str]]:
    """Return list of (tool_name, content) for every ToolMessage."""
    return [(m.name or "tool", str(m.content)) for m in messages if isinstance(m, ToolMessage)]


def _typed_results(tool_results) -> list[tuple[str, dict]]:
    """The (name, facts) pairs for results that arrived as a ToolResult."""
    out: list[tuple[str, dict]] = []
    for name, value in tool_results or ():
        facts = tool_result.facts(value)
        if facts is not None:
            out.append((name, facts))
    return out


def _structured_evidences(tool_results) -> list[dict[str, Any]]:
    """Return non-empty structured ``ToolResult.evidence`` payloads from successful evidence-bearing tools."""
    out: list[dict[str, Any]] = []
    for name, facts in _typed_results(tool_results):
        if name in _NON_RESULT_TOOLS or not facts.get("ok"):
            continue
        if facts.get("kind") in tool_result.NON_EVIDENCE_KINDS and facts.get("status") != answer_status.RESULT_EMPTY:
            continue
        ev = facts.get("evidence")
        if isinstance(ev, dict) and ev:
            out.append(ev)
    return out


def _has_markdown_table(text: str) -> bool:
    lines = text.splitlines()
    pipe_lines = [l for l in lines if l.strip().startswith("|") and "|" in l[1:]]
    return len(pipe_lines) >= 2


# Tool names whose output is a description/status report, never a query
# result — excluded by name so no amount of digits in a schema note (FIPS
# examples, hazard scores, row counts) can be mistaken for grounding data.
_NON_RESULT_TOOLS = {"get_database_schema", "check_data_availability"}

# query_database's own "this wasn't a result" messages. Guards the loose
# "any digit → real data" fast path below, which otherwise misfires on
# these — e.g. "Query returned 0 rows..." contains a digit ('0') and would
# otherwise be wrongly treated as a grounded result.
_NON_DATA_QUERY_PREFIXES = ("query returned 0 rows", "0 rows", "no results")


def _tool_returned_real_data(tool_outputs: list[tuple[str, str]], tool_results=None) -> bool:
    """
    Return True only if a tool returned an actual query result — i.e. output
    that looks like a pandas DataFrame.to_string() with column headers and rows.

    This is stricter than "has digits": query_database's own 0-row/diagnostic
    messages, and get_database_schema's notes, can both contain plenty of
    digits (FIPS examples, row counts) without being a result table. We need
    to distinguish those.

    Heuristics for a real query result:
      - Not from a schema/status tool (_NON_RESULT_TOOLS)
      - NOT starting with an error/non-data prefix
      - Has at least 2 lines
      - Has digit-containing line(s) that look like data rows
    """
    for name, facts in _typed_results(tool_results):  # authoritative: the tool said what it produced
        if facts["ok"] and facts["kind"] not in tool_result.NON_EVIDENCE_KINDS:
            return True

    for name, content in tool_outputs:
        if name in _NON_RESULT_TOOLS:
            continue
        c = content.strip()
        if not c:
            continue
        if answer_status.is_not_answered(c):
            continue  # a deliberate non-answer (agents/answer_status.py) is never data, whatever digits it holds
        # Exclude known error/status outputs
        c_lower = c.lower()
        if (
            c.startswith("ERROR:")
            or c.startswith("EMPTY TABLE")
            or "SQL Error" in c
            or "BinderException" in c
            or c_lower.startswith("error:")
            or c_lower.startswith("sql error")
            or "binder error" in c_lower
        ):
            continue

        lines = [l for l in c.splitlines() if l.strip()]
        if len(lines) < 2:
            continue

        # A real query result: first line has spaced column names, subsequent
        # lines have values. The simplest signal: the tool is query_database
        # AND the output has ≥2 lines AND contains digits — EXCEPT for
        # query_database's own non-data messages (0 rows, etc.), which also
        # tend to contain digits (a row count, a sample FIPS code in a hint)
        # without being a result.
        if name == "query_database" and re.search(r"\d", c) and not c_lower.startswith(_NON_DATA_QUERY_PREFIXES):
            return True

        # For other tools — must have digit-containing lines that look like
        # data rows (not just an incidental number in a status line)
        digit_lines = [l for l in lines if re.search(r"\b\d{2,}\b", l)]
        if len(digit_lines) >= 2:
            return True

    return False


def _classify_tool_failure(tool_outputs: list[tuple[str, str]]) -> str | None:
    """
    If tools failed, return a human-readable reason string.
    Returns None if tools appear to have succeeded.
    """
    for name, content in tool_outputs:
        c = content.strip()
        if "SQL Error" in c or "BinderException" in c or "Binder Error" in c:
            # Extract the useful part of the SQL error
            lines = c.splitlines()
            error_line = next((l for l in lines if "Error" in l), c[:200])
            return f"sql_error:{error_line.strip()}"
        if c.lower().startswith("error:") or "empty tables detected" in c.lower():
            return "missing_data"
        if "not loaded yet" in c.lower() or "do not query" in c.lower():
            return "missing_data"
    return None


# These phrases indicate the tool itself failed — they must appear at the
# START of the tool output or on their own line, not embedded in data rows.
_ERROR_PREFIXES = [
    "error:",
    "empty tables detected",
    "sql error",
    "binder error",
    "✗ empty",
    "0 rows",
]

_ERROR_SIGNALS = [
    "empty table detected",
    "do not query",
    "cannot answer",
    "column not found",
    "no results returned",
    "not loaded yet",
]


def _tool_output_signals_failure(tool_outputs: list[tuple[str, str]]) -> bool:
    """
    Return True only if tool outputs unambiguously signal failure —
    i.e. the content starts with an error prefix or contains a clear failure
    signal on its own line (not buried in a data row).
    """
    for name, content in tool_outputs:
        c = content.strip().lower()
        # Check prefixes (start of output)
        for prefix in _ERROR_PREFIXES:
            if c.startswith(prefix):
                return True
        # Check whole-line signals
        for line in c.splitlines():
            line = line.strip()
            for sig in _ERROR_SIGNALS:
                if line == sig or line.startswith(sig):
                    return True
    return False


def _reply_claims_data(reply: str) -> bool:
    if _has_markdown_table(reply):
        return True
    numbers = _extract_numbers(reply)
    if len(numbers) >= 5:
        return True
    return False


# ── Main validator ────────────────────────────────────────────────────────────

HALLUCINATION_MISSING_DATA = """\
⚠️ **Data not available for this query.**

The required tables are not loaded yet:

{tool_summary}

**What to do:** run `python setup_data.py` then ask again.
"""

HALLUCINATION_SQL_ERROR = """\
⚠️ **Query failed — the SQL referenced a column or table that doesn't exist.**

{error_detail}

Please ask again — the assistant checks the live schema (`get_database_schema`)
before retrying, so it won't guess at column names a second time.
"""

HALLUCINATION_NOT_ANSWERED = """\
⚠️ **This question could not be answered from the loaded data.**

{detail}
"""

HALLUCINATION_NO_TOOLS = """\
⚠️ **No data was fetched for this query.**

The answer above was not backed by a database query.
Please ask again — the agent will use `query_database` to get real numbers.
"""


HALLUCINATION_EMPTY_DATA = """\
⚠️ **No matching records were found for that query in the loaded data.**

The query succeeded cleanly with 0 matching rows; no values exist to tabulate or compare.
"""

_SCALE_FACTORS = {"k": 1_000.0, "thousand": 1_000.0, "m": 1_000_000.0, "million": 1_000_000.0, "b": 1_000_000_000.0}
_HIGHER_WORDS = r"(?:higher|greater|more|larger|bigger|riskier|more\s+expensive|priciest|exceeds|above)"
_LOWER_WORDS = r"(?:lower|less|smaller|fewer|cheaper|safer|below|under)"
_LIST_ITEM_NUM_RE = re.compile(r"^\s*\d{1,2}[.)]\s+")


def _expand_numeric_variants(val: float, allowed: set[float]) -> None:
    allowed.add(val)
    allowed.add(abs(val))
    for scale in (1_000.0, 1_000_000.0, 1_000_000_000.0):
        allowed.add(val / scale)
    allowed.add(val * 100.0)
    if val != 0:
        allowed.add((val - 1.0) * 100.0)  # growth multiples like 1.15 -> 15%
    for digits in (0, 1, 2):
        allowed.add(round(val, digits))


def _allowed_numeric_values(
    evidences: list[dict[str, Any]], tool_text_combined: str, user_text: str = ""
) -> set[float]:
    allowed: set[float] = set()
    for ev in evidences:
        rc = ev.get("row_count")
        if isinstance(rc, (int, float)):
            allowed.add(float(rc))
        for row in ev.get("rows") or ():
            if not isinstance(row, dict):
                continue
            for v in row.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    _expand_numeric_variants(float(v), allowed)
        for ftext in ev.get("applied_filters") or ():
            for num_str in _extract_numbers(str(ftext)):
                try:
                    _expand_numeric_variants(float(num_str), allowed)
                except ValueError:
                    pass
    for extra_text in (tool_text_combined, user_text):
        for num_str in _extract_numbers(extra_text):
            try:
                _expand_numeric_variants(float(num_str), allowed)
            except ValueError:
                pass
    return allowed


def _extract_claim_numbers(reply: str) -> list[float]:
    """Extract quantitative numeric claims from ``reply`` (handling $ and K/M/B suffixes)."""
    cleaned_lines = [_LIST_ITEM_NUM_RE.sub("", line) for line in reply.splitlines()]
    text = re.sub(r"(\d),(\d)", r"\1\2", "\n".join(cleaned_lines))
    claims: list[float] = []
    for m in re.finditer(r"(\$?)(\d+(?:\.\d+)?)\s*([kmb]|thousand|million)?\b", text, re.I):
        raw_num, dollar, suffix = float(m.group(2)), bool(m.group(1)), (m.group(3) or "").lower()
        val = raw_num * _SCALE_FACTORS.get(suffix, 1.0)
        # Skip small conversational list indices/ordinals (1..10) unless formatted as currency or decimal
        if val < 10 and not dollar and "." not in m.group(2) and not suffix:
            continue
        claims.append(val)
    return claims


def _value_matches_allowed(val: float, allowed: set[float]) -> bool:
    for ref in allowed:
        tol = max(0.05, abs(ref) * 0.02)
        if abs(val - ref) <= tol:
            return True
    return False


def _ungrounded_structured_claims(
    reply: str, evidences: list[dict[str, Any]], tool_text_combined: str, user_text: str = ""
) -> list[float]:
    rows = [r for ev in evidences for r in (ev.get("rows") or ()) if isinstance(r, dict)]
    if not rows:
        return []
    allowed = _allowed_numeric_values(evidences, tool_text_combined, user_text=user_text)
    if not allowed:
        return []
    claims = _extract_claim_numbers(reply)
    if not claims:
        return []
    any_truncated = any(bool(ev.get("truncated")) for ev in evidences)
    ungrounded = [c for c in claims if not _value_matches_allowed(c, allowed)]
    if not any_truncated and ungrounded:
        # Even a single clear fabricated price/metric outside allowed tolerance is mixed fabrication.
        significant = [c for c in ungrounded if c >= 10 or (c * 10) % 10 != 0]
        return significant
    if any_truncated and len(ungrounded) > max(1, int(0.4 * len(claims))):
        return ungrounded
    return []


def _excluded_outlier_violation(reply: str, evidences: list[dict[str, Any]]) -> bool:
    """Catch replies presenting a $1 / non-arm's-length excluded record as a normal comp."""
    has_exclusion = any(ev.get("excluded_records") for ev in evidences)
    if not has_exclusion:
        return False
    low = reply.lower()
    if re.search(r"\$1\b(?:\.00)?", reply) and not any(
        w in low for w in ("exclud", "non-arm", "non arm", "filter", "not used", "omit", "ignored")
    ):
        return True
    return False


def _row_label_and_metrics(row: dict[str, Any]) -> tuple[str | None, dict[str, float]]:
    label: str | None = None
    metrics: dict[str, float] = {}
    for k, v in row.items():
        if isinstance(v, str) and v.strip() and label is None and not k.lower().endswith("date"):
            label = v.strip()
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            metrics[k.lower()] = float(v)
    return label, metrics


def _short_entity_aliases(label: str) -> list[str]:
    aliases = [label.lower()]
    head = label.split(",", 1)[0].strip().lower()
    if head and head not in aliases:
        aliases.append(head)
    first = head.split("-", 1)[0].strip()
    if first and len(first) >= 4 and first not in aliases:
        aliases.append(first)
    return aliases


def _check_direction_consistency(reply: str, evidences: list[dict[str, Any]]) -> str | None:
    """Detect unambiguous inverted pairwise comparisons against structured ``evidence`` rows."""
    rows_by_alias: dict[str, dict[str, float]] = {}
    for ev in evidences:
        for row in ev.get("rows") or ():
            if not isinstance(row, dict):
                continue
            label, metrics = _row_label_and_metrics(row)
            if not label or not metrics:
                continue
            for alias in _short_entity_aliases(label):
                rows_by_alias.setdefault(alias, metrics)
    if len(rows_by_alias) < 2:
        return None
    low = reply.lower()
    present = sorted({a for a in rows_by_alias if re.search(rf"\b{re.escape(a)}\b", low)}, key=len, reverse=True)
    if len(present) < 2:
        return None

    for a in present:
        for b in present:
            if a == b or rows_by_alias[a] is rows_by_alias[b]:
                continue
            higher_pat = rf"\b{re.escape(a)}\b[^.!?\n]{{0,60}}\b{_HIGHER_WORDS}\b[^.!?\n]{{0,40}}\bthan\s+(?:the\s+)?\b{re.escape(b)}\b"
            lower_pat = rf"\b{re.escape(a)}\b[^.!?\n]{{0,60}}\b{_LOWER_WORDS}\b[^.!?\n]{{0,40}}\bthan\s+(?:the\s+)?\b{re.escape(b)}\b"
            claimed_higher = bool(re.search(higher_pat, low))
            claimed_lower = bool(re.search(lower_pat, low))
            if not claimed_higher and not claimed_lower:
                continue
            ma, mb = rows_by_alias[a], rows_by_alias[b]
            shared_cols = [c for c in ma if c in mb]
            if not shared_cols:
                continue
            # Require the comparison direction to be supported by at least one shared metric column.
            if len(shared_cols) == 1:
                c = shared_cols[0]
                if claimed_higher and ma[c] < mb[c]:
                    return f"reversed comparison: {a} ({ma[c]}) < {b} ({mb[c]}) on {c}"
                if claimed_lower and ma[c] > mb[c]:
                    return f"reversed comparison: {a} ({ma[c]}) > {b} ({mb[c]}) on {c}"
            else:
                if claimed_higher and not any(ma[c] > mb[c] for c in shared_cols):
                    return f"reversed comparison: {a} is not higher than {b} on any returned metric"
                if claimed_lower and not any(ma[c] < mb[c] for c in shared_cols):
                    return f"reversed comparison: {a} is not lower than {b} on any returned metric"
    return None


def _render_grounded_fallback(evidences: list[dict[str, Any]], reason: str) -> str:
    """Render a deterministic grounded answer from structured ToolEvidence rows."""
    lines: list[str] = [f"Based on the verified tool results ({reason}):", ""]
    for ev in evidences:
        rows = ev.get("rows") or []
        cols = ev.get("columns") or (list(rows[0].keys()) if rows else [])
        if rows and cols:
            lines.append("| " + " | ".join(str(c) for c in cols) + " |")
            lines.append("|" + "|".join("---" for _ in cols) + "|")
            for row in rows[:15]:
                vals = []
                for c in cols:
                    v = row.get(c, "")
                    if isinstance(v, float):
                        vals.append(f"{v:,.2f}" if v != int(v) else f"{int(v):,}")
                    elif isinstance(v, int) and not isinstance(v, bool):
                        vals.append(f"{v:,}")
                    else:
                        vals.append(str(v))
                lines.append("| " + " | ".join(vals) + " |")
            lines.append("")
        for note in ev.get("excluded_records") or ():
            lines.append(f"- Note: {note}")
    return "\n".join(lines).strip()


# Prefixes that mark a reply as a resilience/degradation sentinel produced by the agent's
# own outage-handling path (agents/llm_resilience.py).  These messages are already the
# correct user-facing output — running them through the evidence validator would
# (a) find them "ungrounded" (they contain no tool-query data) and replace them with an
# unhelpful HALLUCINATION template, (b) destroy the deliberate degradation UX.
_DEGRADATION_PREFIXES: tuple[str, ...] = (
    "(The language model service is not responding",
    "The language model service is not responding",
)


def validate_response(
    reply: str,
    all_messages: list[BaseMessage],
    strict: bool = True,
    tool_results=None,
    artifacts=None,
) -> str:
    """
    Validate the agent's reply claim-by-claim against structured ``ToolResult.evidence``
    and tool outputs across five sequential deterministic gates (zero fast-pass bypass).
    """
    # Gate 0: pass degradation / resilience sentinel replies straight through — they are
    # already the correct output from the outage-handling path and must not be rewritten.
    if any(reply.startswith(p) for p in _DEGRADATION_PREFIXES):
        return reply

    tool_outputs = _extract_tool_outputs(all_messages)
    tool_text_combined = " ".join(c for _, c in tool_outputs)
    user_text = " ".join(str(m.content) for m in all_messages if isinstance(m, HumanMessage))
    typed = _typed_results(tool_results)
    evidences = _structured_evidences(tool_results)
    has_real_evidence = _tool_returned_real_data(tool_outputs, tool_results) or any(ev.get("rows") for ev in evidences)

    # ── Gate 1: Conflict / non-answer / missing-evidence gating ────────────
    no_tools_called = not tool_outputs and not typed
    failure_type = _classify_tool_failure(tool_outputs)
    tools_failed = failure_type is not None or _tool_output_signals_failure(tool_outputs)
    reply_has_data = _reply_claims_data(reply)
    not_answered = next((c for _, c in tool_outputs if answer_status.is_not_answered(c)), None)
    typed_failure = next((facts for _, facts in typed if not facts["ok"]), None)

    hallucinated = False
    mixed_reason: str | None = None
    all_empty = (
        evidences
        and all((ev.get("row_count") == 0 or not ev.get("rows")) for ev in evidences)
        and not any(facts.get("ok") and (facts.get("row_count") or 0) > 0 for _, facts in typed)
    )

    if typed_failure is not None and not has_real_evidence and reply_has_data:
        hallucinated = True
        failure_type = "not_answered" if typed_failure["kind"] == "not_answered" else "missing_data"
    elif not_answered is not None and not has_real_evidence and reply_has_data:
        hallucinated = True
        failure_type = "not_answered"
    elif tools_failed and not has_real_evidence and reply_has_data:
        hallucinated = True
    elif no_tools_called and reply_has_data:
        hallucinated = True
    elif all_empty and (_has_markdown_table(reply) or len(_extract_claim_numbers(reply)) >= 2):
        # Gate 2: empty query result yet model fabricated a table/records
        hallucinated = True
        failure_type = "empty_data"

    # ── Gate 2 & 3: Claim-by-claim table / exclusion / number grounding ────
    if not hallucinated and evidences and has_real_evidence:
        if _excluded_outlier_violation(reply, evidences):
            mixed_reason = "reply cited an excluded non-arm's-length outlier without noting exclusion"
        else:
            bad_nums = _ungrounded_structured_claims(reply, evidences, tool_text_combined, user_text=user_text)
            if bad_nums:
                mixed_reason = f"ungrounded figure(s): {', '.join(str(x) for x in bad_nums[:3])}"
    elif not hallucinated and tool_outputs and reply_has_data:
        # Legacy string-only tool outputs (no structured ToolEvidence available)
        reply_numbers = _extract_numbers(reply)
        tool_numbers = _extract_numbers(tool_text_combined)
        ungrounded = reply_numbers - tool_numbers
        if len(reply_numbers) > 5 and len(ungrounded) > 0.6 * len(reply_numbers):
            if has_real_evidence:
                mixed_reason = "ungrounded numbers against tool output"
            else:
                hallucinated = True
                failure_type = "missing_data"

    # ── Gate 4: Multi-entity direction consistency + unconditional score check
    if not hallucinated and not mixed_reason and evidences:
        dir_err = _check_direction_consistency(reply, evidences)
        if dir_err:
            mixed_reason = dir_err

    from services.guardrails import OutputGroundingGuardrail

    score_res = OutputGroundingGuardrail.validate_scores_in_text(reply)

    # ── Gate 5: Artifacts / Strict Repair / Failure Replacement ────────────
    if mixed_reason:
        if not strict:
            return reply + f"\n\n---\n⚠️ **Validation warning:** {mixed_reason}."
        if any(ev.get("rows") for ev in evidences):
            return _render_grounded_fallback(evidences, mixed_reason)
        return HALLUCINATION_MISSING_DATA.format(tool_summary=f"- {mixed_reason}")

    if not hallucinated:
        if not score_res.passed:
            return reply + f"\n\n*(Guardrail note: {'; '.join(score_res.reasons)})*"
        return reply

    if not strict:
        return reply + "\n\n---\n⚠️ **Validation warning:** reply may not reflect loaded data."

    if no_tools_called:
        return HALLUCINATION_NO_TOOLS

    if failure_type == "empty_data":
        return HALLUCINATION_EMPTY_DATA

    if failure_type == "not_answered":
        return HALLUCINATION_NOT_ANSWERED.format(detail=answer_status.status_message(not_answered))

    if failure_type and failure_type.startswith("sql_error:"):
        error_detail = failure_type[len("sql_error:") :]
        return HALLUCINATION_SQL_ERROR.format(error_detail=f"**SQL Error:** `{error_detail}`")

    # Missing data or unknown failure — show what the tools actually returned
    tool_summary_parts = []
    for name, content in tool_outputs:
        snippet = content[:500].strip()
        if snippet:
            tool_summary_parts.append(f"- Tool `{name}` returned:\n  ```\n  {snippet}\n  ```")
    tool_summary = "\n".join(tool_summary_parts) if tool_summary_parts else "_(No tool output)_"
    return HALLUCINATION_MISSING_DATA.format(tool_summary=tool_summary)
