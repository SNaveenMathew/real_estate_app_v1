"""
response_validator.py

Detects hallucinated data in agent responses by checking whether the reply
is grounded in actual tool outputs from the same conversation turn.

Catches:
  - LLM ignoring empty-table errors and fabricating result tables
  - SQL with invented column names failing, then LLM fabricating anyway
  - No tools called but reply contains specific figures

Does NOT flag:
  - Replies grounded in real query_database results
  - Replies that are pure prose explanations
  - Replies where the tool returned actual data rows
"""

import re
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_numbers(text: str) -> set[str]:
    cleaned = re.sub(r'(\d),(\d)', r'\1\2', text)
    return set(re.findall(r'\b\d+\.?\d*\b', cleaned))


def _extract_tool_outputs(messages: list[BaseMessage]) -> list[tuple[str, str]]:
    """Return list of (tool_name, content) for every ToolMessage."""
    return [
        (m.name or "tool", str(m.content))
        for m in messages
        if isinstance(m, ToolMessage)
    ]


def _has_markdown_table(text: str) -> bool:
    lines = text.splitlines()
    pipe_lines = [l for l in lines if l.strip().startswith("|") and "|" in l[1:]]
    return len(pipe_lines) >= 2


def _tool_returned_real_data(tool_outputs: list[tuple[str, str]]) -> bool:
    """
    Return True only if a tool returned an actual query result — i.e. output
    that looks like a pandas DataFrame.to_string() with column headers and rows.

    This is stricter than "has digits": check_data_availability also returns
    multi-line text with digits, but it's a status report, not a query result.
    We need to distinguish those.

    Heuristics for a real query result:
      - NOT starting with an error prefix
      - Has at least 2 lines
      - The first non-empty line looks like column headers (contains 2+ words
        separated by spaces, no ":" at the end — rules out "Column: value" style)
      - A subsequent line has numeric values in the same column positions
    """
    for name, content in tool_outputs:
        c = content.strip()
        if not c:
            continue
        # Exclude known error/status outputs
        c_lower = c.lower()
        if (c.startswith("ERROR:") or c.startswith("EMPTY TABLE")
                or "SQL Error" in c or "BinderException" in c
                or c_lower.startswith("error:") or c_lower.startswith("sql error")
                or "binder error" in c_lower):
            continue
        # check_data_availability returns lines like "  ✓  houses   939 rows"
        # These should NOT count as query results
        if "rows" in c_lower and ("✓" in c or "✗" in c or "loaded" in c_lower):
            continue

        lines = [l for l in c.splitlines() if l.strip()]
        if len(lines) < 2:
            continue

        # A real query result: first line has spaced column names, subsequent
        # lines have values. The simplest signal: the tool is query_database
        # AND the output has ≥2 lines AND contains digits.
        if name == "query_database" and re.search(r'\d', c):
            return True

        # For other tools (get_top_msas_by_flood_risk etc.) — must have
        # digit-containing lines that look like data rows (not just counts)
        digit_lines = [l for l in lines if re.search(r'\b\d{2,}\b', l)]
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
⚠️ **Query failed — wrong column or table name in the SQL.**

{error_detail}

**The correct schema is:**
- `houses`: house_id, address, city, state, zip, lat, lon, price, beds, baths, sqft,
  walk_score, bike_score, transit_score, tract_fips, msa_code
- Join to MSA names: `JOIN census_msa m ON h.msa_code = m.msa_code`
- Join to CBSA:      `JOIN cbsa_counties cb ON h.msa_code = cb.cbsa_code`

There is **no** `msa_name` column in houses — use `city` for city-level grouping,
or join to `census_msa.name` for metro area names.

**Example — walk score by metro area:**
```sql
SELECT m.name AS metro,
       COUNT(*) AS total_houses,
       ROUND(100.0 * SUM(CASE WHEN h.walk_score >= 70 THEN 1 ELSE 0 END)
             / COUNT(*), 1) AS pct_walkable
FROM houses h
JOIN census_msa m ON h.msa_code = m.msa_code
WHERE h.walk_score IS NOT NULL
GROUP BY m.name
ORDER BY pct_walkable DESC;
```
"""

HALLUCINATION_NO_TOOLS = """\
⚠️ **No data was fetched for this query.**

The answer above was not backed by a database query.
Please ask again — the agent will use `query_database` to get real numbers.
"""


def validate_response(
    reply: str,
    all_messages: list[BaseMessage],
    strict: bool = True,
) -> str:
    """
    Validate the agent's reply against actual tool outputs.

    Returns the original reply if grounded, or a specific replacement message
    explaining what went wrong (SQL error vs missing data vs no tools called).
    """
    tool_outputs = _extract_tool_outputs(all_messages)
    tool_text_combined = " ".join(c for _, c in tool_outputs)

    # ── Fast pass: a tool returned a real query result → reply is grounded ─
    if _tool_returned_real_data(tool_outputs):
        return reply

    # ── Determine failure type ─────────────────────────────────────────────
    no_tools_called = not tool_outputs
    failure_type    = _classify_tool_failure(tool_outputs)
    tools_failed    = failure_type is not None or _tool_output_signals_failure(tool_outputs)
    reply_has_data  = _reply_claims_data(reply)

    hallucinated = False
    if tools_failed and reply_has_data:
        hallucinated = True
    elif no_tools_called and reply_has_data:
        hallucinated = True
    elif tool_outputs and reply_has_data:
        # Number grounding check
        reply_numbers = _extract_numbers(reply)
        tool_numbers  = _extract_numbers(tool_text_combined)
        ungrounded    = reply_numbers - tool_numbers
        if (len(reply_numbers) > 5
                and len(ungrounded) > 0.6 * len(reply_numbers)):
            hallucinated = True
            failure_type = "missing_data"

    if not hallucinated:
        return reply

    # ── Return the right replacement message ──────────────────────────────
    if not strict:
        return reply + "\n\n---\n⚠️ **Validation warning:** reply may not reflect loaded data."

    if no_tools_called:
        return HALLUCINATION_NO_TOOLS

    if failure_type and failure_type.startswith("sql_error:"):
        error_detail = failure_type[len("sql_error:"):]
        return HALLUCINATION_SQL_ERROR.format(error_detail=f"**SQL Error:** `{error_detail}`")

    # Missing data or unknown failure — show what the tools actually returned
    tool_summary_parts = []
    for name, content in tool_outputs:
        snippet = content[:500].strip()
        if snippet:
            tool_summary_parts.append(
                f"- Tool `{name}` returned:\n  ```\n  {snippet}\n  ```"
            )
    tool_summary = ("\n".join(tool_summary_parts)
                    if tool_summary_parts else "_(No tool output)_")
    return HALLUCINATION_MISSING_DATA.format(tool_summary=tool_summary)
