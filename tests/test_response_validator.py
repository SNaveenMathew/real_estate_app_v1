"""Claim-by-claim grounding tests for ``agents/response_validator.py`` and structured ``ToolResult.evidence``."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402
from agents import answer_status as st  # noqa: E402
from agents import response_validator as rv  # noqa: E402
from agents import tool_result  # noqa: E402


def _msg(content: str, name: str = "query_database") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="1", name=name)


def _evidence_result(rows: list[dict], *, columns=None, tables=("houses",), excluded=(), filters=(), truncated=False):
    cols = list(columns or (list(rows[0].keys()) if rows else []))
    lines = [" ".join(cols)]
    for r in rows:
        lines.append(" ".join(str(r[c]) for c in cols))
    text = "[GENERATED SQL]\nSELECT ...\n[RESULT]\n" + "\n".join(lines)
    ev = {
        "tables_queried": list(tables),
        "columns": cols,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "applied_filters": list(filters),
        "excluded_records": list(excluded),
        "artifact_refs": [],
    }
    return tool_result.ToolResult(text, kind="data", status=st.OK, row_count=len(rows), columns=cols, evidence=ev)


def test_mixed_fabrication_is_caught_even_when_tool_returned_real_data():
    res = _evidence_result([{"city": "Pittsburgh", "avg_price": 350000}])
    reply = "In Pittsburgh, the average price is $350,000, while Boston averages $825,000."
    validated = rv.validate_response(
        reply,
        [HumanMessage(content="Average list price"), _msg(str(res)), AIMessage(content=reply)],
        strict=True,
        tool_results=[("query_database", res)],
    )
    assert "825,000" not in validated and "Boston" not in validated
    assert "350,000" in validated and "Pittsburgh" in validated


def test_reversed_direction_comparison_is_caught_and_corrected():
    res = _evidence_result(
        [
            {"name": "Miami, FL Metro Area", "rfld_risks": 68.0},
            {"name": "Pittsburgh, PA Metro Area", "rfld_risks": 12.5},
        ],
        tables=("census_msa", "cbsa_counties", "nri_tracts"),
    )
    reply = "Pittsburgh has higher riverine flood risk than Miami (12.5 vs 68.0)."
    validated = rv.validate_response(
        reply,
        [HumanMessage(content="Compare flood risk"), _msg(str(res)), AIMessage(content=reply)],
        strict=True,
        tool_results=[("query_database", res)],
    )
    assert "Pittsburgh has higher" not in validated
    assert "Miami" in validated and "68" in validated and "12.5" in validated


def test_score_bounds_are_never_bypassed_by_real_tool_output():
    plain_data = "[GENERATED SQL]\nSELECT walk_score FROM houses\n[RESULT]\nwalk_score\n82\n90"
    reply = "The neighborhood has a Walk Score of 150 out of 100 and another at 82."
    validated = rv.validate_response(
        reply,
        [HumanMessage(content="Walk scores"), _msg(plain_data), AIMessage(content=reply)],
        strict=True,
        tool_results=[("query_database", plain_data)],
    )
    assert "Guardrail note" in validated


def test_excluded_non_arms_length_outlier_cannot_be_cited_as_normal_comp():
    res = _evidence_result(
        [{"median_sold": 340000}],
        tables=("sold_homes",),
        excluded=["Non-arm's-length and <= $1,000 sales excluded by sold_homes filter."],
    )
    bad = rv.validate_response(
        "The comparable sales are $340,000 and $1.",
        [HumanMessage(content="Estimate price"), _msg(str(res)), AIMessage(content="...")],
        strict=True,
        tool_results=[("estimate_price_with_code", res)],
    )
    assert "excluded non-arm's-length" in bad and "340,000" in bad

    good_reply = "The arm's-length median is $340,000 (the $1 non-arm's-length sale was excluded)."
    good = rv.validate_response(
        good_reply,
        [HumanMessage(content="Estimate price"), _msg(str(res)), AIMessage(content=good_reply)],
        strict=True,
        tool_results=[("estimate_price_with_code", res)],
    )
    assert good == good_reply


def test_empty_result_prevents_fabricated_table():
    empty = tool_result.status_result(
        "The query executed successfully and returned 0 rows.",
        st.RESULT_EMPTY,
        ok=True,
        row_count=0,
        columns=["address", "city"],
        evidence={"tables_queried": ["houses"], "columns": ["address", "city"], "rows": [], "row_count": 0},
    )
    fabricated = "| Address | City | Price |\n|---|---|---|\n| 99 Pine St | Seattle | $750,000 |"
    validated = rv.validate_response(
        fabricated,
        [HumanMessage(content="Houses in Seattle"), _msg(str(empty)), AIMessage(content=fabricated)],
        strict=True,
        tool_results=[("query_database", empty)],
    )
    assert "750,000" not in validated and "No matching records" in validated


def test_valid_rounded_and_user_filter_figures_pass_cleanly():
    res = _evidence_result(
        [{"name": "Pittsburgh, PA Metro Area", "population": 2370930}],
        filters=["year = 2024"],
    )
    reply = "In 2024, the Pittsburgh metro population is 2.37 million (2,370,930)."
    validated = rv.validate_response(
        reply,
        [HumanMessage(content="Population in 2024"), _msg(str(res)), AIMessage(content=reply)],
        strict=True,
        tool_results=[("query_database", res)],
    )
    assert validated == reply


def test_query_database_populates_tool_result_evidence_end_to_end(tmp_path_factory, monkeypatch):
    from agents import tools

    fx = census_fixture.activate(tmp_path_factory.mktemp("ev_fx"))
    try:
        monkeypatch.setattr(tools, "generate_sql", lambda *a, **k: "")
        out = tools.query_database.invoke({"request": "How many houses are in Pittsburgh?"})
        assert isinstance(out, tool_result.ToolResult)
        facts = out.facts()
        assert facts["ok"] is True and facts["row_count"] == 1
        assert facts["evidence"]["tables_queried"] == ["houses"]
        assert facts["evidence"]["rows"]
        # Second call hits semantic cache and still preserves ToolResult.evidence
        cached_out = tools.query_database.invoke({"request": "How many houses are in Pittsburgh?"})
        assert isinstance(cached_out, tool_result.ToolResult)
        assert cached_out.facts()["evidence"]["rows"] == facts["evidence"]["rows"]
    finally:
        census_fixture.deactivate(fx)
