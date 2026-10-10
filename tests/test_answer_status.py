"""agents/answer_status.py and the response validator's handling of deliberate non-answers."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from agents import answer_status as st  # noqa: E402
from agents import response_validator as rv  # noqa: E402


def test_round_trip_and_envelope_parsing():
    text = st.not_answered(st.PLACE_AMBIGUOUS, "  which one?  ")
    assert text == "NOT_ANSWERED[place_ambiguous]: which one?"
    assert st.status_code(text) == "place_ambiguous" and st.status_message(text) == "which one?"
    wrapped = f"[GENERATED SQL]\n(none)\n[RESULT]\n{text}"
    assert st.status_code(wrapped) == "place_ambiguous" and st.status_message(wrapped) == "which one?"


def test_multiline_messages_survive():
    text = st.not_answered(st.UNPLANNABLE, "line one.\nline two.")
    assert st.status_message(f"[RESULT]\n{text}") == "line one.\nline two."


@pytest.mark.parametrize("not_a_status", [
    "", None, 42, "ordinary result text", "NOT_ANSWERED", "NOT_ANSWERED[bad code]: x",
    "[RESULT]\n  name  value\n  NOT_ANSWERED[unplannable]: buried in a data row",     # must sit right after the header
    "a data cell that says NOT_ANSWERED[unplannable]: inside a sentence",
])
def test_only_a_real_status_line_is_a_status(not_a_status):
    assert st.status_code(not_a_status) is None and st.status_message(not_a_status) == "" and not st.is_not_answered(not_a_status)


def test_unknown_codes_are_rejected_at_creation():
    with pytest.raises(ValueError):
        st.not_answered("made_up", "x")


def test_only_requests_a_rewording_could_fix_are_retryable():
    assert {c for c in st.CODES if st.is_retryable(st.not_answered(c, "x"))} == {st.UNPLANNABLE, st.PLACE_MISSING}
    assert not st.is_retryable("Code Agent error: boom")


# --------------------------------------------------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------------------------------------------------
TABLE_REPLY = ("| Metro | Population |\n|---|---|\n| A | 1,100,200 |\n| B | 2,200,300 |\n| C | 3,300,400 |\n"
               "| D | 4,400,500 |\n| E | 5,500,600 |\n")


def tool_msg(content, name="query_database"):
    return ToolMessage(content=content, tool_call_id="1", name=name)


def non_answer(code=st.PLACE_AMBIGUOUS, msg="Portland fits two places; ask which."):
    return f"[GENERATED SQL]\n(no SQL)\n[RESULT]\n{st.not_answered(code, msg)}"


def test_a_non_answer_with_digits_is_still_not_data():
    out = non_answer(st.DATA_UNAVAILABLE, "Coverage is 93% of 700 tracts, too low (2020 vs 2010 vintages).")
    assert rv._tool_returned_real_data([("query_database", out)]) is False


def test_real_data_is_still_data():
    out = "[GENERATED SQL]\nSELECT 1\n[RESULT]\n name  population\n A  1200000\n B  2200000"
    assert rv._tool_returned_real_data([("query_database", out)]) is True


def test_invented_numbers_after_a_non_answer_are_replaced_by_the_tools_own_reason():
    reply = rv.validate_response(TABLE_REPLY, [AIMessage(content="x"), tool_msg(non_answer())])
    assert reply.startswith("⚠️ **This question could not be answered from the loaded data.**")
    assert "Portland fits two places; ask which." in reply and "1,100,200" not in reply
    assert "setup_data.py" not in reply          # NOT the generic "tables are not loaded, run setup" message


def test_an_honest_reply_after_a_non_answer_passes_through():
    honest = "Which Portland do you mean?"
    assert rv.validate_response(honest, [tool_msg(non_answer())]) == honest


def test_lenient_mode_warns_instead_of_replacing():
    reply = rv.validate_response(TABLE_REPLY, [tool_msg(non_answer())], strict=False)
    assert reply.startswith(TABLE_REPLY) and "Validation warning" in reply


def test_real_data_elsewhere_in_the_turn_keeps_the_reply_grounded():
    data = "[GENERATED SQL]\nSELECT 1\n[RESULT]\n name  population\n A  1100200\n B  2200300"
    assert rv.validate_response(TABLE_REPLY, [tool_msg(non_answer()), tool_msg(data)]) == TABLE_REPLY


def test_other_failure_paths_are_unchanged():
    # no tools at all, and a SQL error, still produce their original replacements
    assert rv.validate_response(TABLE_REPLY, []) == rv.HALLUCINATION_NO_TOOLS
    sql_error = "SQL Error: Binder Error: column nope does not exist"
    assert "SQL" in rv.validate_response(TABLE_REPLY, [tool_msg(sql_error)])


def test_message_templates_are_digit_free():
    """Templates (not data-dependent parts) must never contain digits - see agents/answer_status.py."""
    import agents.answer_policy as policy
    import inspect
    src = inspect.getsource(policy)
    literals = re.findall(r'f?"([^"\n]{20,})"', src)
    status_literals = [s for s in literals if "database error" in s or "No query was run" in s]
    assert status_literals and not any(re.search(r"\d", re.sub(r"\{[^}]*\}", "", s)) for s in status_literals)
