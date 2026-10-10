"""General Chat end to end: the layers must cooperate, not just work alone.

Only the language models are stubbed (code agent, SQL agent, answer agent).  The planner, policy, provider, validator,
real input guardrail and orchestration loop are all the production code.
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402

REPORTED = "Verify the land areas of Pittsburgh and Indianapolis"


class Scripted:
    """A model stub: pops queued replies (empty completion when exhausted) and records every prompt it was shown."""

    def __init__(self, replies=()):
        self.replies, self.prompts = list(replies), []

    def invoke(self, messages):
        self.prompts.append("\n".join(str(getattr(m, "content", m)) for m in messages))
        return types.SimpleNamespace(content=self.replies.pop(0) if self.replies else "")


class EvidenceReader:
    """An answer model that only restates numbers it can see in the evidence (an honest one)."""

    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        evidence = str(messages[0].content).split("EXECUTED EVIDENCE:", 1)[-1]
        self.prompts.append(evidence)
        nums = re.findall(r"\b\d+\.\d+\b", evidence)
        return types.SimpleNamespace(content=("Land areas: " + ", ".join(nums)) if nums else "I could not compute that.")


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("orchestration"))
    yield f
    census_fixture.deactivate(f)


@pytest.fixture()
def models(fx, monkeypatch):
    import agents.general_agent as ga
    import agents.tools as tools
    code, sql, answer = Scripted(), Scripted(), EvidenceReader()
    monkeypatch.setattr(ga, "_get_code_agent", lambda: code)
    monkeypatch.setattr(ga, "_get_response_agent", lambda: answer)
    monkeypatch.setattr(tools, "_agent", sql)
    monkeypatch.setattr(tools.vs, "search_data_model", lambda *a, **k: [])      # no vector store / embeddings in tests
    return types.SimpleNamespace(code=code, sql=sql, answer=answer)


def chat(message):
    from agents.general_agent import run_general_chat
    reply, _history = run_general_chat(message)
    return reply


def program(request, requirements=""):
    return f"final_result = query_database(request={request!r}, requirements={requirements!r})"


def test_the_reported_question_is_answered_in_one_step_without_the_sql_model(fx, models):
    models.code.replies = [program(REPORTED, "one row per metro with its land area")]
    reply = chat(REPORTED)
    assert f"{fx.expected['Pittsburgh, PA Metro Area']['area_sq_mi']:.2f}" in reply
    assert f"{fx.expected['Indianapolis-Carmel-Anderson, IN Metro Area']['area_sq_mi']:.2f}" in reply
    assert len(models.code.prompts) == 1 and models.sql.prompts == []


def test_an_empty_code_agent_completion_still_ends_in_the_right_answer(fx, models):
    """The trace's failure mode: empty completions.  The deterministic fallback program must carry the day."""
    reply = chat(REPORTED)                                   # every model reply is empty
    assert "237.84" in reply and "535.14" in reply
    assert models.sql.prompts == []


def test_a_reworded_request_that_loses_the_measure_is_recovered_in_step_two(fx, models):
    models.code.replies = [program("size of Pittsburgh and Indianapolis"),
                           program("land area of Pittsburgh and Indianapolis")]
    reply = chat(REPORTED)
    assert "237.84" in reply and "535.14" in reply
    assert len(models.code.prompts) == 2
    # step two was shown WHY step one failed, in the machine-readable form
    assert "NOT_ANSWERED[unplannable]" in models.code.prompts[1]


def test_retrying_is_bounded_when_the_second_attempt_fails_too(fx, models):
    models.code.replies = [program("size of Pittsburgh"), program("how large is Indianapolis")]
    reply = chat(REPORTED)
    assert len(models.code.prompts) == 2 and models.sql.prompts == []
    assert "NOT_ANSWERED" not in reply


def test_an_ambiguous_place_is_not_retried_because_only_the_user_can_resolve_it(fx, models):
    models.code.replies = [program("population of Portland")]
    chat("population of Portland")
    assert len(models.code.prompts) == 1 and models.sql.prompts == []


def test_an_invented_table_after_a_non_answer_is_replaced_with_the_real_reason(fx, models, monkeypatch):
    import agents.general_agent as ga
    models.code.replies = [program("population of Portland")]
    invented = ("| Metro | Population |\n|---|---|\n| Portland, ME | 68,408 |\n| Portland, OR | 652,503 |\n"
                "| Salem | 175,535 |\n| Eugene | 176,654 |\n| Bend | 99,178 |\n")
    monkeypatch.setattr(ga, "_get_response_agent", lambda: Scripted([invented]))
    reply = chat("population of Portland")
    assert "68,408" not in reply and "652,503" not in reply
    assert "could not be answered from the loaded data" in reply and "matches more than one place" in reply


def test_a_non_answer_never_leaks_its_machine_marker_to_the_user(fx, models, monkeypatch):
    import agents.general_agent as ga
    models.code.replies = [program("median household income in Pittsburgh")]
    monkeypatch.setattr(ga, "_get_response_agent", lambda: Scripted([]))          # answer model returns nothing at all
    reply = chat("median household income in Pittsburgh")
    assert "NOT_ANSWERED" not in reply and "NOT part of the loaded data" in reply


def test_an_honest_answer_after_a_non_answer_is_left_alone(fx, models, monkeypatch):
    import agents.general_agent as ga
    models.code.replies = [program("population of Portland")]
    honest = "Which Portland do you mean - Portland, ME or Portland, OR?"
    monkeypatch.setattr(ga, "_get_response_agent", lambda: Scripted([honest]))
    assert chat("population of Portland") == honest


def test_house_questions_follow_exactly_the_path_they_always_did(fx, models):
    models.code.replies = [program("How many houses are in Pittsburgh?")]
    models.sql.replies = ["SELECT COUNT(houses.house_id) FROM houses WHERE houses.city = 'Pittsburgh'"]
    chat("How many houses are in Pittsburgh?")
    assert len(models.code.prompts) == 1 and len(models.sql.prompts) == 1
