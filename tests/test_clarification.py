"""agents/clarification.py: ask when a place is known and the measure is not - and never otherwise."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402
from agents import answer_policy, answer_status as status, clarification  # noqa: E402
from config import settings  # noqa: E402


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("clarify"))
    yield f
    census_fixture.deactivate(f)


def unplannable(request: str) -> str:
    return answer_policy.unplannable_result(request)


def test_a_place_without_a_measure_is_asked_about(fx):
    asked = clarification.question_for(
        "And how many do I have in Denver?", unplannable("And how many do I have in Denver?")
    )
    assert asked is not None
    assert asked.code == status.UNPLANNABLE
    assert asked.places == ("Denver",)  # one plain name, not 'Denver', 'denver' and the metro variant
    assert "Denver" in asked.text and asked.text.endswith("?")
    assert asked.candidates  # offers measures the catalog actually covers
    assert not any(ch.isdigit() for ch in asked.text)  # the digit-based validator must never see data


def test_every_offered_candidate_is_answerable(fx):
    asked = clarification.question_for("how many do i have in Denver", unplannable("how many do i have in Denver"))
    from agents.query_planner import build_query_plan

    for candidate in asked.candidates:
        plan = build_query_plan(f"how many houses in Denver, using {candidate}")
        assert plan.required_tables or plan.semantic_keys  # an offer a user can pick must be plannable


def test_a_turn_that_answered_is_never_interrogated(fx):
    assert clarification.question_for("How many crime incidents are in Pittsburgh?", "(2 rows)") is None


def test_without_a_recognized_place_there_is_nothing_to_anchor_a_question(fx):
    typo = "how many hosues do i have in pittsbrugh?"
    assert clarification.question_for(typo, unplannable(typo)) is None


def test_an_ambiguous_place_is_left_to_its_own_message(fx):
    """place_ambiguous already asks its question through the answer model; templating it here would only be worse."""
    ambiguous = status.not_answered(
        status.PLACE_AMBIGUOUS, '"springfield" matches more than one place: Springfield, IL; Springfield, MA.'
    )
    assert clarification.question_for("how many houses in Denver", ambiguous) is None


def test_the_orchestrator_asks_instead_of_answering_and_says_so(fx, monkeypatch):
    """End to end through run_general_chat: the question is the reply, and the metadata says why."""
    import agents.general_agent as ga
    import agents.tools as tools

    class Code:
        def invoke(self, messages, **kw):
            return types.SimpleNamespace(
                content="final_result = query_database(request='And how many do I have in Denver?')"
            )

    monkeypatch.setattr(ga, "_get_code_agent", lambda: Code())
    monkeypatch.setattr(tools.vs, "search_data_model", lambda *a, **k: [])
    monkeypatch.setattr(settings, "clarification_enabled", True)
    reply, history, _, metadata = ga.run_general_chat("And how many do I have in Denver?", include_metadata=True)
    assert reply.endswith("?") and "Denver" in reply
    assert metadata["answer_status"] == status.UNPLANNABLE
    assert metadata["clarification"]["places"] == ["Denver"]
    assert history[-1]["content"] == reply  # the question is this turn's answer, so the next turn has it


def test_it_can_be_switched_off(fx, monkeypatch):
    import agents.general_agent as ga
    import agents.tools as tools

    class Code:
        def invoke(self, messages, **kw):
            return types.SimpleNamespace(
                content="final_result = query_database(request='And how many do I have in Denver?')"
            )

    class Echo:
        def invoke(self, messages, **kw):
            return types.SimpleNamespace(content=str(messages[0].content).split("EXECUTED EVIDENCE:", 1)[-1])

    monkeypatch.setattr(ga, "_get_code_agent", lambda: Code())
    monkeypatch.setattr(ga, "_get_response_agent", lambda: Echo())
    monkeypatch.setattr(tools.vs, "search_data_model", lambda *a, **k: [])
    monkeypatch.setattr(settings, "clarification_enabled", False)
    reply, _, _, metadata = ga.run_general_chat("And how many do I have in Denver?", include_metadata=True)
    assert metadata["clarification"] is None
    assert status.status_code(reply) is None or "unplannable" in (status.status_code(reply) or "")
