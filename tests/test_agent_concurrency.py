"""Many whole agent turns at once, through the real turn runner: each must get its own answer, none may see another's.

Models are stateless stubs, so any cross-talk can only come from shared state in the application code (the scope, the
artifact collector, caches, the database connection, the schema registry).
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402
from agents import llm_resilience as lr  # noqa: E402
from agents import turn_runner as tr  # noqa: E402
from config import settings  # noqa: E402

PIT = "Pittsburgh, PA Metro Area"


class Empty:
    def invoke(self, messages, **kw):
        return types.SimpleNamespace(content="")


class UnscopedSql:
    def invoke(self, messages, **kw):
        return types.SimpleNamespace(content="SELECT COUNT(houses.house_id) FROM houses")


class Echo:
    def invoke(self, messages, **kw):
        text = str(messages[0].content)
        return types.SimpleNamespace(content=text.split("EXECUTED EVIDENCE:", 1)[-1].strip() or text[-1500:])


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("agentconc"))
    yield f
    census_fixture.deactivate(f)


@pytest.fixture()
def wired(fx, monkeypatch):
    import agents.general_agent as ga
    import agents.house_agent as ha
    import agents.tools as tools
    lr.reset()
    monkeypatch.setattr(settings, "agent_max_concurrent_turns", 4)
    monkeypatch.setattr(settings, "agent_queue_timeout_s", 60.0)
    monkeypatch.setattr(ga, "_get_code_agent", lambda: Empty())
    monkeypatch.setattr(ga, "_get_response_agent", lambda: Echo())
    monkeypatch.setattr(ha, "_get_house_code_agent", lambda: Empty())
    monkeypatch.setattr(ha, "_get_house_response_agent", lambda: Echo())
    monkeypatch.setattr(tools, "_agent", UnscopedSql())
    monkeypatch.setattr(tools.vs, "search_data_model", lambda *a, **k: [])
    tr._SEMAPHORES.clear()
    return fx


def test_concurrent_general_turns_each_get_their_own_answer(wired):
    from agents.general_agent import run_general_chat
    land = f"{wired.expected[PIT]['area_sq_mi']:.2f}"
    questions = {"What is the population of Pittsburgh?": "24000", "Verify the land areas of Pittsburgh and Indianapolis": land,
                 "How many houses are there?": None}

    async def main():
        jobs = [(q, tr.run_turn(run_general_chat, q)) for q in list(questions) * 8]
        return list(zip([q for q, _ in jobs], await asyncio.gather(*(j for _, j in jobs))))

    results = asyncio.run(main())
    assert len(results) == 24
    for question, (reply, _history) in results:
        expected = questions[question]
        if expected:
            assert expected in reply, (question, reply[:200])
        assert "NOT_ANSWERED" not in reply and "Traceback" not in reply


def test_concurrent_house_turns_never_leak_one_houses_scope_into_anothers(wired):
    from agents.house_agent import run_house_chat
    import agents.house_agent as ha
    program = "final_result = query_database(request='How many houses are there?')"
    ha._get_house_code_agent = lambda: types.SimpleNamespace(invoke=lambda m, **k: types.SimpleNamespace(content=program))

    async def main():
        jobs = [(hid, tr.run_turn(run_house_chat, hid, "How many houses are there?")) for hid in ("h1", "h2", "h3") * 6]
        return list(zip([h for h, _ in jobs], await asyncio.gather(*(j for _, j in jobs))))

    results = asyncio.run(main())
    assert len(results) == 18
    for hid, out in results:
        reply = out[0]
        assert f"houses.house_id = '{hid}'" in reply, (hid, reply[:300])
        for other in {"h1", "h2", "h3"} - {hid}:
            assert f"house_id = '{other}'" not in reply, (hid, other)           # another turn's scope never leaked in


def test_the_scope_is_empty_again_on_every_thread_afterwards(wired):
    from agents import query_scope as qs

    async def probe():
        return await tr.run_turn(lambda: qs.current())

    assert asyncio.run(probe()) == () and qs.current() == ()
