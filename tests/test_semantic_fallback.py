"""agents/semantic_fallback.py: a model may PROPOSE concept keys when no alias matches - and nothing else."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402
from agents import llm_resilience as lr  # noqa: E402
from agents import semantic_fallback as sf  # noqa: E402
from config import settings  # noqa: E402
from plan_battery_queries import GROUPS  # noqa: E402


class Proposer:
    """Stub for the proposer's own model client; records every prompt it was shown."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def invoke(self, messages, **kw):
        self.prompts.append("\n".join(str(m.content) for m in messages))
        out = self.replies.pop(0) if self.replies else '{"concepts": []}'
        if isinstance(out, BaseException):
            raise out
        return types.SimpleNamespace(content=out)

    @property
    def calls(self):
        return len(self.prompts)


class SqlModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages, **kw):
        self.calls += 1
        return types.SimpleNamespace(content="")


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("semantic"))
    yield f
    census_fixture.deactivate(f)


@pytest.fixture()
def proposer(fx, monkeypatch):
    lr.reset()
    sf.reset_cache()
    stub = Proposer()
    monkeypatch.setattr(sf, "_client", stub)
    monkeypatch.setattr(settings, "semantic_fallback_enabled", True)      # the fixture turns it off; these tests opt in, with a stub
    import agents.tools as tools
    monkeypatch.setattr(tools, "_agent", SqlModel())
    return stub


def ask(request, **kw):
    import agents.tools as tools
    return tools.query_database.invoke({"request": request, **kw})


def result_of(out):
    return out.split("[RESULT]\n", 1)[1]


ALLOWED = {"census_land_area", "census_place_population", "house_list_price", "msa_population"}


@pytest.mark.parametrize("reply,expected", [
    ('{"concepts": ["census_land_area"]}', ["census_land_area"]),
    ('Sure! ```json\n{"concepts": ["census_land_area", "msa_population"]}\n```', ["census_land_area", "msa_population"]),
    ('["census_land_area"]', ["census_land_area"]),
    ('{"concepts": ["census_land_area", "census_land_area"]}', ["census_land_area"]),                    # de-duplicated
    ('{"concepts": ["not_a_concept", "census_land_area"]}', ["census_land_area"]),                      # invented keys dropped
    ('{"concepts": ["DROP TABLE houses", "houses"]}', []),                                              # SQL / tables are not keys
    ('{"concepts": ["census_land_area", "msa_population", "house_list_price", "census_place_population"]}',
     ["census_land_area", "msa_population", "house_list_price"]),                                       # capped at three
    ('{"concepts": []}', []), ('{"concepts": "census_land_area"}', []), ("no idea", []), ("", []), ("{broken json", []),
    ('{"concepts": [1, null, {"k": "census_land_area"}]}', []),
])
def test_only_valid_keys_survive_whatever_the_model_says(reply, expected):
    assert sf.parse_keys(reply, ALLOWED) == expected


def test_the_model_is_shown_a_closed_list_and_asked_for_keys_only(proposer):
    proposer.replies = ['{"concepts": ["census_land_area"]}']
    assert sf.propose_concepts("How big is Pittsburgh?") == ["census_land_area"]
    prompt = proposer.prompts[0]
    assert "- census_land_area:" in prompt and "- house_list_price:" in prompt and "QUESTION: How big is Pittsburgh?" in prompt
    assert "Never write SQL" in prompt and "(NOT LOADED)" in prompt                      # gap concepts are marked
    assert len(prompt) < 20_000                                                          # compact: one line per concept


def test_results_are_cached_per_catalog_and_request(proposer):
    proposer.replies = ['{"concepts": ["census_land_area"]}']
    assert sf.propose_concepts("How big is Pittsburgh?") == sf.propose_concepts("how big is pittsburgh") == ["census_land_area"]
    assert proposer.calls == 1


def test_an_unavailable_model_yields_no_proposal_and_is_not_cached(proposer):
    proposer.replies = [ConnectionError("refused"), '{"concepts": ["census_land_area"]}']
    assert sf.propose_concepts("How big is Pittsburgh?") == []
    assert sf.propose_concepts("How big is Pittsburgh?") == ["census_land_area"]         # the failure was not remembered


def test_it_can_be_switched_off(proposer, monkeypatch):
    monkeypatch.setattr(settings, "semantic_fallback_enabled", False)
    assert sf.propose_concepts("How big is Pittsburgh?") == [] and proposer.calls == 0


def test_a_cancelled_turn_is_not_swallowed(proposer):
    with lr.turn() as t:
        t.cancel.set()
        with pytest.raises(lr.TurnCancelled):
            sf.propose_concepts("How big is Pittsburgh?")
    assert proposer.calls == 0


def test_without_a_proposal_such_a_request_is_simply_unplannable(proposer):
    body = result_of(ask("How big is Pittsburgh?"))
    assert body.startswith("NOT_ANSWERED[unplannable]") and proposer.calls == 1


def test_a_valid_proposal_is_planned_routed_and_answered_deterministically(proposer, fx):
    proposer.replies = ['{"concepts": ["census_land_area"]}']
    body = result_of(ask("How big is Pittsburgh?"))
    assert body.startswith("(Interpreted as: Land area of a named metropolitan/micropolitan statistical area")
    assert f"{fx.expected['Pittsburgh, PA Metro Area']['area_sq_mi']:.2f}" in body      # the real derived area, no model involved
    assert proposer.calls == 1


def test_a_proposal_for_a_plain_lookup_runs_through_the_normal_sql_path(proposer):
    from agents.query_planner import build_query_plan
    question = "what is the headcount of Pittsburgh?"
    assert build_query_plan(question).semantic_keys == []          # precondition: no alias matches, so the proposer is consulted
    proposer.replies = ['{"concepts": ["census_place_population"]}']
    out = ask(question)
    assert "SELECT census_msa.name, census_msa.population FROM census_msa WHERE census_msa.name = 'Pittsburgh, PA Metro Area'" in out
    assert "(Interpreted as:" in out and "24000" in result_of(out)


def test_a_proposed_gap_concept_gives_the_honest_not_loaded_answer(proposer):
    proposer.replies = ['{"concepts": ["census_unloaded_topics"]}']
    body = result_of(ask("how wealthy are people in Pittsburgh?"))
    assert body.startswith("NOT_ANSWERED[topic_unavailable]") and "Interpreted as" not in body


def test_a_model_cannot_smuggle_sql_or_tables_through_the_proposal(proposer):
    proposer.replies = ['{"concepts": ["census_land_area"], "sql": "DROP TABLE houses", "tables": ["catalog_concepts"]}']
    body = result_of(ask("How big is Pittsburgh?"))
    import db.duckdb_store as store
    assert store.query("SELECT COUNT(*) AS n FROM houses").iloc[0, 0] == 6 and "catalog_concepts" not in body
    assert "(Interpreted as:" in body


def test_an_invented_concept_leaves_the_request_unplannable(proposer):
    proposer.replies = ['{"concepts": ["definitely_not_a_concept"]}']
    assert result_of(ask("How big is Pittsburgh?")).startswith("NOT_ANSWERED[unplannable]")


def test_requests_that_already_plan_never_reach_the_proposer(proposer):
    from agents.query_planner import build_query_plan
    import agents.tools as tools
    tools._agent = SqlModel()
    checked = 0
    for qs in GROUPS.values():
        for q in qs:
            p = build_query_plan(q)
            if p.required_tables or p.semantic_keys:
                ask(q)
                checked += 1
    assert checked >= 40 and proposer.calls == 0


def test_the_proposal_never_outlives_the_call(proposer):
    import db.schema_catalog as schema
    proposer.replies = ['{"concepts": ["census_land_area"]}']
    ask("How big is Pittsburgh?")
    assert schema._FORCED_CONCEPTS.get() == ()
    from agents.query_planner import build_query_plan
    assert build_query_plan("How big is Pittsburgh?").semantic_keys == []              # a later plan is back to alias matching only


def test_the_orchestrator_answers_in_one_step_instead_of_two(proposer, fx, monkeypatch):
    """Before: unplannable -> retry step.  Now the proposal rescues the request inside the tool."""
    import agents.general_agent as ga
    import agents.tools as tools
    proposer.replies = ['{"concepts": ["census_land_area"]}']
    code = types.SimpleNamespace(prompts=0)

    class Code:
        def invoke(self, messages, **kw):
            code.prompts += 1
            return types.SimpleNamespace(content="final_result = query_database(request='How big is Pittsburgh?')")

    class Echo:
        def invoke(self, messages, **kw):
            return types.SimpleNamespace(content=str(messages[0].content).split("EXECUTED EVIDENCE:", 1)[-1])

    monkeypatch.setattr(ga, "_get_code_agent", lambda: Code())
    monkeypatch.setattr(ga, "_get_response_agent", lambda: Echo())
    monkeypatch.setattr(tools.vs, "search_data_model", lambda *a, **k: [])
    reply = ga.run_general_chat("How big is Pittsburgh?")[0]
    assert f"{fx.expected['Pittsburgh, PA Metro Area']['area_sq_mi']:.2f}" in reply and code.prompts == 1


def test_the_fallback_does_not_second_guess_a_request_that_already_matched_an_alias(proposer):
    """By design it rescues EMPTY plans only: verifying working matches would put a model call on flows that already work."""
    from agents.query_planner import build_query_plan
    question = "how many folks call Pittsburgh home?"                # the alias 'home' matches the house-inventory concept
    assert build_query_plan(question).semantic_keys
    proposer.replies = ['{"concepts": ["census_place_population"]}']
    ask(question)
    assert proposer.calls == 0
