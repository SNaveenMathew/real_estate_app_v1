"""Structural House Chat scope (agents/query_scope.py) and the explicit "top N" limit, end to end."""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402


class StubSqlModel:
    def __init__(self, replies=()):
        self.replies, self.calls = list(replies), 0

    def invoke(self, messages):
        self.calls += 1
        return types.SimpleNamespace(content=self.replies.pop(0) if self.replies else "")


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("scope"))
    yield f
    census_fixture.deactivate(f)


@pytest.fixture()
def model(fx, monkeypatch):
    import agents.tools as tools
    stub = StubSqlModel()
    monkeypatch.setattr(tools, "_agent", stub)
    return stub


def plan(q, **kw):
    from agents.query_planner import build_query_plan
    return build_query_plan(q, **kw)


def test_scope_filters_are_validated_and_escaped():
    from agents import query_scope as qs
    assert qs.ScopeFilter("Houses", "House_ID", "h1").render() == "houses.house_id = 'h1'"
    assert qs.ScopeFilter("houses", "house_id", "o'brien").render() == "houses.house_id = 'o''brien'"
    for bad in (("houses; DROP", "house_id", "x"), ("houses", "house id", "x"), ("houses", "house_id", ""), ("houses", "house_id", None)):
        with pytest.raises(ValueError):
            qs.ScopeFilter(*bad)


def test_scope_is_isolated_per_context_and_always_restored():
    from agents import query_scope as qs
    assert qs.current() == ()
    with qs.scoped(qs.ScopeFilter("houses", "house_id", "a")):
        assert [f.value for f in qs.current()] == ["a"]
        with qs.scoped():
            assert qs.current() == ()                        # an explicit waiver inside is honoured ...
        assert [f.value for f in qs.current()] == ["a"]      # ... and undone
    assert qs.current() == ()
    with pytest.raises(RuntimeError):
        with qs.scoped(qs.ScopeFilter("houses", "house_id", "a")):
            raise RuntimeError
    assert qs.current() == ()                                 # even when the block fails


def test_scope_follows_work_onto_a_worker_thread_but_never_leaks_between_turns():
    from agents import query_scope as qs
    seen = {}

    async def turn(name):
        with qs.scoped(qs.ScopeFilter("houses", "house_id", name)):
            await asyncio.sleep(0.01)
            seen[name] = await asyncio.to_thread(lambda: [f.value for f in qs.current()])   # context is copied to the thread

    async def main():
        await asyncio.gather(*(turn(n) for n in ("h1", "h2", "h3", "h4")))
        return await asyncio.to_thread(lambda: qs.current())

    assert asyncio.run(main()) == ()
    assert seen == {"h1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"]}


def test_a_plan_outside_any_scope_is_exactly_what_it_always_was(fx):
    p = plan("How many houses are there?")
    assert p.scope_filters == [] and "scope_filters" not in p.render()


def test_scope_forces_the_scope_table_and_its_join_path_into_the_plan(fx):
    from agents import query_scope as qs
    with qs.scoped(qs.ScopeFilter("houses", "house_id", "h1")):
        p = plan("average NRI risk score")
    assert p.scope_filters == ["houses.house_id = 'h1'"] and "houses" in p.required_tables and "nri_tracts" in p.required_tables
    assert "MANDATORY" in p.render() and "houses.house_id = 'h1'" in p.render()


def test_the_compiler_adds_the_scope_and_the_contract_accepts_it(fx):
    from agents import plan_contract, query_scope as qs
    from agents.tools import _compile_sql_from_plan
    with qs.scoped(qs.ScopeFilter("houses", "house_id", "h1")):
        p = plan("How many houses are there?")
    sql = _compile_sql_from_plan(p)
    assert "houses.house_id = 'h1'" in sql and plan_contract.verify(sql, p) == []


@pytest.mark.parametrize("variant,expected_ok", [
    ("SELECT COUNT(*) FROM houses WHERE houses.house_id = 'h1'", True),
    ("SELECT COUNT(*) FROM houses h WHERE h.house_id = 'h1' AND h.city = 'Pittsburgh'", True),
    ("SELECT COUNT(*) FROM houses WHERE LOWER(house_id) = 'h1'", True),
    ("SELECT COUNT(*) FROM houses", False),
    ("SELECT COUNT(*) FROM houses WHERE houses.house_id <> 'h1'", False),
    ("SELECT COUNT(*) FROM houses WHERE houses.house_id = 'h1' OR 1 = 1", False),
    ("SELECT COUNT(*) FROM houses WHERE houses.house_id = 'h2'", False),                  # somebody else's house
    ("SELECT COUNT(*) FROM houses WHERE houses.city = 'h1'", False),                      # right literal, wrong column
])
def test_the_contract_requires_the_scope_as_a_positive_filter(fx, variant, expected_ok):
    from agents import plan_contract, query_scope as qs
    with qs.scoped(qs.ScopeFilter("houses", "house_id", "h1")):
        p = plan("How many houses are there?")
    got = {v.code for v in plan_contract.verify(variant, p)}
    assert (("missing_scope_filter" not in got) is expected_ok), (variant, got)


UNSCOPED = "SELECT COUNT(houses.house_id) FROM houses"


def house_query(**kw):
    from agents.house_agent import make_house_approved_functions
    return make_house_approved_functions("h1")["query_database"](**kw)


def test_a_model_that_drops_the_scope_still_gets_only_this_house(fx, model):
    model.replies = [UNSCOPED]
    out = house_query(request="How many houses are there?")
    assert "houses.house_id = 'h1'" in out                  # the statement that ran is the scoped, compiled one
    assert "SELECT COUNT(houses.house_id) FROM houses\n" not in out
    result = out.split("[RESULT]\n", 1)[1]
    assert result.split()[-1] == "1"                        # this house only; the unscoped statement would have counted all of them


def test_all_houses_is_the_explicit_way_to_ask_a_cross_house_question(fx, model):
    model.replies = [UNSCOPED]
    out = house_query(request="How many houses are there?", scope="all_houses")
    assert "house_id = 'h1'" not in out and out.split("[RESULT]\n", 1)[1].split()[-1] == "6"


def test_an_unknown_scope_value_is_refused_with_guidance(fx, model):
    out = house_query(request="How many houses are there?", scope="everything")
    assert out.startswith("Code Agent error: scope must be") and model.calls == 0


def test_the_scope_never_outlives_the_call(fx, model):
    from agents import query_scope as qs
    model.replies = [UNSCOPED]
    house_query(request="How many houses are there?")
    assert qs.current() == ()


def test_house_chat_requests_are_no_longer_polluted_with_prose(fx, model):
    """The old wrapper appended 'This is House Chat for house_id=...' to the request: the planner saw it."""
    seen = {}
    import agents.tools as tools
    real = tools.run_code_query

    def spy(request, *a, **k):
        seen["request"] = request
        return real(request, *a, **k)

    tools.run_code_query = spy
    try:
        house_query(request="How many houses are there?")
    finally:
        tools.run_code_query = real
    assert seen["request"] == "How many houses are there?"


def test_the_house_prompt_documents_the_scope_parameter(fx):
    import agents.house_agent as ha
    import inspect
    src = inspect.getsource(ha)
    assert 'scope: str = "this_house"' in src and 'scope="all_houses"' in src


@pytest.mark.parametrize("text,n", [
    ("top 3 metros by population", 3), ("show the top 25 houses", 25), ("five cheapest houses", 5), ("3 most expensive homes", 3),
    ("top ten cities", 10), ("bottom 4 by price", 4), ("top 5 percent of tracts", None), ("top 5%", None), ("top 0", None),
    ("first 2000 houses", None), ("what is the top selling point", None), ("average price", None),
])
def test_top_n_is_parsed_with_sane_bounds(text, n):
    from agents.query_planner import _top_n
    assert _top_n(text) == n


def test_the_requested_number_reaches_the_plan_and_the_compiler(fx):
    from agents import plan_contract
    from agents.tools import _compile_sql_from_plan
    p = plan("Rank the top 3 MSAs by population")
    assert p.result_limit == 3 and "result_limit: 3" in p.render()
    sql = _compile_sql_from_plan(p)
    assert sql.rstrip().endswith("LIMIT 3") and plan_contract.verify(sql, p) == []
    assert plan("Rank MSAs by population").result_limit is None            # no N asked: nothing is invented


@pytest.mark.parametrize("sql,expected", [
    ("SELECT name FROM census_msa ORDER BY population DESC LIMIT 3", set()),
    ("SELECT name FROM census_msa ORDER BY population DESC LIMIT 2", set()),
    ("SELECT name FROM census_msa ORDER BY population DESC", {"missing_limit"}),
    ("SELECT name FROM census_msa ORDER BY population DESC LIMIT 10", {"limit_too_large"}),
    ("SELECT * FROM (SELECT name FROM census_msa ORDER BY population DESC LIMIT 3) t", {"missing_limit"}),   # inner LIMIT does not bound the result
])
def test_the_contract_enforces_the_limit(fx, sql, expected):
    from agents import plan_contract
    got = {v.code for v in plan_contract.verify(sql, plan("Rank the top 3 MSAs by population"))}
    assert got & {"missing_limit", "limit_too_large"} == expected


def test_a_model_that_forgets_the_limit_is_repaired_not_obeyed(fx, model):
    import agents.tools as tools
    model.replies = ["SELECT census_msa.name, census_msa.population FROM census_msa ORDER BY census_msa.population DESC"]
    out = tools.query_database.invoke({"request": "Rank the top 3 MSAs by population"})
    body = out.split("[RESULT]\n", 1)[1]
    assert "LIMIT 3" in out and len([ln for ln in body.splitlines() if "Metro Area" in ln or "Micro Area" in ln]) == 3
