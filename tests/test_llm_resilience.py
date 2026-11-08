"""agents/llm_resilience.py and the graceful degradation it enables in the tool layer and both chat agents."""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import census_fixture  # noqa: E402
from agents import llm_resilience as lr  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Client:
    """A model stub: scripted outcomes ('ok', or an exception instance), recording how often it was really called."""

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def invoke(self, messages, **kw):
        self.calls += 1
        out = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(out, BaseException):
            raise out
        return types.SimpleNamespace(content=out)


class HttpError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


@pytest.fixture(autouse=True)
def fresh_breaker():
    lr.reset()
    yield
    lr.reset()


class APIConnectionError(Exception): pass          # noqa: E701 - named like the openai / httpx classes on purpose
class ReadTimeout(Exception): pass                 # noqa: E701


@pytest.mark.parametrize("exc,transient", [
    (ConnectionRefusedError("refused"), True), (TimeoutError("t"), True), (APIConnectionError("Connection error."), True),
    (ReadTimeout("The read operation timed out"), True), (HttpError(500), True), (HttpError(503), True), (HttpError(429), True),
    (HttpError(408), True), (HttpError(400), False), (HttpError(401), False), (HttpError(404), False),
    (ValueError("context length exceeded"), False), (KeyError("x"), False), (RuntimeError("something else"), False),
])
def test_only_the_services_failures_are_transient(exc, transient):
    assert lr.is_transient(exc) is transient


def test_the_breaker_opens_after_the_threshold_then_fails_instantly_without_calling_the_model():
    clock = Clock()
    lr.configure_breaker(3, 30, clock)
    client = Client([ConnectionError("x")] * 10)
    for _ in range(3):
        with pytest.raises(lr.ModelUnavailable):
            lr.invoke(client, [])
    assert client.calls == 3 and lr.breaker().state == "open"
    with pytest.raises(lr.ModelUnavailable) as err:
        lr.invoke(client, [])
    assert client.calls == 3 and "not responding" in str(err.value)             # failed fast: the client was not touched


def test_after_the_cooldown_exactly_one_probe_is_allowed_and_its_outcome_decides():
    clock = Clock()
    lr.configure_breaker(2, 30, clock)
    bad = Client([ConnectionError("x")] * 5)
    for _ in range(2):
        with pytest.raises(lr.ModelUnavailable):
            lr.invoke(bad, [])
    clock.now += 29
    with pytest.raises(lr.ModelUnavailable):
        lr.invoke(bad, [])
    assert bad.calls == 2                                                       # still cooling down
    clock.now += 2
    with pytest.raises(lr.ModelUnavailable):
        lr.invoke(bad, [])                                                      # the probe: fails -> re-opens
    assert bad.calls == 3 and lr.breaker().state == "open"
    clock.now += 31
    good = Client(["fine"])
    assert lr.invoke(good, []).content == "fine" and lr.breaker().state == "closed" and lr.breaker().failures == 0


def test_only_one_probe_runs_at_a_time():
    clock = Clock()
    lr.configure_breaker(1, 10, clock)
    with pytest.raises(lr.ModelUnavailable):
        lr.invoke(Client([ConnectionError("x")]), [])
    clock.now += 11
    started, release, results = threading.Event(), threading.Event(), []

    class Slow:
        calls = 0

        def invoke(self, messages):
            Slow.calls += 1
            started.set()
            release.wait(5)
            return types.SimpleNamespace(content="ok")

    def probe():
        results.append(lr.invoke(Slow(), []).content)

    t = threading.Thread(target=probe)
    t.start()
    assert started.wait(5)
    with pytest.raises(lr.ModelUnavailable):                                    # a second caller while the probe is in flight
        lr.invoke(Slow(), [])
    release.set()
    t.join()
    assert results == ["ok"] and Slow.calls == 1 and lr.breaker().state == "closed"


def test_a_success_resets_the_failure_count():
    lr.configure_breaker(3, 30, Clock())
    client = Client([ConnectionError("x"), ConnectionError("x"), "ok", ConnectionError("x"), ConnectionError("x")])
    for expect_fail in (True, True, False, True, True):
        if expect_fail:
            with pytest.raises(lr.ModelUnavailable):
                lr.invoke(client, [])
        else:
            lr.invoke(client, [])
    assert lr.breaker().state == "closed" and lr.breaker().failures == 2


def test_a_bad_request_is_the_callers_problem_not_an_outage():
    lr.configure_breaker(1, 30, Clock())
    client = Client([HttpError(400)] * 3)
    for _ in range(3):
        with pytest.raises(HttpError):                                          # re-raised UNCHANGED
            lr.invoke(client, [])
    assert lr.breaker().state == "closed" and client.calls == 3                 # the service answered: it is up


def test_status_reports_the_breaker_for_health_checks():
    clock = Clock()
    lr.configure_breaker(1, 30, clock)
    assert lr.breaker().status()["state"] == "closed"
    with pytest.raises(lr.ModelUnavailable):
        lr.invoke(Client([ConnectionError("x")]), [])
    clock.now += 10
    assert lr.breaker().status() == {"state": "open", "consecutive_failures": 1, "retry_in_s": 20.0}


def test_a_failing_optional_role_cannot_disable_the_primary_models():
    lr.configure_breaker(3, 30, Clock(), name="semantic")
    optional = Client([ConnectionError("refused")] * 10)
    for _ in range(3):
        with pytest.raises(lr.ModelUnavailable):
            lr.invoke(optional, [], role="semantic_fallback", breaker_name="semantic")
    assert lr.breaker("semantic").state == "open"
    primary = Client(["written answer"])
    assert lr.invoke(primary, [], role="code_agent").content == "written answer"        # the chat models are unaffected
    assert lr.breaker("chat").state == "closed" and primary.calls == 1


def test_statuses_lists_every_breaker_that_exists():
    assert set(lr.statuses()) == {"chat"}
    lr.invoke(Client(["ok"]), [], role="x", breaker_name="semantic")
    assert "semantic" in lr.statuses()


def test_no_turn_means_no_cancellation():
    lr.check_cancelled()


def test_a_cancelled_turn_starts_no_further_model_calls():
    client = Client(["a", "b"])
    with lr.turn(deadline_s=None) as t:
        assert lr.invoke(client, []).content == "a"
        t.cancel.set()
        with pytest.raises(lr.TurnCancelled):
            lr.invoke(client, [])
    assert client.calls == 1


def test_a_deadline_stops_the_turn_at_its_next_step(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(lr.time, "monotonic", lambda: now[0])
    client = Client(["a", "b"])
    with lr.turn(deadline_s=10):
        assert lr.invoke(client, []).content == "a"
        now[0] += 11
        with pytest.raises(lr.TurnCancelled):
            lr.invoke(client, [])
    assert client.calls == 1


def test_a_turn_created_elsewhere_can_be_adopted_by_a_worker_thread():
    handle, seen = lr.Turn(), []

    def worker():
        with lr.use_turn(handle):
            try:
                lr.check_cancelled()
                seen.append("ran")
                handle.cancel.set()
                lr.check_cancelled()
            except lr.TurnCancelled:
                seen.append("cancelled")

    t = threading.Thread(target=worker)
    t.start(); t.join()
    assert seen == ["ran", "cancelled"]


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    f = census_fixture.activate(tmp_path_factory.mktemp("resilience"))
    yield f
    census_fixture.deactivate(f)


@pytest.fixture()
def models_down(fx, monkeypatch):
    import agents.general_agent as ga
    import agents.house_agent as ha
    import agents.tools as tools
    down = Client([ConnectionError("llama-server connection refused")] * 200)
    monkeypatch.setattr(ga, "_get_code_agent", lambda: down)
    monkeypatch.setattr(ga, "_get_response_agent", lambda: down)
    monkeypatch.setattr(ha, "_get_house_code_agent", lambda: down)
    monkeypatch.setattr(ha, "_get_house_response_agent", lambda: down)
    monkeypatch.setattr(tools, "_agent", down)
    monkeypatch.setattr(tools.vs, "search_data_model", lambda *a, **k: [])
    return down


def chat(message):
    from agents.general_agent import run_general_chat
    return run_general_chat(message)[0]


def test_with_every_model_down_a_plannable_question_is_still_answered(models_down):
    reply = chat("What is the population of Pittsburgh?")
    assert reply.startswith("(The language model service is not responding") and "Pittsburgh, PA Metro Area" in reply and "24000" in reply


def test_with_every_model_down_a_derived_measure_needs_no_model_at_all(models_down, fx):
    reply = chat("Verify the land areas of Pittsburgh and Indianapolis")
    assert f"{fx.expected['Pittsburgh, PA Metro Area']['area_sq_mi']:.2f}" in reply and "not a written answer" in reply


def test_with_every_model_down_an_unanswerable_question_gets_the_honest_explanation(models_down):
    reply = chat("tell me a joke about spreadsheets")
    assert "could not be matched" in reply and "NOT_ANSWERED" not in reply


def test_the_breaker_stops_the_hammering_after_the_threshold(models_down):
    first = chat("What is the population of Pittsburgh?")
    calls_after_first = models_down.calls
    assert calls_after_first <= 3 + 1 and lr.breaker().state == "open"
    for _ in range(5):
        chat("What is the population of Pittsburgh?")
    assert models_down.calls == calls_after_first                               # open breaker: zero further model calls
    assert first.startswith("(The language model service")


def test_a_question_that_needs_the_model_says_so_and_is_not_retried(models_down):
    import agents.tools as tools
    from agents import answer_status
    out = tools.query_database.invoke({"request": "list tract population in the Pittsburgh metro"})
    body = out.split("[RESULT]\n", 1)[1]
    assert answer_status.status_code(body) == "model_unavailable" and not answer_status.is_retryable(body)


def test_a_cancelled_turn_is_not_converted_into_an_error_message(models_down, fx, monkeypatch):
    import agents.tools as tools
    cancelled = Client(["SELECT COUNT(houses.house_id) FROM houses"])
    monkeypatch.setattr(tools, "_agent", cancelled)
    with lr.turn() as t:
        t.cancel.set()
        with pytest.raises(lr.TurnCancelled):
            tools.query_database.invoke({"request": "How many houses are there?"})
    assert cancelled.calls == 0


def test_house_chat_degrades_too(models_down):
    from agents.house_agent import run_house_chat
    reply = run_house_chat("h1", "what is the list price?")[0]
    assert reply.startswith("(The language model service is not responding") or "not responding" in reply


def test_chat_clients_are_built_with_bounded_retries():
    from config import settings
    import agents.tools as tools
    tools._agent = None
    try:
        assert tools.get_code_agent().max_retries == settings.llm_max_retries
    finally:
        tools._agent = None


def test_a_dead_semantic_proposer_never_stops_the_chat_model_from_being_called(fx, monkeypatch):
    """Found in the field: with nothing listening on the model port, the optional proposer's connection failures opened the
    single process-wide breaker, and every LATER chat-model call - even a stub - was refused before it was made."""
    import agents.tools as tools
    from agents import semantic_fallback as sf
    from config import settings

    monkeypatch.setattr(settings, "semantic_fallback_enabled", True)
    monkeypatch.setattr(sf, "_client", Client([ConnectionError("refused")] * 50))          # nothing is listening
    sf.reset_cache()
    stub = Client(["SELECT COUNT(houses.house_id) FROM houses WHERE houses.city = 'Pittsburgh'"])
    monkeypatch.setattr(tools, "_agent", stub)
    for i in range(6):                                                                       # unplannable requests hit the proposer
        out = tools.query_database.invoke({"request": f"a question nothing in the catalog can match number {i}"})
        assert "NOT_ANSWERED[unplannable]" in out
    out = tools.query_database.invoke({"request": "How many houses are in Pittsburgh?"})
    # the symptom: with a shared breaker the stub is never called (the compiled SQL answers instead, silently)
    assert stub.calls == 1, f"the chat model was refused before being called; evidence:\n{out}"
    assert "NOT_ANSWERED" not in out and lr.breaker("chat").state == "closed"
