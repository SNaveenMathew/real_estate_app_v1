"""agents/turn_runner.py and the chat endpoints: a turn must not freeze the server, and must be bounded and cancellable."""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import llm_resilience as lr  # noqa: E402
from agents import turn_runner as tr  # noqa: E402
from config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def small_limits(monkeypatch):
    monkeypatch.setattr(settings, "agent_max_concurrent_turns", 2)
    monkeypatch.setattr(settings, "agent_queue_timeout_s", 0.3)
    monkeypatch.setattr(settings, "agent_turn_deadline_s", 5.0)
    tr._state.update(running=0, waiting=0)
    lr.reset()


def run(coro):
    return asyncio.run(coro)


async def _max_loop_lag(duration_s: float) -> float:
    worst, end = 0.0, time.perf_counter() + duration_s
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        await asyncio.sleep(0.01)
        worst = max(worst, time.perf_counter() - t0 - 0.01)
    return worst


def test_a_slow_turn_does_not_block_the_event_loop():
    async def main():
        turn = asyncio.create_task(tr.run_turn(time.sleep, 0.6))
        lag = await _max_loop_lag(0.5)
        await turn
        return lag
    assert run(main()) < 0.1                       # the loop kept running while the "agent" slept for 0.6 s


def test_teeth_the_old_blocking_pattern_froze_the_loop():
    """What the handlers used to do: call the blocking agent directly inside ``async def``."""
    async def main():
        async def old_handler():
            time.sleep(0.6)

        turn = asyncio.create_task(old_handler())
        lag = await _max_loop_lag(0.5)
        await turn
        return lag
    assert run(main()) >= 0.5


def test_the_turn_runs_on_a_worker_thread_with_the_callers_context():
    from agents import query_scope as qs
    seen = {}

    def work():
        seen["thread"] = threading.current_thread().name
        seen["scope"] = [f.value for f in qs.current()]
        seen["turn"] = lr._TURN.get() is not None
        return "done"

    async def main():
        with qs.scoped(qs.ScopeFilter("houses", "house_id", "h9")):
            return await tr.run_turn(work)

    assert run(main()) == "done"
    assert seen["thread"].startswith("agent-turn") and seen["scope"] == ["h9"] and seen["turn"] is True


def test_results_and_exceptions_pass_through_unchanged():
    async def main():
        assert await tr.run_turn(lambda a, b=0: a + b, 2, b=3) == 5
        with pytest.raises(KeyError):
            await tr.run_turn(lambda: {}["nope"])
    run(main())


def test_concurrency_is_bounded_and_extra_turns_wait_their_turn(monkeypatch):
    monkeypatch.setattr(settings, "agent_queue_timeout_s", 5.0)
    lock, live, peak = threading.Lock(), [0], [0]

    def work():
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.15)
        with lock:
            live[0] -= 1

    async def main():
        t0 = time.perf_counter()
        await asyncio.gather(*(tr.run_turn(work) for _ in range(6)))
        return time.perf_counter() - t0

    elapsed = run(main())
    assert peak[0] == 2 and elapsed >= 0.4                    # 6 turns, 2 at a time, 0.15 s each => 3 rounds


def test_a_turn_that_cannot_get_a_slot_is_told_the_assistant_is_busy():
    async def main():
        hold = [asyncio.create_task(tr.run_turn(time.sleep, 0.8)) for _ in range(2)]
        await asyncio.sleep(0.05)
        with pytest.raises(tr.Busy):
            await tr.run_turn(lambda: None)                  # queue timeout is 0.3 s
        await asyncio.gather(*hold)
    run(main())


def test_the_deadline_returns_promptly_and_stops_a_cooperative_turn(monkeypatch):
    monkeypatch.setattr(settings, "agent_turn_deadline_s", 0.25)
    stopped = threading.Event()

    def cooperative():
        try:
            while True:
                lr.check_cancelled()                        # what every model call does first
                time.sleep(0.02)
        except lr.TurnCancelled:
            stopped.set()
            raise

    async def main():
        t0 = time.perf_counter()
        with pytest.raises(tr.DeadlineExceeded):
            await tr.run_turn(cooperative)
        return time.perf_counter() - t0

    assert run(main()) < 1.0
    assert stopped.wait(2)                                   # the worker really stopped, it was not just abandoned


def test_the_slot_is_held_until_the_thread_really_ends(monkeypatch):
    """A turn stuck in a model call cannot be killed; its slot must not be handed to a new turn while it still runs."""
    monkeypatch.setattr(settings, "agent_max_concurrent_turns", 1)
    monkeypatch.setattr(settings, "agent_turn_deadline_s", 0.1)
    monkeypatch.setattr(settings, "agent_queue_timeout_s", 0.2)
    tr._SEMAPHORES.clear()

    async def main():
        with pytest.raises(tr.DeadlineExceeded):
            await tr.run_turn(time.sleep, 0.8)               # ignores cancellation (a hung HTTP call)
        monkeypatch.setattr(settings, "agent_turn_deadline_s", 5.0)
        with pytest.raises(tr.Busy):                         # the zombie still owns the only slot
            await tr.run_turn(lambda: "x")
        await asyncio.sleep(0.8)                             # ... until it finishes
        assert await tr.run_turn(lambda: "x") == "x"
    run(main())


def test_a_client_disconnect_cancels_the_turn():
    seen = threading.Event()

    def work():
        for _ in range(100):
            try:
                lr.check_cancelled()
            except lr.TurnCancelled:
                seen.set()
                return
            time.sleep(0.02)

    async def main():
        task = asyncio.create_task(tr.run_turn(work))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    run(main())
    assert seen.wait(2)


def test_status_counts_running_and_waiting_turns():
    async def main():
        hold = [asyncio.create_task(tr.run_turn(time.sleep, 0.4)) for _ in range(3)]
        await asyncio.sleep(0.1)
        snapshot = tr.status()
        await asyncio.gather(*hold, return_exceptions=True)
        return snapshot
    snap = run(main())
    assert snap["running"] == 2 and snap["waiting"] == 1 and snap["max_concurrent"] == 2
    assert tr.status()["running"] == 0 and tr.status()["waiting"] == 0


@pytest.mark.parametrize("exc,status,fragment", [
    (tr.Busy("x"), 503, "busy"), (tr.DeadlineExceeded("x"), 504, "too long"), (lr.TurnCancelled("x"), 504, "too long"),
    (lr.ModelUnavailable("x"), 503, "language model"), (ValueError("boom"), 500, "Agent error: boom"),
])
def test_failures_map_to_meaningful_http_responses(exc, status, fragment):
    code, message, headers = tr.describe_failure(exc)
    assert code == status and fragment in message
    if status == 503:
        assert "Retry-After" in headers


httpx = pytest.importorskip("httpx")


@pytest.fixture()
def app_module(monkeypatch):
    """The real app module, imported with tracing OFF (a test must not start a network exporter), state restored after."""
    import observability
    monkeypatch.setattr(settings, "phoenix_enabled", False)
    saved = observability._tracer
    main = pytest.importorskip("main")
    yield main
    observability._tracer = saved


def test_other_endpoints_stay_responsive_while_a_chat_turn_runs(app_module, monkeypatch):
    def slow_chat(message, history=None, include_metadata=False, session_id=None):
        time.sleep(0.7)
        return "hi", [], [], {}

    monkeypatch.setattr(app_module, "run_general_chat", slow_chat)

    async def main():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            chat = asyncio.create_task(client.post("/api/chat", json={"message": "hello"}))
            await asyncio.sleep(0.1)
            t0 = time.perf_counter()
            status = await client.get("/api/agent/status")
            status_ms = (time.perf_counter() - t0) * 1000
            reply = await chat
            return status, status_ms, reply

    status, status_ms, reply = run(main())
    assert reply.status_code == 200 and reply.json()["reply"] == "hi"
    assert status.status_code == 200 and status.json()["turns"]["running"] == 1 and status.json()["model"]["state"] == "closed"
    assert status_ms < 150                                    # answered while the 0.7 s turn was still running


def test_the_endpoints_translate_failures_into_clear_http_errors(app_module, monkeypatch):
    def model_down(*a, **k):
        raise lr.ModelUnavailable("The language model service is not responding (code_agent: ConnectionError).")

    monkeypatch.setattr(app_module, "run_general_chat", model_down)

    async def main():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.post("/api/chat", json={"message": "hello"})

    resp = run(main())
    assert resp.status_code == 503 and resp.headers["Retry-After"] == "30" and "language model service" in resp.json()["detail"]
