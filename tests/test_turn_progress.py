"""Unit tests for TurnProgressTracker and progress API endpoints."""

import asyncio
import json
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.turn_progress import (
    TurnProgressTracker,
    create_progress_tracker,
    get_progress_tracker,
    get_or_create_progress_tracker,
    BULLET_RUNNING,
    BULLET_COMPLETED,
    BULLET_PENDING,
    BULLET_FAILED,
)
from agents.general_agent import PROGRESS_STEPS as GENERAL_STEPS
from agents.house_agent import PROGRESS_STEPS as HOUSE_STEPS
import api.chat as api_chat
from api.chat import router

test_app = FastAPI()
test_app.include_router(router)
client = TestClient(test_app)


def _texts(tracker: TurnProgressTracker) -> list[str]:
    """Just the step titles, so ordering can be asserted independently of the bullet."""
    return [s["title"] for s in tracker.get_state()["steps"]]


def test_turn_progress_tracker_initial_state():
    """Verify initial state conforms to: ● Starting, o step..., o result."""
    tracker = TurnProgressTracker(
        "test_turn_1",
        default_steps=[("step_a", "step A"), ("step_b", "step B")],
    )
    formatted = tracker.format_steps()
    assert formatted == [
        "● Starting",
        "o step A",
        "o step B",
        "o result",
    ]
    assert not tracker.is_complete()


def test_turn_progress_tracker_without_a_plan_only_shows_start_and_result():
    """A tracker with no declared plan never invents steps that cannot run."""
    tracker = TurnProgressTracker("test_turn_no_plan")
    assert tracker.format_steps() == ["● Starting", "o result"]


def test_turn_progress_tracker_full_step_lifecycle():
    """
    Verify exact progression requested by the user:
    1. ● Starting / o step A / o step B / o result
    2. ✓ Started  / ● step A / o step B / o result
    3. ✓ Started  / ✓ step A / ● step B / o result
    4. ✓ Started  / ✓ step A / ✓ step B / ● result
    5. ✓ Started  / ✓ step A / ✓ step B / ✓ result
    """
    tracker = TurnProgressTracker(
        "test_turn_2",
        default_steps=[("step_a", "step A"), ("step_b", "step B")],
    )

    # State 1: Initial
    assert tracker.format_steps() == [
        "● Starting",
        "o step A",
        "o step B",
        "o result",
    ]

    # State 2: Step A starts
    tracker.start_step("step_a", "step A")
    assert tracker.format_steps() == [
        "✓ Started",
        "● step A",
        "o step B",
        "o result",
    ]

    # State 3: Step B starts
    tracker.start_step("step_b", "step B")
    assert tracker.format_steps() == [
        "✓ Started",
        "✓ step A",
        "● step B",
        "o result",
    ]

    # State 4: Result running
    tracker.set_result_running()
    assert tracker.format_steps() == [
        "✓ Started",
        "✓ step A",
        "✓ step B",
        "● result",
    ]

    # State 5: Complete all
    tracker.complete_all()
    assert tracker.format_steps() == [
        "✓ Started",
        "✓ step A",
        "✓ step B",
        "✓ result",
    ]
    assert tracker.is_complete()


def test_general_chat_plan_runs_in_declared_order():
    """The General Chat plan keeps its order and leaves no unused step behind."""
    tracker = TurnProgressTracker("general_plan_turn", default_steps=GENERAL_STEPS)
    assert tracker.format_steps() == [
        "● Starting",
        "o Query Planning",
        "o Data Model Retrieval",
        "o Code Generation",
        "o Code Execution",
        "o Answer Synthesis",
        "o result",
    ]

    # The calls agents/general_agent.py makes for one successful turn.
    tracker.start_step("query_planning", "Query Planning")
    assert tracker.format_steps()[:3] == ["✓ Started", "● Query Planning", "o Data Model Retrieval"]

    tracker.start_step("retrieval", "Data Model Retrieval")
    tracker.start_step("code_gen_1", "Code Generation")
    assert tracker.format_steps() == [
        "✓ Started",
        "✓ Query Planning",
        "✓ Data Model Retrieval",
        "● Code Generation",
        "o Code Execution",
        "o Answer Synthesis",
        "o result",
    ]

    tracker.start_step("code_exec_1", "Code Execution")
    assert tracker.format_steps()[4] == "● Code Execution"

    tracker.start_step("response_synthesis", "Answer Synthesis")
    assert tracker.format_steps() == [
        "✓ Started",
        "✓ Query Planning",
        "✓ Data Model Retrieval",
        "✓ Code Generation",
        "✓ Code Execution",
        "● Answer Synthesis",
        "o result",
    ]

    tracker.set_result_running()
    assert tracker.format_steps()[-1] == "● result"
    tracker.complete_all()
    assert tracker.format_steps() == [
        "✓ Started",
        "✓ Query Planning",
        "✓ Data Model Retrieval",
        "✓ Code Generation",
        "✓ Code Execution",
        "✓ Answer Synthesis",
        "✓ result",
    ]


def test_house_chat_plan_has_only_house_steps():
    """House Chat announces house steps only - no general-chat retrieval step leaks in."""
    tracker = TurnProgressTracker("house_plan_turn", default_steps=HOUSE_STEPS)
    assert _texts(tracker) == [
        "Starting",
        "Property Analysis",
        "Code Generation",
        "Code Execution",
        "Answer Synthesis",
        "result",
    ]
    assert "Query Planning" not in _texts(tracker)


def test_retry_inserts_an_attempt_step_before_result():
    """A second code-generation attempt is added as its own step, ahead of 'result'."""
    tracker = TurnProgressTracker("retry_plan_turn", default_steps=HOUSE_STEPS)
    tracker.start_step("property_analysis", "Property Analysis")
    tracker.start_step("house_code_gen_1", "Code Generation")
    tracker.start_step("house_code_exec_1", "Code Execution")
    tracker.start_step("house_code_gen_2", "Code Generation (Attempt 2)")

    assert tracker.format_steps() == [
        "✓ Started",
        "✓ Property Analysis",
        "✓ Code Generation",
        "✓ Code Execution",
        "● Code Generation (Attempt 2)",
        "o Answer Synthesis",
        "o result",
    ]
    tracker.set_result_running()
    assert tracker.format_steps()[-1] == "● result"
    assert tracker.format_steps()[-2] == "✓ Answer Synthesis"


def test_turn_progress_tracker_dynamic_step_insertion():
    """If unexpected or dynamic steps are added on the fly, they insert before result."""
    tracker = TurnProgressTracker("test_turn_dynamic")
    tracker.start_step("planning", "Query Planning")
    assert "✓ Started" in tracker.format_steps()[0]
    assert "● Query Planning" in tracker.format_steps()[1]

    tracker.start_step("custom_step", "Custom Recovery Step")
    steps = tracker.format_steps()
    assert any("✓ Query Planning" in s for s in steps)
    assert any("● Custom Recovery Step" in s for s in steps)
    assert steps[-1] == "o result"


def test_turn_progress_tracker_failure():
    """Verify failure marks current running step failed."""
    tracker = TurnProgressTracker("test_turn_fail", default_steps=[("step_a", "step A")])
    tracker.start_step("step_a", "step A")
    tracker.fail_current("Database connection timeout")
    assert tracker.is_complete()
    formatted = tracker.format_steps()
    assert any("✗ step A" in s for s in formatted)


def test_turn_progress_listeners():
    """Verify listeners receive state updates."""
    tracker = TurnProgressTracker("test_turn_listener")
    updates = []
    tracker.add_listener(lambda state: updates.append(state))

    tracker.start_step("step_1", "First Step")
    tracker.complete_all()

    assert len(updates) >= 3  # initial snapshot + start_step + complete_all
    assert updates[-1]["is_complete"] is True


def test_bullet_symbols():
    """The four bullet symbols used by the panel."""
    assert (BULLET_RUNNING, BULLET_COMPLETED, BULLET_PENDING, BULLET_FAILED) == ("●", "✓", "o", "✗")


def test_progress_api_snapshot_endpoint():
    """Test GET /api/chat/progress/{turn_id}."""
    tid = "api_test_turn_99"
    tracker = create_progress_tracker(tid, default_steps=[("step_a", "step A")])
    tracker.start_step("step_a", "step A")

    resp = client.get(f"/api/chat/progress/{tid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["turn_id"] == tid
    assert "formatted" in data
    assert data["formatted"] == [
        "✓ Started",
        "● step A",
        "o result",
    ]

    # Non-existent turn returns 404
    resp_404 = client.get("/api/chat/progress/non_existent_turn")
    assert resp_404.status_code == 404


def test_progress_api_stream_endpoint():
    """Test GET /api/chat/progress/{turn_id}/stream (SSE)."""
    tid = "api_stream_turn_101"
    tracker = get_or_create_progress_tracker(tid)
    tracker.complete_all()

    # Stream should return SSE data with is_complete: true and close
    with client.stream("GET", f"/api/chat/progress/{tid}/stream") as response:
        assert response.status_code == 200
        event_data = None
        for line in response.iter_lines():
            if line.startswith("data: "):
                raw = line.replace("data: ", "").strip()
                event_data = json.loads(raw)
                break
        assert event_data is not None
        assert event_data["turn_id"] == tid
        assert event_data["is_complete"] is True


def test_progress_stream_waits_for_a_turn_registered_after_it_opens():
    """The browser opens the stream before sending the message; the stream must attach to the turn that
    arrives a moment later instead of creating a tracker with the wrong plan."""
    tid = "api_stream_late_turn"

    async def scenario():
        pending = asyncio.create_task(api_chat.stream_turn_progress(tid))
        await asyncio.sleep(0.2)
        assert not pending.done(), "the stream should wait for its turn to register"
        assert get_progress_tracker(tid) is None, "the stream must never create the tracker itself"

        tracker = create_progress_tracker(tid, default_steps=[("step_a", "step A")])
        response = await pending
        assert response.media_type == "text/event-stream"

        async for chunk in response.body_iterator:
            text = chunk.decode() if isinstance(chunk, bytes) else chunk
            assert text.startswith("data: ")
            event = json.loads(text[len("data: ") :].strip())
            assert event["formatted"][:2] == ["● Starting", "o step A"]
            break
        await response.body_iterator.aclose()
        assert tracker.get_state()["is_complete"] is False

    asyncio.run(scenario())


def test_progress_stream_unknown_turn_returns_404(monkeypatch):
    """A stream for a turn that never registers fails fast instead of hanging."""
    monkeypatch.setattr(api_chat, "_TRACKER_WAIT_S", 0.05)
    resp = client.get("/api/chat/progress/never_registered_turn/stream")
    assert resp.status_code == 404


def test_general_chat_turn_registers_the_general_plan(monkeypatch):
    """POST /api/chat exposes its real plan to the panel, marks the turn complete, and reports 'result' last."""
    monkeypatch.delitem(sys.modules, "main", raising=False)  # don't prefer a wrapper from the app entrypoint
    seen: dict = {}

    def fake_general_chat(message, history, include_metadata=False, session_id=None):
        from agents.turn_progress import get_current_tracker

        tracker = get_current_tracker()
        seen["formatted"] = tracker.format_steps() if tracker else None
        return "ok", [{"role": "assistant", "content": "ok"}], [], {}

    monkeypatch.setattr(api_chat, "run_general_chat", fake_general_chat)
    tid = "route_general_plan_turn"
    resp = client.post("/api/chat", json={"message": "hi", "history": [], "turn_id": tid})

    assert resp.status_code == 200
    assert resp.json()["turn_id"] == tid
    # The agent sees the plan while it runs, and the turn is still in progress at that point.
    assert seen["formatted"] == [
        "● Starting",
        "o Query Planning",
        "o Data Model Retrieval",
        "o Code Generation",
        "o Code Execution",
        "o Answer Synthesis",
        "o result",
    ]
    state = get_progress_tracker(tid).get_state()
    assert state["is_complete"] is True
    assert state["formatted"] == [
        "✓ Started",
        "✓ Query Planning",
        "✓ Data Model Retrieval",
        "✓ Code Generation",
        "✓ Code Execution",
        "✓ Answer Synthesis",
        "✓ result",
    ]


def test_house_chat_turn_registers_the_house_plan(monkeypatch):
    """POST /api/house/{id}/chat registers house steps only."""
    import db.duckdb_store as duck_store

    monkeypatch.setattr(duck_store, "get_house", lambda house_id: {"house_id": house_id} if house_id == "h1" else None)
    seen: dict = {}

    def fake_house_chat(house_id, message, history):
        from agents.turn_progress import get_current_tracker

        tracker = get_current_tracker()
        seen["formatted"] = tracker.format_steps() if tracker else None
        return "ok", [{"role": "assistant", "content": "ok"}], []

    monkeypatch.setattr(api_chat, "run_house_chat", fake_house_chat)
    tid = "route_house_plan_turn"
    resp = client.post("/api/house/h1/chat", json={"message": "hi", "history": [], "turn_id": tid})

    assert resp.status_code == 200
    assert seen["formatted"] == [
        "● Starting",
        "o Property Analysis",
        "o Code Generation",
        "o Code Execution",
        "o Answer Synthesis",
        "o result",
    ]
    assert get_progress_tracker(tid).get_state()["formatted"] == [
        "✓ Started",
        "✓ Property Analysis",
        "✓ Code Generation",
        "✓ Code Execution",
        "✓ Answer Synthesis",
        "✓ result",
    ]


def test_failed_turn_marks_the_running_step_failed(monkeypatch):
    """A turn that raises leaves the panel on '✗' for the step that was running."""
    monkeypatch.delitem(sys.modules, "main", raising=False)

    def broken_general_chat(*args, **kwargs):
        from agents.turn_progress import get_current_tracker

        get_current_tracker().start_step("retrieval", "Data Model Retrieval")
        raise RuntimeError("model exploded")

    monkeypatch.setattr(api_chat, "run_general_chat", broken_general_chat)
    tid = "route_failed_turn"
    resp = client.post("/api/chat", json={"message": "hi", "history": [], "turn_id": tid})

    assert resp.status_code == 500
    state = get_progress_tracker(tid).get_state()
    assert state["is_complete"] is True
    failed = [s for s in state["steps"] if s["status"] == "failed"]
    assert [s["title"] for s in failed] == ["Data Model Retrieval"]
    assert state["formatted"][2] == "✗ Data Model Retrieval"
