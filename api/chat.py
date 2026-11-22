"""HTTP API endpoints for General Chat and House Chat agents."""
from __future__ import annotations

import asyncio
import json
import sys
import traceback
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import db.duckdb_store as store
import db.vector_store as vs
from agents.house_agent import run_house_chat, PROGRESS_STEPS as HOUSE_PROGRESS_STEPS
from agents.general_agent import run_general_chat, PROGRESS_STEPS as GENERAL_PROGRESS_STEPS
from agents import llm_resilience, turn_runner, turn_progress
from observability import phoenix_enabled
from api.common import ChatRequest

router = APIRouter(tags=["chat"])


def turn_failure(exc: BaseException) -> HTTPException:
    status_code, message, headers = turn_runner.describe_failure(exc)
    if status_code == 500:
        traceback.print_exception(type(exc), exc, exc.__traceback__)  # unexpected: keep stack in server log
    return HTTPException(status_code, message, headers=headers or None)


_turn_failure = turn_failure  # alias for backwards compatibility

# The browser opens the progress stream before the POST that registers the turn reaches the server.
_TRACKER_WAIT_S = 5.0


def _seed_progress(turn_id: str | None, steps) -> None:
    """Register a turn's step plan before the turn runs.

    The stream must never create the tracker itself: it is opened first, and a tracker created there would keep
    a plan that does not match the agent that ends up running (leaving steps that never execute in the panel).
    """
    if turn_id and not turn_progress.get_progress_tracker(turn_id):
        turn_progress.create_progress_tracker(turn_id, default_steps=list(steps))


@router.post("/api/house/{house_id}/chat")
async def house_chat(house_id: str, req: ChatRequest):
    """Chat with the house-specific agent."""
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")

    # Auto-save description if message looks like a property description (>100 chars, no ?)
    auto_saved = False
    if len(req.message) > 100 and "?" not in req.message[:50]:
        vs.upsert_description(house_id, req.message)  # enforces single description
        auto_saved = True

    _seed_progress(req.turn_id, HOUSE_PROGRESS_STEPS)
    try:
        reply, updated_history, artifacts = await turn_runner.run_turn(
            run_house_chat, house_id, req.message, req.history, turn_id=req.turn_id
        )
    except Exception as e:
        raise turn_failure(e)

    # Prepend acknowledgment if we auto-saved
    if auto_saved and "thank" not in reply.lower()[:50]:
        reply = "Thank you for the property description — it has been saved. " + reply

    trace_url = updated_history[-1].get("trace_url") if updated_history else None
    return {
        "reply": reply,
        "history": updated_history,
        "auto_saved": auto_saved,
        "artifacts": artifacts,
        "trace_url": trace_url,
        "turn_id": req.turn_id,
    }


@router.post("/api/chat")
async def general_chat(req: ChatRequest):
    """General chat — cross-house, MSA, national risk questions."""
    _seed_progress(req.turn_id, GENERAL_PROGRESS_STEPS)
    try:
        main_mod = sys.modules.get("main")
        chat_fn = getattr(main_mod, "run_general_chat", run_general_chat) if main_mod else run_general_chat
        reply, updated_history, artifacts, observability = await turn_runner.run_turn(
            chat_fn,
            req.message,
            req.history,
            include_metadata=True,
            session_id=req.session_id,
            turn_id=req.turn_id,
        )
    except Exception as e:
        raise turn_failure(e)
    return {
        "reply": reply,
        "history": updated_history,
        "artifacts": artifacts,
        "observability": observability,
        "phoenix_enabled": phoenix_enabled(),
        "turn_id": req.turn_id,
    }


@router.get("/api/chat/progress/{turn_id}")
def get_turn_progress(turn_id: str):
    """Get the current progress snapshot for an agent turn."""
    tracker = turn_progress.get_progress_tracker(turn_id)
    if not tracker:
        raise HTTPException(404, "Turn tracker not found")
    return tracker.get_state()


@router.get("/api/chat/progress/{turn_id}/stream")
async def stream_turn_progress(turn_id: str):
    """Stream real-time step progress updates using Server-Sent Events (SSE)."""
    tracker = turn_progress.get_progress_tracker(turn_id)
    waited = 0.0
    while tracker is None and waited < _TRACKER_WAIT_S:   # the POST registers the turn a moment after the stream opens
        await asyncio.sleep(0.05)
        waited += 0.05
        tracker = turn_progress.get_progress_tracker(turn_id)
    if tracker is None:
        raise HTTPException(404, "Turn tracker not found")

    async def event_generator():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        def listener(snapshot: dict):
            try:
                queue.put_nowait(snapshot)
            except Exception:
                pass

        tracker.add_listener(listener)
        try:
            while True:
                try:
                    state = await asyncio.wait_for(queue.get(), timeout=10.0)
                    yield f"data: {json.dumps(state)}\n\n"
                    if state.get("is_complete"):
                        break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            tracker.remove_listener(listener)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/agent/status")
def agent_status():
    """Is the language model reachable, and how busy is the agent? Makes no model call."""
    return {
        "model": llm_resilience.breaker().status(),
        "breakers": llm_resilience.statuses(),
        "turns": turn_runner.status(),
    }
