"""HTTP API endpoints for General Chat and House Chat agents."""
from __future__ import annotations

import traceback
from fastapi import APIRouter, HTTPException

import db.duckdb_store as store
import db.vector_store as vs
from agents.house_agent import run_house_chat
from agents.general_agent import run_general_chat
from agents import llm_resilience, turn_runner
from observability import phoenix_enabled
from api.common import ChatRequest

router = APIRouter(tags=["chat"])


def turn_failure(exc: BaseException) -> HTTPException:
    status_code, message, headers = turn_runner.describe_failure(exc)
    if status_code == 500:
        traceback.print_exception(type(exc), exc, exc.__traceback__)  # unexpected: keep stack in server log
    return HTTPException(status_code, message, headers=headers or None)


_turn_failure = turn_failure  # alias for backwards compatibility


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

    try:
        reply, updated_history, artifacts = await turn_runner.run_turn(
            run_house_chat, house_id, req.message, req.history
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
    }


@router.post("/api/chat")
async def general_chat(req: ChatRequest):
    """General chat — cross-house, MSA, national risk questions."""
    try:
        reply, updated_history, artifacts, observability = await turn_runner.run_turn(
            run_general_chat,
            req.message,
            req.history,
            include_metadata=True,
            session_id=req.session_id,
        )
    except Exception as e:
        raise turn_failure(e)
    return {
        "reply": reply,
        "history": updated_history,
        "artifacts": artifacts,
        "observability": observability,
        "phoenix_enabled": phoenix_enabled(),
    }


@router.get("/api/agent/status")
async def agent_status():
    """Is the language model reachable, and how busy is the agent? Makes no model call."""
    return {
        "model": llm_resilience.breaker().status(),
        "breakers": llm_resilience.statuses(),
        "turns": turn_runner.status(),
    }

