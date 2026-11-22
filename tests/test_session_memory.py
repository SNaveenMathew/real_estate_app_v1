"""Server-side session memory (services/session_memory.py) and its API wiring.

The transcript is the server's own record of a conversation: it must isolate
sessions and scopes from each other, stay bounded, and never override a client
that already sends its own history.
"""
from __future__ import annotations

import pytest

import db.duckdb_store as store
from services import session_memory


@pytest.fixture()
def client(fresh_db, tmp_path, monkeypatch):
    """The real app, on the fixture database (same shape as tests/test_onboarding_api.py)."""
    from fastapi.testclient import TestClient
    from config import settings
    from services import dataset_onboarding as ob
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(ob, "warm_vector_index", lambda: False)
    import main
    return TestClient(main.app)


@pytest.fixture()
def memory(fresh_db):
    """fresh_db already builds the schema, which now includes the transcript table."""
    session_memory.clear("s-test")
    yield session_memory
    session_memory.clear("s-test")


def test_turns_round_trip_in_order(memory):
    memory.append_turn("s-test", "general", "how many houses?", "You have 6.")
    memory.append_turn("s-test", "general", "and in Denver?", "One.")

    history = memory.recent("s-test", "general")
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "how many houses?"
    assert history[-1]["content"] == "One."


def test_sessions_and_scopes_are_isolated(memory):
    memory.append_turn("s-test", "general", "general question", "general answer")
    memory.append_turn("s-test", "house", "house question", "house answer", house_id="h1")
    memory.append_turn("other-session", "general", "elsewhere", "elsewhere answer")

    assert [m["content"] for m in memory.recent("s-test", "general")] == ["general question", "general answer"]
    assert [m["content"] for m in memory.recent("s-test", "house", house_id="h1")] == ["house question", "house answer"]
    assert memory.recent("s-test", "house", house_id="h2") == []
    assert memory.recent("other-session", "general")[0]["content"] == "elsewhere"


def test_transcript_is_bounded(memory):
    for i in range(memory._limit() + 6):
        memory.append_turn("s-test", "general", f"q{i}", f"a{i}")

    history = memory.recent("s-test", "general")
    assert len(history) <= memory._limit()
    assert history[-1]["content"] == f"a{memory._limit() + 5}"
    # the oldest exchanges were trimmed, not the newest
    assert "q0" not in [m["content"] for m in history]


def test_client_history_wins_over_stored_turns(memory):
    memory.append_turn("s-test", "general", "stored question", "stored answer")
    supplied = [{"role": "user", "content": "client question"}, {"role": "assistant", "content": "client answer"}]

    assert memory.history_or(supplied, "s-test", "general") == supplied
    assert memory.history_or([], "s-test", "general")[0]["content"] == "stored question"
    assert memory.history_or(None, None, "general") == []


def test_clear_removes_only_the_named_scope(memory):
    memory.append_turn("s-test", "general", "g", "g")
    memory.append_turn("s-test", "house", "h", "h", house_id="h1")

    assert memory.clear("s-test", "general") == 2
    assert memory.recent("s-test", "general") == []
    assert len(memory.recent("s-test", "house", house_id="h1")) == 2


def test_api_uses_stored_history_when_the_client_sends_none(client, memory, monkeypatch):
    """A turn that sends only a session_id continues from the transcript; the reply extends it."""
    calls = {}

    async def fake_run_turn(fn, *args, **kwargs):
        calls["history"] = list(args[-1] if args else kwargs.get("history") or [])
        return "Stored answer, continued.", [{"role": "assistant", "content": "Stored answer, continued."}], [], {
            "trace_id": None, "trace_url": None,
        }

    monkeypatch.setattr("api.chat.turn_runner.run_turn", fake_run_turn)
    memory.append_turn("s-api", "general", "first question", "first answer")

    response = client.post("/api/chat", json={"message": "and in Denver?", "session_id": "s-api"})
    assert response.status_code == 200, response.text
    assert calls["history"] == [{"role": "user", "content": "first question"},
                                {"role": "assistant", "content": "first answer"}]
    stored = [m["content"] for m in memory.recent("s-api", "general")]
    assert stored[-2:] == ["and in Denver?", "Stored answer, continued."]
    memory.clear("s-api")


def test_api_prefers_explicit_history(client, memory, monkeypatch):
    calls = {}

    async def fake_run_turn(fn, *args, **kwargs):
        calls["history"] = list(args[-1] if args else kwargs.get("history") or [])
        return "ok", [], [], {"trace_id": None, "trace_url": None}

    monkeypatch.setattr("api.chat.turn_runner.run_turn", fake_run_turn)
    memory.append_turn("s-api2", "general", "stored", "stored answer")

    sent = [{"role": "user", "content": "client"}, {"role": "assistant", "content": "client answer"}]
    response = client.post("/api/chat", json={"message": "next", "session_id": "s-api2", "history": sent})
    assert response.status_code == 200, response.text
    assert calls["history"] == sent
    memory.clear("s-api2")


def test_memory_survives_a_new_connection_and_reports_sessions(fresh_db, memory):
    """The transcript is in the shared database, not a module dict."""
    memory.append_turn("s-persist", "general", "q", "a")
    store.close()
    assert [m["content"] for m in memory.recent("s-persist", "general")] == ["q", "a"]
    assert any(row["session_id"] == "s-persist" for row in memory.sessions())
    memory.clear("s-persist")
