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
        return (
            "Stored answer, continued.",
            [{"role": "assistant", "content": "Stored answer, continued."}],
            [],
            {
                "trace_id": None,
                "trace_url": None,
            },
        )

    monkeypatch.setattr("api.chat.turn_runner.run_turn", fake_run_turn)
    memory.append_turn("s-api", "general", "first question", "first answer")

    response = client.post("/api/chat", json={"message": "and in Denver?", "session_id": "s-api"})
    assert response.status_code == 200, response.text
    assert calls["history"] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
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


def test_retention_deletes_only_sessions_past_the_window(memory):
    """Per-session trimming bounds one conversation; retention bounds how many exist."""
    memory.append_turn("s-old", "general", "ancient question", "ancient answer")
    memory.append_turn("s-new", "general", "recent question", "recent answer")
    store.get_conn().execute(
        "UPDATE chat_session_turns SET created_at = now() - INTERVAL 90 DAY WHERE session_id = ?",
        ["s-old"],
    )

    removed = memory.purge(days=30)
    assert removed == 2
    assert memory.recent("s-old", "general") == []
    assert [m["content"] for m in memory.recent("s-new", "general")] == ["recent question", "recent answer"]
    memory.clear("s-new")


def test_retention_is_measured_from_the_newest_turn(memory):
    """A long-running conversation is not cut in half because its first turn is old."""
    store.get_conn().execute(
        "INSERT INTO chat_session_turns (session_id, agent_key, seq, role, content, created_at) "
        "VALUES ('s-long', 'general', 1, 'user', 'first ever question', now() - INTERVAL 90 DAY)",
    )
    memory.append_turn("s-long", "general", "today's question", "today's answer")

    assert memory.purge(days=30) == 0
    assert memory.recent("s-long", "general")[-2:] == [
        {"role": "user", "content": "today's question"},
        {"role": "assistant", "content": "today's answer"},
    ]
    memory.clear("s-long")


def test_retention_can_be_disabled(memory):
    memory.append_turn("s-keep", "general", "q", "a")
    store.get_conn().execute(
        "UPDATE chat_session_turns SET created_at = now() - INTERVAL 900 DAY WHERE session_id = ?", ["s-keep"]
    )
    assert memory.purge(days=0) == 0
    assert memory.count("s-keep", "general") == 2
    memory.clear("s-keep")


def test_pronoun_follow_up_is_resolved_against_the_previous_turn(memory):
    history = [
        {"role": "user", "content": "Which houses do I own in Denver?"},
        {"role": "assistant", "content": "You own one: 45 Oak Ave."},
    ]

    resolved = memory.resolve_followup("and how many bedrooms does it have?", history)
    assert resolved.startswith("and how many bedrooms does it have?")
    assert "Which houses do I own in Denver?" in resolved


def test_standalone_questions_are_left_alone(memory):
    history = [{"role": "user", "content": "Which houses do I own in Denver?"}]
    for question in (
        "How many houses are in Pittsburgh?",  # names its subject
        "What is the average risk score in Allegheny County?",
        "and then summarize every crime incident in the city for the last ten years",  # too long to guess
    ):
        assert memory.resolve_followup(question, history) == question


def test_follow_up_without_history_is_unchanged(memory):
    assert memory.resolve_followup("and in Denver?", []) == "and in Denver?"
    assert memory.resolve_followup("", [{"role": "user", "content": "x"}]) == ""
