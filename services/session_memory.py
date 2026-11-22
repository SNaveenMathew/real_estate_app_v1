"""Conversation memory the server owns, keyed by ``session_id``.

Why
---
The browsers send the whole history back on every request, so the server never
knew what was said: a reload lost the thread, two tabs with the same session were
unaware of each other, and a follow-up could only be evaluated by a test that
handed the agent a fabricated history (``eval/golden_set.py``'s ``history``
field). The agent itself already accepts ``history``; nothing persisted it.

This module stores the turns in the same DuckDB file the rest of the app uses,
one row per message, ordered by ``seq`` and trimmed to the last
``session_memory_max_turns`` exchanges. It is deliberately small:

* **session_id is the key, agent_key scopes it** (``general`` or
  ``house:<house_id>``) so a house conversation and the general one can share a
  session without mixing their context;
* **the client's history still wins** — a caller that sends history is asking
  for that context, and replaying stored turns on top would change an existing
  answer;
* **a memory failure never fails a turn** (the API writes in a try/except): the
  transcript is a convenience, the answer is the product.

No message content is invented here: what is stored is exactly what the user
sent and the agent replied.
"""
from __future__ import annotations

from typing import Iterable

import db.duckdb_store as store
from config import settings

MAX_MESSAGES = 40          # 20 exchanges; keep last
DEFAULT_AGENT = "general"


def ensure_tables(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_session_turns (
            session_id  VARCHAR,
            agent_key   VARCHAR,
            seq         INTEGER,
            role        VARCHAR,      -- 'user' | 'assistant'
            content     VARCHAR,
            turn_id     VARCHAR,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, agent_key, seq)
        )
    """)


def agent_key(agent: str, house_id: str | None = None) -> str:
    """The scope a session's turns are stored under."""
    return f"house:{house_id}" if agent == "house" and house_id else (agent or DEFAULT_AGENT)


def _limit() -> int:
    turns = int(getattr(settings, "session_memory_max_turns", 20) or 20)
    return max(turns, 1) * 2                     # user + assistant per turn


def append_turn(
    session_id: str | None,
    agent: str,
    user_message: str,
    assistant_reply: str,
    *,
    house_id: str | None = None,
    turn_id: str | None = None,
) -> int:
    """Record one exchange; returns how many rows the session now holds (0 when skipped)."""
    if not session_id:
        return 0
    key = agent_key(agent, house_id)
    conn = store.get_conn()
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) FROM chat_session_turns WHERE session_id = ? AND agent_key = ?",
        [session_id, key],
    ).fetchone()
    seq = int(row[0] or 0)
    for offset, (role, content) in enumerate((("user", user_message), ("assistant", assistant_reply)), start=1):
        if not content:
            continue
        conn.execute(
            "INSERT INTO chat_session_turns (session_id, agent_key, seq, role, content, turn_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [session_id, key, seq + offset, role, str(content), turn_id],
        )
    # Keep the transcript bounded: a session is context, not an archive.
    conn.execute(
        "DELETE FROM chat_session_turns WHERE session_id = ? AND agent_key = ? AND seq <= ?",
        [session_id, key, (seq + 2) - _limit()],
    )
    return count(session_id, agent=agent, house_id=house_id)


def recent(
    session_id: str | None,
    agent: str = DEFAULT_AGENT,
    *,
    house_id: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """The last messages for this session/scope, oldest first, in agent ``history`` shape."""
    if not session_id:
        return []
    key = agent_key(agent, house_id)
    rows = store.get_conn().execute(
        "SELECT role, content FROM chat_session_turns WHERE session_id = ? AND agent_key = ? "
        "ORDER BY seq DESC LIMIT ?",
        [session_id, key, int(limit or _limit())],
    ).fetchall()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def history_or(
    provided: Iterable[dict] | None,
    session_id: str | None,
    agent: str = DEFAULT_AGENT,
    *,
    house_id: str | None = None,
) -> list[dict]:
    """What the agent should see: the caller's history when it sent one, else the stored transcript.

    A caller that sends history is stating the context it wants; stored turns are appended only when the caller
    sent nothing, so this cannot silently rewrite a conversation an older client is already maintaining.
    """
    sent = list(provided or [])
    if sent:
        return sent
    return recent(session_id, agent, house_id=house_id)


def count(session_id: str | None, agent: str = DEFAULT_AGENT, *, house_id: str | None = None) -> int:
    if not session_id:
        return 0
    row = store.get_conn().execute(
        "SELECT COUNT(*) FROM chat_session_turns WHERE session_id = ? AND agent_key = ?",
        [session_id, agent_key(agent, house_id)],
    ).fetchone()
    return int(row[0] or 0)


def sessions(limit: int = 50) -> list[dict]:
    """Sessions with how many turns each holds (diagnostics; no message content)."""
    rows = store.get_conn().execute(
        "SELECT session_id, agent_key, COUNT(*) AS rows, MAX(created_at) AS last_at "
        "FROM chat_session_turns GROUP BY session_id, agent_key ORDER BY last_at DESC LIMIT ?",
        [int(limit)],
    ).fetchall()
    return [
        {"session_id": sid, "agent_key": key, "messages": int(n), "last_at": str(last_at)}
        for sid, key, n, last_at in rows
    ]


def clear(session_id: str, agent: str | None = None, *, house_id: str | None = None) -> int:
    """Delete a session's stored turns (all scopes when ``agent`` is None); returns rows removed."""
    conn = store.get_conn()
    if agent is None:
        before = conn.execute(
            "SELECT COUNT(*) FROM chat_session_turns WHERE session_id = ?", [session_id]
        ).fetchone()[0]
        conn.execute("DELETE FROM chat_session_turns WHERE session_id = ?", [session_id])
    else:
        key = agent_key(agent, house_id)
        before = count(session_id, agent=agent, house_id=house_id)
        conn.execute(
            "DELETE FROM chat_session_turns WHERE session_id = ? AND agent_key = ?", [session_id, key]
        )
    return int(before or 0)
