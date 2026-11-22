"""Persistent result cache shared by BOTH chats — prompt/context cache, semantic cache and SQL-result cache.

Why it exists (issue #16): almost every data set this app holds changes rarely, and the same
intermediate work keeps recurring — the same question asked twice, a rephrased question that the
deterministic planner routes to the identical concepts/tables/operation, and the same SQL statement
(including identical CTEs) reached from General Chat and House Chat alike.  Recomputing any of them
costs a full local-model round trip (tens of seconds) for a value that has not changed.

Four ideas, kept deliberately narrow:

1. **Prompt / context cache** (``kind='prompt'``).  Keyed by a hash of the role, the model, the
   temperature, a prompt-template version tag and the ENTIRE message list.  Only deterministic
   (temperature 0) calls are eligible: an unchanged prompt is then guaranteed to produce the same
   completion, so replaying the stored text is not a guess.  Wired into
   ``agents/llm_resilience.invoke()`` — the single place every model call in the app goes through —
   so it applies to General Chat, House Chat, the SQL Code Agent, the semantic fallback and the
   relationship suggester at once, not to one agent.

2. **Semantic cache** (``kind='semantic'``).  Keyed by the *planner signature* — the sorted semantic
   concept keys, the planned tables, the operation and the mandatory query scope — plus a token
   similarity check against the stored question (``cache_semantic_threshold``).  Two questions are
   equivalent only when the deterministic semantic layer (``db/schema_catalog.py``) already routed
   them to the same concepts, the same tables, the same operation and the same scope, AND they read
   as the same question.  That is a stricter, auditable definition of "semantic" than an embedding
   cosine, and it costs no extra service call on the hot path: no embedding endpoint has to be up
   for a cache hit.  Wired into ``agents/tools.py::query_database``, which House Chat delegates to
   as well, so a cached answer is shared between the two chats.

3. **SQL / CTE result cache** (``cache_sql_results``).  Keyed by a hash of the normalized statement
   text plus the data fingerprint.  This is what makes a CTE reached by both chats cheap the second
   time.  Used by ``db/duckdb_store.query_limited_cached``.

4. **Storage and retrieval.**  DuckDB — the same file the application already uses — with
   ``INSERT OR REPLACE`` upserts on a ``PRIMARY KEY`` (ART index) for O(1) key lookups, and
   secondary indexes on the columns the page filters by.  No second database, no new dependency.
   Every write is a single statement, so a crash mid-write can never leave a half-written entry
   that a later read would trust.

**Invalidation** is the part that decides whether a cache is safe, so it is explicit rather than
time-only: every entry stores ``data_version``, a fingerprint of the catalog version, an explicit
epoch (bumped by ``invalidate()``) and the exact row count of every non-cache table.  A hit is only
served when the fingerprint is unchanged.  Row counts catch the realistic changes (a dataset
reloaded, rows added or replaced, a catalog edit); they would not catch an in-place update that
preserved a table's row count, which is why entries also expire on ``cache_ttl_s`` and the Cache
page has a Clear button.

The Cache page (``static/cache.html``) reads ``graph()`` below: abstractions (the live catalog's
semantic concepts — walkability, transit connectivity, and whatever else the catalog holds) on the
left, cached questions in the middle, the tables/CTEs they read on the right, so the connectedness
is visible at the abstraction level and clickable down to the individual statement.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
import threading
import time
from typing import Any, Iterable, Optional

from config import settings

logger = logging.getLogger(__name__)

# Bump when a prompt template changes, so older stored completions stop matching.
PROMPT_CACHE_VERSION = "1"

_DDL = [
    """CREATE TABLE IF NOT EXISTS cache_entries (
        cache_key      VARCHAR PRIMARY KEY,
        kind           VARCHAR,          -- 'prompt' | 'semantic'
        scope          VARCHAR,          -- model role, or the tool a semantic entry came from
        signature      VARCHAR,          -- planner signature (semantic entries only)
        label          VARCHAR,          -- short human label shown on the Cache page
        request_text   VARCHAR,          -- the question / prompt, for matching and display
        response       VARCHAR,          -- the cached completion / answer payload
        meta_json      VARCHAR,
        data_version   VARCHAR,
        hits           INTEGER DEFAULT 0,
        created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_hit_at    TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS cache_sql_results (
        sql_hash       VARCHAR PRIMARY KEY,
        sql_text       VARCHAR,
        request_text   VARCHAR,
        columns_json   VARCHAR,
        result_json    VARCHAR,
        row_count      INTEGER,
        truncated      BOOLEAN,
        tables_json    VARCHAR,
        data_version   VARCHAR,
        hits           INTEGER DEFAULT 0,
        created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_hit_at    TIMESTAMP)""",
    "CREATE TABLE IF NOT EXISTS cache_meta (key VARCHAR PRIMARY KEY, value VARCHAR)",
    "CREATE INDEX IF NOT EXISTS cache_entries_kind_scope ON cache_entries (kind, scope)",
    "CREATE INDEX IF NOT EXISTS cache_entries_signature ON cache_entries (signature)",
    "CREATE INDEX IF NOT EXISTS cache_sql_results_version ON cache_sql_results (data_version)",
]

_BASE_TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# Store plumbing
# ---------------------------------------------------------------------------


def ensure_tables(conn) -> None:
    """Create the cache tables on the shared connection (called from db.duckdb_store._ensure_schema)."""
    for stmt in _DDL:
        try:
            conn.execute(stmt)
        except Exception as exc:  # an older DuckDB without IF NOT EXISTS on indexes
            logger.debug("cache DDL skipped (%s): %s", type(exc).__name__, exc)


def _conn():
    from db import duckdb_store as store

    return store.get_conn()


def enabled() -> bool:
    return bool(getattr(settings, "cache_enabled", True))


def _ttl_s() -> float:
    return float(getattr(settings, "cache_ttl_s", 86400.0) or 0.0)


def _max_entries() -> int:
    return max(0, int(getattr(settings, "cache_max_entries", 5000) or 0))


def _rows(conn, sql: str, params: Iterable[Any] = ()) -> list[tuple]:
    params = list(params)
    return conn.execute(sql, params).fetchall() if params else conn.execute(sql).fetchall()


def _hash(*parts: Any) -> str:
    payload = "\x00".join(str(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8", "replace")).hexdigest()


def _now_epoch() -> float:
    return time.time()


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return str(value)


_WORD = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    """
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
me my of on or our over show than that the their them then there these they this to us was we were what
when where which who why will with would you your please tell give list all any about
""".split()
)


def normalize_text(text: Any) -> str:
    """Lowercase, punctuation-free, whitespace-collapsed — the form two questions are compared in."""
    return " ".join(_WORD.findall(str(text or "").lower()))


def _tokens(text: Any) -> set[str]:
    return {t for t in _WORD.findall(str(text or "").lower()) if len(t) > 2 and t not in _STOPWORDS}


def token_similarity(a: Any, b: Any) -> float:
    """Dice coefficient over content words — deterministic, needs no embedding service.

    Dice rather than Jaccard because these questions are short: one extra word should not by itself
    push two otherwise identical questions below the threshold.  This is only ever the SECOND gate —
    ``semantic_signature`` has already required the same concepts, tables, operation, filters and
    scope, so a reused answer can never come from a differently-scoped question.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return 2.0 * len(ta & tb) / (len(ta) + len(tb))


def _conn_ready() -> bool:
    try:
        _conn()
        return True
    except Exception as exc:
        logger.debug("cache: store unavailable (%s)", exc)
        return False


# ---------------------------------------------------------------------------
# Data fingerprint — what makes an entry trustworthy
# ---------------------------------------------------------------------------

_VER_LOCK = threading.Lock()
_VER: dict[str, Any] = {"value": "", "at": 0.0}


def _catalog_version(conn) -> str:
    try:
        from db import catalog_store

        return str(catalog_store.get_version(conn))
    except Exception:
        return "?"


def _epoch(conn) -> str:
    try:
        row = conn.execute("SELECT value FROM cache_meta WHERE key = 'epoch'").fetchone()
        return str(row[0]) if row and row[0] is not None else "0"
    except Exception:
        return "0"


def bump_epoch(conn=None) -> str:
    """Explicit invalidation: every stored entry stops matching immediately."""
    conn = conn or _conn()
    value = str(int(time.time() * 1000))
    conn.execute("INSERT OR REPLACE INTO cache_meta (key, value) VALUES ('epoch', ?)", [value])
    with _VER_LOCK:
        _VER["value"] = ""
    return value


def data_version(force: bool = False) -> str:
    """Fingerprint of the loaded data + catalog; ``""`` means "unknown", and nothing is cached on that.

    Cheap by construction: DuckDB answers ``COUNT(*)`` from row-group metadata, the whole result is
    memoized for ``cache_fingerprint_ttl_s``, and the ``cache_*`` tables are excluded so writing to
    the cache never invalidates the cache.
    """
    if not force and not _ttl_s():
        return ""
    now = time.monotonic()
    ttl = float(getattr(settings, "cache_fingerprint_ttl_s", 3.0) or 0.0)
    with _VER_LOCK:
        if not force and _VER["value"] and now - _VER["at"] < ttl:
            return _VER["value"]
    try:
        conn = _conn()
        names = [
            r[0]
            for r in _rows(
                conn,
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' ORDER BY table_name",
            )
        ]
        parts = [f"catalog:{_catalog_version(conn)}", f"epoch:{_epoch(conn)}"]
        for name in names:
            if not _BASE_TABLE.match(name) or name.startswith("cache_"):
                continue
            try:
                count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            except Exception:
                count = -1
            parts.append(f"{name}:{count}")
        value = _hash("|".join(parts))
    except Exception as exc:  # no fingerprint -> no caching, never a wrong hit
        logger.debug("cache: cannot compute data_version (%s)", exc)
        return ""
    with _VER_LOCK:
        _VER["value"], _VER["at"] = value, time.monotonic()
    return value


# ---------------------------------------------------------------------------
# Prompt / context cache
# ---------------------------------------------------------------------------


def _content_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                chunks.append(str(item.get("text") or item.get("content") or item))
            else:
                chunks.append(str(getattr(item, "text", item)))
        return "".join(chunks)
    return str(content)


def _messages_text(messages: Iterable[Any]) -> str:
    rendered = []
    for message in messages or []:
        role = getattr(message, "type", None) or getattr(message, "role", None) or type(message).__name__
        rendered.append(f"{role}: {_content_text(message)}")
    return "\n\n".join(rendered)


def _model_identity(client: Any) -> tuple[str, str]:
    model = getattr(client, "model_name", None) or getattr(client, "model", None) or type(client).__name__
    temperature = getattr(client, "temperature", None)
    return str(model), str(temperature)


def prompt_key(scope: str, client: Any, messages: Iterable[Any]) -> Optional[str]:
    """Cache key for one model call, or ``None`` when this call must not be cached.

    Only deterministic calls qualify: replaying a temperature-0 completion for an unchanged prompt is
    exact reuse, whereas caching a sampling call would pin one arbitrary sample as the answer.
    """
    if not enabled() or not getattr(settings, "cache_prompt_enabled", True):
        return None
    model, temperature = _model_identity(client)
    # An UNSPECIFIED temperature is not deterministic: it means "whatever the server defaults to",
    # which for a local llama-server is a sampling temperature. Only an explicit 0 qualifies.
    if temperature not in ("0", "0.0"):
        return None
    body = _messages_text(messages)
    if not body.strip():
        return None
    return _hash("prompt", PROMPT_CACHE_VERSION, scope, model, temperature, body)


def prompt_label(messages: Iterable[Any]) -> str:
    """The last human message, trimmed — enough for the Cache page to identify the call."""
    label = ""
    for message in messages or []:
        role = getattr(message, "type", None) or getattr(message, "role", None) or ""
        if str(role) in ("human", "user"):
            label = _content_text(message)
    label = " ".join(str(label).split())
    return label[:300]


def lookup_prompt(cache_key: str) -> Optional[str]:
    """The stored completion for ``cache_key``, or ``None``.  Never raises."""
    if not cache_key or not _conn_ready():
        return None
    ttl = _ttl_s()
    cutoff = _now_epoch() - ttl if ttl > 0 else 0.0
    version = data_version()
    if not version:
        return None
    try:
        conn = _conn()
        row = conn.execute(
            "SELECT response FROM cache_entries "
            "WHERE cache_key = ? AND kind = 'prompt' AND data_version = ? "
            "AND (? = 0 OR epoch(created_at) >= ?)",
            [cache_key, version, cutoff, cutoff],
        ).fetchone()
        if row is None or not row[0]:
            _evict_expired(conn)
            return None
        _touch(conn, "cache_entries", "cache_key", cache_key)
        return str(row[0])
    except Exception as exc:
        logger.debug("cache: prompt lookup failed (%s)", exc)
        return None


def store_prompt(
    cache_key: str, scope: str, label: str, request_text: str, response: str, meta: Optional[dict] = None
) -> None:
    """Upsert one completion.  Never raises."""
    if not cache_key or not response or not _conn_ready():
        return
    version = data_version()
    if not version:
        return
    try:
        conn = _conn()
        conn.execute(
            """INSERT OR REPLACE INTO cache_entries
               (cache_key, kind, scope, signature, label, request_text, response, meta_json,
                data_version, hits, created_at, last_hit_at)
               VALUES (?, 'prompt', ?, '', ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP, NULL)""",
            [
                cache_key,
                scope,
                label[:300],
                request_text[:4000],
                response,
                json.dumps(meta or {}, ensure_ascii=False, default=str)[:8000],
                version,
            ],
        )
        _maybe_evict(conn)
    except Exception as exc:
        logger.debug("cache: prompt store failed (%s)", exc)


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------


def semantic_signature(query_plan: Any, scope: Optional[Iterable[Any]] = None) -> Optional[str]:
    """The identity of a question's *meaning*, as decided by the deterministic planner.

    ``None`` when the plan matched no semantic concept: a question the catalog could not place is not
    something to reuse for a different question that also matched nothing.
    """
    if not enabled() or not getattr(settings, "cache_semantic_enabled", True):
        return None
    keys = sorted(getattr(query_plan, "semantic_keys", None) or [])
    if not keys:
        return None
    tables = sorted(getattr(query_plan, "required_tables", None) or [])
    operation = str(getattr(query_plan, "operation", "") or "")
    # Resolved entities and semantic filters belong in the signature: "how many houses" and "how many
    # houses in Pittsburgh" select the same concept over the same table with the same operation, so
    # without these two the second would be answered from the first one's cache.
    entities = sorted(str(f) for f in (getattr(query_plan, "entity_filters", None) or []))
    filters = sorted(str(f) for f in (getattr(query_plan, "filters", None) or []))
    if scope is None:
        from agents import query_scope

        scope = query_scope.current()
    scope_filters = sorted(f.render() for f in scope)
    return _hash(
        "sig",
        ",".join(keys),
        ",".join(tables),
        operation,
        "&".join(entities),
        "&".join(filters),
        "&".join(scope_filters),
    )


def lookup_semantic(scope: str, request_text: str, signature: Optional[str]) -> Optional[dict]:
    """A previously answered equivalent question, or ``None``.

    Returns ``{"response", "label", "similarity", "hits", "created_at"}``.
    """
    if not signature or not request_text.strip() or not _conn_ready():
        return None
    version = data_version()
    if not version:
        return None
    ttl = _ttl_s()
    cutoff = _now_epoch() - ttl if ttl > 0 else 0.0
    limit = max(1, int(getattr(settings, "cache_semantic_scan_limit", 400) or 400))
    threshold = float(getattr(settings, "cache_semantic_threshold", 0.85) or 0.85)
    try:
        conn = _conn()
        candidates = conn.execute(
            """SELECT cache_key, request_text, response, label, hits, created_at
               FROM cache_entries
               WHERE kind = 'semantic' AND scope = ? AND signature = ? AND data_version = ?
                 AND (? = 0 OR epoch(created_at) >= ?)
               ORDER BY created_at DESC LIMIT ?""",
            [scope, signature, version, cutoff, cutoff, limit],
        ).fetchall()
    except Exception as exc:
        logger.debug("cache: semantic lookup failed (%s)", exc)
        return None

    normalized = normalize_text(request_text)
    best = None  # (similarity, row)
    for row in candidates:
        cache_key, stored_text, response, label, hits, created_at = row
        if not response:
            continue
        if normalize_text(stored_text) == normalized:  # identical wording: a certain match
            best = (1.0, row)
            break
        similarity = token_similarity(request_text, stored_text)
        if similarity >= threshold and (best is None or similarity > best[0]):
            best = (similarity, row)
    if best is None:
        return None

    similarity, (cache_key, stored_text, response, label, hits, created_at) = best
    _touch(_conn(), "cache_entries", "cache_key", cache_key)
    return {
        "response": response,
        "label": label or stored_text,
        "similarity": round(similarity, 3),
        "hits": int(hits or 0) + 1,
        "created_at": _iso(created_at),
    }


def store_semantic(
    scope: str, request_text: str, signature: Optional[str], response: str, meta: Optional[dict] = None
) -> None:
    """Upsert one answered question under its planner signature.  Never raises."""
    if not signature or not response or not request_text.strip() or not _conn_ready():
        return
    version = data_version()
    if not version:
        return
    cache_key = _hash("semantic", scope, signature, normalize_text(request_text))
    label = " ".join(request_text.split())[:300]
    try:
        conn = _conn()
        conn.execute(
            """INSERT OR REPLACE INTO cache_entries
               (cache_key, kind, scope, signature, label, request_text, response, meta_json,
                data_version, hits, created_at, last_hit_at)
               VALUES (?, 'semantic', ?, ?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP, NULL)""",
            [
                cache_key,
                scope,
                signature,
                label,
                request_text[:4000],
                response,
                json.dumps(meta or {}, ensure_ascii=False, default=str)[:8000],
                version,
            ],
        )
        _maybe_evict(conn)
    except Exception as exc:
        logger.debug("cache: semantic store failed (%s)", exc)


# ---------------------------------------------------------------------------
# SQL / CTE result cache
# ---------------------------------------------------------------------------


def sql_cache_available(params=None) -> bool:
    """Prepared parameters make a statement's result depend on more than its text, so those are skipped."""
    return enabled() and bool(getattr(settings, "cache_sql_enabled", True)) and not params and _conn_ready()


def _sql_hash(sql_text: str) -> str:
    return _hash("sql", " ".join(str(sql_text).split()))


def _tables_in_sql(sql_text: str) -> list[str]:
    import re as _re

    pattern = _re.compile(r"\b(?:FROM|JOIN)\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)", _re.I)
    return sorted({m.group(1).lower() for m in pattern.finditer(str(sql_text))})


def lookup_sql(sql_text: str) -> Optional[tuple]:
    """``(DataFrame, truncated)`` for an identical earlier statement, or ``None``.  Never raises."""
    if not enabled() or not getattr(settings, "cache_sql_enabled", True) or not _conn_ready():
        return None
    version = data_version()
    if not version:
        return None
    ttl = _ttl_s()
    cutoff = _now_epoch() - ttl if ttl > 0 else 0.0
    key = _sql_hash(sql_text)
    try:
        conn = _conn()
        row = conn.execute(
            """SELECT columns_json, result_json, row_count, truncated FROM cache_sql_results
               WHERE sql_hash = ? AND data_version = ?
                 AND (? = 0 OR epoch(created_at) >= ?)""",
            [key, version, cutoff, cutoff],
        ).fetchone()
        if row is None:
            return None
        columns_json, result_json, row_count, truncated = row
        if not result_json:
            return None
        import pandas as pd
        from io import StringIO

        frame = pd.read_json(StringIO(result_json), orient="split")
        columns = json.loads(columns_json) if columns_json else list(frame.columns)
        frame.columns = columns
        _touch(conn, "cache_sql_results", "sql_hash", key)
        return frame, bool(truncated)
    except Exception as exc:
        logger.debug("cache: sql lookup failed (%s)", exc)
        return None


def store_sql(sql_text: str, frame, truncated: bool, request_text: str = "") -> None:
    """Upsert an executed statement's result.  Empty frames are not stored.  Never raises."""
    if not enabled() or not getattr(settings, "cache_sql_enabled", True) or not _conn_ready():
        return
    try:
        if frame is None or len(frame) == 0:
            return
    except Exception:
        return
    version = data_version()
    if not version:
        return
    try:
        payload = frame.to_json(orient="split", date_format="iso", force_ascii=False)
        if not payload or payload == "null":
            return
        conn = _conn()
        conn.execute(
            """INSERT OR REPLACE INTO cache_sql_results
               (sql_hash, sql_text, request_text, columns_json, result_json, row_count, truncated,
                tables_json, data_version, hits, created_at, last_hit_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP, NULL)""",
            [
                _sql_hash(sql_text),
                str(sql_text)[:20000],
                str(request_text)[:2000],
                json.dumps([str(c) for c in frame.columns], ensure_ascii=False),
                payload,
                int(len(frame)),
                bool(truncated),
                json.dumps(_tables_in_sql(sql_text), ensure_ascii=False),
                version,
            ],
        )
        _maybe_evict(conn)
    except Exception as exc:
        logger.debug("cache: sql store failed (%s)", exc)


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

_WRITES = {"n": 0}


def _touch(conn, table: str, key_column: str, key: str) -> None:
    """Record one hit (never fatal: a counter is not worth failing a served answer over)."""
    try:
        conn.execute(
            f"UPDATE {table} SET hits = hits + 1, last_hit_at = CURRENT_TIMESTAMP WHERE {key_column} = ?", [key]
        )
    except Exception:
        pass


def _evict_expired(conn) -> None:
    ttl = _ttl_s()
    if ttl <= 0:
        return
    cutoff = _now_epoch() - ttl
    try:
        conn.execute("DELETE FROM cache_entries WHERE epoch(created_at) < ?", [cutoff])
        conn.execute("DELETE FROM cache_sql_results WHERE epoch(created_at) < ?", [cutoff])
    except Exception:
        pass


def _maybe_evict(conn) -> None:
    """Keep the store bounded: drop expired rows periodically, then the least recently used."""
    _WRITES["n"] += 1
    if _WRITES["n"] % 25:
        return
    _evict_expired(conn)
    limit = _max_entries()
    if limit <= 0:
        return
    for table, key in (("cache_entries", "cache_key"), ("cache_sql_results", "sql_hash")):
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if total and int(total) > limit:
                conn.execute(
                    f"DELETE FROM {table} WHERE {key} IN ("
                    f"  SELECT {key} FROM {table} ORDER BY COALESCE(last_hit_at, created_at) ASC "
                    f"  LIMIT ?)",
                    [int(total) - limit],
                )
        except Exception:
            pass


def invalidate(kind: Optional[str] = None) -> dict:
    """Forget stored entries (all of them, or one kind) and move the epoch forward."""
    if not _conn_ready():
        return {"ok": False, "error": "The cache store is not available."}
    conn = _conn()
    try:
        if kind in ("prompt", "semantic"):
            conn.execute("DELETE FROM cache_entries WHERE kind = ?", [kind])
        elif kind == "sql":
            conn.execute("DELETE FROM cache_sql_results")
        else:
            conn.execute("DELETE FROM cache_entries")
            conn.execute("DELETE FROM cache_sql_results")
        bump_epoch(conn)
        return {"ok": True, "cleared": kind or "all", "data_version": data_version(force=True)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Reading the cache (stats + the Cache page's graph model)
# ---------------------------------------------------------------------------


def _totals(conn, table: str, where: str = "", params: Iterable[Any] = ()) -> dict:
    clause = f" WHERE {where}" if where else ""
    try:
        row = conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(hits), 0), MIN(created_at), MAX(created_at) FROM {table}{clause}",
            list(params),
        ).fetchone()
    except Exception:
        return {"entries": 0, "hits": 0, "oldest": "", "newest": ""}
    return {"entries": int(row[0] or 0), "hits": int(row[1] or 0), "oldest": _iso(row[2]), "newest": _iso(row[3])}


def stats() -> dict:
    """Counters for the Cache page header and the health endpoint.  Never raises."""
    out = {"enabled": enabled(), "data_version": "", "prompt": {}, "semantic": {}, "sql": {}, "total_entries": 0}
    if not _conn_ready():
        return out
    conn = _conn()
    out["data_version"] = data_version()
    out["prompt"] = {
        "enabled": bool(getattr(settings, "cache_prompt_enabled", True)),
        **_totals(conn, "cache_entries", "kind = 'prompt'"),
    }
    out["semantic"] = {
        "enabled": bool(getattr(settings, "cache_semantic_enabled", True)),
        **_totals(conn, "cache_entries", "kind = 'semantic'"),
    }
    out["sql"] = {"enabled": bool(getattr(settings, "cache_sql_enabled", True)), **_totals(conn, "cache_sql_results")}
    out["total_entries"] = out["prompt"]["entries"] + out["semantic"]["entries"] + out["sql"]["entries"]
    out["thresholds"] = {
        "semantic": float(getattr(settings, "cache_semantic_threshold", 0.85) or 0.85),
        "ttl_s": _ttl_s(),
    }
    return out


def entries(kind: Optional[str] = None, limit: int = 300, query: str = "") -> list[dict]:
    """Raw cache rows for the inspector / API.  Cached answers are data, rendered as text only."""
    if not _conn_ready():
        return []
    conn = _conn()
    limit = max(1, min(int(limit or 300), 2000))
    q = str(query or "").strip().lower()
    out: list[dict] = []
    try:
        if kind != "sql":
            params: list[Any] = []
            where = []
            if kind in ("prompt", "semantic"):
                where.append("kind = ?")
                params.append(kind)
            if q:
                where.append("lower(COALESCE(request_text, '') || ' ' || COALESCE(label, '')) LIKE ?")
                params.append(f"%{q}%")
            clause = f" WHERE {' AND '.join(where)}" if where else ""
            params.append(limit)
            for row in conn.execute(
                f"""SELECT cache_key, kind, scope, signature, label, request_text, meta_json,
                           data_version, hits, created_at, last_hit_at,
                           LENGTH(response) AS response_len
                    FROM cache_entries{clause}
                    ORDER BY COALESCE(last_hit_at, created_at) DESC LIMIT ?""",
                params,
            ).fetchall():
                out.append(
                    {
                        "cache_key": row[0],
                        "kind": row[1],
                        "scope": row[2],
                        "signature": row[3],
                        "label": row[4],
                        "request_text": row[5],
                        "meta": _loads(row[6], {}),
                        "data_version": row[7],
                        "hits": int(row[8] or 0),
                        "created_at": _iso(row[9]),
                        "last_hit_at": _iso(row[10]),
                        "response_length": int(row[11] or 0),
                    }
                )
        if kind in (None, "sql"):
            params = [limit]
            clause = ""
            if q:
                clause = " WHERE lower(COALESCE(request_text, '') || ' ' || COALESCE(sql_text, '')) LIKE ?"
                params = [f"%{q}%", limit]
            for row in conn.execute(
                f"""SELECT sql_hash, sql_text, request_text, row_count, truncated, tables_json,
                           data_version, hits, created_at, last_hit_at
                    FROM cache_sql_results{clause}
                    ORDER BY COALESCE(last_hit_at, created_at) DESC LIMIT ?""",
                params,
            ).fetchall():
                out.append(
                    {
                        "cache_key": row[0],
                        "kind": "sql",
                        "scope": "sql",
                        "signature": "",
                        "label": (row[2] or row[1] or "")[:300],
                        "request_text": row[2] or "",
                        "sql": row[1],
                        "meta": {
                            "tables": _loads(row[5], []),
                            "row_count": int(row[3] or 0),
                            "truncated": bool(row[4]),
                        },
                        "data_version": row[6],
                        "hits": int(row[7] or 0),
                        "created_at": _iso(row[8]),
                        "last_hit_at": _iso(row[9]),
                        "response_length": int(row[3] or 0),
                    }
                )
    except Exception as exc:
        logger.debug("cache: entries query failed (%s)", exc)
        return []
    return out


def get_entry(cache_key: str) -> Optional[dict]:
    """One stored entry in full, for the page's inspector.  ``None`` when it is gone."""
    if not cache_key or not _conn_ready():
        return None
    conn = _conn()
    try:
        row = conn.execute(
            """SELECT cache_key, kind, scope, label, request_text, response, meta_json,
                      data_version, hits, created_at, last_hit_at
               FROM cache_entries WHERE cache_key = ?""",
            [cache_key],
        ).fetchone()
        if row is not None:
            return {
                "cache_key": row[0],
                "kind": row[1],
                "scope": row[2],
                "label": row[3],
                "request_text": row[4] or "",
                "response": row[5] or "",
                "meta": _loads(row[6], {}),
                "data_version": row[7],
                "hits": int(row[8] or 0),
                "created_at": _iso(row[9]),
                "last_hit_at": _iso(row[10]),
            }
        row = conn.execute(
            """SELECT sql_hash, sql_text, request_text, row_count, truncated, tables_json,
                      data_version, hits, created_at, last_hit_at
               FROM cache_sql_results WHERE sql_hash = ?""",
            [cache_key],
        ).fetchone()
        if row is None:
            return None
        return {
            "cache_key": row[0],
            "kind": "sql",
            "scope": "sql",
            "label": (row[2] or "")[:300],
            "request_text": row[2] or "",
            "sql": row[1] or "",
            "response": "",
            "meta": {"tables": _loads(row[5], []), "row_count": int(row[3] or 0), "truncated": bool(row[4])},
            "data_version": row[6],
            "hits": int(row[7] or 0),
            "created_at": _iso(row[8]),
            "last_hit_at": _iso(row[9]),
        }
    except Exception as exc:
        logger.debug("cache: reading one entry failed (%s)", exc)
        return None


def _loads(text: Any, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _concept_label(concept: dict) -> str:
    aliases = concept.get("aliases") or []
    if aliases:
        return str(aliases[0])
    key = str(concept.get("key") or "")
    return key.replace("_", " ")


def graph(include_empty: bool = True) -> dict:
    """The Cache page's node/edge model.

    Level 1 **abstractions** come from the live catalog's semantic concepts, so "walkability",
    "transit connectivity" and anything else the catalog (or a dataset you upload) declares are the
    vocabulary the graph is drawn in — not a hardcoded list.  Level 2 is the cached questions that
    the planner attributed to those concepts.  Level 3 is the tables/CTEs those questions read.
    """
    out: dict = {
        "stats": stats(),
        "abstractions": [],
        "questions": [],
        "tables": [],
        "roles": [],
        "edges": [],
        "domains": [],
    }
    if not _conn_ready():
        return out
    conn = _conn()

    catalog = {"tables": [], "concepts": [], "domains": []}
    try:
        import db.schema_catalog as schema_catalog

        catalog = schema_catalog.describe_catalog()
    except Exception as exc:
        logger.debug("cache: catalog unavailable for the graph (%s)", exc)

    table_meta = {t["name"]: t for t in catalog.get("tables", [])}
    out["domains"] = catalog.get("domains", [])

    # ── cached rows ────────────────────────────────────────────────────────
    semantic_rows: list[dict] = []
    prompt_rows: list[dict] = []
    try:
        for row in conn.execute(
            """SELECT cache_key, kind, scope, label, request_text, meta_json, hits, created_at,
                      LENGTH(response) AS response_len
               FROM cache_entries ORDER BY COALESCE(last_hit_at, created_at) DESC LIMIT 1000"""
        ).fetchall():
            item = {
                "cache_key": row[0],
                "scope": row[2],
                "label": row[3] or "",
                "request_text": row[4] or "",
                "meta": _loads(row[5], {}),
                "hits": int(row[6] or 0),
                "created_at": _iso(row[7]),
                "response_length": int(row[8] or 0),
            }
            (prompt_rows if row[1] == "prompt" else semantic_rows).append(item)
    except Exception as exc:
        logger.debug("cache: graph rows unavailable (%s)", exc)

    # ── abstraction nodes (every catalog concept, cached or not) ───────────
    concepts = catalog.get("concepts", [])
    concept_ids = set()
    for concept in concepts:
        key = str(concept.get("key") or "")
        if not key:
            continue
        tables = [t for t in (concept.get("tables") or []) if t in table_meta] or list(concept.get("tables") or [])
        domain = (table_meta.get(tables[0], {}).get("domain") if tables else None) or "other"
        node = {
            "id": f"c:{key}",
            "kind": "concept",
            "key": key,
            "label": _concept_label(concept),
            "description": concept.get("description", ""),
            "domain": domain,
            "tables": tables,
            "origin": concept.get("origin", "builtin"),
            "questions": 0,
            "hits": 0,
        }
        out["abstractions"].append(node)
        concept_ids.add(key)

    for row in semantic_rows:
        # A concept the catalog no longer knows (retired dataset) is dropped from the links but the
        # cached question stays visible, so nothing silently disappears from the picture.
        row["concepts"] = [k for k in (row["meta"].get("concepts") or []) if k in concept_ids]

    # ── question nodes ────────────────────────────────────────────────────
    for row in semantic_rows:
        node = {
            "id": f"q:{row['cache_key']}",
            "kind": "question",
            "label": row["label"] or row["request_text"],
            "request_text": row["request_text"],
            "concepts": row["concepts"],
            "tables": [t for t in (row["meta"].get("tables") or [])],
            "operation": row["meta"].get("operation", ""),
            "hits": row["hits"],
            "created_at": row["created_at"],
            "cache_key": row["cache_key"],
            "scope": row["scope"],
        }
        out["questions"].append(node)

    # ── table nodes ───────────────────────────────────────────────────────
    used_tables: dict[str, dict] = {}
    for node in out["questions"]:
        for name in node["tables"]:
            if not _BASE_TABLE.match(name):
                continue
            used_tables.setdefault(
                name,
                {
                    "id": f"t:{name}",
                    "kind": "table",
                    "label": name,
                    "domain": table_meta.get(name, {}).get("domain", "other"),
                    "rows": table_meta.get(name, {}).get("rows"),
                    "questions": 0,
                    "hits": 0,
                },
            )
            used_tables[name]["questions"] += 1
            used_tables[name]["hits"] += node["hits"]
    out["tables"] = sorted(used_tables.values(), key=lambda t: (t["domain"], t["label"]))

    # ── edges + rollups ───────────────────────────────────────────────────
    edges: list[dict] = []
    abstraction_by_id = {a["id"]: a for a in out["abstractions"]}
    for node in out["questions"]:
        for key in node["concepts"]:
            target = abstraction_by_id.get(f"c:{key}")
            if target:
                edges.append({"source": target["id"], "target": node["id"], "kind": "asks"})
                target["questions"] += 1
                target["hits"] += node["hits"]
        for name in node["tables"]:
            if name in used_tables:
                edges.append({"source": node["id"], "target": f"t:{name}", "kind": "reads"})
    out["edges"] = edges

    # An abstraction that no cached question reaches is still shown (that is the point of the page:
    # what the assistant can reason about, and what it has actually reused), unless asked to hide it.
    if not include_empty:
        out["abstractions"] = [a for a in out["abstractions"] if a["questions"]]

    # ── prompt-cache family: roles -> calls ───────────────────────────────
    roles: dict[str, dict] = {}
    for row in prompt_rows:
        node = roles.setdefault(
            row["scope"] or "model",
            {
                "id": f"r:{row['scope'] or 'model'}",
                "kind": "role",
                "label": row["scope"] or "model",
                "calls": 0,
                "hits": 0,
            },
        )
        node["calls"] += 1
        node["hits"] += row["hits"]
    out["roles"] = sorted(roles.values(), key=lambda r: -r["calls"])
    out["prompt_calls"] = [
        {
            "id": f"p:{r['cache_key']}",
            "kind": "prompt",
            "label": r["label"],
            "role": r["scope"],
            "hits": r["hits"],
            "created_at": r["created_at"],
            "cache_key": r["cache_key"],
            "response_length": r["response_length"],
        }
        for r in prompt_rows
    ]
    for call in out["prompt_calls"]:
        edges.append({"source": f"r:{call['role']}", "target": call["id"], "kind": "made"})
    return out
