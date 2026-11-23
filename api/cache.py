"""HTTP API behind the Cache page (``/cache``).

    GET  /api/cache                 counters: prompt / semantic / SQL entries, hits, fingerprint
    GET  /api/cache/graph           the reuse view: abstractions -> cached questions -> tables
                                    (the /ontology page draws db/ontology.py instead)
    GET  /api/cache/entries         raw rows for the inspector (``kind``, ``q``, ``limit``)
    POST /api/cache/clear           forget everything, or one kind (``{"kind": "prompt|semantic|sql"}``)

Everything here is read-only apart from ``/clear``, which only deletes cache rows - no data table is
ever touched.  Threading note: the handlers are plain ``def``, so FastAPI runs them in its worker
pool and each gets its own DuckDB cursor (db/duckdb_store.py); the queries are small (indexed
primary-key reads and bounded scans over the cache tables only), so the Cache page never waits
behind the map.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query

import db.duckdb_store as store
from services import cache as result_cache

router = APIRouter(prefix="/api/cache", tags=["cache"])


@router.get("")
def cache_stats():
    store.get_conn()
    return result_cache.stats()


@router.get("/graph")
def cache_graph(include_empty: bool = Query(default=True)):
    """``include_empty=false`` hides catalog concepts that no cached question has reached yet."""
    store.get_conn()
    return result_cache.graph(include_empty=include_empty)


@router.get("/entries")
def cache_entries(
    kind: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    q: str = Query(default=""),
):
    store.get_conn()
    return {"entries": result_cache.entries(kind=kind, limit=limit, query=q)}


@router.get("/entry/{cache_key}")
def cache_entry(cache_key: str):
    """One stored entry in full (the page's inspector).  The answer is data, surfaced as text only."""
    store.get_conn()
    entry = result_cache.get_entry(cache_key)
    if entry is None:
        raise HTTPException(status_code=404, detail="No such cache entry (it may have expired).")
    return entry


@router.post("/clear")
def cache_clear(payload: Optional[dict] = Body(default=None)):
    store.get_conn()
    kind = str((payload or {}).get("kind") or "") or None
    return result_cache.invalidate(kind)
