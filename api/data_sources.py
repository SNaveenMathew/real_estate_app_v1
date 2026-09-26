"""HTTP API behind the "Data sources" section of the Data page (``/data``) — download
links plus upload-and-refresh for the built-in datasets. See services/data_sources.py
for the registry and orchestration this wraps.

    GET  /api/data-sources                  registry + last-refreshed info per source
    POST /api/data-sources/{key}/refresh    upload a file (multipart) -> save + reload
    POST /api/data-sources/health           check each source's public link is reachable

Threading note: same as api/onboarding.py — every handler here touches the shared
DuckDB connection, so it runs on the event-loop thread like the rest of the app.
There's no model call here that needs a worker thread. The health-check handler is
the one exception worth calling out: it makes up to 15 outbound HTTP requests, but
they run concurrently via asyncio.gather (see services/data_sources.py), so this
still only blocks the event loop for the CPU-bound moments between awaits, not for
however long the slowest third-party site takes to respond.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

import db.duckdb_store as store
from config import settings
from services import data_sources as ds

router = APIRouter(prefix="/api/data-sources", tags=["data-sources"])


@router.get("")
async def list_data_sources():
    store.get_conn()
    return {"sources": ds.list_sources()}


@router.post("/health")
async def check_data_source_health(keys: Optional[list[str]] = Body(default=None, embed=True)):
    """Reachability check for each source's public link (or just the ones in `keys`,
    if given). Triggered explicitly from the Data page — see services/data_sources.py:
    check_source_links for what this can and can't guarantee about an upstream site
    staying at the same URL."""
    return {"results": await ds.check_source_links(keys)}


@router.post("/{key}/refresh")
async def refresh_data_source(key: str, file: UploadFile = File(...)):
    store.get_conn()
    source = ds.get_source(key)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Unknown data source: {key}")

    name = Path(file.filename or "upload").name
    folder = Path(settings.uploads_dir) / "data-source-refresh" / uuid.uuid4().hex[:10]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / name
    limit = int(getattr(settings, "onboarding_max_upload_mb", 250)) * 1024 * 1024
    size = 0
    try:
        with open(dest, "wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413,
                                         detail=f"The file is larger than {limit // (1024 * 1024)} MB.")
                fh.write(chunk)
        result = ds.refresh_source(key, dest, name)
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("error", "Refresh failed."))
    return result
