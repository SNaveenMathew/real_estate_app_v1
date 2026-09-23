"""HTTP API behind the "Data sources" section of the Data page (``/data``) — download
links plus upload-and-refresh for the built-in datasets. See services/data_sources.py
for the registry and orchestration this wraps.

    GET  /api/data-sources                  registry + last-refreshed info per source
    POST /api/data-sources/{key}/refresh    upload a file (multipart) -> save + reload

Threading note: same as api/onboarding.py — every handler here touches the shared
DuckDB connection, so it runs on the event-loop thread like the rest of the app.
There's no model call here that needs a worker thread.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

import db.duckdb_store as store
from config import settings
from services import data_sources as ds

router = APIRouter(prefix="/api/data-sources", tags=["data-sources"])


@router.get("")
async def list_data_sources():
    store.get_conn()
    return {"sources": ds.list_sources()}


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
