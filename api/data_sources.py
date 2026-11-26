"""HTTP API behind the "Data sources" section of the Data page (``/data``) — download
links plus upload-and-refresh for the built-in datasets. See services/data_sources.py
for the registry and orchestration this wraps.

    GET  /api/data-sources                  registry + last-refreshed info per source
    POST /api/data-sources/{key}/refresh    upload a file (multipart) -> save + reload
    POST /api/data-sources/health           check each source's public link is reachable

Threading: the listing is a plain ``def`` handler and runs in FastAPI's worker pool
(DuckDB hands each thread its own cursor, see db/duckdb_store.py). The refresh handler
must stay ``async`` for the upload itself, so its two blocking blocks - writing the file
and reloading the table - are handed to a worker thread with ``asyncio.to_thread``; a
Redfin export takes long enough that running it on the loop froze every other request.
The health check makes one outbound HTTP request per source, but they run concurrently via
``asyncio.gather`` (see services/data_sources.py), so it only blocks between awaits.

Uploads are treated as hostile input: the name is sanitized, the size is capped while it
streams, and no file survives a rejected request (api/uploads.py).
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

import db.duckdb_store as store
from api.uploads import sanitize_filename, save_stream
from config import settings
from services import data_sources as ds
from services import jobs

router = APIRouter(prefix="/api/data-sources", tags=["data-sources"])


@router.get("")
def list_data_sources():
    store.get_conn()
    return {"sources": ds.list_sources()}


@router.post("/health")
async def check_data_source_health(keys: Optional[list[str]] = Body(default=None, embed=True)):
    """Reachability check for each source's public link (or just the ones in `keys`,
    if given). Triggered explicitly from the Data page — see services/data_sources.py:
    check_source_links for what this can and can't guarantee about an upstream site
    staying at the same URL."""
    return {"results": await ds.check_source_links(keys)}


def _save_refresh_upload(file: UploadFile, folder: Path) -> tuple[Path, str]:
    name = sanitize_filename(file.filename, fallback="upload")
    dest = folder / name
    save_stream(file.file, dest, max_bytes=int(getattr(settings, "onboarding_max_upload_mb", 600)) * 1024 * 1024)
    return dest, name


def _refresh_in_background(job, key: str, dest: Path, name: str, folder: Path):
    """Reload one source on the job thread; the staged folder always goes away."""
    try:
        job.progress(f"Reloading {key} from {name}...")
        result = ds.refresh_source(key, dest, name)
        if not result.get("ok"):
            raise HTTPException(status_code=422, detail=result.get("error", "Refresh failed."))
        return result
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@router.post("/{key}/refresh", status_code=202)
async def refresh_data_source(key: str, file: UploadFile = File(...)):
    """Stage the upload, reload the source in a background job (202 + /api/jobs/{id}).

    The streamed save is capped and sanitized; the loader - which parses a county export or
    a zipped shapefile and rewrites a table - is what used to hold a request open for
    minutes.
    """
    store.get_conn()
    source = ds.get_source(key)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Unknown data source: {key}")

    folder = Path(settings.uploads_dir) / "data-source-refresh" / uuid.uuid4().hex[:10]
    folder.mkdir(parents=True, exist_ok=True)
    try:
        dest, name = await asyncio.to_thread(_save_refresh_upload, file, folder)
        allowed = {e.strip() for e in source.accept.split(",")}
        ext = Path(name).suffix.lower()
        if ext not in allowed:  # cheap, client-input validation: answer it now, not from the job
            raise HTTPException(
                status_code=422, detail=f"Expected one of {sorted(allowed)} for this source, got '{ext}'."
            )
        try:
            job = jobs.start(
                "refresh", f"Reloading {key}", lambda j: _refresh_in_background(j, key, dest, name, folder)
            )
        except jobs.JobBusy as busy:
            raise HTTPException(status_code=409, detail=str(busy), headers={"Retry-After": "5"})
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return job.snapshot()
