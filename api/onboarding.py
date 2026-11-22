"""HTTP API behind the Data page (``/data``).

    GET  /api/onboarding/catalog[?dataset_id=]     the unified catalog (+ a draft overlay for a dataset under review)
    GET  /api/onboarding/datasets                  datasets added so far + supported formats
    POST /api/onboarding/datasets                  upload a file (multipart) -> staged, profiled draft
    GET  /api/onboarding/datasets/{id}             one dataset with columns, preview, proposals
    PUT  /api/onboarding/datasets/{id}             save the human's description
    POST /api/onboarding/datasets/{id}/draft       optional: draft descriptions with the local model
    POST /api/onboarding/datasets/{id}/enrich      opt-in: tract from lat/lon, or geocode addresses
    POST /api/onboarding/datasets/{id}/analyze     propose the table + relationships, with evidence
    POST /api/onboarding/proposals/{id}/decision   approve / reject (approval applies immediately)
    POST /api/onboarding/relationships/revoke      undo an approved relationship
    POST /api/onboarding/datasets/{id}/retire      retire a published dataset / discard a draft
    GET  /api/onboarding/llm/status                the catalog model router
    POST /api/onboarding/llm/selftest              score the configured model

Threading: DuckDB access needs no event loop - db/duckdb_store.py hands every thread its
own cursor on the shared database, so a plain ``def`` handler simply runs in FastAPI's
worker pool. The short catalog reads and writes below are exactly that.

The ingest path is different: saving a 250 MB upload, profiling it, proposing relationships
across the whole catalog and enriching thousands of rows take seconds to minutes, and doing
that on the event loop froze every map pan and chat turn behind it. Those blocks therefore
stay in ``async`` handlers and are handed to a worker thread with ``asyncio.to_thread``, so
they occupy neither the loop nor one of the request threads the UI needs. The same rule
applies to the one pure-HTTP model call (``catalog_llm``), which may not touch DuckDB inside
the worker.

Uploads are treated as hostile input: the name is sanitized, the size is capped while it
streams, and no file survives a rejected request (api/uploads.py).
"""
from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Body, File, Form, HTTPException, UploadFile

import db.duckdb_store as store
import db.schema_catalog as schema
from api.uploads import sanitize_filename, save_stream
from config import settings
from services import catalog_llm
from services import dataset_onboarding as ob
from services import jobs
from services import dataset_readers as readers
from services import data_sources

router = APIRouter(prefix="/api/onboarding", tags=["data-onboarding"])


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ob.OnboardingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except readers.DatasetReadError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _warm_index() -> None:
    """After a catalog change, refresh the metadata embeddings (best effort; self-heals otherwise).

    A plain function on purpose: Starlette runs a sync background task in a worker thread,
    while an ``async`` one would embed on the event loop after the response is already sent.
    """
    ob.warm_vector_index()


def _max_upload_bytes() -> int:
    return int(getattr(settings, "onboarding_max_upload_mb", 250)) * 1024 * 1024


def _start_job(kind: str, label: str, fn):
    """Claim a job slot, or answer 409 with the job that already holds it."""
    try:
        job = jobs.start(kind, label, fn)
    except jobs.JobBusy as busy:
        raise HTTPException(status_code=409, detail=str(busy), headers={"Retry-After": "5"})
    return job.snapshot()


def _job_call(job, message: str, fn, *args, **kwargs):
    job.progress(message)
    return _call(fn, *args, **kwargs)


# ---------------------------------------------------------------------------
# Catalog + datasets
# ---------------------------------------------------------------------------

@router.get("/catalog")
def get_catalog(dataset_id: str | None = None):
    store.get_conn()
    out = schema.describe_catalog()
    out["draft"] = _call(ob.draft_overlay, dataset_id) if dataset_id else None
    out["house_linked"] = [d["name"] for d in schema.house_linked_datasets()]   # reachable from House Chat
    return out


@router.get("/datasets")
def get_datasets():
    store.get_conn()
    return {"datasets": ob.list_datasets(), "formats": readers.SUPPORTED_EXTENSIONS,
            "max_upload_mb": int(getattr(settings, "onboarding_max_upload_mb", 250)),
            "domains": [{"key": k, "label": schema.DOMAIN_LABELS[k]} for k in ob.VALID_DOMAINS],
            "roles": list(ob.VALID_ROLES)}


def _save_upload(file: UploadFile) -> tuple[Path, str]:
    """Sanitize the name, check the reader supports the extension, stream under the cap."""
    name = sanitize_filename(file.filename, fallback="upload")
    ext = Path(name).suffix.lower()
    if ext not in readers.SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"'{ext or 'that'}' files are not supported. "
                                                   f"Use one of: {', '.join(readers.SUPPORTED_EXTENSIONS)}.")
    dest = Path(settings.uploads_dir) / "datasets" / uuid.uuid4().hex[:10] / name
    save_stream(file.file, dest, max_bytes=_max_upload_bytes())
    return dest, name


def _ingest_upload(dest: Path, name: str, sheet: str | None, skiprows: int | None):
    """Turn a staged file into a draft dataset - or into a refresh of a built-in source."""
    # Before treating this as a brand-new dataset, check whether its columns
    # match one of the built-in sources (see services/data_sources.py) — if so,
    # this is a refresh of an existing table, not a new one: no new dataset
    # draft, no new proposals/relationships, just upsert into what's already
    # there (same path the dedicated "Data sources" panel uses).
    matched_key = None
    for cols in data_sources.peek_header_candidates(dest, dest.suffix.lower()):
        matched_key = data_sources.detect_source_for_columns(cols)
        if matched_key:
            break
    if matched_key:
        result = data_sources.refresh_source(matched_key, dest, name)
        shutil.rmtree(dest.parent, ignore_errors=True)      # refresh_source already copied what it needs
        if not result.get("ok"):
            raise HTTPException(status_code=422, detail=result.get("error", "Refresh failed."))
        return {"matched_source": matched_key, "refresh_result": result}
    return _call(ob.create_dataset, dest, name, sheet=sheet or None, skiprows=skiprows)


def _ingest_in_background(job, dest: Path, name: str, sheet, skiprows):
    """The minutes-long half of an upload, on the job thread."""
    job.progress(f"Reading and profiling {name}...")
    try:
        return _ingest_upload(dest, name, sheet, skiprows)
    except BaseException:
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise


@router.post("/datasets", status_code=202)
async def upload_dataset(file: UploadFile = File(...), sheet: str | None = Form(None),
                         skiprows: int | None = Form(None)):
    """Stage the upload, then ingest it in a background job.

    The bytes are streamed to disk here - capped and sanitized (api/uploads.py) - because
    the request body dies with the response. Everything that takes minutes (profiling,
    relationship discovery, the DDL, the catalog reload) runs in the job, so the page polls
    /api/jobs/{id} instead of holding one request open and hoping the tab stays on it.
    """
    store.get_conn()
    dest, name = await asyncio.to_thread(_save_upload, file)
    try:
        return _start_job("ingest", f"Ingesting {name}",
                          lambda job: _ingest_in_background(job, dest, name, sheet, skiprows))
    except BaseException:
        shutil.rmtree(dest.parent, ignore_errors=True)     # nothing staged survives a refused upload
        raise


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    store.get_conn()
    return _call(ob.detail, dataset_id)


@router.put("/datasets/{dataset_id}")
def save_description(dataset_id: str, payload: dict = Body(...)):
    store.get_conn()
    return _call(ob.update_description, dataset_id, payload)


@router.post("/datasets/{dataset_id}/draft")
async def draft_description(dataset_id: str):
    store.get_conn()
    ds = _call(ob.get_draftable, dataset_id)
    result = await asyncio.to_thread(catalog_llm.draft_dataset, ds)      # pure HTTP: no DuckDB in the worker
    return _call(ob.apply_draft, dataset_id, result)


@router.post("/datasets/{dataset_id}/enrich", status_code=202)
def enrich_dataset(dataset_id: str, payload: dict = Body(...)):
    """Enrichment geocodes and joins thousands of rows: 202, then poll /api/jobs/{id}."""
    store.get_conn()
    kind = str(payload.get("kind", ""))
    return _start_job("enrich", f"Enriching {dataset_id}",
                      lambda job: _job_call(job, f"Enriching by {kind or 'rules'}...",
                                            ob.enrich, dataset_id, kind))


@router.post("/datasets/{dataset_id}/analyze", status_code=202)
def analyze_dataset(dataset_id: str):
    """Key-overlap probes across every catalog table: 202, then poll /api/jobs/{id}."""
    store.get_conn()
    return _start_job("analyze", f"Testing {dataset_id}",
                      lambda job: _job_call(job, "Testing columns against the catalog...",
                                            ob.analyze, dataset_id))


@router.post("/proposals/{proposal_id}/decision")
async def decide(proposal_id: str, background: BackgroundTasks, payload: dict = Body(...)):
    store.get_conn()
    out = await asyncio.to_thread(
        _call, ob.decide, proposal_id, str(payload.get("decision", "")),
        note=str(payload.get("note", "")), edits=payload.get("edits") or {},
    )
    if payload.get("decision") == "approve":
        background.add_task(_warm_index)
    return out


@router.post("/relationships/revoke")
async def revoke(background: BackgroundTasks, payload: dict = Body(...)):
    store.get_conn()
    out = await asyncio.to_thread(
        _call, ob.revoke_relationship, str(payload.get("rel_key", "")), str(payload.get("reason", "")),
    )
    background.add_task(_warm_index)
    return out


@router.post("/datasets/{dataset_id}/retire")
async def retire(dataset_id: str, background: BackgroundTasks):
    store.get_conn()
    out = await asyncio.to_thread(_call, ob.retire_dataset, dataset_id)
    background.add_task(_warm_index)
    return out


# ---------------------------------------------------------------------------
# Catalog model router
# ---------------------------------------------------------------------------

@router.get("/llm/status")
async def llm_status():
    return await asyncio.to_thread(catalog_llm.router().status)


@router.post("/llm/selftest")
async def llm_selftest():
    return await asyncio.to_thread(catalog_llm.selftest)
