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

Threading note: the application shares one DuckDB connection on the event-loop thread, so every
handler that touches DuckDB runs there (like the rest of the app).  Only pure-HTTP work - the model
call and the self-test - is moved to a worker thread so a slow model never freezes the map or chats.
"""
from __future__ import annotations

import asyncio
import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Body, File, Form, HTTPException, UploadFile

import db.duckdb_store as store
import db.schema_catalog as schema
from config import settings
from services import catalog_llm
from services import dataset_onboarding as ob
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


async def _warm_index() -> None:
    """After a catalog change, refresh the metadata embeddings (best effort; self-heals otherwise)."""
    ob.warm_vector_index()


# ---------------------------------------------------------------------------
# Catalog + datasets
# ---------------------------------------------------------------------------

@router.get("/catalog")
async def get_catalog(dataset_id: str | None = None):
    store.get_conn()
    out = schema.describe_catalog()
    out["draft"] = _call(ob.draft_overlay, dataset_id) if dataset_id else None
    out["house_linked"] = [d["name"] for d in schema.house_linked_datasets()]   # reachable from House Chat
    return out


@router.get("/datasets")
async def get_datasets():
    store.get_conn()
    return {"datasets": ob.list_datasets(), "formats": readers.SUPPORTED_EXTENSIONS,
            "max_upload_mb": int(getattr(settings, "onboarding_max_upload_mb", 250)),
            "domains": [{"key": k, "label": schema.DOMAIN_LABELS[k]} for k in ob.VALID_DOMAINS],
            "roles": list(ob.VALID_ROLES)}


async def _save_upload(file: UploadFile) -> tuple[Path, str]:
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(file.filename or "upload").name)[:120] or "upload"
    ext = Path(name).suffix.lower()
    if ext not in readers.SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"'{ext or 'that'}' files are not supported. "
                                                   f"Use one of: {', '.join(readers.SUPPORTED_EXTENSIONS)}.")
    folder = Path(settings.uploads_dir) / "datasets" / uuid.uuid4().hex[:10]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / name
    limit = int(getattr(settings, "onboarding_max_upload_mb", 250)) * 1024 * 1024
    size = 0
    with open(dest, "wb") as fh:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                fh.close()
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(status_code=413, detail=f"The file is larger than {limit // (1024 * 1024)} MB.")
            fh.write(chunk)
    return dest, name


@router.post("/datasets")
async def upload_dataset(file: UploadFile = File(...), sheet: str | None = Form(None),
                         skiprows: int | None = Form(None)):
    store.get_conn()
    dest, name = await _save_upload(file)
    try:
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
            shutil.rmtree(dest.parent, ignore_errors=True)  # refresh_source already copied what it needs
            if not result.get("ok"):
                raise HTTPException(status_code=422, detail=result.get("error", "Refresh failed."))
            return {"matched_source": matched_key, "refresh_result": result}
        return _call(ob.create_dataset, dest, name, sheet=sheet or None, skiprows=skiprows)
    except HTTPException:
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    store.get_conn()
    return _call(ob.detail, dataset_id)


@router.put("/datasets/{dataset_id}")
async def save_description(dataset_id: str, payload: dict = Body(...)):
    store.get_conn()
    return _call(ob.update_description, dataset_id, payload)


@router.post("/datasets/{dataset_id}/draft")
async def draft_description(dataset_id: str):
    store.get_conn()
    ds = _call(ob.get_draftable, dataset_id)
    result = await asyncio.to_thread(catalog_llm.draft_dataset, ds)      # pure HTTP: no DuckDB in the worker
    return _call(ob.apply_draft, dataset_id, result)


@router.post("/datasets/{dataset_id}/enrich")
async def enrich_dataset(dataset_id: str, payload: dict = Body(...)):
    store.get_conn()
    return _call(ob.enrich, dataset_id, str(payload.get("kind", "")))


@router.post("/datasets/{dataset_id}/analyze")
async def analyze_dataset(dataset_id: str):
    store.get_conn()
    return _call(ob.analyze, dataset_id)


@router.post("/proposals/{proposal_id}/decision")
async def decide(proposal_id: str, background: BackgroundTasks, payload: dict = Body(...)):
    store.get_conn()
    out = _call(ob.decide, proposal_id, str(payload.get("decision", "")), note=str(payload.get("note", "")),
                edits=payload.get("edits") or {})
    if payload.get("decision") == "approve":
        background.add_task(_warm_index)
    return out


@router.post("/relationships/revoke")
async def revoke(background: BackgroundTasks, payload: dict = Body(...)):
    store.get_conn()
    out = _call(ob.revoke_relationship, str(payload.get("rel_key", "")), str(payload.get("reason", "")))
    background.add_task(_warm_index)
    return out


@router.post("/datasets/{dataset_id}/retire")
async def retire(dataset_id: str, background: BackgroundTasks):
    store.get_conn()
    out = _call(ob.retire_dataset, dataset_id)
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
