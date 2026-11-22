"""HTTP API endpoints for houses, property documents, favorites, photos, and NRI risk.

Threading: these handlers are plain ``def``. Every one of them reads or writes DuckDB or
Chroma and nothing on the way waits, so FastAPI runs them in its worker pool and the
store's per-thread cursors (db/duckdb_store.py) serve each thread its own handle. As
``async def`` they held the event loop for the whole query - a 1.4 MB ``/api/houses``
payload froze every other request for the duration.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.common import safe_dict, DocumentRequest
from api.uploads import sanitize_filename, save_stream, sniff_image_extension
from config import settings
import db.duckdb_store as store
import db.vector_store as vs

router = APIRouter(tags=["houses"])


@router.get("/api/houses")
def get_houses():
    """Return all houses as GeoJSON FeatureCollection."""
    houses = store.get_all_houses()
    features = []
    for h in houses:
        h = safe_dict(h)
        if h.get("lat") is None or h.get("lon") is None:
            continue
        props = {k: v for k, v in h.items() if k not in ("lat", "lon", "raw_json")}
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [h["lon"], h["lat"]]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


@router.get("/api/house/{house_id}")
def get_house(house_id: str):
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")
    house = safe_dict(house)
    docs = vs.get_house_documents(house_id)
    nri = None
    if house.get("tract_fips"):
        nri = store.get_nri_for_tract(house["tract_fips"])
        if nri:
            nri = safe_dict(nri)
    history = [safe_dict(h) for h in store.get_house_history(house_id)]
    return {"house": house, "documents": docs, "nri": nri, "history": history}


@router.post("/api/house/{house_id}/favorite")
def toggle_favorite(house_id: str):
    """Toggle the favorite status of a house."""
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")
    new_val = store.toggle_favorite(house_id)
    return {"house_id": house_id, "is_favorite": new_val}


@router.put("/api/house/{house_id}/description")
def update_description(house_id: str, req: DocumentRequest):
    """Replace the single stored description for a house."""
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")
    if not req.text.strip():
        raise HTTPException(400, "Text is empty")
    doc_id = vs.upsert_description(house_id, req.text)
    return {"status": "saved", "doc_id": doc_id}


@router.delete("/api/house/{house_id}/description")
def delete_description(house_id: str):
    """Delete the stored description for a house."""
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")
    desc = vs.get_description(house_id)
    if not desc:
        return {"status": "not_found"}
    vs.delete_document(desc["id"])
    return {"status": "deleted"}


@router.get("/api/house/{house_id}/description")
def get_description_endpoint(house_id: str):
    """Get the stored description for a house."""
    desc = vs.get_description(house_id)
    return {"description": desc}


@router.post("/api/house/{house_id}/photo")
def upload_photo(
    house_id: str,
    file: UploadFile = File(...),
    caption: str = Form(""),
):
    """Upload a photo for a house. Stores the file and a reference in the vector DB.

    The client supplies neither the path nor the type: ``house_id`` is sanitized because
    it becomes a directory name, the file is streamed under ``photo_max_upload_mb``, and
    the stored extension comes from the bytes Pillow decoded - never from the filename or
    Content-Type.
    """
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")

    limit = int(getattr(settings, "photo_max_upload_mb", 10)) * 1024 * 1024
    house_dir = Path(settings.uploads_dir) / sanitize_filename(house_id, fallback="house")
    house_dir.mkdir(parents=True, exist_ok=True)

    staged = house_dir / f".{uuid.uuid4().hex}.part"        # never visible under a serving name while it is checked
    save_stream(file.file, staged, max_bytes=limit)
    try:
        ext = sniff_image_extension(staged)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise

    stem = Path(sanitize_filename(file.filename, fallback="photo")).stem[:60]
    dest = house_dir / f"{stem}-{uuid.uuid4().hex[:8]}{ext}"
    staged.replace(dest)

    doc_id = vs.add_photo(house_id, dest, caption=caption[:500])
    return {"status": "uploaded", "doc_id": doc_id, "path": str(dest)}


@router.get("/api/house/{house_id}/documents")
def get_documents(house_id: str):
    docs = vs.get_house_documents(house_id)
    return {"documents": docs}


@router.get("/api/nri/{tract_fips}")
def get_nri(tract_fips: str):
    nri = store.get_nri_for_tract(tract_fips)
    if not nri:
        raise HTTPException(404, "NRI data not found for this tract")
    return nri
