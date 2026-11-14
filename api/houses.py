"""HTTP API endpoints for houses, property documents, favorites, photos, and NRI risk."""
from __future__ import annotations

import shutil
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import settings
import db.duckdb_store as store
import db.vector_store as vs
from api.common import safe_dict, DocumentRequest

router = APIRouter(tags=["houses"])


@router.get("/api/houses")
async def get_houses():
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
async def get_house(house_id: str):
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
async def toggle_favorite(house_id: str):
    """Toggle the favorite status of a house."""
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")
    new_val = store.toggle_favorite(house_id)
    return {"house_id": house_id, "is_favorite": new_val}


@router.put("/api/house/{house_id}/description")
async def update_description(house_id: str, req: DocumentRequest):
    """Replace the single stored description for a house."""
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")
    if not req.text.strip():
        raise HTTPException(400, "Text is empty")
    doc_id = vs.upsert_description(house_id, req.text)
    return {"status": "saved", "doc_id": doc_id}


@router.delete("/api/house/{house_id}/description")
async def delete_description(house_id: str):
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
async def get_description_endpoint(house_id: str):
    """Get the stored description for a house."""
    desc = vs.get_description(house_id)
    return {"description": desc}


@router.post("/api/house/{house_id}/photo")
async def upload_photo(
    house_id: str,
    file: UploadFile = File(...),
    caption: str = Form(""),
):
    """Upload a photo for a house. Stores reference in vector DB."""
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")

    house_dir = settings.uploads_dir / house_id
    house_dir.mkdir(parents=True, exist_ok=True)
    dest = house_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    doc_id = vs.add_photo(house_id, dest, caption=caption)
    return {"status": "uploaded", "doc_id": doc_id, "path": str(dest)}


@router.get("/api/house/{house_id}/documents")
async def get_documents(house_id: str):
    docs = vs.get_house_documents(house_id)
    return {"documents": docs}


@router.get("/api/nri/{tract_fips}")
async def get_nri(tract_fips: str):
    nri = store.get_nri_for_tract(tract_fips)
    if not nri:
        raise HTTPException(404, "NRI data not found for this tract")
    return nri

