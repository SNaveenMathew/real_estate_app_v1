"""
FastAPI backend for the Real Estate Analysis App.
Run: uvicorn main:app --reload --port 8000
"""
import json
import shutil
import traceback
import math
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
import db.duckdb_store as store
import db.vector_store as vs
from services import layers as layer_service
from services import bike_routing
from agents.house_agent import run_house_chat
from agents.general_agent import run_general_chat


def _safe(v):
    """Replace NaN / Inf / -Inf with None so json.dumps never raises."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _safe_dict(d: dict) -> dict:
    return {k: _safe(v) for k, v in d.items()}


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Real Estate Analysis", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
settings.uploads_dir.mkdir(parents=True, exist_ok=True)


# ── Pydantic models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []   # [{"role": "user"|"assistant", "content": "..."}]


class DocumentRequest(BaseModel):
    text: str
    doc_type: str = "description"


# ── Cache-busting version stamp ───────────────────────────────────────────────
# Recomputed each time the server starts, so any static file change takes effect
# immediately after restarting uvicorn — no manual version bumping needed.

import hashlib as _hashlib

def _static_version() -> str:
    """MD5 of app.js + style.css contents — changes whenever either file changes."""
    h = _hashlib.md5()
    for name in ("app.js", "style.css"):
        p = Path("static") / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:8]

_STATIC_VER = _static_version()


# ── HTML entry point ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html = Path("static/index.html").read_text(encoding="utf-8")
    # Replace any existing ?v=... or inject fresh version into static asset URLs
    import re
    html = re.sub(r'(/static/[^"\']+\.(?:js|css))(\?v=[^"\']*)?',
                  lambda m: f'{m.group(1)}?v={_STATIC_VER}', html)
    return HTMLResponse(content=html,
                        headers={"Cache-Control": "no-store"})


# ── Map data ─────────────────────────────────────────────────────────────────

@app.get("/api/houses")
async def get_houses():
    """Return all houses as GeoJSON FeatureCollection."""
    houses = store.get_all_houses()
    features = []
    for h in houses:
        h = _safe_dict(h)
        if h.get("lat") is None or h.get("lon") is None:
            continue
        props = {k: v for k, v in h.items() if k not in ("lat", "lon", "raw_json")}
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [h["lon"], h["lat"]]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/house/{house_id}")
async def get_house(house_id: str):
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")
    house = _safe_dict(house)
    docs = vs.get_house_documents(house_id)
    nri = None
    if house.get("tract_fips"):
        nri = store.get_nri_for_tract(house["tract_fips"])
        if nri:
            nri = _safe_dict(nri)
    history = [_safe_dict(h) for h in store.get_house_history(house_id)]
    return {"house": house, "documents": docs, "nri": nri, "history": history}


# ── Chat endpoints ────────────────────────────────────────────────────────────

@app.post("/api/house/{house_id}/chat")
async def house_chat(house_id: str, req: ChatRequest):
    """Chat with the house-specific agent."""
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")

    # Auto-save description if message looks like a property description (>100 chars, no ?)
    auto_saved = False
    if len(req.message) > 100 and "?" not in req.message[:50]:
        vs.upsert_description(house_id, req.message)  # enforces single description
        auto_saved = True

    try:
        reply, updated_history = run_house_chat(house_id, req.message, req.history)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Agent error: {e}")

    # Prepend acknowledgment if we auto-saved
    if auto_saved and "thank" not in reply.lower()[:50]:
        reply = ("Thank you for the property description — it has been saved. " + reply)

    return {"reply": reply, "history": updated_history, "auto_saved": auto_saved}


@app.post("/api/chat")
async def general_chat(req: ChatRequest):
    """General chat — cross-house, MSA, national risk questions."""
    try:
        reply, updated_history, visualization = run_general_chat(req.message, req.history, include_metadata=True)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Agent error: {e}")
    return {"reply": reply, "history": updated_history, "visualization": visualization}


# ── Document / photo upload ──────────────────────────────────────────────────

@app.post("/api/house/{house_id}/favorite")
async def toggle_favorite(house_id: str):
    """Toggle the favorite status of a house."""
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")
    new_val = store.toggle_favorite(house_id)
    return {"house_id": house_id, "is_favorite": new_val}


@app.put("/api/house/{house_id}/description")
async def update_description(house_id: str, req: DocumentRequest):
    """Replace the single stored description for a house."""
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")
    if not req.text.strip():
        raise HTTPException(400, "Text is empty")
    doc_id = vs.upsert_description(house_id, req.text)
    return {"status": "saved", "doc_id": doc_id}

@app.delete("/api/house/{house_id}/description")
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


@app.get("/api/house/{house_id}/description")
async def get_description_endpoint(house_id: str):
    """Get the stored description for a house."""
    desc = vs.get_description(house_id)
    return {"description": desc}


@app.post("/api/house/{house_id}/photo")
async def upload_photo(
    house_id: str,
    file: UploadFile = File(...),
    caption: str = Form(""),
):
    """Upload a photo for a house. Stores reference in vector DB."""
    house = store.get_house(house_id)
    if not house:
        raise HTTPException(404, "House not found")

    # Save file
    house_dir = settings.uploads_dir / house_id
    house_dir.mkdir(parents=True, exist_ok=True)
    dest = house_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Store in vector DB
    doc_id = vs.add_photo(house_id, dest, caption=caption)
    return {"status": "uploaded", "doc_id": doc_id, "path": str(dest)}


@app.get("/api/house/{house_id}/documents")
async def get_documents(house_id: str):
    docs = vs.get_house_documents(house_id)
    return {"documents": docs}


# ── NRI data ─────────────────────────────────────────────────────────────────

@app.get("/api/nri/{tract_fips}")
async def get_nri(tract_fips: str):
    nri = store.get_nri_for_tract(tract_fips)
    if not nri:
        raise HTTPException(404, "NRI data not found for this tract")
    return nri


# ── Map layers (Crime / NRI overlays) ─────────────────────────────────────────
# Both are viewport-scoped: the frontend passes the current Leaflet map
# bounds (map.getBounds()) and re-requests on pan/zoom, so a national-scale
# dataset never has to be shipped whole just to render what's on screen. See
# services/layers.py for the aggregation/filtering logic.

@app.get("/api/layers/crime")
async def get_crime_layer(west: float, south: float, east: float, north: float,
                           grid_deg: float = 0.003, city: Optional[str] = None):
    """Severity-weighted crime heatmap points within the given map bounds,
    aggregated into a coarse grid. `city` optionally restricts to one
    crime_incidents.city key (e.g. 'pittsburgh')."""
    return layer_service.get_crime_heatmap(west, south, east, north, grid_deg=grid_deg, city=city)


@app.get("/api/layers/nri")
async def get_nri_layer(west: float, south: float, east: float, north: float):
    """FEMA National Risk Index census-tract choropleth (GeoJSON) within the
    given map bounds."""
    return layer_service.get_nri_choropleth(west, south, east, north)


@app.get("/api/layers/bike")
async def get_bike_layer(west: float, south: float, east: float, north: float,
                         city: Optional[str] = None,
                         exclusive: bool = False):
    """BikePGH overlay within the current viewport.

    ``exclusive=true`` is used by visualizations to resolve known overlapping
    BikePGH classifications into one canonical display category without
    altering the underlying bike_routes data.
    """
    return layer_service.get_bike_routes(
        west, south, east, north, city=city, exclusive=exclusive
    )


class BikeRouteRequest(BaseModel):
    start: str
    end: str
    city: str = "Pittsburgh, PA"


@app.post("/api/bike/route")
async def get_bike_route(req: BikeRouteRequest):
    """Find a bicycle route between place names or addresses."""
    try:
        return await bike_routing.route_bike(req.start, req.end, city=req.city)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Bike routing service error: {e}")



# ── Stats / health ────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def stats():
    houses = store.query("SELECT COUNT(*) as n FROM houses").iloc[0]["n"]
    tracts = store.query("SELECT COUNT(*) as n FROM nri_tracts").iloc[0]["n"]
    sold   = store.query("SELECT COUNT(*) as n FROM sold_homes").iloc[0]["n"]
    crime  = store.query("SELECT COUNT(*) as n FROM crime_incidents").iloc[0]["n"]
    docs   = vs.collection_count()
    return {
        "houses": int(houses),
        "nri_tracts": int(tracts),
        "sold_homes": int(sold),
        "crime_incidents": int(crime),
        "vector_documents": docs,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.app_host,
                port=settings.app_port, reload=True)
