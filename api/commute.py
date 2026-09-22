"""HTTP API behind the Commute tab.

    GET    /api/commute/config          work location, modes, providers, counts, job state
    PUT    /api/commute/work            {"address": "..."} or {"lat": .., "lon": .., "label": ".."} -> save + start computing
    DELETE /api/commute/work            forget the work location and its commute rows
    POST   /api/commute/refresh         {"scope": "missing" | "all"} -> start computing in the background
    GET    /api/commute/status          progress of the running/last job
    GET    /api/commute/house/{id}      one house's estimates
    GET    /api/commute/summary         every fresh estimate (drives the map layer)

Threading: like the rest of the app, DuckDB is used on the event-loop thread; only the pure-HTTP address
lookup runs in a worker thread, and the compute job does its routing calls in worker threads too.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Body, HTTPException

import db.duckdb_store as store
from services import commute

router = APIRouter(prefix="/api/commute", tags=["commute"])


@router.get("/config")
async def get_config():
    store.get_conn()
    return commute.status()


@router.put("/work")
async def put_work(payload: dict = Body(...)):
    store.get_conn()
    address = str(payload.get("address") or "").strip()
    try:
        if address:
            place = await asyncio.to_thread(commute.geocode_work_address, address)     # HTTP only; no DuckDB in the worker
        elif payload.get("lat") is not None and payload.get("lon") is not None:
            lat, lon = float(payload["lat"]), float(payload["lon"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise commute.RoutingError("Latitude must be between -90 and 90 and longitude between -180 and 180.")
            place = commute.Place(lat, lon, str(payload.get("label") or f"{lat:.5f}, {lon:.5f}")[:200], "map")
        else:
            raise commute.RoutingError("Enter an address, or pick a point on the map.")
    except commute.RoutingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Latitude and longitude must be numbers.")
    commute.set_work(place)
    job = commute.start_refresh("missing")        # every row is stale for a new location, so this computes all of them
    return {"work": place.as_dict(), "job": job}


@router.delete("/work")
async def delete_work():
    store.get_conn()
    commute.clear_work()
    return commute.status()


@router.post("/refresh")
async def post_refresh(payload: dict = Body(default={})):
    store.get_conn()
    scope = str(payload.get("scope") or "missing")
    if scope not in ("missing", "all"):
        raise HTTPException(status_code=422, detail="scope must be 'missing' or 'all'.")
    if not commute.get_work():
        raise HTTPException(status_code=409, detail="Set a work location first.")
    return {"job": commute.start_refresh(scope)}


@router.get("/status")
async def get_status():
    store.get_conn()
    s = commute.status()
    return {"job": s["job"], "counts": s["counts"]}


@router.get("/house/{house_id}")
async def get_house_commute(house_id: str):
    store.get_conn()
    place = commute.get_work()
    return {"work": place.as_dict() if place else None, "commute": commute.commute_for_house(house_id) if place else None}


@router.get("/summary")
async def get_summary():
    store.get_conn()
    return commute.summary()
