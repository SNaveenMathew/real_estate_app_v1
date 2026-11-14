"""HTTP API endpoints for system stats and client configuration."""
from __future__ import annotations

from fastapi import APIRouter

from config import settings
import db.duckdb_store as store
import db.vector_store as vs

router = APIRouter(tags=["system"])


@router.get("/api/stats")
async def stats():
    houses = store.query("SELECT COUNT(*) as n FROM houses").iloc[0]["n"]
    tracts = store.query("SELECT COUNT(*) as n FROM nri_tracts").iloc[0]["n"]
    sold = store.query("SELECT COUNT(*) as n FROM sold_homes").iloc[0]["n"]
    crime = store.query("SELECT COUNT(*) as n FROM crime_incidents").iloc[0]["n"]
    docs = vs.collection_count()
    return {
        "houses": int(houses),
        "nri_tracts": int(tracts),
        "sold_homes": int(sold),
        "crime_incidents": int(crime),
        "vector_documents": docs,
    }


@router.get("/api/config")
async def get_client_config():
    """Return runtime client configuration (e.g. Map tiles API key)."""
    return {
        "cartoApiKey": settings.carto_api_key,
    }

