"""HTTP API endpoints for public transit routing."""
from __future__ import annotations

import traceback
from fastapi import APIRouter, HTTPException

from services import transit_routing
from api.common import TransitRouteRequest

router = APIRouter(tags=["transit"])


@router.post("/api/transit/route")
async def get_transit_route(req: TransitRouteRequest):
    """Find a public transit route between place names or addresses."""
    try:
        return await transit_routing.route_transit(req.start, req.end, city=req.city)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Transit routing service error: {e}")
