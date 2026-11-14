"""HTTP API endpoints for bicycle routing."""
from __future__ import annotations

import traceback
from fastapi import APIRouter, HTTPException

from services import bike_routing
from api.common import BikeRouteRequest

router = APIRouter(tags=["bike"])


@router.post("/api/bike/route")
async def get_bike_route(req: BikeRouteRequest):
    """Find a bicycle route between place names or addresses."""
    try:
        return await bike_routing.route_bike(req.start, req.end, city=req.city)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Bike routing service error: {e}")

