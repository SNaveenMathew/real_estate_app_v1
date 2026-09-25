"""HTTP API behind the map's layer panel.

    GET /api/layers                 every layer the live catalog currently supports, with its kind
                                    (points/heat/lines/polygons/choropleth/decoration), which toggle
                                    group it belongs to, and whether it's ready to draw right now
    GET /api/layers/{name}          that layer's data for the given viewport (and, for choropleth/
                                    polygons, an optional ?measure=; for heat, an optional ?weight=)

There is one route per KIND of request, not one per table: a table becomes visitable here the moment
``services/map_layers.py`` classifies it, which happens automatically for anything approved on the
Data page. ``houses`` and ``commute`` are listed by ``GET /api/layers`` (so the frontend has one place
to build its toggle panel from) but have no data route here — they are already served by
``/api/houses`` and ``/api/commute/*`` respectively, with the interactive behavior (selection,
favoriting, chat handoff) that only makes sense built into the map page itself.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

import db.duckdb_store as store
from services import map_layers as ml

router = APIRouter(prefix="/api/layers", tags=["map-layers"])

_NO_DATA_HERE = {"houses", "commute"}


@router.get("")
async def list_layers():
    store.get_conn()
    return ml.describe_layers()


@router.get("/{name}")
async def get_layer(
    name: str,
    west: float = Query(..., description="Viewport west longitude"),
    south: float = Query(..., description="Viewport south latitude"),
    east: float = Query(..., description="Viewport east longitude"),
    north: float = Query(..., description="Viewport north latitude"),
    measure: str | None = Query(None, description="Choropleth/polygon column to color by"),
    weight: str | None = Query(None, description="Heat-layer column to weight by (omit for a uniform count)"),
    city: str | None = Query(None, description="Restrict a heat layer to one city, if the table has one"),
    grid_deg: float | None = Query(None, gt=0, description="Heat-grid cell size in degrees"),
    limit: int | None = Query(None, gt=0, le=20000, description="Row/feature cap for this request"),
):
    if name in _NO_DATA_HERE:
        raise HTTPException(status_code=404, detail=f"'{name}' is drawn by the map page directly, not served here.")
    store.get_conn()
    kwargs = {"west": west, "south": south, "east": east, "north": north}
    if measure is not None:
        kwargs["measure"] = measure
    if weight is not None:
        kwargs["weight"] = weight
    if city is not None:
        kwargs["city"] = city
    if grid_deg is not None:
        kwargs["grid_deg"] = grid_deg
    if limit is not None:
        kwargs["limit"] = limit
    try:
        return ml.get_layer_data(name, **kwargs)
    except ml.LayerError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
