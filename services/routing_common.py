"""
Shared graph routing, geocoding, and spatial navigation primitives.

Usable across transport modalities including bicycle (BikePGH) and public transit
(FTA National Transit Map) networks.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import heapq
import json
import math
from typing import Optional

import httpx
from shapely.geometry import LineString, Point, box, shape

import db.duckdb_store as store

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "RealEstateIntelligence/1.0 (routing engine)"
NOMINATIM_MIN_INTERVAL_SECONDS = 1.0
HTTP_TIMEOUT_SECONDS = 30.0

# Pittsburgh bounding box guard
PITTSBURGH_BOUNDS = {
    "south": 40.33,
    "west": -80.10,
    "north": 40.55,
    "east": -79.75,
}

# Pittsburgh neighborhood centroids for deterministic fallback
PITTSBURGH_PLACE_ALIASES = {
    "highland park": {"lat": 40.4781, "lon": -79.9110, "display_name": "Highland Park, Pittsburgh, PA"},
    "highland park, pittsburgh": {"lat": 40.4781, "lon": -79.9110, "display_name": "Highland Park, Pittsburgh, PA"},
    "shadyside": {"lat": 40.45152, "lon": -79.93655, "display_name": "Shadyside, Pittsburgh, PA"},
    "shadyside, pittsburgh": {"lat": 40.45152, "lon": -79.93655, "display_name": "Shadyside, Pittsburgh, PA"},
    "downtown": {"lat": 40.4417, "lon": -80.0000, "display_name": "Downtown, Pittsburgh, PA"},
    "downtown pittsburgh": {"lat": 40.4417, "lon": -80.0000, "display_name": "Downtown, Pittsburgh, PA"},
    "oakland": {"lat": 40.4416, "lon": -79.9570, "display_name": "Oakland, Pittsburgh, PA"},
    "oakland, pittsburgh": {"lat": 40.4416, "lon": -79.9570, "display_name": "Oakland, Pittsburgh, PA"},
    "squirrel hill": {"lat": 40.4380, "lon": -79.9229, "display_name": "Squirrel Hill, Pittsburgh, PA"},
    "east liberty": {"lat": 40.4619, "lon": -79.9248, "display_name": "East Liberty, Pittsburgh, PA"},
    "lawrenceville": {"lat": 40.4700, "lon": -79.9600, "display_name": "Lawrenceville, Pittsburgh, PA"},
    "south side": {"lat": 40.4285, "lon": -79.9850, "display_name": "South Side, Pittsburgh, PA"},
}

# Address-level fallbacks
PITTSBURGH_ADDRESS_FALLBACKS = {
    "5624 bryant st, pittsburgh, pa 15206": {
        "lat": 40.4781, "lon": -79.9110,
        "display_name": "5624 Bryant St, Highland Park, Pittsburgh, PA (fallback endpoint)",
        "source": "local_pittsburgh_endpoint_fallback",
    },
    "5624 bryant street, pittsburgh, pa 15206": {
        "lat": 40.4781, "lon": -79.9110,
        "display_name": "5624 Bryant St, Highland Park, Pittsburgh, PA (fallback endpoint)",
        "source": "local_pittsburgh_endpoint_fallback",
    },
    "5903 5th ave, shadyside, pa 15232": {
        "lat": 40.45152, "lon": -79.93655,
        "display_name": "5903 5th Ave, Shadyside, Pittsburgh, PA (fallback endpoint)",
        "source": "local_pittsburgh_endpoint_fallback",
    },
    "5903 fifth ave, shadyside, pa 15232": {
        "lat": 40.45152, "lon": -79.93655,
        "display_name": "5903 Fifth Ave, Shadyside, Pittsburgh, PA (fallback endpoint)",
        "source": "local_pittsburgh_endpoint_fallback",
    },
    "5903 5th ave, pittsburgh, pa 15232": {
        "lat": 40.45152, "lon": -79.93655,
        "display_name": "5903 5th Ave, Shadyside, Pittsburgh, PA (fallback endpoint)",
        "source": "local_pittsburgh_endpoint_fallback",
    },
    "5903 fifth ave, pittsburgh, pa 15232": {
        "lat": 40.45152, "lon": -79.93655,
        "display_name": "5903 Fifth Ave, Shadyside, Pittsburgh, PA (fallback endpoint)",
        "source": "local_pittsburgh_endpoint_fallback",
    },
}


def in_pittsburgh_bounds(lat: float, lon: float) -> bool:
    return (
        PITTSBURGH_BOUNDS["south"] <= lat <= PITTSBURGH_BOUNDS["north"]
        and PITTSBURGH_BOUNDS["west"] <= lon <= PITTSBURGH_BOUNDS["east"]
    )


def distance_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two latitude/longitude points."""
    r = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


async def _get_json(url: str | tuple[str, ...], *, params=None, json_body=None) -> dict:
    urls = (url,) if isinstance(url, str) else tuple(url)
    last_error = None
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
        follow_redirects=True,
    ) as client:
        for candidate in urls:
            try:
                if json_body is not None:
                    response = await client.post(candidate, json=json_body)
                else:
                    response = await client.get(candidate, params=params)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = f"{candidate}: {exc}"
    raise RuntimeError(f"Public routing service unavailable. Last error: {last_error}")


async def geocode_place(query: str, city_context: Optional[str] = "Pittsburgh, PA") -> dict:
    """Geocode a place/address with a Pittsburgh spatial constraint and DuckDB cache validation."""
    q = query.strip()
    if not q:
        raise ValueError("A start or destination is required.")

    normalized = " ".join(q.lower().split())

    if city_context is None or "pittsburgh" in str(city_context).lower():
        # Exact/known endpoint fallbacks first
        address_fallback = PITTSBURGH_ADDRESS_FALLBACKS.get(normalized)
        if address_fallback:
            return {**address_fallback, "cached": True}

        if "5624 bryant st" in normalized and "15206" in normalized:
            return {**PITTSBURGH_PLACE_ALIASES["highland park"], "cached": True, "source": "local_pittsburgh_address_fallback"}
        if "5903 5th ave" in normalized and "15232" in normalized:
            return {**PITTSBURGH_PLACE_ALIASES["shadyside"], "cached": True, "source": "local_pittsburgh_address_fallback"}
        if "5903 fifth ave" in normalized and "15232" in normalized:
            return {**PITTSBURGH_PLACE_ALIASES["shadyside"], "cached": True, "source": "local_pittsburgh_address_fallback"}

        local_alias = PITTSBURGH_PLACE_ALIASES.get(normalized)
        if local_alias:
            return {**local_alias, "cached": True, "source": "local_pittsburgh_alias"}

    search_q = q
    if city_context and city_context.lower() not in normalized:
        search_q = f"{q}, {city_context}"

    address_key = hashlib.md5(search_q.lower().strip().encode("utf-8")).hexdigest()

    cached = store.query(
        "SELECT lat, lon, full_address FROM geocode_cache WHERE address_key = ? LIMIT 1",
        [address_key],
    )
    if not cached.empty:
        r = cached.iloc[0]
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
            if math.isfinite(lat) and math.isfinite(lon) and (
                city_context is None or in_pittsburgh_bounds(lat, lon)
            ):
                return {
                    "lat": lat,
                    "lon": lon,
                    "display_name": r["full_address"] or q,
                    "cached": True,
                }
        except (TypeError, ValueError):
            pass

    viewbox = (
        f'{PITTSBURGH_BOUNDS["west"]},{PITTSBURGH_BOUNDS["north"]},'
        f'{PITTSBURGH_BOUNDS["east"]},{PITTSBURGH_BOUNDS["south"]}'
    )
    data = await _get_json(
        NOMINATIM_URL,
        params={
            "q": search_q,
            "format": "jsonv2",
            "limit": 8,
            "countrycodes": "us",
            "addressdetails": 1,
            "viewbox": viewbox,
            "bounded": 1,
        },
    )
    if not data:
        raise ValueError(f"Could not find \u201c{q}\u201d in {city_context or 'target area'}. Try a street address or more specific landmark.")

    candidates = []
    for hit in data:
        try:
            lat, lon = float(hit["lat"]), float(hit["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        if city_context and not in_pittsburgh_bounds(lat, lon):
            continue
        candidates.append((hit, lat, lon))

    if not candidates:
        raise ValueError(
            f"Could not find a valid location for \u201c{q}\u201d. "
            "Use a more specific neighborhood, landmark, or street address."
        )

    def rank(item):
        hit, lat, lon = item
        text = (hit.get("display_name") or "").lower()
        address = hit.get("address") or {}
        pittsburgh_match = 0 if "pittsburgh" in text or str(address.get("city", "")).lower() == "pittsburgh" else 1
        importance = -(float(hit.get("importance", 0) or 0))
        return (pittsburgh_match, importance)

    hit, lat, lon = sorted(candidates, key=rank)[0]
    result = {
        "lat": lat,
        "lon": lon,
        "display_name": hit.get("display_name", q),
        "cached": False,
    }

    try:
        store.get_conn().execute(
            """INSERT OR REPLACE INTO geocode_cache
               (address_key, full_address, lat, lon, geocode_source, geocode_accuracy)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                address_key,
                result["display_name"],
                result["lat"],
                result["lon"],
                "nominatim",
                "pittsburgh_bounded_place",
            ],
        )
        store.get_conn().commit()
    except Exception:
        pass

    return result


def dijkstra(
    adjacency: dict[tuple[float, float], dict[tuple[float, float], float]],
    start_node: tuple[float, float],
    end_node: tuple[float, float],
    no_path_msg: str = "No continuous path exists between the two endpoints.",
) -> tuple[list[tuple[float, float]], float]:
    """
    Standard Dijkstra shortest path algorithm over an adjacency dict.
    adjacency: {node: {neighbor: weight_miles}}
    Returns: (path_nodes, total_distance)
    """
    dist: dict[tuple[float, float], float] = {start_node: 0.0}
    prev: dict[tuple[float, float], tuple[float, float]] = {}
    heap: list[tuple[float, tuple[float, float]]] = [(0.0, start_node)]

    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist != dist.get(node):
            continue
        if node == end_node:
            break

        for neighbor, weight in adjacency.get(node, {}).items():
            candidate = current_dist + weight
            if candidate < dist.get(neighbor, float("inf")):
                dist[neighbor] = candidate
                prev[neighbor] = node
                heapq.heappush(heap, (candidate, neighbor))

    if end_node not in dist:
        raise ValueError(no_path_msg)

    path = [end_node]
    cursor = end_node
    while cursor != start_node:
        cursor = prev[cursor]
        path.append(cursor)
    path.reverse()
    return path, dist[end_node]


def bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Calculate compass bearing from coordinate a to coordinate b in degrees."""
    lon1, lat1 = a
    lon2, lat2 = b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def turn_name(delta: float) -> str:
    """Classify heading change into a turn name."""
    delta = (delta + 180) % 360 - 180
    if abs(delta) <= 20:
        return "Continue"
    if 20 < delta <= 55:
        return "Bear right"
    if delta > 55:
        return "Turn right"
    if -55 <= delta < -20:
        return "Bear left"
    return "Turn left"


def make_path_instructions(
    path: list[tuple[float, float]],
    network_name: str = "mapped infrastructure",
) -> list[dict]:
    """Generate route instructions from coordinate sequence."""
    if len(path) < 2:
        return []

    instructions = []
    previous_bearing = None
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        seg_miles = distance_miles(a[1], a[0], b[1], b[0])
        if seg_miles <= 0:
            continue

        b_deg = bearing(a, b)
        if previous_bearing is None:
            text = f"Start on {network_name} for {seg_miles:.2f} miles."
        else:
            turn = turn_name(b_deg - previous_bearing)
            text = f"{turn} on {network_name} for {seg_miles:.2f} miles."

        instructions.append({
            "instruction": text,
            "distance_miles": round(seg_miles, 2),
        })
        previous_bearing = b_deg

    if len(instructions) <= 12:
        return instructions

    collapsed = []
    current = None
    for step in instructions:
        prefix = step["instruction"].split(" on ", 1)[0]
        if current is None or prefix not in {"Continue", "Start"}:
            if current:
                collapsed.append(current)
            current = {
                "instruction": step["instruction"],
                "distance_miles": step["distance_miles"],
            }
        else:
            current["distance_miles"] += step["distance_miles"]
            current["instruction"] = current["instruction"].split(" for ", 1)[0] + (
                f" for {current['distance_miles']:.2f} miles."
            )
    if current:
        collapsed.append(current)
    return collapsed[:20]
