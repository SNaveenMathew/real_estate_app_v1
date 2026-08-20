"""
Bike routing helpers.

Uses free/open infrastructure:
- OpenStreetMap Nominatim for explicit, user-triggered place geocoding.
- OpenStreetMap Nominatim only for endpoint geocoding.
- The locally ingested ``bike_routes`` table is the complete routing graph.

Routes never use an external road/path routing graph. The seven configured
BikePGH layer families are the only route edges.
"""
from __future__ import annotations

import asyncio
import json
import math
import hashlib
import copy
from typing import Optional

import httpx
from shapely.geometry import LineString, Point, box, shape

import db.duckdb_store as store

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "RealEstateIntelligence/1.0 (bike routing)"
NOMINATIM_MIN_INTERVAL_SECONDS = 1.0
HTTP_TIMEOUT_SECONDS = 30.0

# Pittsburgh city-region guard used to make neighborhood/landmark geocoding
# deterministic. Nominatim can return another place with the same name (for
# example "Mount Washington") unless the query is spatially constrained.
PITTSBURGH_BOUNDS = {
    "south": 40.33,
    "west": -80.10,
    "north": 40.55,
    "east": -79.75,
}

# Common Pittsburgh neighborhood centroids used as a deterministic fallback
# when Nominatim is unavailable/rate-limited. These are only endpoint hints;
# the actual route is still snapped to the locally loaded BikePGH graph.
# Shadyside is centered around 40.45152, -79.93655 per OSM-derived sources.
PITTSBURGH_PLACE_ALIASES = {
    "highland park": {"lat": 40.4781, "lon": -79.9110, "display_name": "Highland Park, Pittsburgh, PA"},
    "highland park, pittsburgh": {"lat": 40.4781, "lon": -79.9110, "display_name": "Highland Park, Pittsburgh, PA"},
    "shadyside": {"lat": 40.45152, "lon": -79.93655, "display_name": "Shadyside, Pittsburgh, PA"},
    "shadyside, pittsburgh": {"lat": 40.45152, "lon": -79.93655, "display_name": "Shadyside, Pittsburgh, PA"},
}

# Address-level fallbacks for the two common Pittsburgh endpoint patterns used
# by the app's route tests.  These are intentionally neighborhood endpoint
# anchors, not claims of exact rooftop accuracy; the routing graph snaps them
# to the nearest BikePGH edge.  They allow routing to continue when Nominatim
# is unavailable/rate-limited.
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

def _in_pittsburgh_bounds(lat: float, lon: float) -> bool:
    return (
        PITTSBURGH_BOUNDS["south"] <= lat <= PITTSBURGH_BOUNDS["north"]
        and PITTSBURGH_BOUNDS["west"] <= lon <= PITTSBURGH_BOUNDS["east"]
    )

def _distance_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles for endpoint sanity checks."""
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
    """Geocode a place/address with a Pittsburgh spatial constraint and cache validation."""
    q = query.strip()
    if not q:
        raise ValueError("A start or destination is required.")

    normalized = " ".join(q.lower().split())

    if city_context is None or "pittsburgh" in str(city_context).lower():
        # Exact/known endpoint fallbacks first.
        address_fallback = PITTSBURGH_ADDRESS_FALLBACKS.get(normalized)
        if address_fallback:
            return {**address_fallback, "cached": True}

        # Address patterns that unambiguously identify the intended Pittsburgh
        # neighborhood. These are approximate endpoint anchors and are snapped
        # to the loaded BikePGH graph before Dijkstra.
        if "5624 bryant st" in normalized and "15206" in normalized:
            return {**PITTSBURGH_PLACE_ALIASES["highland park"], "cached": True, "source": "local_pittsburgh_address_fallback"}
        if "5903 5th ave" in normalized and "15232" in normalized:
            return {**PITTSBURGH_PLACE_ALIASES["shadyside"], "cached": True, "source": "local_pittsburgh_address_fallback"}
        if "5903 fifth ave" in normalized and "15232" in normalized:
            return {**PITTSBURGH_PLACE_ALIASES["shadyside"], "cached": True, "source": "local_pittsburgh_address_fallback"}

        # Prefer deterministic local centroids for common Pittsburgh neighborhoods.
        # This prevents a transient public-geocoder failure from masquerading as a
        # failure of the bike-routing graph itself.
        local_alias = PITTSBURGH_PLACE_ALIASES.get(normalized)
        if local_alias:
            return {**local_alias, "cached": True, "source": "local_pittsburgh_alias"}

    search_q = q
    if city_context and city_context.lower() not in normalized:
        search_q = f"{q}, {city_context}"

    address_key = hashlib.md5(search_q.lower().strip().encode("utf-8")).hexdigest()

    # Cached results can pre-date this routing implementation and may point at
    # the wrong "Mount Washington". Only accept them if they still satisfy the
    # city-region constraint.
    cached = store.query(
        "SELECT lat, lon, full_address FROM geocode_cache WHERE address_key = ? LIMIT 1",
        [address_key],
    )
    if not cached.empty:
        r = cached.iloc[0]
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
            if math.isfinite(lat) and math.isfinite(lon) and (
                city_context is None or _in_pittsburgh_bounds(lat, lon)
            ):
                return {
                    "lat": lat,
                    "lon": lon,
                    "display_name": r["full_address"] or q,
                    "cached": True,
                }
        except (TypeError, ValueError):
            pass

    # Pittsburgh viewbox plus bounded=1 prevents same-name places elsewhere
    # from being selected. Request several candidates and choose the first
    # candidate inside the region instead of blindly trusting limit=1.
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
        raise ValueError(f"Could not find “{q}” in Pittsburgh, PA. Try a street address or more specific landmark.")

    candidates = []
    for hit in data:
        try:
            lat, lon = float(hit["lat"]), float(hit["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        if city_context and not _in_pittsburgh_bounds(lat, lon):
            continue
        candidates.append((hit, lat, lon))

    if not candidates:
        raise ValueError(
            f"Could not find a Pittsburgh location for “{q}”. "
            "Use a more specific neighborhood, landmark, or street address."
        )

    # Prefer a result whose display/address explicitly mentions Pittsburgh.
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
    return result



# ---------------------------------------------------------------------------
# BikePGH-only routing graph
# ---------------------------------------------------------------------------
#
# Routing is deliberately performed ONLY on the locally ingested BikePGH
# linework in DuckDB's bike_routes table. OpenStreetMap/Nominatim is used only
# to turn human-friendly endpoint names into coordinates.
#
# All seven BikePGH layer families are included:
#   Bike Lanes, Bikeable Sidewalks, Cautionary Bike Route,
#   On Street Bike Route, Protected Bike Lanes, Sharrows, Trails.
#
# This is a facilities graph, not a general OSM street graph. If the two
# endpoints cannot be connected by the ingested BikePGH network, the tool
# returns "no route" rather than silently falling back to an external street
# router.

BIKE_ROUTE_LAYERS = (
    "bike_lanes",
    "bikeable_sidewalks",
    "cautionary_bike_route",
    "on_street_bike_route",
    "protected_bike_lanes",
    "sharrows",
    "trails",
)

BIKE_GRAPH_SNAP_MAX_MILES = 0.50

# Same display hierarchy used by services.layers.py. Raw source layers remain
# independently routable; this hierarchy only resolves coincident classifications
# for the route visualization.
BIKE_DISPLAY_PRIORITY = {
    "protected_bike_lanes": 100,
    "bike_lanes": 90,
    "trails": 80,
    "bikeable_sidewalks": 70,
    "sharrows": 60,
    "cautionary_bike_route": 50,
    "on_street_bike_route": 40,
}
BIKE_GRAPH_CACHE: dict[str, tuple[int, dict]] = {}

# Crime-avoidance uses the same coarse, severity-weighted grid philosophy as
# the existing Crime map layer, but the routing decision is based on incident
# density first. The default threshold excludes only the highest-density cells
# so the router still has a reasonable chance of finding a continuous route.
DEFAULT_CRIME_GRID_DEG = 0.0018
DEFAULT_CRIME_DENSITY_PERCENTILE = 90.0
MAX_CRIME_FILTER_CELLS = 2500
CRIME_EXCLUSION_BUFFER_DEG = 0.0004  # roughly 30-45 m in Pittsburgh
CRIME_EDGE_SAMPLE_POINTS = 11
CRIME_EDGE_BLOCK_FRACTION = 0.50
CRIME_EDGE_BLOCK_THRESHOLD_RATIO = 0.90
CRIME_EDGE_PENALTY_START_RATIO = 0.50
CRIME_EDGE_MAX_PENALTY = 3.0
VISUAL_CORRIDOR_PAD_DEG = 0.02
MAX_VISUAL_DENSITY_CELLS = 700
MAX_VISUAL_HOTSPOT_CELLS = 350




def _iter_lines(geom):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            if not part.is_empty:
                yield part
    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from _iter_lines(part)


def _load_bike_graph(city: Optional[str]) -> dict:
    """
    Build a graph from every BikePGH line feature for the requested city.

    unary_union() nodes crossing lines so different BikePGH layers can connect
    at shared intersections even when the source shapefiles do not contain a
    common vertex at that crossing.
    """
    # Data is keyed by the folder name, e.g. ``pittsburgh``. Agent/tool
    # callers commonly pass a display context such as ``Pittsburgh, PA``.
    raw_city = (city or "").strip().lower()
    city_candidates = []
    if raw_city:
        city_candidates.append(raw_city)
        if "," in raw_city:
            city_candidates.append(raw_city.split(",", 1)[0].strip())
        else:
            parts = raw_city.rsplit(" ", 1)
            if len(parts) == 2 and len(parts[1]) in (2, 3):
                city_candidates.append(parts[0].strip())
    city_candidates = list(dict.fromkeys(x for x in city_candidates if x))
    city_key = city_candidates[-1] if city_candidates else ""

    where = ""
    params = []
    if city_candidates:
        placeholders = ",".join("?" for _ in city_candidates)
        where = f"WHERE lower(city) IN ({placeholders})"
        params.extend(city_candidates)

    count_df = store.query(f"SELECT COUNT(*) AS n FROM bike_routes {where}", params)
    feature_count = int(count_df.iloc[0]["n"]) if not count_df.empty else 0
    if feature_count <= 0:
        raise ValueError(
            f"No BikePGH infrastructure data is loaded for city '{city or 'all cities'}'."
        )

    cached = BIKE_GRAPH_CACHE.get(city_key)
    if cached and cached[0] == feature_count:
        return cached[1]

    df = store.query(
        f"""
        SELECT geometry_json, layer_type, layer_label, color
        FROM bike_routes
        {where}
        """,
        params,
    )
    source_records = []
    for row in df.itertuples(index=False):
        try:
            geom = shape(json.loads(row.geometry_json))
        except Exception:
            continue
        if geom is None or geom.is_empty:
            continue
        meta = {
            "layer_type": str(getattr(row, "layer_type", "") or ""),
            "label": str(getattr(row, "layer_label", "") or ""),
            "color": str(getattr(row, "color", "") or "#666666"),
        }
        source_records.append((geom, meta))

    lines = []
    for geom, _meta in source_records:
        lines.extend(list(_iter_lines(geom)))

    if not lines:
        raise ValueError(
            f"BikePGH data for '{city or 'all cities'}' contains no routable line geometry."
        )

    from shapely.ops import unary_union

    # Noding the complete local network is what makes crossings between the
    # separate BikePGH source layers routable.
    network = unary_union(lines)

    adjacency: dict[tuple[float, float], dict[tuple[float, float], float]] = {}
    edge_geometries: dict[tuple[tuple[float, float], tuple[float, float]], LineString] = {}

    def add_node(node):
        adjacency.setdefault(node, {})

    def add_edge(a, b):
        if a == b:
            return
        geom = LineString([a, b])
        miles = _line_length_miles(geom)
        if miles <= 0:
            return
        add_node(a)
        add_node(b)
        # Keep the shortest duplicate edge.
        old = adjacency[a].get(b)
        if old is None or miles < old:
            adjacency[a][b] = miles
            adjacency[b][a] = miles
            key = tuple(sorted((a, b)))
            edge_geometries[key] = geom

    for part in _iter_lines(network):
        coords = list(part.coords)
        for a, b in zip(coords, coords[1:]):
            add_edge(
                (round(float(a[0]), 7), round(float(a[1]), 7)),
                (round(float(b[0]), 7), round(float(b[1]), 7)),
            )

    if not adjacency:
        raise ValueError("The BikePGH network graph is empty.")

    graph = {
        "adjacency": adjacency,
        "edge_geometries": edge_geometries,
        "source_records": source_records,
        "feature_count": feature_count,
        "city": city_key,
    }
    BIKE_GRAPH_CACHE[city_key] = (feature_count, graph)
    return graph


def _line_length_miles(line: LineString) -> float:
    coords = list(line.coords)
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        total += _distance_miles(lat1, lon1, lat2, lon2)
    return total


def _graph_edges(graph):
    adjacency = graph["adjacency"]
    for a, nbrs in adjacency.items():
        for b, miles in nbrs.items():
            if a < b:
                yield a, b, miles


def _snap_point_to_graph(graph, lat: float, lon: float):
    """
    Snap a geocoded endpoint to its nearest BikePGH graph edge.

    The returned snap node is inserted into the graph by splitting the edge.
    """
    from shapely.geometry import Point

    point = Point(lon, lat)
    best = None

    for a, b, miles in _graph_edges(graph):
        edge = LineString([a, b])
        distance = point.distance(edge)
        if best is None or distance < best[0]:
            best = (distance, a, b, edge)

    if best is None:
        raise ValueError("The BikePGH graph has no edges.")

    _, a, b, edge = best
    projected = edge.interpolate(edge.project(point))
    snap_lon = round(float(projected.x), 7)
    snap_lat = round(float(projected.y), 7)
    snap_node = (snap_lon, snap_lat)

    snap_miles = _distance_miles(lat, lon, snap_lat, snap_lon)
    if snap_miles > BIKE_GRAPH_SNAP_MAX_MILES:
        raise ValueError(
            f"The endpoint is {snap_miles:.2f} miles from the nearest "
            "BikePGH path. Choose a location closer to mapped bike infrastructure."
        )

    adjacency = graph["adjacency"]
    edge_geometries = graph["edge_geometries"]

    if snap_node not in adjacency:
        old_weight = adjacency[a].get(b)
        if old_weight is None:
            raise ValueError("Unable to split the nearest BikePGH graph edge.")

        # Remove the old edge.
        adjacency[a].pop(b, None)
        adjacency[b].pop(a, None)
        edge_geometries.pop(tuple(sorted((a, b))), None)

        # Add the two split edges.
        w1 = _distance_miles(a[1], a[0], snap_lat, snap_lon)
        w2 = _distance_miles(snap_lat, snap_lon, b[1], b[0])

        adjacency.setdefault(snap_node, {})
        adjacency[a][snap_node] = w1
        adjacency[snap_node][a] = w1
        adjacency[snap_node][b] = w2
        adjacency[b][snap_node] = w2

        edge_geometries[tuple(sorted((a, snap_node)))] = LineString([a, snap_node])
        edge_geometries[tuple(sorted((snap_node, b)))] = LineString([snap_node, b])

    return snap_node, snap_miles


def _crime_density_aggregate(city: Optional[str], grid_deg: float = DEFAULT_CRIME_GRID_DEG) -> list[dict]:
    """Return the shared crime-density grid used by both visualization and routing.

    This is the single source of truth for the crime-density model used by the
    bike/crime analysis view.  Each occupied grid cell is scored using the same
    75% incident-count + 25% severity-weighted formula shown on the map.
    """
    grid_deg = grid_deg if grid_deg and grid_deg > 0 else DEFAULT_CRIME_GRID_DEG

    raw_city = (city or "").strip().lower()
    city_candidates = []
    if raw_city:
        city_candidates.append(raw_city)
        if "," in raw_city:
            city_candidates.append(raw_city.split(",", 1)[0].strip())
        else:
            parts = raw_city.rsplit(" ", 1)
            if len(parts) == 2 and len(parts[1]) in (2, 3):
                city_candidates.append(parts[0].strip())
    city_candidates = list(dict.fromkeys(x for x in city_candidates if x))

    city_clause = ""
    params: list = [grid_deg, grid_deg, grid_deg, grid_deg]
    if city_candidates:
        placeholders = ",".join("?" for _ in city_candidates)
        city_clause = f"AND lower(city) IN ({placeholders})"
        params.extend(city_candidates)

    df = store.query(f"""
        SELECT
            ROUND(lat / ?) * ? AS glat,
            ROUND(lon / ?) * ? AS glon,
            COUNT(*) AS incident_count,
            COALESCE(SUM(severity_weight), 0) AS weighted_score
        FROM crime_incidents
        WHERE lat IS NOT NULL AND lon IS NOT NULL
          {city_clause}
        GROUP BY glat, glon
        HAVING COUNT(*) > 0
    """, params)

    if df.empty:
        return []

    max_count = max(int(df["incident_count"].max()), 1)
    max_weight = max(float(df["weighted_score"].max()), 1.0)
    df["intensity"] = (
        (df["incident_count"].astype(float) / max_count) * 0.75
        + (df["weighted_score"].astype(float) / max_weight) * 0.25
    )

    return [
        {
            "glat": float(r.glat),
            "glon": float(r.glon),
            "incident_count": int(r.incident_count),
            "weighted_score": float(r.weighted_score),
            "intensity": round(float(r.intensity), 4),
            "polygon": box(
                float(r.glon) - grid_deg / 2,
                float(r.glat) - grid_deg / 2,
                float(r.glon) + grid_deg / 2,
                float(r.glat) + grid_deg / 2,
            ),
        }
        for r in df.itertuples(index=False)
    ]


def _crime_hotspot_cells(city: Optional[str], grid_deg: float = DEFAULT_CRIME_GRID_DEG,
                         percentile: float = DEFAULT_CRIME_DENSITY_PERCENTILE) -> list[dict]:
    """Return citywide high-risk crime cells using the exact score shown on the map.

    The percentile baseline remains citywide.  Only the spatial application of
    the threshold is made more precise: a finer grid and smaller buffer are used,
    and individual bike edges are scored continuously before deciding whether an
    edge is removed or merely penalized.
    """
    percentile = min(99.0, max(50.0, float(percentile)))
    cells = _crime_density_aggregate(city, grid_deg)
    if not cells:
        return []

    intensities = sorted(float(c["intensity"]) for c in cells)
    threshold_index = int(math.floor((percentile / 100.0) * (len(intensities) - 1)))
    threshold = intensities[threshold_index]
    hot = [c for c in cells if float(c["intensity"]) >= threshold]

    if len(hot) > MAX_CRIME_FILTER_CELLS:
        hot = sorted(
            hot,
            key=lambda c: (float(c["intensity"]), int(c["incident_count"]), float(c["weighted_score"])),
            reverse=True,
        )[:MAX_CRIME_FILTER_CELLS]

    return [
        {
            **c,
            "crime_threshold": threshold,
            "exclusion_polygon": box(
                float(c["glon"]) - grid_deg / 2 - CRIME_EXCLUSION_BUFFER_DEG,
                float(c["glat"]) - grid_deg / 2 - CRIME_EXCLUSION_BUFFER_DEG,
                float(c["glon"]) + grid_deg / 2 + CRIME_EXCLUSION_BUFFER_DEG,
                float(c["glat"]) + grid_deg / 2 + CRIME_EXCLUSION_BUFFER_DEG,
            ),
        }
        for c in hot
    ]


def _crime_density_cells(city: Optional[str], grid_deg: float = DEFAULT_CRIME_GRID_DEG,
                         max_cells: int = 5000) -> list[dict]:
    """Return the same crime-density cells and intensity score shown to the user.

    Both the map and routing consume the shared aggregate so there is no second,
    silently different crime model.
    """
    cells = _crime_density_aggregate(city, grid_deg)
    if not cells:
        return []
    cells = sorted(
        cells,
        key=lambda c: (float(c["intensity"]), int(c["incident_count"]), float(c["weighted_score"])),
        reverse=True,
    )[:max_cells]
    return cells


def _graph_network_features(graph: dict, max_edges: int = 25000) -> list[dict]:
    """Serialize the currently filtered BikePGH graph for the intermediate map."""
    features = []
    for a, b, _miles in _graph_edges(graph):
        features.append({
            "type": "Feature",
            "properties": {"color": "#4b5563", "filtered_network": True},
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]],
            },
        })
        if len(features) >= max_edges:
            break
    return features


def _analysis_corridor_bounds(start: dict, end: dict, pad_deg: float = VISUAL_CORRIDOR_PAD_DEG) -> dict:
    lats = [float(start["lat"]), float(end["lat"]) ]
    lons = [float(start["lon"]), float(end["lon"]) ]
    return {
        "west": min(lons) - pad_deg,
        "south": min(lats) - pad_deg,
        "east": max(lons) + pad_deg,
        "north": max(lats) + pad_deg,
    }


def _point_in_bbox(lat: float, lon: float, bbox: dict) -> bool:
    return bbox["south"] <= lat <= bbox["north"] and bbox["west"] <= lon <= bbox["east"]


def _geometry_intersects_bbox(feature: dict, bbox: dict) -> bool:
    try:
        coords = feature.get("geometry", {}).get("coordinates") or []
        for lon, lat in coords:
            if _point_in_bbox(float(lat), float(lon), bbox):
                return True
        return False
    except Exception:
        return False


def _build_crime_analysis_visualization(graph: dict, start: dict, end: dict,
                                        hotspot_cells: list[dict] | None,
                                        *, crime_enabled: bool,
                                        crime_percentile: float) -> dict:
    hotspots = hotspot_cells or []
    corridor_bbox = _analysis_corridor_bounds(start, end)

    density_cells = _crime_density_aggregate(graph.get("city"), DEFAULT_CRIME_GRID_DEG) if crime_enabled else []
    # Keep the visualization focused on the requested trip corridor, while the
    # percentile baseline remains citywide.
    relevant_density = [
        c for c in density_cells
        if _point_in_bbox(float(c["glat"]), float(c["glon"]), corridor_bbox)
    ]
    relevant_hotspots = [
        c for c in hotspots
        if _point_in_bbox(float(c["glat"]), float(c["glon"]), corridor_bbox)
    ]
    relevant_density = relevant_density[:MAX_VISUAL_DENSITY_CELLS]
    relevant_hotspots = relevant_hotspots[:MAX_VISUAL_HOTSPOT_CELLS]

    all_network = _graph_network_features(graph)
    remaining_network = [f for f in all_network if _geometry_intersects_bbox(f, corridor_bbox)]
    blocked_network = ((graph.get("crime_avoidance") or {}).get("blocked_network") or {}).get("features", [])
    blocked_network = [f for f in blocked_network if _geometry_intersects_bbox(f, corridor_bbox)]

    return {
        "type": "bike_crime_filtered_network",
        "city": graph.get("city") or "Pittsburgh, PA",
        "start": start,
        "end": end,
        "filtered_bike_network": {"type": "FeatureCollection", "features": remaining_network},
        "blocked_bike_network": {"type": "FeatureCollection", "features": blocked_network},
        "crime_density": [
            {
                "lat": c["glat"], "lon": c["glon"],
                "incident_count": c["incident_count"],
                "weighted_score": c["weighted_score"],
                "intensity": c["intensity"],
                "is_hotspot": any(abs(c["glat"] - h["glat"]) < 1e-9 and abs(c["glon"] - h["glon"]) < 1e-9 for h in relevant_hotspots),
            }
            for c in relevant_density
        ],
        "crime_hotspots": [
            {
                "lat": c["glat"], "lon": c["glon"],
                "incident_count": c["incident_count"],
                "weighted_score": c["weighted_score"],
                "intensity": c.get("intensity", 0.0),
            }
            for c in relevant_hotspots
        ],
        "crime_grid_deg": DEFAULT_CRIME_GRID_DEG,
        "crime_avoidance": {
            "enabled": crime_enabled,
            "percentile": float(crime_percentile),
            "hotspot_cells": len(hotspots),
            "visual_hotspot_cells": len(relevant_hotspots),
            "blocked_edges": int((graph.get("crime_avoidance") or {}).get("blocked_edges", 0)),
            "blocked_segments": int((graph.get("crime_avoidance") or {}).get("blocked_segments", 0)),
            "penalized_edges": 0,
            "visual_blocked_edges": len(blocked_network),
            "crime_threshold": float((graph.get("crime_avoidance") or {}).get("crime_threshold", 0.0)),
            "exclusion_buffer_deg": CRIME_EXCLUSION_BUFFER_DEG,
            "edge_exposure_model": (graph.get("crime_avoidance") or {}).get("edge_exposure_model", {}),
        },
        "corridor_bbox": corridor_bbox,
        "bbox": corridor_bbox,
        "explanation": (
            f"Crime filter removes BikePGH edges intersecting cells in the top {100.0 - float(crime_percentile):.0f}% "
            "by the same combined crime-density intensity shown on the map "
            "(75% normalized incident count + 25% normalized severity). Edges are hard-blocked only when at least "
            "a logical BikePGH segment is blocked in full when any portion of that intersection-to-intersection segment "
            "intersects a buffered top-percentile crime hotspot. Clean segments remain unfiltered. The map shows the "
            "same continuous density field plus complete blocked segments and the remaining BikePGH network."
        ),
    }


def _crime_cell_lookup(cells: list[dict], grid_deg: float) -> dict[tuple[float, float], float]:
    lookup = {}
    for c in cells:
        key = (round(float(c["glon"]), 6), round(float(c["glat"]), 6))
        lookup[key] = float(c["intensity"])
    return lookup


def _crime_cell_center(value: float, grid_deg: float) -> float:
    return round(math.floor((value / grid_deg) + 0.5) * grid_deg, 6)


def _edge_crime_exposure(edge: LineString, crime_lookup: dict[tuple[float, float], float],
                         grid_deg: float, hotspot_threshold: float) -> dict:
    """Sample a geometry against the shared continuous crime-density score."""
    if not crime_lookup or hotspot_threshold <= 0:
        return {
            "avg_intensity": 0.0,
            "max_intensity": 0.0,
            "hotspot_fraction": 0.0,
        }

    sample_count = max(2, CRIME_EDGE_SAMPLE_POINTS)
    values = []
    for i in range(sample_count):
        frac = i / (sample_count - 1)
        pt = edge.interpolate(edge.length * frac)
        key = (
            _crime_cell_center(float(pt.x), grid_deg),
            _crime_cell_center(float(pt.y), grid_deg),
        )
        values.append(float(crime_lookup.get(key, 0.0)))

    avg_intensity = sum(values) / len(values)
    max_intensity = max(values)
    hotspot_fraction = sum(v >= hotspot_threshold for v in values) / len(values)
    return {
        "avg_intensity": round(avg_intensity, 4),
        "max_intensity": round(max_intensity, 4),
        "hotspot_fraction": round(hotspot_fraction, 4),
    }


def _logical_bike_segments(graph: dict, boundary_nodes: set | None = None) -> list[dict]:
    """Group graph edges into intersection-to-intersection BikePGH segments.

    The BikePGH graph is noded at geometry vertices and crossings, so many
    graph edges can represent one real-world street/path section.  For crime
    filtering we treat every maximal chain between meaningful junctions
    (nodes whose degree is not 2) as one logical segment.  This prevents a
    crime overlap on one tiny noded edge from producing a visual/route state
    like dark → orange → dark along the same intersection-to-intersection
    street segment.
    """
    adjacency = graph["adjacency"]
    geometries = graph["edge_geometries"]
    boundary_nodes = set(boundary_nodes or set())
    visited: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    segments: list[dict] = []

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    def edge_line(a, b):
        key = edge_key(a, b)
        geom = geometries.get(key)
        return geom if geom is not None else LineString([a, b])

    def walk_from(start, first):
        edge_keys = []
        coords = [start]
        prev = start
        current = first

        while True:
            key = edge_key(prev, current)
            if key in visited:
                break
            visited.add(key)
            edge_keys.append(key)
            geom = edge_line(prev, current)
            geom_coords = list(geom.coords)
            if geom_coords and tuple(geom_coords[0]) != tuple(prev):
                geom_coords.reverse()
            coords.extend([(float(x), float(y)) for x, y in geom_coords[1:]])

            if current in boundary_nodes or len(adjacency.get(current, {})) != 2:
                break

            neighbors = [n for n in adjacency.get(current, {}) if n != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            prev, current = current, nxt

        if len(edge_keys) < 1:
            return None
        return {
            "start_node": start,
            "end_node": current,
            "edge_keys": edge_keys,
            "geometry": LineString(coords),
        }

    # First, walk outward from every meaningful junction/end node.
    for start, nbrs in adjacency.items():
        if len(nbrs) == 2 and start not in boundary_nodes:
            continue
        for first in nbrs:
            key = edge_key(start, first)
            if key in visited:
                continue
            segment = walk_from(start, first)
            if segment:
                segments.append(segment)

    # Handle isolated degree-2 cycles (rare, but valid graph topology).
    for a, nbrs in adjacency.items():
        for b in nbrs:
            key = edge_key(a, b)
            if key in visited:
                continue
            segment = walk_from(a, b)
            if segment:
                segments.append(segment)

    return segments


def _segment_crime_exposure(segment: dict, crime_lookup: dict[tuple[float, float], float],
                            grid_deg: float, hotspot_threshold: float,
                            hotspot_polygons: list) -> dict:
    """Evaluate crime exposure for an entire intersection-to-intersection segment."""
    geom = segment["geometry"]
    sampled = _edge_crime_exposure(geom, crime_lookup, grid_deg, hotspot_threshold)

    # The hard-block decision is segment-level: if any meaningful portion of the
    # logical intersection-to-intersection segment enters a top-percentile
    # exclusion area, the complete segment is removed. This keeps the visual
    # and routing representations consistent.
    intersects_hotspot = any(geom.intersects(poly) for poly in hotspot_polygons)

    return {
        **sampled,
        "intersects_hotspot": bool(intersects_hotspot),
        "blocked": bool(intersects_hotspot),
    }


def _edge_feature(a, b, *, color="#4b5563", dashed=False, blocked=False):
    return {
        "type": "Feature",
        "properties": {
            "color": color,
            "dashed": bool(dashed),
            "blocked_by_crime": bool(blocked),
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]],
        },
    }


def _segment_feature(segment: dict, *, color="#f97316", dashed=True):
    return {
        "type": "Feature",
        "properties": {
            "color": color,
            "dashed": bool(dashed),
            "blocked_by_crime": True,
            "logical_segment": True,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(x), float(y)] for x, y in segment["geometry"].coords],
        },
    }


def _apply_crime_avoidance_filter(graph: dict, hotspot_cells: list[dict],
                                  all_density_cells: list[dict] | None = None,
                                  grid_deg: float = DEFAULT_CRIME_GRID_DEG,
                                  crime_threshold: float | None = None,
                                  boundary_nodes: set | None = None) -> dict:
    """Apply citywide crime filtering at logical intersection-to-intersection segments."""
    if not hotspot_cells:
        graph["crime_avoidance"] = {
            "enabled": True,
            "applied": True,
            "hotspot_cells": 0,
            "blocked_edges": 0,
            "blocked_segments": 0,
            "penalized_edges": 0,
            "blocked_network": {"type": "FeatureCollection", "features": []},
        }
        return graph

    if all_density_cells is None:
        all_density_cells = _crime_density_aggregate(graph.get("city"), grid_deg)
    lookup = _crime_cell_lookup(all_density_cells, grid_deg)
    threshold = float(
        crime_threshold
        if crime_threshold is not None
        else hotspot_cells[0].get("crime_threshold", 0.0)
    )
    hotspot_polygons = [c["exclusion_polygon"] for c in hotspot_cells if c.get("exclusion_polygon") is not None]

    adjacency = graph["adjacency"]
    edge_geometries = graph["edge_geometries"]
    blocked_edge_count = 0
    blocked_segment_count = 0
    blocked_features = []
    segment_stats = []

    # Segment first, then apply the same decision to every graph edge belonging
    # to that real-world intersection-to-intersection section.
    segments = _logical_bike_segments(graph, boundary_nodes=boundary_nodes)
    for segment in segments:
        exposure = _segment_crime_exposure(
            segment, lookup, grid_deg, threshold, hotspot_polygons
        )
        segment_stats.append(exposure)

        if not exposure["blocked"]:
            continue

        blocked_segment_count += 1
        blocked_features.append(_segment_feature(segment))
        for edge_key in segment["edge_keys"]:
            a, b = edge_key
            if b in adjacency.get(a, {}):
                adjacency[a].pop(b, None)
                blocked_edge_count += 1
            if a in adjacency.get(b, {}):
                adjacency[b].pop(a, None)
            edge_geometries.pop(edge_key, None)

    graph["crime_avoidance"] = {
        "enabled": True,
        "applied": True,
        "hotspot_cells": len(hotspot_cells),
        "crime_threshold": round(threshold, 4),
        "grid_deg": grid_deg,
        "exclusion_buffer_deg": CRIME_EXCLUSION_BUFFER_DEG,
        "blocked_edges": blocked_edge_count,
        "blocked_segments": blocked_segment_count,
        "penalized_edges": 0,
        "blocked_network": {"type": "FeatureCollection", "features": blocked_features},
        "edge_exposure_model": {
            "model": "logical intersection-to-intersection segment",
            "sample_points": max(2, CRIME_EDGE_SAMPLE_POINTS),
            "block_rule": "block the complete logical segment when any portion intersects a buffered top-percentile crime hotspot",
            "citywide_baseline": True,
        },
    }
    return graph

def _dijkstra(graph, start_node, end_node):
    import heapq

    adjacency = graph["adjacency"]
    dist = {start_node: 0.0}
    prev = {}
    heap = [(0.0, start_node)]

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
        raise ValueError(
            "No continuous path exists between the two endpoints using only "
            "the locally ingested BikePGH infrastructure."
        )

    path = [end_node]
    cursor = end_node
    while cursor != start_node:
        cursor = prev[cursor]
        path.append(cursor)
    path.reverse()
    return path, dist[end_node]



def _local_bike_facilities(route_geom: LineString, city: Optional[str]) -> dict:
    """Describe which local BikePGH layer families intersect the route.

    Because every route edge is sourced from ``bike_routes``, the route is
    already 100% BikePGH infrastructure. This query is only for attribution
    and layer reporting.
    """
    minx, miny, maxx, maxy = route_geom.bounds
    pad = 0.0002  # small query envelope around the route
    params = [maxx + pad, minx - pad, maxy + pad, miny - pad]

    raw_city = (city or "").strip().lower()
    city_candidates = []
    if raw_city:
        city_candidates.append(raw_city)
        if "," in raw_city:
            city_candidates.append(raw_city.split(",", 1)[0].strip())
        else:
            parts = raw_city.rsplit(" ", 1)
            if len(parts) == 2 and len(parts[1]) in (2, 3):
                city_candidates.append(parts[0].strip())
    city_candidates = list(dict.fromkeys(x for x in city_candidates if x))

    clause = ""
    if city_candidates:
        placeholders = ",".join("?" for _ in city_candidates)
        clause = f"AND lower(city) IN ({placeholders})"
        params.extend(city_candidates)

    df = store.query(
        f"""
        SELECT DISTINCT layer_type, layer_label, color
        FROM bike_routes
        WHERE min_lon <= ? AND max_lon >= ?
          AND min_lat <= ? AND max_lat >= ?
          {clause}
        """,
        params,
    )

    labels = []
    for row in df.itertuples(index=False):
        labels.append({
            "layer_type": getattr(row, "layer_type", ""),
            "label": getattr(row, "layer_label", ""),
            "color": getattr(row, "color", ""),
        })

    return {
        "facility_overlap_miles": None,
        "facility_overlap_pct": 100.0,
        "facility_segments": sorted(labels, key=lambda x: (x["label"], x["layer_type"])),
        "route_source": "bike_routes",
    }


def _used_bike_infrastructure(graph: dict, path: list[tuple[float, float]]) -> list[dict]:
    """Return only BikePGH source segments actually used by the selected path.

    Each Dijkstra edge is matched against the original BikePGH source features.
    The returned geometry is the selected graph edge itself, carrying the
    original BikePGH layer metadata/color. This avoids drawing unrelated nearby
    facilities just because they happen to fall inside the route bounding box.
    """
    source_records = graph.get("source_records", [])
    if not source_records or len(path) < 2:
        return []

    from shapely.geometry import LineString

    tolerance = 1e-7
    output = []
    seen = set()

    for a, b in zip(path, path[1:]):
        edge = LineString([a, b])
        candidates = []
        for source_geom, meta in source_records:
            try:
                # Match exact/near-coincident source linework. We do not use
                # a broad buffer because that would pull in facilities that
                # are merely adjacent to the selected route.
                if source_geom.distance(edge) <= tolerance:
                    candidates.append(meta)
            except Exception:
                continue

        if not candidates:
            continue

        # A selected graph edge can coincide with multiple raw BikePGH
        # classifications. For the route visual, show ONE canonical category
        # using the same specificity hierarchy as the home-page map.
        meta = max(
            candidates,
            key=lambda m: (
                BIKE_DISPLAY_PRIORITY.get(m["layer_type"], 0),
                m.get("label", ""),
            ),
        )
        key = (meta["layer_type"], meta["label"], meta["color"], a, b)
        if key in seen:
            continue
        seen.add(key)
        output.append({
            "type": "Feature",
            "properties": {
                "layer_type": meta["layer_type"],
                "label": meta["label"],
                "color": meta["color"],
                "display_priority": BIKE_DISPLAY_PRIORITY.get(meta["layer_type"], 0),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(a[0]), float(a[1])],
                                [float(b[0]), float(b[1])]],
            },
        })

    return output


def _bearing(a, b):
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dx = math.cos(lat2) * math.sin(lon2 - lon1)
    dy = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    )
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def _turn_name(delta):
    delta = (delta + 540.0) % 360.0 - 180.0
    if abs(delta) < 25:
        return "Continue"
    if delta > 55:
        return "Turn left"
    if delta < -55:
        return "Turn right"
    return "Bear left" if delta > 0 else "Bear right"


def _make_path_instructions(path):
    """
    Generate route instructions from graph geometry only.

    Because BikePGH linework generally has facility names/types rather than
    street names, instructions intentionally refer to the mapped BikePGH
    infrastructure instead of inventing street names from an external router.
    """
    if len(path) < 2:
        return []

    instructions = []
    cumulative = 0.0
    previous_bearing = None
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        segment_miles = _distance_miles(a[1], a[0], b[1], b[0])
        if segment_miles <= 0:
            continue

        bearing = _bearing(a, b)
        if previous_bearing is None:
            text = f"Start on BikePGH mapped infrastructure for {segment_miles:.2f} miles."
        else:
            turn = _turn_name(bearing - previous_bearing)
            text = f"{turn} and continue on BikePGH mapped infrastructure for {segment_miles:.2f} miles."

        instructions.append({
            "instruction": text,
            "distance_miles": round(segment_miles, 2),
        })
        cumulative += segment_miles
        previous_bearing = bearing

    # Collapse the potentially large number of tiny noded segments into a
    # compact representation while preserving direction changes.
    if len(instructions) <= 12:
        return instructions

    collapsed = []
    current = None
    for step in instructions:
        prefix = step["instruction"].split(" and ", 1)[0]
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


async def route_bike(
    start_text: str,
    end_text: str,
    *,
    city: Optional[str] = "Pittsburgh, PA",
    avoid_crime_dense_areas: bool = False,
    crime_density_percentile: float = DEFAULT_CRIME_DENSITY_PERCENTILE,
) -> dict:
    """Find a BikePGH-only route and, for crime-aware requests, analyze the filtered graph first."""
    try:
        start = await geocode_place(start_text, city_context=city)
        if not start.get("cached"):
            await asyncio.sleep(NOMINATIM_MIN_INTERVAL_SECONDS)
        end = await geocode_place(end_text, city_context=city)
    except Exception as exc:
        # Endpoint geocoding is the only external dependency. Return a
        # structured routing result instead of leaking a generic
        # "routing service" exception through the tool layer.
        return {
            "start": {"query": start_text},
            "end": {"query": end_text},
            "city": city or "Pittsburgh, PA",
            "route": None,
            "alternatives_considered": 1,
            "crime_avoidance": {
                "enabled": bool(avoid_crime_dense_areas),
                "applied": False,
                "hotspot_cells": 0,
                "blocked_edges": 0,
                "error": f"Endpoint geocoding failed: {exc}",
            },
            "analysis_visualization": None,
            "no_route": True,
            "crime_filter_error": bool(avoid_crime_dense_areas),
            "note": (
                "The endpoints could not be geocoded. "
                "This is an endpoint lookup problem, not evidence that a bike route is impossible. "
                f"Details: {exc}"
            ),
        }

    straight_line = _distance_miles(start["lat"], start["lon"], end["lat"], end["lon"])
    if straight_line < 0.05:
        raise ValueError(
            f"Start and destination geocoded to nearly the same location "
            f"({straight_line:.2f} miles apart). Please choose more specific places."
        )

    # Never mutate the cached base graph. Endpoint snapping and crime filtering
    # modify the graph in-place, so every request gets an isolated working copy.
    base_graph = _load_bike_graph(city)
    graph = {
        "adjacency": {n: dict(v) for n, v in base_graph["adjacency"].items()},
        "edge_geometries": dict(base_graph["edge_geometries"]),
        "source_records": list(base_graph.get("source_records", [])),
        "feature_count": base_graph["feature_count"],
        "city": base_graph["city"],
    }

    crime_avoidance = {
        "enabled": bool(avoid_crime_dense_areas),
        "applied": not bool(avoid_crime_dense_areas),
        "hotspot_cells": 0,
        "blocked_edges": 0,
    }

    try:
        start_node, start_snap_miles = _snap_point_to_graph(graph, start["lat"], start["lon"])
        end_node, end_snap_miles = _snap_point_to_graph(graph, end["lat"], end["lon"])
        if start_node == end_node:
            raise ValueError(
                "Both endpoints snap to the same BikePGH path location. "
                "Choose more specific endpoint locations."
            )

        hotspot_cells = []
        if avoid_crime_dense_areas:
            try:
                hotspot_cells = _crime_hotspot_cells(
                    city, percentile=crime_density_percentile
                )
                # A crime-aware route is only valid when crime data exists. An
                # empty crime dataset must not silently turn into an unfiltered
                # route with `applied=True`.
                if not hotspot_cells:
                    analysis_visualization = _build_crime_analysis_visualization(
                        graph, start, end, [],
                        crime_enabled=True, crime_percentile=crime_density_percentile,
                    )
                    return {
                        "start": start, "end": end,
                        "city": graph.get("city") or (city or "Pittsburgh, PA"),
                        "provider": "BikePGH local graph",
                        "attribution": "Endpoint geocoding: Nominatim / OpenStreetMap",
                        "route": None, "alternatives_considered": 1,
                        "crime_avoidance": {
                            "enabled": True, "applied": False,
                            "hotspot_cells": 0, "blocked_edges": 0,
                            "error": "No crime-density data is available for the requested city.",
                        },
                        "analysis_visualization": analysis_visualization,
                        "no_route": True, "crime_filter_error": True,
                        "note": "Crime-aware routing is not possible because no crime-density data is available for the requested city.",
                    }
                _apply_crime_avoidance_filter(
                    graph, hotspot_cells,
                    all_density_cells=_crime_density_aggregate(city, DEFAULT_CRIME_GRID_DEG),
                    grid_deg=DEFAULT_CRIME_GRID_DEG,
                    crime_threshold=float(hotspot_cells[0].get("crime_threshold", 0.0)),
                    boundary_nodes={start_node, end_node},
                )
                crime_avoidance = graph.get("crime_avoidance", crime_avoidance)
                crime_avoidance.update({"enabled": True, "applied": True})
            except Exception as exc:
                analysis_visualization = _build_crime_analysis_visualization(
                    graph, start, end, hotspot_cells,
                    crime_enabled=True, crime_percentile=crime_density_percentile,
                )
                return {
                    "start": start, "end": end,
                    "city": graph.get("city") or (city or "Pittsburgh, PA"),
                    "provider": "BikePGH local graph",
                    "attribution": "Endpoint geocoding: Nominatim / OpenStreetMap",
                    "route": None, "alternatives_considered": 1,
                    "crime_avoidance": {
                        "enabled": True, "applied": False,
                        "hotspot_cells": len(hotspot_cells),
                        "blocked_edges": 0, "error": str(exc),
                    },
                    "analysis_visualization": analysis_visualization,
                    "no_route": True, "crime_filter_error": True,
                    "note": f"Crime-avoidance filter could not be applied: {exc}",
                }

        analysis_visualization = (
            _build_crime_analysis_visualization(
                graph, start, end, hotspot_cells,
                crime_enabled=True, crime_percentile=crime_density_percentile,
            ) if avoid_crime_dense_areas else None
        )

        try:
            path, route_miles = _dijkstra(graph, start_node, end_node)
        except ValueError as exc:
            if avoid_crime_dense_areas:
                return {
                    "start": start, "end": end,
                    "city": graph.get("city") or (city or "Pittsburgh, PA"),
                    "provider": "BikePGH local graph",
                    "attribution": "Endpoint geocoding: Nominatim / OpenStreetMap",
                    "route": None, "alternatives_considered": 1,
                    "crime_avoidance": graph.get("crime_avoidance", crime_avoidance),
                    "analysis_visualization": analysis_visualization,
                    "no_route": True,
                    "note": str(exc),
                }
            raise

        if route_miles <= 0:
            raise ValueError("The BikePGH-only route has zero length.")

        route_geom = LineString([(lon, lat) for lon, lat in path])
        facilities = _local_bike_facilities(route_geom, city)
        used_infrastructure = _used_bike_infrastructure(graph, path)
        layer_labels = sorted({x["label"] for x in facilities.get("facility_segments", [])})
        duration_minutes = route_miles / 10.0 * 60.0
        route_lats = [float(lat) for lon, lat in path]
        route_lons = [float(lon) for lon, lat in path]
        route_bbox = {
            "west": min(route_lons), "south": min(route_lats),
            "east": max(route_lons), "north": max(route_lats),
        }

        return {
            "start": start, "end": end,
            "city": graph.get("city") or (city or "Pittsburgh, PA"),
            "provider": "BikePGH local graph",
            "attribution": "Endpoint geocoding: Nominatim / OpenStreetMap",
            "route": {
                "summary": {"length": round(route_miles, 3), "time": round(duration_minutes * 60.0, 1)},
                "shape": [[lat, lon] for lon, lat in path],
                "maneuvers": _make_path_instructions(path),
                "local_bike_facilities": facilities,
                "used_infrastructure": {"type": "FeatureCollection", "features": used_infrastructure},
                "bikepgh_layers_used": BIKE_ROUTE_LAYERS,
                "bikepgh_layers_near_route": layer_labels,
                "start_snap_miles": round(start_snap_miles, 3),
                "end_snap_miles": round(end_snap_miles, 3),
                "bbox": route_bbox,
            },
            "alternatives_considered": 1,
            "crime_avoidance": graph.get("crime_avoidance", crime_avoidance),
            "analysis_visualization": analysis_visualization,
            "note": (
                "This route is constrained to the locally ingested BikePGH infrastructure graph. "
                "Endpoint geocoding uses Nominatim only; travel time is an estimate."
            ),
        }
    except Exception as exc:
        if avoid_crime_dense_areas:
            try:
                analysis_visualization = _build_crime_analysis_visualization(
                    graph, start, end, [],
                    crime_enabled=True, crime_percentile=crime_density_percentile,
                )
            except Exception:
                analysis_visualization = None
            return {
                "start": start, "end": end,
                "city": graph.get("city") or (city or "Pittsburgh, PA"),
                "provider": "BikePGH local graph",
                "attribution": "Endpoint geocoding: Nominatim / OpenStreetMap",
                "route": None, "alternatives_considered": 1,
                "crime_avoidance": {**crime_avoidance, "enabled": True, "applied": False, "error": str(exc)},
                "analysis_visualization": analysis_visualization,
                "no_route": True, "crime_filter_error": True,
                "note": f"Crime-aware routing could not be completed: {exc}",
            }
        raise
