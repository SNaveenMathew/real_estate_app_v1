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
from typing import Optional

import httpx
from shapely.geometry import LineString, shape

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
) -> dict:
    """
    Find a route using ONLY locally ingested BikePGH infrastructure.

    Endpoint names are geocoded with Nominatim, but no external road/path
    routing service is consulted. The route graph is built from every BikePGH
    layer stored in ``bike_routes`` for the requested city.
    """
    start = await geocode_place(start_text, city_context=city)
    if not start.get("cached"):
        await asyncio.sleep(NOMINATIM_MIN_INTERVAL_SECONDS)
    end = await geocode_place(end_text, city_context=city)

    straight_line = _distance_miles(start["lat"], start["lon"], end["lat"], end["lon"])
    if straight_line < 0.05:
        raise ValueError(
            f"Start and destination geocoded to nearly the same location "
            f"({straight_line:.2f} miles apart). Please choose more specific places."
        )

    # Endpoint insertion mutates the graph. Copy the cached graph for this
    # request so subsequent requests cannot inherit another request's snap nodes.
    graph = _load_bike_graph(city)
    graph = {
        "adjacency": {n: dict(v) for n, v in graph["adjacency"].items()},
        "edge_geometries": dict(graph["edge_geometries"]),
        "source_records": list(graph.get("source_records", [])),
        "feature_count": graph["feature_count"],
        "city": graph["city"],
    }

    start_node, start_snap_miles = _snap_point_to_graph(
        graph, start["lat"], start["lon"]
    )
    end_node, end_snap_miles = _snap_point_to_graph(
        graph, end["lat"], end["lon"]
    )

    if start_node == end_node:
        raise ValueError(
            "Both endpoints snap to the same BikePGH path location. "
            "Choose more specific endpoint locations."
        )

    path, route_miles = _dijkstra(graph, start_node, end_node)
    if route_miles <= 0:
        raise ValueError("The BikePGH-only route has zero length.")

    route_geom = LineString([(lon, lat) for lon, lat in path])
    facilities = _local_bike_facilities(route_geom, city)
    used_infrastructure = _used_bike_infrastructure(graph, path)

    # Replace the broad overlap statistic with a clear statement of what the
    # route actually uses: every edge came from the union of BikePGH layers.
    layer_labels = sorted(
        {x["label"] for x in facilities.get("facility_segments", [])}
    )

    # Approximate bicycle travel speed for a facilities-only path. This is
    # intentionally an estimate because the BikePGH linework has no speed/time
    # model attached to it.
    duration_minutes = route_miles / 10.0 * 60.0

    route_lats = [float(x[0]) for x in [[lat, lon] for lon, lat in path]]
    route_lons = [float(x[1]) for x in [[lat, lon] for lon, lat in path]]
    route_bbox = {
        "west": min(route_lons), "south": min(route_lats),
        "east": max(route_lons), "north": max(route_lats),
    }

    return {
        "start": start,
        "end": end,
        "city": graph.get("city") or (city or "Pittsburgh, PA"),
        "provider": "BikePGH local graph",
        "attribution": "Endpoint geocoding: Nominatim / OpenStreetMap",
        "route": {
            "summary": {
                "length": round(route_miles, 3),
                "time": round(duration_minutes * 60.0, 1),
            },
            "shape": [[lat, lon] for lon, lat in path],
            "maneuvers": _make_path_instructions(path),
            "local_bike_facilities": facilities,
            "used_infrastructure": {
                "type": "FeatureCollection",
                "features": used_infrastructure,
            },
            "bikepgh_layers_used": BIKE_ROUTE_LAYERS,
            "bikepgh_layers_near_route": layer_labels,
            "start_snap_miles": round(start_snap_miles, 3),
            "end_snap_miles": round(end_snap_miles, 3),
            "bbox": route_bbox,
        },
        "alternatives_considered": 1,
        "note": (
            "This route is constrained to the locally ingested BikePGH "
            "infrastructure graph. It does not use an external road or trail "
            "routing graph. All seven configured BikePGH layer families are "
            "eligible: Bike Lanes, Bikeable Sidewalks, Cautionary Bike Route, "
            "On Street Bike Route, Protected Bike Lanes, Sharrows, and Trails. "
            "Endpoint geocoding uses Nominatim only; travel time is an estimate."
        ),
    }
