"""
Transit shortest path routing engine using the locally ingested FTA National Transit Map network.

Provides graph-based multimodal transit routing with transfers across agencies
and modes (bus, light rail, heavy rail, commuter rail, etc.).
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from shapely.geometry import shape, LineString

import db.duckdb_store as store
from services.routing_common import (
    distance_miles,
    dijkstra,
    geocode_place,
    bearing,
    turn_name,
)

logger = logging.getLogger(__name__)

# Average transit operational speed in mph (including typical dwell times)
AVERAGE_TRANSIT_SPEED_MPH = 14.0
INITIAL_WAIT_MINUTES = 5.0
TRANSFER_WAIT_MINUTES = 4.0
MAX_SNAP_DISTANCE_MILES = 3.0
DEFAULT_TRANSFER_MAX_MILES = 0.045  # ~75 meters transfer walk radius


async def route_transit(
    start: str,
    end: str,
    city: Optional[str] = "Pittsburgh, PA",
    max_transfer_dist_miles: float = DEFAULT_TRANSFER_MAX_MILES,
) -> dict:
    """Find a transit route between two locations using the National Transit Map graph."""
    start_info = await geocode_place(start, city_context=city)
    end_info = await geocode_place(end, city_context=city)

    start_lat, start_lon = float(start_info["lat"]), float(start_info["lon"])
    end_lat, end_lon = float(end_info["lat"]), float(end_info["lon"])

    # Expand search corridor bounding box
    pad = 0.04  # ~2.5 miles corridor buffer
    west = min(start_lon, end_lon) - pad
    east = max(start_lon, end_lon) + pad
    south = min(start_lat, end_lat) - pad
    north = max(start_lat, end_lat) + pad

    rows = store.query(
        """
        SELECT route_id, agency_id, route_short_name, route_long_name, route_desc,
               route_type_name, route_color, route_text_color, geometry_json
        FROM transit_routes
        WHERE min_lon <= ? AND max_lon >= ? AND min_lat <= ? AND max_lat >= ?
        """,
        [east, west, north, south],
    )

    if len(rows) == 0:
        # Retry with wider corridor
        pad = 0.08
        west = min(start_lon, end_lon) - pad
        east = max(start_lon, end_lon) + pad
        south = min(start_lat, end_lat) - pad
        north = max(start_lat, end_lat) + pad
        rows = store.query(
            """
            SELECT route_id, agency_id, route_short_name, route_long_name, route_desc,
                   route_type_name, route_color, route_text_color, geometry_json
            FROM transit_routes
            WHERE min_lon <= ? AND max_lon >= ? AND min_lat <= ? AND max_lat >= ?
            """,
            [east, west, north, south],
        )

    if len(rows) == 0:
        raise ValueError(
            f"No transit routes found in the area between '{start}' and '{end}'. "
            "Please verify the endpoints are in an area covered by National Transit Map data."
        )

    adjacency: dict[tuple[float, float], dict[tuple[float, float], float]] = {}
    edge_meta: dict[tuple[tuple[float, float], tuple[float, float]], dict] = {}

    for r in rows.itertuples():
        geom_str = getattr(r, "geometry_json", None)
        if not geom_str:
            continue
        try:
            g = shape(json.loads(geom_str))
        except Exception:
            continue

        parts = [g] if g.geom_type == "LineString" else list(getattr(g, "geoms", []))
        meta = {
            "route_id": getattr(r, "route_id", ""),
            "agency_id": getattr(r, "agency_id", ""),
            "route_short_name": str(getattr(r, "route_short_name", "") or "").strip(),
            "route_long_name": str(getattr(r, "route_long_name", "") or "").strip(),
            "route_type_name": str(getattr(r, "route_type_name", "Transit") or "Transit").strip(),
            "route_color": str(getattr(r, "route_color", "#2563eb") or "#2563eb").strip(),
            "is_transfer": False,
        }

        for line in parts:
            coords = list(line.coords)
            for a, b in zip(coords, coords[1:]):
                pa = (round(float(a[0]), 5), round(float(a[1]), 5))
                pb = (round(float(b[0]), 5), round(float(b[1]), 5))
                if pa == pb:
                    continue
                dist = distance_miles(pa[1], pa[0], pb[1], pb[0])
                if dist <= 0:
                    continue

                if pa not in adjacency:
                    adjacency[pa] = {}
                if pb not in adjacency:
                    adjacency[pb] = {}

                old = adjacency[pa].get(pb)
                if old is None or dist < old:
                    adjacency[pa][pb] = dist
                    adjacency[pb][pa] = dist
                    ekey = tuple(sorted((pa, pb)))
                    edge_meta[ekey] = meta

    if not adjacency:
        raise ValueError("No routable transit line geometry found in the bounding corridor.")

    # Spatial transfer links between nearby transit vertices across different lines
    grid: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for node in adjacency.keys():
        gk = (round(node[0], 3), round(node[1], 3))
        grid.setdefault(gk, []).append(node)

    transfer_penalty_miles = 0.02  # ~1 min walking penalty to favor remaining on line
    for gk, nodes in grid.items():
        for dx in (-0.001, 0, 0.001):
            for dy in (-0.001, 0, 0.001):
                nbr_key = (round(gk[0] + dx, 3), round(gk[1] + dy, 3))
                for other_node in grid.get(nbr_key, []):
                    for n1 in nodes:
                        if n1 < other_node and other_node not in adjacency.get(n1, {}):
                            d = distance_miles(n1[1], n1[0], other_node[1], other_node[0])
                            if 0 < d <= max_transfer_dist_miles:
                                weight = d + transfer_penalty_miles
                                adjacency[n1][other_node] = weight
                                adjacency[other_node][n1] = weight
                                ekey = tuple(sorted((n1, other_node)))
                                edge_meta[ekey] = {
                                    "is_transfer": True,
                                    "distance_miles": d,
                                    "route_type_name": "Transfer Walk",
                                }

    # Snap start and end to transit network
    start_node = min(adjacency.keys(), key=lambda n: distance_miles(start_lat, start_lon, n[1], n[0]))
    end_node = min(adjacency.keys(), key=lambda n: distance_miles(end_lat, end_lon, n[1], n[0]))

    start_snap_miles = distance_miles(start_lat, start_lon, start_node[1], start_node[0])
    end_snap_miles = distance_miles(end_lat, end_lon, end_node[1], end_node[0])

    if start_snap_miles > MAX_SNAP_DISTANCE_MILES:
        raise ValueError(
            f"Start location '{start}' is {start_snap_miles:.1f} miles from the nearest transit route. "
            "Please choose a location closer to mapped transit service."
        )
    if end_snap_miles > MAX_SNAP_DISTANCE_MILES:
        raise ValueError(
            f"Destination '{end}' is {end_snap_miles:.1f} miles from the nearest transit route. "
            "Please choose a location closer to mapped transit service."
        )

    # Dijkstra shortest path
    path, total_weight = dijkstra(
        adjacency,
        start_node,
        end_node,
        no_path_msg=(
            f"No continuous transit path exists between '{start}' and '{end}' "
            "using the locally available transit routes and pedestrian transfers."
        ),
    )

    # Compute actual physical route length (excluding transfer virtual penalty)
    actual_route_miles = 0.0
    segments = []
    current_line = None
    current_meta = None
    current_seg_dist = 0.0

    for pa, pb in zip(path, path[1:]):
        ekey = tuple(sorted((pa, pb)))
        meta = edge_meta.get(ekey, {})
        d = distance_miles(pa[1], pa[0], pb[1], pb[0])
        actual_route_miles += d

        is_trans = meta.get("is_transfer", False)
        line_key = "TRANSFER" if is_trans else (
            meta.get("route_id") or meta.get("route_short_name") or meta.get("route_long_name")
        )

        if line_key == current_line:
            current_seg_dist += d
        else:
            if current_line is not None:
                segments.append((current_line, current_seg_dist, current_meta))
            current_line = line_key
            current_seg_dist = d
            current_meta = meta

    if current_line is not None:
        segments.append((current_line, current_seg_dist, current_meta))

    # Generate maneuvers
    maneuvers = []
    start_disp = start_info.get("display_name", start)
    end_disp = end_info.get("display_name", end)

    maneuvers.append({
        "instruction": f"Start near {start_disp} (walk {start_snap_miles:.2f} mi to transit stop).",
        "distance_miles": round(start_snap_miles, 2),
    })

    transfer_count = 0
    used_lines_dict = {}

    for line_key, seg_dist, meta in segments:
        if meta.get("is_transfer"):
            transfer_count += 1
            maneuvers.append({
                "instruction": f"Walk {round(seg_dist * 5280):d} ft ({seg_dist:.2f} mi) to transfer connection.",
                "distance_miles": round(seg_dist, 2),
            })
        else:
            route_name = meta.get("route_short_name") or meta.get("route_long_name") or "Transit Line"
            route_type = meta.get("route_type_name") or "Transit"
            long_desc = f" ({meta.get('route_long_name')})" if meta.get("route_long_name") and meta.get("route_long_name") != route_name else ""
            line_label = f"{route_type} {route_name}{long_desc}"

            used_lines_dict[line_label] = {
                "route_short_name": meta.get("route_short_name"),
                "route_long_name": meta.get("route_long_name"),
                "route_type_name": meta.get("route_type_name"),
                "route_color": meta.get("route_color"),
                "agency_id": meta.get("agency_id"),
            }

            est_ride_mins = round((seg_dist / AVERAGE_TRANSIT_SPEED_MPH) * 60)
            maneuvers.append({
                "instruction": f"Board {line_label} and ride for {seg_dist:.2f} mi (~{max(1, est_ride_mins)} mins).",
                "distance_miles": round(seg_dist, 2),
                "route_name": route_name,
                "route_type": route_type,
                "route_color": meta.get("route_color"),
            })

    maneuvers.append({
        "instruction": f"Alight and arrive near {end_disp} (walk {end_snap_miles:.2f} mi to destination).",
        "distance_miles": round(end_snap_miles, 2),
    })

    transit_lines_used = list(used_lines_dict.values())

    # Duration estimate
    in_transit_minutes = (actual_route_miles / AVERAGE_TRANSIT_SPEED_MPH) * 60.0
    total_duration_minutes = (
        in_transit_minutes
        + INITIAL_WAIT_MINUTES
        + (transfer_count * TRANSFER_WAIT_MINUTES)
        + ((start_snap_miles + end_snap_miles) / 3.0 * 60.0)  # ~3 mph walk speed for access
    )

    route_lats = [float(p[1]) for p in path]
    route_lons = [float(p[0]) for p in path]
    bbox = {
        "west": min(route_lons),
        "south": min(route_lats),
        "east": max(route_lons),
        "north": max(route_lats),
    }

    return {
        "start": start_info,
        "end": end_info,
        "city": city or "Pittsburgh, PA",
        "provider": "FTA National Transit Map",
        "attribution": "Transit routes: Federal Transit Administration / Bureau of Transportation Statistics; Geocoding: Nominatim / OpenStreetMap",
        "route": {
            "summary": {
                "length": round(actual_route_miles, 2),
                "time": round(total_duration_minutes * 60.0, 1),
                "duration_minutes": round(total_duration_minutes, 1),
                "transfer_count": transfer_count,
            },
            "shape": [[lat, lon] for lon, lat in path],
            "maneuvers": maneuvers,
            "transit_lines_used": transit_lines_used,
            "start_snap_miles": round(start_snap_miles, 2),
            "end_snap_miles": round(end_snap_miles, 2),
            "bbox": bbox,
        },
        "transit_lines": transit_lines_used,
        "transfers": transfer_count,
        "note": "Shortest transit route calculated using National Transit Map linework and pedestrian transfers.",
    }
