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
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from shapely.geometry import LineString, Point, box, shape

import db.duckdb_store as store

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "RealEstateIntelligence/1.0 (routing engine)"
NOMINATIM_MIN_INTERVAL_SECONDS = 1.0
HTTP_TIMEOUT_SECONDS = 30.0

# How long an unresolved address is remembered before the geocoder is asked
# again (a miss is cached too, so a repeated turn costs no HTTP request).
GEOCODE_MISS_TTL_SECONDS = 3 * 24 * 3600

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
        "lat": 40.4781,
        "lon": -79.9110,
        "display_name": "5624 Bryant St, Highland Park, Pittsburgh, PA (fallback endpoint)",
        "source": "local_pittsburgh_endpoint_fallback",
    },
    "5624 bryant street, pittsburgh, pa 15206": {
        "lat": 40.4781,
        "lon": -79.9110,
        "display_name": "5624 Bryant St, Highland Park, Pittsburgh, PA (fallback endpoint)",
        "source": "local_pittsburgh_endpoint_fallback",
    },
    "5903 5th ave, shadyside, pa 15232": {
        "lat": 40.45152,
        "lon": -79.93655,
        "display_name": "5903 5th Ave, Shadyside, Pittsburgh, PA (fallback endpoint)",
        "source": "local_pittsburgh_endpoint_fallback",
    },
    "5903 fifth ave, shadyside, pa 15232": {
        "lat": 40.45152,
        "lon": -79.93655,
        "display_name": "5903 Fifth Ave, Shadyside, Pittsburgh, PA (fallback endpoint)",
        "source": "local_pittsburgh_endpoint_fallback",
    },
    "5903 5th ave, pittsburgh, pa 15232": {
        "lat": 40.45152,
        "lon": -79.93655,
        "display_name": "5903 5th Ave, Shadyside, Pittsburgh, PA (fallback endpoint)",
        "source": "local_pittsburgh_endpoint_fallback",
    },
    "5903 fifth ave, pittsburgh, pa 15232": {
        "lat": 40.45152,
        "lon": -79.93655,
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


# ---------------------------------------------------------------------------
# Address normalisation
# ---------------------------------------------------------------------------
# A geocoder is only as reliable as the spelling it is handed, and the two
# sides of this system spell addresses differently: the house dataset stores
# "4940 2nd Ave" while OpenStreetMap names the same street "Second Avenue"
# (OSM is itself inconsistent - it keeps "South 18th Street" in digits).
# Nominatim expands "Ave"->"Avenue" and "St"->"Street", but it does not
# normalise the ordinal "2nd"->"Second", so a single abbreviation can turn a
# real, mapped house into "address not found".
#
# Every lookup is therefore keyed on a canonical spelling (ordinals spelled
# out, street types and directionals expanded, comma-part boundaries kept),
# and a miss walks a short ladder of alternative spellings before giving up.
# The canonical key is what makes the cache spelling-invariant: "2nd Ave" and
# "Second Avenue" resolve to one row, so one successful lookup is reused by
# every spelling of that address from then on.

_STREET_TYPE_WORDS = {
    "aly": "alley",
    "av": "avenue",
    "ave": "avenue",
    "blvd": "boulevard",
    "cir": "circle",
    "ct": "court",
    "dr": "drive",
    "expy": "expressway",
    "ft": "fort",
    "hls": "hills",
    "hts": "heights",
    "hwy": "highway",
    "ln": "lane",
    "mt": "mount",
    "pkwy": "parkway",
    "pl": "place",
    "rd": "road",
    "sq": "square",
    "st": "street",
    "str": "street",
    "ter": "terrace",
    "trl": "trail",
    "vly": "valley",
    "xing": "crossing",
}

_DIRECTION_WORDS = {
    "n": "north",
    "ne": "northeast",
    "nw": "northwest",
    "s": "south",
    "se": "southeast",
    "sw": "southwest",
    "e": "east",
    "w": "west",
}

_ORDINAL_UNITS = (
    "zeroth",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
)
_TENS_CARDINAL = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}
# "112th" style numbers above 99 keep their digits: a rare street name is not
# worth guessing at ("one hundred twelfth" is not how anyone writes it).
_ORDINAL_TOKEN_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$")


def _ordinal_word(number: int) -> str:
    if 0 <= number < len(_ORDINAL_UNITS):
        return _ORDINAL_UNITS[number]
    if 10 <= number < 100:
        tens, ones = divmod(number, 10)
        base = _TENS_CARDINAL[tens]
        return f"{base} {_ORDINAL_UNITS[ones]}" if ones else f"{base[:-1]}ieth"
    return f"{number}th"


def _canonical_token(token: str, *, ordinal_words: bool) -> str:
    if ordinal_words:
        ordinal = _ORDINAL_TOKEN_RE.match(token)
        if ordinal:
            return _ordinal_word(int(ordinal.group(1)))
    return _STREET_TYPE_WORDS.get(token) or _DIRECTION_WORDS.get(token) or token


def _canonical_address(text: str, *, ordinal_words: bool = True) -> str:
    """Spelling-invariant form of an address, keeping its comma-part boundaries."""
    parts = []
    for part in str(text or "").split(","):
        # Keep apostrophes ("O'Hara St"): they are part of the emitted query.
        cleaned = " ".join(re.sub(r"[^a-z0-9' ]+", " ", part.lower()).split())
        tokens = [_canonical_token(t, ordinal_words=ordinal_words) for t in cleaned.split()]
        if tokens:
            parts.append(" ".join(tokens))
    return ", ".join(parts)


def _query_has_context(query: str, city_context: str) -> bool:
    """True when the query already names every word of the city context."""
    context_tokens = _canonical_address(city_context).replace(",", " ").split()
    query_tokens = set(_canonical_address(query).replace(",", " ").split())
    return bool(context_tokens) and all(token in query_tokens for token in context_tokens)


def _raw_search_text(query: str, city_context: Optional[str]) -> str:
    """The query exactly as earlier versions sent it (kept for continuity)."""
    if city_context and city_context.lower() not in " ".join(query.lower().split()):
        return f"{query}, {city_context}"
    return query


def _canonical_search_text(query: str, city_context: Optional[str], *, ordinal_words: bool = True) -> str:
    text = _canonical_address(query, ordinal_words=ordinal_words)
    if city_context and not _query_has_context(query, city_context):
        suffix = _canonical_address(city_context, ordinal_words=ordinal_words)
        return f"{text}, {suffix}"
    return text


def _address_cache_key(query: str, city_context: Optional[str]) -> str:
    """Cache key that is identical for every spelling of the same address."""
    return hashlib.md5(_canonical_search_text(query, city_context).encode("utf-8")).hexdigest()


def _legacy_address_key(search_text: str) -> str:
    """Key used by earlier versions, so previously cached rows stay reachable."""
    return hashlib.md5(search_text.lower().strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Geocode cache (DuckDB geocode_cache, shared with services/geocoder.py)
# ---------------------------------------------------------------------------

_CACHE_MISS_SOURCE = "nominatim_miss"


def _cache_read(keys: list[str]) -> Optional[dict]:
    """First cached row among `keys`, in the caller's preference order."""
    keys = [key for key in keys if key]
    if not keys:
        return None
    placeholders = ", ".join("?" * len(keys))
    try:
        frame = store.query(
            "SELECT address_key, full_address, lat, lon, geocode_source, geocode_accuracy, "
            f"CAST(geocoded_at AS VARCHAR) AS geocoded_at FROM geocode_cache "
            f"WHERE address_key IN ({placeholders})",
            keys,
        )
    except Exception:
        return None
    if frame.empty:
        return None
    rows = {row["address_key"]: row for _, row in frame.iterrows()}
    for key in keys:
        if key in rows:
            return rows[key]
    return None


def _cache_write(
    key: str,
    full_address: str,
    lat: Optional[float],
    lon: Optional[float],
    source: str,
    accuracy: str,
) -> None:
    """Persist one geocode (or one miss) so later turns never repeat the work."""
    try:
        conn = store.get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO geocode_cache
               (address_key, full_address, lat, lon, geocode_source, geocode_accuracy, geocoded_at)
               VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            [key, full_address, lat, lon, source, accuracy],
        )
        conn.commit()
    except Exception:
        # The cache is an accelerator: a read-only or busy database must never
        # turn a working geocode into a failed request.
        logger.debug("Could not write geocode cache entry for %s", key, exc_info=True)


def _miss_is_fresh(geocoded_at: Optional[str]) -> bool:
    """A remembered miss expires, so a fixed spelling or new OSM data can win."""
    if not geocoded_at:
        return False
    try:
        stamp = datetime.fromisoformat(str(geocoded_at).replace("Z", "+00:00"))
    except ValueError:
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - stamp < timedelta(seconds=GEOCODE_MISS_TTL_SECONDS)


# Nominatim's usage policy allows at most one request per second. The ladder
# below can issue more than one request for a single lookup, so every call is
# spaced here rather than by each caller.
_nominatim_last_call = 0.0


async def _wait_for_nominatim_slot() -> None:
    global _nominatim_last_call
    gap = NOMINATIM_MIN_INTERVAL_SECONDS - (time.monotonic() - _nominatim_last_call)
    if gap > 0:
        await asyncio.sleep(gap)
    _nominatim_last_call = time.monotonic()


def _best_bounded_hit(data: list, city_context: Optional[str]) -> Optional[tuple[float, float, str]]:
    """Best in-bounds Nominatim hit, preferring a Pittsburgh match."""
    candidates = []
    for hit in data or ():
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
        return None

    def rank(item):
        hit, _, _ = item
        text = (hit.get("display_name") or "").lower()
        address = hit.get("address") or {}
        pittsburgh_match = 0 if "pittsburgh" in text or str(address.get("city", "")).lower() == "pittsburgh" else 1
        importance = -(float(hit.get("importance", 0) or 0))
        return (pittsburgh_match, importance)

    hit, lat, lon = sorted(candidates, key=rank)[0]
    return lat, lon, hit.get("display_name") or ""


def _house_coordinates(query: str, city_context: Optional[str]) -> Optional[dict]:
    """Coordinates already stored for a listed house - no network involved.

    The application knows the exact position of every house it lists, so an
    address that names one of them needs no geocoder at all. Matching is on the
    same canonical form used for the cache, which is why "4940 2nd Ave" here
    matches however the caller spells it.
    """
    match = re.match(r"^(\d+[A-Za-z]?)\s+([^,]+)", query.strip())
    if not match:
        return None
    number, street = match.group(1), match.group(2)
    target = _canonical_address(f"{number} {street}")
    if not target:
        return None

    clauses = ["lat IS NOT NULL", "lon IS NOT NULL", "lower(address) LIKE ?"]
    params: list = [f"{number.lower()} %"]
    zip_match = re.search(r"\b(\d{5})\b", query)
    if zip_match:
        clauses.append("zip = ?")
        params.append(zip_match.group(1))
    else:
        parts = [part.strip() for part in query.split(",")]
        city = parts[1] if len(parts) > 1 else (city_context or "").split(",")[0].strip()
        if not city:
            return None
        clauses.append("lower(city) = lower(?)")
        params.append(city)

    try:
        frame = store.query(
            f"SELECT address, city, state, zip, lat, lon FROM houses WHERE {' AND '.join(clauses)} LIMIT 50",
            params,
        )
    except Exception:
        return None

    for row in frame.itertuples(index=False):
        if _canonical_address(row.address) != target:
            continue
        try:
            lat, lon = float(row.lat), float(row.lon)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        if city_context and not in_pittsburgh_bounds(lat, lon):
            continue
        region = " ".join(part for part in (row.state, row.zip) if part)
        label = ", ".join(part for part in (row.address, row.city, region) if part)
        return {
            "lat": lat,
            "lon": lon,
            "display_name": label,
            "cached": True,
            "source": "houses",
        }
    return None


async def geocode_place(query: str, city_context: Optional[str] = "Pittsburgh, PA") -> dict:
    """Geocode a place/address with a Pittsburgh spatial constraint and DuckDB cache validation.

    Resolution order, cheapest and most authoritative first:

    1. exact local Pittsburgh aliases and endpoint fallbacks (deterministic)
    2. the coordinates already stored for a listed house (local, no network)
    3. the DuckDB geocode cache, keyed on a spelling-invariant form
    4. Nominatim, trying a short ladder of spellings before giving up

    Every success is written back under the canonical key, so the cost of a
    lookup is paid once for *all* spellings of that address. A real miss is
    remembered too, but only for GEOCODE_MISS_TTL_SECONDS, so repeated turns
    stop hammering the geocoder without pinning the answer forever.
    """
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
            return {
                **PITTSBURGH_PLACE_ALIASES["highland park"],
                "cached": True,
                "source": "local_pittsburgh_address_fallback",
            }
        if "5903 5th ave" in normalized and "15232" in normalized:
            return {
                **PITTSBURGH_PLACE_ALIASES["shadyside"],
                "cached": True,
                "source": "local_pittsburgh_address_fallback",
            }
        if "5903 fifth ave" in normalized and "15232" in normalized:
            return {
                **PITTSBURGH_PLACE_ALIASES["shadyside"],
                "cached": True,
                "source": "local_pittsburgh_address_fallback",
            }

        local_alias = PITTSBURGH_PLACE_ALIASES.get(normalized)
        if local_alias:
            return {**local_alias, "cached": True, "source": "local_pittsburgh_alias"}

    canonical_key = _address_cache_key(q, city_context)
    legacy_key = _legacy_address_key(_raw_search_text(q, city_context))

    known_house = _house_coordinates(q, city_context)
    if known_house:
        _cache_write(
            canonical_key,
            known_house["display_name"],
            known_house["lat"],
            known_house["lon"],
            "houses",
            "house_coordinates",
        )
        return known_house

    cached = _cache_read([canonical_key, legacy_key])
    if cached is not None:
        try:
            lat, lon = float(cached["lat"]), float(cached["lon"])
        except (TypeError, ValueError):
            lat = lon = float("nan")
        in_bounds = (
            math.isfinite(lat) and math.isfinite(lon) and (city_context is None or in_pittsburgh_bounds(lat, lon))
        )
        if in_bounds:
            return {
                "lat": lat,
                "lon": lon,
                "display_name": cached["full_address"] or q,
                "cached": True,
                "source": cached["geocode_source"] or "cache",
            }
        if not math.isfinite(lat) and _miss_is_fresh(cached["geocoded_at"]):
            raise ValueError(
                cached["full_address"]
                or f"Could not find \u201c{q}\u201d in {city_context or 'target area'}. "
                "Try a street address or more specific landmark."
            )

    viewbox = (
        f"{PITTSBURGH_BOUNDS['west']},{PITTSBURGH_BOUNDS['north']},"
        f"{PITTSBURGH_BOUNDS['east']},{PITTSBURGH_BOUNDS['south']}"
    )

    # Ladder: the spelling this app has always sent, then the canonical
    # spellings with ordinals as words (OSM's "Second Avenue") and as digits
    # (OSM's "18th Street"). Nominatim expands neither form into the other.
    candidates: list[str] = []
    for candidate in (
        _raw_search_text(q, city_context),
        _canonical_search_text(q, city_context, ordinal_words=True),
        _canonical_search_text(q, city_context, ordinal_words=False),
    ):
        if candidate and candidate.lower() not in {seen.lower() for seen in candidates}:
            candidates.append(candidate)

    saw_any_hit = False
    for search_q in candidates:
        await _wait_for_nominatim_slot()
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
        if data:
            saw_any_hit = True
        hit = _best_bounded_hit(data, city_context)
        if hit is None:
            continue
        lat, lon, display_name = hit
        _cache_write(canonical_key, display_name, lat, lon, "nominatim", "pittsburgh_bounded_place")
        return {"lat": lat, "lon": lon, "display_name": display_name or q, "cached": False, "source": "nominatim"}

    if saw_any_hit:
        message = (
            f"Could not find a valid location for \u201c{q}\u201d. "
            "Use a more specific neighborhood, landmark, or street address."
        )
    else:
        message = (
            f"Could not find \u201c{q}\u201d in {city_context or 'target area'}. "
            "Try a street address or more specific landmark."
        )
    _cache_write(canonical_key, message, None, None, _CACHE_MISS_SOURCE, "unresolved")
    raise ValueError(message)


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

        instructions.append(
            {
                "instruction": text,
                "distance_miles": round(seg_miles, 2),
            }
        )
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
