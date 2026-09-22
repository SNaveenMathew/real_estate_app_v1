"""Commute times from each house to the user's work location.

Free and open services only - no API keys, accounts, or credits:

  * Work address -> coordinates: the US Census one-line geocoder, then OpenStreetMap Nominatim
    (both already used elsewhere in the app). You can also type "40.4406, -79.9959" or click the map.
  * Drive / bike / walk: OSRM (open-source routing over OpenStreetMap). The public demo servers are
    the default; run your own ``osrm-routed`` and point ``OSRM_*_URL`` at it to keep coordinates private.
  * Transit (optional): a self-hosted OpenTripPlanner over a GTFS feed you download yourself. Unset
    ``OTP_BASE_URL`` and transit is simply off - there is no free hosted transit router.

Honest limits: drive/bike/walk are FREE-FLOW estimates (no traffic, no signal timing, no departure-time
effect), computed house -> work. Transit is schedule-based (weekday morning) when OTP is configured.

Threading: the application shares one DuckDB connection on the event-loop thread. Everything here that
touches DuckDB runs on that thread; only pure-HTTP work (routing, geocoding) runs in worker threads.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import datetime as dt
import hashlib
import ipaddress
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, NamedTuple
from urllib.parse import urlparse

import httpx

import db.duckdb_store as store
from config import settings

METERS_PER_MILE = 1609.344
WORK_SETTING = "work_location"
MODE_ORDER = ["drive", "bike", "walk", "transit"]
MODE_LABELS = {"drive": "Drive", "bike": "Bike", "walk": "Walk", "transit": "Transit"}
# Modes are not estimated beyond these straight-line distances: the answer would be meaningless
# (a 40-mile walk) and it would waste requests on public servers.
MODE_MAX_MILES = {"drive": 250.0, "bike": 30.0, "walk": 6.0, "transit": 100.0}
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _cfg(name: str, default: Any = "") -> Any:
    value = getattr(settings, name, default)
    return default if value is None else value


class RoutingError(RuntimeError):
    """A routing/geocoding problem whose message is safe to show to the user."""


@dataclass(frozen=True)
class Place:
    lat: float
    lon: float
    label: str = ""
    source: str = ""

    def as_dict(self) -> dict:
        return {"lat": self.lat, "lon": self.lon, "label": self.label, "source": self.source}


class Leg(NamedTuple):
    seconds: float
    meters: float | None


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 3958.7613 * math.asin(math.sqrt(a))


def _short(exc: Any) -> str:
    return re.sub(r"\s+", " ", str(exc))[:160]


# ---------------------------------------------------------------------------
# Work location: geocoding
# ---------------------------------------------------------------------------

_LATLON = re.compile(r"^\s*(-?\d{1,2}\.\d+)\s*[, ]\s*(-?\d{1,3}\.\d+)\s*$")


def parse_latlon(text: str) -> Place | None:
    """'40.4406, -79.9959' -> Place. Both numbers need a decimal point so '12 34' is never read as a coordinate."""
    m = _LATLON.match(text or "")
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return Place(lat, lon, f"{lat:.5f}, {lon:.5f}", "coordinates")


def _census_lookup(client: httpx.Client, text: str) -> Place | None:
    url = str(_cfg("census_geocoder_url", "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"))
    try:
        r = client.get(url, params={"address": text, "benchmark": "Public_AR_Current", "format": "json"})
        matches = r.json().get("result", {}).get("addressMatches", [])
        if not matches:
            return None
        m = matches[0]
        c = m["coordinates"]
        return Place(float(c["y"]), float(c["x"]), m.get("matchedAddress") or text, "census")
    except Exception:
        return None


def _nominatim_lookup(client: httpx.Client, text: str) -> Place | None:
    base = str(_cfg("nominatim_base_url", "https://nominatim.openstreetmap.org")).rstrip("/")
    params: dict[str, Any] = {"q": text, "format": "jsonv2", "limit": 1}
    countries = str(_cfg("nominatim_countrycodes", "us")).strip()
    if countries:
        params["countrycodes"] = countries
    if _cfg("nominatim_email"):
        params["email"] = _cfg("nominatim_email")
    try:
        # Nominatim's usage policy: identify the application, at most 1 request/second. One lookup per user action.
        r = client.get(f"{base}/search", params=params,
                       headers={"User-Agent": str(_cfg("nominatim_user_agent", "RealEstateIntelligence/1.0 (local personal app)"))})
        rows = r.json()
        if not rows:
            return None
        return Place(float(rows[0]["lat"]), float(rows[0]["lon"]), rows[0].get("display_name") or text, "nominatim")
    except Exception:
        return None


def geocode_work_address(text: str, http: httpx.Client | None = None) -> Place:
    """Coordinates for a typed address: 'lat, lon' as-is, else Census one-line, else Nominatim."""
    text = (text or "").strip()
    if not text:
        raise RoutingError("Enter an address, or coordinates like 40.4406, -79.9959.")
    direct = parse_latlon(text)
    if direct:
        return direct
    client = http or httpx.Client(timeout=15.0, follow_redirects=True)
    try:
        place = _census_lookup(client, text) or _nominatim_lookup(client, text)
    finally:
        if http is None:
            client.close()
    if not place:
        raise RoutingError(f"Could not find \"{text}\". Add the city and state, or enter coordinates like 40.4406, -79.9959.")
    return place


# ---------------------------------------------------------------------------
# Routing clients (OSRM for drive/bike/walk, OpenTripPlanner for transit)
# ---------------------------------------------------------------------------

class _Http:
    """Polite HTTP: a minimum interval between requests, bounded retries with backoff."""

    def __init__(self, timeout: float, min_interval: float, http: httpx.Client | None = None):
        self.timeout, self.min_interval = timeout, min_interval
        self._http = http
        self._last = 0.0

    @property
    def http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self.timeout, follow_redirects=True)
        return self._http

    def _throttle(self) -> None:
        gap = self.min_interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()

    def request(self, method: str, url: str, retries: int = 2, **kw) -> Any:
        last: Exception | None = None
        for attempt in range(retries + 1):
            self._throttle()
            try:
                r = self.http.request(method, url, timeout=self.timeout, **kw)
                if r.status_code in _RETRY_STATUS and attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, ValueError) as exc:     # timeouts, connection errors, bad JSON
                last = exc
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
            except httpx.HTTPStatusError as exc:
                last = exc
                break
        raise RoutingError(f"{urlparse(url).netloc or url}: {_short(last)}")

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None


class OsrmClient:
    """OSRM (Project-OSRM/osrm-backend). One profile per server; the profile name in the path is a placeholder."""

    def __init__(self, base_url: str, *, timeout: float = 25.0, min_interval: float = 1.0, table_max: int = 90,
                 http: httpx.Client | None = None):
        self.base = base_url.rstrip("/")
        self.table_max = max(1, int(table_max))
        self.last_error = ""            # why the most recent request failed (surfaced to the user as a warning)
        self._h = _Http(timeout, min_interval, http)

    @staticmethod
    def _coords(places: list[Place]) -> str:
        return ";".join(f"{p.lon:.6f},{p.lat:.6f}" for p in places)            # OSRM wants lon,lat

    def route(self, a: Place, b: Place) -> Leg | None:
        data = self._h.request("GET", f"{self.base}/route/v1/driving/{self._coords([a, b])}", params={"overview": "false"})
        if data.get("code") == "Ok" and data.get("routes"):
            r = data["routes"][0]
            return Leg(float(r["duration"]), float(r["distance"]) if r.get("distance") is not None else None)
        return None

    def _route_each(self, sources: list[Place], dest: Place) -> list[Leg | None]:
        out: list[Leg | None] = []
        for p in sources:
            try:
                out.append(self.route(p, dest))
            except RoutingError as exc:
                self.last_error = str(exc)
                out.append(None)
        return out

    def _table(self, sources: list[Place], dest: Place) -> list[Leg | None]:
        n = len(sources)
        params = {"sources": ";".join(str(i) for i in range(n)), "destinations": str(n), "annotations": "duration,distance"}
        try:
            data = self._h.request("GET", f"{self.base}/table/v1/driving/{self._coords(sources + [dest])}", params=params)
        except RoutingError as exc:
            self.last_error = str(exc)
            return self._route_each(sources, dest)         # some servers disable /table: fall back to /route
        if data.get("code") != "Ok":
            return self._route_each(sources, dest)
        durations, distances = data.get("durations") or [], data.get("distances") or []
        legs: list[Leg | None] = []
        for i in range(n):
            sec = durations[i][0] if i < len(durations) and durations[i] else None
            met = distances[i][0] if i < len(distances) and distances[i] else None
            legs.append(Leg(float(sec), float(met) if met is not None else None) if sec is not None else None)
        return legs

    def matrix(self, sources: list[Place], dest: Place) -> list[Leg | None]:
        """Travel time/distance from every source to ``dest``, in as few requests as the server allows."""
        out: list[Leg | None] = []
        for i in range(0, len(sources), self.table_max):
            out += self._table(sources[i:i + self.table_max], dest)
        return out

    def close(self) -> None:
        self._h.close()


def next_weekday(today: dt.date | None = None) -> dt.date:
    d = (today or dt.date.today()) + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


class OtpClient:
    """OpenTripPlanner 2.x (self-hosted). Uses the GTFS GraphQL ``plan`` query, or the legacy REST ``plan``.

    Not verified against a live OpenTripPlanner in this repository's tests (a mock server is used), so
    treat transit as experimental until you have seen it work against your own instance.
    """

    def __init__(self, base_url: str, *, api: str = "graphql", depart_time: str = "08:00", timeout: float = 60.0,
                 workers: int = 4, http: httpx.Client | None = None):
        self.base, self.api, self.depart, self.workers = base_url.rstrip("/"), api.lower(), depart_time, max(1, workers)
        self.last_error = ""
        self._h = _Http(timeout, 0.0, http)

    def trip(self, a: Place, b: Place, date: dt.date | None = None) -> tuple[float, int] | None:
        date = date or next_weekday()
        if self.api == "rest":
            hh, mm = (int(x) for x in self.depart.split(":"))
            params = {"fromPlace": f"{a.lat},{a.lon}", "toPlace": f"{b.lat},{b.lon}", "mode": "TRANSIT,WALK",
                      "date": date.strftime("%m-%d-%Y"), "time": f"{hh % 12 or 12}:{mm:02d}{'am' if hh < 12 else 'pm'}",
                      "arriveBy": "false", "numItineraries": "1"}
            data = self._h.request("GET", f"{self.base}/otp/routers/default/plan", params=params)
            its = ((data.get("plan") or {}).get("itineraries")) or []
            if not its:
                return None
            return float(its[0]["duration"]), int(its[0].get("transfers", 0) or 0)
        query = ("{ plan(from: {lat: %.6f, lon: %.6f}, to: {lat: %.6f, lon: %.6f}, date: \"%s\", time: \"%s\", "
                 "transportModes: [{mode: TRANSIT}, {mode: WALK}], numItineraries: 1) "
                 "{ itineraries { duration legs { mode } } } }") % (a.lat, a.lon, b.lat, b.lon, date.isoformat(), self.depart)
        data = self._h.request("POST", f"{self.base}/otp/gtfs/v1", json={"query": query})
        its = (((data.get("data") or {}).get("plan") or {}).get("itineraries")) or []
        if not its:
            return None
        modes = [leg.get("mode") for leg in its[0].get("legs", [])]
        transit_legs = sum(1 for m in modes if m not in {"WALK", "BICYCLE", "CAR", None})
        return float(its[0]["duration"]), max(0, transit_legs - 1)

    def trips(self, sources: list[Place], dest: Place) -> list[tuple[float, int] | None]:
        def one(p: Place):
            try:
                return self.trip(p, dest)
            except (RoutingError, KeyError, ValueError, TypeError) as exc:
                self.last_error = _short(exc)
                return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(one, sources))

    def close(self) -> None:
        self._h.close()


def enabled_modes() -> list[str]:
    want = [m.strip().lower() for m in str(_cfg("commute_modes", "drive,bike,walk")).split(",") if m.strip()]
    modes = [m for m in MODE_ORDER if m in want and m != "transit"]
    if str(_cfg("otp_base_url", "")).strip():
        modes.append("transit")           # configuring OpenTripPlanner is what turns transit on
    return modes


def _mode_urls() -> dict[str, str]:
    return {"drive": str(_cfg("osrm_drive_url", "")), "bike": str(_cfg("osrm_bike_url", "")), "walk": str(_cfg("osrm_foot_url", ""))}


def build_clients(modes: list[str]) -> dict[str, Any]:
    timeout = float(_cfg("osrm_timeout_s", 25.0))
    gap = float(_cfg("osrm_min_interval_s", 1.0))
    table_max = int(_cfg("osrm_table_max", 90))
    clients: dict[str, Any] = {}
    for mode, url in _mode_urls().items():
        if mode in modes and url.strip():
            clients[mode] = OsrmClient(url, timeout=timeout, min_interval=gap, table_max=table_max)
    if "transit" in modes and str(_cfg("otp_base_url", "")).strip():
        clients["transit"] = OtpClient(str(_cfg("otp_base_url")), api=str(_cfg("otp_api", "graphql")),
                                       depart_time=str(_cfg("commute_depart_time", "08:00")))
    return clients


def is_public_url(url: str) -> bool:
    """True when routing requests leave this machine/LAN (drives the privacy note in the UI)."""
    host = (urlparse(url).hostname or "").lower()
    if not host or host in {"localhost", "host.docker.internal"} or host.endswith((".local", ".internal")):
        return False
    try:
        ip = ipaddress.ip_address(host)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local)
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Work location + freshness
# ---------------------------------------------------------------------------

def get_work() -> Place | None:
    v = store.get_setting(WORK_SETTING)
    if not v:
        return None
    return Place(float(v["lat"]), float(v["lon"]), v.get("label", ""), v.get("source", ""))


def set_work(place: Place) -> None:
    store.set_setting(WORK_SETTING, place.as_dict())


def clear_work() -> None:
    store.delete_setting(WORK_SETTING)
    store.get_conn().execute("DELETE FROM house_commute")


def work_key(place: Place, modes: list[str]) -> str:
    """Identifies the inputs the stored numbers depend on; a change makes every stored row stale."""
    parts = [f"{place.lat:.5f}", f"{place.lon:.5f}", ",".join(modes), *(_mode_urls()[m] for m in ("drive", "bike", "walk")),
             str(_cfg("otp_base_url", "")), str(_cfg("commute_drive_factor", 1.0))]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Computing (network in worker threads, DuckDB on the loop thread)
# ---------------------------------------------------------------------------

_COLS = ["house_id", "work_key", "work_label", "drive_min", "drive_miles", "bike_min", "bike_miles", "walk_min",
         "walk_miles", "transit_min", "transit_transfers", "straight_line_miles", "status", "source"]


def compute_chunk(chunk: list[tuple[str, float, float]], dest: Place, modes: list[str], clients: dict[str, Any],
                  key: str) -> tuple[list[dict], list[str]]:
    """Commute rows for one chunk of houses. Pure HTTP; a failing mode never sinks the others."""
    factor = float(_cfg("commute_drive_factor", 1.0))
    rows = {hid: {c: None for c in _COLS} | {"house_id": hid, "work_key": key, "work_label": dest.label,
                                             "straight_line_miles": round(haversine_miles(lat, lon, dest.lat, dest.lon), 2)}
            for hid, lat, lon in chunk}
    expected: dict[str, set[str]] = {hid: set() for hid in rows}
    got: dict[str, set[str]] = {hid: set() for hid in rows}
    warnings: list[str] = []
    for mode in modes:
        client = clients.get(mode)
        if client is None:
            continue
        eligible = [(hid, Place(lat, lon)) for hid, lat, lon in chunk if rows[hid]["straight_line_miles"] <= MODE_MAX_MILES[mode]]
        if not eligible:
            continue
        for hid, _ in eligible:
            expected[hid].add(mode)
        client.last_error = ""
        try:
            results = (client.trips if mode == "transit" else client.matrix)([p for _, p in eligible], dest)
        except Exception as exc:
            warnings.append(f"{MODE_LABELS[mode]}: {_short(exc)}")
            continue
        if all(r is None for r in results):
            why = f" ({_short(client.last_error)})" if client.last_error else ""
            warnings.append(f"{MODE_LABELS[mode]}: no routes were returned{why}")
        for (hid, _), res in zip(eligible, results):
            if res is None:
                continue
            got[hid].add(mode)
            if mode == "transit":
                rows[hid]["transit_min"], rows[hid]["transit_transfers"] = round(res[0] / 60.0, 1), res[1]
            else:
                seconds = res.seconds * (factor if mode == "drive" else 1.0)
                rows[hid][f"{mode}_min"] = round(seconds / 60.0, 1)
                rows[hid][f"{mode}_miles"] = round(res.meters / METERS_PER_MILE, 2) if res.meters is not None else None
    for hid, row in rows.items():
        row["status"] = ("out_of_range" if not expected[hid] else "ok" if got[hid] >= expected[hid]
                         else "partial" if got[hid] else "failed")
        row["source"] = "+".join(sorted(got[hid], key=MODE_ORDER.index)) or None
    return list(rows.values()), warnings


def _write_rows(rows: list[dict]) -> None:
    if not rows:
        return
    conn = store.get_conn()
    conn.executemany(
        f"INSERT OR REPLACE INTO house_commute ({', '.join(_COLS)}) VALUES ({', '.join('?' for _ in _COLS)})",
        [[r[c] for c in _COLS] for r in rows])


def _houses_to_compute(scope: str, key: str) -> list[tuple[str, float, float]]:
    sql = "SELECT h.house_id, h.lat, h.lon FROM houses h WHERE h.lat IS NOT NULL AND h.lon IS NOT NULL"
    params: list[Any] = []
    if scope != "all":
        sql += " AND NOT EXISTS (SELECT 1 FROM house_commute c WHERE c.house_id = h.house_id AND c.work_key = ?)"
        params.append(key)
    rows = store.get_conn().execute(sql + " ORDER BY h.house_id", params).fetchall()
    return [(r[0], float(r[1]), float(r[2])) for r in rows]


@dataclass
class Job:
    state: str = "idle"            # idle | running | done | error
    scope: str = ""
    total: int = 0
    done: int = 0
    failed: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {"state": self.state, "scope": self.scope, "total": self.total, "done": self.done, "failed": self.failed,
                "message": self.message, "warnings": list(self.warnings),
                "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1) if self.started_at else 0.0}


_JOB = Job()
_RUNNING = False
_TASKS: set[asyncio.Task] = set()


def job_snapshot() -> dict:
    return _JOB.snapshot()


def reset_job() -> None:
    global _JOB, _RUNNING
    _JOB, _RUNNING = Job(), False


def _claim() -> bool:
    global _RUNNING
    if _RUNNING:
        return False
    _RUNNING = True
    return True


async def refresh_now(scope: str = "missing", *, _claimed: bool = False) -> dict:
    """Compute commute rows for houses without a fresh one (``scope='all'``: every house)."""
    global _RUNNING
    if not _claimed and not _claim():
        return job_snapshot()
    job = _JOB
    try:
        place = get_work()
        if not place:
            job.state, job.message = "error", "Set a work location first."
            return job.snapshot()
        modes = enabled_modes()
        if not modes:
            job.state, job.message = "error", "No travel modes are configured (COMMUTE_MODES / OSRM_*_URL)."
            return job.snapshot()
        key = work_key(place, modes)
        houses = _houses_to_compute(scope, key)
        job.state, job.scope, job.total, job.done, job.failed = "running", scope, len(houses), 0, 0
        job.started_at, job.finished_at, job.message, job.warnings = time.time(), 0.0, "", []
        clients = build_clients(modes)
        size = max(1, int(_cfg("osrm_table_max", 90)))
        try:
            for i in range(0, len(houses), size):
                chunk = houses[i:i + size]
                rows, warns = await asyncio.to_thread(compute_chunk, chunk, place, modes, clients, key)
                _write_rows(rows)
                job.done += len(chunk)
                job.failed += sum(1 for r in rows if r["status"] in ("failed", "partial"))
                for w in warns:
                    if w not in job.warnings:
                        job.warnings.append(w)
        finally:
            for c in clients.values():
                c.close()
        job.state, job.finished_at = "done", time.time()
        job.message = (f"Computed {job.done} house(s)" if job.done else "Nothing to compute: every house is up to date.")
        if job.failed:
            job.message += f"; {job.failed} could not be fully estimated"
    except Exception as exc:                       # never leave the job stuck in "running"
        job.state, job.finished_at, job.message = "error", time.time(), _short(exc)
    finally:
        _RUNNING = False
    return job.snapshot()


def start_refresh(scope: str = "missing") -> dict:
    """Start a background refresh on the running event loop (returns immediately)."""
    if not _claim():
        return job_snapshot()
    _JOB.state, _JOB.scope, _JOB.message = "running", scope, "Starting..."
    task = asyncio.get_running_loop().create_task(refresh_now(scope, _claimed=True))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return job_snapshot()


# ---------------------------------------------------------------------------
# Reading (API, House Chat)
# ---------------------------------------------------------------------------

def status() -> dict:
    place, modes = get_work(), enabled_modes()
    key = work_key(place, modes) if place else None
    conn = store.get_conn()
    houses, with_xy = conn.execute("SELECT COUNT(*), COUNT(*) FILTER (WHERE lat IS NOT NULL AND lon IS NOT NULL) FROM houses").fetchone()
    fresh = conn.execute("SELECT COUNT(*) FROM house_commute WHERE work_key = ?", [key]).fetchone()[0] if key else 0
    urls = _mode_urls()
    providers = {m: {"url": urls[m], "public": is_public_url(urls[m])} for m in ("drive", "bike", "walk") if m in modes and urls[m]}
    if "transit" in modes:
        providers["transit"] = {"url": str(_cfg("otp_base_url")), "public": is_public_url(str(_cfg("otp_base_url")))}
    return {"configured": place is not None, "work": place.as_dict() if place else None, "env_address": str(_cfg("work_address", "")),
            "modes": modes, "transit_configured": "transit" in modes, "providers": providers,
            "drive_factor": float(_cfg("commute_drive_factor", 1.0)),
            "counts": {"houses": houses, "with_coordinates": with_xy, "fresh": fresh, "to_compute": max(0, with_xy - fresh) if place else 0},
            "job": job_snapshot()}


def _fetch_dicts(sql: str, params: list | None = None) -> list[dict]:
    """Rows as plain Python values. NULL stays None: going through pandas would turn a partly-NULL float
    column into NaN, which is not valid JSON (a far house has no walking time, for example)."""
    cur = store.get_conn().execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def commute_for_house(house_id: str) -> dict | None:
    place = get_work()
    rows = _fetch_dicts("SELECT * FROM house_commute WHERE house_id = ?", [house_id])
    if not rows:
        return None
    row = rows[0]
    row["fresh"] = bool(place and row.get("work_key") == work_key(place, enabled_modes()))
    row.pop("work_key", None)
    row["computed_at"] = str(row.get("computed_at") or "")
    return row


def summary() -> dict:
    """Fresh rows only (numbers computed for a previous work location would mislead on the map)."""
    place = get_work()
    if not place:
        return {"work": None, "rows": {}}
    key = work_key(place, enabled_modes())
    rows = _fetch_dicts("SELECT house_id, drive_min, drive_miles, bike_min, walk_min, transit_min FROM house_commute WHERE work_key = ?", [key])
    return {"work": place.as_dict(), "rows": {r["house_id"]: {k: r[k] for k in r if k != "house_id"} for r in rows}}


def describe_for_chat(house_id: str) -> str:
    """House Chat's get_commute_info(): a compact, honest summary for one house."""
    place = get_work()
    if not place:
        return "No work location is set, so there is no commute estimate. The user can set one in the Commute tab of the sidebar."
    row = commute_for_house(house_id)
    if not row:
        return ("A work location is set, but the commute for this house has not been computed yet. "
                "The user can open the Commute tab and choose Compute.")
    est = {}
    for mode in MODE_ORDER:
        mins = row.get(f"{mode}_min")
        if mins is not None:
            item: dict[str, Any] = {"minutes": round(float(mins))}
            if row.get(f"{mode}_miles") is not None:
                item["miles"] = round(float(row[f"{mode}_miles"]), 1)
            if mode == "transit" and row.get("transit_transfers") is not None:
                item["transfers"] = int(row["transit_transfers"])
            est[mode] = item
    house = store.get_house(house_id) or {}
    return json.dumps({
        "work_location": place.label or f"{place.lat:.4f}, {place.lon:.4f}",
        "estimates": est or "none of the travel modes could be estimated for this house",
        "straight_line_miles": row.get("straight_line_miles"),
        "basis": ("Drive/bike/walk are free-flow estimates from OpenStreetMap routing (OSRM) with no traffic or signal timing; "
                  "transit, when present, is schedule-based for a weekday morning."),
        "up_to_date": row["fresh"], "computed_at": row["computed_at"],
        "redfin_scores": {k: house.get(k) for k in ("walk_score", "bike_score", "transit_score")},
    }, indent=2)
