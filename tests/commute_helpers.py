"""A local mock of every external service the Commute feature talks to (no network, no keys)."""
import http.server
import json
import threading
import urllib.parse

import pytest

from services import commute

WORK = (40.4406, -79.9959)              # "downtown Pittsburgh"


def add_houses(conn, n=6, step=0.01):
    """n houses NE of the work location, each ~0.7 mile further out than the last."""
    for i in range(1, n + 1):
        conn.execute("INSERT INTO houses (house_id, address, city, state, zip, lat, lon, status, price) "
                     "VALUES (?,?,?,?,?,?,?,'Active',200000)",
                     [f"h{i}", f"{i} Test St", "Pittsburgh", "PA", "15213", WORK[0] + step * i, WORK[1] + step * i])


def expected_minutes(profile, lat, lon):
    return commute.haversine_miles(lat, lon, *WORK) * MockRouting.SEC_PER_MILE[profile] / 60.0


class MockRouting:
    """OSRM (car/bike/foot), Census + Nominatim geocoders and OpenTripPlanner behind one port."""
    SEC_PER_MILE = {"car": 60.0, "bike": 300.0, "foot": 1200.0}     # 60 / 12 / 3 mph
    ROAD_FACTOR = 1.3                                              # road distance / straight-line distance

    def __init__(self):
        self.calls = []                   # (service, profile, n_coords)
        self.raw = []                     # (path, query, headers)
        self.otp_bodies = []
        self.fail_table = set()           # profiles whose /table answers 400
        self.down = set()                 # profiles that always answer 500
        self.flaky = 0                    # the first N requests answer 503
        self.null_sources = set()         # source indices reported unroutable
        self.census, self.nominatim = {}, {}
        self.otp_graphql = self.otp_rest = None
        self.delay = 0.0
        self._lock = threading.Lock()
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, payload):
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                outer._handle(self, "GET")

            def do_POST(self):
                outer._handle(self, "POST")

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _handle(self, h, method):
        parsed = urllib.parse.urlsplit(h.path)      # NOT urlparse: it splits ';params' off the last path segment
        q = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        parts = parsed.path.strip("/").split("/")
        body = h.rfile.read(int(h.headers.get("Content-Length") or 0)) if method == "POST" else b""
        with self._lock:
            self.raw.append((parsed.path, q, {k.lower(): v for k, v in h.headers.items()}))
            flaky = self.flaky > 0
            if flaky:
                self.flaky -= 1
        if self.delay:
            threading.Event().wait(self.delay)
        if flaky:
            return h._send(503, {"code": "Error"})
        head = parts[0]
        if head in self.SEC_PER_MILE:
            profile, service = head, parts[1]
            coords = [tuple(float(x) for x in c.split(",")) for c in parts[4].split(";")]     # (lon, lat)
            with self._lock:
                self.calls.append((service, profile, len(coords)))
            if profile in self.down:
                return h._send(500, {"code": "Error"})
            if service == "table":
                if profile in self.fail_table:
                    return h._send(400, {"code": "NotImplemented"})
                sources = [int(i) for i in q["sources"].split(";")]
                dest = coords[int(q["destinations"])]
                durs, dists = [], []
                for i in sources:
                    if i in self.null_sources:
                        durs.append([None]); dists.append([None]); continue
                    miles = commute.haversine_miles(coords[i][1], coords[i][0], dest[1], dest[0])
                    durs.append([miles * self.SEC_PER_MILE[profile]])
                    dists.append([miles * 1609.344 * self.ROAD_FACTOR])
                return h._send(200, {"code": "Ok", "durations": durs, "distances": dists})
            miles = commute.haversine_miles(coords[0][1], coords[0][0], coords[1][1], coords[1][0])
            return h._send(200, {"code": "Ok", "routes": [{"duration": miles * self.SEC_PER_MILE[profile],
                                                            "distance": miles * 1609.344 * self.ROAD_FACTOR}]})
        if head == "census":
            hit = self.census.get(q.get("address"))
            matches = [{"matchedAddress": hit[0], "coordinates": {"x": hit[1], "y": hit[2]}}] if hit else []
            return h._send(200, {"result": {"addressMatches": matches}})
        if head == "nominatim":
            hit = self.nominatim.get(q.get("q"))
            return h._send(200, [{"lat": str(hit[2]), "lon": str(hit[1]), "display_name": hit[0]}] if hit else [])
        if parts[:3] == ["otp", "gtfs", "v1"]:
            self.otp_bodies.append(json.loads(body or b"{}"))
            return h._send(200, self.otp_graphql or {"data": {"plan": {"itineraries": []}}})
        if parts[:4] == ["otp", "routers", "default", "plan"]:
            return h._send(200, self.otp_rest or {"plan": {"itineraries": []}})
        return h._send(404, {"error": "unknown route"})


@pytest.fixture()
def routing(fresh_db, monkeypatch):
    """Point every commute setting at the mock and silence the politeness/backoff sleeps."""
    from config import settings
    srv = MockRouting()
    for name, value in {"osrm_drive_url": srv.url + "/car", "osrm_bike_url": srv.url + "/bike", "osrm_foot_url": srv.url + "/foot",
                        "osrm_min_interval_s": 0.0, "osrm_table_max": 90, "commute_drive_factor": 1.0, "otp_base_url": "",
                        "commute_modes": "drive,bike,walk", "census_geocoder_url": srv.url + "/census",
                        "nominatim_base_url": srv.url + "/nominatim", "work_address": ""}.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(commute.time, "sleep", lambda s: None)
    commute.reset_job()
    yield srv
    srv.close()
    commute.reset_job()
