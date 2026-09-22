"""The Commute tab's HTTP API (a mini app around the router, so the background job's event loop persists)."""
import re
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.commute import router
from commute_helpers import WORK, add_houses, routing  # noqa: F401


def _wait(client, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get("/api/commute/status").json()["job"]
        if job["state"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_commute_api_flow(routing, fresh_db):
    add_houses(fresh_db.get_conn(), 4)
    fresh_db.get_conn().execute("INSERT INTO houses (house_id, address, lat, lon) VALUES ('hfar', 'Far St', ?, ?)", [WORK[0] + 0.3, WORK[1]])
    routing.census["500 Grant St, Pittsburgh, PA"] = ("500 GRANT ST, PITTSBURGH, PA, 15219", WORK[1], WORK[0])
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:                                       # the context keeps one event loop for background jobs
        cfg = c.get("/api/commute/config").json()
        assert cfg["configured"] is False and cfg["modes"] == ["drive", "bike", "walk"] and cfg["counts"]["houses"] == 5
        assert c.post("/api/commute/refresh", json={}).status_code == 409                       # nothing to compute yet
        bad = c.put("/api/commute/work", json={"address": "nowhere at all"})
        assert bad.status_code == 422 and "Could not find" in bad.json()["detail"]
        assert c.put("/api/commute/work", json={}).status_code == 422
        assert c.put("/api/commute/work", json={"lat": 91, "lon": 0}).status_code == 422
        assert c.put("/api/commute/work", json={"lat": "abc", "lon": 0}).status_code == 422
        ok = c.put("/api/commute/work", json={"address": "500 Grant St, Pittsburgh, PA"})
        assert ok.status_code == 200 and ok.json()["work"]["source"] == "census" and ok.json()["job"]["state"] == "running"
        job = _wait(c)
        assert (job["state"], job["done"], job["failed"]) == ("done", 5, 0)
        summary = c.get("/api/commute/summary").json()
        assert len(summary["rows"]) == 5 and summary["work"]["label"].startswith("500 GRANT")
        assert summary["rows"]["hfar"]["walk_min"] is None                                        # a NULL mode must not break the JSON
        assert summary["rows"]["h1"]["drive_min"] < summary["rows"]["h4"]["drive_min"]           # nearer house, shorter drive
        one = c.get("/api/commute/house/h1").json()
        assert one["commute"]["fresh"] is True and "work_key" not in one["commute"] and one["work"]["source"] == "census"
        cfg = c.get("/api/commute/config").json()
        assert cfg["configured"] and cfg["counts"]["fresh"] == 5 and cfg["counts"]["to_compute"] == 0
        assert all(p["public"] is False for p in cfg["providers"].values())                       # the mock is on localhost
        pin = c.put("/api/commute/work", json={"lat": WORK[0] + 0.02, "lon": WORK[1], "label": "Map pin"}).json()
        assert pin["work"]["source"] == "map" and _wait(c)["done"] == 5
        assert c.post("/api/commute/refresh", json={"scope": "bogus"}).status_code == 422
        assert c.post("/api/commute/refresh", json={"scope": "all"}).status_code == 200
        _wait(c)
        cleared = c.delete("/api/commute/work").json()
        assert cleared["configured"] is False and cleared["counts"]["fresh"] == 0
        assert c.get("/api/commute/summary").json() == {"work": None, "rows": {}}
        assert c.get("/api/commute/house/h1").json() == {"work": None, "commute": None}


def test_provider_outage_is_reported_in_the_job(routing, fresh_db):
    add_houses(fresh_db.get_conn(), 3)
    routing.down = {"bike"}
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        c.put("/api/commute/work", json={"lat": WORK[0], "lon": WORK[1], "label": "Office"})
        job = _wait(c)
        assert job["state"] == "done" and job["failed"] == 3 and any(w.startswith("Bike:") for w in job["warnings"])
        row = c.get("/api/commute/house/h1").json()["commute"]
        assert row["status"] == "partial" and row["drive_min"] is not None and row["bike_min"] is None


def test_the_real_app_serves_the_commute_tab_script_and_api(routing, fresh_db):
    import main
    c = TestClient(main.app)
    html = c.get("/").text
    assert 'data-tab="commute"' in html and 'id="tab-commute"' in html and 'id="commute-content"' in html
    assert re.search(r"/static/commute\.js\?v=\w{8}", html)                # cache-busted with the app's static version
    assert c.get("/static/commute.js").status_code == 200
    assert c.get("/api/commute/config").json()["configured"] is False        # the router is mounted on the real app
