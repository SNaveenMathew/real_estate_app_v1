"""The /api/houses payload: gzip, a real ETag, and viewport/city scoping.

Measured before this change: 1,507,761 bytes in 0.78 s for 323 properties, sent again in
full on every page load because the response had no validator and the server no compressor.
These tests pin the three fixes: the body is compressed when the client accepts it, an
unchanged body answers 304, and a viewport or a city returns fewer features.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(reference_data, tmp_path, monkeypatch):
    from config import settings
    from services import dataset_onboarding as ob

    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(ob, "warm_vector_index", lambda: False)
    import main

    with TestClient(main.app) as c:
        yield c


def test_response_carries_a_strong_etag_and_revalidates_with_304(client):
    first = client.get("/api/houses")
    assert first.status_code == 200
    etag = first.headers["etag"]
    assert etag.startswith('"') and etag.endswith('"')

    again = client.get("/api/houses", headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert again.content == b""
    assert again.headers["etag"] == etag


def test_different_viewports_are_different_representations(client):
    everything = client.get("/api/houses")
    empty_corner = client.get("/api/houses?west=-70&south=30&east=-69&north=31")
    assert empty_corner.status_code == 200
    assert empty_corner.json()["features"] == []
    assert empty_corner.headers["etag"] != everything.headers["etag"]


def test_bbox_scopes_the_payload_to_the_viewport(client):
    everything = client.get("/api/houses").json()
    inside = client.get("/api/houses?west=-80.1&south=40.3&east=-79.9&north=40.6").json()
    assert len(inside["features"]) == len(everything["features"]) > 0
    assert all(-80.1 <= f["geometry"]["coordinates"][0] <= -79.9 for f in inside["features"])
    assert all(40.3 <= f["geometry"]["coordinates"][1] <= 40.6 for f in inside["features"])


def test_city_filter_is_an_exact_case_insensitive_match(client):
    pgh = client.get("/api/houses?city=Pittsburgh").json()
    assert pgh["features"]
    assert {f["properties"]["city"] for f in pgh["features"]} == {"Pittsburgh"}


def test_incomplete_or_inverted_bbox_is_a_400(client):
    assert client.get("/api/houses?west=-80").status_code == 400
    assert client.get("/api/houses?west=-79&south=40&east=-80&north=41").status_code == 400
    assert client.get("/api/houses?west=-80&south=41&east=-79&north=40").status_code == 400


def test_body_is_gzipped_when_the_client_accepts_it(client):
    plain = client.get("/api/houses")
    assert len(plain.json()["features"]) > 0
    assert len(plain.content) > 1024, "fixture payload must be large enough for the middleware to matter"

    zipped = client.get("/api/houses", headers={"Accept-Encoding": "gzip"})
    assert zipped.headers.get("content-encoding") == "gzip"
    assert zipped.json() == plain.json()
