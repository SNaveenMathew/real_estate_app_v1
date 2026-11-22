"""Uploads are reachable from 0.0.0.0: a client filename is data, not a path, and an
image's type comes from its bytes.

Covers api/uploads.py directly and the two endpoints that accept files through the real
application object (the Data page's dataset upload and a house photo).
"""
import io
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from api.uploads import sanitize_filename, save_stream, sniff_image_extension


@pytest.fixture()
def client(reference_data, tmp_path, monkeypatch):
    from config import settings
    import db.vector_store as vs
    from services import dataset_onboarding as ob

    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(ob, "warm_vector_index", lambda: False)          # no Chroma/Ollama in unit tests
    monkeypatch.setattr(vs, "add_photo", lambda house_id, path, caption="": f"photo-{path.name}")
    import main
    return TestClient(main.app)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


# ── api/uploads.py ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("hostile", [
    "../../etc/passwd",
    "..\\..\\Windows\\evil.csv",
    "C:\\temp\\report.csv",
    "a\"b';c.csv",
    "  ..csv  ",
    "/absolute/path.csv",
])
def test_a_client_filename_never_becomes_a_path(hostile):
    name = sanitize_filename(hostile)
    assert name not in ("", ".", "..")
    assert Path(name).name == name and "/" not in name and "\\" not in name
    assert not name.startswith(".")


def test_sanitize_filename_handles_empty_reserved_and_overlong_names():
    assert sanitize_filename(None) == "upload"
    assert sanitize_filename("") == "upload"
    assert sanitize_filename(".") == "upload"
    assert sanitize_filename(".csv") == "csv"                 # no stem: the caller's allow-list rejects it
    assert sanitize_filename("CON.csv") == "_CON.csv"         # still a Windows device name with an extension
    assert len(sanitize_filename("x" * 400 + ".csv")) <= 104


def test_save_stream_removes_a_partial_file_when_the_cap_is_hit(tmp_path):
    dest = tmp_path / "nested" / "big.csv"
    with pytest.raises(HTTPException) as err:
        save_stream(io.BytesIO(b"a" * 64), dest, max_bytes=32)
    assert err.value.status_code == 413
    assert not dest.exists()                                   # nothing truncated left on disk


def test_save_stream_rejects_an_empty_upload(tmp_path):
    dest = tmp_path / "empty.csv"
    with pytest.raises(HTTPException) as err:
        save_stream(io.BytesIO(b""), dest, max_bytes=1024)
    assert err.value.status_code == 422 and not dest.exists()


def test_sniff_image_extension_reads_the_bytes_not_the_name(tmp_path):
    real = tmp_path / "really-a-png.php"
    real.write_bytes(_png_bytes())
    assert sniff_image_extension(real) == ".png"

    fake = tmp_path / "really-a-png.png"
    fake.write_bytes(b"<?php system($_GET['c']); ?>")
    with pytest.raises(HTTPException) as err:
        sniff_image_extension(fake)
    assert err.value.status_code == 415

    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(_png_bytes()[:40])
    with pytest.raises(HTTPException) as err:
        sniff_image_extension(truncated)
    assert err.value.status_code == 415


# ── house photo endpoint ─────────────────────────────────────────────────────

def test_photo_upload_stays_inside_uploads_and_names_the_file_from_its_bytes(client, tmp_path):
    response = client.post(
        "/api/house/h1/photo",
        files={"file": ("../../escape.php", _png_bytes(), "image/png")},
        data={"caption": "front elevation"},
    )
    assert response.status_code == 200, response.text

    uploads = (tmp_path / "uploads").resolve()
    stored = Path(response.json()["path"]).resolve()
    assert uploads in stored.parents                     # no traversal out of the upload directory
    assert stored.suffix == ".png"                       # from the decoded bytes, not the client's name
    assert stored.read_bytes() == _png_bytes()
    assert not (tmp_path / "escape.php").exists()


def test_photo_upload_rejects_a_file_that_is_not_an_image(client, tmp_path):
    response = client.post(
        "/api/house/h1/photo",
        files={"file": ("front.png", b"<?php system($_GET['c']); ?>", "image/png")},
    )
    assert response.status_code == 415
    uploads = tmp_path / "uploads"
    assert not [p for p in uploads.rglob("*") if p.is_file()] if uploads.exists() else True


def test_photo_upload_enforces_its_own_size_cap(client, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "photo_max_upload_mb", 1)
    response = client.post(
        "/api/house/h1/photo",
        files={"file": ("big.png", b"\x89PNG\r\n\x1a\n" + b"0" * (1024 * 1024 + 16), "image/png")},
    )
    assert response.status_code == 413


def test_photo_upload_for_an_unknown_house_creates_nothing(client, tmp_path):
    response = client.post("/api/house/nope/photo", files={"file": ("p.png", _png_bytes(), "image/png")})
    assert response.status_code == 404
    assert not (tmp_path / "uploads").exists() or not list((tmp_path / "uploads").rglob("*.png"))


# ── dataset upload endpoint ──────────────────────────────────────────────────

def _wait_job(client, job_id, timeout=120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["state"] in ("done", "error"):
            return snap
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_dataset_upload_sanitizes_the_client_filename(client, tmp_path):
    csv = b"Address,City,State,Zip\n10 Test Street,Pittsburgh,PA,15213\n"
    response = client.post(
        "/api/onboarding/datasets",
        files={"file": ("../../evil.csv", csv, "text/csv")},
    )
    assert response.status_code == 202, response.text
    assert _wait_job(client, response.json()["job_id"])["state"] == "done"
    assert not (tmp_path / "evil.csv").exists()
    uploads = (tmp_path / "uploads").resolve()
    for stored in (tmp_path / "uploads").rglob("*"):
        if stored.is_file():
            assert uploads in stored.resolve().parents
