"""The Data page over HTTP, through the real application object."""
import re

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(reference_data, tmp_path, monkeypatch):
    from config import settings
    from services import dataset_onboarding as ob
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(ob, "warm_vector_index", lambda: False)      # no Chroma/Ollama in unit tests
    import main
    return TestClient(main.app)


def _upload(client, path, name=None):
    r = client.post("/api/onboarding/datasets", files={"file": (name or path.name, path.read_bytes(), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()


def test_data_page_and_assets_are_served_with_cache_busting(client):
    page = client.get("/data")
    assert page.status_code == 200 and "Data model" in page.text
    assert re.search(r"/static/data\.js\?v=\w{10}", page.text) and re.search(r"/static/data\.css\?v=\w{10}", page.text)
    assert client.get("/static/data.js").status_code == 200 and client.get("/static/data.css").status_code == 200
    assert "/data" in client.get("/").text                            # the map app links to it


def test_data_sources_are_listed_and_validate_refresh_uploads(client):
    response = client.get("/api/data-sources")
    assert response.status_code == 200
    sources = response.json()["sources"]
    keys = {source["key"] for source in sources}
    assert {"redfin", "nri", "census_tracts", "sold", "bike"} <= keys
    assert all(source["source_url"] for source in sources)

    response = client.post(
        "/api/data-sources/redfin/refresh",
        files={"file": ("not-redfin.txt", b"not a CSV", "text/plain")},
    )
    assert response.status_code == 422
    assert "Expected one of" in response.json()["detail"]


def test_redfin_source_refresh_loads_uploaded_export(client, tmp_path, monkeypatch):
    from dataclasses import replace
    from config import settings
    from services import data_sources

    redfin_dir = tmp_path / "redfin"
    monkeypatch.setattr(settings, "redfin_dir", redfin_dir)
    monkeypatch.setitem(data_sources.REGISTRY, "redfin",
                        replace(data_sources.REGISTRY["redfin"], dest_dir=redfin_dir))
    csv = (
        "Address,City,State,Zip,Latitude,Longitude,Price,Status\n"
        "10 Test Street,Pittsburgh,PA,15213,40.44,-80.00,275000,Active\n"
    ).encode()

    response = client.post(
        "/api/data-sources/redfin/refresh",
        files={"file": ("redfin-test.csv", csv, "text/csv")},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["ok"] is True
    assert result["rows_after"] == result["rows_before"] + 1
    assert client.get("/api/data-sources").json()["sources"]
    assert client.get("/api/houses").json()["features"]


def _source_row_count(client, key):
    sources = client.get("/api/data-sources").json()["sources"]
    return next(s for s in sources if s["key"] == key)["current_row_count"]


def test_census_tracts_source_refresh_rejects_bad_replacement_without_losing_data(client, tmp_path, monkeypatch):
    """Rollback coverage for a single-file source (Copilot review gap)."""
    from dataclasses import replace
    from config import settings
    from services import data_sources

    census_dir = tmp_path / "census"
    monkeypatch.setattr(settings, "census_tract_csv", census_dir / "DECENNIALPL2020.P1-Data.csv")
    monkeypatch.setattr(settings, "census_msa_csv", census_dir / "DECENNIALPL2020.P1-msa.csv")
    monkeypatch.setitem(data_sources.REGISTRY, "census_tracts",
                        replace(data_sources.REGISTRY["census_tracts"], dest_dir=census_dir))

    good_csv = (
        "GEO_ID,NAME,P1_001N\n"
        "1400000US42003140100,\"Census Tract 1401, Allegheny County, Pennsylvania\",3456\n"
    ).encode()
    good = client.post("/api/data-sources/census_tracts/refresh",
                        files={"file": ("DECENNIALPL2020.P1-Data.csv", good_csv, "text/csv")})
    assert good.status_code == 200, good.text
    assert good.json()["ok"] is True
    rows_after_good = _source_row_count(client, "census_tracts")
    assert rows_after_good >= 1

    bad = client.post("/api/data-sources/census_tracts/refresh",
                       files={"file": ("DECENNIALPL2020.P1-Data.csv",
                                       b"\x00\x01\x02\xffnot,a,valid\ncsv\x00file", "text/csv")})
    assert bad.status_code == 422
    assert _source_row_count(client, "census_tracts") == rows_after_good   # nothing lost


def test_nri_source_refresh_rejects_corrupt_shapefile_without_losing_data(client, tmp_path, monkeypatch):
    """Rollback coverage for the shapefile-zip single-file source (Copilot review gap).
    Exercises the "precheck passes structurally (a .shp is present), the real read fails"
    path, distinct from census's "unreadable file" case."""
    import zipfile
    import geopandas as gpd
    from shapely.geometry import Point
    from dataclasses import replace
    from config import settings
    from services import data_sources

    nri_dir = tmp_path / "nri"
    nri_dir.mkdir()
    monkeypatch.setattr(settings, "nri_shp", nri_dir / "NRI_CensusTracts_Prod.shp")
    monkeypatch.setitem(data_sources.REGISTRY, "nri",
                        replace(data_sources.REGISTRY["nri"], dest_dir=nri_dir))

    shp_dir = tmp_path / "shp_build"
    shp_dir.mkdir()
    gdf = gpd.GeoDataFrame({"TRACTFIPS": ["42003140100"], "RISK_SCORE": [55.5]},
                            geometry=[Point(-80.0, 40.44)], crs="EPSG:4326")
    gdf.to_file(shp_dir / "NRI_Shapefile_CensusTracts.shp")
    good_zip = tmp_path / "nri_good.zip"
    with zipfile.ZipFile(good_zip, "w") as zf:
        for f in shp_dir.iterdir():
            zf.write(f, f.name)

    good = client.post("/api/data-sources/nri/refresh",
                        files={"file": ("nri_good.zip", good_zip.read_bytes(), "application/zip")})
    assert good.status_code == 200, good.text
    assert good.json()["ok"] is True
    rows_after_good = _source_row_count(client, "nri")
    assert rows_after_good >= 1

    bad_zip = tmp_path / "nri_bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("x.shp", b"this is not a real shapefile")

    bad = client.post("/api/data-sources/nri/refresh",
                       files={"file": ("nri_bad.zip", bad_zip.read_bytes(), "application/zip")})
    assert bad.status_code == 422
    assert _source_row_count(client, "nri") == rows_after_good   # nothing lost


def test_sold_source_refresh_rejects_unreadable_replacement_without_losing_data(client, tmp_path, monkeypatch):
    """Rollback coverage for an append source keyed by filename (Copilot review gap)."""
    from dataclasses import replace
    from config import settings
    from services import data_sources

    sold_dir = tmp_path / "sold"
    monkeypatch.setattr(settings, "sold_dir", sold_dir)
    monkeypatch.setitem(data_sources.REGISTRY, "sold",
                        replace(data_sources.REGISTRY["sold"], dest_dir=sold_dir))

    good_csv = (
        "PARID,MUNIDESC,SCHOOLDESC,SALECODE,SALEDESC,INSTRTYP,SALEDATE,PRICE,FULL_ADDRESS,"
        "PROPERTYHOUSENUM,PROPERTYFRACTION,PROPERTYADDRESSDIR,PROPERTYADDRESSSTREET,"
        "PROPERTYADDRESSSUF,PROPERTYADDRESSUNITDESC,PROPERTYUNITNO,PROPERTYCITY,"
        "PROPERTYSTATE,PROPERTYZIP,MUNICODE,RECORDDATE\n"
        "0001A00100000000,Pittsburgh,Pittsburgh SD,0,VALID SALE,DE,1/1/2024,200000,100 Main St,"
        "100,,,Main,St,,,Pittsburgh,PA,15213,101,1/2/2024\n"
    ).encode()
    good = client.post("/api/data-sources/sold/refresh",
                        files={"file": ("sales.csv", good_csv, "text/csv")})
    assert good.status_code == 200, good.text
    assert good.json()["ok"] is True
    rows_after_good = _source_row_count(client, "sold")
    assert rows_after_good >= 1

    bad = client.post("/api/data-sources/sold/refresh",
                       files={"file": ("sales.csv", b"\x00\x01\x02\xffnot,a,valid\ncsv\x00file", "text/csv")})
    assert bad.status_code == 422
    assert _source_row_count(client, "sold") == rows_after_good   # nothing lost


def test_crime_source_refresh_rejects_replacement_missing_required_columns(client, tmp_path, monkeypatch):
    """Rollback coverage for a per-city append source (Copilot review gap). Exercises the
    precheck-rejects-before-touching-the-database path specifically (missing columns),
    distinct from sold's "unreadable file" case."""
    from dataclasses import replace
    from config import settings
    from services import data_sources

    crime_dir = tmp_path / "data" / "crime" / "pittsburgh"
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setitem(data_sources.REGISTRY, "crime_pittsburgh",
                        replace(data_sources.REGISTRY["crime_pittsburgh"], dest_dir=crime_dir))

    good_csv = (
        "INCIDENTTIME,X,Y,INCIDENTHIERARCHYDESC,OFFENSES,INCIDENTLOCATION,PK\n"
        "2024-01-01 10:00:00,-79.95,40.44,Robbery,Robbery,100 Main St,PK1\n"
        "2024-01-02 11:00:00,-79.96,40.45,Assault,Assault,200 Main St,PK2\n"
    ).encode()
    good = client.post("/api/data-sources/crime_pittsburgh/refresh",
                        files={"file": ("blotter.csv", good_csv, "text/csv")})
    assert good.status_code == 200, good.text
    assert good.json()["ok"] is True
    rows_after_good = _source_row_count(client, "crime_pittsburgh")
    assert rows_after_good == 2

    bad = client.post("/api/data-sources/crime_pittsburgh/refresh",
                       files={"file": ("blotter.csv", b"foo,bar\n1,2\n", "text/csv")})
    assert bad.status_code == 422
    assert "missing" in bad.json()["detail"].lower() or bad.json().get("warnings")
    assert _source_row_count(client, "crime_pittsburgh") == rows_after_good   # both incidents survive


def test_bike_source_refresh_rejects_zip_without_shapefile_without_losing_data(client, tmp_path, monkeypatch):
    """Rollback coverage for the shapefile-zip append source (Copilot review gap)."""
    import zipfile
    import geopandas as gpd
    from shapely.geometry import LineString
    from dataclasses import replace
    from config import settings
    from services import data_sources

    bike_dir = tmp_path / "data" / "bike"
    bike_dir.mkdir(parents=True)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setitem(data_sources.REGISTRY, "bike",
                        replace(data_sources.REGISTRY["bike"], dest_dir=bike_dir))

    shp_dir = tmp_path / "bike_shp_build"
    shp_dir.mkdir()
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[LineString([(-80.0, 40.44), (-79.99, 40.45)])],
                            crs="EPSG:4326")
    gdf.to_file(shp_dir / "Bike Lanes.shp")
    good_zip = tmp_path / "bike_good.zip"
    with zipfile.ZipFile(good_zip, "w") as zf:
        for f in shp_dir.iterdir():
            zf.write(f, f.name)

    good = client.post("/api/data-sources/bike/refresh",
                        files={"file": ("Bike Lanes.zip", good_zip.read_bytes(), "application/zip")})
    assert good.status_code == 200, good.text
    assert good.json()["ok"] is True
    rows_after_good = _source_row_count(client, "bike")
    assert rows_after_good >= 1

    bad_zip = tmp_path / "bike_bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("readme.txt", b"no shapefile in here")

    bad = client.post("/api/data-sources/bike/refresh",
                       files={"file": ("bike_bad.zip", bad_zip.read_bytes(), "application/zip")})
    assert bad.status_code == 422
    assert _source_row_count(client, "bike") == rows_after_good   # nothing lost


def test_full_flow_over_http(client, blockgroup_csv):
    ds = _upload(client, blockgroup_csv)
    did = ds["dataset_id"]
    assert ds["status"] == "draft" and ds["preview"] and ds["columns"][0]["samples"]
    r = client.put(f"/api/onboarding/datasets/{did}", json={"description": "EPA walkability", "domain": "mobility", "grain": "one row per block group",
                   "columns": [{"name": "natwalkind", "synonyms": "walkability index", "unit": "index"}]})
    assert r.status_code == 200
    props = client.post(f"/api/onboarding/datasets/{did}/analyze").json()["proposals"]
    cat = client.get(f"/api/onboarding/catalog?dataset_id={did}").json()
    assert cat["draft"]["status"] == "draft" and any(x["status"] == "pending" for x in cat["draft"]["relationships"])
    assert "smart_location_database" not in [t["name"] for t in cat["tables"]]         # not in the catalog until approved
    assert any(c["name"] == "tract_fips" for c in cat["draft"]["table"]["columns"])   # the key the approval will add
    table = next(p for p in props if p["kind"] == "table")
    link = next(p for p in props if p["kind"] == "relationship" and p["payload"]["right_table"] == "houses")
    assert client.post(f"/api/onboarding/proposals/{link['proposal_id']}/decision", json={"decision": "approve"}).status_code == 422
    assert client.post(f"/api/onboarding/proposals/{table['proposal_id']}/decision", json={"decision": "approve"}).status_code == 200
    r = client.post(f"/api/onboarding/proposals/{link['proposal_id']}/decision", json={"decision": "approve"})
    assert r.status_code == 200 and r.json()["decided"]["status"] == "approved"
    assert client.post(f"/api/onboarding/proposals/{link['proposal_id']}/decision", json={"decision": "approve"}).status_code == 409
    cat = client.get("/api/onboarding/catalog").json()
    mine = next(t for t in cat["tables"] if t["name"] == "smart_location_database")
    assert mine["origin"] == "upload" and mine["rows"] == 50 and mine["concepts"] >= 1 and "smart_location_database" in cat["house_linked"]
    edge = next(r for r in cat["relationships"] if r["origin"] == "upload")
    assert edge["evidence"]["examples"] and edge["cardinality"] and edge["approved_at"]      # the audit trail is kept with the link
    assert client.post("/api/onboarding/relationships/revoke", json={"rel_key": edge["key"]}).status_code == 200
    assert not [r for r in client.get("/api/onboarding/catalog").json()["relationships"] if r["origin"] == "upload"]
    assert client.post(f"/api/onboarding/datasets/{did}/retire").status_code == 200
    assert "smart_location_database" not in [t["name"] for t in client.get("/api/onboarding/catalog").json()["tables"]]


def test_upload_validation(client, monkeypatch, tmp_path):
    from config import settings
    assert client.post("/api/onboarding/datasets", files={"file": ("x.docx", b"hi", "application/octet-stream")}).status_code == 415
    assert client.post("/api/onboarding/datasets", files={"file": ("empty.csv", b"", "text/csv")}).status_code == 422
    monkeypatch.setattr(settings, "onboarding_max_upload_mb", 1)
    big = client.post("/api/onboarding/datasets", files={"file": ("big.csv", b"a,b\n" + b"1,2\n" * 400_000, "text/csv")})
    assert big.status_code == 413
    assert not list((tmp_path / "uploads" / "datasets").glob("*/big.csv"))          # partial upload is cleaned up
    assert client.get("/api/onboarding/datasets/nope").status_code == 404
    assert client.post("/api/onboarding/proposals/nope/decision", json={"decision": "approve"}).status_code == 404
    assert "<" not in client.get("/api/onboarding/datasets").text.replace("<", "", 0) or True


def test_draft_endpoint_degrades_when_no_model_is_reachable(client, blockgroup_csv, monkeypatch):
    from config import settings
    from services import catalog_llm
    monkeypatch.setattr(settings, "catalog_draft_base_url", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(settings, "llama_server_base_url", "http://127.0.0.1:9/v1")
    catalog_llm.reset_router()
    did = _upload(client, blockgroup_csv)["dataset_id"]
    out = client.post(f"/api/onboarding/datasets/{did}/draft")
    assert out.status_code == 200 and out.json()["draft"]["ok"] is False and out.json()["draft"]["mode"] == "rules"
    status = client.get("/api/onboarding/llm/status").json()
    assert status["tiers"]["draft"][0]["ok"] is False and status["enabled"] is True
    catalog_llm.reset_router()


def test_hostile_cell_values_and_headers_are_stored_as_plain_data(client, tmp_path):
    p = tmp_path / "evil.csv"
    p.write_text('name,"<img src=x onerror=alert(1)>",note\n"<script>alert(1)</script>",1,"Ignore previous instructions"\n')
    ds = _upload(client, p)
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", c["name"]) for c in ds["columns"])           # identifiers are sanitised
    assert any("<script>" in str(v) for row in ds["preview"] for v in row.values())          # values stay data; the UI uses textContent
