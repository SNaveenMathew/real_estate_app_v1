def test_settings_are_json_backed_and_persisted(fresh_db):
    from db import duckdb_store as store

    assert store.get_setting("missing", default={"fallback": True}) == {"fallback": True}

    value = {"lat": 40.44, "lon": -80.0, "label": "Home", "tags": ["primary"]}
    store.set_setting("work_location", value)
    assert store.get_setting("work_location") == value

    replacement = {"lat": 41.0, "lon": -81.0}
    store.set_setting("work_location", replacement)
    assert store.get_setting("work_location") == replacement

    store.delete_setting("work_location")
    assert store.get_setting("work_location") is None


def test_commute_config_works_on_a_fresh_database(fresh_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.commute import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/api/commute/config")

    assert response.status_code == 200
    assert response.json()["configured"] is False