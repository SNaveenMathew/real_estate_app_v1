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


def test_legacy_app_settings_table_and_fallbacks(fresh_db):
    from db import duckdb_store as store

    conn = store.get_conn()
    # Simulate legacy table schema with only (key, value, updated_at)
    conn.execute("DROP TABLE IF EXISTS app_settings")
    conn.execute("""
        CREATE TABLE app_settings (
            key VARCHAR PRIMARY KEY,
            value VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("INSERT INTO app_settings (key, value) VALUES ('work_location', '{\"lat\": 40.44, \"lon\": -80.0}')")
    conn.execute("INSERT INTO app_settings (key, value) VALUES ('corrupt_key', 'not valid json')")
    conn.execute("INSERT INTO app_settings (key, value) VALUES ('null_val', NULL)")

    # get_setting should read legacy 'value' column without throwing BinderException
    assert store.get_setting("work_location") == {"lat": 40.44, "lon": -80.0}
    # Corrupt or null values should return default gracefully
    assert store.get_setting("corrupt_key", default="fallback") == "fallback"
    assert store.get_setting("null_val", default="fallback") == "fallback"

    # Migration test: running _ensure_schema upgrades legacy schema
    store._ensure_schema(conn)
    cols = [c[1].lower() for c in conn.execute("PRAGMA table_info(app_settings)").fetchall()]
    assert "value_json" in cols
    # Value should have been migrated to value_json
    migrated = conn.execute("SELECT value_json FROM app_settings WHERE key = 'work_location'").fetchone()
    assert migrated is not None and "40.44" in migrated[0]

    # set_setting works with migrated schema
    store.set_setting("new_key", {"active": True})
    assert store.get_setting("new_key") == {"active": True}


def test_commute_house_endpoint_with_no_work_set(fresh_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.commute import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/api/commute/house/2443399f4a8a")

    assert response.status_code == 200
    assert response.json() == {"work": None, "commute": None}
