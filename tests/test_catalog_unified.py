"""One catalog for built-in and uploaded data: seeded, persisted, transactional, hot-reloaded."""
import pytest

import db.schema_catalog as schema
from db import catalog_seed as seed
from db import catalog_store
from db.catalog_model import AVG, Relationship, TableMeta, _concept


def test_builtin_definitions_are_seeded_into_the_store_and_loaded_in_order(fresh_db):
    conn = fresh_db.get_conn()
    assert [r[0] for r in conn.execute("SELECT name FROM catalog_tables ORDER BY ord").fetchall()] == list(seed.TABLES)
    assert list(schema.TABLES) == list(seed.TABLES)
    assert [r.key() for r in schema.RELATIONSHIPS] == [r.key() for r in seed.RELATIONSHIPS]
    assert list(schema.SEMANTIC_GLOSSARY) == list(seed.SEMANTIC_GLOSSARY)
    assert [d.name for d in schema.ENTITY_DOMAINS] == [d.name for d in seed.ENTITY_DOMAINS]
    assert {m.origin for m in schema.TABLES.values()} == {"builtin"}


def test_seed_sync_is_idempotent(fresh_db):
    conn = fresh_db.get_conn()
    version = catalog_store.get_version(conn)
    assert catalog_store.sync_seed(conn) is False
    assert catalog_store.get_version(conn) == version


def test_edited_rows_survive_a_seed_refresh_but_untouched_rows_are_refreshed(fresh_db):
    conn = fresh_db.get_conn()
    schema.update_table_description("houses", "My own words about houses")
    conn.execute("UPDATE catalog_tables SET seed_hash = 'stale' WHERE name IN ('houses', 'nri_tracts')")
    conn.execute("UPDATE catalog_tables SET description = 'stale text' WHERE name = 'nri_tracts'")
    assert catalog_store.sync_seed(conn) is True
    schema.reload()
    assert schema.TABLES["houses"].description == "My own words about houses"
    assert schema.TABLES["nri_tracts"].description == seed.TABLES["nri_tracts"].description


def test_registry_containers_are_refreshed_in_place(fresh_db):
    held_tables, held_rels = schema.TABLES, schema.RELATIONSHIPS
    schema.register_table(TableMeta(name="extra_data", description="x"), origin="upload", dataset_id="d1")
    schema.register_relationship(Relationship("extra_data", "tract_fips", "houses", "tract_fips", cardinality="many-to-one"),
                                 origin="upload", dataset_id="d1")
    assert schema.TABLES is held_tables and "extra_data" in held_tables       # references held elsewhere stay valid
    assert schema.RELATIONSHIPS is held_rels and held_rels[-1].key() == "extra_data:tract_fips=houses:tract_fips"
    assert schema.TABLES["extra_data"].origin == "upload"


def test_a_failed_change_is_rolled_back_and_never_visible(fresh_db):
    before = list(schema.TABLES)
    with pytest.raises(RuntimeError):
        with schema.catalog_transaction():
            schema.register_table(TableMeta(name="half_done", description="x"), origin="upload")
            fresh_db.get_conn().execute("CREATE TABLE half_done (a INTEGER)")
            raise RuntimeError("boom")
    assert list(schema.TABLES) == before
    conn = fresh_db.get_conn()
    assert conn.execute("SELECT COUNT(*) FROM catalog_tables WHERE name = 'half_done'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'half_done'").fetchone()[0] == 0


def test_changes_persist_across_reconnects_and_the_registry_follows_the_database(fresh_db, tmp_path, monkeypatch):
    from config import settings
    schema.register_table(TableMeta(name="extra_data", description="x"), origin="upload", dataset_id="d1")
    fresh_db.close()
    fresh_db.get_conn()
    assert "extra_data" in schema.TABLES                       # same file: still there after a restart
    monkeypatch.setattr(settings, "duckdb_path", tmp_path / "other.duckdb")   # like the eval harness's fixture DB
    fresh_db.close()
    assert "extra_data" not in schema.TABLES and "houses" in schema.TABLES


def test_revoke_and_retire_remove_objects_from_the_agents_view(fresh_db):
    schema.register_table(TableMeta(name="extra_data", description="x"), origin="upload", dataset_id="d1")
    rel = Relationship("extra_data", "tract_fips", "houses", "tract_fips", cardinality="many-to-one")
    schema.register_relationship(rel, origin="upload", dataset_id="d1")
    assert schema.relationship_path({"houses", "extra_data"})
    assert schema.revoke_relationship(rel.key(), "test") is True
    assert not schema.relationship_path({"houses", "extra_data"})
    schema.register_relationship(rel, origin="upload", dataset_id="d1")
    counts = schema.retire_dataset_objects("d1")
    assert counts["tables"] == 1 and "extra_data" not in schema.list_table_names()
    assert not schema.relationship_path({"houses", "extra_data"})


def test_a_registered_concept_is_matched_by_the_planner_immediately(fresh_db):
    schema.register_table(TableMeta(name="walk_data", description="w"), origin="upload", dataset_id="d1")
    concept = _concept("walk_data_idx", ["walk_data"], ["walkability index"], "walk", columns=("walk_data.idx",),
                       operations=(AVG("walk_data.idx"),), grain="row", default_operation="avg")
    schema.register_concept("walk_data_idx", concept, dataset_id="d1")
    keys = [m["key"] for m in schema.semantic_matches("what is the average walkability index")]
    assert "walk_data_idx" in keys
    schema.retire_dataset_objects("d1")
    assert "walk_data_idx" not in [m["key"] for m in schema.semantic_matches("what is the average walkability index")]


def test_planner_and_prompts_are_unchanged_for_the_built_in_catalog(fresh_db):
    """Regression guard for the refactor: headers the response validator and agents key on still exist."""
    text = schema.render_schema_for_agent()
    assert "RELATIONSHIPS" in text and "_relationships()" not in text
    assert schema.diagnose_empty_or_error("SELECT * FROM houses JOIN nri_tracts ON 1=1").startswith("EMPTY TABLES:")
