"""General Chat and House Chat see approved changes on their very next turn (real agent code paths)."""
import pytest

import db.schema_catalog as schema
from services import dataset_onboarding as ob


def test_house_chat_gets_the_function_and_prompt_section_immediately(walk_dataset):
    from agents import house_agent as ha
    fns = ha.make_house_approved_functions("h1")
    out = fns["get_linked_dataset_records"](dataset="smart_location_database")
    assert "Joined via" in out and "natwalkind" in out and "42003040100" in out
    prompt = ha._linked_datasets_prompt()
    assert "get_linked_dataset_records" in prompt and "smart_location_database" in prompt and "walkability index" in prompt
    ha._validate_house_program('final_result = get_linked_dataset_records(dataset="smart_location_database")', set(fns))


def test_general_chat_context_and_sql_guards_accept_the_new_table(walk_dataset):
    from agents import general_agent as ga
    from agents.tools import validate_sql
    ctx = ga._get_data_availability_context()
    assert "USER-ADDED DATASETS" in ctx and "smart_location_database" in ctx and "smart_location_database.tract_fips = houses.tract_fips" in ctx
    validate_sql("SELECT AVG(natwalkind) FROM smart_location_database")
    for internal in ("catalog_tables", "stg_abc123", "geocode_cache"):
        with pytest.raises(Exception):
            validate_sql(f"SELECT * FROM {internal}")


def test_nothing_changes_for_the_agents_when_no_dataset_was_added(reference_data):
    from agents import general_agent as ga
    from agents import house_agent as ha
    assert ha._linked_datasets_prompt() == "" and schema.added_datasets_briefing() == ""
    assert "USER-ADDED" not in ga._get_data_availability_context()
    assert ha.make_house_approved_functions("h1")["get_linked_dataset_records"]() == "No user-added datasets are linked to houses yet."


def test_retiring_a_dataset_removes_it_from_both_chats_at_once(walk_dataset):
    from agents import general_agent as ga
    from agents import house_agent as ha
    ob.retire_dataset(walk_dataset["dataset_id"])
    assert ha._linked_datasets_prompt() == "" and "smart_location_database" not in ga._get_data_availability_context()
    assert "No user-added datasets" in ha.make_house_approved_functions("h1")["get_linked_dataset_records"]()


def test_generated_concepts_do_not_trip_the_unplanned_filter_guard(walk_dataset):
    from agents.query_planner import build_query_plan
    from agents.tools import _validate_sql_against_plan
    qp = build_query_plan("show smart location database rows for my houses")
    sql = ("SELECT smart_location_database.natwalkind FROM houses JOIN smart_location_database "
           "ON houses.tract_fips = smart_location_database.tract_fips WHERE natwalkind > 10")
    _validate_sql_against_plan(sql, qp)


class _FakeCollection:
    def __init__(self):
        self.docs = {}

    def get(self, include=None):
        return {"ids": list(self.docs), "documents": list(self.docs.values())}

    def delete(self, ids):
        for i in ids:
            self.docs.pop(i, None)

    def upsert(self, ids, embeddings, documents, metadatas):
        self.docs.update(dict(zip(ids, documents)))


class _FakeEmbeddings:
    def __init__(self):
        self.embedded = 0

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return [[0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, q):
        raise RuntimeError("Ollama is down")


def test_vector_index_syncs_incrementally_and_the_fallback_reads_the_live_catalog(walk_dataset, monkeypatch):
    import db.vector_store as vs
    col, emb = _FakeCollection(), _FakeEmbeddings()
    monkeypatch.setattr(vs, "_schema_collection", lambda: col)
    monkeypatch.setattr(vs, "_get_embeddings", lambda: emb)
    monkeypatch.setattr(vs, "_SYNCED_CATALOG_VERSION", None)
    total = vs.ensure_schema_metadata_index()
    assert emb.embedded == total and any("smart_location_database" in d for d in col.docs.values())
    first = emb.embedded
    vs.ensure_schema_metadata_index()
    assert emb.embedded == first                                    # same catalog version: nothing is re-embedded
    schema.update_table_description("smart_location_database", "Walkability, edited by hand")
    vs.ensure_schema_metadata_index()
    assert 0 < emb.embedded - first <= 3                            # only the changed document(s)
    hits = vs.search_data_model("walkability index of block groups")   # embed_query fails -> lexical fallback
    assert hits and any("smart_location_database" in h["text"] for h in hits)
    ob.retire_dataset(walk_dataset["dataset_id"])
    hits = vs.search_data_model("walkability index of block groups")
    assert not any("smart_location_database" in h["text"] for h in hits)
