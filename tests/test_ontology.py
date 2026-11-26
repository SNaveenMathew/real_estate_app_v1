"""The ontology is complete over the live catalog, structurally valid, and extensible by upload.

These tests are about the *contracts* - every catalog object appears, no edge dangles, the three
granularities return only what they promise, and a dataset approved on the Data page joins the
ontology with no code change - not about a particular seed's contents.
"""

from fastapi.testclient import TestClient

import db.catalog_store as catalog_store
import db.duckdb_store as store
import db.ontology as ontology
import db.schema_catalog as schema
from db import catalog_lint
from db.catalog_model import Relationship
from db import transforms as tx


def _ids(doc: dict) -> set[str]:
    return {n["id"] for n in doc["classes"] + doc["properties"]}


def _concept_definition(key: str, alias: str, **extra) -> dict:
    """A minimal but complete concept definition, as the catalog's DSL would write it."""
    return {
        "key": key,
        "tables": ["houses"],
        "columns": ["houses.walk_score"],
        "aliases": [alias],
        "description": f"{alias} as declared by a test.",
        "operations": [{"op": "avg", "aliases": ["average"], "expr": "AVG(houses.walk_score)"}],
        "filters": [],
        "null_policy": "",
        "orderings": [],
        "groupings": [],
        "grain": "house",
        "entity_types": [],
        "rollup": False,
        "rollup_spec": {},
        "required_terms": [],
        "excluded_terms": [],
        "default_operation": "avg",
        **extra,
    }


# ---------------------------------------------------------------------------
# Coverage of the live catalog
# ---------------------------------------------------------------------------


def test_ontology_covers_every_catalog_table_column_concept_domain_and_join(reference_data):
    doc = ontology.build_ontology()
    ids = _ids(doc)

    for table in schema._tables():
        assert f"table:{table}" in ids
        for column, _type in schema._live_columns(table):
            assert f"column:{table}.{column}" in ids

    for key in schema._glossary():
        assert f"concept:{key}" in ids

    for domain in schema._entity_domains():
        assert f"entity_type:{domain.entity_type}" in ids
        assert f"column:{domain.table}.{domain.column}" in ids

    for rel in schema._relationships():
        assert any(edge["predicate"] == "joins" and edge["key"] == rel.key() for edge in doc["edges"])

    for key in schema.DOMAIN_LABELS:
        assert f"domain:{key}" in ids


def test_identifiers_are_unique_and_every_edge_endpoint_exists(reference_data):
    doc = ontology.build_ontology(with_counts=True)
    nodes = doc["classes"] + doc["properties"]

    assert len(nodes) == len({n["id"] for n in nodes})
    assert len(nodes) == len({n["iri"] for n in nodes})

    ids = _ids(doc)
    for edge in doc["edges"]:
        assert edge["source"] in ids
        assert edge["target"] in ids

    # The fixtures are internally consistent, so structural validation finds nothing to report.
    assert [i for i in doc["issues"] if i["severity"] == "error"] == []


def test_every_measure_points_at_a_real_column(reference_data):
    doc = ontology.build_ontology()
    ids = _ids(doc)
    measures = [e for e in doc["edges"] if e["predicate"] == "measures"]
    assert measures
    assert all(e["target"] in ids for e in measures)
    assert not [i for i in doc["issues"] if i["code"] == "unresolved_measure"]


# ---------------------------------------------------------------------------
# Granularity
# ---------------------------------------------------------------------------


def test_granularity_levels_add_detail_without_dangling_edges(reference_data):
    overview = ontology.build_graph("overview")
    assert {n["kind"] for n in overview["nodes"]} <= {"domain", "dataset", "concept", "entity_type"}
    assert not [e for e in overview["edges"] if e["predicate"] in {"joins", "measures", "in_table"}]

    standard = ontology.build_graph("standard")
    assert any(n["kind"] == "table" for n in standard["nodes"])
    assert any(n["kind"] == "concept" for n in standard["nodes"])
    assert not any(n["kind"] == "column" for n in standard["nodes"])
    assert any(e["predicate"] == "joins" for e in standard["edges"])

    full = ontology.build_graph("full")
    assert any(n["kind"] == "column" for n in full["nodes"])
    assert any(e["predicate"] == "measures" for e in full["edges"])

    assert overview["counts"]["nodes"] < standard["counts"]["nodes"] < full["counts"]["nodes"]
    for level in (overview, standard, full):
        ids = {n["id"] for n in level["nodes"]}
        assert all(e["source"] in ids and e["target"] in ids for e in level["edges"])


def test_an_unknown_granularity_falls_back_to_standard(reference_data):
    model = ontology.build_graph("nonsense")
    assert model["detail"] == "standard"


# ---------------------------------------------------------------------------
# Declared subsumption (the extensible part of the hierarchy)
# ---------------------------------------------------------------------------


def test_a_declared_broader_relation_becomes_a_subclass_edge(reference_data):
    schema.register_concept(
        "walkability_index",
        _concept_definition("walkability_index", "walkability index", broader=["house_walk_score"]),
    )

    doc = ontology.build_ontology()
    edge = next(
        e for e in doc["edges"] if e["predicate"] == "subclass_of" and e["source"] == "concept:walkability_index"
    )
    assert edge["target"] == "concept:house_walk_score"

    # Without a declaration there is no invented hierarchy between the same two concepts.
    assert not any(e["predicate"] == "subclass_of" and e["source"] == "concept:house_walk_score" for e in doc["edges"])


def test_lint_rejects_a_broader_that_names_nothing(reference_data):
    schema.register_concept(
        "mystery_index",
        _concept_definition("mystery_index", "mystery index", broader=["no_such_concept"]),
        origin="manual",
    )
    issues = catalog_lint.lint_catalog()
    assert any(i.code == "unknown_broader" and i.severity == "warning" for i in issues)


def test_a_broader_cycle_is_reported_not_silently_accepted(reference_data):
    schema.register_concept("cycle_a", _concept_definition("cycle_a", "cycle a", broader=["cycle_b"]))
    schema.register_concept("cycle_b", _concept_definition("cycle_b", "cycle b", broader=["cycle_a"]))

    doc = ontology.build_ontology()
    cycles = [i for i in doc["issues"] if i["code"] == "subclass_cycle"]
    assert {i["subject"] for i in cycles} == {"concept:cycle_a", "concept:cycle_b"}


# ---------------------------------------------------------------------------
# Extensibility: an approved upload needs no ontology code change
# ---------------------------------------------------------------------------


def test_an_approved_dataset_enters_the_ontology(walk_dataset):
    table = walk_dataset.get("table_name") or ""
    assert table, walk_dataset

    doc = ontology.build_ontology()
    ids = _ids(doc)
    assert f"table:{table}" in ids
    assert f"dataset:{walk_dataset['dataset_id']}" in ids
    assert any(e["predicate"] == "in_dataset" and e["source"] == f"table:{table}" for e in doc["edges"])
    # The upload's own columns are properties with the notes the user gave them.
    assert any(n["kind"] == "column" and n["table"] == table and n["unit"] == "index" for n in doc["properties"])

    # A link the user approves on the Data page becomes a join edge, through the same catalog API
    # the page uses; no ontology code knows this dataset exists.
    columns = {n["column"] for n in doc["properties"] if n["table"] == table}
    join_column = "tract_fips" if "tract_fips" in columns else sorted(columns)[0]
    schema.register_relationship(
        Relationship(table, join_column, "nri_tracts", "tract_fips", cardinality="many-to-one"),
        origin="upload",
        dataset_id=walk_dataset["dataset_id"],
    )
    linked = ontology.build_ontology()
    assert any(
        e["predicate"] == "joins" and e["source"] == f"table:{table}" and e["target"] == "table:nri_tracts"
        for e in linked["edges"]
    )
    assert any(
        n["kind"] == "dataset" and n["dataset_id"] == walk_dataset["dataset_id"] and n["tables"] for n in doc["classes"]
    )


# ---------------------------------------------------------------------------
# HTTP surface + page
# ---------------------------------------------------------------------------


def test_the_api_serves_the_ontology_jsonld_and_each_granularity(reference_data):
    import main

    client = TestClient(main.app)

    doc = client.get("/api/ontology").json()
    assert doc["stats"]["tables"] and doc["stats"]["concepts"] and doc["stats"]["columns"]
    assert doc["namespace"] == "urn:rei:"

    linked = client.get("/api/ontology?format=jsonld").json()
    assert "@graph" in linked and "@context" in linked
    assert all("@id" in item for item in linked["@graph"])

    for detail in ("overview", "standard", "full"):
        model = client.get(f"/api/ontology/graph?detail={detail}").json()
        assert model["detail"] == detail
        ids = {n["id"] for n in model["nodes"]}
        assert ids and all(e["source"] in ids and e["target"] in ids for e in model["edges"])


def test_the_graph_page_is_served_on_both_paths(reference_data):
    import main

    client = TestClient(main.app)
    for path in ("/ontology", "/cache"):
        res = client.get(path)
        assert res.status_code == 200
        assert "Ontology" in res.text
        assert "/static/cache.js" in res.text
    # The page's script is bound to the ontology API, not to the cache graph it used to draw.
    script = (main.static_dir / "cache.js").read_text(encoding="utf-8")
    assert "/api/ontology/graph" in script
    assert "/api/cache" not in script


def test_the_ontology_is_memoised_against_the_catalog_version(reference_data):
    first = ontology.build_ontology()
    assert ontology.build_ontology() is first  # same version: no rebuild

    schema.register_concept("cache_buster", _concept_definition("cache_buster", "cache buster"))
    second = ontology.build_ontology()
    assert second is not first
    assert "concept:cache_buster" in _ids(second)

    # The document is derived from the store, never a private copy of it.
    rows = catalog_store.load_all(store.get_conn())
    assert any(r["key"] == "cache_buster" for r in rows["concepts"])


# ---------------------------------------------------------------------------
# Join transformations and coverage
# ---------------------------------------------------------------------------


def test_join_transformations_are_first_class_nodes_and_edges(reference_data):
    doc = ontology.build_ontology()
    nodes = {n["id"]: n for n in doc["classes"] + doc["properties"]}

    for kind in (*tx.KINDS, "expression"):
        node = nodes[f"transform:{kind}"]
        assert node["kind"] == "transform"
        assert node["label"] and node["description"] and node["sql_pattern"]
        assert isinstance(node["needs_width"], bool)

    city = next(
        e
        for e in doc["edges"]
        if e["predicate"] == "joins" and e.get("left_expr") == "lower(trim(city))"
    )
    assert city["join_key"] == "houses.lower(trim(city))=transit_agencies.lower(trim(hq_city))"
    assert city["transform"]["left"] == {
        "kind": "lower",
        "column": "city",
        "expr": "lower(trim(city))",
    }
    assert city["transform"]["right"]["column"] == "hq_city"
    assert "lower case" in city["transform_rule"]
    assert any(
        e["predicate"] == "transformed_by"
        and e["source"] == "table:houses"
        and e["target"] == "transform:lower"
        and e["side"] == "left"
        and e["relationship"] == city["key"]
        for e in doc["edges"]
    )

    # A composite key is still represented honestly, as the expression transform rather than a
    # guessed simple rule; the expression remains authoritative for query generation.
    composite = next(e for e in doc["edges"] if e["predicate"] == "joins" and "state_fips || county_fips" in e["left_expr"])
    assert composite["transform"]["left"]["kind"] == "expression"
    assert any(
        e["predicate"] == "transformed_by"
        and e["target"] == "transform:expression"
        and e["relationship"] == composite["key"]
        for e in doc["edges"]
    )
    assert any(
        item["relationship"] == composite["join_key"] for item in doc["coverage"]["custom_expressions"]
    )


def test_rich_table_column_concept_and_entity_metadata_is_preserved(reference_data):
    doc = ontology.build_ontology()
    nodes = {n["id"]: n for n in doc["classes"] + doc["properties"]}

    table = nodes["table:houses"]
    assert "setup_hint" in table and "filter_hint" in table
    assert isinstance(table["hidden_columns"], list)

    column = next(p for p in doc["properties"] if p["table"] == "houses" and p["primary_key"])
    assert column["not_null"] is True and isinstance(column["ordinal"], int)

    concept = next(n for n in doc["classes"] if n["kind"] == "concept" and n.get("filters"))
    assert isinstance(concept["filters"], list)
    assert isinstance(concept["required_terms"], list)
    assert "default_operation" in concept and "rollup_spec" in concept

    entity = next(n for n in doc["classes"] if n["kind"] == "entity_type" and n.get("columns"))
    source = entity["columns"][0]
    assert "description" in source and "preferred_for" in source and "origin" in source


def test_coverage_is_measured_for_every_table_and_has_no_visible_gaps(reference_data):
    doc = ontology.build_ontology()
    coverage = doc["coverage"]
    assert {t["table"] for t in coverage["tables"]} == set(schema._tables())
    assert coverage["totals"]["columns"] == doc["stats"]["columns"]
    assert coverage["totals"]["relationships"] == doc["stats"]["relationships"]
    assert coverage["totals"]["relationships_with_transform"] == doc["stats"]["relationships_with_transform"]

    assert coverage["gaps"] == []
    for entry in coverage["tables"]:
        if entry["agent_visible"]:
            assert entry["concepts"], entry["table"]
            assert entry["joins"], entry["table"]


def test_graph_and_jsonld_keep_the_transformation_context(reference_data):
    import main

    client = TestClient(main.app)
    overview = client.get("/api/ontology/graph?detail=overview").json()
    assert not any(n["kind"] == "transform" for n in overview["nodes"])

    standard = client.get("/api/ontology/graph?detail=standard").json()
    assert any(n["kind"] == "transform" for n in standard["nodes"])
    assert any(e["predicate"] == "transformed_by" for e in standard["edges"])
    ids = {n["id"] for n in standard["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in standard["edges"])

    linked = client.get("/api/ontology?format=jsonld").json()
    assert linked["coverage"]["gaps"] == []
    assert len(linked["transformations"]) == len(tx.KINDS) + 1
    city = next(
        item
        for item in linked["@graph"]
        if item.get("predicate") == "joins" and "lower(trim(city))" in item.get("join_key", "")
    )
    assert city["transform"]["left"]["kind"] == "lower"
