"""The app's ontology: classes, properties and typed relations derived from the live catalog.

Why it exists
-------------
``db/catalog_*`` already records every fact the agents use - tables, columns with roles and units,
semantic concepts with synonyms and operations, entity domains and approved join relationships -
but only as rows read one query at a time.  Nothing assembled those facts into the *structure* they
form: which concepts belong to a subject area, which physical columns a concept measures, which
tables a concept reads, which join the catalog blesses between two datasets, which entity a filter
is anchored on, and where a user-added dataset enters the picture.

This module assembles that structure, once, from the catalog:

* **classes** - subject areas (``domain``), datasets (``dataset``), tables, semantic concepts and
  entity types.  A concept is the ontology's unit of meaning ("walkability", "arms-length sale");
  it has a label, a description, synonyms, properties (the columns it measures), and relations
  (the tables it reads, the entity type it is about).
* **properties** - every physical column, with the role / unit / derivation the catalog records.
* **edges** - typed, directed relations whose endpoints are always a class or a property above.

Nothing here is hand-written per dataset.  A dataset approved on the Data page registers its table,
columns, concepts, entity domain and links in one catalog transaction; the next call includes them
with no code change and no migration.  That is the extensibility contract, and
``tests/test_ontology.py`` proves it with an uploaded dataset rather than a promise.

What it deliberately does not do
--------------------------------
* **It invents no taxonomy.**  ``subclass_of`` edges exist only where a concept declares
  ``broader`` (an optional catalog knob); ``overrides`` edges only where it declares alias
  precedence.  Everything else is containment or provenance taken verbatim from the catalog.
* **It is not a reasoner.**  There is no inference and no OWL entailment.  ``issues`` reports only
  structural problems: dangling edge endpoints, duplicate identifiers, ``broader`` cycles, and
  catalog column notes for columns that no longer exist.

Cost and freshness
------------------
The document is cached in memory against ``schema.catalog_version()`` - which moves on every catalog
write and on every new DuckDB connection - so a stale ontology cannot outlive a catalog edit.
Building it reads the catalog tables plus one ``PRAGMA table_info`` and one ``COUNT(*)`` per table:
metadata operations DuckDB answers without scanning rows.

The HTTP layer is ``api/ontology.py``; the page that draws it is ``/ontology`` (``static/cache.html``).
"""

from __future__ import annotations

import json
import threading
from typing import Any, Iterable

import db.catalog_store as catalog_store
import db.duckdb_store as store
import db.schema_catalog as schema

NAMESPACE = "urn:rei:"
BUILTIN_DATASET_ID = "__builtin__"
DETAIL_LEVELS = ("overview", "standard", "full")

# The relations the ontology can express.  Every one is produced from catalog metadata; a new
# predicate is a code change here *and* in the builder below, never a per-dataset special case.
PREDICATES: dict[str, str] = {
    "in_domain": "belongs to subject area",
    "in_dataset": "comes from dataset",
    "in_table": "is a column of",
    "uses_table": "reads table",
    "measures": "measures column",
    "about": "is about entity type",
    "requires_entity": "only applies when the request names",
    "subclass_of": "is a narrower concept than",
    "overrides": "takes precedence over",
    "joins": "joins",
    "identified_by": "is identified by",
    "derived_from": "is derived from",
}

# Which node kinds each granularity level includes.  "overview" is the abstraction layer the issue
# asks to surface first (areas, datasets, concepts, entity types); "standard" adds the physical
# tables and the approved joins; "full" adds every column and the measure/identity relations.
_DETAIL_KINDS: dict[str, frozenset[str]] = {
    "overview": frozenset({"domain", "dataset", "concept", "entity_type"}),
    "standard": frozenset({"domain", "dataset", "concept", "entity_type", "table"}),
    "full": frozenset({"domain", "dataset", "concept", "entity_type", "table", "column"}),
}

# Small LRU-ish cache.  Keyed by (catalog version, counts) so a catalog edit can never serve a
# stale document; four entries cover the page's detail levels plus the counts variant.
_CACHE: dict[tuple, dict] = {}
_CACHE_LOCK = threading.RLock()
_CACHE_LIMIT = 4


def _loads(text: Any, default: Any) -> Any:
    if text is None or text == "":
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _iri(kind: str, key: str) -> str:
    return f"{NAMESPACE}{kind}:{key}"


def _humanize(key: str) -> str:
    return str(key or "").replace("_", " ").strip() or str(key)


def _cached(key: tuple, build) -> dict:
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit
    doc = build()
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()
        _CACHE[key] = doc
    return doc


def invalidate() -> None:
    """Drop the memoised document (tests, or a caller that just wrote the catalog)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def build_ontology(with_counts: bool = False) -> dict:
    """The full ontology document for the current catalog.

    ``with_counts`` adds the number of live values behind each entity type (one ``COUNT(DISTINCT)``
    per domain).  It is opt-in because those scans are the only non-metadata reads in the build.
    """
    key = ("ontology", schema.catalog_version(), bool(with_counts))
    return _cached(key, lambda: _build(bool(with_counts)))


def _build(with_counts: bool) -> dict:
    conn = store.get_conn()
    data = catalog_store.load_all(conn)
    issues: list[dict] = []

    def issue(severity: str, code: str, subject: str, message: str) -> None:
        issues.append({"severity": severity, "code": code, "subject": subject, "message": message})

    table_rows: dict[str, dict] = {r["name"]: r for r in data.get("tables", [])}
    table_names = set(table_rows)
    # Every declared area gets a class, including an area the label table does not name yet (a table
    # may declare "transit" while DOMAIN_LABELS has no entry for it).  The label table supplies the
    # display name and the order; an unlabelled area is humanised and grouped after the known ones.
    domain_labels: dict[str, str] = dict(schema.DOMAIN_LABELS)
    table_domain: dict[str, str] = {}
    for name, row in table_rows.items():
        key = str(row.get("domain") or "other")
        table_domain[name] = key
        if key not in domain_labels:
            domain_labels[key] = _humanize(key).title()
    classes: list[dict] = []
    properties: list[dict] = []
    ids: set[str] = set()

    def add_class(node: dict) -> None:
        if node["id"] in ids:
            issue("error", "duplicate_id", node["id"], "two classes share this identifier")
            return
        node.setdefault("iri", _iri(node["kind"], node.get("key") or node["id"].split(":", 1)[-1]))
        ids.add(node["id"])
        classes.append(node)

    def add_property(node: dict) -> None:
        if node["id"] in ids:
            issue("error", "duplicate_id", node["id"], "two properties share this identifier")
            return
        node.setdefault("iri", _iri("column", node["key"]))
        ids.add(node["id"])
        properties.append(node)

    # ── subject areas ──────────────────────────────────────────────────────
    for key, label in domain_labels.items():
        add_class(
            {
                "id": f"domain:{key}",
                "kind": "domain",
                "key": key,
                "label": label,
                "description": f"Subject area “{label}”, declared by the catalog.",
                "origin": "builtin",
                "dataset_id": "",
            }
        )

    # ── datasets (built-in pseudo-dataset + every catalogued upload) ────────
    builtin_tables = sorted(name for name, r in table_rows.items() if (r.get("origin") or "builtin") == "builtin")
    add_class(
        {
            "id": f"dataset:{BUILTIN_DATASET_ID}",
            "kind": "dataset",
            "key": BUILTIN_DATASET_ID,
            "label": "Built-in data model",
            "description": "Tables the application ships with (seeded from db/catalog_seed.py).",
            "dataset_id": "",
            "origin": "builtin",
            "synthetic": True,
            "status": "builtin",
            "tables": builtin_tables,
        }
    )
    try:
        datasets = catalog_store.list_datasets(conn)
    except Exception as exc:  # a database predating the workflow tables
        datasets = []
        issue("warning", "datasets_unavailable", "catalog_datasets", f"datasets could not be read: {exc}")
    for ds in datasets:
        dataset_id = str(ds.get("dataset_id") or "")
        if not dataset_id:
            continue
        tables = sorted(name for name, r in table_rows.items() if str(r.get("dataset_id") or "") == dataset_id)
        add_class(
            {
                "id": f"dataset:{dataset_id}",
                "kind": "dataset",
                "key": dataset_id,
                "label": ds.get("title") or ds.get("table_name") or dataset_id,
                "description": ds.get("description") or "",
                "dataset_id": dataset_id,
                "origin": "upload",
                "synthetic": False,
                "status": ds.get("status") or "",
                "domain": ds.get("domain") or "other",
                "grain": ds.get("grain") or "",
                "rows": ds.get("row_count"),
                "source": ds.get("source_filename") or "",
                "tables": tables,
            }
        )

    # ── tables ─────────────────────────────────────────────────────────────
    def _row_count(name: str) -> int | None:
        try:
            row = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()
            return int(row[0]) if row else None
        except Exception:
            return None

    for name, r in table_rows.items():
        domain = table_domain[name]
        add_class(
            {
                "id": f"table:{name}",
                "kind": "table",
                "key": name,
                "label": name,
                "description": r.get("description") or "",
                "domain": domain,
                "grain": r.get("grain") or "",
                "default_filter": r.get("default_filter") or "",
                "origin": r.get("origin") or "builtin",
                "dataset_id": r.get("dataset_id") or "",
                "agent_visible": bool(r.get("agent_visible")),
                "rows": _row_count(name),
            }
        )

    # ── columns (data properties) ──────────────────────────────────────────
    notes: dict[str, dict[str, dict]] = {}
    for c in data.get("columns", []):
        notes.setdefault(c.get("table_name") or "", {})[c.get("column_name") or ""] = c

    for name, r in table_rows.items():
        hidden = set(_loads(r.get("hidden_columns"), []) or [])
        live: list[tuple[str, str]] = []
        try:
            # PRAGMA table_info: (cid, name, type, notnull, default, pk).
            live = [(str(row[1]), str(row[2] or "")) for row in conn.execute(f"PRAGMA table_info('{name}')").fetchall()]
        except Exception as exc:
            issue("warning", "columns_unavailable", name, f"columns could not be introspected: {exc}")
        seen = set()
        for column, ctype in live:
            note = notes.get(name, {}).get(column, {})
            seen.add(column)
            add_property(
                {
                    "id": f"column:{name}.{column}",
                    "kind": "column",
                    "key": f"{name}.{column}",
                    "label": column,
                    "description": note.get("note") or "",
                    "table": name,
                    "column": column,
                    "type": ctype,
                    "role": note.get("role") or "",
                    "unit": note.get("unit") or "",
                    "source_name": note.get("source_name") or "",
                    "derived_from": note.get("derived_from") or "",
                    "hidden": column in hidden,
                    "origin": r.get("origin") or "builtin",
                    "dataset_id": r.get("dataset_id") or "",
                }
            )
        for column in notes.get(name, {}):
            if column not in seen:
                issue(
                    "warning",
                    "stale_column_note",
                    f"{name}.{column}",
                    "the catalog has a note for a column the table no longer has",
                )

    # ── semantic concepts ──────────────────────────────────────────────────
    concepts: dict[str, dict] = {}
    for row in data.get("concepts", []):
        key = str(row.get("key") or "")
        item = schema._json(row.get("definition"), {})
        if not key or not item:
            continue
        missing = [t for t in item.get("tables", []) if t not in table_names]
        if missing:
            issue(
                "warning",
                "concept_dropped",
                key,
                f"references table(s) {missing} that are not active, so the planner already drops it",
            )
            continue
        concepts[key] = item
        tables = [t for t in item.get("tables", []) if t in table_names]
        concept_domains = sorted({table_domain[t] for t in tables} - {""})
        # A gap concept (a topic the data does not cover) has no tables at all: it belongs to no area.
        domain = concept_domains[0] if concept_domains else "other"
        aliases = [str(a) for a in item.get("aliases", []) if str(a).strip()]
        add_class(
            {
                "id": f"concept:{key}",
                "kind": "concept",
                "key": key,
                "label": aliases[0] if aliases else _humanize(key),
                "description": item.get("description") or "",
                "domain": domain,
                "domains": concept_domains,
                "tables": tables,
                "columns": [str(c) for c in item.get("columns", [])],
                "aliases": aliases,
                "operations": [
                    {
                        "op": str(op.get("op") or ""),
                        "expr": str(op.get("expr") or ""),
                        "direction": str(op.get("direction") or ""),
                        "group_by": str(op.get("group_by") or ""),
                    }
                    for op in item.get("operations", [])
                    if isinstance(op, dict)
                ],
                "entity_types": [str(e) for e in item.get("entity_types", [])],
                "null_policy": item.get("null_policy") or "",
                "grain": item.get("grain") or "",
                "rollup": bool(item.get("rollup")),
                "gap": bool(item.get("gap")),
                "derived": bool(item.get("derived")),
                "broader": [str(b) for b in item.get("broader", [])],
                "overrides": [str(o) for o in item.get("overrides", [])],
                "origin": row.get("origin") or "builtin",
                "dataset_id": row.get("dataset_id") or "",
            }
        )

    # ── entity types (one class per entity_type; each domain is a way to find it) ──
    entity_rows = [d for d in data.get("entity_domains", []) if d.get("table_name") in table_names]
    by_type: dict[str, list[dict]] = {}
    for d in entity_rows:
        by_type.setdefault(str(d.get("entity_type") or ""), []).append(d)
    entity_counts: dict[str, int | None] = {}
    for entity_type, domains in sorted(by_type.items()):
        if not entity_type:
            continue
        columns = [
            {
                "table": d.get("table_name"),
                "column": d.get("column_name"),
                "match_mode": d.get("match_mode") or "exact_or_prefix",
                "domain": d.get("name") or "",
            }
            for d in domains
        ]
        add_class(
            {
                "id": f"entity_type:{entity_type}",
                "kind": "entity_type",
                "key": entity_type,
                # The entity type is a catalog identifier ("MSA", "tract_fips"); showing it verbatim
                # keeps an acronym or a snake_case name from being mangled by a title-caser.
                "label": entity_type,
                "description": "Named values a request can be anchored to: "
                + ", ".join(f"{c['table']}.{c['column']}" for c in columns),
                "columns": columns,
                "domains": sorted({c["domain"] for c in columns if c["domain"]}),
                "origin": sorted({str(d.get("origin") or "builtin") for d in domains})[0],
                "dataset_id": "",
                "individuals": None,
            }
        )
        if with_counts:
            # Count distinct live values; a failure means "unknown", never "zero".
            total: int | None = None
            for c in columns:
                try:
                    row = conn.execute(
                        f'SELECT COUNT(DISTINCT "{c["column"]}") FROM "{c["table"]}" WHERE "{c["column"]}" IS NOT NULL'
                    ).fetchone()
                    total = (total or 0) + int(row[0] or 0)
                except Exception:
                    total = None
                    break
            entity_counts[entity_type] = total

    # ── relations ──────────────────────────────────────────────────────────
    pending: list[tuple[str, str, str, dict]] = []

    def relate(source: str, target: str, predicate: str, **attrs) -> None:
        pending.append((source, target, predicate, attrs))

    for name, r in table_rows.items():
        relate(f"table:{name}", f"domain:{table_domain[name]}", "in_domain")
        dataset_id = str(r.get("dataset_id") or "")
        target = f"dataset:{dataset_id}" if dataset_id else f"dataset:{BUILTIN_DATASET_ID}"
        relate(f"table:{name}", target, "in_dataset")

    def _resolve_column(ref: str, tables: Iterable[str], known: set[str]) -> str | None:
        ref = str(ref or "").strip()
        if not ref:
            return None
        if "." in ref:
            table, column = ref.rsplit(".", 1)
            candidate = f"column:{table}.{column}"
            return candidate if candidate in known else None
        for table in tables:
            candidate = f"column:{table}.{ref}"
            if candidate in known:
                return candidate
        return None

    for prop in properties:
        relate(prop["id"], f"table:{prop['table']}", "in_table")
        if prop.get("derived_from"):
            source = _resolve_column(str(prop["derived_from"]), {prop["table"]}, ids)
            if source:
                relate(prop["id"], source, "derived_from")

    for key, item in concepts.items():
        node_id = f"concept:{key}"
        tables = [t for t in item.get("tables", []) if t in table_names]
        for table in tables:
            relate(node_id, f"table:{table}", "uses_table")
        for ref in item.get("columns", []):
            column = _resolve_column(str(ref), tables, ids)
            if column:
                relate(node_id, column, "measures")
            else:
                issue(
                    "warning",
                    "unresolved_measure",
                    f"{key}:{ref}",
                    "the concept names a column that is not in any of its active tables",
                )
        for entity_type in item.get("entity_types", []):
            if f"entity_type:{entity_type}" in ids:
                relate(node_id, f"entity_type:{entity_type}", "about")
            else:
                issue(
                    "warning",
                    "unknown_entity_type",
                    key,
                    f"entity_types names {entity_type!r}, which no entity domain provides",
                )
        for entity_type in item.get("requires_entity", []) or []:
            if f"entity_type:{entity_type}" in ids:
                relate(node_id, f"entity_type:{entity_type}", "requires_entity")
            else:
                issue(
                    "warning",
                    "unknown_entity_type",
                    key,
                    f"requires_entity names {entity_type!r}, which no entity domain provides",
                )
        domains = sorted({table_domain[t] for t in tables})
        for domain in domains or ["other"]:
            if f"domain:{domain}" in ids:
                relate(node_id, f"domain:{domain}", "in_domain")
        dataset_id = str(item.get("dataset_id") or "")
        if dataset_id and f"dataset:{dataset_id}" in ids:
            relate(node_id, f"dataset:{dataset_id}", "in_dataset")
        for broader in item.get("broader", []) or []:
            if f"concept:{broader}" in ids:
                relate(node_id, f"concept:{broader}", "subclass_of")
            else:
                issue(
                    "error",
                    "unknown_broader",
                    key,
                    f"declares broader {broader!r}, which is not an active concept; the class is left unparented",
                )
        for other in item.get("overrides", []) or []:
            if f"concept:{other}" in ids:
                relate(node_id, f"concept:{other}", "overrides")
            else:
                issue("warning", "unknown_override", key, f"overrides {other!r}, which is not an active concept")
        if item.get("gap") and tables:
            issue(
                "error", "gap_with_tables", key, "a gap concept declares an unavailable topic and must have no tables"
            )

    for domain in entity_rows:
        entity_type = str(domain.get("entity_type") or "")
        column = f"column:{domain.get('table_name')}.{domain.get('column_name')}"
        if entity_type and column in ids:
            relate(
                f"entity_type:{entity_type}",
                column,
                "identified_by",
                match_mode=domain.get("match_mode") or "exact_or_prefix",
                domain=domain.get("name") or "",
            )

    for rel in data.get("relationships", []):
        left, right = str(rel.get("left_table") or ""), str(rel.get("right_table") or "")
        if left not in table_names or right not in table_names:
            issue(
                "warning",
                "relationship_dropped",
                str(rel.get("rel_key") or f"{left}->{right}"),
                "an endpoint table is not active, so the join is not part of the model",
            )
            continue
        relate(
            f"table:{left}",
            f"table:{right}",
            "joins",
            key=str(rel.get("rel_key") or ""),
            left_expr=str(rel.get("left_expr") or ""),
            right_expr=str(rel.get("right_expr") or ""),
            cardinality=str(rel.get("cardinality") or ""),
            confidence=str(rel.get("confidence") or ""),
            bridge=bool(rel.get("bridge")),
            preferred=bool(rel.get("preferred")),
            grain_effect=str(rel.get("grain_effect") or ""),
            note=str(rel.get("note") or ""),
            origin=str(rel.get("origin") or "builtin"),
        )

    for ds in classes:
        if ds["kind"] == "dataset" and ds.get("domain") and f"domain:{ds['domain']}" in ids:
            relate(ds["id"], f"domain:{ds['domain']}", "in_domain")

    # ── resolve, deduplicate and validate edges ───────────────────────────
    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    for source, target, predicate, attrs in pending:
        if source not in ids or target not in ids:
            issue(
                "error",
                "dangling_edge",
                f"{source} -{predicate}-> {target}",
                "an edge endpoint is not part of the ontology",
            )
            continue
        signature = (source, target, predicate, str(attrs.get("key") or ""))
        if signature in seen_edges:
            continue
        seen_edges.add(signature)
        edges.append({"source": source, "target": target, "predicate": predicate, **attrs})

    _check_subclass_cycles(edges, issues)

    for entity_type, count in entity_counts.items():
        for node in classes:
            if node["id"] == f"entity_type:{entity_type}":
                node["individuals"] = count

    stats = {
        "domains": sum(1 for n in classes if n["kind"] == "domain"),
        "datasets": sum(1 for n in classes if n["kind"] == "dataset"),
        "tables": sum(1 for n in classes if n["kind"] == "table"),
        "columns": len(properties),
        "concepts": sum(1 for n in classes if n["kind"] == "concept"),
        "entity_types": sum(1 for n in classes if n["kind"] == "entity_type"),
        "relationships": sum(1 for e in edges if e["predicate"] == "joins"),
        "edges": len(edges),
    }
    return {
        "version": schema.catalog_version(),
        "namespace": NAMESPACE,
        "detail_levels": list(DETAIL_LEVELS),
        "predicates": dict(PREDICATES),
        "domains": [{"key": k, "label": v} for k, v in domain_labels.items()],
        "classes": classes,
        "properties": properties,
        "edges": edges,
        "stats": stats,
        "issues": issues,
    }


def _check_subclass_cycles(edges: Iterable[dict], issues: list[dict]) -> None:
    """Report ``subclass_of`` cycles.  A cycle is a modeling error, so it is surfaced, not repaired."""
    parents: dict[str, set[str]] = {}
    for edge in edges:
        if edge["predicate"] == "subclass_of":
            parents.setdefault(edge["source"], set()).add(edge["target"])

    def reaches(start: str, current: str, path: set[str]) -> bool:
        for parent in parents.get(current, ()):
            if parent == start:
                return True
            if parent not in path:
                if reaches(start, parent, path | {parent}):
                    return True
        return False

    for child in parents:
        if reaches(child, child, {child}):
            issues.append(
                {
                    "severity": "error",
                    "code": "subclass_cycle",
                    "subject": child,
                    "message": "the declared broader relations form a cycle, so no class is a strict parent",
                }
            )


def build_graph(detail: str = "standard") -> dict:
    """The page's node/edge projection of the ontology at one granularity level.

    Unknown levels fall back to ``standard`` so a stale bookmark cannot break the page.  Only nodes
    of the level's kinds are returned, and only edges whose two endpoints survived the filter.
    """
    level = detail if detail in DETAIL_LEVELS else "standard"
    doc = build_ontology()
    kinds = _DETAIL_KINDS[level]
    nodes = [{**n} for n in doc["classes"] + doc["properties"] if n["kind"] in kinds]
    ids = {n["id"] for n in nodes}
    edges = [{**e} for e in doc["edges"] if e["source"] in ids and e["target"] in ids]
    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1
    for node in nodes:
        node["degree"] = degree.get(node["id"], 0)
    return {
        "version": doc["version"],
        "detail": level,
        "detail_levels": doc["detail_levels"],
        "predicates": doc["predicates"],
        "domains": doc["domains"],
        "nodes": nodes,
        "edges": edges,
        "stats": doc["stats"],
        "issues": doc["issues"],
        "counts": {"nodes": len(nodes), "edges": len(edges)},
    }


def as_jsonld(doc: dict | None = None) -> dict:
    """Serialize the ontology as JSON-LD ``@graph`` (relations reified, one node per edge).

    Values are copied verbatim from the derived document; nothing is inferred here either.
    """
    doc = doc or build_ontology()
    iri_of = {n["id"]: n["iri"] for n in doc["classes"] + doc["properties"]}
    graph: list[dict] = []
    for node in doc["classes"] + doc["properties"]:
        item: dict[str, Any] = {
            "@id": node["iri"],
            "@type": f"rei:{node['kind']}",
            "label": node.get("label") or "",
            "description": node.get("description") or "",
        }
        for field in ("domain", "unit", "role", "grain", "origin", "dataset_id", "tables"):
            if node.get(field):
                item[field] = node[field]
        graph.append(item)
    for index, edge in enumerate(doc["edges"]):
        graph.append(
            {
                "@id": f"{NAMESPACE}relation:{index}",
                "@type": "rei:Relation",
                "predicate": edge["predicate"],
                "source": iri_of.get(edge["source"], edge["source"]),
                "target": iri_of.get(edge["target"], edge["target"]),
            }
        )
    return {
        "@context": {
            "@vocab": NAMESPACE,
            "rei": NAMESPACE,
            "label": "http://www.w3.org/2000/01/rdf-schema#label",
            "description": "http://www.w3.org/2000/01/rdf-schema#comment",
            "source": {"@id": f"{NAMESPACE}source", "@type": "@id"},
            "target": {"@id": f"{NAMESPACE}target", "@type": "@id"},
        },
        "@graph": graph,
    }
