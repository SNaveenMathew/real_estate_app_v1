"""Unified data-model catalog: the single contract the planner, SQL agent and chats read.

There is ONE catalog.  Built-in sources (Redfin, NRI, Census, sold homes, crime, bike) and
datasets added later through the Data page are the same kind of object: rows in the
``catalog_*`` tables of the application's DuckDB database (see ``db/catalog_store.py``).
``db/catalog_seed.py`` only *seeds* the built-in rows on first run and refreshes them on
upgrade (unless you edited them); after that the store is the source of truth and this
module is a thin, in-memory view over it.  Adding a data source therefore never requires
editing this file.

The catalog is deliberately *not* a routing table.  It describes facts that are true of the
physical model: tables, fields, grain, aliases, operations, entity domains and join
relationships.  The planner composes these facts into a query plan; it does not contain
domain-specific question branches.

Compatibility
-------------
``TABLES``, ``RELATIONSHIPS``, ``SEMANTIC_GLOSSARY`` and ``ENTITY_DOMAINS`` are still
importable module attributes, but they are now *live containers*: they load lazily from the
store on first access, are refreshed **in place** by ``reload()`` (so anything that already
holds a reference keeps seeing current data), and reload automatically when the DuckDB
connection changes (e.g. the evaluation harness pointing at a fixture database).
"""
from __future__ import annotations

import json
import re
import threading
from collections import deque
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

import db.duckdb_store as store
from db import catalog_store
from db.catalog_model import (
    ColumnNote, TableMeta, Relationship, EntityDomain,
    _concept, _op, COUNT, AVG, SUM, MEDIAN, MIN, MAX, RANK_DESC, RANK_ASC,
)
from db.catalog_seed import NRI_HAZARD_COLUMNS   # fixed hazard-code -> label map (not per-source metadata)


DOMAIN_LABELS = {
    "housing": "Housing",
    "geography": "Geography",
    "risk": "Hazard risk",
    "sales": "Sales",
    "safety": "Crime and safety",
    "mobility": "Mobility",
    "environment": "Environment",
    "demographics": "Demographics",
    "other": "Other",
    "system": "System",
}


# ---------------------------------------------------------------------------
# Live registry (loaded from the catalog store)
# ---------------------------------------------------------------------------

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"loaded": False, "generation": -1, "version": 0, "tx_depth": 0}
_TABLES: dict[str, TableMeta] = {}
_RELATIONSHIPS: list[Relationship] = []
_GLOSSARY: dict[str, dict] = {}
_ENTITY_DOMAINS: list[EntityDomain] = []


def __getattr__(name: str):
    """PEP 562: keep the historical module attributes, backed by the live registry."""
    if name == "TABLES":
        _ensure(); return _TABLES
    if name == "RELATIONSHIPS":
        _ensure(); return _RELATIONSHIPS
    if name == "SEMANTIC_GLOSSARY":
        _ensure(); return _GLOSSARY
    if name == "ENTITY_DOMAINS":
        _ensure(); return _ENTITY_DOMAINS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _tables() -> dict[str, TableMeta]:
    _ensure(); return _TABLES


def _relationships() -> list[Relationship]:
    _ensure(); return _RELATIONSHIPS


def _glossary() -> dict[str, dict]:
    _ensure(); return _GLOSSARY


def _entity_domains() -> list[EntityDomain]:
    _ensure(); return _ENTITY_DOMAINS


def _ensure() -> None:
    # A closed connection means the database may be about to change (e.g. the evaluation harness closes it
    # and points settings.duckdb_path at a fixture DB), so reload through a fresh connection.
    if _STATE["loaded"] and store.is_connected() and _STATE["generation"] == store.connection_generation():
        return
    reload()


def _json(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def reload() -> str:
    """Rebuild the in-memory registry from the catalog store, in place."""
    with _LOCK:
        conn = store.get_conn()
        data = catalog_store.load_all(conn)

        notes: dict[str, list[ColumnNote]] = {}
        for c in data["columns"]:
            notes.setdefault(c["table_name"], []).append(ColumnNote(
                column=c["column_name"], note=c.get("note") or "", role=c.get("role") or "",
                unit=c.get("unit") or "", source_name=c.get("source_name") or ""))

        tables: dict[str, TableMeta] = {}
        for r in data["tables"]:
            tables[r["name"]] = TableMeta(
                name=r["name"], description=r.get("description") or "",
                setup_hint=r.get("setup_hint") or "", filter_hint=r.get("filter_hint") or "",
                column_notes=tuple(notes.get(r["name"], ())),
                hidden_columns=tuple(_json(r.get("hidden_columns"), [])),
                agent_visible=bool(r.get("agent_visible")), grain=r.get("grain") or "",
                default_filter=r.get("default_filter") or "", domain=r.get("domain") or "other",
                origin=r.get("origin") or "builtin", dataset_id=r.get("dataset_id") or "")

        rels = [Relationship(
            left_table=r["left_table"], left_expr=r["left_expr"], right_table=r["right_table"],
            right_expr=r["right_expr"], note=r.get("note") or "", cardinality=r.get("cardinality") or "",
            confidence=r.get("confidence") or "high", bridge=bool(r.get("bridge")),
            preferred=bool(r.get("preferred")), grain_effect=r.get("grain_effect") or "",
            origin=r.get("origin") or "builtin", dataset_id=r.get("dataset_id") or "")
            for r in data["relationships"] if r["left_table"] in tables and r["right_table"] in tables]

        glossary: dict[str, dict] = {}
        for r in data["concepts"]:
            item = _json(r.get("definition"), {})
            if not item or any(t not in tables for t in item.get("tables", [])):
                continue
            item.setdefault("origin", r.get("origin") or "builtin")
            item.setdefault("dataset_id", r.get("dataset_id") or "")
            glossary[r["key"]] = item

        domains = [EntityDomain(
            name=r["name"], table=r["table_name"], column=r["column_name"], entity_type=r["entity_type"],
            description=r.get("description") or "", display_column=r.get("display_column") or None,
            match_mode=r.get("match_mode") or "exact_or_prefix",
            preferred_for=tuple(_json(r.get("preferred_for"), [])),
            origin=r.get("origin") or "builtin", dataset_id=r.get("dataset_id") or "")
            for r in data["entity_domains"] if r["table_name"] in tables]

        _TABLES.clear(); _TABLES.update(tables)
        _RELATIONSHIPS[:] = rels
        _GLOSSARY.clear(); _GLOSSARY.update(glossary)
        _ENTITY_DOMAINS[:] = domains
        _STATE.update(loaded=True, generation=store.connection_generation(), version=int(data["version"]))
        return f"{_STATE['generation']}:{_STATE['version']}"


def catalog_version() -> str:
    """Changes whenever the catalog changes (or the DB connection is replaced)."""
    _ensure()
    return f"{_STATE['generation']}:{_STATE['version']}"


# ---------------------------------------------------------------------------
# Unified mutation API (built-in and uploaded sources use the same calls)
# ---------------------------------------------------------------------------

@contextmanager
def catalog_transaction() -> Iterator[None]:
    """Group physical DDL and catalog writes into one atomic change.

    The registry reloads once, after the outermost transaction ends, so the chats never see
    a half-applied change.  Standalone ``register_*`` calls are each their own transaction.
    """
    conn = store.get_conn()
    with _LOCK:
        outermost = _STATE["tx_depth"] == 0
        _STATE["tx_depth"] += 1
        if outermost:
            conn.execute("BEGIN")
        try:
            yield
            if outermost:
                catalog_store.bump_version(conn)
                conn.execute("COMMIT")
        except BaseException:
            if outermost:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
            raise
        finally:
            _STATE["tx_depth"] -= 1
            if outermost:
                reload()


def _write(fn) -> None:
    with catalog_transaction():
        fn(store.get_conn())


def register_table(meta: TableMeta, *, columns: Iterable[dict] = (), origin: str = "upload",
                   dataset_id: str = "", domain: str | None = None, user_modified: bool = False) -> None:
    cols = [dict(c) for c in columns]
    _write(lambda conn: catalog_store.upsert_table(
        conn, meta, columns=cols, origin=origin, dataset_id=dataset_id,
        domain=domain, user_modified=user_modified))


def register_relationship(rel: Relationship, *, origin: str = "upload", dataset_id: str = "",
                          evidence: dict | None = None, status: str = "approved") -> None:
    _write(lambda conn: catalog_store.upsert_relationship(
        conn, rel, origin=origin, dataset_id=dataset_id, evidence=evidence or {}, status=status))


def revoke_relationship(rel_key: str, reason: str = "") -> bool:
    result: dict[str, bool] = {}

    def _do(conn):
        result["ok"] = catalog_store.set_relationship_status(conn, rel_key, "revoked", reason)
    _write(_do)
    return bool(result.get("ok"))


def register_concept(key: str, definition: dict, *, origin: str = "upload", dataset_id: str = "") -> None:
    _write(lambda conn: catalog_store.upsert_concept(conn, key, definition, origin=origin, dataset_id=dataset_id))


def register_entity_domain(domain: EntityDomain, *, origin: str = "upload", dataset_id: str = "") -> None:
    _write(lambda conn: catalog_store.upsert_entity_domain(conn, domain, origin=origin, dataset_id=dataset_id))


def retire_dataset_objects(dataset_id: str) -> dict:
    counts: dict[str, int] = {}

    def _do(conn):
        counts.update(catalog_store.retire_dataset(conn, dataset_id))
    _write(_do)
    return counts


def update_table_description(name: str, description: str) -> None:
    """Edit a table description; marks the row user-modified so a seed refresh never overwrites it."""
    _write(lambda conn: catalog_store.update_table_field(conn, name, "description", description))


# ---------------------------------------------------------------------------
# Join-expression helper (shared by the SQL compiler and the House Chat link plan)
# ---------------------------------------------------------------------------

def qualify_join_expr(table: str, expr: str) -> str:
    """Table-qualify every bare column reference in a relationship-key expr.

    Most catalog relationships key on a single bare column, where a plain
    f"{table}.{expr}" prefix is correct. A few key on a compound expression
    instead - e.g. "state_fips || county_fips" (concatenation) or
    "LEFT(tract_fips, 5)" (a function call) - and naively prefixing the whole string
    only qualifies the first token.  Qualify each bare identifier individually instead,
    leaving SQL function names (identifier immediately followed by '(') and
    already-qualified references untouched.
    """
    def _replace(match: re.Match) -> str:
        token = match.group(0)
        if expr[match.end():match.end() + 1] == "(":
            return token  # function name, e.g. LEFT( - not a column
        return f"{table}.{token}"

    return re.sub(r"(?<!\.)\b[A-Za-z_][A-Za-z0-9_]*\b", _replace, expr)


# ---------------------------------------------------------------------------
# Views for the chats and the Data page
# ---------------------------------------------------------------------------

def _direct_links(table: str, limit: int = 3) -> str:
    out = []
    for r in _relationships():
        if table in (r.left_table, r.right_table):
            out.append(f"{r.left_table}.{r.left_expr} = {r.right_table}.{r.right_expr}")
    return "; ".join(out[:limit])


def added_datasets_briefing(limit: int = 12) -> str:
    """One compact block for the General Chat prompt describing user-added datasets (or '')."""
    rows = [m for m in _tables().values() if m.agent_visible and m.origin != "builtin"]
    if not rows:
        return ""
    lines = ["USER-ADDED DATASETS (approved on the Data page; query them like any other table):"]
    for m in sorted(rows, key=lambda x: x.name)[:limit]:
        links = _direct_links(m.name)
        lines.append(f"- {m.name}: {m.description} grain={m.grain or 'unspecified'}; "
                     + (f"joins: {links}" if links else "no approved joins yet"))
    return "\n".join(lines)


def house_link_plan(table: str) -> dict | None:
    """SQL that returns the rows of ``table`` linked to ONE house via approved relationships.

    Built purely from the relationship graph (bridge tables allowed), so a dataset linked to
    houses directly, or only via ``nri_tracts``/``census_tracts``, is handled the same way.
    Returns ``{"sql": ..., "join_text": ...}`` (one ``?`` parameter: the house_id) or None.
    """
    tables = _tables()
    if "houses" not in tables or table not in tables or table == "houses":
        return None
    rels = relationship_path({"houses", table})
    if not rels:
        return None
    connected = {"houses"}
    joins: list[str] = []
    remaining = list(rels)
    progressed = True
    while remaining and progressed:
        progressed = False
        for rel in list(remaining):
            if rel.left_table in connected and rel.right_table not in connected:
                new = rel.right_table
            elif rel.right_table in connected and rel.left_table not in connected:
                new = rel.left_table
            else:
                continue
            on = (f"{qualify_join_expr(rel.left_table, rel.left_expr)} = "
                  f"{qualify_join_expr(rel.right_table, rel.right_expr)}")
            joins.append(f"JOIN {new} ON {on}")
            connected.add(new)
            remaining.remove(rel)
            progressed = True
    if table not in connected:
        return None
    sql = f"SELECT {table}.* FROM houses " + " ".join(joins) + " WHERE houses.house_id = ? LIMIT 25"
    join_text = "; ".join(f"{r.left_table}.{r.left_expr} = {r.right_table}.{r.right_expr}" for r in rels)
    return {"sql": sql, "join_text": join_text}


def house_linked_datasets() -> list[dict]:
    """User-added, agent-visible tables that can be reached from a house via the join graph."""
    out = []
    for m in sorted(_tables().values(), key=lambda x: x.name):
        if not m.agent_visible or m.origin == "builtin" or m.name == "houses":
            continue
        plan = house_link_plan(m.name)
        if not plan:
            continue
        aliases: list[str] = []
        for item in _glossary().values():
            if m.name in item.get("tables", []) and item.get("aliases"):
                aliases.append(item["aliases"][0])
        out.append({"name": m.name, "description": m.description, "grain": m.grain,
                    "join": plan["join_text"], "measures": aliases[:4]})
    return out


def describe_catalog() -> dict:
    """The unified catalog as plain JSON for the Data page's schema map."""
    _ensure()
    conn = store.get_conn()
    rel_meta = {r["rel_key"]: r for r in catalog_store.relationship_meta(conn)}
    counts: dict[str, int] = {}
    concepts = []
    for key, item in _GLOSSARY.items():
        for t in item.get("tables", []):
            counts[t] = counts.get(t, 0) + 1
        concepts.append({"key": key, "tables": item.get("tables", []), "aliases": item.get("aliases", []),
                         "description": item.get("description", ""), "origin": item.get("origin", "builtin"),
                         "columns": item.get("columns", [])})
    tables = []
    for name, m in _TABLES.items():
        notes = {n.column: n for n in m.column_notes}
        cols = []
        for cname, ctype in _live_columns(name):
            if cname in m.hidden_columns:
                continue
            n = notes.get(cname)
            cols.append({"name": cname, "type": ctype, "note": n.note if n else "",
                         "role": n.role if n else "", "unit": n.unit if n else "",
                         "source_name": n.source_name if n else ""})
        tables.append({"name": name, "description": m.description, "grain": m.grain,
                       "domain": m.domain or "other", "origin": m.origin, "dataset_id": m.dataset_id,
                       "agent_visible": m.agent_visible, "rows": _row_count(name), "columns": cols,
                       "concepts": counts.get(name, 0)})
    rels = []
    for r in _RELATIONSHIPS:
        meta = rel_meta.get(r.key(), {})
        rels.append({"key": r.key(), "left_table": r.left_table, "left_expr": r.left_expr,
                     "right_table": r.right_table, "right_expr": r.right_expr,
                     "cardinality": r.cardinality, "confidence": r.confidence, "bridge": r.bridge,
                     "preferred": r.preferred, "note": r.note, "grain_effect": r.grain_effect,
                     "origin": r.origin, "dataset_id": r.dataset_id,
                     "approved_at": str(meta.get("approved_at") or ""),
                     "evidence": _json(meta.get("evidence"), {})})
    return {"version": catalog_version(),
            "domains": [{"key": k, "label": v} for k, v in DOMAIN_LABELS.items()],
            "tables": tables, "relationships": rels, "concepts": concepts,
            "entity_domains": [{"name": d.name, "table": d.table, "column": d.column,
                                "entity_type": d.entity_type, "description": d.description,
                                "origin": d.origin} for d in _ENTITY_DOMAINS]}


# ---------------------------------------------------------------------------
# Planner support (unchanged logic; reads the live registry)
# ---------------------------------------------------------------------------

REQUEST_INTENTS = [
    {"name": "comparison", "aliases": ["compare", "comparison", "versus", "vs", "tradeoff"]},
    {"name": "ranking", "aliases": ["rank", "ranking", "top", "highest", "lowest", "largest", "smallest", "best", "worst"]},
    {"name": "aggregation", "aliases": ["how many", "count", "number of", "average", "avg", "mean", "median", "total", "sum"]},
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _alias_spans(text: str, alias: str):
    t = _normalize(text)
    a = _normalize(alias)
    if not a:
        return []
    pattern = r"(?<!\w)" + re.escape(a) + r"(?!\w)"
    return [(m.start(), m.end(), len(a.split())) for m in re.finditer(pattern, t)]


def _alias_matches(text: str, alias: str) -> bool:
    return bool(_alias_spans(text, alias))


def semantic_matches(query: str) -> list[dict]:
    text = _normalize(query)
    hits = []
    for key, item in _glossary().items():
        matches = []
        for alias in item.get("aliases", []):
            matches.extend((len(alias.split()), start, end, alias) for start, end, _ in _alias_spans(text, alias))
        if not matches:
            continue
        required_terms = item.get("required_terms", [])
        if required_terms and not all(_alias_matches(text, term) for term in required_terms):
            continue
        excluded_terms = item.get("excluded_terms", [])
        if any(_alias_matches(text, term) for term in excluded_terms):
            continue
        specificity = max(m[0] for m in matches)
        hits.append((specificity, len(matches), key, item, [(m[1], m[2]) for m in matches]))
    hits.sort(key=lambda x: (-x[0], -x[1], x[2]))
    # Declared precedence. A concept may list ``overrides``: other concepts whose phrase sits inside one of
    # its own longer phrases (e.g. "walkability index" over "walkability"). It is honored only when EVERY
    # phrase the overridden concept matched lies inside a phrase the overriding concept matched, so a
    # question that also uses the shorter phrase on its own still selects both. Without this, adding a
    # dataset whose alias extends an existing alias would silently pull the old concept's tables and
    # null-policy filters into every question that uses the new phrase.
    spans_by_key = {h[2]: h[4] for h in hits}
    overridden = set()
    for h in hits:
        for other in h[3].get("overrides", []):
            if other in spans_by_key and all(
                    any(s2 <= s1 and e1 <= e2 and (e2 - s2) > (e1 - s1) for s2, e2 in h[4])
                    for s1, e1 in spans_by_key[other]):
                overridden.add(other)
    hits = [h for h in hits if h[2] not in overridden]
    # Suppress generic inventory if a more-specific sale/history concept is matched.
    specific_keys = {h[2] for h in hits if h[0] >= 2 or h[2] in {"sold_price", "arms_length_sale", "history", "nri_overall_risk", "nri_riverine_flood"}}
    out = []
    for spec, count, key, item, _spans in hits:
        if key == "house_inventory" and any(k in specific_keys for k in {"sold_price", "arms_length_sale", "history"}):
            continue
        row = dict(item)
        row["key"] = key
        row["match_score"] = spec * 10 + count
        out.append(row)
    return out


def match_request_intents(query: str) -> list[dict]:
    t = _normalize(query)
    hits = []
    for item in REQUEST_INTENTS:
        score = max((len(a.split()) for a in item["aliases"] if _alias_matches(t, a)), default=0)
        if score:
            hits.append((score, item))
    hits.sort(key=lambda x: -x[0])
    return [i for _, i in hits]


def tables_mentioned_in_text(query: str) -> list[str]:
    t = _normalize(query)
    return [name for name in _tables() if _alias_matches(t, name.replace("_", " "))]


def list_table_names(agent_visible_only: bool = True) -> list[str]:
    return sorted(n for n, m in _tables().items() if not agent_visible_only or m.agent_visible)


def _live_columns(table_name: str):
    try:
        df = store.query(f"DESCRIBE {table_name}")
        return [(str(r[0]), str(r[1])) for r in df.itertuples(index=False, name=None)]
    except Exception:
        return []


def _row_count(table_name: str) -> int:
    try:
        df = store.query(f"SELECT COUNT(*) AS n FROM {table_name}")
        return int(df.iloc[0, 0]) if not df.empty else 0
    except Exception:
        return 0


def availability_report():
    counts = {t: _row_count(t) for t in list_table_names()}
    lines = ["Live data availability:"] + [f"- {t}: {n} rows" for t, n in counts.items()]
    return "\n".join(lines), counts


def _fetch_values(table: str, column: str, limit: int = 5000):
    try:
        df = store.query(f"SELECT DISTINCT {column} AS value FROM {table} WHERE {column} IS NOT NULL LIMIT {int(limit)}")
        return [str(v) for v in df["value"].tolist()]
    except Exception:
        return []


def resolve_request_entities(query: str, candidate_tables: set[str] | None = None) -> list[dict]:
    t = _normalize(query)
    domains = [d for d in _entity_domains() if not candidate_tables or d.table in candidate_tables]
    results = []
    # Tract FIPS are unambiguous literals and should be resolved first.
    for d in domains:
        if d.entity_type == "tract_fips":
            for fips in re.findall(r"\b\d{11}\b", query):
                values = _fetch_values(d.table, d.column)
                if fips in values:
                    results.append({"entity_type": d.entity_type, "table": d.table, "column": d.column, "value": fips, "domain": d.name, "score": 100})
    # Prefer MSA domain when the request contains MSA/metro language or an MSA concept matched.
    msa_context = any(_alias_matches(t, a) for a in ["msa", "msas", "metro", "metro area", "metropolitan area", "metro areas"])
    matched_concepts = semantic_matches(query)
    if any("MSA" in c.get("entity_types", []) for c in matched_concepts):
        msa_context = True
    for d in domains:
        values = _fetch_values(d.table, d.column)
        for value in values:
            nv = _normalize(value)
            if d.match_mode == "exact_or_prefix":
                if _alias_matches(t, nv):
                    results.append({"entity_type": d.entity_type, "table": d.table, "column": d.column, "value": value, "domain": d.name, "score": 80})
            elif d.match_mode == "prefix":
                # Stored display value may be 'Pittsburgh, PA Metro Area'; a query token 'Pittsburgh' matches the leading label.
                raw_prefix = value.split(",", 1)[0].strip()
                prefix = _normalize(raw_prefix)
                if prefix and (_alias_matches(t, prefix) or _alias_matches(t, nv)):
                    score = 95 if msa_context else 55
                    results.append({"entity_type": d.entity_type, "table": d.table, "column": d.column, "value": value, "domain": d.name, "score": score})
    # Keep the best domain for each normalized value/entity type.
    best = {}
    for e in results:
        k = (e["entity_type"], e["table"], _normalize(e["value"]))
        if k not in best or e["score"] > best[k]["score"]:
            best[k] = e
    return sorted(best.values(), key=lambda e: (-e["score"], e["entity_type"], e["value"]))


def _entity_predicate(entity: dict) -> str:
    value = entity["value"].replace("'", "''")
    return f"{entity['table']}.{entity['column']} = '{value}'"


def relationship_path(required_tables: set[str]) -> list[Relationship]:
    """Return the minimal high-quality relationship forest connecting required tables.

    Intermediate bridge tables are allowed; they come from the declarative
    RELATIONSHIPS graph and are never selected from user-language rules.
    """
    required = set(required_tables)
    if len(required) <= 1:
        return []

    graph = {}
    for rel in _relationships():
        weight = (0 if rel.confidence == "high" else 10) + (0 if rel.preferred else 5) + (0 if rel.bridge else 1)
        graph.setdefault(rel.left_table, []).append((rel.right_table, rel, weight))
        graph.setdefault(rel.right_table, []).append((rel.left_table, rel, weight))

    def shortest_to_connected(source, connected):
        import heapq
        heap = [(0, source, [])]
        seen = {source: 0}
        while heap:
            cost, node, path = heapq.heappop(heap)
            if node in connected:
                return path, cost
            for nxt, rel, weight in graph.get(node, []):
                nc = cost + weight
                if nc < seen.get(nxt, float("inf")):
                    seen[nxt] = nc
                    heapq.heappush(heap, (nc, nxt, path + [rel]))
        return None, float("inf")

    seed = sorted(required)[0]
    connected = {seed}
    chosen = []
    while not required.issubset(connected):
        candidates = []
        for source in sorted(required - connected):
            path, cost = shortest_to_connected(source, connected)
            if path:
                candidates.append((cost, source, path))
        if not candidates:
            break
        _, _, path = min(candidates, key=lambda x: (x[0], x[1]))
        for rel in path:
            chosen.append(rel)
            connected.add(rel.left_table)
            connected.add(rel.right_table)

    # Preserve order but remove duplicate relationship edges.
    out = []
    seen = set()
    for rel in chosen:
        k = rel.key()
        if k not in seen:
            seen.add(k)
            out.append(rel)
    return out


def expand_required_tables(tables: set[str]) -> list[str]:
    # Connect the selected tables using the minimal relationship forest.
    path = relationship_path(tables)
    out = set(tables)
    for rel in path:
        out.add(rel.left_table); out.add(rel.right_table)
    return sorted(out)


def relationships_for_tables(tables: set[str]) -> list[Relationship]:
    return relationship_path(tables)


def _concept_requires_rollup(concepts: list[dict], entity_tables: set[str]) -> bool:
    return bool(entity_tables) and any(c.get("rollup") for c in concepts)


def build_query_context(request: str, requirements: str = "", plan: str = "", focused: bool = False) -> str:
    concepts = semantic_matches(request)
    if focused and plan:
        selected = set()
        marker = re.search(r"semantic_keys:\s*([^\n]+)", plan)
        if marker:
            selected.update(x.strip() for x in marker.group(1).split(",") if x.strip() not in {"none", "null"})
        if selected:
            concepts = [c for c in concepts if c.get("key") in selected or c.get("key") == "msa_cbsa_membership"]
    target_tables = {t for c in concepts for t in c.get("tables", [])}
    target_tables.update(tables_mentioned_in_text(request))
    resolved = resolve_request_entities(request, None)
    msa_entities = [e for e in resolved if e["entity_type"] == "MSA"]
    tract_entities = [e for e in resolved if e["entity_type"] == "tract_fips"]
    # Entity domain + semantic target determines the geography without hardcoded query branches.
    if msa_entities and _concept_requires_rollup(concepts, {e["table"] for e in msa_entities}):
        target_tables.add("census_msa")
        if any(c.get("rollup") for c in concepts):
            target_tables.add("cbsa_counties")
    if tract_entities:
        target_tables.add("census_tracts")
        if any(c.get("rollup") for c in concepts):
            target_tables.add("nri_tracts")
    target_tables = set(expand_required_tables(target_tables))

    parts = ["TARGETED DATA MODEL"]
    for table in sorted(target_tables):
        meta = _tables()[table]
        live = _live_columns(table)
        cols = [c for c, _ in live if c not in meta.hidden_columns]
        parts.append(f"TABLE {table}: {meta.description}; grain={meta.grain}; columns={', '.join(cols)}")
        for note in meta.column_notes:
            if note.column in cols:
                parts.append(f"  {table}.{note.column}: {note.note}")
    rels = relationships_for_tables(target_tables)
    if rels:
        parts.append("RELATIONSHIPS")
        parts.extend(f"- {r.render()}" for r in rels)
    if concepts:
        parts.append("SEMANTIC CONCEPTS")
        for c in concepts:
            parts.append(f"- {c['key']}: {c['description']}; columns={c['columns']}; operations={c['operations']}; filters={c['filters']}; grain={c['grain']}; entity_types={c['entity_types']}; rollup={c['rollup']}; rollup_spec={c.get('rollup_spec', {})}")
    if resolved:
        parts.append("RESOLVED LIVE ENTITY VALUES")
        for e in resolved[:20]:
            parts.append(f"- {e['entity_type']}: {e['table']}.{e['column']} = {e['value']}")
    return "\n".join(parts)


def render_schema_for_agent() -> str:
    lines = ["DATABASE MODEL"]
    for table in list_table_names():
        m = _tables()[table]
        cols = ", ".join(c for c, _ in _live_columns(table) if c not in m.hidden_columns)
        lines.append(f"TABLE {table}: {m.description}; grain={m.grain}; columns={cols}")
    lines.append("RELATIONSHIPS")
    lines.extend(f"- {r.render()}" for r in _relationships() if r.preferred)
    return "\n".join(lines)


def diagnose_empty_or_error(sql: str) -> str:
    refs = set(re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)", sql, re.I))
    empty = [t for t in refs if t in _tables() and _row_count(t) == 0]
    if empty:
        return "EMPTY TABLES: " + ", ".join(sorted(empty))
    rels = [r.render() for r in _relationships() if r.left_table in refs or r.right_table in refs]
    return "Potential relationship context:\n" + "\n".join(rels[:10]) if rels else ""
