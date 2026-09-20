"""Persistent store for the unified data-model catalog (DuckDB).

Everything the agents know about the data model lives here, in the same DuckDB file as the
data itself:

    catalog_tables / catalog_columns          tables, grain, hints, per-column notes
    catalog_relationships                     the join graph (with the evidence it was approved on)
    catalog_concepts                          semantic concepts (aliases, operations, filters)
    catalog_entity_domains                    live entity resolution domains
    catalog_meta                              a monotonically increasing ``version``
    catalog_audit                             who changed what, and when

    catalog_datasets / catalog_proposals      the *workflow* behind uploaded data: what was
                                              uploaded, and the changes waiting for a human

Built-in and uploaded objects share the same rows; ``origin`` ('builtin' | 'upload' |
'manual') is provenance, not a separate layer.  Built-in rows are seeded from
``db/catalog_seed.py``; a refresh never overwrites a row that was edited (``user_modified``).

DuckDB note: deleting and re-inserting the same primary key inside one transaction trips its
constraint checker, so every write here is ``INSERT OR REPLACE`` (plus deletes of *stale* rows
only), never delete-then-insert of the same key.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from db.catalog_model import EntityDomain, Relationship, TableMeta

USER_ORD_BASE = 1_000_000

_DDL = [
    "CREATE SEQUENCE IF NOT EXISTS catalog_audit_seq",
    "CREATE TABLE IF NOT EXISTS catalog_meta (key VARCHAR PRIMARY KEY, value VARCHAR)",
    """CREATE TABLE IF NOT EXISTS catalog_tables (
        name VARCHAR PRIMARY KEY, description VARCHAR, setup_hint VARCHAR, filter_hint VARCHAR,
        grain VARCHAR, default_filter VARCHAR, hidden_columns VARCHAR,
        agent_visible BOOLEAN DEFAULT TRUE, domain VARCHAR, origin VARCHAR, dataset_id VARCHAR,
        status VARCHAR DEFAULT 'active', ord INTEGER, seed_hash VARCHAR,
        user_modified BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS catalog_columns (
        table_name VARCHAR, column_name VARCHAR, note VARCHAR, role VARCHAR, unit VARCHAR,
        source_name VARCHAR, derived_from VARCHAR, ord INTEGER,
        PRIMARY KEY (table_name, column_name))""",
    """CREATE TABLE IF NOT EXISTS catalog_relationships (
        rel_key VARCHAR PRIMARY KEY, left_table VARCHAR, left_expr VARCHAR,
        right_table VARCHAR, right_expr VARCHAR, note VARCHAR, cardinality VARCHAR,
        confidence VARCHAR, bridge BOOLEAN, preferred BOOLEAN, grain_effect VARCHAR,
        origin VARCHAR, dataset_id VARCHAR, status VARCHAR DEFAULT 'approved', evidence VARCHAR,
        ord INTEGER, seed_hash VARCHAR, user_modified BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, approved_at TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS catalog_concepts (
        key VARCHAR PRIMARY KEY, definition VARCHAR, origin VARCHAR, dataset_id VARCHAR,
        status VARCHAR DEFAULT 'active', ord INTEGER, seed_hash VARCHAR,
        user_modified BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS catalog_entity_domains (
        name VARCHAR PRIMARY KEY, table_name VARCHAR, column_name VARCHAR, entity_type VARCHAR,
        description VARCHAR, display_column VARCHAR, match_mode VARCHAR, preferred_for VARCHAR,
        origin VARCHAR, dataset_id VARCHAR, status VARCHAR DEFAULT 'active', ord INTEGER,
        seed_hash VARCHAR, user_modified BOOLEAN DEFAULT FALSE)""",
    """CREATE TABLE IF NOT EXISTS catalog_audit (
        id BIGINT PRIMARY KEY, ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP, actor VARCHAR,
        action VARCHAR, object_type VARCHAR, object_key VARCHAR, detail VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS catalog_datasets (
        dataset_id VARCHAR PRIMARY KEY, table_name VARCHAR, title VARCHAR, description VARCHAR,
        grain VARCHAR, domain VARCHAR, status VARCHAR, source_filename VARCHAR,
        source_path VARCHAR, format VARCHAR, row_count BIGINT, staging_table VARCHAR,
        columns_json VARCHAR, notes_json VARCHAR, enrichments_json VARCHAR,
        pipeline_json VARCHAR, options_json VARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS catalog_proposals (
        proposal_id VARCHAR PRIMARY KEY, dataset_id VARCHAR, kind VARCHAR, status VARCHAR,
        title VARCHAR, summary VARCHAR, group_key VARCHAR, payload_json VARCHAR,
        evidence_json VARCHAR, ord INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, decided_at TIMESTAMP,
        decision_note VARCHAR)""",
]


def ensure_tables(conn) -> None:
    for stmt in _DDL:
        conn.execute(stmt)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _rows(conn, sql: str, params: Iterable[Any] = ()) -> list[dict]:
    params = list(params)
    cur = conn.execute(sql, params) if params else conn.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _replace(conn, table: str, row: dict) -> None:
    cols = list(row)
    conn.execute(
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        [row[c] for c in cols])


def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
                        .encode("utf-8")).hexdigest()[:16]


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _loads(text: Any, default: Any) -> Any:
    if text is None or text == "":
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _next_ord(conn, table: str) -> int:
    row = conn.execute(f"SELECT COALESCE(MAX(ord), 0) FROM {table}").fetchone()
    return max(int(row[0] or 0) + 1, USER_ORD_BASE)


@contextmanager
def transaction(conn) -> Iterator[None]:
    conn.execute("BEGIN")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Version + audit
# ---------------------------------------------------------------------------

def get_version(conn) -> int:
    row = conn.execute("SELECT value FROM catalog_meta WHERE key = 'version'").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def bump_version(conn) -> int:
    v = get_version(conn) + 1
    _replace(conn, "catalog_meta", {"key": "version", "value": str(v)})
    return v


def audit(conn, action: str, object_type: str, object_key: str,
          detail: Any = None, actor: str = "user") -> None:
    conn.execute(
        "INSERT INTO catalog_audit (id, actor, action, object_type, object_key, detail) "
        "VALUES (nextval('catalog_audit_seq'), ?, ?, ?, ?, ?)",
        [actor, action, object_type, object_key, _dumps(detail if detail is not None else {})])


# ---------------------------------------------------------------------------
# Row builders (shared by the seed sync and by uploads)
# ---------------------------------------------------------------------------

def _table_row(meta: TableMeta, *, domain: str | None, origin: str, dataset_id: str,
               ord_: int, status: str = "active") -> dict:
    return {
        "name": meta.name, "description": meta.description, "setup_hint": meta.setup_hint,
        "filter_hint": meta.filter_hint, "grain": meta.grain, "default_filter": meta.default_filter,
        "hidden_columns": _dumps(list(meta.hidden_columns)), "agent_visible": bool(meta.agent_visible),
        "domain": domain or meta.domain or "other", "origin": origin, "dataset_id": dataset_id or "",
        "status": status, "ord": ord_,
    }


def _rel_row(rel: Relationship, *, origin: str, dataset_id: str, evidence: dict, status: str, ord_: int) -> dict:
    return {
        "rel_key": rel.key(), "left_table": rel.left_table, "left_expr": rel.left_expr,
        "right_table": rel.right_table, "right_expr": rel.right_expr, "note": rel.note,
        "cardinality": rel.cardinality, "confidence": rel.confidence, "bridge": bool(rel.bridge),
        "preferred": bool(rel.preferred), "grain_effect": rel.grain_effect, "origin": origin,
        "dataset_id": dataset_id or "", "status": status, "evidence": _dumps(evidence or {}), "ord": ord_,
    }


def _domain_row(d: EntityDomain, *, origin: str, dataset_id: str, ord_: int, status: str = "active") -> dict:
    return {
        "name": d.name, "table_name": d.table, "column_name": d.column, "entity_type": d.entity_type,
        "description": d.description, "display_column": d.display_column or "",
        "match_mode": d.match_mode, "preferred_for": _dumps(list(d.preferred_for)),
        "origin": origin, "dataset_id": dataset_id or "", "status": status, "ord": ord_,
    }


# ---------------------------------------------------------------------------
# Seed sync (built-in definitions -> rows; never clobbers user edits)
# ---------------------------------------------------------------------------

def _write_notes(conn, table: str, notes: list[dict]) -> None:
    for j, n in enumerate(notes):
        _replace(conn, "catalog_columns", {
            "table_name": table, "column_name": n["column_name"], "note": n.get("note", ""),
            "role": n.get("role", ""), "unit": n.get("unit", ""), "source_name": n.get("source_name", ""),
            "derived_from": n.get("derived_from", ""), "ord": j})
    keep = [n["column_name"] for n in notes]
    if keep:
        marks = ", ".join("?" for _ in keep)
        conn.execute(f"DELETE FROM catalog_columns WHERE table_name = ? AND column_name NOT IN ({marks})",
                     [table, *keep])
    else:
        conn.execute("DELETE FROM catalog_columns WHERE table_name = ?", [table])


def sync_seed(conn) -> bool:
    """Insert missing built-in rows; refresh unmodified built-in rows whose definition changed."""
    from db import catalog_seed as seed

    changed = False

    have = {r["name"]: r for r in _rows(conn, "SELECT name, origin, seed_hash, user_modified FROM catalog_tables")}
    for i, (name, meta) in enumerate(seed.TABLES.items()):
        notes = [{"column_name": n.column, "note": n.note} for n in meta.column_notes]
        row = _table_row(meta, domain=seed.SEED_DOMAINS.get(name, "other"), origin="builtin",
                         dataset_id="", ord_=i)
        h = _hash([row, notes])
        cur = have.get(name)
        if cur is None or (cur["origin"] == "builtin" and not cur["user_modified"] and cur["seed_hash"] != h):
            row["seed_hash"] = h
            row["user_modified"] = False
            _replace(conn, "catalog_tables", row)
            _write_notes(conn, name, notes)
            changed = True

    have = {r["rel_key"]: r for r in _rows(conn, "SELECT rel_key, origin, seed_hash, user_modified FROM catalog_relationships")}
    for i, rel in enumerate(seed.RELATIONSHIPS):
        row = _rel_row(rel, origin="builtin", dataset_id="", evidence={}, status="approved", ord_=i)
        h = _hash(row)
        cur = have.get(rel.key())
        if cur is None or (cur["origin"] == "builtin" and not cur["user_modified"] and cur["seed_hash"] != h):
            row["seed_hash"] = h
            row["user_modified"] = False
            _replace(conn, "catalog_relationships", row)
            changed = True

    have = {r["key"]: r for r in _rows(conn, "SELECT key, origin, seed_hash, user_modified FROM catalog_concepts")}
    for i, (key, item) in enumerate(seed.SEMANTIC_GLOSSARY.items()):
        definition = _dumps(item)
        h = _hash([key, definition])
        cur = have.get(key)
        if cur is None or (cur["origin"] == "builtin" and not cur["user_modified"] and cur["seed_hash"] != h):
            _replace(conn, "catalog_concepts", {
                "key": key, "definition": definition, "origin": "builtin", "dataset_id": "",
                "status": "active", "ord": i, "seed_hash": h, "user_modified": False})
            changed = True

    have = {r["name"]: r for r in _rows(conn, "SELECT name, origin, seed_hash, user_modified FROM catalog_entity_domains")}
    for i, d in enumerate(seed.ENTITY_DOMAINS):
        row = _domain_row(d, origin="builtin", dataset_id="", ord_=i)
        h = _hash(row)
        cur = have.get(d.name)
        if cur is None or (cur["origin"] == "builtin" and not cur["user_modified"] and cur["seed_hash"] != h):
            row["seed_hash"] = h
            row["user_modified"] = False
            _replace(conn, "catalog_entity_domains", row)
            changed = True

    if changed:
        bump_version(conn)
    return changed


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_all(conn) -> dict:
    return {
        "version": get_version(conn),
        "tables": _rows(conn, "SELECT * FROM catalog_tables WHERE status = 'active' ORDER BY ord, name"),
        "columns": _rows(
            conn,
            "SELECT c.* FROM catalog_columns c JOIN catalog_tables t ON t.name = c.table_name "
            "WHERE t.status = 'active' ORDER BY c.table_name, c.ord, c.column_name"),
        "relationships": _rows(
            conn, "SELECT * FROM catalog_relationships WHERE status = 'approved' ORDER BY ord, rel_key"),
        "concepts": _rows(conn, "SELECT * FROM catalog_concepts WHERE status = 'active' ORDER BY ord, key"),
        "entity_domains": _rows(
            conn, "SELECT * FROM catalog_entity_domains WHERE status = 'active' ORDER BY ord, name"),
    }


def relationship_meta(conn) -> list[dict]:
    return _rows(conn, "SELECT rel_key, evidence, approved_at, status FROM catalog_relationships")


# ---------------------------------------------------------------------------
# Upserts used by uploads / manual edits
# ---------------------------------------------------------------------------

def upsert_table(conn, meta: TableMeta, *, columns: list[dict] | None = None, origin: str = "upload",
                 dataset_id: str = "", domain: str | None = None, user_modified: bool = False) -> None:
    existing = conn.execute("SELECT ord FROM catalog_tables WHERE name = ?", [meta.name]).fetchone()
    ord_ = int(existing[0]) if existing and existing[0] is not None else _next_ord(conn, "catalog_tables")
    row = _table_row(meta, domain=domain, origin=origin, dataset_id=dataset_id, ord_=ord_)
    row["seed_hash"] = ""
    row["user_modified"] = bool(user_modified)
    _replace(conn, "catalog_tables", row)
    notes = list(columns) if columns else [{"column_name": n.column, "note": n.note, "role": n.role,
                                            "unit": n.unit, "source_name": n.source_name}
                                           for n in meta.column_notes]
    _write_notes(conn, meta.name, notes)
    audit(conn, "register_table", "table", meta.name, {"origin": origin, "dataset_id": dataset_id})


def upsert_column_note(conn, table: str, column: str, *, note: str = "", role: str = "", unit: str = "",
                       source_name: str = "", derived_from: str = "") -> None:
    """Add or replace the note for one column (e.g. a derived key column added on approval)."""
    row = conn.execute("SELECT ord FROM catalog_columns WHERE table_name = ? AND column_name = ?", [table, column]).fetchone()
    if row and row[0] is not None:
        ord_ = int(row[0])
    else:
        ord_ = int(conn.execute("SELECT COALESCE(MAX(ord), -1) + 1 FROM catalog_columns WHERE table_name = ?", [table]).fetchone()[0])
    _replace(conn, "catalog_columns", {
        "table_name": table, "column_name": column, "note": note, "role": role, "unit": unit,
        "source_name": source_name, "derived_from": derived_from, "ord": ord_})


def update_table_field(conn, name: str, field: str, value: Any) -> None:
    if field not in {"description", "grain", "setup_hint", "filter_hint", "default_filter", "domain"}:
        raise ValueError(f"Field '{field}' is not editable.")
    conn.execute(f"UPDATE catalog_tables SET {field} = ?, user_modified = TRUE WHERE name = ?", [value, name])
    audit(conn, "edit_table", "table", name, {field: value})


def upsert_relationship(conn, rel: Relationship, *, origin: str = "upload", dataset_id: str = "",
                        evidence: dict | None = None, status: str = "approved") -> None:
    existing = conn.execute("SELECT ord FROM catalog_relationships WHERE rel_key = ?", [rel.key()]).fetchone()
    ord_ = int(existing[0]) if existing and existing[0] is not None else _next_ord(conn, "catalog_relationships")
    row = _rel_row(rel, origin=origin, dataset_id=dataset_id, evidence=evidence or {}, status=status, ord_=ord_)
    row["seed_hash"] = ""
    row["user_modified"] = origin != "builtin"
    if status == "approved":
        row["approved_at"] = datetime.datetime.now()
    _replace(conn, "catalog_relationships", row)
    audit(conn, "register_relationship", "relationship", rel.key(), {"origin": origin, "status": status})


def set_relationship_status(conn, rel_key: str, status: str, reason: str = "") -> bool:
    if not conn.execute("SELECT 1 FROM catalog_relationships WHERE rel_key = ?", [rel_key]).fetchone():
        return False
    conn.execute("UPDATE catalog_relationships SET status = ?, user_modified = TRUE WHERE rel_key = ?",
                 [status, rel_key])
    audit(conn, f"relationship_{status}", "relationship", rel_key, {"reason": reason})
    return True


def upsert_concept(conn, key: str, definition: dict, *, origin: str = "upload", dataset_id: str = "") -> None:
    existing = conn.execute("SELECT ord FROM catalog_concepts WHERE key = ?", [key]).fetchone()
    ord_ = int(existing[0]) if existing and existing[0] is not None else _next_ord(conn, "catalog_concepts")
    _replace(conn, "catalog_concepts", {
        "key": key, "definition": _dumps(definition), "origin": origin, "dataset_id": dataset_id or "",
        "status": "active", "ord": ord_, "seed_hash": "", "user_modified": origin != "builtin"})
    audit(conn, "register_concept", "concept", key, {"origin": origin})


def upsert_entity_domain(conn, domain: EntityDomain, *, origin: str = "upload", dataset_id: str = "") -> None:
    existing = conn.execute("SELECT ord FROM catalog_entity_domains WHERE name = ?", [domain.name]).fetchone()
    ord_ = int(existing[0]) if existing and existing[0] is not None else _next_ord(conn, "catalog_entity_domains")
    row = _domain_row(domain, origin=origin, dataset_id=dataset_id, ord_=ord_)
    row["seed_hash"] = ""
    row["user_modified"] = origin != "builtin"
    _replace(conn, "catalog_entity_domains", row)
    audit(conn, "register_entity_domain", "entity_domain", domain.name, {"origin": origin})


def retire_dataset(conn, dataset_id: str) -> dict:
    """Deactivate everything a dataset added (rows stay for audit; the data table is kept)."""
    names = [r[0] for r in conn.execute("SELECT name FROM catalog_tables WHERE dataset_id = ?", [dataset_id]).fetchall()]
    counts = {"tables": len(names)}
    conn.execute("UPDATE catalog_tables SET status = 'archived' WHERE dataset_id = ?", [dataset_id])
    conn.execute("UPDATE catalog_concepts SET status = 'archived' WHERE dataset_id = ?", [dataset_id])
    conn.execute("UPDATE catalog_entity_domains SET status = 'archived' WHERE dataset_id = ?", [dataset_id])
    n_rel = 0
    for key, in conn.execute(
            "SELECT rel_key FROM catalog_relationships WHERE dataset_id = ? AND status = 'approved'", [dataset_id]).fetchall():
        n_rel += 1
        conn.execute("UPDATE catalog_relationships SET status = 'revoked' WHERE rel_key = ?", [key])
    counts["relationships"] = n_rel
    audit(conn, "retire_dataset", "dataset", dataset_id, {"tables": names})
    return counts


# ---------------------------------------------------------------------------
# Workflow: datasets and proposals
# ---------------------------------------------------------------------------

_DATASET_JSON = {"columns_json": "columns", "notes_json": "notes", "enrichments_json": "enrichments",
                 "pipeline_json": "pipeline", "options_json": "options"}


def _dataset_from_row(r: dict) -> dict:
    d = {k: r.get(k) for k in ("dataset_id", "table_name", "title", "description", "grain", "domain",
                              "status", "source_filename", "source_path", "format", "row_count",
                              "staging_table")}
    for col, key in _DATASET_JSON.items():
        d[key] = _loads(r.get(col), {} if key == "options" else [])
    d["created_at"] = str(r.get("created_at") or "")
    d["updated_at"] = str(r.get("updated_at") or "")
    return d


def save_dataset(conn, d: dict) -> None:
    row = {k: d.get(k) for k in ("dataset_id", "table_name", "title", "description", "grain", "domain",
                                 "status", "source_filename", "source_path", "format", "row_count",
                                 "staging_table")}
    for col, key in _DATASET_JSON.items():
        row[col] = _dumps(d.get(key, {} if key == "options" else []))
    if conn.execute("SELECT 1 FROM catalog_datasets WHERE dataset_id = ?", [row["dataset_id"]]).fetchone():
        sets = [c for c in row if c != "dataset_id"]
        conn.execute(
            f"UPDATE catalog_datasets SET {', '.join(c + ' = ?' for c in sets)}, "
            f"updated_at = CURRENT_TIMESTAMP WHERE dataset_id = ?",
            [row[c] for c in sets] + [row["dataset_id"]])
    else:
        _replace(conn, "catalog_datasets", row)


def get_dataset(conn, dataset_id: str) -> dict | None:
    rows = _rows(conn, "SELECT * FROM catalog_datasets WHERE dataset_id = ?", [dataset_id])
    return _dataset_from_row(rows[0]) if rows else None


def list_datasets(conn) -> list[dict]:
    return [_dataset_from_row(r) for r in _rows(conn, "SELECT * FROM catalog_datasets ORDER BY created_at DESC, dataset_id")]


def _proposal_from_row(r: dict) -> dict:
    return {"proposal_id": r["proposal_id"], "dataset_id": r["dataset_id"], "kind": r["kind"],
            "status": r["status"], "title": r["title"], "summary": r["summary"],
            "group_key": r["group_key"] or "", "payload": _loads(r["payload_json"], {}),
            "evidence": _loads(r["evidence_json"], {}), "ord": r["ord"],
            "created_at": str(r.get("created_at") or ""), "decided_at": str(r.get("decided_at") or ""),
            "decision_note": r.get("decision_note") or ""}


def replace_proposals(conn, dataset_id: str, proposals: list[dict]) -> list[dict]:
    """Supersede the dataset's pending proposals and store a fresh set."""
    conn.execute("UPDATE catalog_proposals SET status = 'superseded' WHERE dataset_id = ? AND status = 'pending'",
                 [dataset_id])
    out = []
    for i, p in enumerate(proposals):
        pid = p.get("proposal_id") or uuid.uuid4().hex[:12]
        _replace(conn, "catalog_proposals", {
            "proposal_id": pid, "dataset_id": dataset_id, "kind": p["kind"], "status": "pending",
            "title": p.get("title", ""), "summary": p.get("summary", ""), "group_key": p.get("group_key", ""),
            "payload_json": _dumps(p.get("payload", {})), "evidence_json": _dumps(p.get("evidence", {})),
            "ord": i})
        out.append(get_proposal(conn, pid))
    return out


def get_proposal(conn, proposal_id: str) -> dict | None:
    rows = _rows(conn, "SELECT * FROM catalog_proposals WHERE proposal_id = ?", [proposal_id])
    return _proposal_from_row(rows[0]) if rows else None


def list_proposals(conn, dataset_id: str, include_superseded: bool = False) -> list[dict]:
    sql = "SELECT * FROM catalog_proposals WHERE dataset_id = ?"
    if not include_superseded:
        sql += " AND status <> 'superseded'"
    return [_proposal_from_row(r) for r in _rows(conn, sql + " ORDER BY ord, proposal_id", [dataset_id])]


def update_proposal(conn, proposal_id: str, *, status: str | None = None, decision_note: str | None = None,
                    payload: dict | None = None, evidence: dict | None = None) -> None:
    sets, params = [], []
    if status is not None:
        sets.append("status = ?"); params.append(status)
        if status in {"approved", "rejected"}:
            sets.append("decided_at = CURRENT_TIMESTAMP")
    if decision_note is not None:
        sets.append("decision_note = ?"); params.append(decision_note)
    if payload is not None:
        sets.append("payload_json = ?"); params.append(_dumps(payload))
    if evidence is not None:
        sets.append("evidence_json = ?"); params.append(_dumps(evidence))
    if sets:
        conn.execute(f"UPDATE catalog_proposals SET {', '.join(sets)} WHERE proposal_id = ?", params + [proposal_id])
