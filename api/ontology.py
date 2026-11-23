"""HTTP API for the app's ontology and its graph projection (``/ontology``).

    GET  /api/ontology                classes, properties, typed edges, vocabulary, stats, issues
    GET  /api/ontology?format=jsonld  the same document as JSON-LD ``@graph``
    GET  /api/ontology/graph          nodes/edges at one granularity level (overview|standard|full)

Everything here is read-only and derived from the live catalog (``db/ontology.py``), so an approved
dataset appears immediately with no restart.  The handlers are plain ``def``, so FastAPI runs them
in its worker pool and each gets its own DuckDB cursor; the document is memoised against the catalog
version, and the graph is a cheap filter over that memoised document.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

import db.duckdb_store as store
from db import ontology

router = APIRouter(prefix="/api/ontology", tags=["ontology"])


@router.get("")
def ontology_document(
    format: str = Query(default="json", description="json (default) or jsonld"),
    counts: bool = Query(default=False, description="also count the live values behind each entity type"),
):
    store.get_conn()
    doc = ontology.build_ontology(with_counts=counts)
    if format.lower() in {"jsonld", "json-ld"}:
        return ontology.as_jsonld(doc)
    return doc


@router.get("/graph")
def ontology_graph(
    detail: str = Query(default="standard", description="overview | standard | full"),
):
    """The page's model: only the node kinds the level includes, and only edges between them."""
    store.get_conn()
    return ontology.build_graph(detail=detail)
