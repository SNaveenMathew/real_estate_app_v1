"""
ChromaDB vector store.
One collection `house_documents` stores all text/image descriptions.
Each document is tagged with house_id so we can filter per house.
"""
import hashlib
import base64
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_ollama import OllamaEmbeddings

from config import settings


_client: Optional[chromadb.ClientAPI] = None
_collection = None
_embeddings: Optional[OllamaEmbeddings] = None


def _get_client():
    global _client
    if _client is None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def _get_collection():
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(
            name="house_documents",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )
    return _embeddings


def _make_id(house_id: str, text: str) -> str:
    h = hashlib.md5(f"{house_id}:{text[:200]}".encode()).hexdigest()
    return f"{house_id}_{h}"


def add_document(house_id: str, text: str, doc_type: str = "description",
                 metadata: dict = None) -> str:
    """Embed and store a text document for a house. Returns document id."""
    col = _get_collection()
    emb = _get_embeddings()

    doc_id = _make_id(house_id, text)
    meta = {"house_id": house_id, "doc_type": doc_type}
    if metadata:
        meta.update(metadata)

    # Check duplicate
    existing = col.get(ids=[doc_id])
    if existing["ids"]:
        return doc_id  # already stored

    vector = emb.embed_query(text)
    col.add(ids=[doc_id], embeddings=[vector], documents=[text], metadatas=[meta])
    return doc_id


def add_photo(house_id: str, file_path: Path, caption: str = "") -> str:
    """Store a photo reference. Caption (if provided) is embedded."""
    text = f"[Photo] {file_path.name}. {caption}".strip()
    return add_document(house_id, text, doc_type="photo",
                        metadata={"file_path": str(file_path)})


def search_house(house_id: str, query: str, n_results: int = 4) -> list[dict]:
    """Semantic search within a single house's documents."""
    col = _get_collection()
    emb = _get_embeddings()
    vector = emb.embed_query(query)
    results = col.query(
        query_embeddings=[vector],
        n_results=n_results,
        where={"house_id": {"$eq": house_id}},
        include=["documents", "metadatas", "distances"],
    )
    return _format_results(results)


def search_all(query: str, n_results: int = 6) -> list[dict]:
    """Semantic search across all houses."""
    col = _get_collection()
    emb = _get_embeddings()
    vector = emb.embed_query(query)
    results = col.query(
        query_embeddings=[vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return _format_results(results)


def get_house_documents(house_id: str) -> list[dict]:
    """Return all stored documents for a house."""
    col = _get_collection()
    results = col.get(where={"house_id": {"$eq": house_id}},
                      include=["documents", "metadatas"])
    if not results["ids"]:
        return []
    return [
        {"id": i, "text": d, "metadata": m}
        for i, d, m in zip(results["ids"], results["documents"], results["metadatas"])
    ]


def has_document(house_id: str, text: str) -> bool:
    doc_id = _make_id(house_id, text)
    col = _get_collection()
    return bool(col.get(ids=[doc_id])["ids"])


def get_description(house_id: str) -> Optional[dict]:
    """Return the single stored description document for a house, or None.
    A "description" is just a document tagged doc_type='description' —
    filtered out here from other doc types (e.g. 'photo') stored in the
    same collection."""
    for d in get_house_documents(house_id):
        if d["metadata"].get("doc_type") == "description":
            return d
    return None


def upsert_description(house_id: str, text: str) -> str:
    """Replace the single stored description for a house. Enforces at most
    one description per house by deleting any existing one first (unlike
    add_document, which is append-only / dedupes by content hash)."""
    existing = get_description(house_id)
    if existing:
        delete_document(existing["id"])
    return add_document(house_id, text, doc_type="description")


def delete_document(doc_id: str) -> None:
    """Delete a single stored document (description or photo) by its id."""
    _get_collection().delete(ids=[doc_id])


def _format_results(results: dict) -> list[dict]:
    out = []
    if not results["ids"] or not results["ids"][0]:
        return out
    for doc_id, doc, meta, dist in zip(
        results["ids"][0], results["documents"][0],
        results["metadatas"][0], results["distances"][0]
    ):
        out.append({"id": doc_id, "text": doc, "metadata": meta, "score": 1 - dist})
    return out


def collection_count() -> int:
    return _get_collection().count()

# ─────────────────────────────────────────────────────────────────────────────
# Data-model RAG
# ─────────────────────────────────────────────────────────────────────────────

def _schema_collection():
    client = _get_client()
    return client.get_or_create_collection(
        name="data_model_metadata",
        metadata={"hnsw:space": "cosine", "purpose": "schema_relationship_rag"},
    )


def _schema_documents():
    """Build deterministic retrieval documents from the declarative catalog."""
    from db import schema_catalog as schema
    docs = []
    for name, meta in schema.TABLES.items():
        if not meta.agent_visible:
            continue
        cols = ", ".join(c for c, _ in schema._live_columns(name) if c not in meta.hidden_columns)
        notes = " ".join(f"{n.column}: {n.note}" for n in meta.column_notes)
        docs.append((f"table:{name}", f"TABLE {name}. {meta.description} Grain={meta.grain}. Columns={cols}. Notes={notes}"))
    for rel in schema.RELATIONSHIPS:
        docs.append((f"relationship:{rel.key()}", f"RELATIONSHIP {rel.render()}"))
    for domain in schema.ENTITY_DOMAINS:
        docs.append((f"entity:{domain.name}", f"ENTITY DOMAIN {domain.name}. Type={domain.entity_type}. Source={domain.table}.{domain.column}. Match={domain.match_mode}. {domain.description}. Preferred for={domain.preferred_for}"))
    for key, item in schema.SEMANTIC_GLOSSARY.items():
        ops = "; ".join(str(o) for o in item.get("operations", []))
        docs.append((f"semantic:{key}", f"SEMANTIC {key}. {item.get('description','')} Tables={item.get('tables',[])} Columns={item.get('columns',[])} Aliases={item.get('aliases',[])} Operations={ops} Filters={item.get('filters',[])} Grain={item.get('grain','')} EntityTypes={item.get('entity_types',[])} Rollup={item.get('rollup',False)} RollupSpec={item.get('rollup_spec',{})} RequiredTerms={item.get('required_terms',[])}"))
    return docs


_SYNCED_CATALOG_VERSION = None


def ensure_schema_metadata_index(force: bool = False) -> int:
    """Synchronize the small metadata corpus into Chroma - incrementally.

    The corpus is built from the live, unified catalog (built-in and user-added sources alike).
    It is re-synchronized only when the catalog version changes (or ``force``), and only the
    documents whose text changed are re-embedded. Approving a dataset on the Data page therefore
    reaches the next chat turn without re-embedding the whole corpus on every question.

    The application must remain usable when Ollama is not running: in that case vector indexing is
    skipped and ``search_data_model`` falls back to the deterministic lexical path.
    """
    global _SYNCED_CATALOG_VERSION
    from db import schema_catalog as schema
    docs = _schema_documents()
    version = schema.catalog_version()
    if not force and _SYNCED_CATALOG_VERSION == version:
        return len(docs)
    col = _schema_collection()
    try:
        emb = _get_embeddings()
    except Exception:
        return len(docs)
    desired = dict(docs)
    existing = col.get(include=["documents"])
    have = dict(zip(existing.get("ids", []), existing.get("documents", [])))
    stale = [i for i in have if i not in desired]
    if stale:
        col.delete(ids=stale)
    changed = [(i, t) for i, t in docs if have.get(i) != t]
    if changed:
        vectors = emb.embed_documents([t for _, t in changed])
        col.upsert(
            ids=[i for i, _ in changed],
            embeddings=vectors,
            documents=[t for _, t in changed],
            metadatas=[{"kind": i.split(":", 1)[0]} for i, _ in changed],
        )
    _SYNCED_CATALOG_VERSION = version
    return len(docs)


def search_data_model(query: str, n_results: int = 10) -> list[dict]:
    """Retrieve relevant data-model metadata: vector search, then a lexical fallback over the LIVE catalog."""
    try:
        ensure_schema_metadata_index()
        col = _schema_collection()
        emb = _get_embeddings()
        vector = emb.embed_query(query)
        results = col.query(
            query_embeddings=[vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        formatted = _format_results(results)
        if formatted:
            return formatted
    except Exception:
        # Metadata retrieval is a grounding aid, not a reason to fail the whole chat.
        pass

    # Deterministic fallback: rank the metadata corpus by token overlap. The corpus comes from the same
    # unified catalog the planner reads, so it is current even if Chroma/Ollama are down or the vector
    # index has not been refreshed since the last approval.
    import re
    q_tokens = set(re.findall(r"[a-z0-9_]+", (query or "").lower()))
    scored = []
    for doc_id, doc in _schema_documents():
        d_tokens = set(re.findall(r"[a-z0-9_]+", (doc or "").lower()))
        score = len(q_tokens & d_tokens)
        if score:
            scored.append({"id": doc_id, "text": doc, "metadata": {"kind": doc_id.split(":", 1)[0]}, "score": score})
    scored.sort(key=lambda x: (-x["score"], x["id"]))
    return scored[:n_results]
