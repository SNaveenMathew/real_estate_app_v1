"""Semantic fallback: when no catalog alias matches, a model may PROPOSE concept keys - and nothing else.

Alias matching is precise and instant but cannot cover every phrasing ("How big is Austin?" names no alias), and the
catalog will keep growing.  A model is good at exactly this narrow job - "which of these named concepts is the user
asking about?" - and bad at the rest, so it is given only that job, under a strict contract:

* **a closed list in, keys out** - the model sees concept keys with one-line descriptions and returns JSON keys.  It never
  writes SQL, never names a table or a column, and cannot invent a concept: anything not in the list is dropped;
* **proposal, not decision** - proposed keys enter the plan exactly as if an alias had matched (``schema.forcing_concepts``).
  Entity resolution, answer routing (ambiguity, derived measures, known gaps), SQL generation, the plan contract and
  execution are all unchanged and deterministic;
* **transparent** - the evidence tells the answer model (and so the user) how the request was interpreted;
* **bounded** - used only when the plan is otherwise EMPTY (so no working request pays for it), one call at most, its own
  short timeout and token cap, no retries, cached per (catalog, request), and skipped when the model is known to be down
  (``agents/llm_resilience``).  Disabled entirely by ``settings.semantic_fallback_enabled = False``.

Boundary, by design: it rescues EMPTY plans only.  A request that matches the wrong alias is not second-guessed - verifying
working matches would put a model call on flows that already work.
"""
from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import db.schema_catalog as schema
from agents import llm_resilience
from config import settings

MAX_KEYS = 3
_SYSTEM = (
    "You map a user's question to the data-model concepts that can answer it.\n"
    "Choose ONLY from the CONCEPTS list. Reply with JSON only, no prose: {\"concepts\": [\"key\", ...]}\n"
    "Use at most 3 keys, best first. If nothing in the list fits, reply {\"concepts\": []}.\n"
    "Never write SQL. Never invent a key. A concept marked (NOT LOADED) means the data does not exist: choose it when the "
    "user asks about that topic."
)

_client: ChatOpenAI | None = None
_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_lock = threading.Lock()
_CACHE_SIZE = 256


def _get_client() -> ChatOpenAI:
    global _client
    if _client is None:
        _client = ChatOpenAI(
            base_url=settings.llama_server_base_url, api_key="not-needed", model=settings.llama_server_model,
            temperature=0.0, max_tokens=120,                       # a key list is a few dozen tokens
            timeout=settings.semantic_fallback_timeout_s, max_retries=0)
    return _client


def _menu() -> list[tuple[str, str]]:
    """(key, one-line description) for every concept; compact so the prompt stays small as the catalog grows."""
    items = []
    for key, concept in sorted(schema.SEMANTIC_GLOSSARY.items()):
        summary = (concept.get("description") or "").split(". ")[0].strip().rstrip(".")[:120]
        if concept.get("gap"):
            summary = f"(NOT LOADED) {summary}"
        items.append((key, summary))
    return items


def parse_keys(text: str, allowed: set) -> list[str]:
    """The valid keys in a model reply (tolerates prose or code fences around the JSON); [] when nothing usable."""
    match = re.search(r"\{.*\}|\[.*\]", text or "", re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except Exception:
        return []
    raw = data.get("concepts") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return []
    keys: list[str] = []
    for k in raw:
        if isinstance(k, str) and k in allowed and k not in keys:
            keys.append(k)
    return keys[:MAX_KEYS]


def propose_concepts(request: str) -> list[str]:
    """Validated concept keys the model proposes for ``request`` ([] when disabled, unavailable, or nothing fits)."""
    if not settings.semantic_fallback_enabled or not request or not request.strip():
        return []
    menu = _menu()
    cache_key = (hash(tuple(menu)), schema._normalize(request))
    with _lock:
        hit = _cache.get(cache_key)
        if hit is not None:
            _cache.move_to_end(cache_key)
            return list(hit)
    listing = "\n".join(f"- {k}: {d}" for k, d in menu)
    try:
        reply = llm_resilience.invoke(
            _get_client(), [SystemMessage(content=_SYSTEM), HumanMessage(content=f"CONCEPTS:\n{listing}\n\nQUESTION: {request.strip()[:500]}")],
            role="semantic_fallback")
    except llm_resilience.ModelUnavailable:
        return []                                                  # transient: not cached, and the request is then simply unplannable
    content = getattr(reply, "content", reply)
    keys = parse_keys(content if isinstance(content, str) else str(content), {k for k, _ in menu})
    with _lock:
        _cache[cache_key] = tuple(keys)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return keys


def describe(keys: list[str]) -> str:
    """How the request was interpreted, for the evidence."""
    parts = []
    for k in keys:
        concept = schema.SEMANTIC_GLOSSARY.get(k, {})
        parts.append((concept.get("description") or k).split(". ")[0].strip().rstrip("."))
    return "; ".join(parts)


def reset_cache() -> None:
    with _lock:
        _cache.clear()
