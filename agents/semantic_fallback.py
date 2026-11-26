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
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, create_model

import db.schema_catalog as schema
from agents import llm_resilience
from config import LLM_STOP_SEQUENCES, settings
from observability import set_span_input, set_span_output, trace_span
from services.guardrails import OutlinesStructuredClient

MAX_KEYS = 3
_SYSTEM = (
    "You map a user's question to the data-model concepts that can answer it.\n"
    'Choose ONLY from the CONCEPTS list. Reply with JSON only, no prose: {"concepts": ["key", ...]}\n'
    'Use at most 3 keys, best first. If nothing in the list fits, reply {"concepts": []}.\n'
    "Never write SQL. Never invent a key. A concept marked (NOT LOADED) means the data does not exist: choose it when the "
    "user asks about that topic."
)

_client: Any = None
_client_menu_sig: tuple[str, ...] | None = None
_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_lock = threading.Lock()
_CACHE_SIZE = 256


class _EmptyConceptProposal(BaseModel):
    concepts: list[str] = Field(default_factory=list, max_length=MAX_KEYS)


def _proposal_schema(allowed_keys: tuple[str, ...]) -> type[BaseModel]:
    if not allowed_keys:
        return _EmptyConceptProposal
    key_type = Literal[tuple(allowed_keys)]  # type: ignore[valid-type]
    return create_model(
        "ConceptProposal",
        concepts=(list[key_type], Field(default_factory=list, max_length=MAX_KEYS)),  # type: ignore[valid-type]
    )


def _get_client(allowed_keys: tuple[str, ...] = ()) -> Any:
    global _client, _client_menu_sig
    # Respect test stubs injected via `monkeypatch.setattr(sf, "_client", stub)`.
    if _client is not None and not isinstance(_client, OutlinesStructuredClient):
        return _client
    sig = tuple(allowed_keys) if allowed_keys else tuple(sorted(schema.SEMANTIC_GLOSSARY))
    if _client is None or (sig and _client_menu_sig != sig):
        _client = OutlinesStructuredClient(
            base_url=settings.llama_server_base_url,
            api_key="not-needed",
            model=settings.llama_server_model,
            output_schema=_proposal_schema(sig),
            temperature=0.0,
            max_tokens=256,
            timeout=settings.semantic_fallback_timeout_s,
            max_retries=0,
            stop=LLM_STOP_SEQUENCES,
        )
        _client_menu_sig = sig
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


def _extract_valid_json(text: str) -> tuple[bool, Any]:
    match = re.search(r"\{.*\}|\[.*\]", text or "", re.S)
    if not match:
        return False, None
    try:
        return True, json.loads(match.group(0))
    except Exception:
        return False, None


def parse_keys(text: str, allowed: set) -> list[str]:
    """The valid keys in a model reply (tolerates prose or code fences around the JSON); [] when nothing usable."""
    ok, data = _extract_valid_json(text)
    if not ok:
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
    with trace_span(
        "semantic_fallback.propose_concepts",
        attributes={"openinference.span.kind": "CHAIN"},
    ) as span:
        set_span_input(span, {"request": request.strip()})
        menu = _menu()
        cache_key = (hash(tuple(menu)), schema._normalize(request))
        with _lock:
            hit = _cache.get(cache_key)
            if hit is not None:
                _cache.move_to_end(cache_key)
                keys = list(hit)
                set_span_output(span, {"cache_hit": True, "concepts": keys}, mime_type="application/json")
                return keys
        listing = "\n".join(f"- {k}: {d}" for k, d in menu)
        allowed_keys = tuple(k for k, _ in menu)
        try:
            reply = llm_resilience.invoke(
                _get_client(allowed_keys),
                [
                    SystemMessage(content=_SYSTEM),
                    HumanMessage(content=f"CONCEPTS:\n{listing}\n\nQUESTION: {request.strip()[:500]}"),
                ],
                role="semantic_fallback",
                breaker_name="semantic",
            )  # its own breaker: it must never disable the chat models
        except llm_resilience.ModelUnavailable as exc:
            set_span_output(span, {"cache_hit": False, "error": str(exc), "concepts": []}, mime_type="application/json")
            return []  # transient: not cached, and the request is then simply unplannable
        content = getattr(reply, "content", reply)
        raw_text = content if isinstance(content, str) else str(content)
        valid_json, _parsed = _extract_valid_json(raw_text)
        keys = parse_keys(raw_text, set(allowed_keys))
        set_span_output(
            span,
            {
                "cache_hit": False,
                "raw_length": len(raw_text),
                "concepts": keys,
            },
            mime_type="application/json",
        )
        if valid_json:
            # Never poison the cache with unparseable / truncated completions; only cache valid structured replies.
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
