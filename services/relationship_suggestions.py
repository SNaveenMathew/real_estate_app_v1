"""Model-suggested relationship CANDIDATES for uploaded datasets: the model widens what is TESTED, never what is ACCEPTED.

``discover_relationships`` only evaluates a pair of columns when their NAMES look alike (or a value pattern places both in
one key family).  A relationship such as ``locale`` <-> ``houses.city`` is therefore never even tested, however complete
the overlap of values.  A model reading the column names, types and sample values can spot such pairs - and it is also
exactly the kind of guess that must not be trusted on its own.  So:

* the model proposes (source column, target table.column) PAIRS, from closed lists, as JSON - nothing else;
* every pair is validated (both columns exist, same type family) and then goes through the SAME value-containment
  evaluation and the SAME acceptance thresholds as a name-based candidate: a pair whose values do not actually match
  produces no proposal at all;
* an accepted pair is still only a PROPOSAL (needs the user's approval in the onboarding flow), labelled
  ``origin: llm-suggested`` with a warning, so the person reviewing it knows how it was found;
* opt-in (``settings.relationship_llm_suggestions_enabled``), one bounded call per dataset (own client, short timeout, token
  cap, no retries, own circuit breaker), and it can never raise into onboarding: any failure means "no extra candidates",
  i.e. exactly the previous behaviour.
"""
from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents import llm_resilience
from config import settings

log = logging.getLogger(__name__)

MAX_PAIRS = 10
MAX_PER_SOURCE = 3
MAX_TARGETS_SHOWN = 150
_SYSTEM = (
    "You help join a user's uploaded table to existing tables.\n"
    "From SOURCE COLUMNS (name, type, value statistics, sample values) and TARGET COLUMNS (table.column, type), list pairs "
    "whose VALUES likely refer to the same thing (same identifier, same place names, same codes), even when the names differ.\n"
    "Reply with JSON only, no prose: {\"pairs\": [{\"source\": \"<source column>\", \"target\": \"<table.column>\"}]}\n"
    f"At most {MAX_PAIRS} pairs, best first. Use only names from the lists. If nothing is plausible reply {{\"pairs\": []}}."
)

_client: ChatOpenAI | None = None


def _get_client() -> ChatOpenAI:
    global _client
    if _client is None:
        _client = ChatOpenAI(
            base_url=settings.llama_server_base_url, api_key="not-needed", model=settings.llama_server_model,
            temperature=0.0, max_tokens=400, timeout=settings.relationship_llm_timeout_s, max_retries=0)
    return _client


def parse_pairs(text: str, sources: set[str], targets: dict[str, dict]) -> list[tuple[str, dict]]:
    """Valid (source column, target dict) pairs from a model reply; everything else is dropped."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0)).get("pairs")
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: list[tuple[str, dict]] = []
    per_source: dict[str, int] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        src, tgt = item.get("source"), item.get("target")
        if not isinstance(src, str) or not isinstance(tgt, str) or src not in sources or tgt not in targets:
            continue                                                   # unknown names: invented, not suggested
        if any(src == s and targets[tgt] is t for s, t in out) or per_source.get(src, 0) >= MAX_PER_SOURCE:
            continue
        out.append((src, targets[tgt]))
        per_source[src] = per_source.get(src, 0) + 1
        if len(out) >= MAX_PAIRS:
            break
    return out


def suggest_pairs(prof: dict, included: list[dict], targets: list[dict]) -> dict[str, list[dict]]:
    """``{source column: [target dicts]}`` the model suggests; ``{}`` when disabled, unavailable, or nothing plausible."""
    if not settings.relationship_llm_suggestions_enabled:
        return {}
    sources = {}
    for col in included:
        p = prof.get(col["name"])
        if not p or p.get("family") not in ("text", "int") or col.get("role") in ("measure", "geo", "date", "other"):
            continue
        if p.get("distinct", 0) < 2 or col.get("key_kind") == "address":
            continue
        sources[col["name"]] = p
    families = {p["family"] for p in sources.values()}
    shown = [t for t in targets if t["family"] in families][:MAX_TARGETS_SHOWN]
    if not sources or not shown:
        return {}
    target_index = {f"{t['table']}.{t['column']}": t for t in shown}
    src_lines = []
    for name, p in sources.items():
        stats = f"{p['family']}, {p['distinct']} distinct"
        if p["family"] == "text":
            stats += f", length {p.get('min_len')}-{p.get('max_len')}, {round(100 * p.get('digit_frac', 0))}% all-digits"
        src_lines.append(f"- {name} ({stats}); samples: {', '.join(p.get('samples') or [])[:120]}")
    tgt_lines = [f"- {key} ({t['family']})" for key, t in target_index.items()]
    prompt = "SOURCE COLUMNS:\n" + "\n".join(src_lines) + "\n\nTARGET COLUMNS:\n" + "\n".join(tgt_lines)
    try:
        reply = llm_resilience.invoke(_get_client(), [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
                                      role="relationship_suggestions", breaker_name="relationships")
        content = getattr(reply, "content", reply)
        pairs = parse_pairs(content if isinstance(content, str) else str(content), set(sources), target_index)
    except llm_resilience.ModelUnavailable:
        return {}
    except Exception as exc:                       # best-effort: a bad reply must never break onboarding
        log.warning("relationship suggestions skipped (%s: %s)", type(exc).__name__, str(exc)[:200])
        return {}
    out: dict[str, list[dict]] = {}
    for src, target in pairs:
        out.setdefault(src, []).append(target)
    return out
