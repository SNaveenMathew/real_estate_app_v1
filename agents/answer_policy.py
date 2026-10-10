"""Answer policy: decide, from the QueryPlan alone, whether a request should reach SQL generation at all - and when it
should not, say exactly why.

``query_database`` asks this module three questions:

1. ``route_before_sql``     Must this request be handled WITHOUT the SQL model?  Cases, all driven by catalog metadata:

   * a place name fits several places (``ambiguous`` entities)  -> ask the user which one;
   * a matched concept declares ``derived``                      -> run the registered provider (polygon areas, ...);
   * only ``gap`` concepts matched                               -> say the topic is not loaded, estimate nothing.

2. ``unplannable_result``   Nothing in the catalog matched.  No SQL the model could write would survive plan validation
   (every statement must use a planned table), so asking it - up to three times - only burns time.  Return an honest,
   actionable message instead.

3. ``caveats_after_sql``    A real answer was produced but part of the request touches a known-unloaded topic: append
   the caveat so the answer model cannot present a partial result as complete.

Nothing here mentions a particular dataset.  A new dataset changes behaviour by declaring metadata on its concepts
(``requires_entity``, ``gap``, ``derived``, ``derived_support`` - see ``db/catalog_model.CONCEPT_KNOBS``), not by
editing this file.
"""
from __future__ import annotations

import db.duckdb_store as store
import db.schema_catalog as schema
from agents import answer_status as status
from agents.artifacts import classify_dataframe, emit_artifact
from db import entity_lexicon
from observability import mark_span_error, set_span_output, trace_span
from services import derived_measures

NOT_SQL_LABEL = "(no SQL: this request is not answered by a database query)"


def wrap(label: str, result: str) -> str:
    """The evidence envelope every query_database result uses; downstream code parses the [RESULT] header."""
    return f"[GENERATED SQL]\n{label}\n[RESULT]\n{result}"


def _matched_concepts(plan) -> list[dict]:
    glossary = schema.SEMANTIC_GLOSSARY
    return [glossary[k] for k in plan.semantic_keys if k in glossary]


# ---------------------------------------------------------------------------
# 1. Pre-SQL routing
# ---------------------------------------------------------------------------
def route_before_sql(plan, request: str, *, presentation: str = "auto") -> str | None:
    """Return a complete query_database result when SQL must not be used, else ``None``."""
    ambiguous = _ambiguity_message(plan)
    if ambiguous:
        return wrap(NOT_SQL_LABEL, ambiguous)
    concepts = _matched_concepts(plan)
    if any(c.get("derived") for c in concepts):
        return _derived_answer(plan, request, concepts, presentation)
    gap = _gap_only_message(concepts)
    if gap:
        return wrap(NOT_SQL_LABEL, gap)
    return None


def _example_qualifier(value: str) -> str:
    """'Portland-South Portland, ME Metro Area' -> 'Portland, ME' (an example of how to disambiguate)."""
    try:
        parsed = entity_lexicon.parse(value)
        if parsed.components and parsed.states:
            return f"{parsed.components[0]}, {parsed.states[0].upper()}"
    except Exception:
        pass
    return ""


def _ambiguity_message(plan) -> str | None:
    groups: dict[str, list[str]] = {}
    for e in plan.resolved_entities:
        if e.get("ambiguous"):
            groups.setdefault(e.get("phrase") or "that name", []).extend(e.get("ambiguous_with") or [e["value"]])
    if not groups:
        return None
    parts, example = [], ""
    for phrase, values in groups.items():
        unique = sorted(set(values))
        parts.append(f'"{phrase}" matches more than one place: {"; ".join(unique)}')
        example = example or _example_qualifier(unique[0])
    hint = f' (adding the state works, for example "{example}")' if example else ""
    return status.not_answered(
        status.PLACE_AMBIGUOUS,
        f"{'. '.join(parts)}. Ask the user which one they mean{hint}. No query was run and nothing was estimated; "
        "this is not a database error.")


def _gap_only_message(concepts: list[dict]) -> str | None:
    gaps = [c for c in concepts if c.get("gap")]
    if not gaps or any(not c.get("gap") for c in concepts):
        return None
    text = " ".join(c.get("description", "") for c in gaps)
    return status.not_answered(status.TOPIC_UNAVAILABLE, f"{text} This is not a database error.")


# ---------------------------------------------------------------------------
# Derived measures
# ---------------------------------------------------------------------------
def _derived_answer(plan, request: str, concepts: list[dict], presentation: str) -> str:
    wanted: dict[str, set[str]] = {}
    for c in concepts:
        spec = c.get("derived")
        if spec:
            wanted.setdefault(spec["provider"], set()).update(spec.get("measures", []))
    for c in concepts:                      # supporting measures ride along only when a provider is already running
        spec = c.get("derived_support")
        if spec and spec.get("provider") in wanted:
            wanted[spec["provider"]].update(spec.get("measures", []))

    sections = []
    for provider_id, measures in wanted.items():
        outcome = _run_provider(provider_id, measures, plan, request, presentation)
        if status.is_not_answered(outcome):
            return wrap(_evidence_label([provider_id]), outcome)
        sections.append(outcome)

    skipped = [c["aliases"][0] for c in concepts
               if c.get("tables") and not c.get("derived") and not c.get("derived_support") and not c.get("gap") and c.get("aliases")]
    if skipped:
        sections.append("Not computed by this call (they need their own query_database request): " + ", ".join(skipped) + ".")
    return wrap(_evidence_label(list(wanted)), "\n".join(sections))


def _evidence_label(provider_ids: list[str]) -> str:
    """What stands where the SQL normally appears in the evidence: the provider's own label, else its id."""
    labels = []
    for pid in provider_ids:
        try:
            labels.append(getattr(derived_measures.get_provider(pid), "evidence_label", None) or f"Derived measures: {pid}")
        except Exception:
            labels.append(f"Derived measures: {pid}")
    return "; ".join(dict.fromkeys(labels))


def _order_by_request(entities: list[dict], request: str, entity_type: str) -> list[dict]:
    """Entities in the order the user named them, each carrying the phrase that matched (for the 'matched as' note)."""
    phrase_of = {e["value"]: e.get("phrase") for e in schema.recognized_entities(request, {entity_type})}
    normalized = schema._normalize(request)

    def position(e):
        phrase = phrase_of.get(e["value"])
        at = normalized.find(schema._normalize(phrase)) if phrase else -1
        return (at if at >= 0 else len(normalized), e["value"])

    return [dict(e, phrase=phrase_of.get(e["value"])) for e in sorted(entities, key=position)]


def _run_provider(provider_id: str, measures: set[str], plan, request: str, presentation: str) -> str:
    try:
        return _compute_with_provider(provider_id, measures, plan, request, presentation)
    except derived_measures.DerivedError as exc:
        return status.not_answered(exc.code, exc.message)
    except Exception as exc:
        # Anything unexpected keeps the long-standing contract: a "Code Agent error:" result, which the orchestrator
        # recognises and retries - never an exception that tears down the whole turn.
        return f"Code Agent error: {exc}"


def _compute_with_provider(provider_id: str, measures: set[str], plan, request: str, presentation: str) -> str:
    try:
        provider = derived_measures.get_provider(provider_id)
    except KeyError:
        raise RuntimeError(f"derived-measure provider {provider_id!r} is not registered") from None
    # A specific concept already selected this provider; let it add the other measures named in the same sentence.
    measures = set(measures) | derived_measures.detect_measures(provider, request)
    entities = [e for e in plan.resolved_entities if e["entity_type"] == provider.entity_type and not e.get("ambiguous")]
    if not entities:
        return status.not_answered(
            status.PLACE_MISSING,
            f"This question needs a named {provider.entity_label}, and none in the request matches the data (a principal "
            f"city such as the one in the metro's name works). Derived measures are computed for the {provider.entity_label}s "
            f"the user names; ranking or listing every one of them by these measures is not supported. Ask which "
            f"{provider.entity_label} or {provider.entity_label}s they mean.")
    entities = _order_by_request(entities, request, provider.entity_type)
    with trace_span("derived_measures", attributes={"derived.provider": provider_id, "derived.entity_count": len(entities),
                                                    "derived.measures": ",".join(sorted(measures))}) as span:
        try:
            result = provider.compute(entities, sorted(measures), request=request)
            if span is not None:
                set_span_output(span, result.frame.to_dict(orient="records"), mime_type="application/json")
        except Exception as exc:
            mark_span_error(span, exc)
            raise
    try:
        emit_artifact(classify_dataframe(result.frame, request=request, presentation=presentation))
    except Exception:
        pass
    lines = list(result.method)
    matched = [f'"{e["phrase"]}" -> {e["value"]}' for e in entities if e.get("phrase")]
    if matched:
        lines.append("Places matched: " + "; ".join(matched))
    lines.append(result.frame.to_string(index=False))
    lines.extend(result.notes)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Unplannable requests
# ---------------------------------------------------------------------------
def _example_measures(entity_types=None, limit: int = 10, per_table: int = 3) -> list[str]:
    """Phrases the data answers, drawn from the live catalog: the newest ``per_table`` concepts of each primary table."""
    wanted = set(entity_types) if entity_types else None
    by_table: dict[str, list[str]] = {}
    for c in schema.SEMANTIC_GLOSSARY.values():
        if c.get("gap") or not c.get("aliases"):
            continue
        if wanted and not (set(c.get("entity_types", [])) & wanted):
            continue
        alias = next((a for a in c["aliases"] if len(a.split()) >= 2), c["aliases"][0])
        if any(ch.isdigit() for ch in alias):
            continue
        by_table.setdefault((c.get("tables") or [""])[0], []).append(alias)
    picked = []
    for aliases in by_table.values():
        for alias in aliases[-per_table:]:
            if alias not in picked:
                picked.append(alias)
    return sorted(picked)[:limit]


def _empty_entity_tables() -> list[str]:
    """Tables that name matching reads (entity domains) but that have no rows.  An empty one silently disables every
    question that names a place, so it is the one 'data not loaded' cause worth reporting on a failed match."""
    out = []
    for name in sorted({d.table for d in schema.ENTITY_DOMAINS}):
        try:
            if int(store.query(f"SELECT COUNT(*) AS n FROM {name}").iloc[0, 0]) == 0:
                out.append(name)
        except Exception:
            continue
    return out


def unplannable_result(request: str) -> str:
    parts = ["The request could not be matched to any measure or table in the data model, so no SQL was generated or run. "
             "This is a planning limitation, not a database error."]
    places = [e for e in schema.recognized_entities(request) if e["entity_type"] != "tract_fips"]
    if places:
        names = sorted({e["value"] for e in places if not any(ch.isdigit() for ch in e["value"])})[:5]
        examples = _example_measures({e["entity_type"] for e in places})
        parts.append(f"A place was recognized ({'; '.join(names)}) but no measure." if names else "A place was recognized but no measure.")
        parts.append("Say which measure is wanted" + (f", for example: {', '.join(examples)}." if examples else "."))
    else:
        examples = _example_measures()
        parts.append("Name the measure wanted and, where it applies, the place" +
                     (f". Examples of what the data covers: {', '.join(examples)}." if examples else "."))
    empty = _empty_entity_tables()
    if empty:
        parts.append("Place names are matched against these tables, which currently have no rows (so places in them cannot "
                     "be recognized until the data is loaded): " + ", ".join(empty) + ".")
    return status.not_answered(status.UNPLANNABLE, " ".join(parts))


# ---------------------------------------------------------------------------
# 3. Post-SQL caveats
# ---------------------------------------------------------------------------
def caveats_after_sql(plan) -> str:
    gaps = [c for c in _matched_concepts(plan) if c.get("gap")]
    if not gaps:
        return ""
    return "\n\nNOTE (not part of the query result): " + " ".join(c.get("description", "") for c in gaps)
