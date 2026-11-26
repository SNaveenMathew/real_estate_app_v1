"""Generic semantic planner built entirely from the schema catalog."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import re
import db.schema_catalog as schema
from agents import plan_status, query_scope


@dataclass(frozen=True)
class MeasureSpec:
    """One semantic measure inside a (potentially multi-measure) ``QueryPlan``."""

    concept_key: str
    expr: str
    operation: str
    tables: tuple[str, ...] = ()
    source_grain: str = ""
    target_grain: str = ""
    group_key: str = ""
    rollup_spec: dict = field(default_factory=dict)
    filters: tuple[str, ...] = ()
    null_policy: str = ""


@dataclass
class QueryPlan:
    question_type: str = "analytical"
    semantic_keys: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    required_tables: list[str] = field(default_factory=list)
    required_relationships: list[str] = field(default_factory=list)
    aggregation: str | None = None
    ordering: str | None = None
    grouping: list[str] = field(default_factory=list)
    result_limit: int | None = None
    resolved_entities: list[dict] = field(default_factory=list)
    entity_filters: list[str] = field(default_factory=list)
    null_policy: list[str] = field(default_factory=list)
    select_expression: str | None = None
    operation: str | None = None
    rollup_spec: dict = field(default_factory=dict)
    scope_filters: list[str] = field(
        default_factory=list
    )  # mandatory, set by code (agents/query_scope.py), never by the model
    measures: list[MeasureSpec] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def render(self) -> str:
        lines = [
            f"question_type: {self.question_type}",
            f"semantic_keys: {', '.join(self.semantic_keys) or 'none'}",
            f"required_tables: {', '.join(self.required_tables) or 'none'}",
            f"required_relationships: {', '.join(self.required_relationships) or 'none'}",
            f"metrics: {', '.join(self.metrics) or 'none'}",
            f"aggregation: {self.aggregation or 'none'}",
            f"ordering: {self.ordering or 'none'}",
            f"grouping: {', '.join(self.grouping) or 'none'}",
            f"filters: {' | '.join(self.filters) or 'none'}",
            f"null_policy: {' | '.join(self.null_policy) or 'none'}",
            f"select_expression: {self.select_expression or 'none'}",
            f"operation: {self.operation or 'none'}",
            f"rollup_spec: {self.rollup_spec or {}}",
            f"result_limit: {self.result_limit if self.result_limit is not None else 'none'}",
            f"resolved_entities: {self.resolved_entities or 'none'}",
            f"entity_filters: {' | '.join(self.entity_filters) or 'none'}",
        ]
        if self.scope_filters:  # only when set, so every plan without a scope renders exactly as before
            lines.append(
                f"scope_filters (MANDATORY - the SQL must contain each as a positive condition): {' | '.join(self.scope_filters)}"
            )
        if len(self.measures) > 1:  # keep single-measure plans identical to golden plan_battery expectations
            rendered_measures = "; ".join(
                f"{m.concept_key}: {m.operation.upper()}({m.expr}) [grain={m.source_grain or '?'} -> {m.target_grain or m.source_grain or '?'}, group_key={m.group_key or 'none'}]"
                for m in self.measures
            )
            lines.append(
                f"measures (MULTI-MEASURE CTE REQUIRED - compute each measure at its own grain in a separate CTE and join on group key): {rendered_measures}"
            )
        return "\n".join(lines)


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
}
_N = r"(\d{1,4}|" + "|".join(_NUMBER_WORDS) + r")"
# "top 5", "bottom 3", "first 10", "best 5", ...  /  "5 cheapest", "3 most expensive", ...   (not "top 5 percent")
_TOP_N = re.compile(
    r"\b(?:top|bottom|first|best|worst|highest|lowest)\s+" + _N + r"\b(?!\s*(?:%|percent|per\s*cent|percentile))", re.I
)
_N_SUPERLATIVE = re.compile(
    r"\b"
    + _N
    + r"\s+(?:cheapest|most\s+expensive|largest|biggest|smallest|highest|lowest|best|worst|safest|riskiest)\b",
    re.I,
)


def _top_n(text):
    """The N the user asked for ("top 25" -> 25, "five cheapest" -> 5), or None.  Sane bounds only."""
    m = _TOP_N.search(text) or _N_SUPERLATIVE.search(text)
    if not m:
        return None
    raw = m.group(1).lower()
    n = _NUMBER_WORDS.get(raw) if raw in _NUMBER_WORDS else int(raw)
    return n if 1 <= n <= 1000 else None


def _concept_explicit_op(concept: dict, text_low: str):
    """Return ``(score, op_dict, alias)`` for the longest operation alias matching ``text_low``, or ``None``."""
    best = None
    for op in concept.get("operations", []):
        for alias in op.get("aliases", []):
            if schema._alias_matches(text_low, alias):
                score = len(alias.split())
                if best is None or score > best[0]:
                    best = (score, op, alias)
    return best


def _alias_occurrences(text_low: str, alias: str) -> list[tuple[int, int]]:
    """Every whole-word occurrence span of ``alias`` in ``text_low``."""
    pattern = r"(?<![a-z0-9])" + re.escape(alias.lower()) + r"(?![a-z0-9])"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text_low)]


def _alias_is_fully_subsumed(text_low: str, alias: str, longer: str) -> bool:
    """True when every occurrence of ``alias`` sits inside an occurrence of ``longer``.

    ``"how many transit routes"`` contains the generic COUNT alias ``"how many"``: that is one
    request, not two measures.  A question that names both ("how many houses and how many
    transit routes") still has an uncovered ``"how many"``, so the shorter concept keeps its
    own measure.
    """
    short_spans = _alias_occurrences(text_low, alias)
    long_spans = _alias_occurrences(text_low, longer)
    if not short_spans or not long_spans:
        return False
    return all(any(ls <= s and e <= le for ls, le in long_spans) for s, e in short_spans)


def _select_operation(concepts, text):
    best = None
    for c in concepts:
        matched = _concept_explicit_op(c, text)
        if matched and (best is None or matched[0] > best[0]):
            best = (matched[0], matched[1], c)
    if best:
        return best[1], best[2]
    return None, None


def _default_op_for_concept(concept: dict) -> dict | None:
    """Resolve the fallback operation for a concept when no operation alias matched."""
    desired = concept.get("default_operation")
    if desired:
        cand = next((x for x in concept.get("operations", []) if x.get("op") == desired), None)
        if cand:
            return cand
    ops = [op for op in concept.get("operations", ()) if op.get("op") != "lookup"]
    if len(ops) == 1 and ops[0].get("op") in {"avg", "sum", "median", "min", "max", "count"}:
        return ops[0]
    lookup = next((x for x in concept.get("operations", ()) if x.get("op") == "lookup" and not x.get("aliases")), None)
    return lookup


def _measure_for_concept(concept: dict, op_dict: dict) -> MeasureSpec:
    rollup_spec = dict(concept.get("rollup_spec") or {})
    groupings = tuple(concept.get("groupings") or ())
    group_key = str(op_dict.get("group_by") or rollup_spec.get("group_key") or (groupings[0] if groupings else ""))
    filters = list(concept.get("filters") or ())
    null_policy = str(concept.get("null_policy") or "")
    if null_policy and op_dict.get("op") in {"avg", "min", "max", "sum", "median", "rank"}:
        for col in concept.get("columns", ()):
            if (
                col.endswith("walk_score")
                or col.endswith("bike_score")
                or col.endswith("transit_score")
                or null_policy == "exclude NULL"
            ):
                if f"{col} IS NOT NULL" not in filters:
                    filters.append(f"{col} IS NOT NULL")
    return MeasureSpec(
        concept_key=str(concept.get("key") or ""),
        expr=str(op_dict.get("expr") or ""),
        operation=str(op_dict.get("op") or ""),
        tables=tuple(concept.get("tables") or ()),
        source_grain=str(concept.get("grain") or ""),
        target_grain=str(rollup_spec.get("target_grain") or ""),
        group_key=group_key,
        rollup_spec=rollup_spec,
        filters=tuple(dict.fromkeys(filters)),
        null_policy=null_policy,
    )


def _extract_measures(concepts: list[dict], text_low: str) -> list[MeasureSpec]:
    """Extract active ``MeasureSpec`` entries from matched concepts without inventing subject-noun counts."""
    explicit_ops = {c.get("key"): _concept_explicit_op(c, text_low) for c in concepts}
    # Phrase subsumption: a generic operation alias ("how many") wholly contained in a longer,
    # more specific alias from another matched concept ("how many transit routes") describes the
    # SAME request.  Left alone, the generic concept adds a second, spurious measure whose
    # single-table CTE cannot be grouped on the specific concept's geography (and would show an
    # unrelated dataset-wide count beside the scoped answer).
    for c in concepts:
        key = c.get("key")
        matched = explicit_ops.get(key)
        if not matched:
            continue
        alias = matched[2]
        for other in concepts:
            if other is c:
                continue
            other_matched = explicit_ops.get(other.get("key"))
            if not other_matched or len(other_matched[2]) <= len(alias):
                continue
            if _alias_is_fully_subsumed(text_low, alias, other_matched[2]):
                explicit_ops[key] = None
                break
    active_tables = {
        t
        for c in concepts
        if explicit_ops.get(c.get("key")) or c.get("default_operation")
        for t in (c.get("tables") or ())
    }
    measures: list[MeasureSpec] = []
    seen_columns: set[tuple[str, ...]] = set()
    seen_exprs: set[tuple[tuple[str, ...], str, str]] = set()

    # Order concepts with explicit operation matches or default_operation first so an explicit aggregate
    # (e.g. msa_population SUM) claims its underlying column before a passive lookup concept on the same column.
    ordered = sorted(
        concepts,
        key=lambda c: (0 if explicit_ops.get(c.get("key")) else (1 if c.get("default_operation") else 2),),
    )
    multi_table_concepts = (
        len({t for cc in concepts if cc.get("operations") and not cc.get("gap") for t in (cc.get("tables") or ())}) > 1
    )

    for c in ordered:
        if c.get("gap") or not c.get("operations"):
            continue
        matched = explicit_ops.get(c.get("key"))
        op_dict = matched[1] if matched else _default_op_for_concept(c)
        if op_dict is None and multi_table_concepts:
            # Cross-table comparison questions ("flood risk and metro population") often name a second
            # numeric metric without repeating an aggregate word; never promote an unrequested COUNT().
            c_tables = set(c.get("tables") or ())
            if not (c_tables <= active_tables):
                op_dict = next(
                    (x for x in c.get("operations", ()) if x.get("op") in {"avg", "sum", "median", "min", "max"}),
                    None,
                )
        if op_dict is None:
            continue
        c_tables = set(c.get("tables") or ())
        if not matched and not c.get("default_operation"):
            if op_dict.get("op") == "count" and len(concepts) > 1:
                continue
            if (c_tables & active_tables) and len(c.get("operations", ())) > 1:
                continue
        spec = _measure_for_concept(c, op_dict)
        # Deduplicate measures targeting either the same metric column or the exact same (tables, op, expr)
        # (e.g. house_missing_walk COUNT(houses.house_id) already covers house_inventory COUNT(houses.house_id)).
        metric_cols = tuple(
            sorted(col for col in (c.get("columns") or ()) if not col.endswith(".name") and not col.endswith("_id"))
        ) or (spec.tables + (spec.expr,))
        expr_sig = (spec.tables, spec.operation, spec.expr)
        if spec.expr and metric_cols not in seen_columns and expr_sig not in seen_exprs:
            seen_columns.add(metric_cols)
            seen_exprs.add(expr_sig)
            measures.append(spec)

    return measures


def build_query_plan(request: str, requirements: str = "", plan: str = "", scope=None) -> QueryPlan:
    """``scope`` defaults to the scope set by code around the call (agents/query_scope.py); pass ``()`` to force none."""
    scope = query_scope.current() if scope is None else tuple(scope)
    text = request.strip()
    concepts = schema.semantic_matches(text)
    # A concept may pin the join keys its grain is read at (``join``, db/catalog_model.py): ZHVI stores zip and
    # neighborhood rows in one table, so the table-level default (the house ZIP key) is right only for zip rows, a
    # metro concept needs the CBSA key, and the neighborhood level has no house key at all. Dropping concepts that
    # live only outside the pinned grain also keeps their metrics and filters out of the plan.
    pins = schema.concept_join_pins(concepts)
    if pins:
        concepts = schema.concepts_at_pinned_grain(concepts, pins)
    intent = schema.match_request_intents(text)
    qp = QueryPlan(question_type=intent[0]["name"] if intent else "analytical")
    qp.semantic_keys = [c["key"] for c in concepts]
    qp.metrics = sorted({m for c in concepts for m in c.get("columns", [])})
    qp.filters = list(dict.fromkeys(f for c in concepts for f in c.get("filters", [])))
    qp.null_policy = list(dict.fromkeys(c.get("null_policy", "") for c in concepts if c.get("null_policy")))
    qp.grouping = []
    qp.rollup_spec = next((c.get("rollup_spec", {}) for c in concepts if c.get("rollup_spec")), {})

    norm_text, _repaired = schema.normalize_query_typos(text)
    op, op_concept = _select_operation(concepts, norm_text)
    if op:
        qp.operation = op.get("op")
        qp.select_expression = op.get("expr")
        if op.get("op") in {"avg", "sum", "median", "min", "max", "count"}:
            qp.aggregation = op.get("expr")
        if op.get("op") == "rank":
            qp.ordering = f"{op['expr']} {op.get('direction', 'DESC')}"
        if op.get("group_by"):
            qp.grouping = [op["group_by"]]
            qp.select_expression = f"{op['group_by']}, {op['expr']}"
    else:
        default_concept = next((c for c in concepts if c.get("default_operation")), None)
        if default_concept:
            desired = default_concept.get("default_operation")
            candidate = next((x for x in default_concept.get("operations", []) if x.get("op") == desired), None)
            if candidate:
                qp.operation = candidate.get("op")
                qp.select_expression = candidate.get("expr")
                qp.aggregation = (
                    candidate.get("expr") if desired in {"avg", "sum", "median", "min", "max", "count"} else None
                )
        if not qp.operation:
            for c in concepts:
                ops = c.get("operations", [])
                if len(ops) == 1:
                    if ops[0].get("op") == "count":
                        qp.aggregation = ops[0]["expr"]
                        qp.operation = "count"
                        qp.select_expression = ops[0]["expr"]
                    elif ops[0].get("op") in {"avg", "sum", "median", "min", "max"}:
                        qp.aggregation = ops[0]["expr"]
                        qp.operation = ops[0]["op"]
                        qp.select_expression = ops[0]["expr"]

    if not qp.operation:
        # Plain "show the named entity's columns" lookups: only apply anonymous (empty-alias) LOOKUP ops here
        # so explicit listing phrases ("what houses ...") stay opt-in without drifting unrelated plans.
        for c in concepts:
            lookup = next(
                (x for x in c.get("operations", []) if x.get("op") == "lookup" and not x.get("aliases")), None
            )
            if lookup:
                qp.operation = "lookup"
                qp.select_expression = lookup.get("expr")
                break

    qp.measures = _extract_measures(concepts, norm_text)

    missing_semantic = "house_missing_walk" in {c.get("key") for c in concepts}
    for c in concepts:
        if (
            c.get("null_policy")
            and not missing_semantic
            and any(op.get("op") in {"avg", "min", "max", "sum", "median", "rank"} for op in c.get("operations", []))
        ):
            for col in c.get("columns", []):
                if col.endswith("walk_score") or col.endswith("bike_score") or col.endswith("transit_score"):
                    qp.filters.append(f"{col} IS NOT NULL")

    resolved_all = schema.resolve_request_entities(text, None)
    requested_entity_types = {et for c in concepts for et in c.get("entity_types", [])}
    preferred_tables = {t for c in concepts for t in c.get("tables", [])}
    concept_keys = {c.get("key") for c in concepts}
    explicit_tracts = [e for e in resolved_all if e["entity_type"] == "tract_fips"]
    if requested_entity_types:
        resolved = [e for e in resolved_all if e["entity_type"] in requested_entity_types]
        # A concept that opts into county/MSA/tract entities must still resolve a city named in its own
        # table (bike_routes.city is the only city label a bike route has).  A bare city name also matches
        # the MSA whose title starts with that city, so prefer the own-table city unless the question
        # explicitly names a county/metro/tract.
        own_entities = [e for e in resolved_all if e["table"] in preferred_tables]
        if own_entities and not any(
            term in text.lower()
            for term in ("county", "metro", "msa", "metropolitan", "tract", "city limits", "city boundary")
        ):
            resolved = own_entities
        elif not resolved:
            resolved = own_entities
        lowered = text.lower()
        if "county" in lowered:
            resolved = [e for e in resolved if e["entity_type"] != "MSA"]
        elif any(term in lowered for term in ("metro", "msa", "metropolitan")):
            resolved = [e for e in resolved if e["entity_type"] != "county"]
        if explicit_tracts:
            preferred_domains = {e["domain"] for e in explicit_tracts if e["table"] in preferred_tables}
            if not preferred_domains:
                preferred_domains = {
                    d.name
                    for d in schema.ENTITY_DOMAINS
                    if d.entity_type == "tract_fips" and any(k in concept_keys for k in d.preferred_for)
                }
            if preferred_domains:
                resolved = [e for e in resolved if e["entity_type"] != "tract_fips" or e["domain"] in preferred_domains]
    else:
        resolved = [e for e in resolved_all if e["table"] in preferred_tables]
        for e in explicit_tracts:
            if e["table"] in preferred_tables and e not in resolved:
                resolved.append(e)
    qp.resolved_entities = resolved
    qp.entities = sorted({e["entity_type"] for e in resolved})

    if len(qp.measures) > 1 and not qp.grouping:
        group_keys = [m.group_key for m in qp.measures if m.group_key]
        if group_keys and (
            len(resolved) >= 2
            or any(
                token in text.lower()
                for token in (" vs ", " versus ", "compare", "deciding", "weigh", "between", " by ")
            )
        ):
            # Pick the most common declared group_key so every measure CTE aligns on the comparison entity.
            best_group = max(dict.fromkeys(group_keys), key=group_keys.count)
            if not any(tok in text.lower() for tok in ("combined ", "total across ", "all together")):
                qp.grouping = [best_group]

    tables = {t for c in concepts for t in c.get("tables", [])}
    tables.update(e["table"] for e in resolved)
    tables.update(schema.tables_mentioned_in_text(text))

    # Geography is selected from the entity domain, not from a question branch.
    if any(e["entity_type"] == "MSA" for e in resolved) and any("MSA" in c.get("entity_types", []) for c in concepts):
        tables.add("census_msa")
        if any(c.get("rollup") for c in concepts):
            tables.update({"cbsa_counties", "nri_tracts"})
    if any(e["entity_type"] == "tract_fips" for e in resolved):
        if any(c.get("key") == "census_tract_population" for c in concepts):
            tables.add("census_tracts")
        if any(c.get("rollup") for c in concepts):
            tables.add("nri_tracts")

    # For city-named tract/NRI questions, MSA is a useful semantic geography anchor.
    if any(c.get("rollup") for c in concepts) and any(e["entity_type"] == "MSA" for e in resolved):
        tables.update({"census_msa", "cbsa_counties", "nri_tracts"})

    # A projection or ordering expression can name a table that no concept selected (e.g. the commute concept's
    # "houses.address, house_commute.bike_min").  Left alone, the compiler emits FROM house_commute and the database
    # fails with "Referenced table houses" - a database error describing a planning bug (agents/plan_status.py).  Add
    # such a table only when the catalog has a relationship path to what is already planned, so the repair can never
    # introduce an unjoinable table; anything that cannot be repaired is reported by plan_status.report().
    referenced = plan_status.referenced_tables(qp)
    if referenced:
        known = {t.lower() for t in schema.list_table_names(agent_visible_only=True)}
        for candidate in sorted(referenced - {t.lower() for t in tables}):
            if candidate not in known:
                continue
            try:
                if len(schema.relationship_path(tables | {candidate}, pins)) < len(tables):
                    continue  # no catalog path: leave it out, and let the plan report say so
            except Exception:
                continue
            tables.add(candidate)

    tables.update(f.table.lower() for f in scope)  # the scoped table must be in the plan (and joined to the rest)
    if pins:  # tables added from entities/text/scope must still sit at the same grain as the concept
        tables = schema.grain_connected_tables(tables, pins)
    tables = set(schema.expand_required_tables(tables, pins))
    qp.required_tables = sorted(tables)
    qp.required_relationships = [r.render() for r in schema.relationships_for_tables(tables, pins)]
    qp.entity_filters = [
        f"{e['table']}.{e['column']} = '{e['value'].replace(chr(39), chr(39) * 2)}'"
        for e in resolved
        if e["table"] in tables
    ]

    qp.scope_filters = [f.render() for f in scope]

    top_n = _top_n(text)
    if top_n is not None:
        qp.result_limit = top_n  # honour the number the user actually asked for
    return qp


def plan_report(request_or_plan):
    """The typed ``agents/plan_status.PlanReport`` for a request or an already-built plan."""
    plan = request_or_plan if isinstance(request_or_plan, QueryPlan) else build_query_plan(str(request_or_plan))
    return plan_status.report(plan, relationship_path=schema.relationship_path)
