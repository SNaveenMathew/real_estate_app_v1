"""Generic semantic planner built entirely from the schema catalog."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import re
import db.schema_catalog as schema
from agents import query_scope

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
    scope_filters: list[str] = field(default_factory=list)   # mandatory, set by code (agents/query_scope.py), never by the model

    def to_dict(self): return asdict(self)
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
        if self.scope_filters:        # only when set, so every plan without a scope renders exactly as before
            lines.append(f"scope_filters (MANDATORY - the SQL must contain each as a positive condition): {' | '.join(self.scope_filters)}")
        return "\n".join(lines)

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
_N = r"(\d{1,4}|" + "|".join(_NUMBER_WORDS) + r")"
# "top 5", "bottom 3", "first 10", "best 5", ...  /  "5 cheapest", "3 most expensive", ...   (not "top 5 percent")
_TOP_N = re.compile(r"\b(?:top|bottom|first|best|worst|highest|lowest)\s+" + _N + r"\b(?!\s*(?:%|percent|per\s*cent|percentile))", re.I)
_N_SUPERLATIVE = re.compile(r"\b" + _N + r"\s+(?:cheapest|most\s+expensive|largest|biggest|smallest|highest|lowest|best|worst|safest|riskiest)\b", re.I)

def _top_n(text):
    """The N the user asked for ("top 25" -> 25, "five cheapest" -> 5), or None.  Sane bounds only."""
    m = _TOP_N.search(text) or _N_SUPERLATIVE.search(text)
    if not m:
        return None
    raw = m.group(1).lower()
    n = _NUMBER_WORDS.get(raw) if raw in _NUMBER_WORDS else int(raw)
    return n if 1 <= n <= 1000 else None

def _select_operation(concepts, text):
    best = None
    for c in concepts:
        for op in c.get("operations", []):
            for alias in op.get("aliases", []):
                if schema._alias_matches(text, alias):
                    score = len(alias.split())
                    cand = (score, op, c)
                    if best is None or score > best[0]: best = cand
    if best:
        return best[1], best[2]
    return None, None

def build_query_plan(request: str, requirements: str = "", plan: str = "", scope=None) -> QueryPlan:
    """``scope`` defaults to the scope set by code around the call (agents/query_scope.py); pass ``()`` to force none."""
    scope = query_scope.current() if scope is None else tuple(scope)
    text = request.strip()
    concepts = schema.semantic_matches(text)
    intent = schema.match_request_intents(text)
    qp = QueryPlan(question_type=intent[0]["name"] if intent else "analytical")
    qp.semantic_keys = [c["key"] for c in concepts]
    qp.metrics = sorted({m for c in concepts for m in c.get("columns", [])})
    qp.filters = list(dict.fromkeys(f for c in concepts for f in c.get("filters", [])))
    qp.null_policy = list(dict.fromkeys(c.get("null_policy", "") for c in concepts if c.get("null_policy")))
    qp.grouping = []
    qp.rollup_spec = next((c.get("rollup_spec", {}) for c in concepts if c.get("rollup_spec")), {})

    op, op_concept = _select_operation(concepts, text.lower())
    if op:
        qp.operation = op.get("op")
        qp.select_expression = op.get("expr")
        if op.get("op") in {"avg", "sum", "median", "min", "max", "count"}: qp.aggregation = op.get("expr")
        if op.get("op") == "rank":
            qp.ordering = f"{op['expr']} {op.get('direction','DESC')}"
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
                qp.aggregation = candidate.get("expr") if desired in {"avg", "sum", "median", "min", "max", "count"} else None
        if not qp.operation:
            for c in concepts:
                ops = c.get("operations", [])
                if len(ops) == 1:
                    if ops[0].get("op") == "count": qp.aggregation = ops[0]["expr"]; qp.operation = "count"; qp.select_expression = ops[0]["expr"]
                    elif ops[0].get("op") in {"avg", "sum", "median", "min", "max"}: qp.aggregation = ops[0]["expr"]; qp.operation = ops[0]["op"]; qp.select_expression = ops[0]["expr"]

    if not qp.operation:
        # Plain "show the named entity's columns" lookups: no aggregate or ranking phrase matched, so use the concept's
        # declared LOOKUP operation (db/catalog_model.py).  Existing concepts declare none, so their plans are unchanged.
        for c in concepts:
            lookup = next((x for x in c.get("operations", []) if x.get("op") == "lookup"), None)
            if lookup:
                qp.operation = "lookup"
                qp.select_expression = lookup.get("expr")
                break

    missing_semantic = "house_missing_walk" in {c.get("key") for c in concepts}
    for c in concepts:
        if c.get("null_policy") and not missing_semantic and any(op.get("op") in {"avg","min","max","sum","median","rank"} for op in c.get("operations", [])):
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
        if explicit_tracts:
            preferred_domains = {e["domain"] for e in explicit_tracts if e["table"] in preferred_tables}
            if not preferred_domains:
                preferred_domains = {
                    d.name for d in schema.ENTITY_DOMAINS
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

    tables = {t for c in concepts for t in c.get("tables", [])}
    tables.update(e["table"] for e in resolved)
    tables.update(schema.tables_mentioned_in_text(text))

    # Geography is selected from the entity domain, not from a question branch.
    if any(e["entity_type"] == "MSA" for e in resolved) and any("MSA" in c.get("entity_types", []) for c in concepts):
        tables.add("census_msa")
        if any(c.get("rollup") for c in concepts): tables.update({"cbsa_counties", "nri_tracts"})
    if any(e["entity_type"] == "tract_fips" for e in resolved):
        if any(c.get("key") == "census_tract_population" for c in concepts): tables.add("census_tracts")
        if any(c.get("rollup") for c in concepts): tables.add("nri_tracts")

    # For city-named tract/NRI questions, MSA is a useful semantic geography anchor.
    if any(c.get("rollup") for c in concepts) and any(e["entity_type"] == "MSA" for e in resolved):
        tables.update({"census_msa", "cbsa_counties", "nri_tracts"})

    tables.update(f.table.lower() for f in scope)         # the scoped table must be in the plan (and joined to the rest)
    tables = set(schema.expand_required_tables(tables))
    qp.required_tables = sorted(tables)
    qp.required_relationships = [r.render() for r in schema.relationships_for_tables(tables)]
    qp.entity_filters = [f"{e['table']}.{e['column']} = '{e['value'].replace(chr(39), chr(39)*2)}'" for e in resolved if e['table'] in tables]

    qp.scope_filters = [f.render() for f in scope]

    top_n = _top_n(text)
    if top_n is not None: qp.result_limit = top_n          # honour the number the user actually asked for
    return qp
