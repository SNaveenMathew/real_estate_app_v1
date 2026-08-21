"""Deterministic query planning metadata for General Chat.

The planner does not generate SQL. It resolves user-language concepts to
physical tables/columns, documented relationship paths, aggregation guidance,
and data-quality constraints before the SQL Code Agent runs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

import db.schema_catalog as schema


@dataclass
class QueryPlan:
    question_type: str = "analytical"
    entities: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    required_tables: list[str] = field(default_factory=list)
    required_relationships: list[str] = field(default_factory=list)
    aggregation: str | None = None
    ordering: str | None = None
    limit: int | None = None
    universe_limit: int | None = None
    result_limit: int | None = None
    scope: dict = field(default_factory=dict)
    data_quality_rules: list[str] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        lines = [
            f"question_type: {self.question_type}",
            f"entities: {', '.join(self.entities) or 'none'}",
            f"metrics: {', '.join(self.metrics) or 'none'}",
            f"filters: {', '.join(self.filters) or 'none'}",
            f"required_tables: {', '.join(self.required_tables) or 'none'}",
            f"required_relationships: {', '.join(self.required_relationships) or 'none'}",
            f"aggregation: {self.aggregation or 'none'}",
            f"ordering: {self.ordering or 'none'}",
            f"limit: {self.limit if self.limit is not None else 'none'}",
            f"universe_limit: {self.universe_limit if self.universe_limit is not None else 'none'}",
            f"result_limit: {self.result_limit if self.result_limit is not None else 'none'}",
            f"scope: {self.scope or 'none'}",
            f"data_quality_rules: {' | '.join(self.data_quality_rules) or 'none'}",
            f"unresolved_items: {', '.join(self.unresolved_items) or 'none'}",
        ]
        return "\n".join(lines)


def build_query_plan(request: str, requirements: str = "", plan: str = "") -> QueryPlan:
    text = " ".join(x for x in (request, requirements, plan) if x).strip()
    lower = text.lower()
    qp = QueryPlan()

    if any(x in lower for x in ("top ", "highest", "lowest", "rank", "best", "worst")):
        qp.question_type = "ranking"
    elif any(x in lower for x in ("how many", "count", "number of")):
        qp.question_type = "aggregation"
    elif any(x in lower for x in ("compare", "versus", "vs.")):
        qp.question_type = "comparison"

    sem = schema.semantic_matches(text)
    for item in sem:
        qp.metrics.extend(item.get("columns", []))
        if item.get("filter"):
            qp.filters.append(item["filter"])
        qp.required_tables.extend(item.get("tables", []))

    if "flood risk" in lower and "coastal" not in lower:
        qp.metrics.append("nri_tracts.rfld_risks")
        qp.required_tables.append("nri_tracts")
        qp.aggregation = "AVG(rfld_risks) at MSA grain unless another statistic is requested"
        qp.data_quality_rules.append("Exclude NULL rfld_risks; do not average coastal and riverine scores together")

    if "overall risk" in lower or "composite risk" in lower:
        qp.metrics.append("nri_tracts.risk_score")
        qp.required_tables.append("nri_tracts")
        qp.aggregation = "AVG(risk_score) at MSA grain unless another statistic is requested"
        qp.data_quality_rules.append("Exclude NULL risk_score")

    if "walk score" in lower or "walkability" in lower:
        qp.metrics.append("houses.walk_score")
        qp.required_tables.append("houses")
        qp.ordering = "houses.walk_score DESC for highest/best, ASC for lowest/worst"
        qp.data_quality_rules.append("Exclude NULL walk_score unless missing values are explicitly requested")

    if "my list" in lower or "saved houses" in lower or "favorites" in lower or "favorite houses" in lower:
        qp.scope["type"] = "saved_houses"
        qp.scope["filter"] = "houses.is_favorite = TRUE"
        qp.required_tables.append("houses")

    m = re.search(r"\btop\s+(\d+)\b", lower)
    if m:
        qp.universe_limit = int(m.group(1))
        # Ranking language refers to the number of rows returned, not the size
        # of the universe being ranked. Keep those concepts separate so
        # "Among the top 50 MSAs, which have the lowest ..." means:
        #   1) population-rank 50 MSAs, then
        #   2) risk-rank that 50-MSA universe and return the best few.
        qp.result_limit = 10
        qp.limit = qp.result_limit
        qp.filters.append(f"universe = top {qp.universe_limit} MSAs by population")

    if "msa" in lower or "metro" in lower:
        qp.entities.append("MSA")
        qp.required_tables.extend(["census_msa", "cbsa_counties", "nri_tracts"] if "risk" in lower else ["census_msa"])
        qp.required_relationships.extend([
            "census_msa.msa_code = cbsa_counties.cbsa_code",
            "cbsa_counties.state_fips || cbsa_counties.county_fips = nri_tracts.county_fips",
        ] if "risk" in lower else ["census_msa"])
        qp.data_quality_rules.append("Filter census_msa.msa_code NOT LIKE 'X%' before joining to cbsa_counties")
        if qp.universe_limit is not None:
            qp.ordering = "population DESC inside the MSA universe; requested NRI metric ASC for lowest/best or DESC for highest/worst"

    qp.required_tables = sorted(set(qp.required_tables))
    qp.metrics = sorted(set(qp.metrics))
    qp.required_relationships = sorted(set(qp.required_relationships))
    return qp
