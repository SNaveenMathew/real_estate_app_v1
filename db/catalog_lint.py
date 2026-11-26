"""Catalog lint: static checks that make adding a dataset or a concept safe.

The agents are driven entirely by catalog metadata, which means a metadata mistake does not fail loudly - a concept
whose table name has a typo is silently dropped by ``schema_catalog.reload()``, a misspelled knob (``require_entity``)
silently does nothing, and a generic alias quietly steals a word from every future dataset.  This module turns those
silent failures into explicit findings, for built-in and uploaded objects alike.

    python -m db.catalog_lint            # prints findings; exit status 1 when there are errors
    python -m db.catalog_lint --orphans  # additionally reconcile join keys against the loaded rows

``tests/test_catalog_contract.py`` runs it on every test run with a *ratchet*: findings that already existed are listed
in ``tests/golden/catalog_lint_baseline.json``; a NEW finding fails the build.  Fix it, or - when it is deliberate -
add it to the baseline in the same change so a reviewer sees the decision.

Severities
----------
error     the object cannot behave as declared (unknown knob, missing table, unregistered provider, ...)
warning   it works but is risky for future datasets (generic alias, alias owned twice, no declared precedence)
info      a suggestion (e.g. an entity domain whose values would be better served by ``components`` matching)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import db.catalog_store as catalog_store
import db.duckdb_store as store
import db.schema_catalog as schema
from db import entity_lexicon
from db.catalog_model import CONCEPT_KNOBS
from db.text_match import normalize

KNOWN_MATCH_MODES = ("exact", "exact_or_prefix", "prefix", "components")
BASE_CONCEPT_KEYS = {
    "key",
    "tables",
    "columns",
    "aliases",
    "description",
    "operations",
    "filters",
    "null_policy",
    "orderings",
    "groupings",
    "grain",
    "entity_types",
    "rollup",
    "rollup_spec",
    "required_terms",
    "excluded_terms",
    "default_operation",
    "origin",
    "dataset_id",
}
# Mirrors services/dataset_onboarding._GENERIC_ALIASES (imported when available so there is a single source of truth).
_FALLBACK_GENERIC = {
    "value",
    "values",
    "score",
    "rate",
    "data",
    "number",
    "count",
    "total",
    "index",
    "rating",
    "level",
    "type",
    "name",
    "id",
    "code",
    "year",
    "date",
    "status",
    "city",
    "state",
    "price",
    "risk",
    "population",
    "average",
    "area",
    "size",
    "list",
    "rank",
    "house",
    "home",
    "houses",
}


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    subject: str
    message: str

    def key(self) -> str:
        return f"{self.severity}:{self.code}:{self.subject}"

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code} {self.subject}: {self.message}"


def _generic_aliases() -> set[str]:
    try:
        from services.dataset_onboarding import _GENERIC_ALIASES

        return set(_GENERIC_ALIASES)
    except Exception:
        return set(_FALLBACK_GENERIC)


def _alias_is_generic(alias: str, generic: set[str]) -> bool:
    toks = normalize(alias).split()
    if not toks:
        return True
    if len(toks) == 1:
        return len(toks[0]) < 6 or toks[0] in generic
    return all(t in generic for t in toks)


def _is_builtin(row: dict) -> bool:
    return (row.get("origin") or "builtin") == "builtin"


def lint_catalog(include_info: bool = True) -> list[Issue]:
    conn = store.get_conn()
    data = catalog_store.load_all(conn)
    tables = {r["name"] for r in data["tables"]}
    issues: list[Issue] = []
    add = lambda sev, code, subject, msg: issues.append(Issue(sev, code, subject, msg))  # noqa: E731

    domains = data["entity_domains"]
    domain_types = {d["entity_type"] for d in domains}
    concept_rows = []
    for r in data["concepts"]:
        item = schema._json(r.get("definition"), {})
        if item:
            concept_rows.append((r, item))
    keys = {r["key"] for r, _ in concept_rows}
    generic = _generic_aliases()

    from services import derived_measures

    providers = {}
    for pid in derived_measures.known_provider_ids():
        try:
            providers[pid] = derived_measures.get_provider(pid)
        except Exception as exc:  # pragma: no cover - import problems
            add("error", "provider_import_failed", pid, f"derived provider {pid!r} could not be imported: {exc}")

    # A concept's ``join`` knob names the edges its grain is read at; a key that is not an approved relationship (or
    # does not touch the concept's own table) is a silent no-op that leaves the table-level default in place.
    join_edges = {
        f"{r['left_table']}.{r['left_expr']}={r['right_table']}.{r['right_expr']}": (r["left_table"], r["right_table"])
        for r in data["relationships"]
    }
    owners: dict[str, list[str]] = {}
    for row, item in concept_rows:
        key = row["key"]
        builtin = _is_builtin(row)
        sev_missing = "error" if builtin else "warning"

        for field in ("tables", "aliases", "description"):
            if field not in item:
                add("error", "missing_field", key, f"concept has no '{field}'")
        unknown = sorted(set(item) - BASE_CONCEPT_KEYS - set(CONCEPT_KNOBS))
        if unknown:
            add(
                "error",
                "unknown_key",
                key,
                f"unknown concept key(s) {unknown}; known knobs are {sorted(CONCEPT_KNOBS)} "
                "(a misspelled knob silently does nothing)",
            )
        missing = [t for t in item.get("tables", []) if t not in tables]
        if missing:
            add(
                sev_missing,
                "concept_dropped",
                key,
                f"references table(s) {missing} that are not active, so the concept is "
                "silently dropped from the registry",
            )

        seen = set()
        for alias in item.get("aliases", []):
            n = normalize(alias)
            if not n:
                add("error", "empty_alias", key, f"alias {alias!r} normalizes to nothing")
                continue
            if n in seen:
                add("warning", "duplicate_alias", key, f"alias {alias!r} is listed twice")
            seen.add(n)
            if not item.get("gap"):  # a known-gap topic yields to any dataset that covers it: it owns nothing
                owners.setdefault(n, []).append(key)
            if builtin and _alias_is_generic(alias, generic) and not item.get("gap"):
                add(
                    "warning",
                    "generic_alias",
                    f"{key}:{n}",
                    f"alias {alias!r} is too generic for a built-in concept: "
                    "onboarding would stop every future dataset from using that word",
                )

        # ---- knobs ------------------------------------------------------------------------------------------------
        joins = item.get("join")
        if joins is not None:
            if not isinstance(joins, list) or not all(isinstance(x, str) for x in joins):
                add("error", "bad_knob", key, "join must be a list of relationship keys like houses.zip=zhvi.region_name")
            else:
                for join_key in joins:
                    tables_in_edge = join_edges.get(join_key)
                    if tables_in_edge is None:
                        add(
                            "error",
                            "unknown_join",
                            key,
                            f"join names {join_key!r}, which is not an approved relationship, so the planner keeps "
                            "the table-level default instead",
                        )
                    elif not set(tables_in_edge) & set(item.get("tables", [])):
                        add(
                            "error",
                            "join_off_table",
                            key,
                            f"join {join_key!r} does not touch this concept's table(s) "
                            f"{item.get('tables', [])}, so it can never apply",
                        )
        needs = item.get("requires_entity")
        if needs is not None:
            if not isinstance(needs, list) or not all(isinstance(x, str) for x in needs):
                add("error", "bad_knob", key, "requires_entity must be a list of entity-type names")
            else:
                for et in needs:
                    if et not in domain_types:
                        add(
                            "error",
                            "unknown_entity_type",
                            key,
                            f"requires_entity names {et!r}, but no entity domain has "
                            "that entity_type, so the concept can never apply",
                        )
        if item.get("level"):
            # The knob says "this filter selects a dataset level"; without an equality filter there is nothing to
            # probe, so the declaration could never fire.
            pinned = [
                f
                for f in item.get("filters", ())
                if isinstance(f, str) and re.match(r"^[A-Za-z_]\w*\.[A-Za-z_]\w*\s*=\s*'[^']*'$", f.strip())
            ]
            if not pinned:
                add(
                    "error",
                    "level_without_pin",
                    key,
                    "level=True marks a concept that pins a dataset level, but it declares no "
                    "`table.column = 'value'` filter for the availability check to probe",
                )
        for knob in ("derived", "derived_support"):
            spec = item.get(knob)
            if spec is None:
                continue
            if not isinstance(spec, dict) or "provider" not in spec or not isinstance(spec.get("measures"), list):
                add("error", "bad_knob", key, f"{knob} must be {{'provider': id, 'measures': [...]}}")
                continue
            provider = providers.get(spec["provider"])
            if provider is None:
                add(
                    "error",
                    "unknown_provider",
                    key,
                    f"{knob} names provider {spec['provider']!r}, which is not registered (known: {sorted(providers)})",
                )
                continue
            for m in spec["measures"]:
                if m not in provider.measures:
                    add(
                        "error",
                        "unknown_measure",
                        key,
                        f"{knob} asks provider {spec['provider']!r} for {m!r}; it offers {sorted(provider.measures)}",
                    )
            if knob == "derived" and provider.entity_type not in item.get("entity_types", []):
                add(
                    "error",
                    "derived_needs_entity",
                    key,
                    f"provider {spec['provider']!r} needs an entity of type "
                    f"{provider.entity_type!r}; the concept must list it in entity_types or no place will be resolved",
                )
        if item.get("gap"):
            if item.get("tables"):
                add(
                    "error",
                    "gap_with_tables",
                    key,
                    "a gap concept declares an UNAVAILABLE topic and must have no tables",
                )
            if not str(item.get("description", "")).strip():
                add(
                    "error",
                    "gap_without_description",
                    key,
                    "a gap concept's description is the message shown to the user",
                )
            elif re.search(r"\d", item["description"]):
                add(
                    "warning",
                    "digits_in_gap_description",
                    key,
                    "avoid digits: the message is shown as evidence and digits are what the validator treats as data",
                )
        for other in item.get("overrides", []):
            if other not in keys:
                add("warning", "unknown_override", key, f"overrides {other!r}, which is not an active concept")
        # ``broader`` is the only declared subsumption in the model; a typo would silently draw an
        # unparented ontology class, so an unknown target is an error for a built-in concept.
        broader = item.get("broader")
        if broader is not None:
            if not isinstance(broader, list) or not all(isinstance(x, str) and x for x in broader):
                add("error", "bad_knob", key, "broader must be a list of concept keys")
            else:
                for broad in broader:
                    if broad == key:
                        add("error", "self_broader", key, "broader names the concept itself")
                    elif broad not in keys:
                        add(
                            "error" if builtin else "warning",
                            "unknown_broader",
                            key,
                            f"broader names {broad!r}, which is not an active concept, so the class has no parent",
                        )
        for op in item.get("operations", []):
            if op.get("op") == "lookup" and not any(
                re.search(rf"\b{re.escape(t)}\.", op.get("expr", "")) for t in item.get("tables", [])
            ):
                add(
                    "warning",
                    "lookup_expr_tables",
                    key,
                    f"lookup expression {op.get('expr')!r} references none of the concept's tables",
                )

    for alias, who in sorted(owners.items()):
        who = sorted(set(who))
        if len(who) > 1:
            add(
                "warning",
                "alias_collision",
                alias,
                f"alias {alias!r} is claimed by {who}; the planner will select all of them",
            )

    # ---- entity domains ------------------------------------------------------------------------------------------
    for d in domains:
        name = d["name"]
        mode = d.get("match_mode") or "exact_or_prefix"
        if mode not in KNOWN_MATCH_MODES:
            add(
                "error",
                "unknown_match_mode",
                name,
                f"match_mode {mode!r} is not one of {KNOWN_MATCH_MODES}; such a domain never matches anything",
            )
        if d["table_name"] not in tables:
            add("warning", "domain_dropped", name, f"table {d['table_name']!r} is not active, so the domain is ignored")
            continue
        try:
            live = {r[1] for r in conn.execute(f"PRAGMA table_info('{d['table_name']}')").fetchall()}
            if live and d["column_name"] not in live:
                add("error", "domain_column_missing", name, f"{d['table_name']}.{d['column_name']} does not exist")
        except Exception:
            pass
        for k in schema._json(d.get("preferred_for"), []):
            if k not in keys:
                add(
                    "warning",
                    "unknown_preferred_for",
                    name,
                    f"preferred_for names {k!r}, which is not an active concept",
                )
        if include_info and mode in ("exact_or_prefix", "prefix"):
            try:
                values = [
                    str(v)
                    for v in store.query(
                        f"SELECT DISTINCT {d['column_name']} AS v FROM {d['table_name']} WHERE {d['column_name']} IS NOT NULL LIMIT 500"
                    )["v"].tolist()
                ]
                if entity_lexicon.suggest_match_mode(values) == "components":
                    add(
                        "info",
                        "suggest_components",
                        name,
                        "values look like 'Name, ST' labels; match_mode='components' lets "
                        "people use any city of the label and a state qualifier",
                    )
            except Exception:
                pass

    for r in data["relationships"]:
        if r["left_table"] not in tables or r["right_table"] not in tables:
            add(
                "warning",
                "relationship_dropped",
                r["rel_key"],
                "an endpoint table is not active, so the relationship is ignored",
            )

    return [i for i in issues if include_info or i.severity != "info"]


# Join keys the catalog declares as relationships. DuckDB does not enforce them on tables that
# already exist, so the reconciliation happens here: a house whose tract is not in nri_tracts is
# silently dropped by every risk join, and nothing else would ever say so.
_ORPHAN_CHECKS = (
    ("houses", "tract_fips", "census_tracts", "tract_fips"),
    ("houses", "msa_code", "census_msa", "msa_code"),
    ("sold_homes", "tract_fips", "census_tracts", "tract_fips"),
    ("nri_tracts", "county_fips", "cbsa_counties", "county_fips"),
)


def orphan_join_keys(conn=None) -> list[Issue]:
    """Rows whose declared join key has no match: the ones inner joins silently drop.

    A warning below 50% of the child table and an error above it: a handful of unmatched tracts
    is real-world geocoding noise, while a majority means the load pipeline is joining on the
    wrong column and every risk/census answer is wrong in the same direction.
    """
    conn = conn or store.get_conn()
    issues: list[Issue] = []
    for child, child_col, parent, parent_col in _ORPHAN_CHECKS:
        try:
            row = conn.execute(
                f"SELECT (SELECT COUNT(*) FROM {child} WHERE {child_col} IS NOT NULL), "
                f"(SELECT COUNT(*) FROM {child} c WHERE c.{child_col} IS NOT NULL AND NOT EXISTS "
                f"(SELECT 1 FROM {parent} p WHERE p.{parent_col} = c.{child_col})), "
                f"(SELECT COUNT(*) FROM {parent})"
            ).fetchone()
        except Exception:
            continue  # a table this build does not have
        total, orphans, parent_rows = int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)
        if not total or not orphans or not parent_rows:
            continue  # nothing to compare against is not an orphan report
        share = 100.0 * orphans / total
        issues.append(
            Issue(
                "error" if share >= 50 else "warning",
                "orphan_join_key",
                f"{child}.{child_col}",
                f"{orphans:,} of {total:,} rows ({share:.1f}%) have a value that does not exist in "
                f"{parent}.{parent_col}; every join on this key drops them silently",
            )
        )
    return issues


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    issues = lint_catalog()
    if "--orphans" in args:
        issues = issues + orphan_join_keys()
    for i in issues:
        print(i)
    errors = [i for i in issues if i.severity == "error"]
    print(
        f"\n{len(errors)} error(s), {sum(i.severity == 'warning' for i in issues)} warning(s), "
        f"{sum(i.severity == 'info' for i in issues)} info"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
