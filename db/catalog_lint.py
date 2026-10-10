"""Catalog lint: static checks that make adding a dataset or a concept safe.

The agents are driven entirely by catalog metadata, which means a metadata mistake does not fail loudly - a concept
whose table name has a typo is silently dropped by ``schema_catalog.reload()``, a misspelled knob (``require_entity``)
silently does nothing, and a generic alias quietly steals a word from every future dataset.  This module turns those
silent failures into explicit findings, for built-in and uploaded objects alike.

    python -m db.catalog_lint            # prints findings; exit status 1 when there are errors

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
    "key", "tables", "columns", "aliases", "description", "operations", "filters", "null_policy", "orderings",
    "groupings", "grain", "entity_types", "rollup", "rollup_spec", "required_terms", "excluded_terms",
    "default_operation", "origin", "dataset_id",
}
# Mirrors services/dataset_onboarding._GENERIC_ALIASES (imported when available so there is a single source of truth).
_FALLBACK_GENERIC = {"value", "values", "score", "rate", "data", "number", "count", "total", "index", "rating", "level",
                     "type", "name", "id", "code", "year", "date", "status", "city", "state", "price", "risk",
                     "population", "average", "area", "size", "list", "rank", "house", "home", "houses"}


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
    add = lambda sev, code, subject, msg: issues.append(Issue(sev, code, subject, msg))     # noqa: E731

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
        except Exception as exc:                                            # pragma: no cover - import problems
            add("error", "provider_import_failed", pid, f"derived provider {pid!r} could not be imported: {exc}")

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
            add("error", "unknown_key", key, f"unknown concept key(s) {unknown}; known knobs are {sorted(CONCEPT_KNOBS)} "
                                              "(a misspelled knob silently does nothing)")
        missing = [t for t in item.get("tables", []) if t not in tables]
        if missing:
            add(sev_missing, "concept_dropped", key, f"references table(s) {missing} that are not active, so the concept is "
                                                     "silently dropped from the registry")

        seen = set()
        for alias in item.get("aliases", []):
            n = normalize(alias)
            if not n:
                add("error", "empty_alias", key, f"alias {alias!r} normalizes to nothing")
                continue
            if n in seen:
                add("warning", "duplicate_alias", key, f"alias {alias!r} is listed twice")
            seen.add(n)
            if not item.get("gap"):                 # a known-gap topic yields to any dataset that covers it: it owns nothing
                owners.setdefault(n, []).append(key)
            if builtin and _alias_is_generic(alias, generic) and not item.get("gap"):
                add("warning", "generic_alias", f"{key}:{n}", f"alias {alias!r} is too generic for a built-in concept: "
                    "onboarding would stop every future dataset from using that word")

        # ---- knobs ------------------------------------------------------------------------------------------------
        needs = item.get("requires_entity")
        if needs is not None:
            if not isinstance(needs, list) or not all(isinstance(x, str) for x in needs):
                add("error", "bad_knob", key, "requires_entity must be a list of entity-type names")
            else:
                for et in needs:
                    if et not in domain_types:
                        add("error", "unknown_entity_type", key, f"requires_entity names {et!r}, but no entity domain has "
                                                                 "that entity_type, so the concept can never apply")
        for knob in ("derived", "derived_support"):
            spec = item.get(knob)
            if spec is None:
                continue
            if not isinstance(spec, dict) or "provider" not in spec or not isinstance(spec.get("measures"), list):
                add("error", "bad_knob", key, f"{knob} must be {{'provider': id, 'measures': [...]}}")
                continue
            provider = providers.get(spec["provider"])
            if provider is None:
                add("error", "unknown_provider", key, f"{knob} names provider {spec['provider']!r}, which is not registered "
                                                      f"(known: {sorted(providers)})")
                continue
            for m in spec["measures"]:
                if m not in provider.measures:
                    add("error", "unknown_measure", key, f"{knob} asks provider {spec['provider']!r} for {m!r}; it offers "
                                                         f"{sorted(provider.measures)}")
            if knob == "derived" and provider.entity_type not in item.get("entity_types", []):
                add("error", "derived_needs_entity", key, f"provider {spec['provider']!r} needs an entity of type "
                    f"{provider.entity_type!r}; the concept must list it in entity_types or no place will be resolved")
        if item.get("gap"):
            if item.get("tables"):
                add("error", "gap_with_tables", key, "a gap concept declares an UNAVAILABLE topic and must have no tables")
            if not str(item.get("description", "")).strip():
                add("error", "gap_without_description", key, "a gap concept's description is the message shown to the user")
            elif re.search(r"\d", item["description"]):
                add("warning", "digits_in_gap_description", key, "avoid digits: the message is shown as evidence and digits "
                                                                 "are what the validator treats as data")
        for other in item.get("overrides", []):
            if other not in keys:
                add("warning", "unknown_override", key, f"overrides {other!r}, which is not an active concept")
        for op in item.get("operations", []):
            if op.get("op") == "lookup" and not any(re.search(rf"\b{re.escape(t)}\.", op.get("expr", "")) for t in item.get("tables", [])):
                add("warning", "lookup_expr_tables", key, f"lookup expression {op.get('expr')!r} references none of the concept's tables")

    for alias, who in sorted(owners.items()):
        who = sorted(set(who))
        if len(who) > 1:
            add("warning", "alias_collision", alias, f"alias {alias!r} is claimed by {who}; the planner will select all of them")

    # ---- entity domains ------------------------------------------------------------------------------------------
    for d in domains:
        name = d["name"]
        mode = d.get("match_mode") or "exact_or_prefix"
        if mode not in KNOWN_MATCH_MODES:
            add("error", "unknown_match_mode", name, f"match_mode {mode!r} is not one of {KNOWN_MATCH_MODES}; such a domain "
                                                     "never matches anything")
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
                add("warning", "unknown_preferred_for", name, f"preferred_for names {k!r}, which is not an active concept")
        if include_info and mode in ("exact_or_prefix", "prefix"):
            try:
                values = [str(v) for v in store.query(
                    f"SELECT DISTINCT {d['column_name']} AS v FROM {d['table_name']} WHERE {d['column_name']} IS NOT NULL LIMIT 500"
                )["v"].tolist()]
                if entity_lexicon.suggest_match_mode(values) == "components":
                    add("info", "suggest_components", name, "values look like 'Name, ST' labels; match_mode='components' lets "
                        "people use any city of the label and a state qualifier")
            except Exception:
                pass

    for r in data["relationships"]:
        if r["left_table"] not in tables or r["right_table"] not in tables:
            add("warning", "relationship_dropped", r["rel_key"], "an endpoint table is not active, so the relationship is ignored")

    return [i for i in issues if include_info or i.severity != "info"]


def main(argv=None) -> int:
    issues = lint_catalog()
    for i in issues:
        print(i)
    errors = [i for i in issues if i.severity == "error"]
    print(f"\n{len(errors)} error(s), {sum(i.severity == 'warning' for i in issues)} warning(s), "
          f"{sum(i.severity == 'info' for i in issues)} info")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
