"""Registry of *derived-measure providers*.

A derived measure is a number the stored tables cannot produce with SQL: polygon-derived land area, routing times,
model scores...  They used to be special-cased inside ``query_database`` (one hard-coded branch for "population
density").  Now a concept declares what it needs in catalog metadata::

    derived={"provider": "msa_geometry", "measures": ["land_area"]}

and ``agents/answer_policy.py`` routes any plan whose concepts declare ``derived`` to the named provider.  Adding a new
kind of derived measure for a future dataset is therefore three steps, with no edit to the tool or planner layers:

1. write a provider module exposing ``PROVIDER`` (the protocol below),
2. register it here (``_PROVIDER_MODULES``, or ``register_provider_module`` at start-up / in tests),
3. declare ``derived`` / ``derived_support`` on the concept(s) in the catalog.

Provider protocol (duck-typed)::

    PROVIDER.id            str            same id used in concept metadata
    PROVIDER.entity_type   str            entity type the provider needs resolved from the request (e.g. "MSA")
    PROVIDER.entity_label  str            how to say it to a user ("metropolitan area")
    PROVIDER.evidence_label str           optional: shown in the evidence where the SQL normally appears
    PROVIDER.measures      dict[str, MeasureInfo]
    PROVIDER.compute(entities, measures, *, request) -> DerivedResult      # raises DerivedError, never a bare error
    PROVIDER.detect_measures(request) -> set[str]                          # optional: measures named in the request

Provider modules are imported lazily (only when a request actually needs one), so heavy dependencies such as
geopandas are not loaded by requests that never touch them.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class MeasureInfo:
    id: str
    label: str
    unit: str
    description: str
    # Words that mean this measure.  They are NOT catalog aliases (generic words like "density" or "population" would
    # capture unrelated questions); they are consulted only AFTER a specific concept has already routed the request to
    # this provider, to pick up the other measures asked for in the same sentence ("population, land area and density").
    phrases: tuple[str, ...] = ()


@dataclass
class DerivedResult:
    frame: pd.DataFrame
    method: list[str] = field(default_factory=list)   # how the numbers were produced; shown before the table
    notes: list[str] = field(default_factory=list)    # caveats about THIS result; shown after the table


class DerivedError(Exception):
    """The provider could not produce the measure.  ``code`` is an ``agents.answer_status`` code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# provider id -> module that exposes PROVIDER
_PROVIDER_MODULES: dict[str, str] = {"msa_geometry": "services.census_metrics"}


def register_provider_module(provider_id: str, module_path: str) -> None:
    _PROVIDER_MODULES[provider_id] = module_path


def detect_measures(provider, request: str) -> set[str]:
    """Measures of ``provider`` whose declared phrases appear in ``request`` (works for any provider)."""
    from db.text_match import has_phrase
    return {m.id for m in provider.measures.values() if any(has_phrase(request, ph) for ph in m.phrases)}


def known_provider_ids() -> list[str]:
    return sorted(_PROVIDER_MODULES)


def get_provider(provider_id: str):
    """Import (once) and return the provider; ``KeyError`` for an unregistered id."""
    module_path = _PROVIDER_MODULES[provider_id]
    return importlib.import_module(module_path).PROVIDER
