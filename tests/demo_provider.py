"""A stand-in for a FUTURE derived-measure provider, used only by test_future_dataset_extensibility.py.

It shows how little a new provider needs: a module with PROVIDER (services/derived_measures.py documents the protocol).
Nothing in the planner, the tool layer, the policy or the validator is edited to use it.
"""
from __future__ import annotations

import db.duckdb_store as store
from services.derived_measures import DerivedResult, MeasureInfo


class DoublePopulationProvider:
    id = "demo_provider"
    entity_type = "MSA"
    entity_label = "metropolitan area"
    measures = {"double_population": MeasureInfo("double_population", "doubled population", "people",
                                                 "Twice the Census MSA population.", phrases=("doubled",))}

    def compute(self, entities, measures, *, request: str = "") -> DerivedResult:
        names = [e["value"] for e in entities]
        marks = ",".join("?" for _ in names)
        df = store.query(f"SELECT name AS msa_name, population * 2 AS double_population FROM census_msa WHERE name IN ({marks})", names)
        return DerivedResult(df, method=["Doubled population is twice the Census MSA population."])


PROVIDER = DoublePopulationProvider()
