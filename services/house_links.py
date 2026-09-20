"""Records from user-added datasets that link to ONE house (used by House Chat).

The join is derived from the unified catalog's relationship graph (``schema.house_link_plan``), so a
dataset linked directly to houses, or only through nri_tracts / census_tracts, is handled the same way.
"""
from __future__ import annotations

import db.duckdb_store as store
import db.schema_catalog as schema

MAX_ROWS = 10
MAX_COLUMNS = 25


def linked_records(house_id: str, dataset: str = "") -> str:
    datasets = schema.house_linked_datasets()
    if not datasets:
        return "No user-added datasets are linked to houses yet."
    dataset = (dataset or "").strip()
    chosen = [d for d in datasets if not dataset or d["name"] == dataset]
    if not chosen:
        return ("Unknown dataset '%s'. Linked datasets: %s." % (dataset, ", ".join(d["name"] for d in datasets)))
    blocks = []
    for d in chosen:
        plan = schema.house_link_plan(d["name"])
        if not plan:
            continue
        try:
            df = store.query(plan["sql"], [house_id])
        except Exception as exc:
            blocks.append(f"[{d['name']}] could not be queried: {exc}")
            continue
        header = f"[{d['name']}] {d['description']} Joined via {plan['join_text']}."
        if df.empty:
            blocks.append(header + " No rows match this house.")
            continue
        df = df.iloc[:MAX_ROWS, :MAX_COLUMNS]
        blocks.append(f"{header} ({len(df)} row(s) shown)\n" + df.to_string(index=False))
    return "\n\n".join(blocks) if blocks else "No linked records were found for this house."
