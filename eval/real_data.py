"""A small evaluation tier whose answers come from the real database, not the fixture.

Every expected value in ``eval/golden_set.py`` is derived from ``eval/fixtures.py``: the suite
proves the agent computes what the fixture says, not that ingestion, tract assignment and the
joins are right on the 85 k-tract / 740 k-incident dataset the app actually serves. This tier
closes that gap for ten headline numbers.

The ground truth here is a checked-in SQL statement per case, executed read-only against
``settings.duckdb_path`` - a different code path from the agent under test, which plans its own
SQL. Statements are pure SELECTs (the loader below refuses anything else), so running this
cannot modify the database it measures.

These are integration cases: they need a populated database and are excluded from the default
pytest run (see pytest.ini). Run the questions through the agent with
``python run_eval.py``-style tooling only when a model server is available; the testable part -
that the SQL still returns a number and that a reply is scored correctly against it - runs
without one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RealDataCase:
    id: str
    question: str
    sql: str
    tolerance: float = 0.001  # fraction of the true value accepted
    unit: str = ""


CASES: tuple[RealDataCase, ...] = (
    RealDataCase("real-house-count", "How many houses are in the database?", "SELECT COUNT(*) FROM houses"),
    RealDataCase(
        "real-avg-price",
        "What is the average list price of the houses?",
        "SELECT AVG(price) FROM houses WHERE price IS NOT NULL",
    ),
    RealDataCase(
        "real-crime-latest-year",
        "How many crime incidents are recorded for the most recent year?",
        "SELECT COUNT(*) FROM crime_incidents WHERE year = (SELECT MAX(year) FROM crime_incidents)",
    ),
    RealDataCase(
        "real-avg-risk-score",
        "What is the average FEMA risk score across the loaded tracts?",
        "SELECT AVG(risk_score) FROM nri_tracts WHERE risk_score IS NOT NULL",
        tolerance=0.01,
    ),
    RealDataCase(
        "real-tract-population",
        "What is the total population of the census tracts loaded?",
        "SELECT SUM(population) FROM census_tracts WHERE population IS NOT NULL",
    ),
    RealDataCase("real-sold-count", "How many county sold-home records are loaded?", "SELECT COUNT(*) FROM sold_homes"),
    RealDataCase(
        "real-bike-segments", "How many bike infrastructure segments are loaded?", "SELECT COUNT(*) FROM bike_routes"
    ),
    RealDataCase(
        "real-transit-ridership",
        "What was total transit ridership in the most recent year with ridership data?",
        "SELECT SUM(upt) FROM transit_ridership WHERE year = (SELECT MAX(year) FROM transit_ridership)",
        tolerance=0.01,
    ),
    RealDataCase(
        "real-zhvi-latest",
        "What is the average home value in the latest Zillow month?",
        "SELECT AVG(home_value) FROM zhvi WHERE date = (SELECT MAX(date) FROM zhvi)",
        tolerance=0.01,
    ),
    RealDataCase(
        "real-transit-agencies", "How many transit agencies are loaded?", "SELECT COUNT(*) FROM transit_agencies"
    ),
)

_NUMBER_RE = re.compile(r"-?\$?\s?\d[\d,]*\.?\d*%?")


def _numbers(text: str) -> list[float]:
    out = []
    for match in _NUMBER_RE.findall(text or ""):
        cleaned = match.replace("$", "").replace(",", "").replace("%", "").strip()
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def connect():
    """Read-only connection to the live database (raises if it has not been built)."""
    import duckdb
    from config import settings

    path = Path(getattr(settings, "duckdb_path"))
    if not path.exists():
        raise FileNotFoundError(path)
    return duckdb.connect(str(path), read_only=True)


def ground_truth(case: RealDataCase, conn=None) -> float:
    """The case's answer, computed by its own SQL rather than by the agent's plan."""
    sql = case.sql.strip().rstrip(";")
    if not sql.lower().startswith("select"):
        raise ValueError(f"{case.id}: ground-truth SQL must be a SELECT")
    owned = conn is None
    conn = conn or connect()
    try:
        row = conn.execute(sql).fetchone()
    finally:
        if owned:
            conn.close()
    if not row or row[0] is None:
        raise ValueError(f"{case.id}: ground-truth query returned no value")
    return float(row[0])


def evaluate_reply(case: RealDataCase, reply: str, truth: float | None = None) -> bool:
    """Whether a textual reply states the true value within the case's tolerance."""
    target = float(truth if truth is not None else ground_truth(case))
    allowed = abs(target) * case.tolerance
    return any(abs(number - target) <= allowed for number in _numbers(reply))
