r"""
Update eval/golden_set.py from SQL ground truth in eval/fixture_data/eval_fixture.duckdb.

Run this BEFORE run_eval.py from the project root:

    python .\update_eval_ground_truth.py
    python .\run_eval.py

This script intentionally derives the structured expected values and the
numeric facts embedded in the free-text rubrics from SQL executed against the
actual evaluation fixture.  It does NOT ask the agent to generate SQL and it
never uses the agent's previous answers as ground truth.

The script also validates fixture invariants, but house-count/city truth is based
on the `houses` table itself rather than `is_favorite`. The evaluation questions
intentionally test the houses present in the fixture, not saved/favorite semantics.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

try:
    import duckdb
except ImportError as exc:
    raise SystemExit(
        "duckdb is required. Install it in the house_agent environment with: "
        "pip install duckdb"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "eval" / "fixture_data" / "eval_fixture.duckdb"
GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_set.py"


# ---------------------------------------------------------------------------
# SQL ground truth
# ---------------------------------------------------------------------------

SQL = {
    # House inventory: intentionally do NOT filter on is_favorite.
    "house_cities": """
        SELECT DISTINCT city
        FROM houses
        WHERE city IS NOT NULL
        ORDER BY city
    """,
    "house_count_pittsburgh": """
        SELECT COUNT(*)
        FROM houses
        WHERE LOWER(city) = 'pittsburgh'
    """,
    "house_count_total": """
        SELECT COUNT(*)
        FROM houses
        WHERE TRUE
    """,
    "austin_avg_walk_score": """
        SELECT AVG(walk_score)
        FROM houses
        WHERE TRUE
          AND LOWER(city) = 'austin'
          AND walk_score IS NOT NULL
    """,
    "austin_walk_score_rows": """
        SELECT COUNT(*) AS total_rows, COUNT(walk_score) AS non_null_walk_scores
        FROM houses
        WHERE TRUE
          AND LOWER(city) = 'austin'
    """,
    "miami_house_walk_score": """
        SELECT walk_score
        FROM houses
        WHERE house_id = 'h4'
    """,
    "pittsburgh_house_nri_score": """
        SELECT n.risk_score
        FROM houses h
        JOIN nri_tracts n
          ON h.tract_fips = n.tract_fips
        WHERE h.house_id = 'h1'
    """,
    "pittsburgh_arms_length_avg_sold_price": """
        SELECT AVG(s.sold_price)
        FROM sold_homes s
        WHERE s.tract_fips = '42003140300'
          AND (s.is_arms_length IS NULL OR s.is_arms_length = TRUE)
          AND s.sold_price > 1000
    """,
    "pittsburgh_tract_population": """
        SELECT population
        FROM census_tracts
        WHERE tract_fips = '42003140300'
    """,

    # Four-metro population ranking.  Use short labels so the golden scorer
    # can match either "Miami" or a full MSA title in an agent response.
    "msa_population_rank": """
        SELECT
            split_part(m.name, ',', 1) AS metro,
            CAST(m.population AS BIGINT) AS population
        FROM census_msa m
        WHERE m.population IS NOT NULL
          AND m.msa_code NOT LIKE 'X%'
          AND (
              LOWER(m.name) LIKE 'pittsburgh%'
              OR LOWER(m.name) LIKE 'denver%'
              OR LOWER(m.name) LIKE 'miami%'
              OR LOWER(m.name) LIKE 'austin%'
          )
        ORDER BY population DESC
    """,

    # Riverine risk must be calculated through the documented relationship:
    # census_msa -> cbsa_counties -> nri_tracts.
    "msa_flood_risk_rank": """
        WITH target_msas AS (
            SELECT m.msa_code, split_part(m.name, ',', 1) AS metro
            FROM census_msa m
            WHERE m.msa_code NOT LIKE 'X%'
              AND (
                  LOWER(m.name) LIKE 'pittsburgh%'
                  OR LOWER(m.name) LIKE 'denver%'
                  OR LOWER(m.name) LIKE 'miami%'
                  OR LOWER(m.name) LIKE 'austin%'
              )
        )
        SELECT
            t.metro,
            AVG(n.rfld_risks) AS avg_riverine_flood_risk
        FROM target_msas t
        JOIN cbsa_counties cb
          ON t.msa_code = cb.cbsa_code
        JOIN nri_tracts n
          ON n.county_fips = cb.state_fips || cb.county_fips
        GROUP BY t.metro
        ORDER BY avg_riverine_flood_risk DESC
    """,

    # Free-text tradeoff facts.
    "tradeoff_population_and_flood": """
        WITH target_msas AS (
            SELECT m.msa_code, split_part(m.name, ',', 1) AS metro,
                   CAST(m.population AS BIGINT) AS population
            FROM census_msa m
            WHERE m.msa_code NOT LIKE 'X%'
              AND (
                  LOWER(m.name) LIKE 'pittsburgh%'
                  OR LOWER(m.name) LIKE 'miami%'
              )
        ),
        flood AS (
            SELECT t.metro, AVG(n.rfld_risks) AS avg_riverine_flood_risk
            FROM target_msas t
            JOIN cbsa_counties cb ON t.msa_code = cb.cbsa_code
            JOIN nri_tracts n ON n.county_fips = cb.state_fips || cb.county_fips
            GROUP BY t.metro
        )
        SELECT t.metro, t.population, f.avg_riverine_flood_risk
        FROM target_msas t
        JOIN flood f USING (metro)
        ORDER BY t.metro
    """,

    # Unmatched MSA / CBSA explanation.
    "unmatched_msa": """
        SELECT
            m.name,
            m.msa_code,
            c.cbsa_code
        FROM census_msa m
        LEFT JOIN cbsa_counties c
          ON m.msa_code = c.cbsa_code
        WHERE m.name = 'Sample Micro Area'
    """,

    "seattle_house_count": """
        SELECT COUNT(*)
        FROM houses
        WHERE TRUE
          AND LOWER(city) = 'seattle'
    """,
}


def q(conn, name: str):
    """Execute one named SQL query and return rows as dictionaries."""
    cur = conn.execute(SQL[name])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def one_value(conn, name: str, column: str | None = None):
    rows = q(conn, name)
    if len(rows) != 1:
        raise RuntimeError(f"{name}: expected exactly one row, got {len(rows)}: {rows!r}")
    row = rows[0]
    return next(iter(row.values())) if column is None else row[column]


def normalize_number(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    value = float(value)
    return int(value) if value.is_integer() else value


def first_number(rows, key):
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row, got {len(rows)}: {rows!r}")
    return normalize_number(rows[0][key])


def build_truth(conn) -> dict:
    house_cities = [r["city"] for r in q(conn, "house_cities")]
    pgh_count = int(one_value(conn, "house_count_pittsburgh"))
    total_count = int(one_value(conn, "house_count_total"))
    austin_avg = normalize_number(one_value(conn, "austin_avg_walk_score"))
    austin_rows = q(conn, "austin_walk_score_rows")[0]
    miami_walk = normalize_number(one_value(conn, "miami_house_walk_score"))
    pgh_nri = normalize_number(one_value(conn, "pittsburgh_house_nri_score"))
    pgh_sold = normalize_number(one_value(conn, "pittsburgh_arms_length_avg_sold_price"))
    pgh_pop = int(one_value(conn, "pittsburgh_tract_population"))
    seattle_count = int(one_value(conn, "seattle_house_count"))

    pop_rows = q(conn, "msa_population_rank")
    flood_rows = q(conn, "msa_flood_risk_rank")
    tradeoff_rows = q(conn, "tradeoff_population_and_flood")
    unmatched = q(conn, "unmatched_msa")

    # These are fixture integrity checks.  They validate the actual houses table
    # without imposing saved/favorite semantics.
    if total_count == 0:
        raise RuntimeError(
            "Fixture integrity failure: houses contains zero rows. "
            "The eval fixture must contain the houses expected by the golden set."
        )
    if total_count != 6:
        raise RuntimeError(
            f"Fixture integrity failure: expected 6 houses, got {total_count}."
        )
    if sorted(house_cities) != ["Austin", "Denver", "Miami", "Pittsburgh"]:
        raise RuntimeError(
            "Fixture integrity failure: house cities differ from the intended fixture: "
            f"{house_cities!r}"
        )
    if pgh_count == 0:
        raise RuntimeError(
            "Fixture integrity failure: Pittsburgh has zero houses. "
            "Expected the house fixture to contain Pittsburgh entries."
        )
    if not house_cities:
        raise RuntimeError("Fixture integrity failure: house_cities is empty.")
    if seattle_count != 0:
        raise RuntimeError(f"Fixture integrity failure: expected zero Seattle houses, got {seattle_count}")
    if int(austin_rows["total_rows"]) == 0:
        raise RuntimeError("Fixture integrity failure: no Austin houses.")
    if austin_avg is None:
        raise RuntimeError("Fixture integrity failure: Austin average walk score is NULL.")
    if len(pop_rows) != 4:
        raise RuntimeError(f"Expected 4 target MSAs for population ranking, got {len(pop_rows)}")
    if len(flood_rows) != 4:
        raise RuntimeError(f"Expected 4 target MSAs for flood ranking, got {len(flood_rows)}")
    if len(tradeoff_rows) != 2:
        raise RuntimeError(f"Expected Pittsburgh + Miami tradeoff rows, got {len(tradeoff_rows)}")
    if len(unmatched) != 1:
        raise RuntimeError(f"Expected one unmatched MSA row, got {len(unmatched)}")

    unmatched_row = unmatched[0]
    no_cbsa_match = unmatched_row["cbsa_code"] is None
    if not no_cbsa_match:
        raise RuntimeError(
            "Fixture integrity failure: Sample Micro Area unexpectedly matched a CBSA: "
            f"{unmatched_row['cbsa_code']}"
        )

    # Numeric tradeoff facts by metro.
    tradeoff = {r["metro"]: r for r in tradeoff_rows}
    for metro in ("Pittsburgh", "Miami"):
        if metro not in tradeoff:
            raise RuntimeError(f"Tradeoff query did not return {metro}: {tradeoff_rows!r}")

    return {
        "cities_with_houses": house_cities,
        "pittsburgh_house_count": pgh_count,
        "total_house_count": total_count,
        "austin_avg_walk_score": austin_avg,
        "miami_house_walk_score": miami_walk,
        "pittsburgh_house_nri_score": pgh_nri,
        "pittsburgh_arms_length_avg_sold_price": pgh_sold,
        "pittsburgh_tract_population": pgh_pop,
        "msa_population_rank": [r["metro"] for r in pop_rows],
        "msa_flood_risk_rank": [r["metro"] for r in flood_rows],
        "tradeoff": {
            metro: {
                "population": int(tradeoff[metro]["population"]),
                "flood_risk": normalize_number(tradeoff[metro]["avg_riverine_flood_risk"]),
            }
            for metro in ("Pittsburgh", "Miami")
        },
        "no_houses_in_seattle": seattle_count,
        "austin_has_missing_walk_score": int(austin_rows["non_null_walk_scores"]),
        "unmatched_msa_has_cbsa_match": False,
        "unmatched_msa_code": unmatched_row["msa_code"],
    }


# ---------------------------------------------------------------------------
# Golden-set patching
# ---------------------------------------------------------------------------


def replace_assignment(text: str, variable: str, value) -> str:
    """Replace a top-level simple assignment while preserving surrounding code."""
    pattern = re.compile(rf"^({re.escape(variable)}\s*=\s*).*?$", re.MULTILINE)
    replacement = f"{variable} = {repr(value)}"
    new_text, n = pattern.subn(replacement, text, count=1)
    if n != 1:
        raise RuntimeError(f"Could not find assignment for {variable}")
    return new_text


def patch_golden(text: str, truth: dict) -> str:
    """
    Replace the imported fixture constants in golden_set.py by replacing the
    corresponding `expected=` expressions in the individual GoldenExample
    blocks.  This keeps the golden file readable and reviewable.

    The free-text rubrics are also updated because they embed the SQL-derived
    population/flood-risk facts.
    """

    def patch_expected(example_id: str, new_expr: str, source: str) -> str:
        marker = f'id="{example_id}"'
        start = source.find(marker)
        if start < 0:
            raise RuntimeError(f"Example {example_id} not found")
        block_start = source.rfind("GoldenExample(", 0, start)
        block_end = source.find("    ),", start)
        if block_start < 0 or block_end < 0:
            raise RuntimeError(f"Could not locate block for {example_id}")
        block = source[block_start:block_end]
        # Expected values are emitted on a single GoldenExample field line.
        # Replace the entire physical line rather than stopping at the first
        # comma, because list literals legitimately contain commas.  Replacing
        # the whole line also repairs a previously-corrupted line such as:
        #   expected=['Austin', 'Denver'], 'Miami', ...],
        # which older versions of this updater could generate.
        new_block, n = re.subn(
            r"(?m)^(?P<indent>[ \t]*)expected\s*=.*$",
            lambda m: f"{m.group('indent')}expected={new_expr},",
            block,
            count=1,
        )
        if n != 1:
            raise RuntimeError(f"Could not patch expected for {example_id}")
        return source[:block_start] + new_block + source[block_end:]

    text = patch_expected("cities_with_houses", repr(truth["cities_with_houses"]), text)
    text = patch_expected("msa_population_rank", repr(truth["msa_population_rank"]), text)
    text = patch_expected("msa_flood_risk_rank", repr(truth["msa_flood_risk_rank"]), text)
    text = patch_expected("pittsburgh_house_count", repr([truth["pittsburgh_house_count"]]), text)
    text = patch_expected("total_house_count", repr([truth["total_house_count"]]), text)
    text = patch_expected("austin_avg_walk_score", repr([truth["austin_avg_walk_score"]]), text)
    text = patch_expected("pittsburgh_arms_length_avg_sold_price", repr([truth["pittsburgh_arms_length_avg_sold_price"]]), text)
    text = patch_expected("pittsburgh_tract_population", repr([truth["pittsburgh_tract_population"]]), text)
    text = patch_expected("miami_house_walk_score", repr([truth["miami_house_walk_score"]]), text)
    text = patch_expected("pittsburgh_house_nri_score", repr([truth["pittsburgh_house_nri_score"]]), text)

    # Replace the two free-text rubrics completely so their numeric claims are
    # guaranteed to be derived from the current fixture.
    p = truth["tradeoff"]["Pittsburgh"]
    m = truth["tradeoff"]["Miami"]
    tradeoff_rubric = (
        "A good answer notes that Miami has much higher riverine flood risk "
        f"(~{m['flood_risk']} vs ~{p['flood_risk']} for Pittsburgh) "
        f"and a larger metro population (~{m['population']:,} vs "
        f"~{p['population']:,}), and frames this as a genuine tradeoff rather than "
        "just repeating numbers with no takeaway. It should not claim Pittsburgh "
        "has higher flood risk than Miami, and should not invent figures not "
        "derivable from the two metros' data."
    )

    def patch_rubric(example_id: str, rubric: str, source: str) -> str:
        marker = f'id="{example_id}"'
        start = source.find(marker)
        if start < 0:
            raise RuntimeError(f"Example {example_id} not found")
        block_start = source.rfind("GoldenExample(", 0, start)
        close_marker = "\n    ),"
        close_start = source.find(close_marker, start)
        block_end = close_start + len(close_marker) if close_start >= 0 else -1
        if block_start < 0 or block_end < 0:
            raise RuntimeError(f"Could not locate {example_id} GoldenExample block")
        block = source[block_start:block_end]

        rubric_match = re.search(
            r'(?ms)^(?P<indent>[ \t]*)rubric\s*=\s*.*?(?=^    \),\s*$)',
            block,
        )
        if not rubric_match:
            raise RuntimeError(f"Could not locate rubric for {example_id}")

        indent = rubric_match.group("indent")
        replacement = f"{indent}rubric={repr(rubric)},\n"
        patched_block = (
            block[:rubric_match.start()]
            + replacement
            + block[rubric_match.end():]
        )
        return source[:block_start] + patched_block + source[block_end:]

    text = patch_rubric("tradeoff_pittsburgh_vs_miami", tradeoff_rubric, text)

    unmatched_rubric = (
        "'Sample Micro Area' has no matching row in cbsa_counties. A good answer "
        "says no CBSA match was found. It should not claim a specific CBSA/metro "
        "affiliation that is not present in the data."
    )
    text = patch_rubric("unmatched_msa_explanation", unmatched_rubric, text)

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"Fixture database not found: {args.db}")
    if not args.golden.exists():
        raise SystemExit(f"Golden set not found: {args.golden}")

    print(f"Ground truth DB : {args.db}")
    print(f"Golden file     : {args.golden}")

    conn = duckdb.connect(str(args.db), read_only=True)
    try:
        truth = build_truth(conn)
    finally:
        conn.close()

    print("\nSQL ground truth:")
    print(json.dumps(truth, indent=2, sort_keys=True))

    source = args.golden.read_text(encoding="utf-8")
    updated = patch_golden(source, truth)

    # Syntax-check the generated Python before touching the golden file.
    ast.parse(updated, filename=str(args.golden))

    if args.check_only:
        print("\nCHECK ONLY: golden_set.py would be updated, but was not modified.")
        return

    backup = args.golden.with_suffix(args.golden.suffix + ".bak")
    backup.write_text(source, encoding="utf-8")
    args.golden.write_text(updated, encoding="utf-8")

    print(f"\nUpdated {args.golden}")
    print(f"Backup  {backup}")
    print("Run: python .\\run_eval.py")


if __name__ == "__main__":
    main()
