"""
eval/tests/test_scoring.py

Unit tests for eval/scoring.py's assert-equal logic, against hand-crafted
reply strings. No LLM, no database — these test the scorer itself: does it
actually discriminate a correct reply from a wrong one, in both directions?
A scorer that only ever passes (or only ever fails) is worse than no scorer.

Run directly: python3 eval/tests/test_scoring.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.golden_set import GoldenExample
from eval.scoring import score_structured

_passed = 0
_failed = 0


def check(label: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [pass] {label}")
    else:
        _failed += 1
        print(f"  [FAIL] {label}  {detail}")


def verdict(example: GoldenExample, reply: str) -> str:
    return score_structured(example, reply).verdict


# ── Set comparison (order_matters=False) ────────────────────────────────────
print("Set comparison")
ex = GoldenExample(id="t_set", agent="general", question="q",
                    expected=["Pittsburgh", "Denver", "Miami", "Austin"],
                    order_matters=False, value_type="string",
                    distractors=["Seattle"])

check("correct set, any order -> PASS",
      verdict(ex, "You have houses in Austin, Miami, Pittsburgh, and Denver.") == "PASS")
check("missing one city -> FAIL",
      verdict(ex, "You have houses in Pittsburgh, Denver, and Miami.") == "FAIL")
check("all correct but distractor also present -> FAIL",
      verdict(ex, "Pittsburgh, Denver, Miami, Austin, and Seattle.") == "FAIL")

# ── Ordered comparison (order_matters=True) ─────────────────────────────────
print("\nOrdered comparison")
ex_ord = GoldenExample(id="t_order", agent="general", question="q",
                        expected=["Miami", "Denver", "Pittsburgh", "Austin"],
                        order_matters=True, value_type="string")

check("correct order -> PASS",
      verdict(ex_ord, "Ranked by population: 1. Miami 2. Denver 3. Pittsburgh 4. Austin.") == "PASS")
check("right set, wrong order -> FAIL",
      verdict(ex_ord, "Ranked: 1. Denver 2. Miami 3. Austin 4. Pittsburgh.") == "FAIL")
check("missing item -> FAIL",
      verdict(ex_ord, "Miami, then Denver, then Pittsburgh.") == "FAIL")
check("(documented limitation) earlier out-of-order mention before the real "
      "list can still cause a false FAIL",
      verdict(ex_ord, "Denver and Miami are both large. Ranked: "
                       "1. Miami 2. Denver 3. Pittsburgh 4. Austin.") == "FAIL")

# ── Numeric, exact (tolerance=0 — counts) ───────────────────────────────────
print("\nNumeric, exact match required (counts)")
ex_count = GoldenExample(id="t_count", agent="general", question="q",
                          expected=[2], value_type="number", tolerance=0)

check("exact digit match -> PASS", verdict(ex_count, "You have 2 houses in Pittsburgh.") == "PASS")
check("spelled-out number matches too -> PASS",
      verdict(ex_count, "You have two houses in Pittsburgh.") == "PASS")
check("off by one -> FAIL", verdict(ex_count, "You have 3 houses in Pittsburgh.") == "FAIL")

# ── Numeric, with tolerance + distractor (NULL-handling case) ───────────────
print("\nNumeric with tolerance + distractor (NULL-handling)")
ex_null = GoldenExample(id="t_null", agent="general", question="q",
                         expected=[55.0], value_type="number",
                         distractors=[27.5])

check("correct NULL-safe average -> PASS",
      verdict(ex_null, "The average walk score in Austin is 55.0.") == "PASS")
check("small rounding within tolerance -> PASS",
      verdict(ex_null, "The average walk score in Austin is approximately 55.") == "PASS")
check("NULL treated as zero (distractor) -> FAIL even though not the only number",
      verdict(ex_null, "The average walk score in Austin is 27.5 "
                        "(averaging both houses, treating the missing "
                        "score as zero).") == "FAIL")

# ── Currency, with tolerance + distractor (filter-skipped case) ─────────────
print("\nCurrency with distractor (arm's-length filter)")
ex_cur = GoldenExample(id="t_cur", agent="general", question="q",
                        expected=[340000.0], value_type="currency",
                        distractors=[170000.5])

check("correct filtered average -> PASS",
      verdict(ex_cur, "The average arm's-length sold price is $340,000.") == "PASS")
check("unfiltered average (distractor) -> FAIL",
      verdict(ex_cur, "The average sold price is $170,000.50.") == "FAIL")

# ── Edge cases ────────────────────────────────────────────────────────────
print("\nEdge cases")
check("empty reply -> FAIL", verdict(ex_count, "") == "FAIL")
check("whitespace-only reply -> FAIL", verdict(ex_count, "   \n  ") == "FAIL")
check("a large unrelated number (e.g. a FIPS code) doesn't false-match a small target",
      verdict(GoldenExample(id="t_fips", agent="general", question="q",
                             expected=[4200], value_type="number", tolerance=0),
              "The tract FIPS code is 42003140300.") == "FAIL")

print(f"\n{_passed} passed, {_failed} failed")
if _failed:
    sys.exit(1)
