"""
diagnose_msa.py — Run this against your local DuckDB to see exactly why
X-coded MSA names don't match cbsa_counties entries.

Usage:
    python diagnose_msa.py           # show all X-coded rows with best candidates
    python diagnose_msa.py --apply   # apply suggested fixes automatically
"""
import sys
import re
import unicodedata
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db.duckdb_store as store


# ── Same normalizer as data_loader.py ────────────────────────────────────────

def normalize(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r'\s+(Metro|Micro)\s+Area$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'-{2,}', '-', name)
    name = re.sub(r'\s+', ' ', name).strip()
    decomposed = unicodedata.normalize('NFD', name)
    return ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn').lower()


def first_city_state(norm: str):
    m = re.match(r'^(.*),\s*([a-z]{2}(?:-[a-z]{2})*)$', norm)
    if not m:
        return norm, ''
    first_city  = m.group(1).split('-')[0].strip()
    first_state = m.group(2).split('-')[0]
    return first_city, first_state


def all_words(norm: str) -> set:
    """All significant words (>2 chars) in a normalized name, for fuzzy scoring."""
    return {w for w in re.split(r'[\s,\-]+', norm) if len(w) > 2}


def similarity_score(norm_a: str, norm_b: str) -> float:
    """Jaccard similarity on word sets — good enough for city name fuzzy matching."""
    wa, wb = all_words(norm_a), all_words(norm_b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_cbsa() -> dict:
    """Load all CBSA titles → codes from the database."""
    df = store.query("""
        SELECT DISTINCT cbsa_code, cbsa_title
        FROM cbsa_counties
        WHERE cbsa_title IS NOT NULL
        ORDER BY cbsa_code
    """)
    return {row['cbsa_code']: row['cbsa_title']
            for _, row in df.iterrows()}


def find_best_candidate(census_name: str, cbsa_by_code: dict) -> tuple:
    """
    Find the best matching CBSA title for a census MSA name.
    Returns (cbsa_code, cbsa_title, match_strategy, score).
    """
    norm_census = normalize(census_name)
    city_c, state_c = first_city_state(norm_census)

    best_code  = None
    best_title = None
    best_strat = None
    best_score = 0.0

    for code, title in cbsa_by_code.items():
        norm_cbsa = normalize(title)
        city_cb, state_cb = first_city_state(norm_cbsa)

        # Exact match (shouldn't happen for X-codes, but check anyway)
        if norm_census == norm_cbsa:
            return code, title, 'exact', 1.0

        # City+state match
        if city_c == city_cb and state_c == state_cb:
            return code, title, 'city+state', 0.95

        # Same state + Jaccard similarity > 0.5
        if state_c and state_c == state_cb:
            score = similarity_score(norm_census, norm_cbsa)
            if score > best_score:
                best_score = score
                best_code  = code
                best_title = title
                best_strat = f'fuzzy({score:.2f})'

    if best_score >= 0.4:
        return best_code, best_title, best_strat, best_score
    return None, None, 'no_match', 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main(apply: bool = False):
    print("Loading X-coded MSA rows...")
    x_rows = store.query(
        "SELECT msa_code, name, population FROM census_msa WHERE msa_code LIKE 'X%' ORDER BY name"
    )
    if x_rows.empty:
        print("✓ No X-coded rows found — all MSAs are matched!")
        return

    print(f"Found {len(x_rows)} X-coded rows\n")

    cbsa_by_code = load_cbsa()
    print(f"Loaded {len(cbsa_by_code)} distinct CBSA entries from cbsa_counties\n")

    suggestions: list[dict] = []
    no_match:    list[str]  = []

    print(f"{'Census name':<55} {'Strategy':<15} {'Best CBSA candidate'}")
    print("─" * 130)

    for _, row in x_rows.iterrows():
        name    = row['name']
        old_key = row['msa_code']
        code, title, strat, score = find_best_candidate(name, cbsa_by_code)

        if code:
            print(f"  {name:<55} {strat:<15} [{code}] {title}")
            suggestions.append({
                'old_key': old_key,
                'name':    name,
                'new_code': code,
                'cbsa_title': title,
                'strategy': strat,
                'score': score,
            })
        else:
            print(f"  {name:<55} {'NO MATCH':<15} (not in 2023 CBSA delineation file)")
            no_match.append(name)

    print()
    print(f"✓ Can fix:    {len(suggestions)}")
    print(f"✗ No match:   {len(no_match)} (these MSAs don't exist in your CBSA file vintage)")

    if no_match:
        print("\nMSAs with no CBSA match (likely abolished/renamed between 2020 and 2023):")
        for n in no_match:
            print(f"  • {n}")

    if apply and suggestions:
        print(f"\nApplying {len(suggestions)} fixes...")
        conn = store.get_conn()
        applied = 0
        for s in suggestions:
            conn.execute(
                "UPDATE census_msa SET msa_code = ? WHERE msa_code = ?",
                [s['new_code'], s['old_key']]
            )
            applied += 1
        print(f"✓ Applied {applied} fixes")

        remaining = store.query(
            "SELECT COUNT(*) as n FROM census_msa WHERE msa_code LIKE 'X%'"
        ).iloc[0]['n']
        print(f"  Remaining X-codes: {int(remaining)}")

    elif suggestions and not apply:
        print("\nRun with --apply to automatically apply these fixes:")
        print("  python diagnose_msa.py --apply")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true',
                        help='Apply the suggested msa_code fixes to the database')
    args = parser.parse_args()
    main(apply=args.apply)
