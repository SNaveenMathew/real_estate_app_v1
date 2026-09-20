"""
debug_flood_query.py  — Run this to pinpoint exactly where the flood risk
query breaks down. Executes each CTE independently and prints what it finds.

Usage:
    python debug_flood_query.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import db.duckdb_store as store

SEP = "─" * 70

def q(sql, label=""):
    try:
        df = store.query(sql)
        return df
    except Exception as e:
        print(f"  SQL ERROR in {label}: {e}")
        return None


def run():
    print(SEP)
    print("FLOOD RISK QUERY DEBUGGER")
    print(SEP)

    # ── 1. Table sizes ────────────────────────────────────────────────────
    print("\n[1] Table row counts")
    for tbl in ["census_msa", "cbsa_counties", "nri_tracts"]:
        df = q(f"SELECT COUNT(*) as n FROM {tbl}", tbl)
        if df is not None:
            print(f"  {tbl:<20} {int(df.iloc[0]['n']):>10,} rows")

    # ── 2. census_msa — population column ────────────────────────────────
    print("\n[2] census_msa — top 5 by population (raw values)")
    df = q("""
        SELECT msa_code, name, population,
               typeof(population) AS pop_type
        FROM census_msa
        WHERE population IS NOT NULL
        ORDER BY population DESC
        LIMIT 5
    """, "census_msa top5")
    if df is not None:
        print(df.to_string(index=False))

    print("\n[2b] census_msa — top 5 with CAST to BIGINT (used in query)")
    df = q("""
        SELECT msa_code, name,
               CAST(population AS BIGINT) AS pop_bigint
        FROM census_msa
        WHERE population IS NOT NULL
        ORDER BY CAST(population AS BIGINT) DESC
        LIMIT 5
    """, "census_msa cast bigint")
    if df is not None:
        print(df.to_string(index=False))

    print("\n[2c] census_msa — X-coded rows in top 50 by population")
    df = q("""
        WITH top50 AS (
            SELECT msa_code, name, CAST(population AS BIGINT) AS pop
            FROM census_msa
            WHERE population IS NOT NULL
            ORDER BY CAST(population AS BIGINT) DESC
            LIMIT 50
        )
        SELECT msa_code, name, pop
        FROM top50
        WHERE msa_code LIKE 'X%'
        ORDER BY pop DESC
    """, "x-codes in top 50")
    if df is not None:
        n = len(df)
        print(f"  {n} X-coded MSA(s) in top 50 by population"
              + (" — these WON'T join to NRI" if n else ""))
        if n:
            print(df.to_string(index=False))

    # ── 3. cbsa_counties ─────────────────────────────────────────────────
    print("\n[3] cbsa_counties — sample key values")
    df = q("""
        SELECT state_fips, county_fips,
               state_fips || county_fips AS joined_key,
               cbsa_code, cbsa_title
        FROM cbsa_counties
        LIMIT 5
    """, "cbsa sample")
    if df is not None:
        print(df.to_string(index=False))

    # ── 4. nri_tracts ─────────────────────────────────────────────────────
    print("\n[4] nri_tracts — sample county_fips values")
    df = q("""
        SELECT county_fips, COUNT(*) as tract_count,
               AVG(rfld_risks) as avg_flood
        FROM nri_tracts
        GROUP BY county_fips
        LIMIT 5
    """, "nri sample")
    if df is not None:
        print(df.to_string(index=False))

    # ── 5. cbsa ↔ nri join ────────────────────────────────────────────────
    print("\n[5] cbsa_counties ↔ nri_tracts join (county FIPS key)")
    df = q("""
        SELECT COUNT(DISTINCT cb.cbsa_code) AS matching_cbsas,
               COUNT(DISTINCT n.tract_fips) AS matching_tracts
        FROM cbsa_counties cb
        JOIN nri_tracts n
          ON (cb.state_fips || cb.county_fips) = n.county_fips
        WHERE n.rfld_risks IS NOT NULL
    """, "cbsa-nri join")
    if df is not None:
        print(df.to_string(index=False))
        cbsas_with_nri = int(df.iloc[0]["matching_cbsas"])
        if cbsas_with_nri == 0:
            print("  !! ZERO MATCHES — this is the problem.")
            print("     Check that state_fips + county_fips lengths match county_fips in nri_tracts")
            df2 = q("""
                SELECT
                  LENGTH(state_fips || county_fips) AS cbsa_key_len,
                  (SELECT LENGTH(county_fips) FROM nri_tracts LIMIT 1) AS nri_key_len
                FROM cbsa_counties LIMIT 1
            """, "key length check")
            if df2 is not None:
                print(df2.to_string(index=False))

    # ── 6. census_msa ↔ cbsa_counties join ───────────────────────────────
    print("\n[6] census_msa.msa_code ↔ cbsa_counties.cbsa_code join")
    df = q("""
        SELECT COUNT(*) AS total_msa_rows,
               COUNT(cb.cbsa_code) AS rows_with_cbsa_match
        FROM census_msa m
        LEFT JOIN (SELECT DISTINCT cbsa_code FROM cbsa_counties) cb
          ON m.msa_code = cb.cbsa_code
        WHERE m.population IS NOT NULL
    """, "msa-cbsa join")
    if df is not None:
        total  = int(df.iloc[0]["total_msa_rows"])
        joined = int(df.iloc[0]["rows_with_cbsa_match"])
        pct    = 100 * joined // max(total, 1)
        print(f"  {joined}/{total} MSA rows have a matching cbsa_code ({pct}%)")
        if joined == 0:
            print("  !! ZERO MATCHES — run: python setup_data.py --only repair")

    # ── 7. Top 50 specifically ────────────────────────────────────────────
    print("\n[7] Top 50 MSAs by population — how many link to NRI via cbsa?")
    df = q("""
        WITH top50 AS (
            SELECT msa_code, name, CAST(population AS BIGINT) AS pop
            FROM census_msa
            WHERE population IS NOT NULL
            ORDER BY CAST(population AS BIGINT) DESC
            LIMIT 50
        ),
        with_risk AS (
            SELECT t.msa_code, t.name, t.pop,
                   AVG(n.rfld_risks) AS avg_flood
            FROM top50 t
            JOIN cbsa_counties cb ON t.msa_code = cb.cbsa_code
            JOIN nri_tracts n
              ON (cb.state_fips || cb.county_fips) = n.county_fips
            WHERE n.rfld_risks IS NOT NULL
            GROUP BY t.msa_code, t.name, t.pop
        )
        SELECT COUNT(*) AS msas_with_full_chain FROM with_risk
    """, "full 3-way join")
    if df is not None:
        n = int(df.iloc[0]["msas_with_full_chain"])
        print(f"  MSAs in top 50 that complete the full chain: {n}")
        if n == 0:
            print("  !! ZERO — the three-way join produces no rows")
            print("     One of the two joins above is failing")

    # ── 8. Full query result ──────────────────────────────────────────────
    print("\n[8] Final query — bottom 10 flood risk among top 50 MSAs")
    df = q("""
        WITH top_msas AS (
            SELECT msa_code, name,
                   CAST(population AS BIGINT) AS population
            FROM census_msa
            WHERE population IS NOT NULL
            ORDER BY CAST(population AS BIGINT) DESC
            LIMIT 50
        ),
        msa_risk AS (
            SELECT cb.cbsa_code,
                   AVG(n.rfld_risks) AS avg_flood,
                   COUNT(DISTINCT n.tract_fips) AS tract_count
            FROM cbsa_counties cb
            JOIN nri_tracts n
              ON (cb.state_fips || cb.county_fips) = n.county_fips
            WHERE n.rfld_risks IS NOT NULL
            GROUP BY cb.cbsa_code
        )
        SELECT m.name AS msa_name,
               m.population,
               ROUND(r.avg_flood, 2) AS avg_rfld_risks,
               r.tract_count
        FROM top_msas m
        JOIN msa_risk r ON m.msa_code = r.cbsa_code
        ORDER BY r.avg_flood ASC
        LIMIT 10
    """, "final query")
    if df is not None:
        if df.empty:
            print("  !! EMPTY — see steps above for where the chain breaks")
        else:
            print(df.to_string(index=False))

    print(f"\n{SEP}")
    print("SUMMARY: check any step that shows 0 rows or !! above")
    print(SEP)


if __name__ == "__main__":
    run()
