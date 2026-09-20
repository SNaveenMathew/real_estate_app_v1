"""
End-to-end integration tests against a LIVE running server.

Usage:
    python tests/e2e_live.py                          # default http://localhost:8000
    python tests/e2e_live.py --base http://localhost:8765

The server must be running before this script is invoked:
    uvicorn main:app --port 8000

Exit code 0 = all pass, 1 = failures.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import textwrap
import time

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://localhost:8000")
args, _ = parser.parse_known_args()
BASE = args.base.rstrip("/")

_GREEN = "\033[32m"
_RED   = "\033[31m"
_RESET = "\033[0m"
PASS = f"{_GREEN}PASS{_RESET}"
FAIL = f"{_RED}FAIL{_RESET}"

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    _results.append((name, ok, detail))
    return ok


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _csv_bytes() -> bytes:
    return textwrap.dedent("""\
        GEOID20,STATEFP,COUNTYFP,TRACTCE,CBSA_Name,NatWalkInd,D3B,lead_pct
        420030401001,42,003,040100,Pittsburgh PA,12.5,150,3.2
        420030401002,42,003,040100,Pittsburgh PA,11.0,145,2.8
        420030501001,42,003,050100,Pittsburgh PA,9.3,130,4.1
        420030501002,42,003,050100,Pittsburgh PA,8.7,128,3.9
        421010001001,42,101,000100,Philadelphia PA,15.1,200,1.2
        421010001002,42,101,000100,Philadelphia PA,14.8,198,1.0
        421010002001,42,101,000200,Philadelphia PA,13.2,180,2.5
        390351001001,39,035,100100,Cleveland OH,7.4,90,5.0
    """).encode()


# ─── 1. Static pages ──────────────────────────────────────────────────────────
section("1. Static pages")

r = requests.get(f"{BASE}/")
check("GET / returns 200", r.status_code == 200)
check("/ contains map HTML", "leaflet" in r.text.lower() or "map" in r.text.lower())

r = requests.get(f"{BASE}/data")
check("GET /data returns 200", r.status_code == 200)
check("/data contains relevant content", "data" in r.text.lower())
check("/data has cache-busting ?v= on assets", "?v=" in r.text)

r = requests.get(f"{BASE}/static/data.js")
check("GET /static/data.js returns 200", r.status_code == 200)

r = requests.get(f"{BASE}/static/data.css")
check("GET /static/data.css returns 200", r.status_code == 200)

# ─── 2. Catalog API — built-in sources ───────────────────────────────────────
section("2. Catalog API — built-in sources visible")

r = requests.get(f"{BASE}/api/onboarding/catalog")
check("GET /api/onboarding/catalog returns 200", r.status_code == 200,
      r.text[:80] if r.status_code != 200 else "")
cat = r.json()
check("catalog has 'tables' key", "tables" in cat)
tables_before = {t["name"]: t for t in cat.get("tables", [])}
check("built-in 'houses' table present", "houses" in tables_before)
check("built-in 'nri_tracts' table present", "nri_tracts" in tables_before)
check("built-in 'census_tracts' table present", "census_tracts" in tables_before)
check("catalog has 'relationships' key", "relationships" in cat)
rels_before = cat.get("relationships", [])
check("at least one built-in relationship", len(rels_before) > 0)
check("catalog has 'domains' key", "domains" in cat)

# ─── 3. Dataset upload ────────────────────────────────────────────────────────
section("3. Dataset upload")

r = requests.post(
    f"{BASE}/api/onboarding/datasets",
    files={"file": ("walkability.csv", io.BytesIO(_csv_bytes()), "text/csv")},
)
check("POST /datasets (upload CSV) returns 200", r.status_code == 200,
      r.text[:120] if r.status_code != 200 else "")
ds = r.json()
dataset_id = ds.get("dataset_id", "")
check("response has dataset_id", bool(dataset_id))
# Upload runs type-inference immediately so status is 'draft', not 'staged'
check("status is 'draft' or 'staged' after upload",
      ds.get("status") in ("staged", "draft"), f"got: {ds.get('status')}")
check("row_count > 0", (ds.get("row_count") or 0) > 0)
check("columns list non-empty", len(ds.get("columns") or []) > 0)

# ─── 4. Datasets list endpoint ────────────────────────────────────────────────
section("4. Datasets list endpoint")

r = requests.get(f"{BASE}/api/onboarding/datasets")
check("GET /datasets returns 200", r.status_code == 200)
body = r.json()
# Server returns an envelope: {"datasets": [...], "formats": [...], "domains": [...], ...}
check("response envelope has 'datasets' key", "datasets" in body)
check("envelope has 'formats' and 'domains'",
      "formats" in body and "domains" in body)
check("uploaded dataset appears in list",
      any(d.get("dataset_id") == dataset_id for d in body.get("datasets", [])))

# ─── 5. Update description ────────────────────────────────────────────────────
section("5. Update description")

desc_payload = {
    "description": "EPA National Walkability Index by census block group",
    "grain": "one row per census block group",
    "domain": "mobility",
    "columns": [
        {"name": "natwalkind", "description": "National Walkability Index (1-20)",
         "synonyms": ["walkability", "walk score"], "unit": "index"},
        {"name": "d3b", "description": "Street intersection density",
         "synonyms": ["intersection density"], "unit": "intersections/sq mi"},
        {"name": "lead_pct", "description": "Estimated % of lead service lines",
         "synonyms": ["lead service line"], "unit": "percent"},
    ],
}
r = requests.put(f"{BASE}/api/onboarding/datasets/{dataset_id}", json=desc_payload)
check("PUT /datasets/{id} returns 200", r.status_code == 200,
      r.text[:120] if r.status_code != 200 else "")
ds2 = r.json()
check("description persisted", "walkability" in (ds2.get("description") or "").lower())

# ─── 6. Draft wording (optional; degrades gracefully) ────────────────────────
section("6. Draft wording (optional LLM)")

r = requests.post(f"{BASE}/api/onboarding/datasets/{dataset_id}/draft")
check("draft endpoint returns 200", r.status_code == 200,
      r.text[:120] if r.status_code != 200 else "")
draft = r.json() if r.status_code == 200 else {}
check("draft response has dataset_id", draft.get("dataset_id") == dataset_id)

# ─── 7. Analyze — propose table and relationships ─────────────────────────────
section("7. Analyze — propose table and relationships")

r = requests.post(f"{BASE}/api/onboarding/datasets/{dataset_id}/analyze")
check("POST /datasets/{id}/analyze returns 200", r.status_code == 200,
      r.text[:120] if r.status_code != 200 else "")
ds3 = r.json()
proposals = ds3.get("proposals") or []
check("at least one proposal produced", len(proposals) > 0)
table_proposals = [p for p in proposals if p.get("kind") == "table"]
rel_proposals   = [p for p in proposals if p.get("kind") == "relationship"]
check("table proposal present", len(table_proposals) > 0,
      f"got kinds: {[p.get('kind') for p in proposals]}")
check("relationship proposal(s) present", len(rel_proposals) > 0)
nri_links = [p for p in rel_proposals
             if "nri_tracts" in json.dumps(p.get("payload", {}))]
check("link to nri_tracts discovered via GEOID derivation", len(nri_links) > 0,
      f"nri_links={len(nri_links)}")

table_pid = table_proposals[0]["proposal_id"] if table_proposals else None
rel_pids  = [p["proposal_id"] for p in rel_proposals]

# ─── 8. Catalog overlay before approval ──────────────────────────────────────
section("8. Catalog overlay (draft view before approval)")

r = requests.get(f"{BASE}/api/onboarding/catalog?dataset_id={dataset_id}")
check("GET /catalog?dataset_id= returns 200", r.status_code == 200)
overlay = r.json()
check("overlay has tables key", "tables" in overlay)
check("overlay tables non-empty", len(overlay.get("tables", [])) > 0)

# ─── 9. Approve table proposal ───────────────────────────────────────────────
section("9. Approve table proposal")

if table_pid:
    r = requests.post(f"{BASE}/api/onboarding/proposals/{table_pid}/decision",
                      json={"decision": "approve"})
    check("POST /proposals/{id}/decision (table) returns 200",
          r.status_code == 200, r.text[:120] if r.status_code != 200 else "")
    result = r.json()
    check("result indicates approved", "approved" in str(result).lower())
else:
    check("table proposal id available", False, "no table proposal")

# ─── 10. Approve relationship proposals ──────────────────────────────────────
section("10. Approve relationship proposals")

approved_rels = 0
approved_rel_keys: list[str] = []
for pid in rel_pids:
    r = requests.post(f"{BASE}/api/onboarding/proposals/{pid}/decision",
                      json={"decision": "approve"})
    if r.status_code == 200:
        approved_rels += 1
        matching = next((p for p in rel_proposals if p["proposal_id"] == pid), None)
        if matching:
            rk = matching.get("payload", {}).get("rel_key", "")
            if rk:
                approved_rel_keys.append(rk)
check("at least one relationship approved",
      approved_rels > 0, f"approved {approved_rels}/{len(rel_pids)}")

# ─── 11. New dataset live in catalog ─────────────────────────────────────────
section("11. New dataset live in catalog immediately")

time.sleep(0.5)
r = requests.get(f"{BASE}/api/onboarding/catalog")
cat2 = r.json()
tables_after = {t["name"]: t for t in cat2.get("tables", [])}
new_names = set(tables_after) - set(tables_before)
check("new table now live in catalog", len(new_names) > 0, f"new: {new_names}")

rels_after = cat2.get("relationships", [])
check("approved relationship(s) live in catalog",
      len(rels_after) > len(rels_before),
      f"was {len(rels_before)}, now {len(rels_after)}")

# ─── 12. Dataset detail reflects published status ────────────────────────────
section("12. Dataset detail reflects published status")

r = requests.get(f"{BASE}/api/onboarding/datasets/{dataset_id}")
check("GET /datasets/{id} returns 200", r.status_code == 200)
detail = r.json()
check("dataset status is 'published'", detail.get("status") == "published",
      f"status={detail.get('status')}")
check("approved table proposal in detail",
      any(p.get("status") == "approved" and p.get("kind") == "table"
          for p in detail.get("proposals", [])))

# ─── 13. Published table is agent-visible ────────────────────────────────────
section("13. Published table is agent-visible in catalog")

check("new table has agent_visible=True",
      any(tables_after[n].get("agent_visible") is True for n in new_names if n in tables_after),
      f"new_names={new_names}")

# ─── 14. Revoke a relationship ───────────────────────────────────────────────
section("14. Revoke a relationship")

# Prefer rel_key captured at approval time; fall back to scanning the live catalog
target_rel_key = approved_rel_keys[0] if approved_rel_keys else None
if not target_rel_key:
    for rel in rels_after:
        if rel.get("origin") != "builtin":
            target_rel_key = rel.get("rel_key")
            break

if target_rel_key:
    r = requests.post(f"{BASE}/api/onboarding/relationships/revoke",
                      json={"rel_key": target_rel_key})
    check("POST /relationships/revoke returns 200", r.status_code == 200,
          r.text[:120] if r.status_code != 200 else "")
    time.sleep(0.3)
    r2 = requests.get(f"{BASE}/api/onboarding/catalog")
    rels_post_revoke = r2.json().get("relationships", [])
    check("revoked relationship gone from catalog",
          not any(rel.get("rel_key") == target_rel_key for rel in rels_post_revoke),
          f"rel_key={target_rel_key}")
else:
    check("non-builtin relationship available to revoke", False,
          "no non-builtin rel found; were any approved?")

# ─── 15. Retire dataset ──────────────────────────────────────────────────────
section("15. Retire dataset")

r = requests.post(f"{BASE}/api/onboarding/datasets/{dataset_id}/retire")
check("POST /datasets/{id}/retire returns 200", r.status_code == 200,
      r.text[:120] if r.status_code != 200 else "")
time.sleep(0.3)
r2 = requests.get(f"{BASE}/api/onboarding/catalog")
remaining = {t["name"] for t in r2.json().get("tables", [])}
check("retired table removed from live catalog",
      not (new_names & remaining), f"still present: {new_names & remaining}")

# ─── 16. Upload validation ───────────────────────────────────────────────────
section("16. Upload validation — hostile / unsupported inputs")

r = requests.post(f"{BASE}/api/onboarding/datasets",
                  files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")})
check("empty CSV upload rejected (4xx)", 400 <= r.status_code < 500,
      f"got {r.status_code}")

# Server returns 415 Unsupported Media Type for unknown extensions
r = requests.post(f"{BASE}/api/onboarding/datasets",
                  files={"file": ("bad.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
check("unsupported extension upload rejected (4xx)", 400 <= r.status_code < 500,
      f"got {r.status_code}")

evil_csv = b'=CMD("calc"),value\n1,2\n3,4\n'
r = requests.post(f"{BASE}/api/onboarding/datasets",
                  files={"file": ("evil.csv", io.BytesIO(evil_csv), "text/csv")})
if r.status_code == 200:
    evil_ds = r.json()
    cols = [c["name"] for c in (evil_ds.get("columns") or [])]
    check("formula-injection header sanitised (no '=' in col name)",
          not any("=" in c for c in cols), f"cols: {cols}")
    requests.post(f"{BASE}/api/onboarding/datasets/{evil_ds['dataset_id']}/retire")
else:
    check("formula-injection upload rejected (also valid)", 400 <= r.status_code < 500,
          f"got {r.status_code}")

# ─── 17. LLM status endpoint ─────────────────────────────────────────────────
section("17. LLM / catalog model status endpoint")

r = requests.get(f"{BASE}/api/onboarding/llm/status")
check("GET /api/onboarding/llm/status returns 200", r.status_code == 200,
      r.text[:80] if r.status_code != 200 else "")
status = r.json()
check("status has 'enabled' field", "enabled" in status)
check("status has 'tiers' field", "tiers" in status, f"keys: {list(status.keys())}")

# ─── 18. Core app endpoints ──────────────────────────────────────────────────
section("18. Core app endpoints")

r = requests.get(f"{BASE}/api/houses")
check("GET /api/houses returns 200", r.status_code == 200,
      f"got {r.status_code}: {r.text[:60]}" if r.status_code != 200 else "")

r = requests.get(f"{BASE}/api/layers/nri?min_lat=40.0&max_lat=41.0&min_lon=-80.5&max_lon=-79.5")
nri_ok = r.status_code == 200
if not nri_ok:
    r = requests.get(f"{BASE}/api/nri-layer?min_lat=40.0&max_lat=41.0&min_lon=-80.5&max_lon=-79.5")
    nri_ok = r.status_code == 200
check("NRI layer returns 200", nri_ok, f"got {r.status_code}")

r = requests.get(f"{BASE}/api/layers/crime?min_lat=40.0&max_lat=41.0&min_lon=-80.5&max_lon=-79.5")
crime_ok = r.status_code == 200
if not crime_ok:
    r = requests.get(f"{BASE}/api/crime-layer?min_lat=40.0&max_lat=41.0&min_lon=-80.5&max_lon=-79.5")
    crime_ok = r.status_code == 200
check("Crime layer returns 200", crime_ok, f"got {r.status_code}")

r = requests.get(f"{BASE}/metrics")
check("GET /metrics (Prometheus) returns 200", r.status_code == 200)

# ─── Summary ─────────────────────────────────────────────────────────────────
section("SUMMARY")
passed = sum(1 for _, ok, _ in _results if ok)
failed = sum(1 for _, ok, _ in _results if not ok)
print(f"\n  Total: {len(_results)}  |  PASS: {passed}  |  FAIL: {failed}\n")
if failed:
    print("  Failed checks:")
    for name, ok, detail in _results:
        if not ok:
            print(f"    - {name}" + (f": {detail}" if detail else ""))
    print()

sys.exit(0 if failed == 0 else 1)
