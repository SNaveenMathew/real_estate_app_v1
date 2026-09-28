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

# ─── 2. Data-source API — public links and refresh contract ──────────────────
section("2. Data-source API — public links and refresh contract")

r = requests.get(f"{BASE}/api/data-sources")
check("GET /api/data-sources returns 200", r.status_code == 200,
      r.text[:120] if r.status_code != 200 else "")
sources = r.json().get("sources", []) if r.status_code == 200 else []
source_keys = {s.get("key") for s in sources}
check("data-source registry includes core sources",
      {"redfin", "nri", "census_tracts", "sold", "bike"} <= source_keys,
      f"keys={sorted(k for k in source_keys if k)}")
check("registered sources expose public links",
      bool(sources) and all(s.get("source_url") for s in sources))

# This checks request validation without changing any source files or tables.
r = requests.post(
    f"{BASE}/api/data-sources/redfin/refresh",
    files={"file": ("invalid.txt", io.BytesIO(b"not a Redfin export"), "text/plain")},
)
check("refresh rejects unsupported file types", 400 <= r.status_code < 500,
      f"got {r.status_code}")

# ─── 2b. Catalog API — built-in sources ──────────────────────────────────────
section("2b. Catalog API — built-in sources visible")

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
            # Proposal payload uses 'key' or 'rel_key' depending on the version
            payload = matching.get("payload", {})
            rk = payload.get("key") or payload.get("rel_key") or ""
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

# Prefer rel_key captured at approval time; fall back to a fresh catalog scan.
# The catalog uses 'key' as the field name; the revoke endpoint calls it 'rel_key' — same value.
target_rel_key = approved_rel_keys[0] if approved_rel_keys else None
if not target_rel_key:
    # Re-fetch the catalog to get the current relationships with origin info
    _fresh_rels = requests.get(f"{BASE}/api/onboarding/catalog").json().get("relationships", [])
    for rel in _fresh_rels:
        if rel.get("origin") != "builtin":
            target_rel_key = rel.get("key") or rel.get("rel_key")
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
          not any((rel.get("key") or rel.get("rel_key")) == target_rel_key for rel in rels_post_revoke),
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

# Layer endpoints use west/south/east/north bbox params
_bbox = {"west": -80.5, "south": 40.0, "east": -79.5, "north": 41.0}
r = requests.get(f"{BASE}/api/layers/nri_tracts", params=_bbox)
check("NRI layer returns 200", r.status_code == 200, f"got {r.status_code}")

r = requests.get(f"{BASE}/api/layers/crime_incidents", params=_bbox)
check("Crime layer returns 200", r.status_code == 200, f"got {r.status_code}")

r = requests.get(f"{BASE}/metrics")
check("GET /metrics (Prometheus) returns 200", r.status_code == 200)

# ─── 19. Crime heat layer — year filter ──────────────────────────────────────
section("19. Crime heat layer — year filter")

_layers_r = requests.get(f"{BASE}/api/layers")
check("GET /api/layers returns 200 for year_options check",
      _layers_r.status_code == 200, f"got {_layers_r.status_code}")

if _layers_r.status_code == 200:
    _crime_spec = next(
        (l for l in _layers_r.json().get("layers", []) if l["name"] == "crime_incidents"),
        None,
    )
    check("crime_incidents layer present in layer list", _crime_spec is not None)
    _year_opts = (_crime_spec or {}).get("year_options", [])
    check("crime_incidents spec includes year_options list",
          isinstance(_year_opts, list) and len(_year_opts) > 0,
          f"year_options={_year_opts!r}")
    check("year_options is sorted ascending",
          _year_opts == sorted(_year_opts),
          f"year_options={_year_opts!r}")
    check("year_options contains only integers",
          all(isinstance(y, int) for y in _year_opts),
          f"types={[type(y).__name__ for y in _year_opts[:5]]!r}")
    # Pick a middle year that should be well-populated
    _pick_year = 2019 if 2019 in _year_opts else (_year_opts[len(_year_opts) // 2] if _year_opts else None)
else:
    _pick_year = 2019

# Filtered call — should succeed and echo the requested year
_r_year = requests.get(f"{BASE}/api/layers/crime_incidents",
                       params={**_bbox, "year": _pick_year, "grid_deg": 0.01})
check(f"GET /api/layers/crime_incidents?year={_pick_year} returns 200",
      _r_year.status_code == 200,
      f"got {_r_year.status_code}: {_r_year.text[:80]}" if _r_year.status_code != 200 else "")

if _r_year.status_code == 200:
    _d_year = _r_year.json()
    check("year-filtered response echoes year field",
          _d_year.get("year") == _pick_year,
          f"year={_d_year.get('year')!r}")
    check("year-filtered response has points list",
          isinstance(_d_year.get("points"), list),
          f"keys={list(_d_year)}")
    check("year-filtered response includes max_year_weight >= max_weight",
          _d_year.get("max_year_weight") is not None and _d_year.get("max_year_weight") >= _d_year.get("max_weight", 0),
          f"max_year_weight={_d_year.get('max_year_weight')!r}, max_weight={_d_year.get('max_weight')!r}")

# Unfiltered call — should return more incidents than the year-filtered one
_r_all = requests.get(f"{BASE}/api/layers/crime_incidents",
                      params={**_bbox, "grid_deg": 0.01})
if _r_all.status_code == 200 and _r_year.status_code == 200:
    _cnt_all = _r_all.json().get("incident_count", 0)
    _cnt_year = _r_year.json().get("incident_count", 0)
    check("unfiltered call returns more incidents than year-filtered call",
          _cnt_all > _cnt_year,
          f"all={_cnt_all}, year={_cnt_year}")
    check("unfiltered response echoes year=null",
          _r_all.json().get("year") is None,
          f"year={_r_all.json().get('year')!r}")
else:
    check("both filtered and unfiltered calls succeeded", False,
          f"all={_r_all.status_code}, year={_r_year.status_code}")

# Non-existent year — must return empty points, not an error
_r_empty = requests.get(f"{BASE}/api/layers/crime_incidents",
                        params={**_bbox, "year": 1900, "grid_deg": 0.01})
check("?year=1900 (no data) returns 200 with empty points",
      _r_empty.status_code == 200 and _r_empty.json().get("points") == [],
      f"status={_r_empty.status_code}, points={_r_empty.json().get('points')!r}"
      if _r_empty.status_code == 200 else f"status={_r_empty.status_code}")

# Non-integer year — must be rejected by API validation
_r_bad = requests.get(f"{BASE}/api/layers/crime_incidents",
                      params={**_bbox, "year": "notanumber"})
check("?year=notanumber rejected with 422",
      _r_bad.status_code == 422,
      f"got {_r_bad.status_code}")

# Year query on layer without year column — must be rejected with 422
_r_no_year = requests.get(f"{BASE}/api/layers/sold_homes",
                          params={**_bbox, "year": 2020})
check("?year=2020 on sold_homes (no year col) rejected with 422",
      _r_no_year.status_code == 422,
      f"got {_r_no_year.status_code}")



# ─── 20. House Chat — artifact contract ──────────────────────────────────────
section("20. House Chat — artifact contract (issue #5)")

# Fetch any real house_id from the live server
_r_houses = requests.get(f"{BASE}/api/houses")
_house_id = None
if _r_houses.status_code == 200:
    _features = _r_houses.json().get("features", [])
    if _features:
        _house_id = _features[0]["properties"].get("house_id")

check("house list available for chat test", _house_id is not None,
      f"status={_r_houses.status_code}")

if _house_id:
    # Ask for NRI risk — get_nri_risk_data() always emits a chart artifact
    _r_chat = requests.post(
        f"{BASE}/api/house/{_house_id}/chat",
        json={"message": "What is the FEMA risk rating for this home?", "history": []},
    )
    check("POST /api/house/{id}/chat returns 200",
          _r_chat.status_code == 200,
          f"got {_r_chat.status_code}: {_r_chat.text[:80]}" if _r_chat.status_code != 200 else "")

    if _r_chat.status_code == 200:
        _chat_body = _r_chat.json()
        check("house chat response has 'reply' field",
              bool(_chat_body.get("reply")))
        check("house chat response has 'history' list",
              isinstance(_chat_body.get("history"), list))
        check("house chat response has 'artifacts' key",
              "artifacts" in _chat_body,
              f"keys={list(_chat_body)}")
        _arts = _chat_body.get("artifacts", [])
        check("house chat 'artifacts' is a list",
              isinstance(_arts, list),
              f"type={type(_arts).__name__}")
        # NRI query should produce at least one artifact (a chart of top hazards)
        check("house chat NRI query produces at least one artifact",
              len(_arts) >= 1,
              f"got {len(_arts)} artifact(s): {[a.get('type') for a in _arts]}")
        if _arts:
            _first = _arts[0]
            check("artifact has a 'type' field",
                  _first.get("type") in ("chart", "table", "map"),
                  f"type={_first.get('type')!r}")

# ─── 21. General Chat — artifact contract ────────────────────────────────────
section("21. General Chat — artifact contract (issue #5)")

_r_gen = requests.post(
    f"{BASE}/api/chat",
    json={"message": "Show me the average house price by city", "history": []},
)
check("POST /api/chat returns 200",
      _r_gen.status_code == 200,
      f"got {_r_gen.status_code}: {_r_gen.text[:80]}" if _r_gen.status_code != 200 else "")

if _r_gen.status_code == 200:
    _gen_body = _r_gen.json()
    check("general chat response has 'reply' field",
          bool(_gen_body.get("reply")))
    check("general chat response has 'artifacts' key",
          "artifacts" in _gen_body,
          f"keys={list(_gen_body)}")
    _gen_arts = _gen_body.get("artifacts", [])
    check("general chat 'artifacts' is a list",
          isinstance(_gen_arts, list),
          f"type={type(_gen_arts).__name__}")
    check("general chat aggregation query produces at least one artifact",
          len(_gen_arts) >= 1,
          f"got {len(_gen_arts)} artifact(s): {[a.get('type') for a in _gen_arts]}")
    if _gen_arts:
        _gen_first = _gen_arts[0]
        check("general chat artifact has valid 'type'",
              _gen_first.get("type") in ("chart", "table", "map"),
              f"type={_gen_first.get('type')!r}")
        # avg-by-city is a 1-dimension / 1-measure query → should be a chart
        check("avg-price-by-city produces chart or table artifact",
              _gen_first.get("type") in ("chart", "table"),
              f"type={_gen_first.get('type')!r}")

    check("general chat response has 'observability' field",
          "observability" in _gen_body,
          f"keys={list(_gen_body)}")

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
