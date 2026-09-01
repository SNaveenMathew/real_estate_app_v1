# 🏠 Real Estate Intelligence

A local, AI-powered map app for analyzing houses with FEMA National Risk Index data, Census demographics, Redfin listings, severity-weighted crime data, and an LLM chat interface — all running on your machine.

## General Chat: Agent Architecture & Design Philosophy

**The LLM writes text; deterministic code decides what's true and what's allowed.**

For every analytical question, *what's relevant* — which tables, which join
path, which entities, which operation — is decided entirely by deterministic
code reading `db/schema_catalog.py`, before any SQL exists. The LLM's job is
narrower and comes later: turn an already-decided, already-validated plan into
one SQL statement, or turn executed results into prose. Every LLM output along
the way is checked by code against that same plan — never by another LLM call
grading the first one.

```
request text
  -> InputGuardrail                                  [deterministic, Outlines/regex]
       prompt injection screening, jailbreak detection, length/null-byte sanitization
  -> build_query_plan()                              [deterministic]
       concept + operation match, entity resolution, relationship-path search
  -> QueryPlan (tables, joins, filters, operation) — now authoritative
  -> generate_program() / generate_sql()             [LLM]
       writes Python code agent step or SELECT statement from the plan text
  -> CodeAgentGuardrail (AST) + SQL validation        [deterministic, AST/Outlines]
       AST sandboxing, read-only SQL check (SELECT/WITH only), plan-conformance check
       - valid -> execute against DuckDB
       - invalid/empty after 3 attempts -> _compile_sql_from_plan()  [deterministic]
             same QueryPlan, template compiler, no LLM -> execute against DuckDB
  -> final answer                                      [LLM]
       writes prose from the executed evidence only
  -> response_validator.py + OutputGroundingGuardrail  [deterministic]
       flags ungrounded claims, enforces metric bounds (Walk/Bike scores in [0, 100])
```

| Stage | Deterministic or LLM |
|---|---|
| Input sanitization & prompt injection guard | Deterministic (`InputGuardrail`) |
| Routing — tables, joins, entities, operation | Deterministic (`build_query_plan`) |
| Code program / SQL generation | LLM |
| Code AST sandboxing & SQL safety checks | Deterministic (`CodeAgentGuardrail`) |
| SQL plan-conformance check | Deterministic (`_validate_sql_against_plan`) |
| Fallback SQL compilation | Deterministic (`_compile_sql_from_plan`) |
| Final answer prose | LLM |
| Reply-vs-evidence check & score bounding | Deterministic (`OutputGroundingGuardrail`) |

The split holds because every LLM call on this path — SQL generation,
orchestration, and the final answer — runs through the same local, quantized
model at `temperature=0.0` (see **LLM: run `llama-server`** above). Two things
follow from that: retrying that model with an unchanged prompt returns the same
result, so a safety net that just asks it again isn't a safety net; and the
same weights writing an answer can't be trusted to independently grade that
answer, so every "is this correct/safe" check is code, never a second model
call. **See [`AGENT_ARCHITECTURE.md`](AGENT_ARCHITECTURE.md) for the full
pipeline mechanics, every validation rule, and the reasoning behind where each
line is drawn.**

### Data-model retrieval

Two separate retrieval steps feed the pipeline above. `general_chat.data_model_rag`
(`db/vector_store.py`) runs once per turn, before the orchestrator decides what
to do — it searches a dedicated Chroma collection (`data_model_metadata`) using
both the question and the structured plan, so the orchestrator has context
before choosing tool calls. This is a **retrieval accelerator, not the
authoritative schema**: live DuckDB `DESCRIBE` output and row counts remain the
source of truth, and if the embedding service is unavailable, retrieval falls
back to lexical matching rather than failing the turn. Separately,
`schema.build_query_context()`, called inside `generate_sql()`, pulls targeted
live schema and relationship text for just the tables the plan already
selected — a direct catalog read, not a search.

`db/schema_catalog.py` is the single source of truth either way: live
introspection (so column names/types can't drift) plus curated notes for what
introspection can't tell you — which columns are reliably populated, which
joins need a non-obvious expression, which tables need a default filter to be
meaningful.

The planner keeps `universe_limit` separate from `result_limit`: "top 50 MSAs
with the lowest risk" first selects the 50 largest MSAs, then ranks those 50 by
the requested NRI metric. The canonical MSA/NRI join path is
`census_msa -> cbsa_counties -> nri_tracts`, aggregated at MSA grain. On
startup, `_ensure_schema()` removes any `house_rankings`/`nri_msa_risk` views
left by older builds — semantic SQL views are intentionally not part of this
architecture; the SQL agent queries physical tables only.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Browser (Leaflet map + chat UI)                                            │
│  ├─ House markers + sidebar                                                │
│  ├─ Layer control: None / Crime / NRI / Bike Lanes                         │
│  ├─ Bike route planner (start/end inputs with local BikePGH route map)     │
│  ├─ House Chat (per-property LangGraph agent)                              │
│  └─ General Chat (cross-city LangGraph agent)                              │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ HTTP (FastAPI)
┌──────────────────────────────▼─────────────────────────────────────────────┐
│ main.py                                                                    │
│  ├─ /api/houses                 GeoJSON of all houses                      │
│  ├─ /api/layers/crime           Severity-weighted heatmap grid             │
│  ├─ /api/layers/nri             NRI tract choropleth (GeoJSON)             │
│  ├─ /api/layers/bike            BikePGH overlay for current viewport       │
│  ├─ /api/bike/route             Local BikePGH routing endpoint             │
│  ├─ /api/house/{id}/chat        House-specific agent                       │
│  ├─ /api/chat                   General agent                              │
│  ├─ /metrics                    Prometheus metrics endpoint                │
│  └─ /api/house/{id}/photo       Photo upload                               │
└───────────────┬───────────────────────────────────────┬────────────────────┘
                │                                       │
┌───────────────▼──────────────┐      ┌────────────────▼─────────────────┐
│ DuckDB                       │      │ ChromaDB (vector)                │
│  ├─ houses                   │      │  ├─ house_documents              │
│  ├─ nri_tracts               │      │  │  descriptions/photos/notes    │
│  ├─ census_*                 │      │  └─ search + retrieval           │
│  ├─ sold_homes               │      └──────────────────────────────────┘
│  ├─ crime_incidents          │
│  ├─ bike_routes              │
│  ├─ cbsa_*                   │
│  └─ geocode_cache            │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│ LLM + observability                                                  │
│  ├─ llama-server / Ollama (default: llama-server)                    │
│  ├─ Phoenix OTEL tracing + UI (optional local collector)             │
│  └─ Prometheus-compatible metrics                                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- Python 3.11+
- `llama-server` (llama.cpp) installed and running, or Ollama as an alternative
- Phoenix is optional. When `PHOENIX_ENABLED=true`, the app starts a local Phoenix server automatically on `http://127.0.0.1:6006` if that port is available.
- Internet access is needed for Nominatim geocoding when sold-home addresses or bike-route endpoints are not already cached.

---

## Quick Start

### 1. Clone / place files

```
real_estate_app/
├── main.py
├── config.py
├── setup_data.py
├── requirements.txt
├── .env.example
├── agents/
├── db/
├── services/
├── static/
├── observability.py
└── data/
    ├── redfin/       ← drop Redfin CSV exports here
    ├── sold/         ← drop sold-homes CSV files here
    ├── nri/          ← NRI_CensusTracts_Prod.shp or NRI_Table_CensusTracts.csv
    ├── census/       ← DECENNIALPL2020.P1-Data.csv, DECENNIALPL2020.P1-2026-03-25T232220.csv, list1_2023.xlsx
    ├── crime/        ← one folder per city, e.g. crime/chicago/, crime/pittsburgh/
    ├── bike/         ← BikePGH route layers by city
    ├── shapefiles/   ← TIGER/Line tract shapefiles (optional but recommended)
    └── chroma/       ← vector DB working files
```

### 2. Install dependencies

```bash
cd real_estate_app
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 3. Copy environment defaults

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # Mac/Linux
```

The project reads `.env` via `pydantic-settings` and includes defaults for:
- local `llama-server` or Ollama-backed LLMs
- model timeout and token caps
- Phoenix observability endpoint
- app host/port and default map center

All settings are optional. The defaults in `config.py` point at a local
`llama-server` on port `8080`, Phoenix on port `6006`, and the database and
data directories under this repository.

### 4. LLM: run `llama-server` (default)

This repository is configured to use `llama-server` (from the `llama.cpp`
project) as the primary local LLM endpoint. Example:

```bash
llama-server -hf DuoNeural/Gemma-4-26B-A4B-it-GGUF:Q3_K_M \
  -ngl 999 -c 28672 -fa on --cache-type-k q8_0 --cache-type-v q8_0
```

```
llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL -c 24576 \
  --cache-type-k q8_0 --cache-type-v q8_0 -ngl -1 -fa on --port 8080
```

If you prefer Ollama, it remains supported as an alternative. Update the relevant
settings in `.env` / `config.py` and run:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve
```

> The app defaults to `nomic-embed-text` for embeddings, even when using a
> separate local LLM server stack.

### 5. Place your data files

See **Data Sources** section below.

### 6. Load data

```bash
python setup_data.py
```

### 7. Run the app

```bash
python main.py
# or:
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## Data Sources

### Redfin Favorites (required to see houses on the map)

1. Go to Redfin → My Redfin → Favorites (or Saved Searches)
2. Export to CSV
3. The app expects columns including `lat`/`lon`, `address`, `city`, `state`, `zip`, `price`, `beds`, `baths`, `sqft`, `status`
4. If you already have Walk/Bike/Transit scores, include them as `walk_score`, `bike_score`, `transit_score`
5. Drop the CSV into **`data/redfin/`**

You can drop multiple CSV files (e.g. one per city).

### FEMA National Risk Index (highly recommended)

1. Go to: https://www.fema.gov/about/openfema/data-sets/national-risk-index-data
2. Download the tract-level data package
3. The project prefers the shapefile version at **`data/nri/NRI_CensusTracts_Prod.shp`**
4. The CSV fallback is also supported at **`data/nri/NRI_Table_CensusTracts.csv`**

### Census Tract Populations

1. Go to https://data.census.gov
2. Search for table **DECENNIALPL2020.P1**
3. Filter: Geography → Census Tracts → All States → All Tracts
4. Download → Save as **`data/census/DECENNIALPL2020.P1-Data.csv`**

### MSA Populations

1. Same table **DECENNIALPL2020.P1** on data.census.gov
2. Filter: Geography → Metropolitan Statistical Areas → All MSAs
3. Download → Save as **`data/census/DECENNIALPL2020.P1-2026-03-25T232220.csv`**

### CBSA County Crosswalk (needed for MSA-level NRI queries)

1. Go to: https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
2. Download the current crosswalk and store it under **`data/census/`** as a file like **`list1_2023.xlsx`** or a CSV equivalent; the app accepts `list*.xlsx`, `list*.xls`, and `list*.csv` variants.

### Census Tract Shapefiles (optional, for fast tract-FIPS assignment)

Without shapefiles, the app uses the Census Geocoder API to resolve each house's census tract — this is slow (~0.15s per house). With shapefiles it's instant.

1. Go to: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
2. Select: Year = 2020 → Layer = Census Tracts → Select your states
3. Unzip into **`data/shapefiles/`** (one or multiple states)
4. Re-run: `python setup_data.py --resolve-tracts`

### BikePGH route data (optional but required for local bike routing)

The route planner uses the locally ingested BikePGH network from the DuckDB `bike_routes` table and its source data under `data/bike/`.

- `data/bike/Bike Lanes/`
- `data/bike/Pittsburgh/`

If those files are present, the app can answer route questions like:
- “Is there a bikeable route from Mount Washington to Point State Park?”
- “Find a safe bike route from A to B”

No external road-routing engine is used for the route graph itself; the service only geocodes place names with Nominatim.

### Sold Homes (optional)

- Drop CSV files with sold home data into **`data/sold/`**
- Required columns: `address`, `lat`, `lon`, `sold_price`, `sqft`, `sold_date`
- Optional: `list_price`, `beds`, `baths`

### Crime Data (optional, powers the "Crime" map layer)

Every city publishes crime data differently — different columns, different
offense vocabularies, sometimes different file formats entirely. Rather than
teach the app one format, each city gets its own small parser (see
`services/crime_sources.py`); currently supported:

**Baltimore · Boston · Buffalo · Chicago · Indianapolis · Minneapolis · Philadelphia · Pittsburgh**

1. Download the city's open-data crime/incident export (.csv or .xlsx — most
   city data portals offer both) — typically named something like "Crime
   Incidents", "Part 1 Crime", or "Police Blotter"
2. Drop it in **`data/crime/<city>/`**, e.g. `data/crime/chicago/crimes_2024.csv`
   — the folder name must match one of the city keys above (lowercase)
3. You can drop multiple files per city (e.g. one per year) — they're all
   read and combined
4. Run `python setup_data.py --only crime`

Each incident is classified into a standardized category (Homicide, Robbery,
Burglary, Theft, ...) and assigned a **severity weight from 1–10** — see
`services/crime_taxonomy.py` for the full category list and the reasoning
behind the weights. The "Crime" map layer's heatmap intensity is driven by
this weight, not raw incident count, so a block with a few thefts doesn't
outweigh a block with one assault. The weights are a plain, editable Python
list — tune them to your own judgment if the defaults don't match how you'd
weigh things.

To cover a city that isn't listed above, add a new parser class to
`services/crime_sources.py` (subclass `CrimeParserBase`, following the
pattern of the existing city classes) and register it — no changes needed
anywhere else.

---

## Usage Guide

### Map

- **Colored markers** by status:
  - 🔵 Blue = Active
  - 🟡 Yellow = Pending
  - 🟣 Purple = Contingent
  - 🟢 Green = Pre-Market
- **Click any marker** to open the sidebar
- **Map Layer control** (top-right): toggle an optional overlay on top of the map — **Crime** (severity-weighted heatmap), **NRI** (FEMA risk choropleth by census tract), or **Bike Lanes** (BikePGH network overlay). Only one layer is shown at a time; select **None** to turn it off. Layers re-fetch automatically as you pan/zoom.
- **Bike route planner**: a control on the map lets you enter a start and end location and route using the local BikePGH network. Resulting route geometry is drawn directly on the map and in chat responses when applicable.

### House Sidebar

| Tab | Contents |
|-----|----------|
| **Details** | Price, beds/baths/sqft, livability scores, NRI summary |
| **Risk** | Full NRI breakdown — 18 hazards, composite score, percentile, EAL |
| **Chat** | AI agent for this house — pricing, risk Q&A, description analysis |
| **Docs** | All stored text/photos; upload new photos |

### House Chat — Example Conversations

```
User: <pastes Redfin/Zillow description>
Agent: Thanks — I've saved that to the knowledge base. How can I help?

User: Estimate a fair price for this house
Agent: [runs price estimator, checks comparables, sold homes in tract]

User: How bad is the flood risk here?
Agent: [fetches NRI, explains RFLD_RISKS score and EAL in plain language]

User: What's the crime like in this neighborhood?
Agent: [queries crime_incidents for this house's crime_city, summarizes by category and severity]

User: Is the HOA fee reasonable?
Agent: [compares HOA against other houses in the same city]
```

### General Chat — Example Questions

- *"Among the top 50 metro areas by population, which have the lowest overall risk?"*
- *"Which of my saved houses has the best combined walk + transit score?"*
- *"Compare tornado risk between Dallas and Houston census tracts"*
- *"What's the median price/sqft across all my Austin houses?"*
- *"Which city has the most severe crime, weighted, not just the most incidents?"*

---

## Reloading Data

After adding new CSV files, re-run:

```bash
python setup_data.py --only redfin    # just Redfin
python setup_data.py --only sold      # just sold homes
python setup_data.py --only crime     # just crime data
python setup_data.py --only bike      # just BikePGH route data
python setup_data.py --only census    # CBSA crosswalk + tract/MSA populations
python setup_data.py --only geocode   # retry pending sold-home geocodes
python setup_data.py --only match     # link sold records to houses
python setup_data.py --only repair    # repair X-coded MSA codes
python setup_data.py --only sold --no-geocoding  # load sold data without network geocoding
python setup_data.py --resolve-tracts # resolve missing house tract FIPS values
python setup_data.py                  # everything
```

The vector database (ChromaDB) grows automatically as you paste descriptions in chat.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Map shows no houses | Run `setup_data.py`, check `data/redfin/*.csv` exists |
| NRI data missing | Download the NRI shapefile or CSV into `data/nri/` (preferred: `NRI_CensusTracts_Prod.shp`; fallback: `NRI_Table_CensusTracts.csv`) |
| Tract FIPS not resolved | Run `python setup_data.py --resolve-tracts` |
| Chat says "I couldn't generate a response" | Make sure `llama-server` is running (or Ollama if you chose that stack) |
| Embedding errors | If using Ollama embeddings: pull `nomic-embed-text` with `ollama pull nomic-embed-text`. If using another embedding provider, configure accordingly. |
| Slow tract resolution | Add TIGER/Line shapefiles to `data/shapefiles/` |
| "Crime" layer is empty | Confirm `data/crime/<city>/` has files for a city your map view overlaps, then `python setup_data.py --only crime` |
| "NRI" layer is empty / shows a warning | It needs both tract *geometry* (from the NRI shapefile, or TIGER/Line shapefiles in `data/shapefiles/`) and tract *attributes* (`python setup_data.py --only nri`) — the layer's warning message says which is missing |
| Phoenix is unavailable | Set `PHOENIX_ENABLED=false` to run without tracing, or start it manually with `python -m phoenix.server.main serve` |
| Bike route returns no route | Confirm BikePGH layers were loaded with `python setup_data.py --only bike`; the router does not fall back to an external road-routing service |

---

## Adding New Data Sets

The architecture is designed to grow. To add a new data set:

1. Add a new `load_xyz()` function in `services/data_loader.py`
2. Add a new table in `db/duckdb_store.py` → `_ensure_schema()`
3. Add ONE `TableMeta` entry (description + notes on any non-obvious columns)
   in `db/schema_catalog.py`. If it joins to an existing table, add one
   `Relationship` entry alongside it. You do NOT need to write a new agent
   tool or teach the LLM a new SQL pattern — `query_database` plus the schema
   catalog is enough for the agent to work out how to query it, including
   joins to other tables.
4. Call `setup_data.py` to load it

`db/schema_catalog.py` is the single source of truth for what the agent knows
about the data: it combines live introspection of the running database
(so column names/types can't go stale) with curated notes for things no
amount of introspection can tell you — which columns are reliably populated,
which joins need a non-obvious expression, which tables need a default filter
to be meaningful. `check_data_availability`, `get_database_schema`,
`setup_data.py`'s summary, and the response validator's fallback message all
read from it, so there's nothing else to keep in sync when you add a table.

---

## Project Structure

```
main.py               FastAPI app + all HTTP endpoints
config.py             All settings (paths, models, ports)
setup_data.py         One-time data loader script
run_eval.py            Agent evaluation pipeline entry point
update_eval_ground_truth.py  Regenerate golden expectations from fixture SQL

agents/
  tools.py            LangChain tools (SQL, vector search, price estimation)
  query_planner.py    Deterministic analytical query planning and semantic mappings
  house_agent.py      Per-house ReAct agent (LangGraph)
  general_agent.py    General ReAct agent (LangGraph)
  response_validator.py  Post-hoc check that replies are grounded in real tool output

db/
  duckdb_store.py     All SQL queries and schema management
  schema_catalog.py   Metadata layer — table/column notes + join graph for the agent
  vector_store.py     ChromaDB — embed, store, search text documents

services/
  data_loader.py      Parsers for Redfin, NRI, Census, Sold, Crime
  crime_sources.py    Per-city crime file parsers (one class per city)
  crime_taxonomy.py   Standardized crime categories + severity weights
  layers.py           Viewport-scoped queries behind the Crime/NRI map layers
  geo_utils.py         Census tract FIPS assignment (shapefile or API)

eval/
  fixtures.py         Builds a small deterministic DB with hand-verifiable answers
  golden_set.py        The golden examples (structured + free-text)
  scoring.py            Assert-equal scoring for structured examples
  judge.py              LLM-as-judge scoring for free-text examples
  mock_agent.py         Scripted stand-in used only by --mock (harness smoke test)
  tests/test_scoring.py Unit tests for the scorer itself, no LLM/DB needed
  reports/              Timestamped JSON + Markdown reports land here

static/
  index.html          Leaflet map + sidebar + chat UI
  style.css            App styles, BikePGH route visuals, route planner UI
  app.js              Frontend logic, layer toggles, bike route rendering

observability.py      Phoenix tracing + Prometheus metrics helper

data/
  redfin/             Drop Redfin CSVs here
  sold/               Drop sold-homes CSVs here
  nri/                FEMA NRI shapefile or CSV
  census/             Census P1 tables + CBSA crosswalk
  crime/<city>/       Drop each city's raw crime export here
  bike/               BikePGH network sources used by the local router
  shapefiles/         TIGER/Line tract shapefiles (optional)
```

---

## Observability & metrics

The app auto-starts a local Phoenix trace collector when
`PHOENIX_ENABLED=true` (the default in `.env.example`) and exposes
Prometheus-compatible metrics at `/metrics`. If Phoenix is already listening on
the configured port, the app reuses it and does not start a second process.

- Trace collection is configured via `PHOENIX_COLLECTOR_ENDPOINT`, `PHOENIX_PROTOCOL`,
  and `PHOENIX_UI_URL`
- The General Chat flow records agent, tool, and validation spans
- This is intentionally fail-open: if Phoenix is unavailable, the app continues
  running normally

```bash
python -m phoenix.server.main serve
```

Use the command above only when starting Phoenix separately, for example after
setting `PHOENIX_ENABLED=false`. Open the Phoenix UI at the configured local URL
(default: http://127.0.0.1:6006).

## HTTP API

The browser uses these main endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/houses` | Return saved houses as GeoJSON |
| `GET /api/layers/crime` | Return the viewport's severity-weighted crime grid |
| `GET /api/layers/nri` | Return NRI tract geometry and attributes |
| `GET /api/layers/bike` | Return BikePGH features for the current viewport |
| `POST /api/bike/route` | Route between geocoded places on the local bike graph |
| `POST /api/chat` | Ask the general cross-city agent |
| `POST /api/house/{id}/chat` | Ask the agent about one house |
| `GET /metrics` | Expose Prometheus metrics |

The API also supports house document and photo operations; the interactive UI
is the recommended way to use those endpoints.

---

## Evaluation

`run_eval.py` runs a small golden set (`eval/golden_set.py`) against the real
agents — the same `run_general_chat` / `run_house_chat` the app itself calls
— on a dedicated, deterministic evaluation database (`eval/fixtures.py`), not
your real data.

```bash
python run_eval.py                 # full run against your configured model
python run_eval.py --list          # see what's in the golden set
python run_eval.py --tags nri,sold_homes    # run a subset
python run_eval.py --mock          # smoke-test the harness itself, no model server needed
python .\run_eval.py --skip-house-agent     # skip evals on house agent
```

Each golden example is one of:
- **structured** — has a definite right answer (a count, a ranking, a set of
  cities). Scored by exact comparison against the reply text: as an ordered
  sequence when order matters, as a set when it doesn't. No LLM involved in
  grading these.
- **free_text** — open-ended (a tradeoff, an explanation). Scored by an LLM
  judge against a rubric in `eval/judge.py`. Configure the judge model via
  `judge_llama_server_base_url` / `judge_llama_server_model` in `.env`; it
  defaults to the same server as the agent under test, but an independent or
  stronger judge is stronger evidence than a model grading itself.

Reports land in `eval/reports/` as both JSON (for tooling) and Markdown (for
reading). Exit code is non-zero if anything failed or errored, so this is
safe to wire into CI. `eval/tests/test_scoring.py` tests the assert-equal
scorer itself against hand-crafted replies — run it directly any time you
change `eval/scoring.py`.

To add a golden example: add fixture data to `eval/fixtures.py` if needed
(prefer deriving expected values from the fixture data programmatically, the
way the existing examples do, over hand-typing a number), then add one
`GoldenExample` to `eval/golden_set.py`.

### Updating evaluation ground truth

`update_eval_ground_truth.py` regenerates structured expected values and the
numeric facts used in free-text rubrics from SQL executed against
`eval/fixture_data/eval_fixture.duckdb`. It does not ask an agent to generate
SQL and never uses previous agent answers as ground truth. It also validates
fixture invariants before rewriting `eval/golden_set.py`.

Run it after changing `eval/fixtures.py` or the fixture schema, and before
running the evaluation suite:

```bash
python update_eval_ground_truth.py
python run_eval.py
```

The updater accepts an alternate fixture database or golden-set path:

```bash
python update_eval_ground_truth.py --db path/to/eval_fixture.duckdb
python update_eval_ground_truth.py --golden path/to/golden_set.py
```

`--skip-house-agent` is accepted for workflow consistency with `run_eval.py`;
it does not remove house-agent examples from the golden file. It only documents
that the subsequent evaluation run may skip those examples. DuckDB must be
installed in the active Python environment.

---

## Capability matrix

This section distinguishes capabilities implemented by the application from
capabilities currently covered by the deterministic evaluation fixture.

### Explicitly supported

| Area | Supported questions and behavior | Fixture coverage |
|------|----------------------------------|------------------|
| House inventory | Counts, city/state/status filters, price aggregates, Walk/Bike/Transit scores, missing-score checks, and rankings | Strong |
| House inventory scope | “My houses” means the full inventory; saved/favorites require explicit list or favorite wording | Supported, but weakly tested because all fixture houses are favorites |
| MSA and tract population | Population totals, rankings, named-MSA comparisons, and specific tract queries | Strong |
| NRI risk | Overall/composite risk, riverine flooding, hurricane, wildfire, other canonical hazards, averages, and rankings | Strong for documented joins and populated fixture columns |
| MSA to NRI analysis | Uses `census_msa -> cbsa_counties -> nri_tracts`, including top-N-by-population universes and MSA-grain aggregation | Strong for the documented pattern |
| Sold homes | Arm’s-length filtering, price averages/rankings, and tract-scoped sold comparables | Strong |
| Individual houses | Stored details, NRI data, risk percentile, EAL, vulnerability/resilience fields, and data-derived price estimates | Strong or partial when fields are NULL |
| House documents | Search a house’s stored descriptions/documents or search across all houses | Supported when documents are loaded |
| Bike routing | Route addresses, neighborhoods, landmarks, parks, or coordinates on the loaded BikePGH network | Supported when BikePGH data is loaded |
| Crime-aware routing | Remove BikePGH edges intersecting buffered high-density crime cells before Dijkstra and return analysis visualization | Supported, but heuristic |
| Crime analytics | Incident counts, severity-weighted crime, standardized categories, and city/month analysis | Supported when crime data is loaded |

### Supported with caveats

- A house’s `msa_code` is usually NULL in Redfin data. Tract-based MSA lookup
  should use the documented county bridge rather than assuming the house row
  carries a reliable CBSA/MSA code.
- The distinction between full inventory and favorites is implemented, but the
  current fixture marks every house `is_favorite = TRUE`, so it cannot detect a
  mistaken favorite filter. Add a non-favorite fixture house to test this rule.
- Arbitrary multi-table questions and unusual ranking variants still depend on
  LLM-generated SQL being written correctly at least once. The application
  provides a three-attempt generation/repair loop and a deterministic fallback
  that compiles SQL directly from the structured plan when generation fails
  (see **General Chat: Agent Architecture & Design Philosophy** above, and
  `AGENT_ARCHITECTURE.md`) — but that fallback can only build joins/operations
  the catalog already declares; it does not invent new relationship paths.
- Sold-home tract questions require geocoded rows; pending rows have NULL
  geography and must not be treated as tract-local.
- Historical house questions require populated `house_snapshots`; the current
  fixture does not populate that table.
- Description search requires documents in the vector store. Crime sources are
  city-specific, and missing or incompatible files are skipped.
- Crime severity means `SUM(severity_weight)`, while incident volume means
  `COUNT(*)`; ambiguous wording can lead to different valid metrics.
- Bike endpoints too far from the loaded network are rejected, and no route
  means no continuous path in the loaded BikePGH graph, not necessarily no
  real-world route. Crime avoidance is a density heuristic, not a safety
  guarantee.
- Geometry/blob columns are hidden from the SQL agent, so arbitrary polygon or
  geometry analytics are not available through `query_database`.
- General Chat is grounded in loaded application data and approved functions; it
  is not a web-search or general recommendation agent.

### Not currently supported

The current data model and approved General Chat functions do not provide:

- Mortgage payment, loan affordability, down-payment, or rate calculations
- Property-tax liability or tax forecasting
- School ratings or school-assignment quality
- Insurance quotes or premium prediction
- Future price-appreciation or five-year value forecasting
- Current Internet market, news, traffic, or live route conditions
- A guarantee that a route is safe

### Evaluation fixture limits

The 53-example golden set covers only data populated by `eval/fixtures.py`:

- Six houses across Pittsburgh, Denver, Miami, and Austin
- Census tract and MSA populations for four named metros
- NRI overall, riverine flood, hurricane, and wildfire values
- Six sold-home records, including one deliberately invalid/non-market $1 sale
  and one pending-geocode record
- One unmatched MSA with placeholder code `Xplaceholder1` and no CBSA bridge row

The fixture does not populate crime incidents, BikePGH routes, or house snapshot
history. Those capabilities need a separate integration fixture using
representative loaded data rather than invented expected values.

### Recommended evaluation strategy

Keep the golden set as the deterministic fixture regression suite and maintain a
separate integration suite for:

1. Crime analysis across at least two loaded cities and multiple categories or months
2. Successful, no-route, and crime-avoidance BikePGH routing cases
3. Multiple price/status observations in `house_snapshots`
4. Several house documents and document types
5. A non-favorite fixture house so inventory scope is genuinely tested

---

## Bike routing

The app exposes `POST /api/bike/route` with:

```json
{"start":"Mount Washington","end":"Point State Park","city":"Pittsburgh, PA"}
```

Endpoints are place strings, not required map clicks. The service geocodes them
with Nominatim, then routes exclusively on the locally ingested BikePGH linework
stored in the `bike_routes` DuckDB table. No external road-routing engine is used.

### Endpoint precision

- Exact coordinates, street addresses, named places, landmarks, or neighborhoods with `city` context are supported.
- For Pittsburgh, ambiguous place names are auto-appended with `Pittsburgh, PA` when appropriate.
- Geocoding results are cached in the `geocode_cache` table.

### BikePGH layers used for routing

- Bike Lanes
- Bikeable Sidewalks
- Cautionary Bike Route
- On Street Bike Route
- Protected Bike Lanes
- Sharrows
- Trails

If the locally ingested network does not contain a continuous path between the snapped endpoints, the request returns **no route** rather than falling back to OSM street routing.

### Free/open services

- Nominatim / OpenStreetMap is used only for endpoint place-name geocoding.
- Shapely + DuckDB build and query the local BikePGH graph.

### Route semantics

- Distance is computed from the local BikePGH graph. Travel time is an estimate.
- Turn instructions refer to mapped BikePGH infrastructure rather than inventing street names.

---

## Developer scripts & tests

These utility scripts and tests are intended for debugging, data validation, evaluation, and regression testing. Run them from the repository root:

- `tests/test_guardrails.py`: Full test suite for multi-layer guardrails (Input prompt injection defense, Code AST sandboxing, read-only SQL validation, output grounding) plus microsecond latency benchmark. Usage: `python tests/test_guardrails.py`
- `tests/test_bikepg_h_visualization.py`: Regression checks verifying BikePGH visualization specs and styling matches ground truth. Usage: `python tests/test_bikepg_h_visualization.py`
- `run_eval.py`: Agent evaluation pipeline. Runs the golden set examples against real agents or mock fixture DB, scores them, and writes timestamped reports to `eval/reports/`. Usage: `python run_eval.py [--mock]`
- `debug_bike_route.py`: Lightweight checks for BikePGH city-key normalization and routing helpers. Usage: `python debug_bike_route.py`
- `debug_flood_query.py`: Step-by-step SQL debugger for the flood-risk query; runs CTEs, prints table counts, join diagnostics, and sample rows to pinpoint where the chain breaks. Usage: `python debug_flood_query.py`
- `debug_nri_columns.py`: Inspect the NRI shapefile's DBF column names and show NULL counts for hazard columns in `nri_tracts`. Usage: `python debug_nri_columns.py`
- `diagnose_msa.py`: Finds `X`-coded MSA rows that don't match `cbsa_counties`, suggests best CBSA candidates using a fuzzy normalizer, and can apply fixes with `--apply`. Usage: `python diagnose_msa.py [--apply]`



## Crime-aware bike routing
Crime-aware bike route requests are now executed deterministically in `agents/general_agent.py` when the user asks for a bike route that avoids crime/high-crime/dangerous areas. This prevents a local LLM from omitting the `find_bike_route` tool call. The resulting `find_bike_route` tool span is visible in observability, and its intermediate filtered BikePGH/crime visualization remains attached to the response.

Example request:

> Find a bike route from Mount Washington to Point State Park that avoids high-crime areas.


### Behavior

For bike-route questions that explicitly ask to avoid crime-dense/high-crime areas, the app now treats crime avoidance as a deterministic spatial filter rather than a language-model preference. The routing graph is cloned per request, the top crime-density cells (default: top 10% of occupied cells) are expanded by a small exclusion buffer, and BikePGH edges intersecting those exclusion areas are removed before Dijkstra routing. The intermediate map shows the relevant high-density crime cells, the BikePGH edges removed by the filter, and the BikePGH network that remains. A final route map is rendered only when a continuous route exists in that filtered graph.

The intermediate map is intentionally corridor-focused so the crime layer remains legible instead of painting the whole city with low-opacity cells. The visual is explanatory only; crime density is a heuristic and not a safety guarantee.


### Consistent hotspot scoring

The bike/crime intermediate visualization and the routing filter now share one crime-density model. Each occupied grid cell receives the same intensity score shown on the map: 75% normalized incident count + 25% normalized severity-weighted score. The routing hotspot percentile is applied to that same intensity score using a citywide baseline. The selected cells are buffered, then BikePGH geometry is evaluated at logical intersection-to-intersection segment granularity so one affected sub-edge does not create an inconsistent dark/orange/dark split within the same real-world segment.
