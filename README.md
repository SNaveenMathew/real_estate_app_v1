# 🏠 Real Estate Intelligence

A local, AI-powered map app for analyzing houses with FEMA National Risk Index data, Census demographics, Redfin listings, and an LLM chat interface — all running on your machine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (Leaflet Map + Chat UI)                            │
│   └─ Markers per house (click → sidebar)                    │
│   └─ House Chat  (per-property, LangGraph ReAct agent)      │
│   └─ General Chat (cross-city, LangGraph ReAct agent)       │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP (FastAPI)
┌──────────────────────▼──────────────────────────────────────┐
│  FastAPI  (main.py)                                         │
│   ├─ /api/houses      GeoJSON of all houses                 │
│   ├─ /api/house/{id}/chat   House-specific agent            │
│   ├─ /api/chat              General agent                   │
│   └─ /api/house/{id}/photo  Photo upload                    │
└───────┬─────────────────────────────┬───────────────────────┘
        │                             │
┌───────▼────────┐         ┌──────────▼───────────┐
│  DuckDB        │         │  ChromaDB (vector)   │
│  ─ houses      │         │  ─ house_documents   │
│  ─ nri_tracts  │         │    (descriptions,    │
│  ─ census_*    │         │     photos, notes)   │
│  ─ sold_homes  │         └──────────────────────┘
│  ─ cbsa_*      │
└───────┬────────┘         ┌──────────────────────┐
        │                  │  llama.cpp / llama-server  │
        └──────────────────│  ─ Gemma or other HF model  │
               │  ─ (see LLM setup below)   │
               └──────────────────────┘
```

---

## Prerequisites

- Python 3.10+
- `llama-server` (llama.cpp) installed and running, or Ollama as an alternative

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
└── data/
    ├── redfin/       ← drop Redfin CSV exports here
    ├── sold/         ← drop sold-homes CSV files here
    ├── nri/          ← NRI_Table_CensusTracts.csv
    ├── census/       ← DECENNIALPL2020_P1_tract.csv, _msa.csv, list1_2020.csv
    └── shapefiles/   ← TIGER/Line tract shapefiles (optional but recommended)
```

### 2. Install dependencies

```bash
cd real_estate_app
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 3. LLM: run `llama-server` (default)

This repository is configured to use `llama-server` (from the `llama.cpp` project)
as the primary local LLM endpoint. A small example command that works with the
Gemma model (adjust paths/flags for your machine):

```bash
# Example: run llama-server serving a HF model (Gemma) with quantized caches
llama-server -hf DuoNeural/Gemma-4-26B-A4B-it-GGUF:Q3_K_M \
  -ngl 999 -c 28672 -fa on --cache-type-k q8_0 --cache-type-v q8_0
```

If you prefer Ollama, it remains supported as an alternative — see the
`config.py` and agent comments for how to switch. For Ollama, you would
pull and run:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve
```

Note: the project is configured to use Ollama's `nomic-embed-text` for embeddings by
default (see `.env.example`). If you run a pure `llama-server` stack, either run
Ollama alongside it for embeddings or update `db/vector_store.py` to use a
different embedding provider.

> **Lower VRAM?** For `llama-server` use a smaller HF model or reduce cache/memory flags.

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # Mac/Linux
# Edit .env if needed (defaults work for local llama-server; Ollama optional)
```

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
3. The app expects columns including `lat`/`lon`, `address`, `city`, `state`, `zip`, `price`, `beds`, `baths`, `square feet`, `status`
4. If you already have Walk/Bike/Transit scores (from the WalkScore API), include them as `walk score`, `bike score`, `transit score`
5. Drop the CSV into **`data/redfin/`**

You can drop multiple CSV files (e.g. one per city).

### FEMA National Risk Index (highly recommended)

1. Go to: https://www.fema.gov/about/openfema/data-sets/national-risk-index-data
2. Download **NRI_Table_CensusTracts.csv** (census tract level, ~800 MB)
3. Place at: **`data/nri/NRI_Table_CensusTracts.csv`**

### Census Tract Populations

1. Go to https://data.census.gov
2. Search for table **DECENNIALPL2020.P1**
3. Filter: Geography → Census Tracts → All States → All Tracts
4. Download → Save as **`data/census/DECENNIALPL2020_P1_tract.csv`**

### MSA Populations

1. Same table **DECENNIALPL2020.P1** on data.census.gov
2. Filter: Geography → Metropolitan Statistical Areas → All MSAs
3. Download → Save as **`data/census/DECENNIALPL2020_P1_msa.csv`**

### CBSA County Crosswalk (needed for MSA-level NRI queries)

1. Go to: https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
2. Download **list1_2020.xls** → save/convert as **`data/census/list1_2020.csv`**

### Census Tract Shapefiles (optional, for fast tract-FIPS assignment)

Without shapefiles, the app uses the Census Geocoder API to resolve each house's census tract — this is slow (~0.15s per house). With shapefiles it's instant.

1. Go to: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
2. Select: Year = 2020 → Layer = Census Tracts → Select your states
3. Unzip into **`data/shapefiles/`** (one or multiple states)
4. Re-run: `python setup_data.py --resolve-tracts`

### Sold Homes (optional)

- Drop CSV files with sold home data into **`data/sold/`**
- Required columns: `address`, `lat`, `lon`, `sold_price`, `sqft`, `sold_date`
- Optional: `list_price`, `beds`, `baths`

---

## Usage Guide

### Map

- **Colored markers** by status:
  - 🔵 Blue = Active
  - 🟡 Yellow = Pending
  - 🟣 Purple = Contingent
  - 🟢 Green = Pre-Market
- **Click any marker** to open the sidebar

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

User: Is the HOA fee reasonable?
Agent: [compares HOA against other houses in the same city]
```

### General Chat — Example Questions

- *"Among the top 50 metro areas by population, which have the lowest overall risk?"*
- *"Which of my saved houses has the best combined walk + transit score?"*
- *"Compare tornado risk between Dallas and Houston census tracts"*
- *"What's the median price/sqft across all my Austin houses?"*

---

## Reloading Data

After adding new CSV files, re-run:

```bash
python setup_data.py --only redfin    # just Redfin
python setup_data.py --only sold      # just sold homes
python setup_data.py                  # everything
```

The vector database (ChromaDB) grows automatically as you paste descriptions in chat.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Map shows no houses | Run `setup_data.py`, check `data/redfin/*.csv` exists |
| NRI data missing | Download & place `NRI_Table_CensusTracts.csv` in `data/nri/` |
| Tract FIPS not resolved | Run `python setup_data.py --resolve-tracts` |
| Chat says "I couldn't generate a response" | Make sure `llama-server` is running (or Ollama if you chose that stack) |
| Embedding errors | If using Ollama embeddings: pull `nomic-embed-text` with `ollama pull nomic-embed-text`. If using another embedding provider, configure accordingly. |
| Slow tract resolution | Add TIGER/Line shapefiles to `data/shapefiles/` |

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

agents/
  tools.py            LangChain tools (SQL, vector search, price estimation)
  house_agent.py      Per-house ReAct agent (LangGraph)
  general_agent.py    General ReAct agent (LangGraph)
  response_validator.py  Post-hoc check that replies are grounded in real tool output

db/
  duckdb_store.py     All SQL queries and schema management
  schema_catalog.py   Metadata layer — table/column notes + join graph for the agent
  vector_store.py     ChromaDB — embed, store, search text documents

services/
  data_loader.py      CSV parsers for Redfin, NRI, Census, Sold
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
  style.css            All styles
  app.js               Frontend logic

data/
  redfin/             Drop Redfin CSVs here
  sold/               Drop sold-homes CSVs here
  nri/                FEMA NRI CSV
  census/             Census P1 tables + CBSA crosswalk
  shapefiles/         TIGER/Line tract shapefiles (optional)
```

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
```

Each golden example is one of:
- **structured** — has a definite right answer (a count, a ranking, a set of
  cities). Scored by exact comparison against the reply text: as an ordered
  sequence when order matters, as a set when it doesn't. No LLM involved in
  grading these.
- **free_text** — open-ended (a tradeoff, an explanation). Scored by an LLM
  judge against a rubric (`eval/judge.py`). Configure the judge model via
  `judge_llama_server_base_url` / `judge_llama_server_model` in `.env` — it
  defaults to the same server as the agent under test, but an independent or
  stronger judge is stronger evidence than a model grading itself.

Reports land in `eval/reports/` as both JSON (for tooling) and Markdown (for
reading). Exit code is non-zero if anything failed or errored, so this is
safe to wire into CI. `eval/tests/test_scoring.py` unit-tests the assert-equal
scorer itself against hand-crafted replies — run it directly any time you
change `eval/scoring.py`.

To add a golden example: add fixture data to `eval/fixtures.py` if needed
(prefer deriving expected values from the fixture data programmatically, the
way the existing examples do, over hand-typing a number), then add one
`GoldenExample` to `eval/golden_set.py`.
