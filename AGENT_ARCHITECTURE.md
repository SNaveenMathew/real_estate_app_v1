# Agent Architecture

This document is the detailed companion to the **General Chat: Agent
Architecture & Design Philosophy** section of `README.md`. That section is
the overview; this is the full mechanics — every stage, every validation
rule, every deterministic-vs-LLM boundary, and the reasoning behind where
each line is drawn. If you're extending the agent, adding a data set the
agent should reason about, or debugging a Phoenix trace, this is the
reference.

It covers **General Chat** (`agents/general_agent.py` + `agents/tools.py`),
since that's where the deterministic/LLM split is richest. House Chat
(`agents/house_agent.py`) is a separate ReAct agent over a smaller,
house-scoped tool set; it shares the same category of design choices
(approved-function sandboxing, evidence-grounded final answers) but not the
query-planning/SQL pipeline described here.

## Table of contents

1. [Design principle](#1-design-principle)
2. [Which agent handles a request](#2-which-agent-handles-a-request)
3. [General Chat: the full pipeline](#3-general-chat-the-full-pipeline)
   - 3.0 [Input guardrails — deterministic](#30-input-guardrails--deterministic)
   - 3.1 [Orchestration loop](#31-orchestration-loop)
   - 3.2 [Query planning (routing) — deterministic](#32-query-planning-routing--deterministic)
   - 3.3 [Data-model retrieval — deterministic](#33-data-model-retrieval--deterministic-two-distinct-steps)
   - 3.4 [SQL generation — LLM](#34-sql-generation--llm)
   - 3.5 [SQL validation — deterministic, three layers](#35-sql-validation--deterministic-three-layers)
   - 3.6 [Deterministic fallback compilation](#36-deterministic-fallback-compilation)
   - 3.7 [Execution](#37-execution)
   - 3.8 [Final answer + response validation](#38-final-answer--response-validation)
   - 3.9 [Guardrails & Security Framework](#39-guardrails--security-framework)
4. [Complete deterministic-vs-LLM inventory](#44-complete-deterministic-vs-llm-inventory)
5. [Why the boundary is where it is](#5-why-the-boundary-is-where-it-is)
6. [Known boundaries and sharp edges](#6-known-boundaries-and-sharp-edges)
7. [Extending the agent](#7-extending-the-agent)
8. [The catalog is a store, and how it changes](#8-the-catalog-is-a-store-and-how-it-changes)

---

## 1. Design principle

**The LLM writes text. Deterministic code decides what's true, what's
relevant, and what's allowed.**

Two different kinds of work happen in every General Chat turn:

- **Open-ended generation** — there's no single correct string. Writing a
  SQL query's exact syntax, and writing the final answer's exact prose, are
  both like this: multiple outputs could be equally correct.
- **Lookup and verification** — there's exactly one correct answer, and it's
  either a fact already sitting in the catalog/database, or a check with a
  yes/no answer. Which tables a question needs, whether an entity actually
  exists, whether a join path is declared, whether generated SQL matches the
  plan it was given, whether a finished reply is consistent with the
  evidence it cites — none of these benefit from creativity, and all of them
  have one right answer.

The rule this codebase follows: **the LLM only ever does the first kind of
work.** Every instance of the second kind is deterministic code, and it runs
either *before* the LLM (to hand it an already-decided scope) or *after* it
(to check what it produced). No step in this pipeline asks the LLM to grade,
audit, or double-check another LLM call's output — every such check is code.

---

## 2. Which agent handles a request

This isn't an agent decision at all — it's fixed by which HTTP endpoint the
browser calls, which is in turn fixed by which UI panel the person is using:

| Endpoint | Handler | UI context |
|---|---|---|
| `POST /api/house/{id}/chat` | `run_house_chat` (`agents/house_agent.py`) | The **Chat** tab in a specific house's sidebar |
| `POST /api/chat` | `run_general_chat` (`agents/general_agent.py`) | The general **General Chat** panel |

There is no classifier, prompt, or model call involved in this choice — it's
resolved before either agent is ever invoked.

---

## 3. General Chat: the full pipeline

### 3.0 Input guardrails — deterministic

Every chat turn (`run_general_chat` and `run_house_chat`) passes through
`services.guardrails::InputGuardrail` before any LLM call or retrieval step:

- **Prompt Injection & Jailbreak Defense**: Detects adversarial injection patterns
  ("ignore all previous instructions", "DAN mode", system prompt extraction,
  unauthorized execution commands).
- **Sanitization & Length Bounds**: Strips null bytes and caps max input characters
  (`MAX_INPUT_CHARS = 25000`).
- **Short-Circuit Protection**: If flagged as an injection attempt, the turn
  immediately returns a safe canned response without executing LLM spans or
  triggering tool calls. Average latency: $\sim 22\ \mu\text{s}$.

### 3.1 Orchestration loop

`run_general_chat` drives a bounded loop over `CODE_AGENT_MAX_STEPS = 3`
"steps," each pairing a `general_chat.code_agent.step_N` span (generate a
small program) with a `general_chat.code_execution.step_N` span (run it).

Two retrieval calls happen once, before the loop starts, and are reused
across every step within the turn:

- `general_chat.query_planner`: `build_query_plan(message)` on the raw user
  message — recorded for observability. (`_generate_program`, below, computes
  its own fresh plan from the same message; this one is not passed forward.)
- `general_chat.data_model_rag`: `retrieve_data_model_context.invoke(message)`
  — the Chroma-backed retrieval described in §3.3. Its result **is** reused
  across every step's prompt.

Each step:

1. **`_generate_program(...)`** builds a prompt from: data-availability
   context, `"MANDATORY STRUCTURED QUERY PLAN:\n" + build_query_plan(user_message).render()`
   (a *fresh* plan computed on the raw message, every step), the reused
   data-model context, `CODE_AGENT_PROMPT`'s rules, conversation history, and
   — from step 1 onward — the previous step's executed evidence. The model
   returns a small program: one or more `variable = approved_function(kw=...)`
   lines.
2. **`_validate_program(source)`** runs `CodeAgentGuardrail.validate_code_program`
   and walks the AST with a hard allow-list:
   - Only `Assign` and bare-call `Expr` statements are allowed at all.
   - Every call must target a name in `APPROVED_FUNCTIONS` — currently
     `check_data_availability`, `get_database_schema`, `query_database`,
     `retrieve_data_model_context`, `find_bike_route`,
     `search_all_house_descriptions`. Nothing else is callable.
   - No `import`, `Attribute` access, `Subscript`, or `Lambda` anywhere.
   - No `if`/`for`/`while`/`try`/`with`, no `def`/`class` — no control flow,
     no new functions. It's a flat sequence of approved calls, nothing more.
   - Positional arguments are rejected at validation time but *tolerated* at
     execution time (`_invoke_approved` remaps them to the callable's real
     parameter order) — a harmless local-model formatting slip shouldn't
     fail the whole turn when the call itself is already allow-listed safe.
   - `CODE_AGENT_MAX_CHARS = 16000` caps program size. Average latency: $\sim 15\ \mu\text{s}$.
3. **`_execute_program(...)`** actually calls the approved functions and
   collects `(name, result)` pairs. For `query_database(request=..., ...)`
   specifically, `request` is whatever string the orchestrating call chose
   to write — sometimes the user's message verbatim, sometimes the
   orchestrator's own decomposition of it (e.g. splitting "compare X and Y
   for A and B" into a request about X and a separate request about Y, since
   `CODE_AGENT_PROMPT` explicitly permits "multiple approved calls" in one
   step for independent evidence needs). Whatever that text is, it gets its
   *own* fresh `build_query_plan()` call inside `query_database` — that plan,
   not the orchestration-level one, is what SQL generation and validation
   are actually checked against.

**Continuation logic is easy to misread from the loop bound.** Only step 0
is conditional:

```python
if step == 0:
    step_failed = any(
        isinstance(result, str) and (
            "Query returned 0 rows" in result
            or "Code Agent error:" in result
            or "does not match" in result.lower()
            or ("join" in result.lower() and "likely" in result.lower())
        )
        for _, result in calls
    )
    if not any(name in {"query_database", "find_bike_route", "search_all_house_descriptions"} for name in names) or step_failed:
        continue
break
```

If step 0 produced no relevant data-fetching call, or one of those four
hardcoded phrases appears in a result, the loop retries once (step 1).
**Every other case — including step 1 itself, whatever its outcome — falls
straight through to `break`.** There is no branch for `step == 1`. So a turn
makes **at most two** orchestration passes in practice, never three, despite
`CODE_AGENT_MAX_STEPS = 3`. A question that genuinely needs a third round of
"gather evidence, evaluate, gather more" doesn't get one today — see §6.

A step whose *execution* raises (a malformed program, an approved-function
exception) is caught, recorded as `("code_execution_error", str(exc))`,
folded into the accumulated evidence string, and the loop `continue`s rather
than aborting the turn — evidence already gathered survives into the next
attempt or the final answer.

After the loop, `evidence = _render_evidence(all_calls)` (the full
accumulated set, not just the last step) feeds `_write_final_answer` (§3.8).

### 3.2 Query planning (routing) — deterministic

Everything in this subsection is `db/schema_catalog.py` +
`agents/query_planner.py::build_query_plan()`. No LLM call. It runs on
whatever request text is current at that pipeline level (§3.1) and produces
a `QueryPlan`: `required_tables`, `required_relationships`, `operation`,
`select_expression`, `grouping`, `ordering`, `filters`, `entity_filters`,
`resolved_entities`, `null_policy`, `semantic_keys`.

- **Concept matching** (`schema.semantic_matches`) — normalized text, regex
  word-boundary alias matching against each catalog concept's alias list,
  scored by specificity. One hardcoded disambiguation rule: the generic
  `house_inventory` concept is suppressed when a more specific concept
  (`sold_price`, `arms_length_sale`, `history`, `nri_overall_risk`,
  `nri_riverine_flood`) also matches, so "sold price" doesn't fall back to a
  bare inventory count.
- **Entity resolution** (`schema.resolve_request_entities`) — a regex for
  literal FIPS codes, plus substring/prefix matching against **values
  actually fetched from the live database** (`_fetch_values`, cached). This
  is a hard guarantee, not a convenience: an entity only resolves if it's a
  real row that exists right now. Nothing here can hallucinate a city that
  isn't in the data.
- **Relationship-path search** (`schema.relationship_path`) — Dijkstra's
  algorithm over the declared `RELATIONSHIPS` graph, edge-weighted by
  `confidence` ("high" vs not), `preferred`, and `bridge` flags, so multiple
  possible join paths resolve to the one the catalog author blessed, not
  whichever one text-matching happens to suggest. Docstring: *"never
  selected from user-language rules."*
- **Operation selection** (`_select_operation`) — same alias-matching
  mechanism, scored by the matching alias's word count; exactly one
  `(operation, concept)` pair wins.

The output `QueryPlan` becomes, from this point forward, **the single
authoritative description of scope** for that request. Everything
downstream — SQL generation, both validators, the fallback compiler — reads
from it and is checked against it. Nothing downstream is allowed to expand
it.

### 3.3 Data-model retrieval — deterministic, two distinct steps

These answer different questions and shouldn't be conflated:

- **`general_chat.data_model_rag`** (`db/vector_store.py`) — semantic search
  over a dedicated Chroma collection (`data_model_metadata`) of small
  metadata documents, queried with both the raw question and the structured
  plan. Runs once per turn (§3.1), before the orchestrator decides what to
  do, so it has schema/relationship context for *choosing* tool calls. This
  is a **retrieval accelerator, not the authoritative schema** — live DuckDB
  `DESCRIBE` output and row counts remain the source of truth for actual
  availability, and if the embedding service is unavailable, retrieval falls
  back to lexical matching rather than failing the turn. Chroma documents
  are kept in sync with `upsert`.
- **`schema.build_query_context()`** — called inside `generate_sql()` (§3.4)
  for one specific request, after its plan is already built. Pulls targeted
  live schema and relationship text for *just* the tables that plan already
  selected. Not a search — a direct, deterministic read of
  `db/schema_catalog.py` for exactly what's already in scope.

`db/schema_catalog.py` is the single source of truth behind both: live
introspection (so column names/types can't drift out of sync) combined with
curated notes for what introspection alone can't tell you — which columns
are reliably populated, which joins need a non-obvious expression, which
tables need a default filter to be meaningful. `check_data_availability`,
`get_database_schema`, `setup_data.py`'s summary, and the response
validator's fallback message all read from the same catalog.

Two catalog-level design notes worth knowing:

- The planner keeps `universe_limit` separate from `result_limit`: "top 50
  MSAs with the lowest risk" first selects the 50 largest MSAs (the
  universe), then ranks *those 50* by the requested NRI metric — population
  filtering and risk ranking are not the same operation and aren't allowed
  to collapse into one.
- The canonical MSA/NRI join path is `census_msa -> cbsa_counties ->
  nri_tracts`, aggregated at MSA grain. Semantic SQL views (`house_rankings`,
  `nri_msa_risk`) are intentionally not part of this architecture — the
  agent queries physical tables only; `_ensure_schema()` removes any such
  views left by older builds on startup.

### 3.4 SQL generation — LLM

`agents/tools.py::generate_sql(request, plan, focused)`. One `ChatOpenAI`
call at `temperature=0.0` against `llama-server`, capped at
`settings.code_agent_max_tokens` (currently 1500). The cap exists for a
specific, documented reason (`get_code_agent`'s own comment): without one, a
call that doesn't hit a recognized stop token keeps decoding instead of
returning — this actually happened once, running to 24k+ tokens and ~5.5
minutes before being cut off at the server's context limit.

The prompt is exactly:

```
USER REQUEST: <request>
STRUCTURED QUERY PLAN (authoritative): <query_plan.render()>
TARGETED LIVE DATA MODEL (authoritative): <schema.build_query_context(...)>
Generate exactly one DuckDB SELECT/WITH query. Do not add semantic scope,
filters, tables, or joins absent from the plan/model.
```

plus a `SYSTEM_PROMPT` establishing the same constraint as a standing rule
(use documented relationship paths only; never invent scope; output SQL
only). `focused=True` on a retry appends a `REPAIR CONTEXT` block explaining
*why* the previous attempt was rejected (§3.6) — this is what makes retrying
worth anything at `temperature=0.0`: the prompt actually changed, even
though sampling didn't become non-deterministic.

`_clean_sql()` strips Markdown code fences the model might wrap around its
answer. This is deliberately still deterministic post-processing rather than
a stricter prompt instruction: `SYSTEM_PROMPT` already says "output SQL
only," but trusting a small local model to *never* deviate from a formatting
instruction is a worse bet than a two-line strip that costs nothing and
can't fail.

### 3.5 SQL validation — deterministic, two layers

**Layer 1 — `validate_sql(sql)`: security and shape.** A hard allow-list,
not a preference:

- Non-empty.
- No `INSERT | UPDATE | DELETE | DROP | CREATE | ALTER | TRUNCATE | COPY |
  EXPORT | ATTACH | DETACH | INSTALL | LOAD | CALL | PRAGMA`, anywhere,
  case-insensitive.
- Must start with `SELECT` or `WITH`.
- Single statement only (rejects an embedded `;` followed by more content).
- Every table referenced via `FROM`/`JOIN`/etc. must be in
  `schema.list_table_names()` — an agent-visible table that actually exists.

**Layer 2 — `_validate_sql_against_plan(sql, query_plan)`: does this SQL
match what was decided.** Four checks, each comparing the generated SQL
against the `QueryPlan` from §3.2, never against the model's own judgment:

1. **Table set** — the referenced tables must equal `required_tables`
   exactly; neither a missing table nor an extra one is allowed.
2. **Entity-literal presence** — every literal value in `entity_filters`
   (a real, live-database value resolved in §3.2) must appear somewhere in
   the SQL text.
3. **Semantic-filter presence** — every filter the plan requires (e.g. the
   arm's-length sale predicate, a NULL-exclusion) must have all of its
   normalized field/value atoms present in the SQL, comparing atoms rather
   than a brittle full-string match.
4. **Unplanned-filter scan** — catches the LLM adding a scope-narrowing
   filter on a column tied to a semantic concept the plan didn't select
   (e.g. quietly filtering on a walk-score-adjacent column for a question
   that never asked about walk scores). **This check is scoped to the WHERE
   clause only**, via a small extractor
   (`_where_clause_text`) that pulls text between `WHERE` and the next
   clause boundary (`GROUP BY`/`ORDER BY`/`LIMIT`/another `WHERE`, for
   multi-`WHERE` shapes like CTEs). This scoping is deliberate: a `JOIN ...
   ON` predicate — e.g. the required bridge `census_msa.msa_code =
   cbsa_counties.cbsa_code` — is structural (it's how two tables in
   `required_tables` get connected, per `required_relationships`), not a
   row-filtering decision, and must never be flagged just because a column
   name is followed by `=` somewhere in the `FROM` clause. Scanning the
   *whole* SQL string here — rather than only the WHERE clause — was a real
   bug in an earlier version of this check: any query joining
   `census_msa`↔`cbsa_counties` (i.e. every MSA-level NRI or tract-population
   question) was rejected outright, because the join predicate's text
   satisfied the same regex a WHERE-clause filter would. The fix is
   structural, not a special case for that one column: the scan only ever
   looks at WHERE-clause text now, so it can't be fooled by a JOIN clause
   again for *any* future column.

A validation failure is non-fatal for the turn: it sets `repair_note` (fed
into the next `generate_sql` call as `REPAIR CONTEXT`, see §3.4) and the
attempt loop continues, up to 3 attempts total (`for attempt in range(3)` in
`run_code_query`).

### 3.6 Deterministic fallback compilation

`_compile_sql_from_plan(query_plan)` — no LLM, ever. Its docstring states
its scope precisely: *"This is a safety-net only: it runs when the LLM SQL
generator returns no SQL... It never changes routing; it uses the tables,
relationships, operation, projection, grouping, ordering, filters and live
entities already selected by the metadata planner."*

**Trigger condition**, from `run_code_query`: on any attempt, if
`generate_sql()` raises (empty output) or its output fails
`_validate_sql_against_plan`, *and* no earlier attempt in this call produced
any SQL at all (`last_sql` still empty), the compiler runs and its output is
immediately re-validated with the same §3.5 layer-2 check before being
executed. It is never used to override or "improve" SQL the LLM already
produced successfully.

**What it builds**, entirely from `QueryPlan` fields, with no free-text
judgment involved:

- **Projection** — `COUNT`/`AVG`/`SUM`/`MEDIAN`/`MIN`/`MAX` wrap `expr`
  directly for aggregate operations; a `rank` operation uses
  `expr`/`aggregation` as-is, since some catalog rank operations already
  carry a full aggregate expression (MSA/NRI-style: `"name, AVG(x)"`) and
  others carry a bare per-row projection (sold-home-style: `"city,
  sold_price"`).
- **GROUP BY** — only emitted when the projection actually contains an
  aggregate call (`AVG`/`SUM`/`COUNT`/`MEDIAN`/`MIN`/`MAX`, checked via
  `_AGGREGATE_CALL_RE`). Pairing a grouping key with a bare, non-aggregated
  column — which several catalog rank operations legitimately do, since not
  every "group by city" is really an aggregation — would otherwise produce
  a `GROUP BY` DuckDB rejects outright (*"column must appear in the GROUP BY
  clause or be part of an aggregate function"*); this check exists
  specifically so that shape compiles to valid, correct SQL instead.
- **Joins** — walks `required_tables`/`required_relationships` via the same
  graph as §3.2, and table-qualifies each relationship's key expression with
  `_qualify_join_expr`, which qualifies every bare identifier individually
  rather than prefixing the whole expression string. Most relationship keys
  are a single bare column, where either approach gives the same result; a
  few key on a compound expression (string concatenation, a `LEFT(...)`
  call), where naive whole-expression prefixing only qualifies the first
  token and leaves a later bare column ambiguous once a second joined table
  happens to share that column name, or turns a function call into invalid
  syntax. Per-identifier qualification handles both correctly and
  generically, for any future compound relationship key, not just the ones
  that exist today.
- **Predicates** — `_group_equality_predicates` combines same-column
  resolved-entity equalities (one per named entity — e.g. four separate
  `census_msa.name = '...'` predicates for a four-metro comparison) into a
  single `IN (...)`, rather than AND-joining them. AND-joining different
  literals on the same column is a logical contradiction (no row can equal
  two different values at once) and would make the query always return zero
  rows regardless of the data. Filters on different columns are still
  AND-ed together as before.
- **NULL exclusion** — for any concept whose `null_policy` includes
  `"exclude NULL"`, every dotted column reference in `select_expression`
  gets an `IS NOT NULL` predicate appended.

**What it explicitly does not do** — see §6 for the full list, but the
headline one: it compiles exactly the single `operation`/`select_expression`
the plan already carries. If a question needs two independent metrics (e.g.
flood risk *and* population), the compiler doesn't fan a plan out into two
queries — that decomposition is the orchestrator's job (§3.1), issuing two
separate `query_database` calls, each independently planned and compiled.

### 3.7 Execution

`store.query(sql)` against DuckDB. Results over 50 rows are truncated to the
first 50 with a `"... (N total rows, showing 50)"` note.

**Zero rows is not automatically treated as an error.** A validated,
successfully executed query that returns no rows is very often the correct,
factual answer (e.g. "does this MSA have a matching CBSA record" — genuinely
no, for an intentionally unmatched fixture MSA), not a broken join. The
first two attempts still treat it as worth a repair pass (`repair_note`
includes `schema.diagnose_empty_or_error(sql)` — a hint listing empty tables
or nearby relationship context, meant to help the *next* generation attempt
reconsider joins/filters). But the **final** message, once every attempt is
exhausted, states the fact plainly first — *"The query executed successfully
against the documented schema and returned 0 rows — no matching records were
found."* — with the diagnostic appended after, rather than returning only
the internal repair-hint text as if it were the answer. An earlier version
of this path returned only the raw diagnostic dump, which a response-writing
LLM (§3.8) had no reliable way to distinguish from "here's what you should
try fixing" versus "this is the final state" — and it doesn't have to make
that inference anymore, because the evidence now says which one it is.

A `Code Agent error: <message>` result means the whole attempt loop was
exhausted without ever producing a validated, executable query — this is
distinct from a clean zero-row result and reads that way in the evidence.

### 3.8 Final answer + response validation

**`_write_final_answer(message, history, evidence)`** — LLM, writing prose
from `FINAL_RESPONSE_PROMPT`'s instruction: *"Use ONLY the evidence produced
by the executed application functions. Do not invent facts, numbers,
routes, or map conclusions."* This call never sees raw database access —
only the rendered evidence string.

**`response_validator.py::validate_response(...)`** — deterministic,
runs after the reply is written. It's a backstop for two failure shapes,
neither requiring another model call to detect:

- The reply claims data (a Markdown table, or five-plus distinct numbers)
  while tool evidence shows failure signals — a short, hardcoded set of
  literal prefixes (`"error:"`, `"empty tables detected"`, `"sql error"`,
  `"binder error"`, `"0 rows"`, …) and whole-line phrases (`"cannot
  answer"`, `"no results returned"`, `"not loaded yet"`, …) checked against
  the tool output text.
- Tools returned real, successful data, but the reply doesn't reflect it.

### 3.9 Guardrails & Security Framework (Outlines + Pydantic)

The application incorporates a centralized open-source guardrail layer
(`services/guardrails.py`) powered by Outlines (`outlines==1.3.3`, `outlines_core==0.2.14`)
and Pydantic:

1. **Input Guardrail (`InputGuardrail`)**:
   - Screened at the entrypoint of `run_general_chat` and `run_house_chat`.
   - Regular expression and heuristic checks for prompt injection attacks,
     jailbreak prompts (e.g. DAN mode, developer mode), null bytes, and
     unauthorized system overrides.
   - Microsecond latency ($\sim 22\ \mu\text{s}$).

2. **Code Agent Guardrail (`CodeAgentGuardrail`)**:
   - Outlines schema / AST parser enforcing approved-function call constraints.
   - Prevents code generation containing `import`, control loops, or attribute
     access.
   - Microsecond latency ($\sim 15\ \mu\text{s}$).

3. **SQL Guardrail (`CodeAgentGuardrail::validate_sql`)**:
   - Restricts all generated SQL queries to read-only `SELECT` or `WITH` queries.
   - Blocks destructive or modifying operations (`DROP`, `DELETE`, `UPDATE`,
     `INSERT`, `ATTACH`, `LOAD`, etc.).
   - Microsecond latency ($\sim 17\ \mu\text{s}$).

4. **Output Grounding Guardrail (`OutputGroundingGuardrail`)**:
   - Enforces valid real estate metric boundaries (Walk Score, Bike Score,
     Transit Score in $[0, 100]$).
   - Replaces fabricated missing score claims with explicit "unavailable in data"
     disclaimers.
   - Microsecond latency ($\sim 9\ \mu\text{s}$).

Total combined guardrail overhead per user turn is $\sim 0.065\ \text{ms}$
($< 0.005\%$ of total request latency), introducing zero perceptible degradation.

---

## 4. Complete deterministic-vs-LLM inventory

| Stage | Location | Deterministic or LLM |
|---|---|---|
| Input sanitization & prompt injection guard | `services/guardrails.py::InputGuardrail` | Deterministic (Outlines / regex) |
| House vs. General routing | `main.py` (endpoint selection) | Neither — fixed by which UI panel called it |
| Concept / operation matching | `db/schema_catalog.py::semantic_matches`, `agents/query_planner.py::_select_operation` | Deterministic |
| Entity resolution | `db/schema_catalog.py::resolve_request_entities` | Deterministic, grounded in live DB values |
| Join-path selection | `db/schema_catalog.py::relationship_path` | Deterministic (Dijkstra over declared graph) |
| Orchestration program generation | `agents/general_agent.py::_generate_program` | LLM |
| Orchestration program sandboxing | `services/guardrails.py::CodeAgentGuardrail`, `agents/general_agent.py::_validate_program` | Deterministic (AST allow-list) |
| Orchestration continue/stop decision | `agents/general_agent.py` (`step_failed` check) | Deterministic (substring match) |
| Data-model retrieval (RAG) | `db/vector_store.py` | Deterministic retrieval, with lexical fallback |
| Data-model retrieval (targeted) | `db/schema_catalog.py::build_query_context` | Deterministic |
| SQL text generation | `agents/tools.py::generate_sql` | LLM |
| SQL output cleanup | `agents/tools.py::_clean_sql` | Deterministic |
| SQL security/shape check | `services/guardrails.py::CodeAgentGuardrail`, `agents/tools.py::validate_sql` | Deterministic (hard allow-list) |
| SQL plan-conformance check | `agents/tools.py::_validate_sql_against_plan` | Deterministic |
| SQL fallback compilation | `agents/tools.py::_compile_sql_from_plan` | Deterministic |
| SQL execution | `db/duckdb_store.py::query` | Deterministic (direct DB call) |
| Zero-rows / error messaging | `agents/tools.py::run_code_query` | Deterministic |
| Final answer prose | `agents/general_agent.py::_write_final_answer` | LLM |
| Reply-vs-evidence check & score bounds | `services/guardrails.py::OutputGroundingGuardrail`, `agents/response_validator.py` | Deterministic |

The only two LLM-driven steps in the entire pipeline are **SQL text
generation** and **final-answer prose**. Both are genuinely open-ended
generation tasks with no single correct output — which is exactly the shape
of problem an LLM should own. Everything else is either a lookup (does this
table/relationship/entity exist, per the catalog) or a check (does this
match what was already decided) — tasks with one correct answer, decided by
code.

---

## 5. Why the boundary is where it is

It's tempting to see the deterministic layers above as scaffolding to
eventually remove in favor of "the agent deciding more." In this pipeline
specifically, doing that would make it less capable, not more — for
concrete, checkable reasons, not stylistic ones.

**Every LLM role in this pipeline is the same model.** SQL generation
(`agents/tools.py::get_code_agent`), orchestration
(`agents/general_agent.py::_get_code_agent`), and final-answer writing
(`agents/general_agent.py::_get_response_agent`) all point at the identical
`settings.llama_server_model` — one local, quantized model playing three
roles in a single turn. A step whose entire purpose is to catch that model
getting something wrong — is this SQL safe, does it match the plan, does the
final answer match the evidence — can't be "ask the same model to check
itself." Self-grading isn't independent verification.

**Every one of those calls runs at `temperature=0.0`.** Sampling is
deterministic: retrying an unchanged prompt returns the same output. That's
not a hypothetical — the fallback compiler in §3.6 exists precisely because
this was observed directly: a request whose SQL generation fails once at
`temperature=0.0` fails identically on every subsequent attempt with an
unchanged prompt. A "smarter" fallback that's just another call to the same
model with the same information isn't a fallback for that failure mode at
all. (The 3-attempt retry loop in §3.5–3.6 *is* worth something, because
each retry's prompt genuinely changes — a repair note describing what went
wrong gets added before every retry. That's different from asking again with
nothing new.)

**Routing has to be independent of generation, or validation has nothing to
check against.** `_validate_sql_against_plan` (§3.5) exists to catch the
model deviating from an authoritative scope. If *scope itself* were an LLM
decision, that check would just be comparing one model call's opinion to
another's — there'd be no independent ground truth left in the loop.

**Security-relevant checks are a hard allow-list on purpose.**
`validate_sql` blocking anything but a single read-only `SELECT`/`WITH`, and
`_validate_program`'s AST walk blocking anything but approved function calls,
are guarantees. Replacing either with "ask the model if this looks safe"
trades a guarantee for a suggestion — for zero benefit, since the check
itself is nearly free to run.

**Retries are not free.** Every additional round-trip through `llama-server`
costs real, measured time. In this project's own evaluation runs, a question
that succeeds on the first attempt typically finishes in 12–45 seconds;
a question that exhausts a 3-attempt retry loop before falling back takes
well over two minutes. Adding an LLM call to a step that already has a
correct, instant, deterministic answer buys nothing and costs seconds to
minutes per request, every time.

None of this is an argument that the deterministic layers are permanent or
sacred — see §7 for how to extend them. It's specifically an argument that
*routing, safety checks, and the safety-net compiler* are the wrong places
to add LLM calls, because each one either has an already-correct
zero-latency answer, or exists specifically to catch the LLM being wrong,
and can't do that job by asking the LLM.

---

## 6. Known boundaries and sharp edges

Honest limitations of the current design, not open bugs:

- **One operation per plan.** `QueryPlan` carries exactly one
  `operation`/`select_expression`. A question needing two independent
  metrics (flood risk *and* population) isn't answered by one compiled
  query — it needs the orchestrator to issue two `query_database` calls in
  one step, which `CODE_AGENT_PROMPT` explicitly permits and encourages, but
  whether it actually happens depends on the orchestrating model's own
  judgment for that turn, not on a structural guarantee.
- **Entity resolution only grounds what already exists as live data.** A
  request referencing something the live-value matcher doesn't catch (a
  typo, a synonym the alias list doesn't cover) doesn't get a pre-verified
  literal to anchor the query. The result still passes through the same
  validators, but without an `entity_filters` literal behind it — whether
  the generated SQL still does something reasonable in that case depends on
  the model, not on a determinism guarantee.
- **The fallback compiler is only as correct as the catalog.** It faithfully
  compiles whatever `RELATIONSHIPS`/operation definitions say, including a
  wrong definition — it has no independent way to know a declared
  relationship or grain is mistaken. This isn't unique to the fallback path:
  the LLM path is handed the same catalog-derived plan text and inherits the
  same dependency. Catalog correctness is the one thing neither path can
  verify on its own.
- **A few checks are coupled to exact evidence wording.** The orchestration
  `step_failed` check (§3.1) and `response_validator.py`'s failure-phrase
  lists (§3.8) pattern-match specific substrings rather than a structured
  status field. Changing how a tool phrases a result elsewhere in the
  codebase should be checked against both of these, or a message that used
  to correctly signal "this needs a retry" (or "this is a real failure") can
  silently stop being recognized as one.
- **Orchestration is effectively 2 steps, not 3.** See §3.1 — the loop bound
  reads `CODE_AGENT_MAX_STEPS = 3`, but the exit logic only branches on
  `step == 0`; step 1 always exits. A question genuinely needing a third
  round of "evaluate evidence, gather more" doesn't get one today.
- **The fallback compiler doesn't invent join paths.** It can only build a
  join that `db/schema_catalog.py::RELATIONSHIPS` already declares between
  the tables a plan selected. A question needing two tables with no declared
  (direct or bridged) relationship path fails at planning time, before any
  SQL exists, for either the LLM or the compiler.

---

## 7. Extending the agent

For adding a new *data set*, see **Adding New Data Sets** in `README.md`: from
the browser (the Data page) or, for built-in sources loaded by script, a loader
plus an entry in `db/catalog_seed.py`. Since the catalog became a store (§8),
`db/schema_catalog.py` is a view over it and is not edited to add a source.

For extending the *agent's* behavior specifically:

- **A new aggregate/rank operation** — add it to the relevant concept's
  `operations=(...)` tuple in `db/catalog_seed.py` (built-in) using the existing
  `AVG`/`SUM`/`RANK_DESC`/etc. helpers. If it's a `RANK_DESC` with
  `group_by=`, decide up front whether the projection is a real aggregate
  (`"AVG(x)"`) or a bare per-row column (`"x"`) — §3.6 explains why that
  distinction determines whether `_compile_sql_from_plan` emits a `GROUP BY`
  correctly. Both shapes are already supported; you don't need new compiler
  code, just a catalog entry that matches one of the two existing patterns.
- **A new relationship** — add a `Relationship(...)` entry to `db/catalog_seed.py`,
  or approve one on the Data page. If its key is a
  single bare column on each side, no further consideration is needed. If
  either side is a compound expression (concatenation, a function call),
  `_qualify_join_expr` already handles per-identifier qualification
  generically — you don't need to hand-write the qualified form, and
  shouldn't: that's exactly the class of mistake §3.6 documents.
- **Don't hand-write a new SQL template outside `_compile_sql_from_plan`.**
  If a new question shape needs the fallback path to produce something the
  current compiler can't, extend the compiler's general logic (as §3.6's
  helpers already do for grouping, joins, and predicates), not a
  special-cased branch for one concept. A general fix protects every future
  catalog entry with the same shape; a special case only protects the one
  you're looking at.
- **Don't add an LLM call to catch another LLM call's mistake.** If you find
  a new failure mode in generated SQL or a generated program, the fix
  belongs in `_validate_sql_against_plan` or `_validate_program` — both
  already-existing, already-tested deterministic gates — not a new prompt
  asking the model to review its own output. §5 covers why.

---

## 8. The catalog is a store, and how it changes

### 8.1 One layer, in a data store

There is one catalog. Built-in sources and datasets added later are the same kind of object: rows in the
`catalog_*` tables of the application's DuckDB file (`catalog_store.py`). `origin` is provenance, not a
separate layer. `db/catalog_seed.py` only seeds the built-in rows; a seed refresh never overwrites a row
that was edited. `db/schema_catalog.py` keeps every public name the rest of the system uses and becomes a
thin in-memory view: `TABLES`, `RELATIONSHIPS`, `SEMANTIC_GLOSSARY` and `ENTITY_DOMAINS` are live
containers (module `__getattr__`) that load lazily, refresh in place, and reload when the connection is
replaced. The planner, SQL generator, both validators and the availability report are unchanged code
reading that view. A comparison of the store-backed module against the previous static module (order,
planner output, generated prompts, join paths, and a query built from every built-in alias) showed no
differences.

### 8.2 How a change reaches the agents

An approved change is written inside one `catalog_transaction()`: the physical DDL, the catalog rows and a
version bump commit together (or not at all), then the registry reloads once. Each agent turn already
reads the catalog fresh, so:

- **General Chat** sees new tables through the planner (`semantic_matches`, `relationship_path`), the
  SQL allow-list (`schema.list_table_names`), the availability report and a `USER-ADDED DATASETS` block.
- **House Chat** has a fixed, AST-validated function set, so it gets a dynamic prompt section and one
  approved function, `get_linked_dataset_records`, built from the live relationship graph
  (`schema.house_link_plan`).
- **Metadata retrieval** re-synchronizes the vector index when the catalog version changes and embeds only
  changed documents; with Ollama down it falls back to lexical search over the live catalog.

Two catalog facts exist so that new concepts cannot disturb old ones: `scope_guard: false` (measure columns
of generated concepts are legitimately filterable, so `_validate_sql_against_plan` does not treat them as
"unplanned filters") and `overrides` (a concept whose phrase extends another's declares precedence; the
planner honors it only when every phrase the overridden concept matched lies inside the overriding phrase).

### 8.3 Updating the catalog: deterministic-vs-model inventory

| Step | Who decides |
|---|---|
| Read a file, infer types, sanitize identifiers | deterministic |
| Detect key kinds (tract/county/ZIP FIPS, city, address, lat/lon) and guess roles | deterministic |
| Which columns of which tables might join, and after what normalization | deterministic (fixed set of transforms) |
| Whether they do join: match rate both ways, fan-out, cardinality, confidence | measured with SQL |
| Tract from coordinates, geocoding, derived key columns | deterministic (`geo_utils`, `geocoder`, SQL) |
| Column descriptions, units, synonyms, dataset title (**optional**) | model drafts; validators check; a person edits |
| Annotate a candidate with a plausibility note (**optional, off**) | model; annotation only |
| Approve a table or a link; change cardinality/preferred | a person |

This is the §5 boundary applied to onboarding: the model gets only what language is good for, its output is
constrained and validated, and it is never a gate.

### 8.4 Orchestration and model routing

The pipeline is a fixed sequence (read -> profile -> describe -> enrich -> analyze -> review -> publish), so
it is orchestrated as a deterministic workflow (`dataset_onboarding.py`), not an LLM planner or a
multi-agent supervisor: an LLM router would add a failure mode to a decision that is already known
statically. The only routing is by capability: `catalog_llm.ModelRouter` maps a tier (`draft`, `judge`) to an
ordered endpoint chain with health checks and fallback, serializes calls per endpoint, and returns
`ok=False` (never raises) when nothing works, in which case the workflow continues with rule-based defaults.
Because a person reviews every proposal, the cost of a poor draft is bounded by a correction, which is what
makes a small local model sufficient for this role; `python -m services.catalog_llm --selftest` measures
whether a given model is.
