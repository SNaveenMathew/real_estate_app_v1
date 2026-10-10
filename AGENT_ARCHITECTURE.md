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
(`agents/house_agent.py`) mirrors the same Code Agent architecture over a
smaller, house-scoped approved-function set — see that file's own module
docstring for the house-specific tool list; it shares the query-planning/
SQL pipeline described here (its `query_database` delegates straight to
`agents/tools.py::query_database`) but not every General Chat tool.
Both agents share one more thing described in §10: how a result, once
computed, decides whether it's shown as a table, a chart, or a map.

## Table of contents

1. [Design principle](#1-design-principle)
2. [Which agent handles a request](#2-which-agent-handles-a-request)
3. [General Chat: the full pipeline](#3-general-chat-the-full-pipeline)
   - 3.0 [Input guardrails — deterministic](#30-input-guardrails--deterministic)
   - 3.1 [Orchestration loop](#31-orchestration-loop)
   - 3.2 [Query planning (routing) — deterministic](#32-query-planning-routing--deterministic)
   - 3.3 [Data-model retrieval — deterministic](#33-data-model-retrieval--deterministic-two-distinct-steps)
   - 3.4 [SQL generation — LLM](#34-sql-generation--llm)
   - 3.5 [SQL validation — deterministic, two layers](#35-sql-validation--deterministic-two-layers)
   - 3.6 [Deterministic fallback compilation](#36-deterministic-fallback-compilation)
   - 3.7 [Execution](#37-execution)
   - 3.8 [Final answer + response validation](#38-final-answer--response-validation)
   - 3.9 [Guardrails & Security Framework](#39-guardrails--security-framework)
4. [Complete deterministic-vs-LLM inventory](#44-complete-deterministic-vs-llm-inventory)
5. [Why the boundary is where it is](#5-why-the-boundary-is-where-it-is)
6. [Known boundaries and sharp edges](#6-known-boundaries-and-sharp-edges)
7. [Extending the agent](#7-extending-the-agent)
8. [The catalog is a store, and how it changes](#8-the-catalog-is-a-store-and-how-it-changes)
9. [Commute times](#9-commute-times)
10. [Presentation layer: tables, charts, and maps](#10-presentation-layer-tables-charts-and-maps)
11. [Answer routing, derived measures, and non-answers](#11-answer-routing-derived-measures-and-non-answers)

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
            or answer_status.is_retryable(result)       # NOT_ANSWERED[unplannable | place_missing], see §11.5
        )
        for _, result in calls
    )
    if not any(name in {"query_database", "find_bike_route", "search_all_house_descriptions"} for name in names) or step_failed:
        continue
break
```

If step 0 produced no relevant data-fetching call, or one of those four
hardcoded phrases - or a *retryable* `NOT_ANSWERED[...]` status (§11.5) - appears in a result, the loop retries once
(step 1). Non-retryable statuses (an ambiguous place, an unloaded topic) are final: only the user can fix them.
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

A `NOT_ANSWERED[...]` status (§11.5) is never counted as data, and a reply that presents a table or many numbers after one
is replaced by the tool's own stated reason - not the generic "tables not loaded" message.

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
| Presentation classification (table/chart/map/none) | `agents/artifacts.py::classify_dataframe` | Deterministic, from the executed result's shape |
| Final answer prose | `agents/general_agent.py::_write_final_answer` | LLM |
| Reply-vs-evidence check & score bounds | `services/guardrails.py::OutputGroundingGuardrail`, `agents/response_validator.py` | Deterministic |

The only two LLM-driven *generation* steps in the entire pipeline are **SQL
text generation** and **final-answer prose** — genuinely open-ended tasks
with no single correct output, which is exactly the shape of problem an
LLM should own. The orchestration program the LLM writes may also set a
`presentation` hint on `query_database` (§10), but that's a bounded choice
among four fixed strings, not open-ended generation, and — like every other
LLM output in this pipeline — it never gets the final word: `classify_
dataframe` decides the actual shape from the data regardless of the hint.
Everything else is either a lookup (does this table/relationship/entity
exist, per the catalog) or a check (does this match what was already
decided) — tasks with one correct answer, decided by code.

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
- **A few checks are still coupled to exact evidence wording.** Non-answers now carry a structured
  `NOT_ANSWERED[code]` status (§11.5) that the orchestration `step_failed` check (§3.1) and
  `response_validator.py` (§3.8) read directly; but SQL errors, zero-row results and the rest of the
  failure-phrase lists still pattern-match specific substrings rather than a structured status field. Changing how a tool phrases a result elsewhere in the
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
- **A measure SQL cannot compute, a known-missing topic, or place-like labels** - declare it in catalog metadata
  (`derived`, `gap`, `requires_entity`, `match_mode="components"`) and, for a derived measure, register a provider. See
  §11.3-§11.8 for the knobs, the provider protocol and the checklist; none of it needs a change to the planner or tool layer.
- **A new approved function that should be able to produce a table, chart,
  or map** — see §10. In short: once you have a `pandas.DataFrame` (or can
  build a small one from whatever structured payload the function already
  returns), call `agents.artifacts.classify_dataframe(df, title=..., presentation=...)`
  and `agents.artifacts.emit_artifact(...)` with the result. Both agents
  already `reset_artifacts()`/`collect_artifacts()` once per turn, so a
  new tool needs no other wiring. Don't hand-write a bespoke map/chart
  payload shape for a new tool the way `find_bike_route` predates this and
  still does (§10) — `classify_dataframe` is the general path now, and a
  one-off shape only benefits the one tool you're looking at.

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
- **The map's layer panel** (`services/map_layers.py`, no LLM involved) classifies every agent-visible
  table by its columns and lists it as a marker/heat/line/choropleth layer, or leaves it out - the same
  "read the live catalog, not a fixed list" approach as the three consumers above, applied to
  visualization rather than chat. See "Map layers" in `README.md`.

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

---

## 9. Commute times

### 9.1 No LLM anywhere in the computation

Commute estimates are deterministic: an address lookup, routing requests, and arithmetic. The LLM only ever sees
the stored numbers, through the same channels as every other fact (the planner and SQL path in General Chat, a
function result in House Chat), so the §5 boundary is unchanged.

```
work address (typed | "lat, lon" | map click)
  -> geocode_work_address()   Census one-line geocoder, then Nominatim              [HTTP, worker thread]
  -> app_settings["work_location"]                                                  [DuckDB]
  -> refresh job (asyncio task, single flight)
       per chunk of ~90 houses, per mode, in a worker thread:
         OSRM /table  (sources = houses, destination = work)  -> falls back to /route per house
         OpenTripPlanner (optional, per house)
       straight-line caps skip absurd modes (walk > 6 mi, bike > 30 mi, drive > 250 mi)
       rows written on the loop thread: INSERT OR REPLACE INTO house_commute
  -> house_commute  (one row per house, keyed to the CURRENT work location by work_key)
```

`work_key` hashes the destination, the enabled modes, the server URLs and the drive factor. A row whose key does not
match the current inputs is *stale*: the map and the summary ignore it (numbers for a previous work location would
mislead), and the tab offers **Recompute**. This is why changing any input needs no migration and no manual cleanup.

### 9.2 Threading

As in the rest of the application, the single DuckDB connection is used only on the event-loop thread. Only pure-HTTP
work (geocoding, routing) runs in worker threads, so a slow public server never freezes the map or the chats. The job
is single-flight (`start_refresh` claims it synchronously), always leaves the `running` state (errors are recorded as
job state, never raised), and reports *why* a mode came back empty (`client.last_error`), so a dead bike server is a
visible warning instead of silently missing numbers.

### 9.3 How the chats reach it

- **General Chat**: `house_commute` is an ordinary built-in catalog table (`db/catalog_seed.py`), related to `houses` by
  `house_id`, with five concepts (drive, bike, walk, transit, distance). Rank operations carry
  `group_by="houses.address"`, so a ranking returns houses. Mode concepts declare `overrides: ["house_commute_drive"]`
  (computed from alias overlap when the seed is built), so "bike commute" is not also read as the generic word
  "commute". Nothing in the planner is commute-specific; a test asserts that no existing alias, in any sentence
  shape, started selecting a commute concept.
- **House Chat**: a new approved function, `get_commute_info()`, returns the estimates, the work location, whether
  they are up to date, and a plain statement of the basis (free-flow, no traffic). The dynamic linked-datasets prompt
  section from section 8 is now numbered 8.
- Why a separate table rather than columns on `houses`: the loader inserts into `houses` positionally, so new columns
  would break it; and a table keyed by house carries its own freshness (`work_key`, `computed_at`, `status`).

### 9.4 Privacy and honesty

The work location and house coordinates go to the configured routing servers; the API reports whether each is public
(`is_public_url`) so the UI can say so, and self-hosting OSRM keeps everything local. Drive, bike and walk are
free-flow estimates and are labeled as such everywhere they appear. Transit needs a self-hosted OpenTripPlanner and is
tested only against a mock.

---

## 10. Presentation layer: tables, charts, and maps

### 10.1 What problem this solves

Before this, exactly one tool produced anything other than reply text:
`find_bike_route`, whose route/crime-density payload was fished out of the
tool-call trace by two bike-specific functions
(`_parse_bike_payloads`/`_extract_bike_visualization`, since removed) and
handed to the frontend as a single `visualization` object. Every other
analytical answer — `query_database`'s result set included — only ever
became prose, and a multi-row result relied on the *LLM* hand-formatting a
markdown table into that prose: exactly the class of task §1 says has one
correct shape and so belongs to code, not the model.

`agents/artifacts.py` generalizes the one working case into a contract any
approved function can use, on the same LLM/code boundary as the rest of
this document: the LLM may *hint* how a result should be shown; code
decides what it actually *is*.

### 10.2 The contract

```
approved function executes                          [LLM decided to call it]
  -> a DataFrame (or small structured payload) is on hand   [deterministic]
  -> classify_dataframe() looks at its actual shape --
     row/column counts, dtypes -- and returns one of
     {table, chart, map, None}, optionally steered by a
     `presentation` hint ("map"/"chart"/"table"/"auto")      [deterministic]
  -> emit_artifact() appends it to this turn's list          [deterministic]
  -> collect_artifacts() hands the whole list to main.py
     once the turn is done, as `artifacts` in the API
     response (POST /api/chat, POST /api/house/{id}/chat)    [deterministic]
```

`reset_artifacts()`/`collect_artifacts()` are called once per turn, in
`run_general_chat` and `run_house_chat` respectively — the same functions
that already reset/read their per-turn `all_calls` trace. In between, any
approved function may call `emit_artifact()` zero or more times as a side
effect of its normal work; most turns emit nothing, some emit one artifact,
and a crime-aware bike route emits two (the crime-density map, then the
route map — the same order the original bike-specific code produced).

### 10.3 The classifier (`classify_dataframe`)

Given a `DataFrame`, in order:

1. **Map** — if the result has a recognizable latitude/longitude column
   pair (by name: `lat`/`latitude` × `lon`/`lng`/`longitude`, etc.), it's a
   map, regardless of hint. Rows with a null coordinate are dropped; a
   label column is picked by name priority (`address`, `name`, `city`, …,
   falling back to the first remaining column); every other column becomes
   a popup field.
2. **Chart** — one non-numeric "dimension" column (two, folded into a
   combined category label, if `presentation="chart"` was hinted) plus one
   to four numeric "measure" columns (six if hinted), with the row count
   under a cap (`config.py::presentation_chart_max_points[_hinted]`),
   becomes a chart: `line` if the dimension is a datetime column or named
   like one (`year`, `month`, `quarter`, …), `bar` otherwise. A
   numeric-*dtype* column named like a time period (an integer `year`
   column, say) is still treated as a dimension, not a measure — nobody
   wants a bar chart averaging years.
3. **Table** — anything else with more than one row.
4. **None** — an empty result, or a single summary row/value (reads better
   as prose than a one-row table) — *unless* `presentation="table"` was
   explicitly hinted, which always wins, including for one row.

A `presentation` hint never fabricates a shape the data can't support: a
`"map"` hint with no coordinate columns anywhere in the result falls
through to chart, then table, exactly as `"auto"` would
(`tests/test_artifacts.py::test_classify_dataframe_map_hint_falls_back_
when_no_coordinates`). Every call is wrapped in a bare
`except Exception: return None` — a bug in this layer degrades to "no
artifact" for that turn, never a broken reply.

### 10.4 Where it's wired in today

| Function | What it emits |
|---|---|
| `agents/tools.py::query_database` (shared by both agents — House Chat's own `query_database` closure delegates straight to it) | `classify_dataframe` on the executed result, honoring an optional `presentation` kwarg the orchestrating LLM may pass |
| `agents/tools.py::find_bike_route` | Its existing route/crime-density payload, normalized into `{"type": "map", "map_kind": "bike_route" \| "bike_crime_analysis", ...}` — the routing/crime logic in `services/bike_routing.py` is untouched; only the outer envelope is new |
| `agents/tools.py::check_data_availability` | A small table of table names and row counts, from the same `counts` dict `schema.availability_report()` already computed and previously discarded |
| `agents/tools.py::search_all_house_descriptions` | A table of vector-search matches (house, doc type, excerpt) |
| `agents/house_agent.py::get_nri_risk_data` | A bar chart of the top hazards by risk score |
| `agents/house_agent.py::get_nearby_sold_homes`, `::estimate_price_with_code` | Table(s) from the same comparable-sales DataFrames already built for the prose answer |

`query_database` is the one to reach for when adding a new source of
table/chart/map output: it's already wired into both agents, so a new
built-in dataset or catalog entry gets the presentation layer for free,
with nothing in `agents/artifacts.py` to touch.

### 10.5 The `presentation` hint, and why final-answer prose stopped asking for markdown tables

`query_database(request, requirements="", plan="", presentation="auto")`'s
new parameter is documented in `CODE_AGENT_PROMPT` (General Chat) and the
house-scoped equivalent as a plain instruction: set it to
`"map"`/`"chart"`/`"table"` only when the user's own words ask to see the
result that way, leave it `"auto"` otherwise. Getting it wrong costs
nothing (§10.3) — this is a hint, not a decision, the same relationship
the rest of this document draws between every other LLM output and the
deterministic check downstream of it (§5).

Both final-answer prompts (`FINAL_RESPONSE_PROMPT` in `general_agent.py`,
`_HOUSE_FINAL_RESPONSE_PROMPT` in `house_agent.py`) previously asked the
model to format a markdown table itself when comparing several rows. That
instruction is gone: the application now renders the full result set as
its own artifact next to the reply, so the model just states the key
figures in prose, in the order the evidence gives them (a requested
ranking stays in ranked order — `eval/golden_set.py`'s `order_matters`
cases only ever check for a ranked list of *names*, never markdown table
syntax, so this doesn't weaken that scoring). This removes a case where
the small local model was trusted to transcribe numbers into correct
pipe/dash syntax with no downstream check on whether it did, replacing it
with a deterministic table built directly from the same DataFrame the
prose is describing.

### 10.6 Frontend

`static/app.js`'s `renderArtifactsInChat(messageEl, artifacts)` is the one
dispatcher, keyed on `artifact.type`/`map_kind`: `renderTableArtifactInChat`
and `renderChartArtifactInChat` (bar/line as inline SVG — no charting
library, consistent with the rest of the frontend's minimal-dependency
approach) are new; `renderPointsMapArtifactInChat` is new and handles a
generic `query_database` map result (a small embedded Leaflet map per
`makeEmbeddedLeafletMap`, one marker per point, a popup per row);
`renderBikeCrimeAnalysisInChat`/`renderBikeFinalRouteInChat` predate this
work and are unchanged — they're reached through the generic dispatcher
now instead of bike-specific logic living in `sendGeneralMessage`.
`sendHouseMessage` calls the same dispatcher, so a table/chart artifact
from a house-scoped tool renders exactly the way a General Chat one does.

### 10.7 Known boundaries and sharp edges

- **Name-based heuristics, not semantic ones.** A numeric column whose
  name doesn't contain a temporal hint (`year_built`, say, rather than
  `year`) is treated as a measure even where a person might read it as a
  dimension; a column genuinely named like a time period but semantically
  a duration (`years_on_market`) is treated as a dimension it isn't. Both
  degrade to a safe, if unexciting, table rather than a wrong chart — never
  the reverse.
- **One dimension in `"auto"` mode, two if `presentation="chart"` is
  hinted.** `{city, avg_price}` charts automatically; `{city, year,
  avg_price}` only charts (folding `year` into the category label as
  `"Pittsburgh / 2023"`) if the agent set the hint. This mirrors §6's
  general stance: the automatic path stays conservative, and an explicit
  ask unlocks more.
- **A single-row, many-column result never becomes a small key/value
  table** (§10.3, point 4). `get_house_details`-shaped answers — "every
  fact about this one house" — still rely on prose, same as before this
  work. Worth revisiting if that shape becomes a common ask.
- **The size caps** (`config.py`: `presentation_table_max_rows`,
  `presentation_chart_max_points[_hinted]`, `presentation_map_max_points`)
  are independent of the ~50-row cap on the text evidence shown to the
  final-answer LLM in `run_code_query` — an artifact can show more rows
  than the model ever reads, since the model only needs enough evidence to
  describe the pattern, not to enumerate a table it isn't drawing anymore.


---

## 11. Answer routing, derived measures, and non-answers

### 11.1 What problem this solves

"Verify the land areas of Pittsburgh and Indianapolis" used to fail three ways at once. The catalog had no concept for
land area, so the plan was empty. An empty plan can never succeed - `_validate_sql_against_plan` rejects every statement
that does not use a planned table - yet `run_code_query` still asked the SQL model three times. And the final text
(`Code Agent error: Cannot compile SQL: plan selected no tables.`) was summarised by the answer model as "the database
query encountered an error", although nothing was ever sent to the database.

The fix is deliberately not "add a land-area branch". Each layer got one small, generic mechanism, and census is simply
the first user of them:

| Layer | Mechanism | Where |
|---|---|---|
| Catalog | concept knobs (`requires_entity`, `gap`, `derived`, `derived_support`, `LOOKUP`) | §11.3, `db/catalog_model.py` |
| Entity matching | `match_mode="components"` place lexicon | §11.4, `db/entity_lexicon.py` |
| Planner | token-index phrase matching; `lookup` fallback operation | §11.6, `db/text_match.py`, `agents/query_planner.py` |
| Tool | answer policy; unplannable short-circuit | §11.2, `agents/answer_policy.py`, `agents/tools.py` |
| Orchestrator / validator | `NOT_ANSWERED[code]` statuses | §11.5, `agents/answer_status.py` |
| Derived numbers | provider registry | §11.7, `services/derived_measures.py`, `services/census_metrics.py` |

### 11.2 The answer policy

`query_database` asks `agents/answer_policy.py` three questions, all answered from the `QueryPlan` and catalog metadata
alone (the module contains no dataset vocabulary):

1. **`route_before_sql`** - must this request be handled *without* the SQL model?
   - a place name fits several places (`ambiguous` entities, §11.4) -> ask the user which one;
   - a matched concept declares `derived` -> run the registered provider (§11.7);
   - only `gap` concepts matched -> state that the topic is not loaded and estimate nothing.
2. **`unplannable_result`** - nothing in the catalog matched. Return a status (not an error) naming any place that *was*
   recognized and measures the data covers (drawn from the live catalog, so new datasets appear automatically), and
   report empty tables that name matching depends on - an empty `census_msa` silently disables every metro question.
   No model call is made.
3. **`caveats_after_sql`** - a real answer was produced but part of the request touches a `gap` topic: append the
   caveat so a partial result is never presented as complete.

Before planning gives up, `query_database` also plans from `request + requirements`: the orchestrator often states the
measure only in `requirements` ("Return one row per metro with its land area") while `request` just names places.

### 11.3 Concept knobs

Optional keys on a concept (`db/catalog_model.CONCEPT_KNOBS` is the single documented list; `db/catalog_lint.py`
rejects unknown keys, because a misspelled knob silently does nothing):

| Knob | Meaning | Use it when |
|---|---|---|
| `requires_entity: ["MSA"]` | applies only if the request names a live value of that entity type | natural phrasing ("population of ...") that must not capture unrelated questions |
| `gap: True` | a known-unavailable topic; no tables; its description is the message shown to the user | you know users will ask for something the data does not hold |
| `derived: {provider, measures}` | measures computed by a registered provider, not SQL | polygon areas, routing times, model scores |
| `derived_support: {provider, measures}` | the provider *can also* supply this concept's measure when it is already running | an SQL concept whose number should ride along in a derived answer |
| `overrides: [...]` | concepts to drop when all their phrases sit inside this concept's longer ones | your phrase extends an existing alias |
| `scope_guard: False` | keep the concept's columns out of the unplanned-filter guard | the columns are legitimately filterable |
| `LOOKUP(expr)` operation | plain "show these columns for the named entity"; chosen only when no aggregate/rank phrase matched | simple lookups the fallback compiler should answer deterministically |

A `gap` concept **yields automatically** to any real concept that matches overlapping words, so loading a dataset that
covers the topic silences the gap with no edit (and retiring the dataset restores it) -
`tests/test_future_dataset_extensibility.py` proves both directions.
The Data page's onboarding honors this too: `_known_aliases()` in `services/dataset_onboarding.py` skips `gap` concepts, so the
dataset that finally covers the topic can claim the same phrases. (Without that, onboarding would report its aliases as
"already used by ..." and create no concept at all - the test drives the real `build_concepts` to keep it that way.)

**Alias hygiene.** Onboarding refuses to let an uploaded dataset claim an alias a built-in concept already owns, so every
alias a built-in concept claims is a word no future dataset can use. Built-in aliases must therefore be specific phrases
("land area", "how many people live in"), never bare generic words ("population", "area", "price"). The lint enforces the
same rule onboarding uses (`_GENERIC_ALIASES`, single tokens shorter than six characters).

### 11.4 Entity matching and `match_mode="components"`

`match_mode` is a property of an entity domain. `exact` (tract FIPS) is resolved only by the literal-number pass;
`exact_or_prefix` and `prefix` behave exactly as before. `components` (`db/entity_lexicon.py`) understands labels of the
form `"City-City--City, ST-ST Metro Area"`:

- each city of the label is a lookup key (principal city = tier 1, others = tier 2), plus the whole label, plus
  state-qualified forms (`portland me`, `portland maine`); `Fort`/`Saint`/`Mount` and `Ft`/`St`/`Mt` are interchangeable;
  accents are folded;
- a phrase that fits several labels is **ambiguous** unless one candidate is strictly better by (tier, metro-before-micro).
  Ambiguity is reported (`ambiguous: True`, `ambiguous_with`, `phrase` on every candidate) and the policy asks the user -
  it is never guessed (the old matcher returned every same-named metro in one unflagged list, so a SUM silently added them);
- everyday words that are also places (`mobile`, `reading`, `bend`, `normal`, `orange`) match only when state-qualified;
- the lexicon is active for a domain only when a matched concept lists the domain's entity type in `entity_types`
  (metro/MSA wording also activates it). Elsewhere `components` behaves exactly like `prefix`, so unrelated requests
  resolve - and render in the plan text - exactly as they always did;
- lexicons are cached by the *content* of the live values, so new data is picked up automatically and a stale lexicon
  cannot be served.

Any column of `"Name, ST"` labels can opt in by declaring the mode; `catalog_lint` suggests it when it sees such values.

### 11.5 Answer statuses

When a request cannot be answered, the result is `NOT_ANSWERED[<code>]: <message safe to show the user>`
(`agents/answer_status.py`) instead of free text that every layer has to pattern-match:

| Code | Meaning | Retried by the orchestrator? |
|---|---|---|
| `unplannable` | nothing in the data model matched; no SQL generated | yes - a reworded request might match |
| `place_missing` | a derived measure needs a named place and the request has none | yes |
| `place_ambiguous` | a name fits several places | no - only the user can choose |
| `topic_unavailable` | a `gap` topic; nothing may be estimated | no |
| `data_unavailable` | needed data is missing or too incomplete in the database | no |
| `limit_exceeded` | larger than one call supports | no |

Consumers: the orchestrator's `step_failed` retries only retryable codes (§3.1); `response_validator` never counts a
status as data (whatever digits its explanation holds), and when the reply then presents a table or a pile of numbers it
replaces the reply with the tool's own reason - not the generic "tables not loaded, run setup_data.py" message;
`_extract_query_result_fallback` shows the user the plain message, never the marker. Messages contain no digits (tests
enforce it): the validator's data heuristic is digit-based. To add a code, add it to `agents/answer_status.py`
(`CODES`, and `RETRYABLE` if rewording can help) and use `not_answered(code, message)`.

### 11.6 The planner's performance contract

The planner runs several times per chat turn, and each run asks one question thousands of times: *does this normalized
alias occur, as whole words, in the request?* It used to build one regular expression per alias and per live entity value.
Python caches a few hundred compiled patterns, so at realistic data size nearly every call recompiled thousands of them
(about 3,400 per warmed call in the benchmark) - the dominant fixed cost of every turn, growing with every dataset added.

`db/text_match.py` answers the same question from a per-request n-gram index, so cost is O(request) however large the
catalog or the data. Semantics are identical to the regex, including its non-overlapping-match rule that the `overrides`
logic relies on - `tests/test_text_match.py` proves it differentially against the original implementation.

Rules that keep it that way:

- never build a regular expression from request text or live values on the planning path;
- gate anything expensive behind something cheap the catalog declares (`requires_entity` is evaluated only after an
  alias already matched, and after `excluded_terms`);
- import heavy modules lazily (geopandas is loaded only when a derived measure is actually computed);
- `tests/test_planner_hot_path.py` enforces the first three deterministically (a regex-compile counter run at a data
  scale large enough to exceed the regex cache - at fixture scale even the old matcher shows zero compilations).

### 11.7 Derived-measure providers

A derived measure is a number SQL over the stored tables cannot produce. A provider is a module exposing `PROVIDER`
(`services/derived_measures.py` documents the protocol), registered by id; concepts name it with `derived`.
Adding one needs no change to the planner, tool, policy, validator or orchestrator
(`tests/test_future_dataset_extensibility.py::test_a_new_derived_provider_plugs_in_through_metadata_alone`).

The built-in provider `msa_geometry` (`services/census_metrics.py`) computes population, land area and density for named
metro areas:

- member counties come from `cbsa_counties` (by CBSA code, or by normalized title for an unresolved `X...` code);
  tract populations from `census_tracts`; polygons from `services/geo_utils`, measured in an equal-area projection;
- every figure is computed over the **footprint** - tracts with both a population and a polygon - so population, area and
  density always reconcile, and any excluded tract is reported. Under 95% tract coverage or 98% population coverage is a
  `data_unavailable` status with the likely cause (different geometry and census vintages), never a quietly wrong number;
- each tract is measured at most once per loaded geometry object (caches follow the object, so reloading geometry cannot
  serve stale areas; lock-protected), and database reads go through a private cursor;
- a provider can recognise the other measures named in the same sentence once a specific concept has selected it
  (`MeasureInfo.phrases`), so "population, land area and density of Pittsburgh" is one call.

### 11.8 Adding a dataset: checklist

1. Load the data and register tables/concepts (Data page, or `db/catalog_seed.py` for built-ins).
2. Choose aliases that are specific multi-word phrases (§11.3).
3. Not expressible as SQL -> write a provider (§11.7). Known-missing topic -> a `gap` concept. Place-like labels ->
   `match_mode="components"` and `requires_entity`.
4. `python -m db.catalog_lint` - no errors, and no new warnings (the ratchet in `tests/golden/catalog_lint_baseline.json`).
5. Add two or three representative questions to `tests/plan_battery_queries.py`, then `python scripts/plan_battery.py --check`.
   Any changed plan outside your own dataset is a collision between concepts: fix the aliases (or add `overrides`) rather
   than accepting the diff. `--update --groups <group>` records an intended change.
6. `python -m pytest tests/test_catalog_contract.py tests/test_plan_battery.py tests/test_planner_hot_path.py`.

### 11.9 Known boundaries and sharp edges

- **Metro scope, not city limits.** A city name resolves to the metro/micropolitan area that contains it; figures describe
  the whole metro area. The answer says so. There is no city-boundary geometry.
- **Only metro areas have derived measures.** No county, state or ZIP roll-ups yet; add a provider (or extend this one).
  Ranking *every* metro by area or density is declined (`MAX_METROS = 25`) rather than attempted.
- **"Land area" is polygon area.** If the tract polygons include water bodies the figure can exceed an official land-only
  (`ALAND`) value; the answer says so. For authoritative land area, load Gazetteer/TIGER `ALAND` and have the provider
  prefer it.
- **One shared database connection.** `db/duckdb_store.py` serves every thread from one connection, whose result state is
  not thread-safe; overlapping `store.query` calls can clobber each other. The derived provider reads through its own
  cursor; the planner and SQL execution do not. Giving `store.query` a lock or cursor would fix it for everyone.
- **Metro vocabulary in the resolver.** The words that put the resolver in metro context ("msa", "metro", "metro area" ...)
  are still a short list in `resolve_request_entities`; a second `components` domain activates through concept
  `entity_types` instead.
- **Concept-level overlap is judged by the planner, not the lint.** The lint reports equal aliases claimed twice; a phrase
  that merely *contains* another concept's alias needs `overrides` or the plan battery will show the collision.
- `map_layers.get_msa_population_density` is no longer called by the agents (the provider replaced it); it remains for
  compatibility and can be deleted once nothing else imports it.
