#!/usr/bin/env python3
"""
run_eval.py — agent evaluation pipeline.

Runs every example in eval/golden_set.py against the REAL agents
(agents/general_agent.py, agents/house_agent.py) on a dedicated evaluation
database (eval/fixtures.py), scores each one, and writes a report.

Supports both one-shot (`--phase all`, default) and separated two-pass runs so
the agent server and independent judge never compete for 16 GB VRAM:
    python run_eval.py --phase generate --run-id gemma-20261125
    python run_eval.py --phase judge    --run-id gemma-20261125
"""

import argparse
import atexit
import hashlib
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Windows consoles default to cp1252, which cannot encode the \u2713/\u2717 status glyphs printed per case.
# Reconfiguring is harmless on UTF-8 terminals (Linux CI included) and keeps a local run from aborting.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings
from eval.fixtures import build_fixture
from eval.golden_set import GOLDEN_SET, GoldenExample, get_example
from eval.scoring import ScoreResult, score_structured
from eval.judge import LLMJudge, MockJudge


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable_artifacts(artifacts) -> list[dict]:
    if not artifacts:
        return []
    try:
        raw = json.loads(json.dumps(list(artifacts), default=str))
        return raw if isinstance(raw, list) else []
    except Exception:
        return []


def _judge_with_artifacts(judge, example: GoldenExample, reply: str, artifacts: list[dict]) -> ScoreResult:
    try:
        return judge.judge(example, reply, artifacts=artifacts)
    except TypeError:
        return judge.judge(example, reply)


def _export_example_trace(
    ex: GoldenExample,
    *,
    trace_id: str | None,
    trace_url: str | None,
    traces_dir: Path,
    metadata: dict,
) -> tuple[str | None, str | None, int, str | None]:
    """Export one trace into the run-scoped immutable directory and return (path, sha256, span_count, error)."""
    trace_path = traces_dir / f"{ex.id}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if trace_id:
            from observability import export_trace

            payload = export_trace(
                trace_id,
                trace_path,
                metadata={**metadata, "trace_url": trace_url},
            )
            span_count = int(payload.get("span_count", 0))
        else:
            payload = {
                "trace_id": None,
                "span_count": 0,
                "metadata": {
                    **metadata,
                    "note": "No OpenTelemetry trace was emitted by this agent path.",
                },
                "spans": [],
            }
            trace_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            span_count = 0
        return str(trace_path), _sha256_file(trace_path), span_count, None
    except Exception as exc:
        return None, None, 0, str(exc)


def generate_example(example: GoldenExample, mock: bool, *, traces_dir: Path | None = None) -> dict:
    """Execute one golden example through the real (or mock) agent and capture reply, artifacts, usage, and trace."""
    t0 = time.time()
    trace_id = None
    trace_url = None
    metadata: dict = {}
    artifacts: list[dict] = []
    asked = 0
    scripted_used = 0
    ended_with_question = False
    reply = ""
    gen_error: str | None = None
    gen_traceback: str | None = None

    if not mock:
        try:
            from observability import clear_local_trace_capture, initialize_observability

            initialize_observability()
            clear_local_trace_capture()
        except Exception:
            pass

    if mock:
        from eval.mock_agent import mock_reply

        try:
            reply = mock_reply(example)
        except Exception as e:
            gen_error = f"mock agent failed: {e}"
            gen_traceback = traceback.format_exc()
    else:
        try:
            if example.agent == "general":
                from agents.general_agent import run_general_chat

                # A turn may legitimately end in a QUESTION (agents/clarification.py). If the example scripts the
                # user's reply, continue the conversation with it; otherwise the question is the result and is rated
                # as such. Each scripted reply is self-contained on purpose - planning is single-turn by design.
                pending = list(example.clarifications)
                turns = list(example.history)
                question = example.question
                while True:
                    reply, _, turn_artifacts, metadata = run_general_chat(
                        question, history=turns, include_metadata=True
                    ) or ("", [], [], {})
                    metadata = metadata or {}
                    artifacts = _jsonable_artifacts(turn_artifacts)
                    trace_id = metadata.get("trace_id") or trace_id
                    trace_url = metadata.get("trace_url") or trace_url
                    ended_with_question = bool(metadata.get("clarification"))
                    if not ended_with_question:
                        break
                    asked += 1
                    if not pending:
                        break
                    turns = turns + [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": reply},
                    ]
                    question = pending.pop(0)
                    scripted_used += 1
            else:
                from agents.house_agent import run_house_chat

                out = run_house_chat(
                    example.house_id,
                    example.question,
                    history=list(example.history),
                    include_metadata=True,
                )
                if len(out) == 4:
                    reply, _, turn_artifacts, metadata = out
                else:
                    reply, _, turn_artifacts = out
                    metadata = {}
                metadata = metadata or {}
                artifacts = _jsonable_artifacts(turn_artifacts)
                trace_id = metadata.get("trace_id")
                trace_url = metadata.get("trace_url")
                if not trace_id:
                    try:
                        from observability import latest_local_trace_id

                        trace_id = latest_local_trace_id()
                    except Exception:
                        trace_id = None
        except Exception as e:
            gen_error = f"agent call failed: {e}"
            gen_traceback = traceback.format_exc()
            try:
                from observability import latest_local_trace_id

                trace_id = latest_local_trace_id()
            except Exception:
                pass

    dt = time.time() - t0
    trace_file = None
    trace_sha256 = None
    span_count = 0
    trace_export_error = None
    if not mock and traces_dir is not None:
        trace_file, trace_sha256, span_count, trace_export_error = _export_example_trace(
            example,
            trace_id=trace_id,
            trace_url=trace_url,
            traces_dir=traces_dir,
            metadata={
                "eval_id": example.id,
                "agent": example.agent,
                "answer_type": example.answer_type,
                "tags": list(example.tags),
                "question": example.question,
                "reply": reply,
                "answer_status": metadata.get("answer_status"),
            },
        )

    return {
        "id": example.id,
        "agent": example.agent,
        "answer_type": example.answer_type,
        "tags": list(example.tags),
        "question": example.question,
        "reply": reply,
        "artifacts": artifacts,
        "clarification_rounds": scripted_used,
        "assisted": scripted_used > 0,
        "asked": asked,
        "ended_with_question": ended_with_question,
        "answer_status": metadata.get("answer_status"),
        "token_usage": metadata.get("token_usage") or {},
        "latency_s": round(dt, 3),
        "trace_id": trace_id,
        "trace_url": trace_url,
        "trace_file": trace_file,
        "trace_sha256": trace_sha256,
        "span_count": span_count,
        "trace_export_error": trace_export_error,
        "error": gen_error,
        "traceback": gen_traceback,
    }


def score_generation(example: GoldenExample, gen: dict, judge) -> ScoreResult:
    """Grade a single captured generation record with either assert_equal or the JudgeBackend."""
    reply = str(gen.get("reply") or "")
    asked = int(gen.get("asked") or 0)
    ended_with_question = bool(gen.get("ended_with_question"))
    scripted_used = int(gen.get("clarification_rounds") or 0)
    artifacts = list(gen.get("artifacts") or [])

    if gen.get("error"):
        result = ScoreResult(
            example.id,
            "ERROR",
            "n/a",
            str(gen["error"]),
            reply,
            extra={"traceback": gen.get("traceback")},
        )
    elif example.expect_clarification and not asked:
        result = ScoreResult(
            example.id,
            "FAIL",
            "assert_equal",
            "answered without asking for the missing measure, which this example expects to be asked",
            reply,
        )
    elif ended_with_question and not example.expect_clarification:
        result = ScoreResult(
            example.id,
            "FAIL",
            "assert_equal",
            "asked a clarifying question and no user reply was scripted for this example",
            reply,
        )
    elif example.answer_type == "structured":
        result = score_structured(example, reply)
    else:
        result = _judge_with_artifacts(judge, example, reply, artifacts)

    if ended_with_question and example.expect_clarification and not scripted_used and not gen.get("error"):
        result = ScoreResult(example.id, "PASS", "assert_equal", "asked the expected clarifying question", reply)

    result.extra.update(
        {
            "clarification_rounds": scripted_used,
            "assisted": bool(gen.get("assisted")),
            "asked": asked,
            "artifacts": artifacts,
            "answer_status": gen.get("answer_status"),
            "token_usage": gen.get("token_usage") or {},
            "latency_s": gen.get("latency_s", 0.0),
        }
    )
    for k in ("trace_id", "trace_url", "trace_file", "trace_sha256", "trace_export_error"):
        if gen.get(k) is not None:
            result.extra[k] = gen[k]
    return result


def run_example(example: GoldenExample, judge, mock: bool, *, traces_dir: Path | None = None):
    """Run and grade one golden example synchronously."""
    gen = generate_example(example, mock, traces_dir=traces_dir)
    return score_generation(example, gen, judge)


def _filter_examples(args) -> list[GoldenExample]:
    examples = list(GOLDEN_SET)
    if args.ids:
        wanted = set(args.ids.split(","))
        examples = [e for e in examples if e.id in wanted]
    if args.tags:
        wanted_tags = set(args.tags.split(","))
        examples = [e for e in examples if wanted_tags & set(e.tags)]
    if args.skip_house_agent:
        examples = [e for e in examples if e.agent != "house"]
    if args.house_agent_only:
        examples = [e for e in examples if e.agent == "house"]
    return examples


def _slug(text: str, *, limit: int = 60) -> str:
    """Filesystem-safe slug: ``DuoNeural/Gemma-4-26B:Q3_K_M`` -> ``duoneural-gemma-4-26b-q3-k-m``."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:limit].rstrip("-") or "model"


def _default_run_id(model: str) -> str:
    """Model-scoped run id, so a GLM or Qwen round can never overwrite this run's artifacts."""
    return f"{_slug(model)}-{time.strftime('%Y%m%d-%H%M%S')}"


def _resolve_run_dir(run_id: str, run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        return Path(run_dir_arg)
    return settings.eval_runs_dir / run_id


def _clear_caches() -> None:
    """Empty the persistent prompt/semantic/SQL caches and the in-process proposal cache."""
    from agents import semantic_fallback

    semantic_fallback.reset_cache()
    from services import cache as result_cache

    try:
        before = result_cache.stats()
        outcome = result_cache.invalidate()
        after = result_cache.stats()
    except Exception as exc:  # a missing cache table must not stop an evaluation
        print(f"warning: could not clear the result cache: {exc}", file=sys.stderr)
        return
    if not outcome.get("ok"):
        print(f"warning: could not clear the result cache: {outcome.get('error')}", file=sys.stderr)
        return
    print(
        "Cleared caches: "
        f"{before.get('total_entries', 0)} -> {after.get('total_entries', 0)} entries "
        f"(prompt={before.get('prompt', {}).get('entries', 0)}, "
        f"semantic={before.get('semantic', {}).get('entries', 0)}, "
        f"sql={before.get('sql', {}).get('entries', 0)})"
    )


def _aggregate_token_usage(items: list[dict]) -> dict:
    totals = {
        "llm_calls": 0,
        "cache_hits": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "llm_latency_seconds": 0.0,
        "estimated_cost_usd": 0.0,
    }
    for item in items:
        usage = item.get("token_usage") or {}
        for k in ("llm_calls", "cache_hits", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
            totals[k] += int(usage.get(k) or 0)
        totals["llm_latency_seconds"] += float(usage.get("llm_latency_seconds") or 0.0)
        totals["estimated_cost_usd"] += float(usage.get("estimated_cost_usd") or 0.0)
    totals["llm_latency_seconds"] = round(totals["llm_latency_seconds"], 4)
    totals["estimated_cost_usd"] = round(totals["estimated_cost_usd"], 6)
    return totals


def _write_manifest(
    run_dir: Path,
    meta: dict,
    generations: list[dict],
    results: list[tuple[GoldenExample, ScoreResult]] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    result_by_id = {ex.id: r for ex, r in (results or [])}
    entries: dict[str, dict] = {}
    for g in generations:
        ex_id = g["id"]
        res = result_by_id.get(ex_id)
        entry = {
            "agent": g.get("agent"),
            "answer_type": g.get("answer_type"),
            "answer_status": g.get("answer_status"),
            "latency_s": g.get("latency_s", 0.0),
            "trace_id": g.get("trace_id"),
            "trace_file": g.get("trace_file"),
            "trace_sha256": g.get("trace_sha256"),
            "span_count": g.get("span_count", 0),
            "token_usage": g.get("token_usage") or {},
        }
        if res is not None:
            entry["verdict"] = res.verdict
            entry["method"] = res.method
            entry["reason"] = res.reason
        entries[ex_id] = entry

    manifest = {
        "run_id": meta.get("run_id"),
        "phase": meta.get("phase"),
        "git_sha": meta.get("git_sha") or _git_sha(),
        "timestamp": meta.get("timestamp") or time.strftime("%Y%m%d-%H%M%S"),
        "command": meta.get("command", ""),
        "agent_model": meta.get("agent_model"),
        "judge_model": meta.get("judge_model"),
        "judge_independent": meta.get("judge_independent"),
        "fixture_path": meta.get("fixture_path"),
        "fixture_sha256": meta.get("fixture_sha256"),
        "complete_coverage": meta.get("complete_coverage"),
        "token_usage_totals": _aggregate_token_usage(generations),
        "entries": entries,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _write_generations(run_dir: Path, generations: list[dict]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "generations.jsonl"
    lines = [json.dumps(g, ensure_ascii=False, default=str) for g in generations]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _load_generations(run_dir: Path) -> tuple[list[dict], dict]:
    gen_path = run_dir / "generations.jsonl"
    if not gen_path.exists():
        raise FileNotFoundError(f"No generations.jsonl found in run directory: {run_dir}")
    generations = [json.loads(line) for line in gen_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return generations, manifest


def main():
    parser = argparse.ArgumentParser(description="Run the agent evaluation pipeline")
    parser.add_argument("--ids", help="comma-separated example ids to run (default: all)")
    parser.add_argument("--tags", help="comma-separated tags to filter by (default: all)")
    parser.add_argument(
        "--skip-fixture-build", action="store_true", help="reuse an already-built fixture DB instead of rebuilding it"
    )
    parser.add_argument("--list", action="store_true", help="list matching examples and exit")
    parser.add_argument(
        "--skip-house-agent",
        action="store_true",
        help="skip examples assigned to the house agent; evaluate General Chat only",
    )
    parser.add_argument(
        "--house-agent-only",
        "--only-house-agent",
        dest="house_agent_only",
        action="store_true",
        help="run only examples assigned to the house agent",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="smoke-test the harness with a scripted stub reply and a "
        "keyword-based judge — no model server required. "
        "NOT a real evaluation of the agent.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="permit a filtered run to exit 0 even though it skipped examples. "
        "Without this, any subset is reported as partial and exits 2.",
    )
    parser.add_argument(
        "--allow-self-judge",
        action="store_true",
        help="allow the model under test to judge its own answers. Off by default: a run where the "
        "judge is the agent's weights is not independent evidence.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="score the known-verdict calibration answers (eval/calibration.py) and exit; catches a "
        "scorer or judge that has drifted to always-pass. Add --mock for the deterministic half only.",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "generate", "judge"),
        default="all",
        help="run generation+scoring ('all', default), generation only ('generate'), or offline scoring ('judge').",
    )
    parser.add_argument(
        "--run-id",
        help="run identifier for immutable run/trace storage (default: <model-slug>-<timestamp>)",
    )
    parser.add_argument("--run-dir", help="explicit directory for run artifacts (defaults to eval/runs/<run-id>)")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="empty the prompt, semantic and SQL result caches (plus the in-process semantic proposal "
        "cache) before generating. The fixture database is rebuilt by default, so this matters when "
        "reusing one with --skip-fixture-build.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="K",
        help="run each example K independent times (default: 1). Reports pass@1 (at least 1 of K "
        "trials passed) and pass^K (all K trials passed). Semantic and prompt caches are disabled "
        "between trials so each is an independent sample.",
    )
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or (Path(args.run_dir).name if args.run_dir else _default_run_id(settings.llama_server_model))
    run_dir = _resolve_run_dir(run_id, args.run_dir)
    traces_dir = run_dir / "traces"

    if args.calibrate:
        judge = MockJudge() if args.mock else LLMJudge()
        from eval import calibration

        rows, ok = calibration.run(judge=None if args.mock else judge)
        print(f"{'case':<30} {'what':<50} {'got':<12} {'expected':<9} ok")
        for row in rows:
            print(
                f"{row['id']:<30} {row['what'][:50]:<50} {row['verdict']:<12} {row['expected']:<9} "
                f"{'yes' if row['ok'] else 'NO'}"
            )
        bad = [row for row in rows if not row["ok"]]
        print(f"\n{len(rows) - len(bad)}/{len(rows)} calibration answers agreed with their known verdict")
        sys.exit(0 if ok else 2)

    examples = _filter_examples(args)
    if not examples:
        print("No golden examples matched the given filters.")
        sys.exit(1)

    if args.list:
        for e in examples:
            print(f"  {e.id:<40s} [{e.agent}] [{e.answer_type}] tags={list(e.tags)}")
        return

    filtered = bool(args.ids or args.tags or args.skip_house_agent or args.house_agent_only)
    covered_all = (not filtered) and len(examples) == len(GOLDEN_SET)
    print(f"Run id: {run_id}\nRun directory: {run_dir}")

    if args.phase == "judge":
        generations, prior_manifest = _load_generations(run_dir)
        by_id = {g["id"]: g for g in generations}
        if filtered:
            wanted_ids = {e.id for e in examples}
            missing = [e.id for e in examples if e.id not in by_id]
            if missing:
                print(
                    f"error: generations.jsonl in {run_dir} is missing example(s): {', '.join(missing[:10])}",
                    file=sys.stderr,
                )
                sys.exit(1)
            generations = [g for g in generations if g["id"] in wanted_ids]
        examples = [get_example(g["id"]) for g in generations]
        covered_all = (not filtered) and len(examples) == len(GOLDEN_SET)
        if not covered_all:
            filtered = True
        judge = MockJudge() if args.mock else LLMJudge()
        judge_model = getattr(judge, "model", None)
        agent_model = prior_manifest.get("agent_model") or settings.llama_server_model
        if not args.mock and judge_model == agent_model and not args.allow_self_judge:
            print(
                f"error: the judge is the model under test ({judge_model}). Pass --allow-self-judge or use an independent judge.",
                file=sys.stderr,
            )
            sys.exit(2)
        meta = {
            "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "phase": "judge",
            "git_sha": prior_manifest.get("git_sha") or _git_sha(),
            "timestamp": ts,
            "ids": [e.id for e in examples],
            "count": len(examples),
            "golden_set_size": len(GOLDEN_SET),
            "complete_coverage": covered_all,
            "filtered": filtered,
            "agent_model": agent_model,
            "judge_model": judge_model or "unknown",
            "judge_is_agent_model": judge_model == agent_model,
            "judge_independent": (not args.mock) and judge_model != agent_model,
            "fixture_path": prior_manifest.get("fixture_path"),
            "fixture_sha256": prior_manifest.get("fixture_sha256"),
            "token_usage_totals": _aggregate_token_usage(generations),
            "mock": args.mock,
        }
        results: list[tuple[GoldenExample, ScoreResult]] = []
        print(f"\nGrading {len(examples)} captured generation(s) from {run_dir}...\n")
        for ex, gen in zip(examples, generations):
            t0 = time.time()
            result = score_generation(ex, gen, judge)
            dt = time.time() - t0
            icon = {"PASS": "\u2713", "FAIL": "\u2717", "ERROR": "!", "INCONCLUSIVE": "?"}[result.verdict]
            print(f"  [{icon}] {ex.id:<42s} {result.verdict:<6s} ({result.method}, {dt:.1f}s)")
            if result.verdict != "PASS":
                print(f"        {result.reason}")
            results.append((ex, result))

        _write_manifest(run_dir, meta, generations, results)
        _write_report(results, mock=args.mock, meta=meta, run_dir=run_dir, run_id=run_id)
        _print_summary(results)
        failed = [r for _, r in results if r.verdict in ("FAIL", "ERROR")]
        unknown = [r for _, r in results if r.verdict == "INCONCLUSIVE"]
        if failed:
            sys.exit(1)
        if unknown or (filtered and not args.allow_partial):
            sys.exit(2)
        return

    # Generation phases ("generate" or "all")
    if not args.skip_fixture_build:
        print(f"Building evaluation fixture at {settings.eval_fixture_duckdb_path} ...")
        build_fixture(settings.eval_fixture_duckdb_path)
    settings.duckdb_path = settings.eval_fixture_duckdb_path
    fixture_sha = _sha256_file(settings.eval_fixture_duckdb_path)
    if args.clear_cache:
        _clear_caches()

    judge = None
    judge_model = None
    if args.phase == "all":
        judge = MockJudge() if args.mock else LLMJudge()
        judge_model = getattr(judge, "model", None)
        if not args.mock and judge_model == settings.llama_server_model and not args.allow_self_judge:
            print(
                f"error: the judge is the model under test ({judge_model}). A model scoring its own answers is not "
                "independent evidence: set JUDGE_LLAMA_SERVER_MODEL (and JUDGE_LLAMA_SERVER_BASE_URL) to a different "
                "model, or pass --allow-self-judge to record a run that says so.",
                file=sys.stderr,
            )
            sys.exit(2)

    if not args.mock:
        from observability import ensure_phoenix_server, initialize_observability, stop_phoenix_server

        atexit.register(stop_phoenix_server)
        if settings.phoenix_enabled and not ensure_phoenix_server():
            settings.phoenix_enabled = False
            print("Phoenix is unavailable; continuing evaluation without tracing.", file=sys.stderr)
        initialize_observability()

    meta = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "phase": args.phase,
        "git_sha": _git_sha(),
        "timestamp": ts,
        "ids": [e.id for e in examples],
        "count": len(examples),
        "golden_set_size": len(GOLDEN_SET),
        "complete_coverage": covered_all,
        "filtered": filtered,
        "agent_model": settings.llama_server_model,
        "judge_model": judge_model or "deferred (--phase generate)",
        "judge_is_agent_model": (judge_model == settings.llama_server_model) if judge_model else False,
        "judge_independent": bool(judge_model and not args.mock and judge_model != settings.llama_server_model),
        "fixture_path": str(settings.eval_fixture_duckdb_path),
        "fixture_sha256": fixture_sha,
        "mock": args.mock,
        "repeat": args.repeat,
    }
    if filtered:
        print(
            f"PARTIAL RUN: {len(examples)} of {len(GOLDEN_SET)} examples "
            f"(filters: ids={bool(args.ids)} tags={bool(args.tags)} "
            f"skip_house={args.skip_house_agent} house_only={args.house_agent_only}).",
            file=sys.stderr,
        )
    if meta["judge_is_agent_model"] and not args.mock:
        print(
            f"warning: the judge is the model under test ({meta['judge_model']}); "
            "set JUDGE_LLAMA_SERVER_MODEL to an independent model for stronger evidence.",
            file=sys.stderr,
        )

    generations: list[dict] = []
    results = []
    repeat = max(1, args.repeat)
    trial_label = f" x{repeat} trials" if repeat > 1 else ""
    print(
        f"\nRunning {len(examples)} example(s) [phase={args.phase}]{' [MOCK]' if args.mock else ''}{trial_label}...\n"
    )

    # Per-example multi-trial records for pass@1 / pass^K aggregation.
    trial_records: dict[str, list[ScoreResult]] = {}

    for ex in examples:
        trial_records[ex.id] = []
        for trial_idx in range(repeat):
            if repeat > 1:
                # Disable both caches between trials so each is an independent sample.
                try:
                    import services.cache as _cache  # noqa: PLC0415
                    import unittest.mock as _mock  # noqa: PLC0415

                    _cache_patch: object = _mock.patch.object(_cache, "enabled", return_value=False)
                    _cache_patch.start()  # type: ignore[union-attr]
                except Exception:
                    _cache_patch = None
            else:
                _cache_patch = None

            t0 = time.time()
            gen = generate_example(ex, args.mock, traces_dir=None if args.mock else traces_dir)
            if repeat > 1:
                gen = {**gen, "trial": trial_idx}
            generations.append(gen)

            if repeat > 1 and _cache_patch is not None:
                try:
                    _cache_patch.stop()  # type: ignore[union-attr]
                except Exception:
                    pass

            if args.phase == "generate":
                status_icon = "!" if gen.get("error") else "\u2713"
                trial_suffix = f" [trial {trial_idx + 1}/{repeat}]" if repeat > 1 else ""
                print(f"  [{status_icon}] {ex.id:<42s} GENERATED ({gen.get('latency_s', 0.0):.1f}s){trial_suffix}")
                if gen.get("error"):
                    print(f"        {gen['error']}")
            else:
                result = score_generation(ex, gen, judge)
                trial_records[ex.id].append(result)
                dt = time.time() - t0
                icon = {"PASS": "\u2713", "FAIL": "\u2717", "ERROR": "!", "INCONCLUSIVE": "?"}[result.verdict]
                trial_suffix = f" [trial {trial_idx + 1}/{repeat}]" if repeat > 1 else ""
                print(f"  [{icon}] {ex.id:<42s} {result.verdict:<6s} ({result.method}, {dt:.1f}s){trial_suffix}")
                if result.verdict != "PASS":
                    print(f"        {result.reason}")
                    if result.verdict == "ERROR" and result.extra.get("traceback"):
                        print("        (full traceback in the report)")

        if args.phase != "generate" and trial_records[ex.id]:
            # Representative result for the report: worst-case across trials so a
            # single failure is surfaced (FAIL < ERROR < INCONCLUSIVE < PASS).
            worst_order = {"FAIL": 0, "ERROR": 1, "INCONCLUSIVE": 2, "PASS": 3}
            rep_result = min(trial_records[ex.id], key=lambda r: worst_order.get(r.verdict, 4))
            results.append((ex, rep_result))

    _write_generations(run_dir, generations)
    meta["token_usage_totals"] = _aggregate_token_usage(generations)

    # Compute pass@1 and pass^K metrics when K > 1 trials were run.
    if repeat > 1 and args.phase != "generate":
        # pass@1: fraction of examples where at least 1 trial passed.
        pass_at_1 = sum(1 for ex in examples if any(r.verdict == "PASS" for r in trial_records.get(ex.id, [])))
        # pass^K: fraction of examples where ALL K trials passed.
        pass_all_k = sum(
            1 for ex in examples if trial_records.get(ex.id) and all(r.verdict == "PASS" for r in trial_records[ex.id])
        )
        n_ex = len(examples)
        meta["multi_trial"] = {
            "K": repeat,
            "pass_at_1": pass_at_1,
            "pass_at_1_rate": round(pass_at_1 / n_ex, 4) if n_ex else 0.0,
            "pass_all_k": pass_all_k,
            "pass_all_k_rate": round(pass_all_k / n_ex, 4) if n_ex else 0.0,
            "per_example": {
                ex.id: {
                    "trials": [r.verdict for r in trial_records.get(ex.id, [])],
                    "pass_at_1": any(r.verdict == "PASS" for r in trial_records.get(ex.id, [])),
                    "pass_all_k": all(r.verdict == "PASS" for r in trial_records.get(ex.id, [])),
                }
                for ex in examples
            },
        }

    _write_manifest(run_dir, meta, generations, results if args.phase == "all" else None)

    if args.phase == "generate":
        print(f"\nGenerations captured to:\n  {run_dir / 'generations.jsonl'}\n  {run_dir / 'manifest.json'}")
        has_err = any(g.get("error") for g in generations)
        if has_err:
            sys.exit(1)
        if filtered and not args.allow_partial:
            sys.exit(2)
        return

    _write_report(results, mock=args.mock, meta=meta, run_dir=run_dir, run_id=run_id)
    _print_summary(results, multi_trial=meta.get("multi_trial"))

    failed = [r for _, r in results if r.verdict in ("FAIL", "ERROR")]
    unknown = [r for _, r in results if r.verdict == "INCONCLUSIVE"]
    if failed:
        sys.exit(1)
    if unknown or (filtered and not args.allow_partial):
        sys.exit(2)


def _print_summary(results: list[tuple[GoldenExample, ScoreResult]], *, multi_trial: dict | None = None):
    n = len(results)
    passed = sum(1 for _, r in results if r.verdict == "PASS")
    failed = sum(1 for _, r in results if r.verdict == "FAIL")
    errored = sum(1 for _, r in results if r.verdict == "ERROR")
    unknown = sum(1 for _, r in results if r.verdict == "INCONCLUSIVE")
    print(f"\n{'=' * 60}")
    print(f"{passed}/{n} passed, {failed} failed, {errored} errored, {unknown} inconclusive")

    if multi_trial and multi_trial.get("K", 1) > 1:
        K = multi_trial["K"]
        p1 = multi_trial["pass_at_1"]
        pk = multi_trial["pass_all_k"]
        print(f"  pass@1 (≥1/{K} trials passed): {p1}/{n} ({multi_trial['pass_at_1_rate']:.1%})")
        print(f"  pass^{K} (all {K} trials passed): {pk}/{n} ({multi_trial['pass_all_k_rate']:.1%})")

    assisted = [r for _, r in results if r.extra.get("assisted")]
    if assisted:
        print(f"  assisted (needed a clarifying round trip): {len(assisted)}/{n}")
        for r in assisted:
            print(f"    - {r.example_id}: {r.verdict} after {r.extra['clarification_rounds']} round(s)")

    for label, method_prefix in [("assert-equal (structured)", "assert_equal"), ("LLM judge (free-text)", "llm_judge")]:
        subset = [(e, r) for e, r in results if r.method.startswith(method_prefix)]
        if subset:
            p = sum(1 for _, r in subset if r.verdict == "PASS")
            print(f"  {label}: {p}/{len(subset)}")

    by_tag: dict[str, list[int]] = {}
    for ex, r in results:
        for tag in ex.tags:
            by_tag.setdefault(tag, [0, 0])
            by_tag[tag][1] += 1
            if r.verdict == "PASS":
                by_tag[tag][0] += 1
    if by_tag:
        print("\nBy dataset:")
        for tag, (p, t) in sorted(by_tag.items()):
            print(f"  {tag:<12s} {p}/{t}")


def _write_report(
    results: list[tuple[GoldenExample, ScoreResult]],
    mock: bool,
    meta: dict,
    *,
    run_dir: Path | None = None,
    run_id: str | None = None,
):
    settings.eval_reports_dir.mkdir(parents=True, exist_ok=True)
    ts = run_id or meta.get("run_id") or time.strftime("%Y%m%d-%H%M%S")
    json_path = settings.eval_reports_dir / f"report-{ts}.json"
    md_path = settings.eval_reports_dir / f"report-{ts}.md"

    payload = {
        "timestamp": ts,
        "mock": mock,
        **meta,
        "results": [
            {
                "id": ex.id,
                "agent": ex.agent,
                "tags": list(ex.tags),
                "answer_type": ex.answer_type,
                "question": ex.question,
                "verdict": r.verdict,
                "method": r.method,
                "reason": r.reason,
                "reply": r.reply,
                "traceback": r.extra.get("traceback"),
                "trace_id": r.extra.get("trace_id"),
                "trace_url": r.extra.get("trace_url"),
                "trace_file": r.extra.get("trace_file"),
                "trace_sha256": r.extra.get("trace_sha256"),
                "trace_export_error": r.extra.get("trace_export_error"),
                "clarification_rounds": r.extra.get("clarification_rounds", 0),
                "assisted": r.extra.get("assisted", False),
                "asked": r.extra.get("asked", 0),
                "answer_status": r.extra.get("answer_status"),
                "latency_s": r.extra.get("latency_s", 0.0),
                "token_usage": r.extra.get("token_usage") or {},
            }
            for ex, r in results
        ],
    }
    payload_text = json.dumps(payload, indent=2)
    json_path.write_text(payload_text, encoding="utf-8")

    n = len(results)
    passed = sum(1 for _, r in results if r.verdict == "PASS")
    lines = [f"# Evaluation report — {ts}", ""]
    if mock:
        lines.append("**MOCK RUN — smoke-test only, not a real evaluation.**\n")
    lines.append(f"**{passed}/{n} passed**\n")
    lines.append("## Run")
    lines.append(f"- **command:** `{meta.get('command', '')}`")
    lines.append(f"- **agent model:** `{meta.get('agent_model')}` · **judge model:** `{meta.get('judge_model')}`")
    if meta.get("judge_is_agent_model"):
        lines.append(
            "- **warning:** the judge is the same model that answered — set `JUDGE_LLAMA_SERVER_MODEL` to an independent model."
        )
    if not meta.get("complete_coverage", True):
        lines.append(
            f"- **partial run:** {meta.get('count')} of {meta.get('golden_set_size')} examples — not a quality gate."
        )
    if meta.get("multi_trial"):
        mt = meta["multi_trial"]
        K = mt["K"]
        p1 = mt["pass_at_1"]
        pk = mt["pass_all_k"]
        n_ex = meta.get("count", len(results))
        lines.append(
            f"- **multi-trial (K={K}):** "
            f"pass@1 = {p1}/{n_ex} ({mt['pass_at_1_rate']:.1%}) · "
            f"pass^{K} = {pk}/{n_ex} ({mt['pass_all_k_rate']:.1%})"
        )
    lines.append(f"- **ids:** {', '.join(meta.get('ids') or [])}")
    lines.append("")
    lines.append("| id | agent | type | verdict | method |")
    lines.append("|---|---|---|---|---|")
    for ex, r in results:
        lines.append(f"| {ex.id} | {ex.agent} | {ex.answer_type} | {r.verdict} | {r.method} |")
    lines.append("\n## Details")
    for ex, r in results:
        reply_snippet = (r.reply or "")[:500]
        lines.append(f"\n### {ex.id} — {r.verdict}")
        lines.append(f"- **question:** {ex.question}")
        lines.append(f"- **reason:** {r.reason}")
        lines.append(f"- **reply:** {reply_snippet}")
        if r.extra.get("trace_file"):
            lines.append(f"- **trace file:** `{r.extra['trace_file']}`")
        if r.extra.get("trace_id"):
            lines.append(f"- **trace id:** `{r.extra['trace_id']}`")
        if r.extra.get("trace_export_error"):
            lines.append(f"- **trace export error:** `{r.extra['trace_export_error']}`")
        if r.verdict == "ERROR" and r.extra.get("traceback"):
            lines.append(f"- **traceback:**\n```\n{r.extra['traceback']}```")
    md_text = "\n".join(lines)
    md_path.write_text(md_text, encoding="utf-8")  # reports are read on other machines too
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.json").write_text(payload_text, encoding="utf-8")
        (run_dir / "report.md").write_text(md_text, encoding="utf-8")

    print(f"\nReport written to:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
